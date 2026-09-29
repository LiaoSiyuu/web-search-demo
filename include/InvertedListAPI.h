#ifndef INVERTED_LIST_API_H
#define INVERTED_LIST_API_H

#include "config.h"
#include "Lexicon.h"

// ========== inverted list pointer ==========
struct ListPointer {
    // list(term) info
    std::string term;              // term, e.g. "apple"
    uint32_t doc_count;            // num(docs) containing this term

    // current chunk info
    uint32_t num_chunks;           // how many chunks list(term) is divided into
    uint32_t current_chunk;        // current chunk index
    bool chunk_loaded;             // if current chunk is loaded into memory

    // current chunk's data(decompressed)
    std::vector<uint32_t> chunk_docids;   // docIDs in current chunk for list(term)
    std::vector<uint32_t> chunk_freqs;    // corresponding frequencies in current chunk for list(term)
    uint32_t chunk_index;                  // index of the current posting in current chunk
    uint32_t chunk_size;                   // num(postings)

    // current posting :{current_docid, current_freq}
    uint32_t current_docid;
    uint32_t current_freq;

    // assisting info for jumping between chunks, forward seek
    std::vector<uint32_t> last_docids;     // last docID in each chunk
    std::vector<uint32_t> chunk_offsets;   // offset of each chunk's data in the index file
    std::vector<float> chunk_max_scores;   // per-chunk BM25 upper bound (block-max)

    // Start of this term's list in memory. Under the whole-file mapping this is
    // just (index_base + entry.offset) — a pointer into a mapping OWNED by the
    // InvertedListAPI. LIFETIME INVARIANT: list_base is valid only while that
    // mapping outlives this ListPointer. True under the current single-threaded
    // design (all ListPointers are created/destroyed within one InvertedListAPI's
    // lifetime). Callers must preserve this ownership constraint.
    const uint8_t* list_base;

    ListPointer() : term(""), doc_count(0),
                    num_chunks(0), current_chunk(0), chunk_loaded(false),
                    chunk_index(0), chunk_size(0),
                    current_docid(0), current_freq(0),
                    list_base(nullptr) {}
    // No destructor needed: the list points into the InvertedListAPI's whole-file
    // mapping (unmapped once in ~InvertedListAPI). See the lifetime invariant above.
};

// ========== inverted list access API ==========
class InvertedListAPI {
public:
    /*
    * constructor, open index file and get file size in bytes
    * @param lexicon: lexicon
    * @param index_path: index file path
    */
    InvertedListAPI(const Lexicon& lexicon, const std::string& index_path)
        : lexicon_(lexicon) {

        // open index file for reading only, by bytes, work for both text and binary files
        index_fd_ = open(index_path.c_str(), O_RDONLY);
        if (index_fd_ == -1) {
            std::cerr << "Cannot open index file: " << index_path << std::endl;
            throw std::runtime_error("Failed to open index file");
        }

        // get file size in bytes，not needed for current operations
        struct stat sb;
        fstat(index_fd_, &sb);
        index_size_ = sb.st_size;

        // Require the IDX2 magic (v2 layout: per-chunk max_score in headers).
        // A missing magic means a stale index — abort loudly. List offsets
        // in the lexicon already account for the 4-byte magic (tellp at write time).
        char magic[INDEX_FILE_MAGIC_LEN] = {0};
        if (pread(index_fd_, magic, INDEX_FILE_MAGIC_LEN, 0) != INDEX_FILE_MAGIC_LEN ||
            std::memcmp(magic, INDEX_FILE_MAGIC, INDEX_FILE_MAGIC_LEN) != 0) {
            std::cerr << "FATAL: " << index_path << " is not an '" << INDEX_FILE_MAGIC
                      << "' index (rebuild with merge_compress)." << std::endl;
            throw std::runtime_error("stale/incompatible index format");
        }

        // Map once; every list borrows a slice of this mapping. Access is
        // scattered across posting lists, hence the random-access advice.
        void* base = mmap(NULL, index_size_, PROT_READ, MAP_PRIVATE, index_fd_, 0);
        if (base == MAP_FAILED) {
            perror("mmap failed (whole index)");
            throw std::runtime_error("Failed to mmap index");
        }
        index_base_ = (const uint8_t*)base;
        index_mmap_size_ = index_size_;
        madvise(base, index_size_, MADV_RANDOM);

    }

    /*
    * destructor, close index file
    * @param index_fd: index file descriptor
    */
    ~InvertedListAPI() {
        if (index_base_ != nullptr) {
            munmap((void*)index_base_, index_mmap_size_);   // whole-file mapping, once
        }
        if (index_fd_ != -1) {
            close(index_fd_);
        }
    }

    /*
    * open inverted list for a given term
    * @param term: term
    * @return: inverted list pointer
    */
    ListPointer* openList(const std::string& term) {
        auto it = lexicon_.entries.find(term);
        if (it == lexicon_.entries.end()) {
            return nullptr;  // term not found
        }

        const LexiconEntry& entry = it->second;
        ListPointer* lp = new ListPointer();

        lp->term = term;
        lp->doc_count = entry.doc_count; // [doc_count] docs contain this term

        // Borrow the list from the existing mapping; no per-list mmap call.
        lp->list_base = index_base_ + entry.offset;

        const uint8_t* data = lp->list_base;
        size_t pos = 0;

        // how many chunks the term list is divided into
        memcpy(&lp->num_chunks, data + pos, sizeof(uint32_t));
        pos += sizeof(uint32_t);

        // preload all chunks' metadata (for jumping)
        loadChunkMetadata(lp, data, pos);

        // initialize
        lp->current_chunk = 0;
        lp->chunk_index = 0;
        lp->chunk_loaded = false;
        lp->current_docid = 0;

        // load the first chunk(idx from 0)
        loadChunk(lp, 0);

        return lp;
    }

    // ===== close inverted list =====
    void closeList(ListPointer* lp) {
        if (lp != nullptr) {
            delete lp;  // mapping remains owned by InvertedListAPI
        }
    }

    /*
    * forward seek, find the first posting with docID >= k, if not found, return UINT32_MAX
    * @param lp: inverted list pointer
    * @param k: target docID
    * @return: the first docID >= k
    */
    uint32_t nextGEQ(ListPointer* lp, uint32_t k) {
        if (lp == nullptr) {
            return UINT32_MAX;
        }

        // already found current docID >= k, return directly
        if (lp->current_docid >= k) {
            return lp->current_docid;
        }

        // last_docid < k, just skip this chunk
        while (lp->current_chunk < lp->num_chunks &&
               lp->last_docids[lp->current_chunk] < k) {
            lp->current_chunk++;
            lp->chunk_loaded = false;
        }

        // reach end
        if (lp->current_chunk >= lp->num_chunks) {
            lp->current_docid = UINT32_MAX;
            return UINT32_MAX;
        }

        // load current chunk(if not loaded)
        if (!lp->chunk_loaded) {
            loadChunk(lp, lp->current_chunk);
        }

        // find the first posting with docID >= k in current chunk
        while (lp->chunk_index < lp->chunk_size &&
               lp->chunk_docids[lp->chunk_index] < k) {
            lp->chunk_index++;
        }

        // current chunk not found, move to next chunk
        if (lp->chunk_index >= lp->chunk_size) {
            lp->current_chunk++;
            if (lp->current_chunk >= lp->num_chunks) {
                lp->current_docid = UINT32_MAX;
                return UINT32_MAX;
            }
            loadChunk(lp, lp->current_chunk);
            lp->chunk_index = 0;
        }

        // update current posting :{current_docid, current_freq}
        lp->current_docid = lp->chunk_docids[lp->chunk_index];
        lp->current_freq = lp->chunk_freqs[lp->chunk_index];

        return lp->current_docid;
    }

    // ===== get frequency of current posting =====
    uint32_t getFreq(ListPointer* lp) {
        if (lp == nullptr || lp->current_docid == UINT32_MAX) {
            return 0;
        }
        return lp->current_freq;
    }

    // ===== block-max: BM25 upper bound for the chunk holding the cursor =====
    float getCurrentChunkMaxScore(ListPointer* lp) {
        if (lp == nullptr || lp->current_chunk >= lp->num_chunks) return 0.0f;
        return lp->chunk_max_scores[lp->current_chunk];
    }

    // ===== last docID of the chunk holding the cursor (for block skipping) =====
    uint32_t getCurrentChunkLastDocID(ListPointer* lp) {
        if (lp == nullptr || lp->current_chunk >= lp->num_chunks) return UINT32_MAX;
        return lp->last_docids[lp->current_chunk];
    }

    // ===== move to next posting =====
    uint32_t nextPosting(ListPointer* lp) {
        if (lp == nullptr || lp->current_docid == UINT32_MAX) {
            return UINT32_MAX;
        }
        return nextGEQ(lp, lp->current_docid + 1);
    }

private:
    const Lexicon& lexicon_;
    int index_fd_;
    size_t index_size_;
    const uint8_t* index_base_ = nullptr;   // whole-file mmap base
    size_t index_mmap_size_ = 0;

    /*
    * preload all chunks' metadata (for jumping)
    * @param lp: inverted list pointer
    * @param data: pointer to list(term) after mmap
    * @param pos: position of the current chunk
    */
    void loadChunkMetadata(ListPointer* lp, const uint8_t* data, size_t& pos) {
        lp->last_docids.reserve(lp->num_chunks);
        lp->chunk_offsets.reserve(lp->num_chunks);
        lp->chunk_max_scores.reserve(lp->num_chunks);

        size_t current_offset = pos;

        // v2 chunk header: [chunk_size u32][last_docid u32][max_score f32][docid_bytes u32][freq_bytes u32]
        for (uint32_t i = 0; i < lp->num_chunks; i++) {
            uint32_t chunk_size, last_docid, docid_bytes, freq_bytes;
            float max_score;

            memcpy(&chunk_size, data + current_offset, sizeof(uint32_t));
            current_offset += sizeof(uint32_t);

            memcpy(&last_docid, data + current_offset, sizeof(uint32_t));
            current_offset += sizeof(uint32_t);

            memcpy(&max_score, data + current_offset, sizeof(float));
            current_offset += sizeof(float);

            memcpy(&docid_bytes, data + current_offset, sizeof(uint32_t));
            current_offset += sizeof(uint32_t);

            memcpy(&freq_bytes, data + current_offset, sizeof(uint32_t));
            current_offset += sizeof(uint32_t);

            lp->last_docids.push_back(last_docid);
            lp->chunk_max_scores.push_back(max_score);
            lp->chunk_offsets.push_back(current_offset);  // start of this chunk's data

            current_offset += docid_bytes + freq_bytes;    // skip the data
        }
    }

    /*
    * load a specific chunk
    * @param lp: inverted list pointer
    * @param chunk_idx: index of the chunk to load
    */
    void loadChunk(ListPointer* lp, uint32_t chunk_idx) {
        if (chunk_idx >= lp->num_chunks) {
            return;
        }

        // list data starts at list_base (whole-file mmap: index_base + offset;
        // owned by InvertedListAPI).
        const uint8_t* data = lp->list_base;

        size_t chunk_start = lp->chunk_offsets[chunk_idx];

        // go back over the 20-byte v2 header (chunk_size, last_docid, max_score f32,
        // docid_bytes, freq_bytes). chunk_start points at the docID|freq data.
        size_t metadata_pos = chunk_start - (4 * sizeof(uint32_t) + sizeof(float));

        uint32_t chunk_size, last_docid, docid_bytes, freq_bytes;
        memcpy(&chunk_size, data + metadata_pos, sizeof(uint32_t));
        metadata_pos += sizeof(uint32_t);
        memcpy(&last_docid, data + metadata_pos, sizeof(uint32_t));
        metadata_pos += sizeof(uint32_t);
        metadata_pos += sizeof(float);   // skip max_score (already cached in metadata)
        memcpy(&docid_bytes, data + metadata_pos, sizeof(uint32_t));
        metadata_pos += sizeof(uint32_t);
        memcpy(&freq_bytes, data + metadata_pos, sizeof(uint32_t));

        // decompress docIDs
        auto docid_gaps = varbyte_decode(data + chunk_start, docid_bytes);

        // decompress frequencies
        auto freqs = varbyte_decode(data + chunk_start + docid_bytes, freq_bytes);

        // pre-allocate space for docIDs and frequencies
        lp->chunk_docids.clear();
        lp->chunk_freqs.clear();
        lp->chunk_docids.reserve(chunk_size); // reset to num(postings) positions for docIDs
        lp->chunk_freqs.reserve(chunk_size); // reset to num(postings)

        // use D-gap in block! start from 0 for each chunk!!
        uint32_t current_docid = 0;
        // reconstruct docIDs and frequencies
        for (size_t i = 0; i < docid_gaps.size(); i++) {
            current_docid += docid_gaps[i];
            lp->chunk_docids.push_back(current_docid);
            lp->chunk_freqs.push_back(freqs[i]);
        }

        lp->chunk_size = chunk_size;
        lp->chunk_index = 0;
        lp->chunk_loaded = true;
        lp->current_chunk = chunk_idx;

        // update current posting :{current_docid, current_freq}
        if (lp->chunk_size > 0) {
            lp->current_docid = lp->chunk_docids[0];
            lp->current_freq = lp->chunk_freqs[0];
        }
    }
};

#endif
