#ifndef KWAY_MERGER_H
#define KWAY_MERGER_H

#include "config.h"
#include "InvertedIndex.h"
#include "IndexWriter.h"
#include <memory>

/*
KWayMerger — single streaming pass that replaces stages 2+3 (pairwise merge +
re-parse + compress). It opens all N temp files at once, keeps a min-heap of
cursors keyed by (term, file_id), and for each term concatenates every file's
postings IN FILE-ID ORDER straight into IndexWriter — no merged.txt, no rewrites.

Why file-id order yields globally ascending docIDs (so write_term's assertion
holds): build_index flushes temp files in passage order, so temp file i holds
strictly lower passage ids than file i+1, except the single passage that straddles
a flush boundary. A term appears at most once per passage (term_freq is a per-
passage map), so that boundary passage's posting lands in exactly one file and is
never duplicated. Hence, for any term, file i's docIDs are all < file i+1's, and
concatenating in file-id order is strictly increasing. (Asserted in write_term.)

Reads the binary temp format (see TempIndex::flush_to_disk). Each file starts
with the "TMP2" magic; a missing magic (e.g. a stale text-format file) aborts loud.
*/
class KWayMerger {
public:
    static void merge(uint32_t num_files, IndexWriter& writer) {
        std::vector<std::unique_ptr<Cursor>> cursors;
        cursors.reserve(num_files);
        for (uint32_t i = 0; i < num_files; i++) {
            auto cur = std::make_unique<Cursor>(i);
            std::string path = std::string(TEMP_INDEX_FOLDER) + "temp_" +
                               std::to_string(i) + TEMP_FILE_SUFFIX;
            cur->open(path);
            if (!cur->in) {
                std::cerr << "FATAL: cannot open temp file " << path << std::endl;
                std::abort();
            }
            cur->advance();
            cursors.push_back(std::move(cur));
        }

        // min-heap: smallest term first, ties broken by smallest file_id so equal
        // terms are popped in ascending file-id (== ascending docID) order.
        auto cmp = [&cursors](uint32_t a, uint32_t b) {
            const std::string& ta = cursors[a]->term;
            const std::string& tb = cursors[b]->term;
            if (ta != tb) return ta > tb;
            return cursors[a]->file_id > cursors[b]->file_id;
        };
        std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(cmp)> heap(cmp);
        for (uint32_t i = 0; i < num_files; i++) {
            if (cursors[i]->valid) heap.push(i);
        }

        std::vector<Posting> combined;   // reused; peak ~largest term's list
        std::vector<uint32_t> group;

        while (!heap.empty()) {
            std::string term = cursors[heap.top()]->term;  // copy: cursor will advance

            combined.clear();
            group.clear();
            while (!heap.empty() && cursors[heap.top()]->term == term) {
                group.push_back(heap.top());
                heap.pop();
            }
            for (uint32_t idx : group) {
                const std::vector<Posting>& ps = cursors[idx]->postings;
                combined.insert(combined.end(), ps.begin(), ps.end());
                cursors[idx]->advance();
                if (cursors[idx]->valid) heap.push(idx);
            }

            writer.write_term(term, combined);   // asserts strictly increasing docIDs

        }

    }

private:
    // One buffered reader over a binary temp file, holding the current record
    // (term + decoded postings).
    struct Cursor {
        std::ifstream in;
        std::vector<char> buffer;
        std::string term;
        std::vector<Posting> postings;
        uint32_t file_id;
        bool valid = false;

        explicit Cursor(uint32_t fid) : buffer(MERGE_BUFFER_SIZE), file_id(fid) {}

        void open(const std::string& path) {
            in.rdbuf()->pubsetbuf(buffer.data(), buffer.size());  // before open
            in.open(path, std::ios::binary);
            if (!in) return;
            char magic[TEMP_FILE_MAGIC_LEN];
            in.read(magic, TEMP_FILE_MAGIC_LEN);
            if (in.gcount() != TEMP_FILE_MAGIC_LEN ||
                std::memcmp(magic, TEMP_FILE_MAGIC, TEMP_FILE_MAGIC_LEN) != 0) {
                std::cerr << "FATAL: " << path << " missing '" << TEMP_FILE_MAGIC
                          << "' magic — stale/text temp file? Re-run build_index."
                          << std::endl;
                std::abort();
            }
        }

        // Decode the next record: [term_len u16][term][count u32] then count pairs
        // of (varbyte docid d-gap, varbyte freq). d-gap base 0 per record.
        void advance() {
            uint16_t term_len;
            if (!in.read((char*)&term_len, sizeof(term_len))) { valid = false; return; }
            term.resize(term_len);
            if (term_len && !in.read(&term[0], term_len)) { valid = false; return; }

            uint32_t count;
            if (!in.read((char*)&count, sizeof(count))) {
                std::cerr << "FATAL: truncated temp record (count) for term '" << term << "'" << std::endl;
                std::abort();
            }

            postings.clear();
            postings.reserve(count);
            uint32_t prev = 0;
            bool ok = true;
            for (uint32_t i = 0; i < count; i++) {
                uint32_t gap = read_varbyte(in, ok);
                uint32_t freq = read_varbyte(in, ok);
                if (!ok) {
                    std::cerr << "FATAL: truncated postings for term '" << term << "'" << std::endl;
                    std::abort();
                }
                prev += gap;
                postings.push_back(Posting(prev, freq));
            }
            valid = true;
        }
    };
};

#endif
