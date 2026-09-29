#ifndef INDEX_WRITER_H
#define INDEX_WRITER_H

#include "config.h"
#include "InvertedIndex.h"   // Posting
#include "Document.h"        // PassageTable (index-time max_score)
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>

/*
IndexWriter — single source of truth for the on-disk index.idx + lexicon.lex
format (used by merge_compress).

index.idx ("IDX2" format):
  ["IDX2" 4 bytes]                                  -- file magic, once at head
  per term:  [num_chunks u32] then per chunk
    [chunk_size u32][last_docid u32][max_score f32][docid_bytes u32][freq_bytes u32]
    [varbyte d-gap docids...][varbyte freqs...]
  D-gap base resets to 0 at every chunk boundary (random access depends on it).
  The per-chunk max_score is the BM25 upper bound over that chunk's postings,
  stored f32 ROUNDED UP (a down-rounded value would not be a valid bound and could
  wrongly skip a real top-k chunk).

lexicon.lex (binary "LEXB"): magic + term_count u32, then per term (sorted order)
  [term_len u16][term][offset u64][length u64][doc_count u32][max_score f64]
  (host-endian / non-portable by design — see write_lexicon / config.h).

INVARIANT: postings handed to write_term() MUST have strictly increasing doc_id
(asserted, loud abort). max_score (per-term and per-chunk) uses the SAME shared
bm25 inlines as query-time scoring, so the bounds are tight and cannot drift.
*/
class IndexWriter {
public:
    struct Entry {
        std::string term;
        uint64_t offset;
        uint64_t length;
        uint32_t doc_count;
        double max_score;     // per-term BM25 upper bound
    };

    IndexWriter(const std::string& index_path, const PassageTable& pt)
        : out_(index_path, std::ios::binary), pt_(pt) {
        if (!out_) {
            throw std::runtime_error("IndexWriter: cannot open index file " + index_path);
        }
        out_.write(INDEX_FILE_MAGIC, INDEX_FILE_MAGIC_LEN);   // file head magic
    }

    // Write one term's full posting list. postings must be strictly ascending by doc_id.
    void write_term(const std::string& term, const std::vector<Posting>& postings) {
        for (size_t i = 1; i < postings.size(); i++) {
            if (postings[i].doc_id <= postings[i - 1].doc_id) {
                std::cerr << "FATAL: non-increasing docID in term '" << term
                          << "': " << postings[i - 1].doc_id << " -> "
                          << postings[i].doc_id << " (index " << i << ")" << std::endl;
                std::abort();
            }
        }

        uint64_t start_offset = (uint64_t)out_.tellp();
        double term_max = write_compressed_list(postings);   // also returns per-term max
        uint64_t end_offset = (uint64_t)out_.tellp();

        entries_.push_back(Entry{term, start_offset, end_offset - start_offset,
                                 (uint32_t)postings.size(), term_max});
    }

    // Binary lexicon ("LEXB"): magic + term_count u32, then per term
    //   [term_len u16][term][offset u64][length u64][doc_count u32][max_score f64].
    // Host-endian / non-portable by design (see config.h). max_score stays f64 so it
    // is bit-exact with the value used at query time (no bound-validity rounding).
    void write_lexicon(const std::string& path) const {
        std::ofstream lx(path, std::ios::binary);
        lx.write(LEXICON_MAGIC, LEXICON_MAGIC_LEN);
        uint32_t count = (uint32_t)entries_.size();
        lx.write((char*)&count, sizeof(count));
        for (const auto& e : entries_) {
            if (e.term.size() > UINT16_MAX) {   // term_len must fit u16
                std::cerr << "FATAL: term length " << e.term.size() << " exceeds u16" << std::endl;
                std::abort();
            }
            uint16_t term_len = (uint16_t)e.term.size();
            lx.write((char*)&term_len, sizeof(term_len));
            lx.write(e.term.data(), term_len);
            lx.write((char*)&e.offset, sizeof(e.offset));
            lx.write((char*)&e.length, sizeof(e.length));
            lx.write((char*)&e.doc_count, sizeof(e.doc_count));
            lx.write((char*)&e.max_score, sizeof(e.max_score));
        }
        lx.close();
    }

    size_t term_count() const { return entries_.size(); }
    void close() { out_.close(); }

private:
    std::ofstream out_;
    const PassageTable& pt_;
    std::vector<Entry> entries_;

    // Round a double upper bound up to the nearest f32 that is still >= it, so the
    // stored bound never falls below the true value (validity of pruning).
    static float up_round_f32(double v) {
        float f = (float)v;
        if ((double)f < v) f = std::nextafterf(f, std::numeric_limits<float>::infinity());
        return f;
    }

    // Encode the term's postings as chunks (varbyte + d-gap). Each chunk header
    // carries its BM25 max_score (f32, up-rounded). Returns the per-term max_score
    // (double, exact) for the lexicon.
    double write_compressed_list(const std::vector<Posting>& postings) {
        uint32_t num_chunks = (postings.size() + POSTINGS_PER_CHUNK - 1) / POSTINGS_PER_CHUNK;
        out_.write((char*)&num_chunks, sizeof(num_chunks));

        const double N = (double)pt_.passages.size();
        const double idf = bm25_idf(N, (double)postings.size());
        const double avg = pt_.avg_passage_length;
        double term_max = 0.0;

        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            size_t start = chunk_idx * POSTINGS_PER_CHUNK;
            size_t end = std::min(start + POSTINGS_PER_CHUNK, postings.size());

            // D-gap encode docIDs; base resets to 0 at each chunk boundary
            std::vector<uint8_t> encoded_docids;
            uint32_t prev_docid = 0;
            for (size_t i = start; i < end; i++) {
                uint32_t gap = postings[i].doc_id - prev_docid;
                auto encoded = varbyte_encode(gap);
                encoded_docids.insert(encoded_docids.end(), encoded.begin(), encoded.end());
                prev_docid = postings[i].doc_id;
            }

            // encode freqs + per-chunk BM25 max_score
            std::vector<uint8_t> encoded_freqs;
            double chunk_max = 0.0;
            for (size_t i = start; i < end; i++) {
                auto encoded = varbyte_encode(postings[i].freq);
                encoded_freqs.insert(encoded_freqs.end(), encoded.begin(), encoded.end());
                double len = (double)pt_.passages[postings[i].doc_id].length;
                double c = bm25_contribution(idf, postings[i].freq, len, avg);
                if (c > chunk_max) chunk_max = c;
            }
            if (chunk_max > term_max) term_max = chunk_max;

            uint32_t chunk_size = end - start;
            uint32_t last_docid = postings[end - 1].doc_id;
            float chunk_max_f32 = up_round_f32(chunk_max);
            uint32_t docid_bytes = encoded_docids.size();
            uint32_t freq_bytes = encoded_freqs.size();

            out_.write((char*)&chunk_size, sizeof(chunk_size));
            out_.write((char*)&last_docid, sizeof(last_docid));
            out_.write((char*)&chunk_max_f32, sizeof(chunk_max_f32));
            out_.write((char*)&docid_bytes, sizeof(docid_bytes));
            out_.write((char*)&freq_bytes, sizeof(freq_bytes));

            out_.write((char*)encoded_docids.data(), docid_bytes);
            out_.write((char*)encoded_freqs.data(), freq_bytes);
        }
        return term_max;
    }
};

#endif
