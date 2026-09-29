#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cctype>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ------ config: file path ------
// DATA_SOURCE_PATH can be overridden at compile time via the Makefile DATASET
// variable, e.g. `make all DATASET=data/collection_dev.tsv`, which passes
// -DDATA_SOURCE_PATH='"..."'. Falls back to the full dataset otherwise.
#ifndef DATA_SOURCE_PATH
#define DATA_SOURCE_PATH "data/collection.tsv"
#endif
#define TEMP_INDEX_FOLDER "./temp/"
// binary temp files: temp_N.bin, each prefixed with a 4-byte magic so the
// k-way merge rejects stale text-format files.
#define TEMP_FILE_SUFFIX ".bin"
#define TEMP_FILE_MAGIC "TMP2"
#define TEMP_FILE_MAGIC_LEN 4
#define PASSAGE_TABLE_FILE "passages.dt"
#define FINAL_INDEX_FILE "index.idx"
#define LEXICON_FILE "lexicon.lex"
// index.idx starts with this 4-byte magic. Its presence marks the v2 chunk
// layout (per-chunk max_score f32 in each chunk header). The reader aborts if absent.
#define INDEX_FILE_MAGIC "IDX2"
#define INDEX_FILE_MAGIC_LEN 4
// binary lexicon ("LEXB"): magic + term_count u32, then per term
//   [term_len u16][term][offset u64][length u64][doc_count u32][max_score f64].
// HOST-ENDIAN / non-portable by design — the index is always rebuilt locally, so
// this avoids parsing textual numeric fields at startup.
#define LEXICON_MAGIC "LEXB"
#define LEXICON_MAGIC_LEN 4

// ------ config: index building parameters ------
#define MEMORY_CHUNK_SIZE (20 * 1024 * 1024)  // 20MB memory chunk
#define MAP_NODE_OVERHEAD 24                  // map node overhead in bytes(estimated, 16 bytes for key and 8 bytes for value)
#define POSTINGS_PER_CHUNK 64                 // 64 postings per chunk
// Per-cursor read buffer for the k-way merge. With ~157 temp files this is
// ~157 MB of buffers (157 * 1MB), in addition to decoded posting lists,
// the combined current term, the passage table, and accumulated lexicon entries.
#define MERGE_BUFFER_SIZE (1024 * 1024)       // 1MB per merge cursor

// ========== query parameters ==========
#define TOP_K 10                              // return top K results
#define BM25_K1 1.2
#define BM25_B 0.75
#define SNIPPET_WINDOW 50                     // snippet window size(in words)
#define SNIPPET_READ_BUF (16 * 1024)          // initial pread buffer; doubles until newline/EOF (no cap)

// ========== query type ==========
#define CONJUNCTIVE 0
#define DISJUNCTIVE 1

#define HIGHLIGHT_START "\033[1;33m"  // ANSI highlight (bold yellow)
#define HIGHLIGHT_END "\033[0m"        // ANSI reset

// ========== tools ==========

// Varbyte encode
inline std::vector<uint8_t> varbyte_encode(uint32_t value) {
    std::vector<uint8_t> encoded;
    while (value >= 128) {
        encoded.push_back((value & 0x7F) | 0x80);
        value >>= 7;
    }
    encoded.push_back(value & 0x7F);
    return encoded;
}

// Varbyte decode
inline std::vector<uint32_t> varbyte_decode(const uint8_t* data, size_t len) {
    std::vector<uint32_t> decoded;
    uint32_t value = 0;
    int shift = 0;

    for (size_t i = 0; i < len; i++) {
        value |= (data[i] & 0x7F) << shift;
        if ((data[i] & 0x80) == 0) {
            decoded.push_back(value);
            value = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }
    return decoded;
}

// Streaming varbyte for the binary temp format — write/read one value to/from
// a stream without allocating (unlike varbyte_encode which returns a vector).
inline void write_varbyte(std::ostream& out, uint32_t value) {
    while (value >= 128) {
        out.put((char)((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.put((char)(value & 0x7F));
}

// Reads one varbyte; sets ok=false on EOF, including a truncated encoding.
inline uint32_t read_varbyte(std::istream& in, bool& ok) {
    uint32_t value = 0;
    int shift = 0;
    int c = in.get();
    if (c == EOF) { ok = false; return 0; }
    while (true) {
        value |= (uint32_t)(c & 0x7F) << shift;
        if ((c & 0x80) == 0) break;
        shift += 7;
        c = in.get();
        if (c == EOF) { ok = false; return value; }  // truncated; caller aborts
    }
    ok = true;
    return value;
}

// ---- BM25 (single source of truth, used by both query-time scoring and the
// index-time max_score in IndexWriter, so the two can never drift) ----
inline double bm25_idf(double N, double df) {
    return std::log((N - df + 0.5) / (df + 0.5));
}
// Per-(term,doc) contribution; clamped non-negative exactly as the original
// QueryProcessor::calculate_bm25. K1/B are frozen.
inline double bm25_contribution(double idf, uint32_t freq, double doc_len, double avg_len) {
    double K = BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avg_len);
    double tf_component = (BM25_K1 + 1) * freq / (K + freq);
    return std::max(0.0, idf * tf_component);
}

// tokenize text content
inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string word;
    const std::string delimiters = " \t\n\r\f.,;:!?[]{}()<>\"'`~@#$%^&*-+=|\\/_";

    for (char c : text) {
        if (delimiters.find(c) == std::string::npos) {
            word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!word.empty()) {
            tokens.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) {
        tokens.push_back(word);
    }
    return tokens;
}

#endif
