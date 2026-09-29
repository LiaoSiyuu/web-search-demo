#ifndef LEXICON_H
#define LEXICON_H

#include "config.h"
#include "InvertedIndex.h"
#include <limits>
#include <unordered_map>

//------ Lexicon Entry ------
// Lexicon offsets and lengths are u64, with a per-term
// BM25 max_score upper bound for MaxScore pruning. Written by IndexWriter
// as the v2 lexicon; build-time encoding lives entirely in IndexWriter.
struct LexiconEntry {
    uint64_t offset;
    uint64_t length;
    uint32_t doc_count;
    double max_score;

    LexiconEntry() : offset(0), length(0), doc_count(0),
                     max_score(std::numeric_limits<double>::infinity()) {}
};

//------ Lexicon Class ------
class Lexicon {
public:
    // unordered_map: the query path only does find(term) (no sorted iteration), so
    // O(1) lookups beat std::map's O(log n). The on-disk lexicon order is set by
    // IndexWriter (a sorted vector), independent of this container.
    std::unordered_map<std::string, LexiconEntry> entries; // term -> lexicon entry

    // Loads the binary LEXB lexicon. Host-endian / non-portable by design (the
    // index is always rebuilt locally). A missing/incompatible magic aborts loudly.
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "Cannot open " << path << std::endl;
            exit(1);
        }

        char magic[LEXICON_MAGIC_LEN] = {0};
        in.read(magic, LEXICON_MAGIC_LEN);
        if (in.gcount() != LEXICON_MAGIC_LEN ||
            std::memcmp(magic, LEXICON_MAGIC, LEXICON_MAGIC_LEN) != 0) {
            std::cerr << "FATAL: " << path << " is not a '" << LEXICON_MAGIC
                      << "' lexicon. Rebuild with merge_compress." << std::endl;
            exit(1);
        }

        // term_count drives reserve() AND an integrity assert (read exactly N)
        uint32_t count = 0;
        in.read((char*)&count, sizeof(count));
        entries.reserve(count + count / 8);   // headroom vs the load factor
        uint32_t read = 0;
        for (uint32_t i = 0; i < count; i++) {
            uint16_t term_len;
            if (!in.read((char*)&term_len, sizeof(term_len))) break;
            std::string term(term_len, '\0');
            if (term_len && !in.read(&term[0], term_len)) break;
            LexiconEntry e;
            in.read((char*)&e.offset, sizeof(e.offset));
            in.read((char*)&e.length, sizeof(e.length));
            in.read((char*)&e.doc_count, sizeof(e.doc_count));
            if (!in.read((char*)&e.max_score, sizeof(e.max_score))) break;
            entries.emplace(std::move(term), e);
            read++;
        }
        if (read != count) {
            std::cerr << "FATAL: lexicon truncated: read " << read << " of " << count
                      << " terms in " << path << std::endl;
            exit(1);
        }
        in.close();
        std::cout << "Loaded lexicon with " << entries.size() << " terms (binary LEXB)" << std::endl;
    }
};

#endif
