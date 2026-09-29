#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

#include "config.h"

// -------- Posting{doc_id, freq} --------
struct Posting {
    uint32_t doc_id; // doc_id is defined by passage_count in build_index.cpp, i.e. doc_id by order they are processed
    uint32_t freq; // term frequency in PassageTable[doc_id]

    Posting(uint32_t d = 0, uint32_t f = 0) : doc_id(d), freq(f) {}
};

// -------- TempIndex in memory{index, memory_usage, file_counter} --------
class TempIndex {
public:
    std::map<std::string, std::vector<Posting>> index; // map: {term, {Posting}}
    size_t memory_usage; // estimated posting memory used to trigger shard flushes
    uint32_t file_counter; // temp_*.bin counter in disk

    TempIndex() : memory_usage(0), file_counter(0) {}

    void add_posting(const std::string& term, uint32_t doc_id, uint32_t freq) {
        if (index.find(term) == index.end()) { // term not in index
            memory_usage += term.length() + MAP_NODE_OVERHEAD;
        }
        index[term].push_back(Posting(doc_id, freq));
        memory_usage += sizeof(Posting);

        if (memory_usage > MEMORY_CHUNK_SIZE) {
            flush_to_disk();
        }
    }

    // binary temp format. File layout:
    //   [magic "TMP2" : 4 bytes]
    //   per term (std::map order, so sorted):
    //     [term_len u16][term bytes][posting_count u32]
    //     count x ( varbyte(docid d-gap) varbyte(freq) )    -- d-gap base 0 per term
    // Postings within a term are already docID-ascending (passage order), so gaps
    // are non-negative. Replaces the old text format (term:docid freq,...).
    void flush_to_disk() {
        if (index.empty()) return;

        std::string filename = std::string(TEMP_INDEX_FOLDER) + "temp_" +
                               std::to_string(file_counter++) + TEMP_FILE_SUFFIX;
        std::ofstream out(filename, std::ios::binary);

        out.write(TEMP_FILE_MAGIC, TEMP_FILE_MAGIC_LEN);

        for (const auto& entry : index) {
            const std::string& term = entry.first;
            const std::vector<Posting>& postings = entry.second;

            if (term.size() > UINT16_MAX) {   // term_len must fit u16
                std::cerr << "FATAL: term length " << term.size()
                          << " exceeds u16 (term='" << term.substr(0, 64) << "...')" << std::endl;
                std::abort();
            }
            uint16_t term_len = (uint16_t)term.size();
            out.write((char*)&term_len, sizeof(term_len));
            out.write(term.data(), term_len);

            uint32_t count = (uint32_t)postings.size();
            out.write((char*)&count, sizeof(count));

            uint32_t prev = 0;
            for (const auto& p : postings) {
                write_varbyte(out, p.doc_id - prev);   // d-gap, base 0 per term
                write_varbyte(out, p.freq);
                prev = p.doc_id;
            }
        }
        out.close();

        index.clear();
        memory_usage = 0;

    }

    uint32_t get_file_count() const { return file_counter; }
};

#endif
