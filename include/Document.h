#ifndef DOCUMENT_H
#define DOCUMENT_H

#include "config.h"

// -------- Passage{pid, length, file_offset} --------
struct Passage {
    uint32_t pid;           // Passage ID in tsv file
    uint32_t length;        // Passage length(word count)
    uint64_t file_offset;   // file offset in tsv file

    Passage() : pid(0), length(0), file_offset(0) {}
};

// -------- Passage Table{passages, avg_passage_length} --------
class PassageTable {
public:
    std::vector<Passage> passages;
    double avg_passage_length;

    PassageTable() : avg_passage_length(0) {}

    void add(const Passage& passage) {
        passages.push_back(passage);
    }

    void calculate_avg_length() {
        uint64_t total = 0;
        for (const auto& passage : passages) {
            total += passage.length;
        }
        avg_passage_length = passages.empty() ? 0 : (double)total / passages.size();
    }

    // save passage table to binary file
    void save(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        uint32_t count = passages.size();
        out.write((char*)&count, sizeof(count));
        out.write((char*)&avg_passage_length, sizeof(avg_passage_length));

        for (const auto& passage : passages) {
            out.write((char*)&passage.pid, sizeof(passage.pid));
            out.write((char*)&passage.length, sizeof(passage.length));
            out.write((char*)&passage.file_offset, sizeof(passage.file_offset));
        }
        out.close();

    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "Cannot open " << path << std::endl;
            exit(1);
        }

        uint32_t count;
        in.read((char*)&count, sizeof(count));
        in.read((char*)&avg_passage_length, sizeof(avg_passage_length));

        passages.resize(count); // resize to include all passages in PASSAGE_TABLE_FILE
        for (auto& passage : passages) {
            in.read((char*)&passage.pid, sizeof(passage.pid));
            in.read((char*)&passage.length, sizeof(passage.length));
            in.read((char*)&passage.file_offset, sizeof(passage.file_offset));
        }
        in.close();

        std::cout << "Loaded " << count << " passages (avg_len="
                  << avg_passage_length << " words)" << std::endl;
    }
};

#endif
