/*
Build index from MS MARCO Passages Ranking Dataset

Usage:
    ./build_index

Output:
    - PASSAGE_TABLE_FILE(passages.dt)
    - TEMP_INDEX_FOLDER/temp_*.bin
*/
#include "config.h"
#include "Document.h"
#include "InvertedIndex.h"
#include "Lexicon.h"
#include <chrono>
#include <sys/stat.h>
#include <dirent.h>

void create_temp_folder() {
    struct stat st;
    if (stat(TEMP_INDEX_FOLDER, &st) != 0) {
        mkdir(TEMP_INDEX_FOLDER, 0755); // create temp folder(no special permission;rwx;r-x;r-x)
        std::cout << "Created temp folder: " << TEMP_INDEX_FOLDER << std::endl;
    }
}

// Remove any stale temp_* files (either old .txt text format or prior .bin) so a
// shorter run can't leave higher-numbered stale shards behind, and so the new
// binary format never mixes with stale text files.
void purge_temp_files() {
    DIR* dir = opendir(TEMP_INDEX_FOLDER);
    if (!dir) return;
    struct dirent* e;
    while ((e = readdir(dir)) != nullptr) {
        std::string name = e->d_name;
        if (name.rfind("temp_", 0) == 0) {   // starts with "temp_"
            std::remove((std::string(TEMP_INDEX_FOLDER) + name).c_str());
        }
    }
    closedir(dir);
}

int main() {
    std::cout << "\n========== Stage 1: Building Index from MS MARCO Passages ==========\n" << std::endl;
    auto start = std::chrono::steady_clock::now();   // wall-clock (clock() under-reports I/O)

    create_temp_folder();
    purge_temp_files();   // clear any stale temp_* before writing fresh shards

    PassageTable passage_table; // passage table for passage meta data
    TempIndex temp_index; // temp inverted index in memory

    std::ifstream infile(DATA_SOURCE_PATH); // open data file(collection.tsv)
    if (!infile) {
        std::cerr << "Cannot open data file: " << DATA_SOURCE_PATH << std::endl;
        return 1;
    }

    std::string line; // each line of tsv(passage_id \t text_content)
    uint32_t passage_count = 0; // count of processed passages
    uint64_t file_offset = 0; // file offset of current line(file_offset += line.length() + 1)

    std::cout << "Reading passages from " << DATA_SOURCE_PATH << "..." << std::endl;

    while (std::getline(infile, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) { // if no tab, skip this line
            file_offset += line.length() + 1;
            continue;
        }

        try {
            // passage_id \t text_content
            uint32_t pid = std::stoul(line.substr(0, tab));
            std::string text = line.substr(tab + 1);

            Passage passage;
            passage.pid = pid;
            passage.file_offset = file_offset;

            // tokenize text content
            auto tokens = tokenize(text);
            passage.length = tokens.size();

            // count term frequency
            std::map<std::string, uint32_t> term_freq;
            for (const auto& token : tokens) {
                if (!token.empty()) {
                    term_freq[token]++;
                }
            }

            for (const auto& [term, frequency] : term_freq) {
                temp_index.add_posting(term, passage_count, frequency);
            }

            passage_table.add(passage);
            passage_count++;

        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << e.what() << std::endl;
        }

        file_offset += line.length() + 1;
    }

    infile.close();
    temp_index.flush_to_disk();
    passage_table.calculate_avg_length();
    passage_table.save(PASSAGE_TABLE_FILE);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n✓ Indexing complete:" << std::endl;
    std::cout << "  Passages: " << passage_count << std::endl;
    std::cout << "  Temp files: " << temp_index.get_file_count() << std::endl;
    std::cout << "  Time: " << elapsed << "s" << std::endl;

    return 0;
}
