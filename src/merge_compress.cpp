/*
Fused merge + compress: replaces stages 2 (merge_index) and 3
(compress_index) with one streaming k-way merge straight into the compressed
index. No merged.txt is ever produced.

Usage:
    ./merge_compress

Input:
    - TEMP_INDEX_FOLDER/temp_*.bin   (produced by build_index, term-sorted binary records)
Output:
    - FINAL_INDEX_FILE (index.idx)
    - LEXICON_FILE     (lexicon.lex)
*/
#include "config.h"
#include "Document.h"
#include "InvertedIndex.h"
#include "IndexWriter.h"
#include "KWayMerger.h"
#include <chrono>
#include <sys/stat.h>
#include <sys/resource.h>

// Best-effort: raise the open-file soft limit so holding all temp files open at
// once can't hit EMFILE (macOS default is 256; CI / a future sharded build can
// have many more shards). Never lowers; ignores failure.
static void ensure_fd_limit(uint32_t files_needed) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return;
    rlim_t want = (rlim_t)files_needed + 64;   // + stdio/index/headroom
    if (rl.rlim_cur >= want) return;
    rlim_t newcur = want;
    if (rl.rlim_max != RLIM_INFINITY && newcur > rl.rlim_max) newcur = rl.rlim_max;
    rl.rlim_cur = newcur;
    setrlimit(RLIMIT_NOFILE, &rl);
}

int main() {
    std::cout << "\n========== Stage 2+3 (fused): Merge + Compress ==========\n" << std::endl;
    auto start = std::chrono::steady_clock::now();

    // count temp files: temp_0.bin, temp_1.bin, ...
    uint32_t file_count = 0;
    while (true) {
        std::string filename = std::string(TEMP_INDEX_FOLDER) + "temp_" +
                               std::to_string(file_count) + TEMP_FILE_SUFFIX;
        std::ifstream test(filename);
        if (!test) break;
        file_count++;
    }
    if (file_count == 0) {
        std::cerr << "No temp files found (" << TEMP_FILE_SUFFIX
                  << "). Run build_index first." << std::endl;
        return 1;
    }
    std::cout << "Found " << file_count << " temp files" << std::endl;

    ensure_fd_limit(file_count);

    // PassageTable (lengths + N + avg_len) drives the v2 lexicon's per-term
    // max_score (BM25 upper bound) computed in IndexWriter.
    PassageTable passage_table;
    passage_table.load(PASSAGE_TABLE_FILE);

    IndexWriter writer(FINAL_INDEX_FILE, passage_table);
    KWayMerger::merge(file_count, writer);
    writer.write_lexicon(LEXICON_FILE);
    writer.close();

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n✓ Fused merge+compress complete:" << std::endl;
    std::cout << "  Terms: " << writer.term_count() << std::endl;
    std::cout << "  Time: " << elapsed << "s (" << (elapsed / 60.0) << " min)" << std::endl;

    struct stat st;
    if (stat(FINAL_INDEX_FILE, &st) == 0) {
        double mb = st.st_size / 1024.0 / 1024.0;
        if (mb >= 1024.0) std::cout << "  index.idx: " << (mb / 1024.0) << " GB" << std::endl;
        else              std::cout << "  index.idx: " << mb << " MB" << std::endl;
    }
    std::cout << "  (merged.txt was never created)" << std::endl;
    return 0;
}
