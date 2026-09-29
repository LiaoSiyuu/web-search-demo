/*
Query Processing

Usage (interactive):
    ./query_engine

Usage (batch / benchmark harness):
    ./query_engine --batch <queries_file> --mode <0|1> --out <results_file> [--snippets]
        mode 0 = Conjunctive (AND), 1 = Disjunctive (OR)
        results file, one line per result:
            query_idx<TAB>rank<TAB>pid<TAB>score(6dp)[<TAB>snippet]
        query_idx is 0-based (line order in queries_file, blank lines skipped).
        Per-query latency is timed with steady_clock (index load excluded) and
        p50/p95/mean reported to stdout.

Output:
    - query results
*/
#include "config.h"
#include "Document.h"
#include "InvertedIndex.h"
#include "Lexicon.h"
#include "QueryProcessor.h"
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

// ---- batch / benchmark mode ----
static int run_batch(const std::string& queries_file, int mode,
                     const std::string& out_file, bool with_snippets) {

    PassageTable passage_table;
    passage_table.load(PASSAGE_TABLE_FILE);

    Lexicon lexicon;
    lexicon.load(LEXICON_FILE);

    // index load is intentionally OUTSIDE the timed region below
    QueryProcessor qp(passage_table, lexicon, FINAL_INDEX_FILE, DATA_SOURCE_PATH);

    std::ifstream qin(queries_file);
    if (!qin) {
        std::cerr << "Cannot open queries file: " << queries_file << std::endl;
        return 1;
    }
    std::ofstream out(out_file);
    if (!out) {
        std::cerr << "Cannot open out file: " << out_file << std::endl;
        return 1;
    }

    std::vector<double> latencies_ms;
    std::string line;
    size_t qidx = 0;
    // MaxScore pruning counters (OR queries use MaxScore)
    uint64_t agg_scored = 0, agg_total = 0, agg_chunks_skipped = 0;
    long agg_essential = 0, stat_queries = 0;

    while (std::getline(qin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF safety
        if (line.empty()) continue;  // skip blank lines; qidx not advanced

        auto t0 = std::chrono::steady_clock::now();
        auto results = qp.search(line, mode);
        auto t1 = std::chrono::steady_clock::now();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (mode == DISJUNCTIVE) {
            agg_scored += qp.stat_postings_scored;
            agg_total += qp.stat_total_postings;
            agg_essential += qp.stat_essential_after_warmup;
            agg_chunks_skipped += qp.stat_chunks_skipped;
            stat_queries++;
        }

        for (size_t r = 0; r < results.size(); r++) {
            out << qidx << '\t' << (r + 1) << '\t' << results[r].pid << '\t'
                << std::fixed << std::setprecision(6) << results[r].score;
            if (with_snippets) {
                // snippet is single-line and tab-free; emit verbatim (incl. ANSI
                // highlight) for reproducible snippet comparisons.
                out << '\t' << results[r].snippet;
            }
            out << '\n';
        }
        qidx++;
    }
    out.close();

    if (latencies_ms.empty()) {
        std::cout << "Batch: 0 queries run." << std::endl;
        return 0;
    }

    std::vector<double> s = latencies_ms;
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) {
        size_t idx = (size_t)std::ceil(p / 100.0 * s.size());
        if (idx > 0) idx -= 1;                 // nearest-rank, 0-based
        if (idx >= s.size()) idx = s.size() - 1;
        return s[idx];
    };
    double sum = 0.0;
    for (double v : s) sum += v;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nBatch complete: " << s.size() << " queries, mode="
              << mode << " (" << (mode == CONJUNCTIVE ? "AND" : "OR") << ")\n";
    std::cout << "  latency (ms): p50=" << pct(50) << "  p95=" << pct(95)
              << "  mean=" << (sum / s.size())
              << "  min=" << s.front() << "  max=" << s.back() << "\n";
    if (stat_queries > 0 && agg_total > 0) {
        std::cout << "  MaxScore pruning: scored " << agg_scored << " / " << agg_total
                  << " postings (" << (100.0 * agg_scored / agg_total) << "% scored, "
                  << (100.0 * (agg_total - agg_scored) / agg_total) << "% pruned)"
                  << "; mean essential lists after warm-up = "
                  << ((double)agg_essential / stat_queries)
                  << "; block-max chunks skipped = " << agg_chunks_skipped << std::endl;
    }
    std::cout << "  results written to " << out_file << std::endl;
    return 0;
}

static void run_interactive() {
    std::cout << "\n========== Query Processing ==========\n" << std::endl;

    std::cout << "Loading index structures..." << std::endl;
    auto load_start = std::chrono::steady_clock::now();

    PassageTable passage_table;
    passage_table.load(PASSAGE_TABLE_FILE);

    Lexicon lexicon;
    lexicon.load(LEXICON_FILE);

    QueryProcessor qp(passage_table, lexicon, FINAL_INDEX_FILE, DATA_SOURCE_PATH);

    auto load_end = std::chrono::steady_clock::now();
    double load_time = std::chrono::duration<double>(load_end - load_start).count();
    std::cout << "Index loaded in " << load_time << "s" << std::endl;

    std::cout << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     MS MARCO Passage Search Engine - Ready            ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nCommands:" << std::endl;
    std::cout << "  - Enter your query and press Enter" << std::endl;
    std::cout << "  - Choose mode: 0 = Conjunctive (AND), 1 = Disjunctive (OR)" << std::endl;
    std::cout << "  - Type 'exit' or 'quit' to quit\n" << std::endl;

    while (true) {
        std::cout << "\n─────────────────────────────────────────────────────────" << std::endl;
        std::cout << "query>> ";
        std::string query;
        if (!std::getline(std::cin, query)) break;

        if (query == "exit" || query == "quit") break;
        if (query.empty()) continue;

        // choose mode: 0 = Conjunctive (AND), 1 = Disjunctive (OR)
        std::cout << "mode (0=AND, 1=OR)>> ";
        std::string mode_str;
        if (!std::getline(std::cin, mode_str)) break;

        int mode = (mode_str == "0") ? CONJUNCTIVE : DISJUNCTIVE;

        auto query_start = std::chrono::steady_clock::now();
        auto results = qp.search(query, mode);
        auto query_end = std::chrono::steady_clock::now();
        double query_time = std::chrono::duration<double>(query_end - query_start).count();

        std::cout << "\n┌─────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│ Found " << results.size() << " result(s) in "
                  << std::fixed << std::setprecision(3) << query_time << "s";

        int padding = 54 - 8 - std::to_string(results.size()).length() - 16 - 1;
        for (int i = 0; i < padding; i++) std::cout << " ";
        std::cout << "│" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────┘\n" << std::endl;

        if (results.empty()) {
            std::cout << "No passages found matching your query." << std::endl;
            continue;
        }

        for (size_t i = 0; i < results.size(); i++) {
            std::cout << "【" << (i + 1) << "】 Passage ID: " << results[i].pid
                      << " | Score: " << std::fixed << std::setprecision(4)
                      << results[i].score << std::endl;
            std::cout << "    " << results[i].snippet << std::endl;
            if (i < results.size() - 1) std::cout << std::endl;
        }
    }

    std::cout << "\nThank you for using MS MARCO Search Engine!" << std::endl;
}

int main(int argc, char** argv) {
    std::string queries_file, out_file;
    int mode = DISJUNCTIVE;
    bool batch = false, with_snippets = false, mode_set = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--batch" && i + 1 < argc) { batch = true; queries_file = argv[++i]; }
        else if (a == "--mode" && i + 1 < argc) {
            std::string value = argv[++i];
            if (value != "0" && value != "1") {
                std::cerr << "Mode must be 0 (AND) or 1 (OR)" << std::endl;
                return 1;
            }
            mode = value == "0" ? CONJUNCTIVE : DISJUNCTIVE;
            mode_set = true;
        }
        else if (a == "--out" && i + 1 < argc) { out_file = argv[++i]; }
        else if (a == "--snippets") { with_snippets = true; }
        else {
            std::cerr << "Unknown/incomplete argument: " << a << std::endl;
            std::cerr << "Usage: ./query_engine [--batch <queries> --mode <0|1> --out <results> [--snippets]]" << std::endl;
            return 1;
        }
    }

    if (batch) {
        if (queries_file.empty() || out_file.empty() || !mode_set) {
            std::cerr << "Usage: ./query_engine --batch <queries> --mode <0|1> --out <results> [--snippets]" << std::endl;
            return 1;
        }
        return run_batch(queries_file, mode, out_file, with_snippets);
    }

    run_interactive();
    return 0;
}
