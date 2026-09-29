/*
verify_blockmax: check stored per-chunk max_score values are valid BM25
upper bound *as computed at query time*. For a sampled set of terms it walks
EVERY posting through the real query path (openList / nextGEQ / nextPosting /
getFreq + the shared bm25_contribution inline + the same passage lengths) and
checks contribution <= the chunk's stored f32 max_score. This catches any
index-time vs query-time floating-point drift that a Python-only check would miss.

Usage: ./verify_blockmax [stride]     (default stride 89; 1 = check every term)
*/
#include "config.h"
#include "Document.h"
#include "InvertedIndex.h"
#include "Lexicon.h"
#include "InvertedListAPI.h"
#include <vector>
#include <string>
#include <limits>

int main(int argc, char** argv) {
    uint32_t stride = 89;
    try {
        if (argc > 2) throw std::invalid_argument("too many arguments");
        if (argc == 2) {
            std::string value = argv[1];
            size_t end = 0;
            unsigned long parsed = std::stoul(value, &end);
            if (value.empty() || value[0] == '-' || end != value.size() ||
                parsed == 0 || parsed > UINT32_MAX)
                throw std::invalid_argument("invalid stride");
            stride = static_cast<uint32_t>(parsed);
        }
    } catch (const std::exception&) {
        std::cerr << "Usage: ./verify_blockmax [positive stride, 1 = all terms]" << std::endl;
        return 1;
    }

    PassageTable pt;
    pt.load(PASSAGE_TABLE_FILE);
    Lexicon lex;
    lex.load(LEXICON_FILE);
    InvertedListAPI api(lex, FINAL_INDEX_FILE);

    const double N = (double)pt.passages.size();
    const double avg = pt.avg_passage_length;

    // explicit terms (always checked, incl. a stop-word) + a strided sample
    std::vector<std::string> explicit_terms = {"the", "city", "water", "new", "york", "pizza"};

    uint64_t terms_checked = 0, postings_checked = 0, violations = 0;
    double min_slack = std::numeric_limits<double>::infinity();
    double max_slack = 0.0;

    auto check_term = [&](const std::string& term, const LexiconEntry& e) {
        ListPointer* lp = api.openList(term);
        if (!lp) return;
        double idf = bm25_idf(N, (double)e.doc_count);
        uint32_t did = api.nextGEQ(lp, 0);
        while (did < UINT32_MAX) {
            uint32_t f = api.getFreq(lp);
            double len = (double)pt.passages[did].length;
            double c = bm25_contribution(idf, f, len, avg);
            double cmax = (double)api.getCurrentChunkMaxScore(lp);
            double slack = cmax - c;
            if (slack < 0) {
                if (++violations <= 10)
                    std::cerr << "VIOLATION term='" << term << "' did=" << did
                              << " contribution=" << c << " chunk_max=" << cmax
                              << " (slack=" << slack << ")" << std::endl;
            }
            if (slack < min_slack) min_slack = slack;
            if (slack > max_slack) max_slack = slack;
            postings_checked++;
            did = api.nextPosting(lp);
        }
        api.closeList(lp);
        terms_checked++;
    };

    for (const auto& t : explicit_terms) {
        auto it = lex.entries.find(t);
        if (it != lex.entries.end()) check_term(t, it->second);
    }
    uint64_t idx = 0;
    for (const auto& kv : lex.entries) {
        if (idx++ % stride == 0) check_term(kv.first, kv.second);
    }

    std::cout << "Checked " << postings_checked << " postings across " << terms_checked
              << " terms (stride " << stride << ")\n"
              << "  violations = " << violations << "\n"
              << "  slack (chunk_max - contribution): min=" << min_slack
              << " max=" << max_slack << std::endl;
    if (violations == 0)
        std::cout << "BLOCK-MAX BOUND CHECK PASS (all visited postings satisfy their chunk bounds)"
                  << std::endl;
    else
        std::cout << "BLOCK-MAX BOUND CHECK FAIL" << std::endl;
    return violations == 0 ? 0 : 1;
}
