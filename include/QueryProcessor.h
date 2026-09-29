#ifndef QUERY_PROCESSOR_H
#define QUERY_PROCESSOR_H

#include "config.h"
#include "Document.h"
#include "Lexicon.h"
#include "InvertedListAPI.h"
#include <unordered_set>

// -------- search results --------
struct SearchResult {
    uint32_t pid; // external passage ID
    uint32_t doc_id; // internal row index used for passage-table lookup
    double score; // score of the result
    std::string snippet; // snippet of the result

    bool operator<(const SearchResult& other) const {
        return score < other.score;
    }
};

// Min-heap comparator for top-k selection: the SMALLEST score sits on top so
// that, once the heap holds more than TOP_K entries, pop() evicts the lowest
// score and the k HIGHEST survive. (A default priority_queue<SearchResult> is a
// max-heap and would evict the maximum, retaining the k lowest — i.e. bottom-k.)
struct ScoreGreater {
    bool operator()(const SearchResult& a, const SearchResult& b) const {
        return a.score > b.score;
    }
};

//------ Query Processor Class ------
class QueryProcessor {
public:
    QueryProcessor(const PassageTable& passage_table, const Lexicon& lexicon,
                   const std::string& index_path, const std::string& data_path)
        : passage_table_(passage_table), lexicon_(lexicon),
          data_path_(data_path),
          list_api_(lexicon, index_path) {  // initialize inverted list API

        // open collection.tsv ONCE here (centralized open => centralized error)
        // for pread-based snippets, instead of a fresh ifstream per result per query.
        snippet_fd_ = open(data_path_.c_str(), O_RDONLY);
        if (snippet_fd_ == -1) {
            std::cerr << "FATAL: cannot open data file for snippets: " << data_path_ << std::endl;
            throw std::runtime_error("Failed to open collection.tsv");
        }

    }

    ~QueryProcessor() {
        if (snippet_fd_ != -1) close(snippet_fd_);
    }

    // main func: search query and return results
    std::vector<SearchResult> search(const std::string& query, int mode) {
        stat_postings_scored = 0;
        stat_total_postings = 0;
        stat_essential_after_warmup = 0;
        stat_chunks_skipped = 0;
        auto terms = tokenize(query);
        if (terms.empty()) return {};

        if (mode == CONJUNCTIVE) {
            return conjunctive_search_with_api(terms);
        }
        return disjunctive_search_maxscore(terms);   // OR: MaxScore
    }

    // MaxScore per-query pruning stats (valid after a MaxScore disjunctive call)
    uint64_t stat_postings_scored = 0;   // (term,doc) contributions actually computed
    uint64_t stat_total_postings = 0;    // sum of query terms' doc_counts (exhaustive cost)
    int stat_essential_after_warmup = 0; // essential-list count once the heap first filled
    uint64_t stat_chunks_skipped = 0;    // essential chunks skipped via block-max

private:
    const PassageTable& passage_table_; // passage table{{pid, length, file_offset}...}
    const Lexicon& lexicon_; // lexicon{term -> {offset, length, doc_count}...}
    std::string data_path_; // data file path(collection.tsv)
    InvertedListAPI list_api_;  // inverted list API instance
    int snippet_fd_ = -1;       // persistent fd into collection.tsv for pread snippets

    //------ conjunctive search (AND) - using API ------
    std::vector<SearchResult> conjunctive_search_with_api(const std::vector<std::string>& terms) {
        std::vector<ListPointer*> list_pointers;

        // -> open all inverted lists
        for (const auto& term : terms) {
            ListPointer* lp = list_api_.openList(term);
            if (lp == nullptr) {
                // some terms not found, close all opened lists
                for (auto* ptr : list_pointers) {
                    list_api_.closeList(ptr);
                }
                return {};
            }
            list_pointers.push_back(lp);
        }

        // -> sort by document frequency(shortest list first)
        std::sort(list_pointers.begin(), list_pointers.end(),
                  [](ListPointer* a, ListPointer* b) {
                      return a->doc_count < b->doc_count;
                  });

        std::map<uint32_t, double> scores;

        // traverse each posting in the shortest list
        uint32_t did = list_api_.nextGEQ(list_pointers[0], 0);

        while (did < UINT32_MAX) { // found such docID in the shortest list
            bool found_in_all = true;
            double score = 0.0;

            // check if this docID also exists in other terms' lists
            for (size_t i = 0; i < list_pointers.size(); i++) {
                uint32_t d = list_api_.nextGEQ(list_pointers[i], did);

                if (d != did) {
                    found_in_all = false;
                    if (d < UINT32_MAX) {
                        did = d;  // move to next candidate
                    } else {
                        did = UINT32_MAX;  // some list reached end
                    }
                    break;
                }

                // calculate the contribution of this term
                uint32_t freq = list_api_.getFreq(list_pointers[i]);
                score += calculate_bm25(list_pointers[i]->term, did, freq);
            }

            if (found_in_all) {
                scores[did] = score;
                did = list_api_.nextGEQ(list_pointers[0], did + 1);
            }
        }

        // close all lists
        for (auto* lp : list_pointers) {
            list_api_.closeList(lp);
        }

        return extract_top_k_from_map(scores, terms);
    }

    //------ disjunctive search (OR) - MaxScore ------
    // Standard MaxScore (Turtle & Flood): lists sorted ascending by max_score with
    // prefix sums ub[]; lists [0..pivot-1] are non-essential (their combined ceiling
    // <= theta), [pivot..n-1] essential. DAAT over essential lists only; non-essential
    // lists are probed by nextGEQ in descending max_score and abandoned once
    // score+ub[i] <= theta. Top-10 is identical to exhaustive OR (only provably-
    // hopeless docs are skipped); boundary ties may swap among equal-score docs.
    std::vector<SearchResult> disjunctive_search_maxscore(const std::vector<std::string>& terms) {
        struct TermList { ListPointer* lp; double max_score; std::string term; };
        std::vector<TermList> lists;
        for (const auto& term : terms) {
            ListPointer* lp = list_api_.openList(term);
            if (lp == nullptr) {
                continue;
            }
            double ms = lexicon_.entries.find(term)->second.max_score;  // term exists
            lists.push_back({lp, ms, term});
        }
        if (lists.empty()) return {};

        std::sort(lists.begin(), lists.end(),
                  [](const TermList& a, const TermList& b) { return a.max_score < b.max_score; });
        const int n = (int)lists.size();

        std::vector<double> ub(n);             // prefix sums of max_score (upper bounds)
        double running = 0.0;
        for (int i = 0; i < n; i++) { running += lists[i].max_score; ub[i] = running; }

        // stats
        stat_total_postings = 0;
        for (auto& tl : lists) stat_total_postings += tl.lp->doc_count;
        stat_postings_scored = 0;
        stat_essential_after_warmup = n;
        stat_chunks_skipped = 0;
        bool warmup_recorded = false;

        std::vector<uint32_t> cur(n);          // current docID per list
        for (int i = 0; i < n; i++) cur[i] = list_api_.nextGEQ(lists[i].lp, 0);

        std::priority_queue<SearchResult, std::vector<SearchResult>, ScoreGreater> heap;
        double theta = 0.0;
        auto pivot_for = [&](double th) {      // smallest p with ub[p] > th
            int p = 0; while (p < n && ub[p] <= th) p++; return p;
        };
        int pivot = pivot_for(theta);

        // Termination (boundary cond. 1): exits when pivot == n (theta grew to
        // >= ub[n-1], every list non-essential) OR when the essential min is
        // UINT32_MAX (all essential cursors exhausted while non-essentials may
        // still hold postings — those docs are bounded by ub[pivot-1] <= theta).
        while (pivot < n) {
            const int p = pivot;               // freeze pivot for this candidate

            // Block-max: drop any essential chunk that provably can't reach
            // theta — chunk_block_max + (sum of the OTHER lists' global max) <= theta
            // means no doc in that chunk is top-k, so jump past it. Correct because
            // such docs have score <= theta and the strict (score > theta) push would
            // reject them anyway; the top-k is unchanged.
            if (theta > 0.0) {
                for (int i = p; i < n; i++) {
                    while (cur[i] != UINT32_MAX &&
                           (double)list_api_.getCurrentChunkMaxScore(lists[i].lp)
                               + (ub[n - 1] - lists[i].max_score) <= theta) {
                        uint32_t chunk_last = list_api_.getCurrentChunkLastDocID(lists[i].lp);
                        cur[i] = list_api_.nextGEQ(lists[i].lp, chunk_last + 1);
                        stat_chunks_skipped++;
                    }
                }
            }

            uint32_t d = UINT32_MAX;           // min docID over essential lists [p..n-1]
            for (int i = p; i < n; i++) if (cur[i] < d) d = cur[i];
            if (d == UINT32_MAX) break;        // essential lists exhausted

            double score = 0.0;
            for (int i = p; i < n; i++) {      // essential contributions at d
                if (cur[i] == d) {
                    score += calculate_bm25(lists[i].term, d, list_api_.getFreq(lists[i].lp));
                    stat_postings_scored++;
                }
            }
            // probe non-essential lists in descending max_score (p-1 .. 0)
            for (int i = p - 1; i >= 0; i--) {
                if (score + ub[i] <= theta) break;          // cannot reach theta
                cur[i] = list_api_.nextGEQ(lists[i].lp, d); // monotonic forward move
                if (cur[i] == d) {
                    score += calculate_bm25(lists[i].term, d, list_api_.getFreq(lists[i].lp));
                    stat_postings_scored++;
                }
            }

            // push iff warming up OR strictly beating the k-th best. With theta==0
            // this reproduces the baseline's score>0 filter exactly: candidates come
            // from essential lists (max_score>0 => idf>0), so their score is > 0.
            if ((int)heap.size() < TOP_K || score > theta) {
                SearchResult r;
                r.pid = passage_table_.passages[d].pid;
                r.doc_id = d;
                r.score = score;
                heap.push(r);
                if ((int)heap.size() > TOP_K) heap.pop();
                if ((int)heap.size() == TOP_K) theta = heap.top().score;
            }

            // advance ONLY essential lists that were at d (non-essentials already
            // moved by their nextGEQ probes) — advancing all is the classic skip bug
            for (int i = p; i < n; i++) {
                if (cur[i] == d) cur[i] = list_api_.nextGEQ(lists[i].lp, d + 1);
            }

            pivot = pivot_for(theta);          // theta may have grown -> pivot moves right
            if (!warmup_recorded && (int)heap.size() == TOP_K) {
                stat_essential_after_warmup = n - pivot;
                warmup_recorded = true;
            }
        }

        for (auto& tl : lists) list_api_.closeList(tl.lp);

        std::vector<SearchResult> results;
        while (!heap.empty()) { results.push_back(heap.top()); heap.pop(); }
        std::reverse(results.begin(), results.end());   // descending by score
        for (auto& r : results) r.snippet = generate_snippet(r.doc_id, terms);
        return results;
    }

    //------ BM25 calculation (delegates to the shared config.h inlines) ------
    double calculate_bm25(const std::string& term, uint32_t doc_id, uint32_t freq) {
        if (doc_id >= passage_table_.passages.size()) return 0.0;

        auto it = lexicon_.entries.find(term);
        if (it == lexicon_.entries.end()) return 0.0;

        double N = passage_table_.passages.size();
        double idf = bm25_idf(N, it->second.doc_count);
        return bm25_contribution(idf, freq,
                                 passage_table_.passages[doc_id].length,
                                 passage_table_.avg_passage_length);
    }

    //------ extract top K (from map) — used by the conjunctive (AND) path ------
    std::vector<SearchResult> extract_top_k_from_map(const std::map<uint32_t, double>& scores,
                                                     const std::vector<std::string>& terms) {
        std::priority_queue<SearchResult, std::vector<SearchResult>, ScoreGreater> heap;

        for (const auto& entry : scores) {
            if (entry.second > 0) {
                SearchResult result;
                result.pid = passage_table_.passages[entry.first].pid;
                result.doc_id = entry.first;
                result.score = entry.second;

                heap.push(result);
                if (heap.size() > TOP_K) {
                    heap.pop();
                }
            }
        }

        std::vector<SearchResult> results;
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
        std::reverse(results.begin(), results.end());

        for (auto& result : results) {
            result.snippet = generate_snippet(result.doc_id, terms);
        }

        return results;
    }

    // Read the TSV line for internal `doc_id` into `line` (without the trailing '\n'). Uses a
    // single pread on the persistent fd (no per-result open). pread is positional,
    // so it's safe even if the engine were multi-threaded.
    bool read_passage_line(uint32_t doc_id, std::string& line) {
        if (doc_id >= passage_table_.passages.size()) return false;
        uint64_t offset = passage_table_.passages[doc_id].file_offset;

        // pread into a buffer that doubles until a newline is found or EOF — NO cap,
        // so (like std::getline) the whole line is read without truncation.
        std::vector<char> buf(SNIPPET_READ_BUF);
        size_t total = 0;
        const char* nl = nullptr;
        while (true) {
            if (total == buf.size()) buf.resize(buf.size() * 2);
            ssize_t n = pread(snippet_fd_, buf.data() + total, buf.size() - total,
                              (off_t)(offset + total));
            if (n < 0) { perror("pread (snippet)"); return false; }
            if (n == 0) {
                if (total == 0) {   // offset came from passages.dt — must be valid
                    std::cerr << "FATAL: pread returned 0 at offset " << offset
                              << " (doc_id " << doc_id << ") — stale collection.tsv / passages.dt"
                              << " mismatch?" << std::endl;
                    std::abort();
                }
                break;              // EOF: final line without a trailing newline
            }
            nl = (const char*)memchr(buf.data() + total, '\n', (size_t)n);
            total += (size_t)n;
            if (nl) break;
        }
        size_t len = nl ? (size_t)(nl - buf.data()) : total;   // exclude '\n', like getline
        line.assign(buf.data(), len);
        return true;
    }

    //------ generate snippet ------
    std::string generate_snippet(uint32_t doc_id, const std::vector<std::string>& query_terms) {
        std::string line;
        if (!read_passage_line(doc_id, line)) {
            return "[Snippet unavailable]";
        }

        size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            return "[Snippet unavailable]";
        }

        std::string text = line.substr(tab + 1);
        auto words = tokenize(text);

        // create query term set (for quick lookup)
        std::unordered_set<std::string> query_set(query_terms.begin(), query_terms.end());

        // mark match positions
        std::vector<bool> is_match(words.size(), false);
        std::vector<int> match_positions;

        for (size_t i = 0; i < words.size(); i++) {
            if (query_set.count(words[i]) > 0) {
                is_match[i] = true;
                match_positions.push_back(i);
            }
        }

        // intelligent window selection: find the window containing the most query terms
        int best_start = 0;
        int best_end = std::min<size_t>(SNIPPET_WINDOW * 2, words.size());
        int max_score = 0;

        if (!match_positions.empty()) {
            for (int center : match_positions) {
                int start = std::max(0, center - SNIPPET_WINDOW);
                int end = std::min<int>(words.size(), center + SNIPPET_WINDOW);

                // calculate window score (number of matching terms + density of matching terms)
                int matches = 0;
                int first_match = -1, last_match = -1;

                for (int i = start; i < end; i++) {
                    if (is_match[i]) {
                        matches++;
                        if (first_match == -1) first_match = i;
                        last_match = i;
                    }
                }

                // score = matches * 1000 - (last_match - first_match)
                int score = matches * 1000;
                if (first_match != -1 && last_match != -1) {
                    score -= (last_match - first_match);
                }

                if (score > max_score) {
                    max_score = score;
                    best_start = start;
                    best_end = end;
                }
            }
        }

        // build highlighted snippet
        std::ostringstream snippet;

        if (best_start > 0) {
            snippet << "... ";
        }

        for (int i = best_start; i < best_end; i++) {
            if (is_match[i]) {
                snippet << HIGHLIGHT_START << words[i] << HIGHLIGHT_END;
            } else {
                snippet << words[i];
            }
            if (i < best_end - 1) {
                snippet << " ";
            }
        }

        if (best_end < (int)words.size()) {
            snippet << " ...";
        }

        return snippet.str();
    }
};

#endif
