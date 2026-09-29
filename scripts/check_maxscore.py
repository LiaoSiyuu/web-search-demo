#!/usr/bin/env python3
"""
Independent check of the binary lexicon's per-term max_score.

For a set of spot terms (including a stop-word, whose negative idf must give
max_score 0), recompute max_i( max(0, idf*tf_i) ) from scratch over the corpus
and compare to the value the index stored. Reuses the byte-faithful tokenizer.
"""
import argparse, math, struct, sys

K1, B = 1.2, 0.75
DEV = "data/collection_dev.tsv"
PASSAGES_DT = "passages.dt"
LEXICON = "lexicon.lex"
SPOT_TERMS = ["the", "city", "water", "pizza", "brooklyn", "new", "york", "manhattan"]

DELIMS = set(b" \t\n\r\f.,;:!?[]{}()<>\"'`~@#$%^&*-+=|\\/_")

def tokenize(text_bytes):
    toks, w = [], bytearray()
    for c in text_bytes:
        if c in DELIMS:
            if w: toks.append(bytes(w).decode("latin-1")); w = bytearray()
        else:
            w.append(c + 32 if 65 <= c <= 90 else c)
    if w: toks.append(bytes(w).decode("latin-1"))
    return toks

def load_passages_dt(path):
    with open(path, "rb") as f: blob = f.read()
    count, avg = struct.unpack_from("=Id", blob, 0)
    lengths = [0]*count
    off = 12
    for i in range(count):
        _pid, length, _foff = struct.unpack_from("=IIQ", blob, off)
        lengths[i] = length; off += 16
    return count, avg, lengths

def load_lexicon(path):
    # binary LEXB: ["LEXB"][term_count u32], then per term
    #   [term_len u16][term][offset u64][length u64][doc_count u32][max_score f64]
    # host-endian (little-endian on this machine). Term bytes decoded latin-1 so a
    # UTF-8 byte inside a term is never mistaken for a field separator.
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"LEXB":
        print("lexicon is not LEXB binary format — rebuild the index"); sys.exit(2)
    (count,) = struct.unpack_from("=I", blob, 4)
    lex, off = {}, 8
    for _ in range(count):
        (tlen,) = struct.unpack_from("=H", blob, off); off += 2
        term = blob[off:off + tlen].decode("latin-1"); off += tlen
        _offset, _length, doc_count, max_score = struct.unpack_from("=QQId", blob, off)
        off += 28                                  # u64 + u64 + u32 + f64, packed
        lex[term] = (doc_count, max_score)
    return lex

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEV, help="corpus used to build the current index")
    args = parser.parse_args()
    N, avg, lengths = load_passages_dt(PASSAGES_DT)
    if N == 0 or not math.isfinite(avg) or avg <= 0:
        sys.exit("MAX_SCORE CHECK FAIL: empty or invalid passage table")
    lex = load_lexicon(LEXICON)
    want = set(SPOT_TERMS)
    tf = {t: {} for t in want}
    docid = 0
    with open(args.corpus, "rb") as f:
        for raw in f:
            line = raw.rstrip(b"\n")
            tab = line.find(b"\t")
            if tab < 0: continue
            toks = tokenize(line[tab+1:])
            if docid >= N or len(toks) != lengths[docid]:
                sys.exit(f"MAX_SCORE CHECK FAIL: corpus/index length mismatch at row {docid}")
            for t in want:
                c = toks.count(t)
                if c: tf[t][docid] = c
            docid += 1

    if docid != N:
        sys.exit(f"MAX_SCORE CHECK FAIL: corpus has {docid} rows, index has {N}")
    if not math.isclose(avg, sum(lengths) / N, rel_tol=1e-12):
        sys.exit("MAX_SCORE CHECK FAIL: invalid average passage length")

    fail = False
    for t in SPOT_TERMS:
        df = len(tf[t])
        if df == 0 and t not in lex:
            print(f"  ok  {t:10s} absent from both corpus and lexicon")
            continue
        if lex.get(t, (None, None))[0] != df:
            print(f"  BAD {t}: corpus/index document frequency mismatch")
            fail = True
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5))
        ref = 0.0
        for d, fr in tf[t].items():
            K = K1 * (1 - B + B * lengths[d] / avg)
            c = max(0.0, idf * (K1 + 1) * fr / (K + fr))
            if c > ref: ref = c
        stored = lex.get(t, (None, None))[1]
        ok = stored is not None and abs(ref - stored) <= 1e-9 * max(1.0, abs(ref))
        if not ok: fail = True
        note = " (stop-word, idf<0 -> 0)" if idf < 0 else ""
        print(f"  {'ok ' if ok else 'BAD'} {t:10s} df={df:6d} idf={idf:+.4f} ref_max={ref:.9f} stored={stored}{note}")
    print("\n" + ("MAX_SCORE CHECK FAIL" if fail else "MAX_SCORE CHECK PASS"))
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
