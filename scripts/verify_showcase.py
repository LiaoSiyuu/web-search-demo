#!/usr/bin/env python3
"""Build and audit the shipped fixture in a disposable directory (stdlib only)."""
import argparse
from collections import Counter, defaultdict
import itertools
import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DELIMITERS = b" \t\n\r\f.,;:!?[]{}()<>\"'`~@#$%^&*-+=|\\/_"
SPLIT = re.compile(b"[" + re.escape(DELIMITERS) + b"]+")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def tokenize(text):
    return [term for term in SPLIT.split(text.lower()) if term]


def varbyte(value):
    encoded = bytearray()
    while value >= 128:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return encoded


class ShowcaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="search-showcase-test-")
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.root = Path(cls.scratch.name)
        for directory in ("include", "src", "scripts", "examples"):
            shutil.copytree(ROOT / directory, cls.root / directory)
        shutil.copyfile(ROOT / "Makefile", cls.root / "Makefile")
        cls.run_command(["make", "-j4", "all", "DATASET=data/collection_dev.tsv",
                         "CXX=" + OPTIONS.cxx, "CXXFLAGS=" + OPTIONS.cxxflags], timeout=180)
        cls.cases = cls.root / "cases"
        cls.cases.mkdir()

    @classmethod
    def run_command(cls, args, cwd=None, input=None, check=True, timeout=60):
        result = subprocess.run(args, cwd=cwd or cls.root, input=input,
                                text=True, capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise AssertionError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
        # UBSan may report a failure without a nonzero process status.
        if "runtime error:" in result.stderr or "ERROR: AddressSanitizer" in result.stderr:
            raise AssertionError(result.stderr)
        return result

    def setUp(self):
        self.case = self.cases / self._testMethodName
        (self.case / "data").mkdir(parents=True)
        self.source = self.case / "data/collection_dev.tsv"
        shutil.copyfile(ROOT / "examples/collection_dev.tsv", self.source)
        self.build_index()

    def engine(self, name, *args, **kwargs):
        return self.run_command([str(self.root / name), *args], cwd=self.case, **kwargs)

    def build_index(self):
        self.engine("build_index")
        self.engine("merge_compress")

    def batch(self, queries, mode, snippets=False):
        (self.case / "queries.txt").write_text("\n".join(queries) + "\n", encoding="utf-8")
        args = ["--batch", "queries.txt", "--mode", str(mode), "--out", "results.tsv"]
        if snippets:
            args.append("--snippets")
        output = self.engine("query_engine", *args).stdout
        results = defaultdict(list)
        for line in (self.case / "results.tsv").read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            results[int(fields[0])].append((int(fields[1]), int(fields[2]),
                                            float(fields[3]), fields[4] if snippets else ""))
        return results, output

    def test_ranking_against_independent_oracle(self):
        corpus = [Counter(tokenize(line.split(b"\t", 1)[1]))
                  for line in self.source.read_bytes().splitlines()]
        n = len(corpus)
        lengths = [sum(document.values()) for document in corpus]
        avg = sum(lengths) / n
        terms = ["city", "water", "pizza", "brooklyn", "new", "york", "manhattan", "the", "absent"]
        rng = random.Random(417)
        queries = [[term] for term in terms] + list(itertools.combinations(terms, 2))
        queries += [rng.choices(terms, k=rng.randint(2, 7)) for _ in range(160)]
        queries += [["CITY,", "pizza!"], ["!!!"]]
        texts = [" ".join(query) for query in queries]
        df = {term.encode(): sum(term.encode() in document for document in corpus) for term in terms}

        def contribution(term, doc):
            frequency = corpus[doc][term]
            if not frequency:
                return 0.0
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5))
            return max(0.0, idf * 2.2 * frequency /
                       (1.2 * (0.25 + 0.75 * lengths[doc] / avg) + frequency))

        for mode in (0, 1):
            results, output = self.batch(texts, mode)
            if mode == 1:
                match = re.search(r"block-max chunks skipped = (\d+)", output)
                self.assertIsNotNone(match)
                self.assertGreater(int(match[1]), 0, "fixture must exercise block skipping")
            self.assertTrue(set(results) <= set(range(len(queries))))
            for qi, text in enumerate(texts):
                tokens = tokenize(text.encode())
                reference = {}
                for doc in range(n):
                    if mode == 0 and not all(term in corpus[doc] for term in tokens):
                        continue
                    score = sum(contribution(term, doc) for term in tokens)
                    if score > 0:
                        reference[doc] = score
                best = sorted(reference.values(), reverse=True)[:10]
                got = results[qi]
                with self.subTest(mode=mode, query=text):
                    self.assertEqual(len(got), len(best))
                    self.assertEqual(len({pid for _, pid, _, _ in got}), len(got))
                    for i, (rank, pid, score, _) in enumerate(got):
                        self.assertEqual(rank, i + 1)
                        self.assertIn(pid, reference)
                        self.assertTrue(math.isfinite(score))
                        self.assertAlmostEqual(score, reference[pid], delta=1e-6)
                        self.assertAlmostEqual(score, best[i], delta=1e-6)
        print(f"Validated {len(queries)} queries in each mode (per-document scores and top-k).")

    def test_merge_across_shard_and_document_boundaries(self):
        original = {name: (self.case / name).read_bytes() for name in ("index.idx", "lexicon.lex")}
        corpus = [Counter(tokenize(line.split(b"\t", 1)[1]))
                  for line in self.source.read_bytes().splitlines()]
        records = [(term, doc, frequency) for doc, counts in enumerate(corpus)
                   for term, frequency in sorted(counts.items())]
        for path in (self.case / "temp").glob("temp_*.bin"):
            path.unlink()
        shard_count = 0
        for start in range(0, len(records), 127):
            postings = defaultdict(list)
            for term, doc, frequency in records[start:start + 127]:
                postings[term].append((doc, frequency))
            blob = bytearray(b"TMP2")
            for term, pairs in sorted(postings.items()):
                blob += struct.pack("=H", len(term)) + term + struct.pack("=I", len(pairs))
                previous = 0
                for doc, frequency in pairs:
                    blob += varbyte(doc - previous) + varbyte(frequency)
                    previous = doc
            (self.case / "temp" / f"temp_{shard_count}.bin").write_bytes(blob)
            shard_count += 1
        self.assertGreater(shard_count, 1)
        self.engine("merge_compress")
        for name, expected in original.items():
            self.assertEqual((self.case / name).read_bytes(), expected, name)
        print(f"Validated byte-identical merge output from {shard_count} shards.")

    def test_score_bounds(self):
        self.engine("verify_blockmax", "1")
        self.run_command([sys.executable, str(self.root / "scripts/check_maxscore.py")], cwd=self.case)
        # A mismatched corpus must not silently pass the reference check.
        self.source.write_bytes(self.source.read_bytes().split(b"\n", 1)[1])
        result = self.run_command([sys.executable, str(self.root / "scripts/check_maxscore.py")],
                                  cwd=self.case, check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_queries_do_not_reuse_counters(self):
        _, one = self.batch(["city"], 1)
        _, mixed = self.batch(["city", "absent", "!!!"], 1)
        pattern = r"scored (\d+) / (\d+) postings"
        self.assertEqual(re.search(pattern, one).groups(), re.search(pattern, mixed).groups())

    def test_snippets_use_row_offsets_not_external_ids(self):
        rows = []
        for i in range(40):
            pid = 5000 + i * 7
            words = f"marker{i} café " + ("pizza " * (i + 1) if i < 12 else "filler")
            rows.append(f"{pid}\t{words}")
        self.source.write_text("\n".join(rows), encoding="utf-8")  # final row has no newline
        self.build_index()
        for mode in (0, 1):
            results, _ = self.batch(["pizza"], mode, snippets=True)
            self.assertEqual(len(results[0]), 10)
            for _, pid, _, snippet in results[0]:
                self.assertIn(f"marker{(pid - 5000) // 7}", ANSI.sub("", snippet))
                self.assertIn("café", snippet)

    def test_cli_input_validation_and_eof(self):
        self.engine("query_engine", input="", timeout=3)
        self.engine("query_engine", input="city\n", timeout=3)
        for mode in ("2", "abc", "-1"):
            result = self.engine("query_engine", "--batch", "missing.txt", "--mode", mode,
                                 "--out", "results.tsv", check=False)
            self.assertEqual(result.returncode, 1)
        for stride in ("0", "-1", "abc", "1x", "4294967296"):
            self.assertEqual(self.engine("verify_blockmax", stride, check=False).returncode, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--cxxflags", default="-std=c++17 -O2 -Wall -Wextra")
    OPTIONS, remaining = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
