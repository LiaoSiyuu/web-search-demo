# Compressed Posting-List Search Engine

Single-threaded C++17 search engine with BM25 ranking, AND/OR queries, compressed
posting lists, and MaxScore/block-max pruning.

## Structure

```text
include/      Index writer, shard merger, posting-list reader, query processor
src/          Build, merge, query, and verification executables
scripts/      Correctness checks
examples/     Synthetic corpus
benchmarks/   Query list
```

Pipeline: `collection.tsv → build_index → merge_compress → query_engine`.

## Run

Requires macOS or Linux, a C++17 compiler, Make, and Python 3.9+.
The current remote requires repository access and a GitHub SSH key.

```sh
git clone git@github.com:LiaoSiyuu/web-search-demo.git
cd web-search-demo

mkdir -p data
cp examples/collection_dev.tsv data/collection_dev.tsv
make -B -j4 all DATASET=data/collection_dev.tsv
./build_index
./merge_compress
./query_engine
```

Enter `city water`, choose `0` for AND or `1` for OR, and type `quit` to exit.

## Test

```sh
make test
```

Runs isolated checks for ranking, shard merging, score bounds, snippets, and CLI
behavior. No dataset download or Python packages are needed.

## MS MARCO data (optional)

Use the v1 passage collection. See Microsoft's [download page and terms](https://microsoft.github.io/msmarco/Datasets.html).
Allow several GB for the archive, extracted corpus, and index.

```sh
mkdir -p data
curl --fail --location --retry 3 --continue-at - \
  --output data/collection.tar.gz \
  https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
gzip -t data/collection.tar.gz
tar -xzf data/collection.tar.gz -C data collection.tsv
head -n 10000 data/collection.tsv > data/collection_dev.tsv
```

Repeat the run commands starting at `make -B` to index these 10,000 passages.
For the full corpus, use `DATASET=data/collection.tsv`. Always rebuild both index
stages after changing the corpus. Use trusted, locally built index files.

## License

See [LICENSE](LICENSE) for the existing usage restrictions.
