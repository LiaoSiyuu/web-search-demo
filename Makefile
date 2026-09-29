# MS MARCO Passage Search Engine — build
CXX      = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
INCLUDES = -Iinclude

# Optional dataset override: build against a different collection.tsv without
# editing config.h, e.g.  make all DATASET=data/collection_dev.tsv
# The value is injected as a string literal via -DDATA_SOURCE_PATH='"..."'.
DATASET ?=
ifneq ($(strip $(DATASET)),)
DATASET_FLAGS = -DDATA_SOURCE_PATH='"$(DATASET)"'
endif

# Build all executables.
all: build_index merge_compress query_engine verify_blockmax

# Stage 1: index build (collection.tsv -> temp_*.bin + passages.dt)
build_index: src/build_index.cpp include/*.h
	$(CXX) $(CPPFLAGS) $(DATASET_FLAGS) $(CXXFLAGS) $(INCLUDES) -o build_index src/build_index.cpp

# Stage 2+3 (fused): single k-way merge streamed straight into the compressed index
merge_compress: src/merge_compress.cpp include/*.h
	$(CXX) $(CPPFLAGS) $(DATASET_FLAGS) $(CXXFLAGS) $(INCLUDES) -o merge_compress src/merge_compress.cpp

# Stage 4: query engine (interactive + --batch harness)
query_engine: src/query_engine.cpp include/*.h
	$(CXX) $(CPPFLAGS) $(DATASET_FLAGS) $(CXXFLAGS) $(INCLUDES) -o query_engine src/query_engine.cpp

# Verify every per-chunk max_score is a valid query-time BM25 upper bound
verify_blockmax: src/verify_blockmax.cpp include/*.h
	$(CXX) $(CPPFLAGS) $(DATASET_FLAGS) $(CXXFLAGS) $(INCLUDES) -o verify_blockmax src/verify_blockmax.cpp

# Builds and runs in a temporary directory; leaves local datasets/indexes alone.
test:
	python3 scripts/verify_showcase.py --cxx "$(CXX)" --cxxflags "$(CXXFLAGS)"

clean:
	rm -f build_index merge_compress query_engine verify_blockmax
	rm -rf temp/
	rm -f *.idx *.lex *.dt

.PHONY: all clean test
