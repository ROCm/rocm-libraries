CXX = g++
HIPCC = hipcc
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
HIPFLAGS = -std=c++17 -O2 --offload-arch=gfx950

HEADERS = mxfp6_types.hpp mxfp6_preprocess.hpp mxfp6_reference.hpp mxfp6_asm_utils.hpp \
          mxfp6_lds.hpp

# Production MXFP6 GEMM (v18): V17 dispatcher (tile-shrink, V2 mixed accumulator,
# depth-1 prefetch, L2 swizzle) + LDS deep-K paradigm as a new tile candidate.
# The cost model routes large CU-filling aligned shapes to the LDS kernel (+1.6~3.9%)
# and everything else to the V17 tiles. Runs correctness + dispatch benchmark.
test_pipeline_v18: test_pipeline_v18.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# Previous production (V17, register-direct, no LDS). Kept as the LDS-free baseline.
test_pipeline_v17: test_pipeline_v17.cpp $(filter-out mxfp6_lds.hpp, $(HEADERS))
	$(HIPCC) $(HIPFLAGS) $< -o $@

# LDS deep-K kernel dev/sweep driver (paradigm experiments: depth, tile, occ).
test_lds: test_lds.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# CPU reference (ground-truth validator, no HIP).
test_reference: test_reference.cpp $(filter-out mxfp6_asm_utils.hpp mxfp6_lds.hpp, $(HEADERS))
	$(CXX) $(CXXFLAGS) -o $@ $<

test_cpu: test_reference
	./test_reference

test_gpu: test_pipeline_v18
	./test_pipeline_v18

clean:
	rm -f test_pipeline_v18 test_pipeline_v17 test_lds test_reference

.PHONY: test_cpu test_gpu clean
