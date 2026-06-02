CXX = g++
HIPCC = hipcc
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
HIPFLAGS = -std=c++17 -O2 --offload-arch=gfx950

HEADERS = mxfp6_types.hpp mxfp6_preprocess.hpp mxfp6_reference.hpp mxfp6_asm_utils.hpp

# Production MXFP6 GEMM: dispatcher + tuned pipeline kernel (tile-shrink, V2 mixed
# accumulator, depth-1 prefetch, L2-aware swizzle). Runs correctness + benchmark.
test_pipeline_v17: test_pipeline_v17.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# CPU reference (ground-truth validator, no HIP).
test_reference: test_reference.cpp $(filter-out mxfp6_asm_utils.hpp, $(HEADERS))
	$(CXX) $(CXXFLAGS) -o $@ $<

test_cpu: test_reference
	./test_reference

test_gpu: test_pipeline_v17
	./test_pipeline_v17

clean:
	rm -f test_pipeline_v17 test_reference

.PHONY: test_cpu test_gpu clean
