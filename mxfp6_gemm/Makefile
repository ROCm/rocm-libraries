CXX = g++
HIPCC = hipcc
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
HIPFLAGS = -std=c++17 -O2 --offload-arch=gfx950

HEADERS = mxfp6_types.hpp mxfp6_preprocess.hpp mxfp6_reference.hpp mxfp6_asm_utils.hpp \
          mxfp6_lds.hpp

PP8_HEADERS = mxfp6_types.hpp mxfp6_preprocess.hpp mxfp6_reference.hpp mxfp6_asm_utils.hpp \
          mxfp6_lds.hpp mxfp6_pp8.hpp

# Production MXFP6 GEMM (v18): V17 dispatcher (tile-shrink, V2 mixed accumulator,
# depth-1 prefetch, L2 swizzle) + LDS deep-K paradigm as a new tile candidate.
# The cost model routes large CU-filling aligned shapes to the LDS kernel (+1.6~3.9%)
# and everything else to the V17 tiles. Runs correctness + dispatch benchmark.
test_pipeline_v18: test_pipeline_v18.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# Previous production (V17, register-direct, no LDS). Kept as the LDS-free baseline.
test_pipeline_v17: test_pipeline_v17.cpp $(filter-out mxfp6_lds.hpp, $(HEADERS))
	$(HIPCC) $(HIPFLAGS) $< -o $@

# LDS deep-K kernel dev/sweep driver (paradigm experiments: depth, tile, occ + F16/BF16).
test_lds: test_lds.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# MS0: 8-wave scaffold (Idea 2 clean-room). 256x256, 32x32x64 MFMA, 4->8 waves,
# occ-2 target, single-buffer K=64, NO ping-pong. Correctness + perf-ref driver.
test_pp8: test_pp8.cpp $(PP8_HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# Warp-specialization experiment (producer/consumer, LDS flag ring). DISPROVEN — kept
# as a documented reference (loses to symmetric double-buffer; see HANDOFF Step 24).
test_ws: test_ws.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# Single-dispatch profiling driver for the production LDS kernel (PMC/ATT/RCV capture).
# Usage: ./profile_lds [warmup] [repeat]  (0 1 = one dispatch; 3 10 = timed benchmark).
profile_lds: profile_lds.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# CPU reference (ground-truth validator, no HIP).
test_reference: test_reference.cpp $(filter-out mxfp6_asm_utils.hpp mxfp6_lds.hpp, $(HEADERS))
	$(CXX) $(CXXFLAGS) -o $@ $<

test_cpu: test_reference
	./test_reference

test_gpu: test_pipeline_v18
	./test_pipeline_v18

clean:
	rm -f test_pipeline_v18 test_pipeline_v17 test_lds test_ws test_pp8 profile_lds test_reference \
	      *.bc *.hipi *.s *.o *-resolution.txt *.0.hipv4-* *.0.host-*

.PHONY: test_cpu test_gpu clean
