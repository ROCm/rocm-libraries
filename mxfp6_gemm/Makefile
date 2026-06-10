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

# HYBRID experiment: A-LDS deep-K + B streamed direct HBM->reg (coalesced preshuffle_B,
# register ring double-buffer). Beats the pure-LDS baseline +16~24% @8192^3 (B's LDS
# round-trip + sub-head ds_read stall removed). test_hybrid = correctness + bench sweep;
# test_hybrid_grid = partial-grid / non-square / odd-K correctness gate.
test_hybrid: test_hybrid.cpp $(HEADERS) mxfp6_lds_hybrid.hpp
	$(HIPCC) $(HIPFLAGS) $< -o $@

test_hybrid_grid: test_hybrid_grid.cpp $(HEADERS) mxfp6_lds_hybrid.hpp
	$(HIPCC) $(HIPFLAGS) $< -o $@

# occ2 compile-gate + bench driver (Phase 1). Instantiates the production lds_gemm_db
# with MIN_OCC=2 + 8-acc tile (128x256). Does NOT touch test_lds.cpp / mxfp6_lds.hpp.
test_lds_occ2: test_lds_occ2.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

# occ2 axis-3 drip kernel (Phase 2): mxfp6_lds_occ2.hpp forks lds_gemm_db DB path
# + drips next-tile buffer_loads across MFMA quartets. Does NOT touch production.
test_occ2_drip: test_occ2_drip.cpp $(HEADERS) mxfp6_lds_occ2.hpp
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
	rm -f test_pipeline_v18 test_pipeline_v17 test_lds test_lds_occ2 test_occ2_drip occ_probe test_ws test_pp8 profile_lds test_reference \
	      *.bc *.hipi *.s *.o *-resolution.txt *.0.hipv4-* *.0.host-*

.PHONY: test_cpu test_gpu clean
