HIPCC = hipcc
HIPFLAGS = -std=c++17 -O2 --offload-arch=gfx950

HEADERS = mxfp6_types.hpp mxfp6_preprocess.hpp mxfp6_reference.hpp mxfp6_asm_utils.hpp \
          mxfp6_lds.hpp mxfp6_lds_hybrid.hpp mxfp6_dispatch.hpp

# Unified MXFP6 GEMM. ONE kernel paradigm — hybrid drip-A (A staged deep-K in LDS double-
# buffered + B streamed coalesced HBM->VGPR ring + dripped A loads + RDB barrier + tiled-
# scale), shape-routed by mxfp6_dispatch.hpp: 256x256 (16-acc sweet spot) for CU-filling
# shapes, 128x256 for WG-starved small-M. Beats the old register-direct + pure-LDS v18
# dispatcher on all 12 benchmarked shapes (+6~98%). test_dispatch = end-to-end correctness
# (fresh-alloc + CPU ref) + 12-shape benchmark.
test_dispatch: test_dispatch.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

clean:
	rm -f test_dispatch *.out

.PHONY: clean
