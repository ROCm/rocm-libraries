#pragma once
#include <algorithm>
// INTERNAL: templated GEMM launch for the unified hybrid drip-A kernel. The public,
// type-erased entry points (mxfp6::gemm / mxfp6::choose_tile) live in mxfp6/gemm.hpp and
// are defined in src/gemm.cpp, which instantiates this template for F32/F16/BF16.
//
// ONE kernel paradigm (hybrid drip-A: A staged deep-K in LDS double-buffered + B streamed
// coalesced HBM->VGPR ring + dripped A loads + RDB barrier + tiled-scale), routed by shape:
//   * 256x256 (16-acc sweet spot) — workhorse for CU-filling shapes.
//   * 128x256 (8-acc)             — WG-starved small-M, to fill idle CUs.
// Same lds_gemm_hybrid_dripA kernel for both (tile-general); only the tile args differ.
#include "kernel.hpp"
#include "mxfp6/gemm.hpp"  // TileChoice, choose_tile (declarations)
namespace mxfp6 {
namespace detail {

// Kp = K padded to a multiple of K_TILE(=192). Caller must have tiled the scales with
// choose_tile(...).MPW/NPW and preshuffled B (preshuffle_B).
template <typename OutT>
inline void dispatch_gemm(int M, int N, int Kp, const void* dA, const void* dBsh,
                          const uint8_t* dsA, const uint8_t* dsB, OutT* dD, int A_row_bytes,
                          int B_row_bytes) {
    constexpr int KT = K_TILE;
    TileChoice tc = choose_tile(M, N);
    dim3 blk(256);
    int kit = Kp / 64;
    if (tc.MT == 128) {
        // occ1. (The 8-acc 128x256 tile CAN be forced to occ2 via MIN_OCC=2 — 251 VGPR,
        // 0 spill — but in STEADY STATE that gives ~0% on small-M: a 256-WG shape has no
        // 2nd WG to fill the occ2 slot. An earlier +6~10% reading was a warm-up artifact
        // of always benchmarking occ1 first on a still-ramping GPU. Kept at occ1.)
        dim3 g(M / 128, N / 256);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 1, 0, true, OutT>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes);
    } else {
        dim3 g(M / 256, N / 256);
        int lds = 2 * (256 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<256, 256, KT, 2, 2, 1, 0, true, OutT>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes);
    }
}

}  // namespace detail
}  // namespace mxfp6
