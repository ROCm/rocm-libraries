#pragma once
// Unified MXFP6 GEMM dispatcher — ONE kernel paradigm (hybrid drip-A: A staged deep-K in
// LDS double-buffered + B streamed coalesced HBM->VGPR ring + dripped A loads + RDB barrier
// + tiled-scale), routed across the (M,N) space by a single rule:
//
//   * 256x256 tile (16-acc arithmetic-intensity sweet spot) — the workhorse for every shape
//     whose 256x256 grid fills the machine (WG >= #CU). Beats the old register-direct +
//     pure-LDS dispatcher by +14~24% on ALL aligned shapes incl. non-pow2 N (no separate
//     mixed-acc 18/20-acc tiles needed — the hybrid paradigm wins outright).
//   * 128x256 tile (8-acc) — ONLY for WG-starved small-M shapes (256x256 grid < #CU). Halves
//     the M-tile to double the WG count and fill idle CUs; +20~98% over 256x256 there.
//
// Same `lds_gemm_hybrid_dripA` kernel for both (tile-general after the cooperative-load fix);
// only the tile template args differ. Validated bit-exact (fresh-alloc) + best-per-shape on
// 12 shapes vs the v18 dispatcher (hybrid wins every shape).
#include "mxfp6_lds_hybrid.hpp"
namespace mxfp6 {

struct TileChoice { int MT, NT, MPW, NPW; };

// MPW/NPW (per-wave 32-blocks, 2x2 waves) drive the host-side tiled-scale grouping.
inline TileChoice choose_tile(int M, int N, int CU = 256) {
    int wg256 = (M / 256) * (N / 256);
    if (wg256 < CU && (M % 128) == 0 && (N % 256) == 0)
        return {128, 256, 2, 4};   // WG-starved small-M: fill CUs
    return {256, 256, 4, 4};       // workhorse: 16-acc sweet spot
}

// Device launch of the chosen tile. Kp = K padded to a multiple of K_TILE(=192); the caller
// must have tiled the scales with choose_tile(...).MPW/NPW and preshuffled B (preshuffle_B).
template <typename OutT>
inline void dispatch_gemm(int M, int N, int Kp, const void* dA, const void* dBsh,
                          const uint8_t* dsA, const uint8_t* dsB, OutT* dD,
                          int A_row_bytes, int B_row_bytes) {
    constexpr int KT = 192;
    TileChoice tc = choose_tile(M, N);
    dim3 blk(256);
    int kit = Kp / 64;
    if (tc.MT == 128) {
        // MIN_OCC=2: the 8-acc 128x256 tile fits in 251 VGPR (123 arch + 128 AGPR, 0 spill),
        // so two waves/SIMD — gives WG-boundary pipelining (one WG's epilogue HBM drain
        // overlaps the next WG's compute). +0.6~9.7% on small-M, biggest at 256-WG shapes
        // (2048x4096). The 256x256 path can't do this (507 VGPR, occ1-locked).
        dim3 g(M / 128, N / 256);
        int lds = 2 * (128 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<128, 256, KT, 2, 2, 2, 0, true, OutT>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes);
    } else {
        dim3 g(M / 256, N / 256);
        int lds = 2 * (256 * (KT * 6 / 8));
        lds_gemm_hybrid_dripA<256, 256, KT, 2, 2, 1, 0, true, OutT>
            <<<g, blk, lds>>>(dA, dBsh, dsA, dsB, dD, N, kit, A_row_bytes, B_row_bytes);
    }
}
} // namespace mxfp6
