#pragma once
// Public API for the unified MXFP6 GEMM (libmxfp6gemm).
//
// One kernel paradigm (hybrid drip-A: A staged deep-K in LDS double-buffered + B streamed
// coalesced HBM->VGPR ring + dripped A loads + RDB barrier + tiled-scale), shape-routed:
//   * 256x256 tile (16-acc) for shapes whose grid fills the machine (WG >= #CU)
//   * 128x256 tile (8-acc)  for WG-starved small-M shapes (fills idle CUs)
//
// This header is device-free (plain declarations) — a host translation unit can include it
// and link libmxfp6gemm without a HIP compiler. Inputs are pre-quantized DEVICE buffers:
// quantize A / preprocess+preshuffle B / tile the scales with mxfp6/preprocess.hpp first,
// using choose_tile(M,N).{MPW,NPW} for the scale grouping. Kp = K padded up to a multiple
// of the K-tile (192).
#include <cstdint>

namespace mxfp6 {

enum class OutType { F32, F16, BF16 };

// K is processed in deep tiles of K_TILE; callers pad K up to kpad(K) (Kp) and pass that as the
// kernel's K. The padded K-tail must be zero on at least one operand (B in the pad-B-only recipe)
// so its contribution is nulled — see mxfp6/preprocess.hpp for the host-side padding helpers.
constexpr int K_TILE = 192;
inline int kpad(int K) {
    return (K + K_TILE - 1) / K_TILE * K_TILE;
}

// Tile chosen for (M,N). MPW/NPW = per-wave 32-block counts (2x2 waves); the host scale
// tiling (tile_scale) must use these for A and B respectively.
struct TileChoice {
    int MT, NT, MPW, NPW;
};
TileChoice choose_tile(int M, int N);

// D[M,N] = A[M,K] * B[K,N], MXFP6 E2M3 inputs + per-32-block E8M0 scales.
//   dA   : packed MXFP6 A (row-major), A_row_bytes per row
//   dBsh : preshuffled MXFP6 B (preshuffle_B layout)
//   dsA  : tiled A scales (tile_scale with MPW)   dsB: tiled B scales (tile_scale with NPW)
//   dD   : output buffer of element type `ot` (float / __half / __hip_bfloat16)
void gemm(OutType ot, int M, int N, int Kp, const void* dA, const void* dBsh, const uint8_t* dsA,
          const uint8_t* dsB, void* dD, int A_row_bytes, int B_row_bytes);

}  // namespace mxfp6
