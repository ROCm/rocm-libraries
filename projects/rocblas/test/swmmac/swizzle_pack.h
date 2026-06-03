// swizzle_pack.h — RDNA4 BF16 SWMMAC Physical Swizzle Packing
// TriQChem proprietary asset: One-Hot DOE reverse-engineered dataflow mapping
//
// Architecture: Pure outer-product engine (verified by 2D DOE)
//   A: column-mapped, broadcast to ALL rows
//      lane = col_group*8 + col_offset  (lane 0..7→col 0..7, 8..15→col 8..15)
//      reg  = k_position within K-stride
//   B: row-mapped, isolated per output lane
//      lane = row  (0..15 for rows 0..15)
//      reg  = k_position
//
//   C[row][col] = Σ_k A[col][k] * B[row][k] (outer product accumulation)
//
// K-axis folding across lanes 0..15 (1st K-half) and 16..31 (2nd K-half)
// DOE confirmed: A[lane=L][0] → output lane 0, elem[L] (+16 unique contribution)
//                B[lane=L] → only affects output lane L (row isolation)
//
// Per-tile data layout (flat array index = cld * stride):
//   A_per_tile = K/16 * 32 lanes * 8 regs = K/2 * 8 values
//   B_per_tile = K/16 * 32 lanes * 16 regs = K/2 * 16 values

#pragma once
#include <cstdint>
#include <cstring>

namespace swizzle {

// A: column data, broadcast to all rows via outer-product engine
// col: 0..15 (output column index within 16x16 tile)
// k:   0..K-1 (K-axis position in source matrix)
// Returns physical lane (0..31) and register (0..7)
inline void a_physical(int col, int k, int& lane, int& reg) {
    int col_group = col / 8;           // 0 or 1 (cols 0..7 vs 8..15)
    int col_off   = col % 8;           // 0..7 within group
    int k_half    = k / 16;            // 0 or 1 (1st/2nd K-half)
    lane = col_group * 8 + col_off + k_half * 16;
    reg  = (k % 16) / 2;              // 2 K elements per register pair
}

// B: row data, isolated per output lane
// row: 0..15 (output row index within 16x16 tile)
// k:   0..K-1
// Returns physical lane (0..31) and register (0..15)
inline void b_physical(int row, int k, int& lane, int& reg) {
    lane = row % 16;                   // row 0..15 → lane 0..15
    reg  = k % 16;                     // 16 registers cover K positions (stride-4 via 2:4 sparsity)
}

} // namespace swizzle
