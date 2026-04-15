// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Mamba-2 SSD (State Space Decomposition) — ck_tile example
// Problem definition and host arguments.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// ---------------------------------------------------------------------------
// SSD problem shape
//   X       [B, EH, D, C, L]
//   DeltaA  [B, EH, C, L]
//   Delta   [B, EH, C, L]
//   B_mat   [B, G,  N, C, L]
//   C_mat   [B, G,  N, C, L]
//   D_param [EH, D]
//   Y       [B, EH, D, C, L]
//   Fstate  [B, EH, D, N]
// ---------------------------------------------------------------------------
struct SsdHostArgs
{
    // Pointers
    const void* p_x;       // [B, EH, D, C, L]
    const void* p_delta_a; // [B, EH, C, L]
    const void* p_delta;   // [B, EH, C, L]
    const void* p_b_mat;   // [B, G, N, C, L]
    const void* p_c_mat;   // [B, G, N, C, L]
    const void* p_d_param; // [EH, D]
    const void* p_z;       // [B, EH, D, C, L] or nullptr
    void* p_y;             // [B, EH, D, C, L]
    void* p_fstate;        // [B, EH, D, N]

    // Problem sizes
    index_t batch;     // B
    index_t groups;    // G
    index_t exp_heads; // EH = E * H
    index_t chunks;    // C
    index_t chunk_len; // L
    index_t head_dim;  // D
    index_t state_dim; // N

    // Derived
    CK_TILE_HOST index_t grp_ratio() const { return exp_heads / groups; }
    CK_TILE_HOST index_t total_heads() const { return batch * exp_heads; }
};

// Tile configuration for the SSD kernel.
// The GEMM tiles are chosen for the fixed sizes L=128, D=64, N=128.
struct SsdTileConfig
{
    // IntraBMM1: [L, L] = C^T @ B, GEMM M=L=128, N=L=128, K=N=128
    static constexpr index_t IntraBMM1_MTile = 64;
    static constexpr index_t IntraBMM1_NTile = 64;
    static constexpr index_t IntraBMM1_KTile = 32;

    // IntraBMM2: [L, D] = Pre @ X^T, GEMM M=L=128, N=D=64, K=L=128
    static constexpr index_t IntraBMM2_MTile = 64;
    static constexpr index_t IntraBMM2_NTile = 64;
    static constexpr index_t IntraBMM2_KTile = 32;

    // InterBMM1: [N, D] = Pre @ X^T, GEMM M=N=128, N=D=64, K=L=128
    static constexpr index_t InterBMM1_MTile = 64;
    static constexpr index_t InterBMM1_NTile = 64;
    static constexpr index_t InterBMM1_KTile = 32;

    // InterBMM2: [L, D] = C^T @ State, GEMM M=L=128, N=D=64, K=N=128
    static constexpr index_t InterBMM2_MTile = 64;
    static constexpr index_t InterBMM2_NTile = 64;
    static constexpr index_t InterBMM2_KTile = 32;
};

} // namespace ck_tile
