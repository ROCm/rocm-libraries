// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_descriptors.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_weight_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp"
#include "ck_tile/ops/direct_convolution/utils/transpose_lds_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/mfma.hpp"
#include "ck_tile/ops/direct_convolution/utils/launch_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/tensor/buffer_view.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/load_tile.hpp"
#include "ck_tile/core/tensor/store_tile.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile::direct_conv::grouped_4c_tile::v3
{

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 16 columns wide.
// Each wave handles 4 output columns.
constexpr int WARP_Q = 4;

// Kernel configuration parameters.

struct Config
{
    // waves_c64 — channel (group) dimension
    // Each wave computes outputs for 64 input channels worth of groups. 
    // If each group has, e.g., exactly 4 channels, 64 channels -> 16 groups per workgroup.
    // This number tells many waves of 64 channels are processed by one workgroup (thread block).
    int waves_c64;

    // waves_q4 — spatial output column dimension
    // Each wave handles 4 output columns (WARP_Q = 4)
    // This number tells how many waves of 4 ouput columns are processed by one workgroup (thread block).
    int waves_q4;

    // Filter width & height
    int kh = 3;
    int kw = 3;

    // Batch folding:
    // The batch dimension is folded into the grid by a factor of n_fold, meaning each block processes n_fold batches.
    // The grid for launching the kernel becomes 
    //      dim3(ceil(out_W / block_q) * n_fold,   ceil(C / block_c),   ceil(N / n_fold))
    // This means that W-tiles are interleaved with n_fold groups of images
    // The n_fold number tells how many image slots are packed into one X-dimension stride.
    // By spreading images into the X dimension rather than only Z, 
    // the GPU can schedule blocks from different images onto different CUs without 
    // waiting for one image's channel tiles to finish first.
    int n_fold = 8;

    // Number of channels per convolution group.
    int channels_per_group = 4;

    // Uniform accessor for shared code (alias for channels_per_group).
    constexpr int group_size() const { return channels_per_group; }

    Direction direction = Direction::Fprop;

    // Swizzle pattern - by default no explicit swizzle.
    SwizzleType swizzle_type = SwizzleType::None;

    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;

    // Total number of waves.
    constexpr int num_waves() const { return waves_c64 * waves_q4; }

    // Tile size in the channel dimension: number of input channels processed by one workgroup.
    constexpr int block_c() const { return waves_c64 * 64; }

    // Tile size in the output column dimension.
    constexpr int block_q() const { return waves_q4 * WARP_Q; }

    // Number of conv groups processed by one workgroup.
    constexpr int block_groups() const { return waves_c64 * 16; }

    // Number of threads per workgroup (thread block).
    constexpr int block_size() const { return num_waves() * WAVE_SIZE; }

    std::string GetName() const
    {
        std::string swz = (swizzle_type == SwizzleType::XOR) ? "swizzleXOR" : "noswizzle";
        std::string waves_c64_str = "_waves_c64_" + std::to_string(waves_c64);
        std::string waves_q4_str = "_waves_q4_" + std::to_string(waves_q4);
        std::string base = "v3_grouped_4c_" + swz + waves_c64_str + waves_q4_str;
        if (epilogue == EpilogueType::RegistersToGlobalMemory)
            return  base + "_skip_lds_epilogue";
        else
            return base + "_lds_epilogue";
    }
};

// All instantiated configurations. The first valid config is expected to be the fastest.
//
// Layout: 4 variant groups × 10 configs each = 40 configs total.
// Each group has 5 Dgrad (indices 0-4) + 5 Fprop (indices 5-9) configs:
//   waves_c64=2,waves_q4=8 / waves_c64=2,waves_q4=4 / waves_c64=2,waves_q4=2 /
//   waves_c64=2,waves_q4=1 / waves_c64=1,waves_q4=1
//
// Group 0 (indices  0- 9): No swizzle, direct DRAM epilogue
// Group 1 (indices 10-19): No swizzle, LDS-staged epilogue
// Group 2 (indices 20-29): XOR swizzle, direct DRAM epilogue
// Group 3 (indices 30-39): XOR swizzle, LDS-staged epilogue
constexpr Config configs[] = {
    // ---- Group 0: No swizzle, direct DRAM epilogue (default) ----
    // Dgrad (indices 0-4)
    {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad},
    {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad},
    // Fprop (indices 5-9)
    {.waves_c64 = 2, .waves_q4 = 8},
    {.waves_c64 = 2, .waves_q4 = 4},
    {.waves_c64 = 2, .waves_q4 = 2},
    {.waves_c64 = 2, .waves_q4 = 1},
    {.waves_c64 = 1, .waves_q4 = 1},
    // ---- Group 1: No swizzle, LDS-staged epilogue ----
    // Dgrad (indices 10-14)
    {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // Fprop (indices 15-19)
    {.waves_c64 = 2, .waves_q4 = 8,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 4,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 2,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 1,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 1, .waves_q4 = 1,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // ---- Group 2: XOR swizzle, direct DRAM epilogue ----
    // Dgrad (indices 20-24)
    {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    // Fprop (indices 25-29)
    {.waves_c64 = 2, .waves_q4 = 8,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 2, .waves_q4 = 4,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 2, .waves_q4 = 2,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 2, .waves_q4 = 1,
     .swizzle_type = SwizzleType::XOR},
    {.waves_c64 = 1, .waves_q4 = 1,
     .swizzle_type = SwizzleType::XOR},
    // ---- Group 3: XOR swizzle, LDS-staged epilogue ----
    // Dgrad (indices 30-34)
    {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // Fprop (indices 35-39)
    {.waves_c64 = 2, .waves_q4 = 8,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 4,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 2,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 2, .waves_q4 = 1,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_c64 = 1, .waves_q4 = 1,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     // Cyclic-shift swizzle
     {.waves_c64 = 2, .waves_q4 = 8,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     {.waves_c64 = 2, .waves_q4 = 8,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
     {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
    {
        return false;
    }
    if((par.groups % cfg.block_groups()) != 0)
    {
        return false;
    }
    const int out_q = (par.direction == Direction::Dgrad) ? par.w : par.q;
    if(out_q < cfg.block_q() && cfg.waves_q4 > 1)
    {
        return false;
    }
    // XOR swizzle constraint: every block_q offset must be a multiple of
    // BLOCK_C8 (= block_c/8). The XOR transform operates in global coords on
    // (x_global, c8), mapping c8 -> c8 ^ (x_global % BLOCK_C8). The LDS read
    // applies XOR in local coords: c8 -> c8 ^ (x_local % BLOCK_C8). For
    // consistency we need (block_q + x_local) % BLOCK_C8 == x_local % BLOCK_C8,
    // i.e. block_q % BLOCK_C8 == 0 for every tile. This is guaranteed when:
    //   (a) cfg.block_q() is a multiple of BLOCK_C8 (all offsets aligned), OR
    //   (b) only a single spatial tile is needed (out_q <= block_q, so offset=0).
    if(cfg.swizzle_type == SwizzleType::XOR)
    {
        const int block_c8 = cfg.block_c() / 8;
        if(cfg.block_q() % block_c8 != 0 && out_q > cfg.block_q())
        {
            return false;
        }
    }
    return true;
}

inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    const auto& cfg = configs[config_idx];

    // Compute the grid size.
    // For Dgrad the output is the input gradient (width = par.w, not par.q).
    const int out_q    = (cfg.direction == Direction::Dgrad) ? par.w : par.q;
    auto blocks_w      = ck_tile::integer_divide_ceil(out_q, cfg.block_q());
    auto blocks_w_n    = blocks_w * cfg.n_fold;
    auto blocks_c      = ck_tile::integer_divide_ceil(par.c_tot, cfg.block_c());
    auto blocks_n_fold = ck_tile::integer_divide_ceil(par.n, cfg.n_fold);

    LaunchParams launch;
    launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
    launch.block_size = dim3(cfg.block_size(), 1, 1);
    return launch;
}

// Tile constants derived from the kernel configuration.
template <Config cfg>
struct TileConstants
{
    static constexpr int GROUP_SIZE   = cfg.channels_per_group; // 4
    static constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4;         // 1

    // Number of input columns loaded by each workgroup (output columns plus halo).
    static constexpr int BLOCK_W = cfg.block_q() + (cfg.kw - 1);

    // uint4 vectors per channel fiber (8 fp16 per uint4).
    static constexpr int BLOCK_C8 = cfg.block_c() / 8;

    // Total channels per block in fp16 elements.
    static constexpr int BLOCK_C = BLOCK_C8 * 8;

    // fp16x4 groups per channel fiber (4 fp16 per group = one MFMA result vector).
    static constexpr int BLOCK_C4 = BLOCK_C / 4;

    // Number of uint4 vectors to store per output row (LDS epilogue path).
    static constexpr int STORE_VECS = cfg.block_q() * BLOCK_C8;

    // LDS double buffering for input loads.
    static constexpr int NUM_INPUT_LDS_BUFFERS    = 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C4 = INPUT_LDS_BUFFER_SIZE_C8 * 2;

    // Tile-level async load constants.
    static constexpr int NUM_WAVES    = cfg.num_waves();
    static constexpr int LANES_PER_ROW = WAVE_SIZE / BLOCK_C8;
    static constexpr int TOTAL_SPATIAL = cfg.block_size() / BLOCK_C8;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C8  = BLOCK_C8 * TOTAL_SPATIAL;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C4  = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_FP16 = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 8;

    // Aliases for cfg fields used as template arguments inside nested structs.
    // Direct member access (cfg.waves_q4) in template argument positions can
    // confuse the parser in nested struct context; aliases avoid this.
    static constexpr int WAVES_Q4  = cfg.waves_q4;
    static constexpr int WAVES_C64 = cfg.waves_c64;
    static constexpr int BLOCK_Q   = cfg.block_q();
    static constexpr int KH_KW     = cfg.kh * cfg.kw;
    static constexpr int KW        = cfg.kw;
    static constexpr SwizzleType SWIZZLE_TYPE = cfg.swizzle_type;

    // -----------------------------------------------------------------------
    // Mfma — shared tile distribution for MFMA operands and results.
    //
    // Maps (P0=warp_id, P1=lane_id) to a 3D tile coordinate
    //   (q_local, c4_local, c_sub) where:
    //   q_local  = warp_q * WARP_Q + lane_col  ∈ [0, block_q)
    //   c4_local = warp_c64 * 16 + lane_batch  ∈ [0, BLOCK_C4)
    //   c_sub    ∈ [0, 4) — vectorization (Y dimension)
    //
    // This is the thread layout produced and consumed by mfma_f32_4x4x4f16:
    //   lane_col   = (lane % 4)       → Q column within warp (4 output cols)
    //   lane_batch = (lane / 4) % 16  → C4 group within warp (16 groups of 4)
    //
    // 3D tile: [block_q, BLOCK_C4, 4]
    //   X0 = block_q [waves_q4, 4]:
    //     factor 0 (waves_q4): warp Q group → P0
    //     factor 1 (4): lane column → P1
    //   X1 = BLOCK_C4 [waves_c64, 16]:
    //     factor 0 (waves_c64): warp C group → P0
    //     factor 1 (16): lane batch → P1
    //   X2 = 4 (vectorization, Y dimension)
    //
    // R = [] (no replication)
    // -----------------------------------------------------------------------
    struct Mfma
    {
        static constexpr auto MakeDistribution()
        {
            // RH-major indices: 0=R(empty), 1=X0(Q), 2=X1(C4), 3=X2(c_sub)
            //
            // P0 (warp_id = num_waves, merge {waves_q4, waves_c64}):
            //   factor 0 (waves_q4)  → X0 factor 0 → major=1, minor=0
            //   factor 1 (waves_c64) → X1 factor 0 → major=2, minor=0
            //
            // P1 (lane_id = 64, merge {16, 4}):
            //   factor 0 (16 = batch) → X1 factor 1 → major=2, minor=1
            //   factor 1 (4 = col)    → X0 factor 1 → major=1, minor=1
            //
            // Y0 (length 4) → X2 factor 0 → major=3, minor=0 (vectorization)
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<WAVES_Q4, 4>,
                                   ck_tile::sequence<WAVES_C64, 16>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<2, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 1>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>>{});
        }
    };

    // -----------------------------------------------------------------------
    // Input — descriptors and distributions for the input activation tensor.
    //
    // Memory stages:
    //   DRAM:  4D [hi, wi_padded, BLOCK_C8, 8] in fp16 — async loaded to LDS.
    //   LDS:   4D [1, TOTAL_SPATIAL, BLOCK_C8, 8] in fp16 — staging buffer.
    //   Regs:  3D [BLOCK_W, BLOCK_C4, 4] in fp16 — MFMA A operand (read via Mfma distribution).
    // -----------------------------------------------------------------------
    struct Input
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Input;

        static CK_TILE_DEVICE auto MakeDramDescriptor(int hi, int wi, int C_total, int px)
        {
            return Shared::MakeDramDescriptor(hi, wi, C_total, px);
        }

        static constexpr auto MakeDramDistribution() { return Shared::MakeDramDistribution(); }

        static constexpr auto MakeLdsStoreDescriptor() { return Shared::MakeLdsStoreDescriptor(); }

        static constexpr auto MakeLdsReadDescriptor() { return Shared::MakeLdsReadDescriptor(); }
    };

    // -----------------------------------------------------------------------
    // Weight — descriptors and distributions for the filter weight tensor.
    //
    // Memory stages:
    //   DRAM:  2D [WEIGHT_LDS_SIZE_UINT4, 8] in fp16 — async loaded to LDS
    //          in NUM_WEIGHT_PASSES passes (one block_size chunk per pass).
    //   LDS:   2D [WEIGHT_LDS_PADDED_UINT4, 8] in fp16 — staging buffer.
    //   Regs:  1D array [kh*kw] of fp16x4 — MFMA B operand.
    //          Fprop reads via MakeLdsReadDistribution.
    //          Dgrad reads via TransposeLDSLayout (ds_read_b64_tr_b16).
    // -----------------------------------------------------------------------
    struct Weight
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Weight;

        // Weight LDS sizing (uniform formula: block_c * kh * kw * GROUP_SIZE / 4 uint2).
        static constexpr int WEIGHT_LDS_SIZE_UINT2   =
            cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE * GROUP_SIZE_4;
        static constexpr int WEIGHT_LDS_SIZE_UINT4   = WEIGHT_LDS_SIZE_UINT2 / 2;
        static constexpr int NUM_WEIGHT_PASSES =
            (WEIGHT_LDS_SIZE_UINT4 + cfg.block_size() - 1) / cfg.block_size();
        static constexpr int WEIGHT_LDS_PADDED_UINT4 = NUM_WEIGHT_PASSES * cfg.block_size();
        static constexpr int WEIGHT_LDS_READ_K = cfg.block_c();

        static constexpr auto MakeDramDescriptor() { return Shared::MakeDramDescriptor(); }
        static constexpr auto MakeDramDistribution() { return Shared::MakeDramDistribution(); }
        static constexpr auto MakeLdsStoreDescriptor() { return Shared::MakeLdsStoreDescriptor(); }
        static constexpr auto MakeLdsReadDescriptor() { return Shared::MakeLdsReadDescriptor(); }

        // Tile distribution for weight LDS reads (Fprop only).
        // R = [waves_q4]: all Q-waves read the same weights (replication).
        static constexpr auto MakeLdsReadDistribution()
        {
            // RH-major indices: 0=R, 1=X0(K-channel), 2=X1(filter pos), 3=X2(sub-channel)
            //
            // P0 (warp_id = num_waves, merge {waves_q4, waves_c64}):
            //   factor 0 (waves_q4) → R dim → major=0, minor=0
            //   factor 1 (waves_c64) → X0 factor 0 → major=1, minor=0
            //
            // P1 (lane_id = 64, merge {16, 4}):
            //   factor 0 (16 = batch) → X0 factor 1 → major=1, minor=1
            //   factor 1 (4 = k)      → X0 factor 2 → major=1, minor=2
            //
            // Y0 (length kh*kw) → X1 factor 0 → major=2, minor=0
            // Y1 (length 4)     → X2 factor 0 → major=3, minor=0
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<WAVES_Q4>,
                    ck_tile::tuple<ck_tile::sequence<WAVES_C64, 16, 4>,
                                   ck_tile::sequence<KH_KW>,
                                   ck_tile::sequence<GROUP_SIZE>>,
                    ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<1, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 2>>,
                    ck_tile::sequence<2, 3>,
                    ck_tile::sequence<0, 0>>{});
        }
    };

    // -----------------------------------------------------------------------
    // Output — descriptors and distributions for the output activation tensor.
    //
    // Memory stages:
    //   Regs:  Distributed fp32x4 accumulators in the MFMA layout
    //          (mapped to/from fp16x4 via Mfma::MakeDistribution).
    //   LDS:   3D [block_q, BLOCK_C4, 4] in fp16 — staging buffer for the
    //          LDS epilogue path (RegistersToLdsToGlobalMemory).
    //   DRAM:  4D [ho, wo_padded, BLOCK_C4, 4] in fp16 — final output.
    //          MakeDramDistribution is also used to read back from LDS before
    //          the coalesced DRAM store (both use the same thread mapping).
    // -----------------------------------------------------------------------
    struct Output
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Output;

        static constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * BLOCK_Q;

        static constexpr auto MakeLdsWriteDescriptor() { return Shared::MakeLdsWriteDescriptor(); }
        static constexpr auto MakeLdsReadDescriptor() { return Shared::MakeLdsReadDescriptor(); }

        static CK_TILE_DEVICE auto MakeDramDescriptor(int ho, int wo, int C)
        {
            return Shared::MakeDramDescriptor(ho, wo, C);
        }

        // Tile distribution for DRAM output writes (variant-specific wave decomposition).
        // 4D tile: [1, block_q, BLOCK_C4, 4]
        static constexpr auto MakeDramDistribution()
        {
            // RH-major indices: 0=R(empty), 1=X0(row), 2=X1(Q), 3=X2(C_group), 4=X3(C_sub)
            //
            // P0 (warp_id), merge {waves_q4, waves_c64}:
            //   factor 0 → X1 factor 0 (waves_q4)  → major=2, minor=0
            //   factor 1 → X2 factor 0 (waves_c64) → major=3, minor=0
            //
            // P1 (lane_id=64), merge {16, 4}:
            //   factor 0 (16=batch) → X2 factor 1 → major=3, minor=1
            //   factor 1 (4=col)    → X1 factor 1 → major=2, minor=1
            //
            // Y0 (length 1) → X0 → major=1, minor=0 (trivial row)
            // Y1 (length 4) → X3 → major=4, minor=0 (vectorization)
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<1>,
                                   ck_tile::sequence<WAVES_Q4, 4>,
                                   ck_tile::sequence<WAVES_C64, 16>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<2, 3>, ck_tile::sequence<3, 2>>,
                    ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 1>>,
                    ck_tile::sequence<1, 4>,
                    ck_tile::sequence<0, 0>>{});
        }
    };
};

// Workgroup-level coordinates derived from blockIdx.
template <Config cfg>
struct BlockCoords
{
    int block_n;      // Batch index for this workgroup (unfolded from blockIdx).
    int block_q;      // Starting output column (W dimension) for this workgroup.
    int block_group;  // Starting convolution group index for this workgroup.
    int block_k;      // Starting output channel index (= block_group * channels_per_group).
    int block_c8;     // block_k expressed in uint4 vector units (block_k / 8).
    int C;            // Total number of input channels (groups * channels_per_group).
    int C8;           // Total number of input channels in uint4 vector units (C / 8).
    int K;            // Total number of output channels (== C for this kernel since K_tot == C_tot).

    __device__ BlockCoords(int groups)
        : C(groups * cfg.channels_per_group), C8(C / 8), K(C)
    {
        const int block_q_n_idx = blockIdx.x;
        block_n     = static_cast<int>(blockIdx.z) * cfg.n_fold + block_q_n_idx % cfg.n_fold;
        block_q     = (block_q_n_idx / cfg.n_fold) * cfg.block_q();
        block_group = static_cast<int>(blockIdx.y) * cfg.block_groups();
        block_k     = block_group * cfg.channels_per_group;
        block_c8    = block_k / 8; // TODO: rename block_c8 to block_k8
    }
};

// Handles input loads from global memory into LDS and then into registers.
template <Config cfg>
using InputLoader = direct_conv::InputLoader<TileConstants<cfg>, cfg>;

// Handles weight reads from LDS into registers.
template <Config cfg>
struct WeightLoader
{
    using TC = TileConstants<cfg>;

    template <typename BlockCoords_>
    __device__ static void load_to_lds(const BlockCoords_& bc,
                                       uint4* weight_lds,
                                       const _Float16* __restrict__ wei)
    {
        direct_conv::weight_load_to_lds<TC, cfg>(bc, weight_lds, wei);
    }

    // Read weights from LDS into registers after sync.
    __device__ static void read_from_lds(
        fp16x4_t (&weights_reg)[cfg.kh * cfg.kw],
        uint4* weight_lds)
    {
        if constexpr(cfg.direction == Direction::Dgrad)
        {
            // Dgrad uses transposed LDS reads (ds_read_b64_tr_b16) which
            // require the TransposeLDSLayout address mapping and cannot be
            // expressed via load_tile_transpose (its quad pattern requires
            // X1 >= 16, but GROUP_SIZE = 4). Compute thread coordinates
            // directly from threadIdx.x.
            const int lane     = static_cast<int>(threadIdx.x) % WAVE_SIZE;
            const int wave_c64 = (static_cast<int>(threadIdx.x) / WAVE_SIZE) % cfg.waves_c64;

            auto output_lds_fp16 = ck_tile::buffer_view<
                ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>{
                reinterpret_cast<_Float16*>(weight_lds),
                static_cast<ck_tile::index_t>(TC::Weight::WEIGHT_LDS_PADDED_UINT4 *
                                              (sizeof(uint4) / sizeof(_Float16)))};

            using TransposeLayout = TransposeLDSLayout<4, 4, 16>;
            const int tr_batch    = TransposeLayout::batch(lane);
            const int tr_row      = TransposeLayout::row(lane);
            int filter_local      = wave_c64 * 64 + tr_batch * TC::GROUP_SIZE + tr_row;

            const ck_tile::index_t weight_base =
                filter_local * cfg.kh * cfg.kw * TC::GROUP_SIZE;

            for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
            {
                weights_reg[khw] = output_lds_fp16.template transpose_get<ck_tile::fp16x4_t>(
                    weight_base + khw * TC::GROUP_SIZE, 0, true);
            }
        }
        else
        {
            // Fprop: use tile distribution to read weights from LDS.
            constexpr auto weight_lds_read_desc = TC::Weight::MakeLdsReadDescriptor();
            auto weight_lds_view =
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    reinterpret_cast<_Float16*>(weight_lds), weight_lds_read_desc);

            constexpr auto weight_lds_read_dist = TC::Weight::MakeLdsReadDistribution();
            auto weight_lds_read_window         = ck_tile::make_tile_window(
                weight_lds_view,
                ck_tile::make_tuple(ck_tile::number<TC::Weight::WEIGHT_LDS_READ_K>{},
                                    ck_tile::number<TC::KH_KW>{},
                                    ck_tile::number<TC::GROUP_SIZE>{}),
                {0, 0, 0},
                weight_lds_read_dist);

            const auto weight_tile = ck_tile::load_tile(weight_lds_read_window);
            const auto& buf        = weight_tile.get_thread_buffer();

            static_for<TC::KH_KW>(
                [&]<int khw>()
                {
                    __builtin_memcpy(&weights_reg[khw],
                                     &buf.get(khw * TC::GROUP_SIZE),
                                     sizeof(fp16x4_t));
                });
        }
    }
};

// Handles writing MFMA accumulator results to global memory.
template <Config cfg>
using OutputWriter = direct_conv::OutputWriter<TileConstants<cfg>>;

// Handles output staging through LDS and writing to global memory.
template <Config cfg>
using OutputWriterLds = direct_conv::OutputWriterLds<TileConstants<cfg>>;

template <Config cfg>
__device__ void conv2d_grouped_4c_fp16_cdna4_nhwc_impl(const _Float16* __restrict__ in,
                                                       const _Float16* __restrict__ wei,
                                                       double alpha,
                                                       double beta,
                                                       _Float16* __restrict__ out,
                                                       int N,
                                                       int groups,
                                                       int c_per_group,
                                                       int k_per_group,
                                                       int hi,
                                                       int wi,
                                                       int ho,
                                                       int wo,
                                                       int fy,
                                                       int fx,
                                                       int sy,
                                                       int sx,
                                                       int dy,
                                                       int dx,
                                                       int py,
                                                       int px)
{
    constexpr bool use_lds_epilogue = (cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory);
    using TC = TileConstants<cfg>;
    using OutputWriterType = std::conditional_t<use_lds_epilogue, OutputWriterLds<cfg>, OutputWriter<cfg>>;

    direct_conv::grouped_conv_compute_loop<
        TC, cfg, Mfma4x4x4,
        BlockCoords<cfg>, InputLoader<cfg>, WeightLoader<cfg>, OutputWriterType>(
        in, wei, out, N, groups, hi, wi, ho, wo, py, px);
}

template <Config cfg>
__global__ void conv2d_grouped_4c_fp16_nhwc_cdna4(const _Float16* __restrict__ in,
                                                  const _Float16* __restrict__ wei,
                                                  double alpha,
                                                  double beta,
                                                  _Float16* __restrict__ out,
                                                  int N,
                                                  int groups,
                                                  int c_per_group,
                                                  int k_per_group,
                                                  int hi,
                                                  int wi,
                                                  int ho,
                                                  int wo,
                                                  int fy,
                                                  int fx,
                                                  int sy,
                                                  int sx,
                                                  int dy,
                                                  int dx,
                                                  int py,
                                                  int px)
{
    conv2d_grouped_4c_fp16_cdna4_nhwc_impl<cfg>(in,
                                                wei,
                                                alpha,
                                                beta,
                                                out,
                                                N,
                                                groups,
                                                c_per_group,
                                                k_per_group,
                                                hi,
                                                wi,
                                                ho,
                                                wo,
                                                fy,
                                                fx,
                                                sy,
                                                sx,
                                                dy,
                                                dx,
                                                py,
                                                px);
}

template <size_t... Is>
void launch_dispatch(int config_idx,
                     std::index_sequence<Is...>,
                     const LaunchParams& lp,
                     const Conv2dParams& par,
                     const void* in,
                     const void* wei,
                     void* out,
                     hipStream_t stream)
{
    auto kernel_launch = [&]<size_t I>()
    {
        auto view = SizeView<configs[I].direction>(par);
        conv2d_grouped_4c_fp16_nhwc_cdna4<configs[I]>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const _Float16*>(in),
                static_cast<const _Float16*>(wei),
                1.0,
                0.0,
                static_cast<_Float16*>(out),
                par.n,
                par.groups,
                par.channels_per_group(),
                par.filters_per_group(),
                view.h(),
                view.w(),
                view.p(),
                view.q(),
                par.kh,
                par.kw,
                par.stride_h,
                par.stride_w,
                par.dilation_h,
                par.dilation_w,
                view.pad_h(),
                view.pad_w());
    };
    (void)((config_idx == static_cast<int>(Is) ? (kernel_launch.template operator()<Is>(), true)
                                               : false) ||
           ...);
}

inline void launch(int config_idx,
                   const LaunchParams& lp,
                   const Conv2dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   void* /*workspace*/,
                   hipStream_t stream)
{
    launch_dispatch(
        config_idx, std::make_index_sequence<NUM_CONFIGS>{}, lp, par, in, wei, out, stream);
}

constexpr KernelVariant make_variant()
{
    return {
        .is_applicable =
            [](const Conv2dParams& par)
        {
            if(par.in_type != DataType::fp16)
                return false;
            if(par.wei_type != DataType::fp16)
                return false;
            if(par.out_type != DataType::fp16)
                return false;
            if(par.order != TensorOrder::NHWC)
                return false;
            if(par.direction != Direction::Fprop &&
               par.direction != Direction::Dgrad)
                return false;
            if(par.kh != 3 || par.kw != 3)
                return false;
            if(par.k_tot != par.c_tot)
                return false;
            if(par.channels_per_group() != 4)
                return false;
            if(par.c_tot % 4 != 0)
                return false;
            if(par.stride_h != 1 || par.stride_w != 1)
                return false;
            if(par.dilation_h != 1 || par.dilation_w != 1)
                return false;
            if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
                return false;
            return true;
        },
        .config_is_compatible = [](const Conv2dParams& par, int idx)
        { return is_valid_config(par, configs[idx]); },
        .get_launch_params  = &get_launch_params,
        .launch             = &launch,
        .get_workspace_size = [](int, const Conv2dParams&) -> size_t { return 0; },
        .num_configs        = NUM_CONFIGS,
    };
}

} // namespace ck_tile::direct_conv::grouped_4c_tile::v3
