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

namespace ck_tile::direct_conv::grouped_16c_tile::v2
{

using namespace ck_tile::direct_conv;

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Thread block output is 16 columns wide (fixed by mfma_f32_16x16x16f16 M=16).
constexpr int BLOCK_Q = 16;

// Kernel configuration parameters.
struct Config
{
    // Number of wave per workgroup.
    // Each wave computes BLOCK_Q=16 columns and waves_per_wg * 16 input channels.
    int waves_per_wg;

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
    int channels_per_group = 16;

    // Uniform accessor for shared code (alias for channels_per_group).
    constexpr int group_size() const { return channels_per_group; }

    Direction direction = Direction::Fprop;

    SwizzleType swizzle_type = SwizzleType::None;

    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;

    // Total number of waves per workgroup.
    constexpr int num_waves() const { return waves_per_wg; }

    // Tile size in the channel dimension: number of input channels processed by one workgroup.
    constexpr int block_c() const { return channels_per_group * waves_per_wg; }

    // Tile size in the output column dimension (fixed by MFMA M=16).
    constexpr int block_q() const { return BLOCK_Q; }

    // Number of conv groups processed by one workgroup.
    constexpr int block_groups() const { return waves_per_wg; }

    // Number of threads per workgroup (thread block).
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    std::string GetName() const
    {
        std::string swz = "noswizzle";
        if (swizzle_type == SwizzleType::XOR)
            swz = "xorswizzle";
        else if (swizzle_type == SwizzleType::CyclicShift)
            swz = "cyclicshift";
        
        std::string base = "tile_v2_grouped_16c_" + swz + "_waves_per_wg_" +
                           std::to_string(waves_per_wg);
        if(epilogue == EpilogueType::RegistersToGlobalMemory)
            return base + "_skip_lds_epilogue";
        else
            return base + "_lds_epilogue";
    }
};

// All instantiated configurations.
//
// Layout: 4 variant groups × 18 configs each = 72 configs total.
// Each group has 9 Dgrad + 9 Fprop configs:
//   waves_per_wg = 16,8,7,6,5,4,3,2,1
//
// Group 0 (indices  0-17): No swizzle, direct DRAM epilogue
// Group 1 (indices 18-35): No swizzle, LDS-staged epilogue
// Group 2 (indices 36-53): XOR swizzle, direct DRAM epilogue
// Group 3 (indices 54-71): XOR swizzle, LDS-staged epilogue
constexpr Config configs[] = {
    // ---- Group 0: No swizzle, direct DRAM epilogue (default) ----
    // Dgrad (indices 0-8)
    {.waves_per_wg = 16, .direction = Direction::Dgrad},
    {.waves_per_wg = 8, .direction = Direction::Dgrad},
    {.waves_per_wg = 7, .direction = Direction::Dgrad},
    {.waves_per_wg = 6, .direction = Direction::Dgrad},
    {.waves_per_wg = 5, .direction = Direction::Dgrad},
    {.waves_per_wg = 4, .direction = Direction::Dgrad},
    {.waves_per_wg = 3, .direction = Direction::Dgrad},
    {.waves_per_wg = 2, .direction = Direction::Dgrad},
    {.waves_per_wg = 1, .direction = Direction::Dgrad},
    // Fprop (indices 9-17)
    {.waves_per_wg = 16},
    {.waves_per_wg = 8},
    {.waves_per_wg = 7},
    {.waves_per_wg = 6},
    {.waves_per_wg = 5},
    {.waves_per_wg = 4},
    {.waves_per_wg = 3},
    {.waves_per_wg = 2},
    {.waves_per_wg = 1},
    // ---- Group 1: No swizzle, LDS-staged epilogue ----
    // Dgrad (indices 18-26)
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 7, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 5, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 3, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 1, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // Fprop (indices 27-35)
    {.waves_per_wg = 16,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 7,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 5,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 3,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 1,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // ---- Group 2: XOR swizzle, direct DRAM epilogue ----
    // Dgrad (indices 36-44)
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 7, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 6, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 5, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 3, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 2, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 1, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    // Fprop (indices 45-53)
    {.waves_per_wg = 16,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 7,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 6,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 5,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 4,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 3,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 2,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 1,
     .swizzle_type = SwizzleType::XOR},
    // ---- Group 3: XOR swizzle, LDS-staged epilogue ----
    // Dgrad (indices 54-62)
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 7, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 5, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 3, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 1, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // Fprop (indices 63-71)
    {.waves_per_wg = 16,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 7,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 5,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 3,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 1,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     // Cyclic shift instances
     {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
     {.waves_per_wg = 8, .direction=Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     {.waves_per_wg = 8, .direction=Direction::Dgrad,
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
    if((par.groups % cfg.waves_per_wg) != 0)
    {
        return false;
    }
    // XOR swizzle constraint: BLOCK_Q must be a multiple of BLOCK_C8 for
    // multi-tile spatial decomposition. BLOCK_C8 = waves_per_wg * 2.
    // BLOCK_Q = 16 is divisible by BLOCK_C8 only when waves_per_wg divides 8
    // (i.e., waves_per_wg ∈ {1,2,4,8}). For other values, XOR is only valid
    // when the output fits in a single spatial tile.
    if(cfg.swizzle_type == SwizzleType::XOR)
    {
        const int block_c8 = cfg.block_c() / 8;
        const int out_q = (par.direction == Direction::Dgrad) ? par.w : par.q;
        if(BLOCK_Q % block_c8 != 0 && out_q > BLOCK_Q)
        {
            return false;
        }
    }
    return true;
}

inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    const auto& cfg = configs[config_idx];

    const int out_q    = (cfg.direction == Direction::Dgrad) ? par.w : par.q;
    auto blocks_w      = ck_tile::integer_divide_ceil(out_q, BLOCK_Q);
    auto blocks_w_n    = blocks_w * cfg.n_fold;
    auto blocks_c      = ck_tile::integer_divide_ceil(par.c_tot, cfg.block_c());
    auto blocks_n_fold = ck_tile::integer_divide_ceil(par.n, cfg.n_fold);

    LaunchParams launch;
    launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
    launch.block_size = dim3(cfg.block_size(), 1, 1);
    return launch;
}

// ===================================================================
// Tile constants derived from the kernel configuration.
// ===================================================================
template <Config cfg>
struct TileConstants
{
    static constexpr int GROUP_SIZE   = cfg.group_size();   // 16
    static constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4;   // 4
    static constexpr int GROUP_SIZE_8 = GROUP_SIZE / 8;   // 2

    static constexpr int BLOCK_Q = cfg.block_q();

    // Number of input columns loaded by each workgroup (output columns plus halo).
    static constexpr int BLOCK_W = BLOCK_Q + (cfg.kw - 1);

    // uint4 vectors per channel fiber (8 fp16 per uint4).
    static constexpr int BLOCK_C8 = cfg.block_c() / 8;

    // Total channels per block in fp16 elements.
    static constexpr int BLOCK_C = BLOCK_C8 * 8;

    // fp16x4 groups per channel fiber.
    static constexpr int BLOCK_C4 = BLOCK_C / 4;

    // Number of uint4 vectors to store per output row (LDS epilogue path).
    static constexpr int STORE_VECS = BLOCK_Q * BLOCK_C8;

    // LDS double buffering for input loads.
    static constexpr int NUM_INPUT_LDS_BUFFERS    = 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C8  = BLOCK_C8 * BLOCK_W;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C4  = INPUT_LDS_BUFFER_SIZE_C8 * 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_FP16 = INPUT_LDS_BUFFER_SIZE_C8 * 8;

    // Tile-level async load constants.
    static constexpr int NUM_WAVES     = cfg.waves_per_wg;
    static constexpr int LANES_PER_ROW = WAVE_SIZE / BLOCK_C8;
    static constexpr int TOTAL_SPATIAL  = cfg.block_size() / BLOCK_C8;

    static constexpr int KH_KW = cfg.kh * cfg.kw;
    static constexpr int KW    = cfg.kw;
    static constexpr SwizzleType SWIZZLE_TYPE = cfg.swizzle_type;

    // -----------------------------------------------------------------------
    // Mfma — shared tile distribution for MFMA operands and results.
    //
    // mfma_f32_16x16x16f16 lane mapping:
    //   lane_q   = lane % 16 → Q column (16 output cols)
    //   lane_c4  = lane / 16 → C4 group (4 groups of fp32x4)
    //
    // 3D tile: [BLOCK_Q=16, BLOCK_C4, 4]
    //   X0 = 16 [16]: lane_q → P1
    //   X1 = BLOCK_C4 [waves_per_wg, 4]:
    //     factor 0 (waves_per_wg): warp_id → P0
    //     factor 1 (4): lane_c4 → P1
    //   X2 = 4: vectorization → Y0
    // -----------------------------------------------------------------------
    struct Mfma
    {
        static constexpr auto MakeAccTileDistribution()
        {
            // P0 (warp_id = NUM_WAVES):
            //   → X1 factor 0 → major=2, minor=0
            // P1 (lane_id = 64, merge {4, 16}):
            //   factor 0 (4 = lane/16) → X1 factor 1 → major=2, minor=1
            //   factor 1 (16 = lane%16) → X0 factor 0 → major=1, minor=0
            // Y0 (length 4) → X2 factor 0 → major=3, minor=0
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<16>,
                                   ck_tile::sequence<NUM_WAVES, 4>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>>{});
        }
    };

    // -----------------------------------------------------------------------
    // Input — descriptors and distributions for the input activation tensor.
    //
    // Memory stages:
    //   DRAM:  4D [hi, wi_padded, BLOCK_C8, 8] in fp16 — async loaded to LDS.
    //   LDS:   4D [1, BLOCK_W, BLOCK_C8, 8] in fp16 — staging buffer.
    //   Regs:  3D [BLOCK_W, BLOCK_C4, 4] in fp16 — MFMA A operand.
    // -----------------------------------------------------------------------
    struct Input
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Input;

        static CK_TILE_DEVICE auto MakeDramReadDescriptor(int hi, int wi, int C_total, int px)
        {
            return Shared::MakeDramReadDescriptor(hi, wi, C_total, px);
        }

        static constexpr auto MakeDramReadTileDistribution() { return Shared::MakeDramReadTileDistribution(); }

        static constexpr auto MakeLdsWriteDescriptor() { return Shared::MakeLdsWriteDescriptor(); }

        static constexpr auto MakeLdsReadDescriptor() { return Shared::MakeLdsReadDescriptor(); }
    };

    // -----------------------------------------------------------------------
    // Weight — descriptors and distributions for the filter weight tensor.
    //
    // Memory stages:
    //   DRAM:  2D [WEIGHT_LDS_SIZE_UINT4, 8] in fp16 — async loaded to LDS.
    //   LDS:   2D [WEIGHT_LDS_PADDED_UINT4, 8] in fp16 — staging buffer.
    //   Regs:  1D array [kh*kw] of fp16x4.
    // -----------------------------------------------------------------------
    struct Weight
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Weight;

        // Weight LDS sizing (uniform formula: block_c * kh * kw * GROUP_SIZE / 4 uint2).
        static constexpr int WEIGHT_LDS_SIZE_UINT2 =
            cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE * GROUP_SIZE_4;
        static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_SIZE_UINT2 / 2;
        static constexpr int NUM_WEIGHT_PASSES =
            (WEIGHT_LDS_SIZE_UINT4 + cfg.block_size() - 1) / cfg.block_size();
        static constexpr int WEIGHT_LDS_PADDED_UINT4 = NUM_WEIGHT_PASSES * cfg.block_size();
        static constexpr int WEIGHT_LDS_READ_K = cfg.block_c();

        static constexpr auto MakeDramReadDescriptor() { return Shared::MakeDramReadDescriptor(); }
        static constexpr auto MakeDramReadTileDistribution() { return Shared::MakeDramReadTileDistribution(); }
        static constexpr auto MakeLdsWriteDescriptor() { return Shared::MakeLdsWriteDescriptor(); }
        static constexpr auto MakeLdsReadDescriptor() { return Shared::MakeLdsReadDescriptor(); }

        // Tile distribution for weight LDS reads (Fprop).
        //
        // mfma_f32_16x16x16f16 B operand:
        //   lane_k   = lane % 16 → K-column (outer-product dim of B)
        //   lane_c4  = lane / 16 → C-reduction group (4 groups, each 4 fp16)
        //
        // 3D tile: [block_c, kh*kw, GROUP_SIZE]
        //   X0 = block_c [waves_per_wg, 16]:
        //     factor 0 (waves_per_wg): wave index → P0
        //     factor 1 (16): lane_k → P1
        //   X1 = kh*kw → Y0 (filter positions)
        //   X2 = GROUP_SIZE [4, 4]:
        //     factor 0 (4): lane_c4 → P1
        //     factor 1 (4): sub-channel → Y1
        //
        // P1 merge = {4, 16}:
        //   factor 0 = lane/16 (range 0-3) → X2 factor 0 (4 elements)
        //   factor 1 = lane%16 (range 0-15) → X0 factor 1 (16 elements)
        //
        // No R dimension — all Q positions come from within a single wave.
        static constexpr auto MakeLdsReadTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<NUM_WAVES, 16>,
                                   ck_tile::sequence<KH_KW>,
                                   ck_tile::sequence<4, 4>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<3, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<0, 1>>,
                    ck_tile::sequence<2, 3>,
                    ck_tile::sequence<0, 1>>{});
        }
    };

    // -----------------------------------------------------------------------
    // Output — descriptors and distributions for the output activation tensor.
    // -----------------------------------------------------------------------
    struct Output
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Output;

        static constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * BLOCK_Q;

        static constexpr auto MakeLdsWriteDescriptor() { return Shared::MakeLdsWriteDescriptor(); }

        static CK_TILE_DEVICE auto MakeDramWriteDescriptorNarrow(int ho, int wo, int C)
        {
            return Shared::MakeDramWriteDescriptorNarrow(ho, wo, C);
        }

        // Store distribution for wider LDS reads and DRAM stores.
        // Maps all block_size threads to [STORE_Q, BLOCK_C8, 8] positions.
        // Only STORE_VECS threads are active (Q < BLOCK_Q).
        //
        // Thread decomposition:
        //   P0 (warp_id, NUM_WAVES) → X0 factor 0 (Q warp group)
        //   P1 (lane_id = 64, merge {LANES_PER_ROW, BLOCK_C8}):
        //     factor 0 (LANES_PER_ROW) → X0 factor 1 (Q within warp)
        //     factor 1 (BLOCK_C8) → X1 (C8)
        //   Y0 (length 8) → X2 (vectorization)
        static constexpr auto MakeDramWriteTileDistributionWide()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<NUM_WAVES, LANES_PER_ROW>,
                                   ck_tile::sequence<BLOCK_C8>,
                                   ck_tile::sequence<8>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>,
                    ck_tile::number<STORE_VECS>>{});
        }

        static constexpr int STORE_Q = TOTAL_SPATIAL;

        static constexpr auto MakeLdsReadDescriptorWide()
        {
            return Shared::template MakeLdsReadDescriptorWide<STORE_Q>();
        }

        static CK_TILE_DEVICE auto MakeDramWriteDescriptorWide(int wo, int C)
        {
            return Shared::MakeDramWriteDescriptorWide(wo, C);
        }

        // Tile distribution for DRAM output writes (variant-specific wave decomposition).
        // 4D: [1, BLOCK_Q, BLOCK_C4, 4]
        //   X0 = 1 (row) → Y0
        //   X1 = 16 (Q) → P1
        //   X2 = BLOCK_C4 [waves_per_wg, 4] → P0, P1
        //   X3 = 4 → Y1
        static constexpr auto MakeDramWriteTileDistributionNarrow()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<1>,
                                   ck_tile::sequence<16>,
                                   ck_tile::sequence<NUM_WAVES, 4>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<3>, ck_tile::sequence<3, 2>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<1, 4>,
                    ck_tile::sequence<0, 0>>{});
        }
    };
};

// ===================================================================
// Workgroup-level coordinates derived from blockIdx.
// ===================================================================
template <Config cfg>
struct BlockCoords
{
    int block_n;
    int block_q;
    int block_group;
    int block_k;
    int block_c8;
    int C;
    int C8;
    int K;

    __device__ BlockCoords(int groups)
        : C(groups * cfg.group_size()), C8(C / 8), K(C)
    {
        const int block_q_n_idx = blockIdx.x;
        block_n     = static_cast<int>(blockIdx.z) * cfg.n_fold + block_q_n_idx % cfg.n_fold;
        block_q     = (block_q_n_idx / cfg.n_fold) * BLOCK_Q;
        block_group = static_cast<int>(blockIdx.y) * cfg.block_groups();
        block_k     = block_group * cfg.group_size();
        block_c8    = block_k / 8;
    }
};

// ===================================================================
// InputLoader — DRAM→LDS async load, double-buffered, MFMA reads.
// ===================================================================
template <Config cfg>
using InputLoader = direct_conv::InputLoader<TileConstants<cfg>, cfg>;

// ===================================================================
// WeightLoader — async weight loads to LDS, then register reads.
// ===================================================================
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

    __device__ static void read_from_lds(
        fp16x4_t (&weights_reg)[cfg.kh * cfg.kw],
        uint4* weight_lds)
    {
        if constexpr(cfg.direction == Direction::Dgrad)
        {
            // Dgrad: transposed LDS reads via ds_read_b64_tr_b16.
            const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
            const int wave = static_cast<int>(threadIdx.x) / WAVE_SIZE;

            auto weight_lds_fp16 = ck_tile::buffer_view<
                ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>{
                reinterpret_cast<_Float16*>(weight_lds),
                static_cast<ck_tile::index_t>(TC::Weight::WEIGHT_LDS_SIZE_UINT4 *
                                              (sizeof(uint4) / sizeof(_Float16)))};

            using TransposeLayout = TransposeLDSLayout<16, 16>;
            const int tr_row = TransposeLayout::row(lane);
            const int tr_col = TransposeLayout::col(lane);
            int filter_local = wave * TC::GROUP_SIZE + tr_row;

            const ck_tile::index_t weight_base =
                filter_local * cfg.kh * cfg.kw * TC::GROUP_SIZE;

            for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
            {
                weights_reg[khw] = weight_lds_fp16.template transpose_get<ck_tile::fp16x4_t>(
                    weight_base + khw * TC::GROUP_SIZE + tr_col, 0, true);
            }
        }
        else
        {
            // Fprop: use tile distribution to read weights from LDS.
            constexpr auto weight_lds_read_desc = TC::Weight::MakeLdsReadDescriptor();
            auto weight_lds_view =
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    reinterpret_cast<_Float16*>(weight_lds), weight_lds_read_desc);

            constexpr auto weight_lds_read_dist = TC::Weight::MakeLdsReadTileDistribution();
            auto weight_lds_read_window = ck_tile::make_tile_window(
                weight_lds_view,
                ck_tile::make_tuple(ck_tile::number<TC::Weight::WEIGHT_LDS_READ_K>{},
                                    ck_tile::number<TC::KH_KW>{},
                                    ck_tile::number<TC::GROUP_SIZE>{}),
                {0, 0, 0},
                weight_lds_read_dist);

            const auto weight_tile = ck_tile::load_tile(weight_lds_read_window);
            const auto& buf = weight_tile.get_thread_buffer();

            // Thread buffer has KH_KW * 4 elements (Y0=KH_KW, Y1=4).
            // X2 = [4, 4]: factor 0 (4) is mapped to P1 (lane/16),
            // factor 1 (4) is Y1 iteration. So per-filter-position stride = 4.
            static_for<TC::KH_KW>(
                [&]<int khw>()
                {
                    __builtin_memcpy(&weights_reg[khw],
                                     &buf.get(khw * 4),
                                     sizeof(fp16x4_t));
                });
        }
    }
};

// ===================================================================
// OutputWriter — direct DRAM writes (RegistersToGlobalMemory epilogue).
// ===================================================================
template <Config cfg>
using OutputWriter = direct_conv::OutputWriter<TileConstants<cfg>>;

// ===================================================================
// OutputWriterLds — LDS-staged writes (RegistersToLdsToGlobalMemory).
// ===================================================================
template <Config cfg>
using OutputWriterLds = direct_conv::OutputWriterLds<TileConstants<cfg>>;

// ===================================================================
// Main device function.
// ===================================================================
template <Config cfg>
__device__ void conv2d_grouped_16c_fp16_cdna4_nhwc_impl_v2(const _Float16* __restrict__ in,
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
        TC, cfg, Mfma16x16x16,
        BlockCoords<cfg>, InputLoader<cfg>, WeightLoader<cfg>, OutputWriterType>(
        in, wei, out, N, groups, hi, wi, ho, wo, py, px);
}

template <Config cfg>
__global__ void conv2d_grouped_16c_fp16_nhwc_cdna4_v2(const _Float16* __restrict__ in,
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
    conv2d_grouped_16c_fp16_cdna4_nhwc_impl_v2<cfg>(in, wei, alpha, beta, out,
                                                     N, groups, c_per_group, k_per_group,
                                                     hi, wi, ho, wo, fy, fx, sy, sx, dy, dx, py, px);
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
        conv2d_grouped_16c_fp16_nhwc_cdna4_v2<configs[I]>
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
            if(par.channels_per_group() != 16)
                return false;
            if(par.c_tot % 16 != 0)
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

} // namespace ck_tile::direct_conv::grouped_16c_tile::v2
