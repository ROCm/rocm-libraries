// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CK Tile v2 implementation of 8-channel grouped convolution using Toeplitz FIR structure.
//
// This kernel uses mfma_f32_16x16x32_f16 with the S-dimension embedded in the MFMA's
// K=32 dimension via a block-Toeplitz matrix, achieving 75% MFMA utilization.
// No explicit S-loop is needed — only the R (filter height) loop.
//
// v2 replaces manual low-level intrinsics from v1 with CK Tile abstractions:
//   - Input DRAM→LDS: shared InputLoader with CK Tile descriptors
//   - Weight DRAM→LDS: shared weight_load_to_lds
//   - Output writes: shared OutputWriter (direct DRAM) or OutputWriterLds (LDS-staged)
//   - Swizzle support: None / XOR / CyclicShift via descriptor transforms
//
// The Toeplitz compute loop is unchanged from v1 (R-loop only, fp16x8_t, mfma_f32_16x16x32_f16).

#pragma once

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_descriptors.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_weight_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp"
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

namespace ck_tile::direct_conv::grouped_8c_tile::v2
{

using namespace ck_tile::direct_conv;

// Resolve ambiguity with ck_tile:: types
using ck_tile::direct_conv::fp16x4_t;
using ck_tile::direct_conv::fp16x8_t;
using ck_tile::direct_conv::fp32x4_t;
using ck_tile::direct_conv::static_for;

// ============================================================================
// Toeplitz FIR transforms for 8-channel grouped convolution
// ============================================================================

// GT is the transpose of the Toeplitz filter matrix G.
//
//            t=0  t=1  t=2  t=3
//          [ g0   g1   g2   0  ]  q=0
//   GT =   [  0   g0   g1   g2 ]  q=1
//
// GT ~ 16 x 32 = [2q x 8k] x [4t x 8c]
struct GT
{
    static constexpr int rows       = 16;
    static constexpr int cols       = 32;
    static constexpr int group_size = 8;

    static constexpr int q(int row) { return row / group_size; }
    static constexpr int t(int col) { return col / group_size; }
    static constexpr int k(int row) { return row % group_size; }
    static constexpr int c(int col) { return col % group_size; }
    static constexpr int c4(int col4) { return col4 % (group_size / 4); }
    static constexpr int s(int row, int col4) { return GT::t(col4 * 4) - GT::q(row); }

    static constexpr int filter_is_zero(int row, int col4)
    {
        int ss = GT::s(row, col4);
        return ss < 0 || ss > 2;
    }
};

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 32 columns wide (16 tiles x 2 outputs per tile).
constexpr int BLOCK_Q = 32;

// Kernel configuration parameters.
struct Config
{
    int waves_per_wg;

    int kh = 3;
    int kw = 3;

    int channels_per_group = 8;

    int n_fold = 8;

    Direction direction = Direction::Fprop;

    SwizzleType swizzle_type = SwizzleType::None;

    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;

    // Size of the vector loads/stores
    int vector_size = 8;

    constexpr int group_size() const { return channels_per_group; }

    constexpr int block_c() const { return channels_per_group * waves_per_wg; }

    constexpr int block_q() const { return BLOCK_Q; }

    constexpr int block_groups() const { return waves_per_wg; }

    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    constexpr int num_waves() const { return waves_per_wg; }

    std::string GetName() const
    {
        std::string swz = "no-swizzle";
        if(swizzle_type == SwizzleType::XOR)
            swz = "xor-swizzle";
        else if(swizzle_type == SwizzleType::CyclicShift)
            swz = "cyclicshift-swizzle";

        std::string base =
            "tile_v2_grouped_8c_" + swz + "_waves_per_wg_" + std::to_string(waves_per_wg);
        if(epilogue == EpilogueType::RegistersToGlobalMemory)
            return base + "_skip_lds_epilogue";
        else
            return base + "_lds_epilogue";
    }
};

// All instantiated configurations.
//
// Layout: 4 variant groups × 18 configs each = 72 configs + 4 cyclic-shift = 76 total.
// Each group has 9 Dgrad + 9 Fprop configs:
//   waves_per_wg = 16,8,7,6,5,4,3,2,1
//
// Group 0 (indices  0-17): No swizzle, direct DRAM epilogue
// Group 1 (indices 18-35): No swizzle, LDS-staged epilogue
// Group 2 (indices 36-53): XOR swizzle, direct DRAM epilogue
// Group 3 (indices 54-71): XOR swizzle, LDS-staged epilogue
// Cyclic-shift (indices 72-75): 4 configs
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
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
        return false;

    if((par.groups % cfg.waves_per_wg) != 0)
        return false;

    // XOR swizzle constraint: BLOCK_Q must be a multiple of BLOCK_C8 for
    // multi-tile spatial decomposition. BLOCK_C8 = waves_per_wg.
    // BLOCK_Q = 32 is divisible by BLOCK_C8 only when waves_per_wg divides 32
    // (i.e., waves_per_wg ∈ {1,2,4,8,16}).
    if(cfg.swizzle_type == SwizzleType::XOR)
    {
        const int block_c8 = cfg.block_c() / 8;
        const int out_q    = (par.direction == Direction::Dgrad) ? par.w : par.q;
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
    static constexpr int GROUP_SIZE   = cfg.group_size();   // 8
    static constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4;     // 2
    static constexpr int GROUP_SIZE_8 = GROUP_SIZE / 8;     // 1

    static constexpr int BLOCK_Q = cfg.block_q();           // 32

    // Number of input columns loaded by each workgroup (output columns plus halo).
    static constexpr int BLOCK_W = BLOCK_Q + (cfg.kw - 1);  // 34

    // uint4 vectors per channel fiber (8 fp16 per uint4).
    static constexpr int BLOCK_C8 = cfg.block_c() / 8;      // waves_per_wg

    // Total channels per block in fp16 elements.
    static constexpr int BLOCK_C = BLOCK_C8 * 8;

    // fp16x4 groups per channel fiber.
    static constexpr int BLOCK_C4 = BLOCK_C / 4;            // 2 * waves_per_wg

    // Number of uint4 vectors to store per output row (LDS epilogue path).
    static constexpr int STORE_VECS = BLOCK_Q * BLOCK_C8;

    // LDS double buffering for input loads.
    static constexpr int NUM_INPUT_LDS_BUFFERS     = 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C8  = BLOCK_C8 * BLOCK_W;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C4  = INPUT_LDS_BUFFER_SIZE_C8 * 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_FP16 = INPUT_LDS_BUFFER_SIZE_C8 * 8;

    // Tile-level async load constants.
    static constexpr int NUM_WAVES     = cfg.waves_per_wg;
    static constexpr int LANES_PER_ROW = WAVE_SIZE / BLOCK_C8;
    static constexpr int TOTAL_SPATIAL = cfg.block_size() / BLOCK_C8;

    static constexpr int KH_KW = cfg.kh * cfg.kw;
    static constexpr int KW    = cfg.kw;
    static constexpr SwizzleType SWIZZLE_TYPE = cfg.swizzle_type;

    // -----------------------------------------------------------------------
    // Mfma — tile distribution for MFMA 16x16x32 results.
    //
    // mfma_f32_16x16x32_f16 result layout:
    //   m = 4*(lane/16) + vec_idx  (0..15)
    //   n = lane % 16               (0..15)
    //
    // GT mapping:
    //   output_col = 2*n + GT::q(m) = 2*(lane%16) + lane/32
    //   c4_within_group = (m/4) % 2 = (lane/16) % 2
    //
    // 3D tile: [BLOCK_Q=32, BLOCK_C4=2*NUM_WAVES, 4]
    //   X0 = BLOCK_Q [16, 2]: result_n(16) × GT::q(2)
    //   X1 = BLOCK_C4 [NUM_WAVES, 2]: wave_id(NUM_WAVES) × c4(2)
    //   X2 = 4: vectorization
    //
    // P1 merge = {2, 2, 16}:
    //   factor 0 = lane/32 (GT::q) → X0 factor 1
    //   factor 1 = (lane/16)%2 (c4) → X1 factor 1
    //   factor 2 = lane%16 (result_n) → X0 factor 0
    // -----------------------------------------------------------------------
    struct Mfma
    {
        static constexpr auto MakeAccTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<16, 2>,        // X0: Q = [16 result_n, 2 GT::q]
                                   ck_tile::sequence<NUM_WAVES, 2>, // X1: C4 = [NUM_WAVES wave, 2 c4]
                                   ck_tile::sequence<4>>,           // X2: vectorization
                    ck_tile::tuple<ck_tile::sequence<2>,            // P0 → X1
                                   ck_tile::sequence<1, 2, 1>>,    // P1 → X0, X1, X0
                    ck_tile::tuple<ck_tile::sequence<0>,            // P0 minor: X1 factor 0
                                   ck_tile::sequence<1, 1, 0>>,    // P1 minor: X0:1, X1:1, X0:0
                    ck_tile::sequence<3>,                           // Y0 → X2
                    ck_tile::sequence<0>>{});
        }
    };

    // -----------------------------------------------------------------------
    // Input — descriptors and distributions for the input activation tensor.
    // -----------------------------------------------------------------------
    struct Input
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Input;

        static CK_TILE_DEVICE auto MakeDramReadDescriptor(int hi, int wi, int C_total, int px)
        {
            return Shared::MakeDramReadDescriptor(hi, wi, C_total, px);
        }

        static constexpr auto MakeDramReadTileDistribution()
        {
            return Shared::MakeDramReadTileDistribution();
        }

        static constexpr auto MakeLdsWriteDescriptor()
        {
            return Shared::MakeLdsWriteDescriptor();
        }

        static constexpr auto MakeLdsReadDescriptor()
        {
            return Shared::MakeLdsReadDescriptor();
        }
    };

    // -----------------------------------------------------------------------
    // Weight — descriptors and distributions for the filter weight tensor.
    // -----------------------------------------------------------------------
    struct Weight
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Weight;

        // Weight LDS sizing: block_groups * GROUP_SIZE * kh * kw * GROUP_SIZE_8 uint4.
        // Each uint4 holds one (k, r, s) tap: all 8 input channels.
        static constexpr int WEIGHT_LDS_SIZE_UINT2 =
            cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE * GROUP_SIZE_4;
        static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_SIZE_UINT2 / 2;
        static constexpr int NUM_WEIGHT_PASSES =
            (WEIGHT_LDS_SIZE_UINT4 + cfg.block_size() - 1) / cfg.block_size();
        static constexpr int WEIGHT_LDS_PADDED_UINT4 = NUM_WEIGHT_PASSES * cfg.block_size();
        static constexpr int WEIGHT_LDS_READ_K = cfg.block_c();

        static constexpr auto MakeDramReadDescriptor()
        {
            return Shared::MakeDramReadDescriptor();
        }
        static constexpr auto MakeDramReadTileDistribution()
        {
            return Shared::MakeDramReadTileDistribution();
        }
        static constexpr auto MakeLdsWriteDescriptor()
        {
            return Shared::MakeLdsWriteDescriptor();
        }
        static constexpr auto MakeLdsReadDescriptor()
        {
            return Shared::MakeLdsReadDescriptor();
        }
    };

    // -----------------------------------------------------------------------
    // Output — descriptors and distributions for the output activation tensor.
    // -----------------------------------------------------------------------
    struct Output
    {
        using Shared = SharedDescriptors<TileConstants<cfg>>::Output;

        static constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * BLOCK_Q;

        static constexpr auto MakeLdsWriteDescriptor()
        {
            return Shared::MakeLdsWriteDescriptor();
        }

        static CK_TILE_DEVICE auto MakeDramWriteDescriptorNarrow(int ho, int wo, int C)
        {
            return Shared::MakeDramWriteDescriptorNarrow(ho, wo, C);
        }

        // Store distribution for wider LDS reads and DRAM stores.
        // Maps all block_size threads to [STORE_Q, BLOCK_C8, 8] positions.
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

        // Tile distribution for DRAM output writes (Toeplitz-specific).
        //
        // 4D: [1, BLOCK_Q=32, BLOCK_C4=2*NUM_WAVES, 4]
        //   X0 = 1 (row) → Y0
        //   X1 = 32 [16, 2]: result_n(16) × GT::q(2)
        //   X2 = BLOCK_C4 [NUM_WAVES, 2]: wave_id × c4_within_group
        //   X3 = 4 → Y1
        //
        // P1 merge = {2, 2, 16}: same factorization as Mfma distribution.
        static constexpr auto MakeDramWriteTileDistributionNarrow()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<1>,
                                   ck_tile::sequence<16, 2>,
                                   ck_tile::sequence<NUM_WAVES, 2>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<3>,     // P0 → X2
                                   ck_tile::sequence<2, 3, 2>>,  // P1 → X1, X2, X1
                    ck_tile::tuple<ck_tile::sequence<0>,     // P0 minor: X2 factor 0
                                   ck_tile::sequence<1, 1, 0>>,  // P1 minor: X1:1, X2:1, X1:0
                    ck_tile::sequence<1, 4>,                 // Y0→X0, Y1→X3
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
// InputLoader — shared DRAM→LDS, Toeplitz-specific LDS→register read.
//
// Reuses the shared InputLoader for DRAM→LDS (async buffer_load_lds with
// CK Tile pad/XOR/CyclicShift transforms, OOB checking, double buffering).
// The LDS→register read is custom for the Toeplitz structure because:
//   - The shared loader reads fp16x4_t at MFMA-distribution offsets (with S-loop)
//   - The Toeplitz kernel needs fp16x8_t at input_x = 2*(lane%16) + lane/16 (no S-loop)
//
// The Toeplitz LDS offset is precomputed using a temporary tile_window with the
// LDS read descriptor (to correctly handle XOR/CyclicShift swizzle transforms).
// ===================================================================
template <Config cfg>
struct InputLoaderToeplitz
{
    using TC = TileConstants<cfg>;

    // Inherit the shared InputLoader for DRAM→LDS infrastructure.
    direct_conv::InputLoader<TC, cfg> shared_loader;

    // Toeplitz-specific: precomputed LDS read offset in fp16 elements.
    ck_tile::index_t toeplitz_lds_offset;

    template <typename BlockCoords_>
    __device__ InputLoaderToeplitz(const BlockCoords_& bc,
                                   uint4* input_lds,
                                   const _Float16* __restrict__ in,
                                   int hi,
                                   int wi,
                                   int px)
        : shared_loader(bc, input_lds, in, hi, wi, px)
    {
        // Compute the Toeplitz-specific LDS read offset.
        // Input position: input_x = 2*(lane%16) + lane/16
        // Channel group: wave (one uint4 = 8 fp16 per group)
        //
        // We use the LDS read descriptor (which handles XOR/CyclicShift inverse
        // transforms) to compute the correct swizzled offset.
        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int wave = static_cast<int>(threadIdx.x) / WAVE_SIZE;

        const int input_x = 2 * (lane % 16) + lane / 16;

        // Create temporary LDS read window with the Mfma distribution to
        // extract the swizzled offset for the Toeplitz read position.
        // The LDS read descriptor is [BLOCK_W, BLOCK_C4, 4] with swizzle.
        // We need to find the byte offset for position (input_x, wave*2, 0)
        // which maps to the start of the 8 fp16 channels for this lane.
        //
        // For the Toeplitz kernel, each lane reads 8 contiguous fp16 (one uint4).
        // In the [BLOCK_W, BLOCK_C8, 8] raw layout:
        //   offset = input_x * BLOCK_C8 * 8 + wave * 8
        // With XOR/CyclicShift swizzle, the [BLOCK_W, BLOCK_C8] dimensions are
        // swizzled but the innermost 8 elements remain contiguous.
        //
        // The raw (no-swizzle) offset computation:
        if constexpr(TC::SWIZZLE_TYPE == SwizzleType::None)
        {
            toeplitz_lds_offset = input_x * TC::BLOCK_C8 * 8 + wave * 8;
        }
        else
        {
            // For XOR/CyclicShift, use the LDS read descriptor to get correct offsets.
            // The LDS read descriptor handles the inverse swizzle.
            // We read at (spatial=input_x, c4=wave*GROUP_SIZE_4, sub=0).
            // The descriptor is [BLOCK_W, BLOCK_C4, 4], and we need element offset
            // for position [input_x, wave*2, 0] (c4 = wave * GROUP_SIZE_4 = wave * 2).
            constexpr auto lds_read_desc = TC::Input::MakeLdsReadDescriptor();
            auto coord = ck_tile::make_tensor_coordinate(
                lds_read_desc,
                ck_tile::make_tuple(input_x, wave * TC::GROUP_SIZE_4, 0));
            toeplitz_lds_offset = coord.get_offset();
        }
    }

    __device__ __forceinline__ void fetch_tile_to_lds(int lds_buffer_index)
    {
        shared_loader.fetch_tile_to_lds(lds_buffer_index);
    }

    __device__ __forceinline__ void prefetch_tile_to_lds(int lds_buffer_index)
    {
        shared_loader.prefetch_tile_to_lds(lds_buffer_index);
    }

    // Read Toeplitz input from LDS: single fp16x8_t (no S-loop needed).
    __device__ __forceinline__ void read_from_lds(
        fp16x8_t& input_reg, int lds_buffer_index) const
    {
        const _Float16* base = reinterpret_cast<const _Float16*>(shared_loader.input_lds_ptr)
                               + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;
        __builtin_memcpy(&input_reg, base + toeplitz_lds_offset, sizeof(input_reg));
    }
};

// ===================================================================
// WeightLoader — shared DRAM→LDS, Toeplitz-specific LDS→register read.
// ===================================================================
template <Config cfg>
struct WeightLoader
{
    using TC = TileConstants<cfg>;

    template <typename BlockCoords_>
    __device__ static void load_to_lds(const BlockCoords_& bc,
                                       uint4* weight_lds,
                                       const _Float16* __restrict__ wei,
                                       int c_per_group,
                                       int k_per_group)
    {
        direct_conv::weight_load_to_lds<TC, cfg>(bc, weight_lds, wei, c_per_group, k_per_group);
    }

    // GT-specific weight read from LDS — same as v1 (unchanged).
    // Each lane reads one fp16x8_t per filter row R.
    __device__ static void read_from_lds(fp16x8_t (&weights_reg)[cfg.kh],
                                         uint4* weight_lds)
    {
        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int wave = static_cast<int>(threadIdx.x) / WAVE_SIZE;

        // Map lane to GT matrix position.
        int row = lane % 16;
        int g   = lane / 16;

        if constexpr(cfg.direction == Direction::Dgrad)
        {
            int s_dgrad = cfg.kw - 1 - g + GT::q(row);
            int c_out   = GT::k(row);

            if(GT::filter_is_zero(row, g * 2))
            {
                for(int r = 0; r < cfg.kh; r++)
                    *reinterpret_cast<uint4*>(&weights_reg[r]) = uint4{0, 0, 0, 0};
            }
            else
            {
                const __half* wei_half = reinterpret_cast<const __half*>(weight_lds);
                int base_k             = wave * TC::GROUP_SIZE;
                for(int r = 0; r < cfg.kh; r++)
                {
                    for(int k = 0; k < TC::GROUP_SIZE; k++)
                    {
                        int idx = (base_k + k) * cfg.kh * cfg.kw * TC::GROUP_SIZE +
                                  r * cfg.kw * TC::GROUP_SIZE + s_dgrad * TC::GROUP_SIZE + c_out;
                        weights_reg[r][k] = wei_half[idx];
                    }
                }
            }
        }
        else
        {
            int k_val = GT::k(row);
            int s_val = GT::s(row, g * 2);

            if(GT::filter_is_zero(row, g * 2))
            {
                for(int r = 0; r < cfg.kh; r++)
                    *reinterpret_cast<uint4*>(&weights_reg[r]) = uint4{0, 0, 0, 0};
            }
            else
            {
                int k_within_wg = wave * TC::GROUP_SIZE + k_val;
                for(int r = 0; r < cfg.kh; r++)
                {
                    int offset     = k_within_wg * cfg.kh * cfg.kw + r * cfg.kw + s_val;
                    weights_reg[r] = *reinterpret_cast<const fp16x8_t*>(&weight_lds[offset]);
                }
            }
        }
    }
};

// ===================================================================
// OutputWriter — direct DRAM writes (RegistersToGlobalMemory epilogue).
// ===================================================================
template <Config cfg>
using OutputWriter8c = direct_conv::OutputWriter<TileConstants<cfg>>;

// ===================================================================
// OutputWriterLds — LDS-staged writes (RegistersToLdsToGlobalMemory).
// ===================================================================
template <Config cfg>
using OutputWriterLds8c = direct_conv::OutputWriterLds<TileConstants<cfg>>;

// ===================================================================
// Main device function — self-contained Toeplitz compute loop.
//
// Unlike the 4c/16c kernels, this does NOT use grouped_conv_compute_loop
// because the Toeplitz structure has:
//   - No S-loop (S embedded in MFMA K=32 via GT)
//   - R-loop only
//   - fp16x8_t operands (not fp16x4_t)
//   - mfma_f32_16x16x32_f16 (not Mfma16x16x16)
// ===================================================================
template <Config cfg>
__device__ void conv2d_grouped_8c_fp16_cdna4_nhwc_impl_v2(const _Float16* __restrict__ in,
                                                            const _Float16* __restrict__ wei,
                                                            _Float16* __restrict__ out,
                                                            int N,
                                                            int groups,
                                                            int hi,
                                                            int wi,
                                                            int ho,
                                                            int wo,
                                                            int py,
                                                            int px)
{
    constexpr bool use_lds_epilogue = (cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory);
    using TC = TileConstants<cfg>;
    using OutputWriterType =
        std::conditional_t<use_lds_epilogue, OutputWriterLds8c<cfg>, OutputWriter8c<cfg>>;

    // --- Unified LDS buffer ---
    static constexpr int INPUT_TOTAL = TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8;
    static constexpr int WEIGHT_LDS  = TC::Weight::WEIGHT_LDS_SIZE_UINT4;
    static constexpr int IO_LDS      = use_lds_epilogue
                                            ? INPUT_TOTAL + TC::Output::OUTPUT_LDS_BUFFER_SIZE
                                            : INPUT_TOTAL;
    static constexpr int UNIFIED_LDS_SIZE = (WEIGHT_LDS > IO_LDS) ? WEIGHT_LDS : IO_LDS;
    __shared__ uint4 lds_buf[UNIFIED_LDS_SIZE];
    uint4* input_lds  = lds_buf;
    uint4* output_lds = lds_buf + INPUT_TOTAL;

    // --- Coordinate setup ---
    BlockCoords<cfg> bc(groups);
    if(bc.block_n >= N)
        return;

    // --- Weight loading (uses start of buffer, before input phase) ---
    fp16x8_t weights_reg[cfg.kh];
    WeightLoader<cfg>::load_to_lds(bc, lds_buf, wei, TC::GROUP_SIZE, TC::GROUP_SIZE);
    wait_vmcnt<0>();
    __syncthreads();

    WeightLoader<cfg>::read_from_lds(weights_reg, lds_buf);

    // --- Setup input loader and output writer ---
    InputLoaderToeplitz<cfg> il(bc, input_lds, in, hi, wi, px);
    OutputWriterType ow(bc, output_lds, out, ho, wo);

    __syncthreads();

    // --- Prefetch first input row ---
    il.prefetch_tile_to_lds(0);

    // --- Circular accumulator buffer ---
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    // --- Main loop: iterate over input rows ---
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                wait_vmcnt<0>();
                __syncthreads();

                int y = y_base + Y_LOCAL;

                // Fetch next input row into LDS.
                if((y + 1) < hi)
                {
                    il.fetch_tile_to_lds(tic);
                }

                // Load input operand: ONE fp16x8_t per lane (no S-loop).
                fp16x8_t input_reg;
                il.read_from_lds(input_reg, toc);

                // Accumulate: R-loop only (S is embedded in the Toeplitz structure).
                static_for<cfg.kh>(
                    [&]<int R>()
                    {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[cfg.kh - 1 - R], input_reg, acc[p_idx], 0, 0, 0);
                        else
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[R], input_reg, acc[p_idx], 0, 0, 0);
                    });

                tic ^= 1;
                toc ^= 1;

                // Flush completed output row.
                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_FLUSH], p_out);
                acc[P_FLUSH] = Zero;
            });
    }

    // --- Remainder: hi % kh leftover rows ---
    {
        int y_rem_base = (hi / cfg.kh) * cfg.kh;
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                if(Y_LOCAL >= hi % cfg.kh)
                    return;
                int y = y_rem_base + Y_LOCAL;

                wait_vmcnt<0>();
                __syncthreads();

                if((y + 1) < hi)
                {
                    il.fetch_tile_to_lds(tic);
                }

                fp16x8_t input_reg;
                il.read_from_lds(input_reg, toc);

                static_for<cfg.kh>(
                    [&]<int R>()
                    {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[cfg.kh - 1 - R], input_reg, acc[p_idx], 0, 0, 0);
                        else
                            acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                weights_reg[R], input_reg, acc[p_idx], 0, 0, 0);
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out             = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_FLUSH], p_out);
                acc[P_FLUSH] = Zero;
            });
    }

    // --- Tail flush: output rows not flushed by the main/remainder loops ---
    for(int p_out = hi - cfg.kh + 1 + py; p_out < ho; p_out++)
    {
        __syncthreads();
        int p_idx = (p_out - py + cfg.kh) % cfg.kh;
        fp32x4_t slot;
        dispatch<cfg.kh>(p_idx,
                         [&]<int P>()
                         {
                             slot   = acc[P];
                             acc[P] = Zero;
                         });
        ow.flush(slot, p_out);
    }
}

// ============================================================================
// Global kernel entry point
// ============================================================================

template <Config cfg>
__global__ void conv2d_grouped_8c_fp16_nhwc_cdna4_v2(const _Float16* __restrict__ in,
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
    conv2d_grouped_8c_fp16_cdna4_nhwc_impl_v2<cfg>(in, wei, out,
                                                     N, groups, hi, wi, ho, wo, py, px);
}

// ============================================================================
// Launch dispatch
// ============================================================================

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
        conv2d_grouped_8c_fp16_nhwc_cdna4_v2<configs[I]>
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
            if(par.channels_per_group() != 8)
                return false;
            if(par.c_tot % 8 != 0)
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

} // namespace ck_tile::direct_conv::grouped_8c_tile::v2
