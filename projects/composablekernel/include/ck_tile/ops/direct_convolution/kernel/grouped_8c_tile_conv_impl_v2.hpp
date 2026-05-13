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

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_kernel_base.hpp"
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
using ck_tile::fp16x4_t;
using ck_tile::fp16x8_t;
using ck_tile::fp32x4_t;
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
template <DataType DT = DataType::fp16>
struct Config
{
    static constexpr DataType data_type = DT;
    int waves_per_wg;

    int kh = 3;
    int kw = 3;

    int channels_per_group = 8;

    int n_fold = 8;

    Direction direction = Direction::Fprop;

    SwizzleType swizzle_type = SwizzleType::CyclicShift;

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

        std::string vector_size_str = "_vec_" + std::to_string(vector_size);
        std::string base =
            "v2_grouped_8c_" + swz + "_waves_per_wg_" + std::to_string(waves_per_wg) + vector_size_str;
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
// Group 0 (indices  0-17): Cyclic-shift swizzle, direct DRAM epilogue
// Group 1 (indices 18-35): Cyclic-shift swizzle, LDS-staged epilogue
// Group 2 (indices 36-53): XOR swizzle, direct DRAM epilogue
// Group 3 (indices 54-71): XOR swizzle, LDS-staged epilogue
// Cyclic-shift (indices 72-75): 4 configs
template <DataType DT = DataType::fp16>
struct KernelConfigurations
{
static constexpr Config<DT> configs[] = {
    // ---- Group 0: Cyclic-shift swizzle, direct DRAM epilogue (default) ----
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
    // ---- Group 1: Cyclic-shift swizzle, LDS-staged epilogue ----
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
     // ---- Group 4: cyclic-shift swizzle, LDS-staged epilogue ----
    // Small vector load/store configs for padding cases (channels_per_group < 8)
    // where we can't use vectorized accesses without out-of-bounds.
    // Dgrad CyclicShift (indices 76-78)
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 4},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 2},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
    // Fprop CyclicShift (indices 79-81)
    {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 4},
    {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 2},
    {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
    // No-swizzle fallback for padding (indices 82-83)
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
    {.waves_per_wg = 8,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
    // Additional cyclisft swizzle instances
    {.waves_per_wg = 16,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 16,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
};
static constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);
}; // KernelConfigurations

template <DataType DT = DataType::fp16>
inline bool is_valid_config(const Conv2dParams& par, const Config<DT>& cfg)
{
    if(par.direction != cfg.direction)
        return false;

    if((par.groups % cfg.waves_per_wg) != 0)
        return false;

    if(cfg.swizzle_type == SwizzleType::XOR && !direct_conv::xor_config_valid(cfg, par))
        return false;

    const bool padding_needed = par.channels_per_group() != 8 || par.filters_per_group() != 8;
    if(padding_needed && par.channels_per_group() % cfg.vector_size != 0)
        return false;
    if(padding_needed && par.filters_per_group() % cfg.vector_size != 0)
        return false;

    return true;
}

template <DataType DT = DataType::fp16>
inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    return get_launch_params_impl(KernelConfigurations<DT>::configs[config_idx], par);
}

// ===================================================================
// Tile constants — inherits shared base, adds 8c-specific distributions.
// ===================================================================
template <auto cfg>
struct TileConstants : direct_conv::TileConstantsBase<cfg>
{
    using Base = direct_conv::TileConstantsBase<cfg>;

    // Bring base constants into scope for use in distributions below.
    using Base::NUM_WAVES;
    using Base::BLOCK_C8;
    using Base::BLOCK_C4;
    using Base::LANES_PER_ROW;
    using Base::STORE_VECS;
    using Base::TOTAL_SPATIAL;

    // -----------------------------------------------------------------------
    // Mfma — tile distribution for MFMA 16x16x32 results (8c-specific).
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
    // Output — inherits base; overrides narrow write distribution (8c-specific).
    // -----------------------------------------------------------------------
    struct Output : Base::Output
    {
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
// Workgroup-level coordinates — shared with all variants.
// ===================================================================
template <auto cfg>
using BlockCoords = direct_conv::BlockCoords<cfg>;

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
template <auto cfg, bool Padded = true>
struct InputLoaderToeplitz
{
    // Register type for MFMA input operand (matches read_from_lds parameter type).
    using input_type = std::conditional_t<cfg.data_type == DataType::bf16, bf16x8_t, fp16x8_t>;

    using TC = TileConstants<cfg>;

    // Inherit the shared InputLoader for DRAM→LDS infrastructure.
    direct_conv::InputLoader<TC, cfg, std::conditional_t<cfg.data_type == DataType::bf16, ck_tile::bf16x4_t, ck_tile::fp16x4_t>, Padded, ToType<cfg.data_type>> shared_loader;

    // Toeplitz-specific: precomputed LDS read offset in fp16 elements.
    ck_tile::index_t toeplitz_lds_offset;

    template <typename BlockCoords_>
    __device__ InputLoaderToeplitz(const BlockCoords_& bc,
                                   uint4* input_lds,
                                   const ToType<cfg.data_type>* __restrict__ in,
                                   int hi,
                                   int wi,
                                   int px,
                                   int py,
                                   int dx,
                                   int dy,
                                   int sx,
                                   int sy,
                                   int c_per_group = TC::GROUP_SIZE)
        : shared_loader(bc, input_lds, in, hi, wi, px, py, dx, dy, sx, sy, c_per_group)
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
    // The S parameter is accepted for interface compatibility with the shared
    // compute loop but is ignored — the Toeplitz kernel uses INNER_KW=1.
    __device__ __forceinline__ void read_from_lds(
        input_type& input_reg, int /*S*/, int lds_buffer_index) const
    {
        const ToType<cfg.data_type>* base = reinterpret_cast<const ToType<cfg.data_type>*>(shared_loader.input_lds_ptr)
                               + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;
        __builtin_memcpy(&input_reg, base + toeplitz_lds_offset, sizeof(input_reg));
    }
};

// ===================================================================
// WeightLoader — shared DRAM→LDS, Toeplitz-specific LDS→register read.
//
// Instance-based: weights are stored in member array for access via
// get<R,S>() / get_transposed<R,S>() (S is ignored — Toeplitz has no S-loop).
// ===================================================================
template <auto cfg>
struct WeightLoader
{
    using TC = TileConstants<cfg>;
    using ElementType = ToType<cfg.data_type>;
    using weight_vec_type = std::conditional_t<cfg.data_type == DataType::bf16, bf16x8_t, fp16x8_t>;

    weight_vec_type weights_[cfg.kh];

    template <bool Padded_ = true, typename BlockCoords_>
    __device__ static void load_to_lds(const BlockCoords_& bc,
                                       uint4* weight_lds,
                                       const ElementType* __restrict__ wei,
                                       int c_per_group,
                                       int k_per_group)
    {
        direct_conv::weight_load_to_lds<TC, cfg, Padded_, BlockCoords_, ElementType>(bc, weight_lds, wei, c_per_group, k_per_group);
    }

    // GT-specific weight read from LDS into weights_ array.
    // Each lane reads one fp16x8_t per filter row R.
    __device__ void read_from_lds(uint4* weight_lds)
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
                    *reinterpret_cast<uint4*>(&weights_[r]) = uint4{0, 0, 0, 0};
            }
            else
            {
                const ElementType* wei_elem = reinterpret_cast<const ElementType*>(weight_lds);
                int base_k             = wave * TC::GROUP_SIZE;
                for(int r = 0; r < cfg.kh; r++)
                {
                    for(int k = 0; k < TC::GROUP_SIZE; k++)
                    {
                        int idx = (base_k + k) * cfg.kh * cfg.kw * TC::GROUP_SIZE +
                                  r * cfg.kw * TC::GROUP_SIZE + s_dgrad * TC::GROUP_SIZE + c_out;
                        weights_[r][k] = wei_elem[idx];
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
                    *reinterpret_cast<uint4*>(&weights_[r]) = uint4{0, 0, 0, 0};
            }
            else
            {
                int k_within_wg = wave * TC::GROUP_SIZE + k_val;
                for(int r = 0; r < cfg.kh; r++)
                {
                    int offset     = k_within_wg * cfg.kh * cfg.kw + r * cfg.kw + s_val;
                    weights_[r] = *reinterpret_cast<const weight_vec_type*>(&weight_lds[offset]);
                }
            }
        }
    }

    // Access weight for filter position (R, S). S is ignored (embedded in Toeplitz).
    template <int R, int S = 0>
    __device__ __forceinline__ weight_vec_type get() const { return weights_[R]; }

    // Access transposed weight for Dgrad: reverses R index.
    template <int R, int S = 0>
    __device__ __forceinline__ weight_vec_type get_transposed() const { return weights_[cfg.kh - 1 - R]; }
};

// ===================================================================
// OutputWriter — direct DRAM writes (RegistersToGlobalMemory epilogue).
// ===================================================================
template <auto cfg, bool Padded = true>
using OutputWriter8c = direct_conv::OutputWriter<TileConstants<cfg>, Padded, ToType<cfg.data_type>>;

// ===================================================================
// OutputWriterLds — LDS-staged writes (RegistersToLdsToGlobalMemory).
// ===================================================================
template <auto cfg, bool Padded = true>
using OutputWriterLds8c = direct_conv::OutputWriterLds<TileConstants<cfg>, Padded, ToType<cfg.data_type>>;

// ===================================================================
// Main device function — delegates to the shared compute loop with
// INNER_KW=1 (Toeplitz: S embedded in MFMA K=32, no explicit S-loop).
// ===================================================================
template <auto cfg, bool Padded = true>
__device__ void ck_tile_conv2d_grouped_8c_nhwc_impl(const ToType<cfg.data_type>* __restrict__ in,
                                                        const ToType<cfg.data_type>* __restrict__ wei,
                                                        ToType<cfg.data_type>* __restrict__ out,
                                                        int N,
                                                        int groups,
                                                        int c_per_group,
                                                        int k_per_group,
                                                        int hi,
                                                        int wi,
                                                        int ho,
                                                        int wo,
                                                        int py,
                                                        int px)
{
    constexpr bool use_lds_epilogue = (cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory);
    using TC = TileConstants<cfg>;
    using ElementType = ToType<cfg.data_type>;
    using MfmaFn = std::conditional_t<cfg.data_type == DataType::bf16, Mfma16x16x32_bf16, Mfma16x16x32>;
    using OutputWriterType =
        std::conditional_t<use_lds_epilogue, OutputWriterLds8c<cfg, Padded>, OutputWriter8c<cfg, Padded>>;

    direct_conv::grouped_conv_compute_loop<
        TC, cfg, Padded, MfmaFn,
        BlockCoords<cfg>, InputLoaderToeplitz<cfg, Padded>, WeightLoader<cfg>, OutputWriterType,
        /*INNER_KW=*/1, ElementType>(
        in, wei, out, N, groups, c_per_group, k_per_group, hi, wi, ho, wo, py, px);
}

// ============================================================================
// Global kernel entry point
// ============================================================================

template <auto cfg, bool Padded = true>
__global__ void ck_tile_conv2d_grouped_8c_nhwc(const ToType<cfg.data_type>* __restrict__ in,
                                                   const ToType<cfg.data_type>* __restrict__ wei,
                                                   double alpha,
                                                   double beta,
                                                   ToType<cfg.data_type>* __restrict__ out,
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
    ck_tile_conv2d_grouped_8c_nhwc_impl<cfg, Padded>(in, wei, out,
                                                     N, groups, c_per_group, k_per_group,
                                                     hi, wi, ho, wo, py, px);
}

// ============================================================================
// Launch dispatch
// ============================================================================

template <DataType DT = DataType::fp16, size_t... Is>
void launch_dispatch(int config_idx,
                     std::index_sequence<Is...>,
                     const LaunchParams& lp,
                     const Conv2dParams& par,
                     const void* in,
                     const void* wei,
                     void* out,
                     hipStream_t stream)
{
    using ElementType = ToType<DT>;
    using KC = KernelConfigurations<DT>;
    const bool needs_padding = par.channels_per_group() != KC::configs[0].group_size() ||
                               par.filters_per_group() != KC::configs[0].group_size();

    auto kernel_launch = [&]<size_t I, bool P>()
    {
        auto view = SizeView<KC::configs[I].direction>(par);
        ck_tile_conv2d_grouped_8c_nhwc<KC::configs[I], P>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const ElementType*>(in),
                static_cast<const ElementType*>(wei),
                1.0,
                0.0,
                static_cast<ElementType*>(out),
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

    auto dispatch_config = [&]<size_t I>()
    {
        if(needs_padding)
            kernel_launch.template operator()<I, true>();
        else
            kernel_launch.template operator()<I, false>();
    };

    (void)((config_idx == static_cast<int>(Is) ? (dispatch_config.template operator()<Is>(), true)
                                               : false) ||
           ...);
}

template <DataType DT = DataType::fp16>
inline void launch(int config_idx,
                   const LaunchParams& lp,
                   const Conv2dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   void* /*workspace*/,
                   hipStream_t stream)
{
    launch_dispatch<DT>(
        config_idx, std::make_index_sequence<KernelConfigurations<DT>::NUM_CONFIGS>{},
        lp, par, in, wei, out, stream);
}

static bool channels_can_be_padded(const Conv2dParams& par)
{
    int c = par.channels_per_group();
    int k = par.filters_per_group();
    // Only pad to 8 if at least one dimension exceeds the 4c kernel's range.
    return c <= 8 && k <= 8 && (c > 4 || k > 4);
}

template <DataType DT = DataType::fp16>
constexpr KernelVariant make_variant()
{
    return {
        .is_applicable =
            [](const Conv2dParams& par)
        {
            if(!is_applicable_base(par))
                return false;
            if(par.in_type != DT || par.wei_type != DT || par.out_type != DT)
                return false;
            if(!channels_can_be_padded(par))
                return false;
            return true;
        },
        .config_is_compatible = [](const Conv2dParams& par, int idx)
        { return is_valid_config<DT>(par, KernelConfigurations<DT>::configs[idx]); },
        .get_launch_params  = &get_launch_params<DT>,
        .launch             = &launch<DT>,
        .get_workspace_size = [](int, const Conv2dParams&) -> size_t { return 0; },
        .num_configs        = KernelConfigurations<DT>::NUM_CONFIGS,
    };
}

} // namespace ck_tile::direct_conv::grouped_8c_tile::v2
