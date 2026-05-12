// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CK Tile v2 implementation of 32-channel grouped convolution.
//
// This kernel uses mfma_f32_16x16x32_f16 with full 32-channel reduction and explicit S-loop. 
// Each group's 32 channels are split across 2 waves (wave_half=0 handles K[0:15],
// wave_half=1 handles K[16:31]).
//
// v2 uses CK Tile abstractions for all data movement:
//   - Input DRAM→LDS: shared InputLoader with CK Tile descriptors
//   - Weight DRAM→LDS: shared weight_load_to_lds
//   - Output writes: shared OutputWriter (direct DRAM) or OutputWriterLds (LDS-staged)
//   - Swizzle support: None / XOR / CyclicShift via descriptor transforms

#pragma once

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_kernel_base.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp"
#include "ck_tile/ops/direct_convolution/utils/mfma.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/load_tile.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile::direct_conv::grouped_32c_tile::v2
{

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Thread block output is 16 columns wide (fixed by mfma_f32_16x16x32_f16 M=16).
constexpr int BLOCK_Q = 16;

// Kernel configuration parameters.
struct Config
{
    // Number of waves per workgroup.
    // Must be even: each group uses 2 waves (wave_half=0 and wave_half=1).
    // Each pair of waves computes BLOCK_Q=16 columns and 32 output channels.
    int waves_per_wg;

    // Filter width & height
    int kh = 3;
    int kw = 3;

    // Batch folding factor.
    int n_fold = 8;

    // Number of channels per convolution group.
    int channels_per_group = 32;

    // Uniform accessor for shared code (alias for channels_per_group).
    constexpr int group_size() const { return channels_per_group; }

    Direction direction = Direction::Fprop;

    SwizzleType swizzle_type = SwizzleType::None;

    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;

    // Size of the vector loads/stores
    int vector_size = 8;

    // Number of waves per group (always 2 for 32c: each wave handles 16 of 32 channels).
    constexpr int waves_per_group() const { return 2; }

    // Total number of waves per workgroup.
    constexpr int num_waves() const { return waves_per_wg; }

    // Tile size in the channel dimension: number of input channels processed by one workgroup.
    constexpr int block_c() const { return channels_per_group * block_groups(); }

    // Tile size in the output column dimension (fixed by MFMA M=16).
    constexpr int block_q() const { return BLOCK_Q; }

    // Number of conv groups processed by one workgroup.
    constexpr int block_groups() const { return waves_per_wg / waves_per_group(); }

    // Number of threads per workgroup (thread block).
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    std::string GetName() const
    {
        std::string swz = "no-swizzle";
        if (swizzle_type == SwizzleType::XOR)
            swz = "xor-swizzle";
        else if (swizzle_type == SwizzleType::CyclicShift)
            swz = "cyclicshift-swizzle";

        std::string vector_size_str = "_vec_" + std::to_string(vector_size);
        std::string base = "v2_grouped_32c_" + swz + "_waves_per_wg_" + std::to_string(waves_per_wg) + vector_size_str;
        if(epilogue == EpilogueType::RegistersToGlobalMemory)
            return base + "_skip_lds_epilogue";
        else
            return base + "_lds_epilogue";
    }
};

// All instantiated configurations.
//
// Layout: 4 variant groups × 16 configs each = 64 configs total.
// Each group has 8 Dgrad + 8 Fprop configs:
//   waves_per_wg = 16,14,12,10,8,6,4,2  (must be even)
//
// Group 0 (indices  0-15): No swizzle, direct DRAM epilogue
// Group 1 (indices 16-31): No swizzle, LDS-staged epilogue
// Group 2 (indices 32-47): XOR swizzle, direct DRAM epilogue
// Group 3 (indices 48-63): XOR swizzle, LDS-staged epilogue
constexpr Config configs[] = {
    // TODO: Prune the configurations: we need 2, 4, 8, 16 waves per workgroup.
    // ---- Group 0: No swizzle, direct DRAM epilogue (default) ----
    // Dgrad (indices 0-7)
    {.waves_per_wg = 16, .direction = Direction::Dgrad},
    {.waves_per_wg = 14, .direction = Direction::Dgrad},
    {.waves_per_wg = 12, .direction = Direction::Dgrad},
    {.waves_per_wg = 10, .direction = Direction::Dgrad},
    {.waves_per_wg = 8, .direction = Direction::Dgrad},
    {.waves_per_wg = 6, .direction = Direction::Dgrad},
    {.waves_per_wg = 4, .direction = Direction::Dgrad},
    {.waves_per_wg = 2, .direction = Direction::Dgrad},
    // Fprop (indices 8-15)
    {.waves_per_wg = 16},
    {.waves_per_wg = 14},
    {.waves_per_wg = 12},
    {.waves_per_wg = 10},
    {.waves_per_wg = 8},
    {.waves_per_wg = 6},
    {.waves_per_wg = 4},
    {.waves_per_wg = 2},
    // ---- Group 1: No swizzle, LDS-staged epilogue ----
    // Dgrad (indices 16-23)
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 14, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 12, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 10, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2, .direction = Direction::Dgrad,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // Fprop (indices 24-31)
    {.waves_per_wg = 16,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 14,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 12,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 10,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // ---- Group 2: XOR swizzle, direct DRAM epilogue ----
    // Dgrad (indices 32-39)
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 14, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 12, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 10, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 6, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 2, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR},
    // Fprop (indices 40-47)
    {.waves_per_wg = 16,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 14,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 12,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 10,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 6,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 4,
     .swizzle_type = SwizzleType::XOR},
    {.waves_per_wg = 2,
     .swizzle_type = SwizzleType::XOR},
    // ---- Group 3: XOR swizzle, LDS-staged epilogue ----
    // Dgrad (indices 48-55)
    {.waves_per_wg = 16, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 14, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 12, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 10, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    // Fprop (indices 56-63)
    {.waves_per_wg = 16,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 14,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 12,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 10,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 8,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 6,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 4,
     .swizzle_type = SwizzleType::XOR,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
    {.waves_per_wg = 2,
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
     // Small vector load/store configs for padding cases (channels_per_group < 32)
     // Dgrad CyclicShift (indices 68-72)
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 16}, // TODO: Remove
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 8}, // TODO: Remove
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 4},
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 2},
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
     // Fprop CyclicShift (indices 73-77)
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 16}, // TODO: Remove
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 8},  // TODO: Remove
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 4},
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 2},
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
     // No-swizzle fallback for padding (indices 78-79)
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
     {.waves_per_wg = 8,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
      // Additional cyclic-shift epilogue
    {.waves_per_wg = 4,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     {.waves_per_wg = 4,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
     {.waves_per_wg = 4, .direction=Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToLdsToGlobalMemory},
     {.waves_per_wg = 4, .direction=Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift,
     .epilogue = EpilogueType::RegistersToGlobalMemory},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
        return false;
    // block_groups = waves_per_wg / 2; groups must be a multiple.
    if((par.groups % cfg.block_groups()) != 0)
        return false;
    if(cfg.swizzle_type == SwizzleType::XOR && !xor_config_valid(cfg, par))
        return false;

    const bool padding_needed = par.channels_per_group() != 32 || par.filters_per_group() != 32;
    if(padding_needed && par.channels_per_group() % cfg.vector_size != 0)
        return false;
    if(padding_needed && par.filters_per_group() % cfg.vector_size != 0)
        return false;

    return true;
}

inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    return get_launch_params_impl(configs[config_idx], par);
}

// -----------------------------------------------------------------------
// TileConstants — inherits all shared constants from TileConstantsBase.
// Only adds the three variant-specific tile distributions:
//   Mfma::MakeAccTileDistribution       (mfma_f32_16x16x32_f16 lane layout)
//   Weight::MakeLdsReadTileDistribution (Fprop weight read from LDS)
//   Output::MakeDramWriteTileDistributionNarrow
// -----------------------------------------------------------------------
template <Config cfg>
struct TileConstants : direct_conv::TileConstantsBase<cfg>
{
    using Base = direct_conv::TileConstantsBase<cfg>;

    static constexpr int WAVES_PER_WG = cfg.waves_per_wg;
    static constexpr int KH_KW_       = cfg.kh * cfg.kw;

    // -----------------------------------------------------------------------
    // Mfma — tile distribution for mfma_f32_16x16x32_f16 results.
    //
    // mfma_f32_16x16x32_f16 lane mapping (64-lane wave):
    //   lane_q   = lane % 16 → Q column (16 output cols)
    //   lane_c4  = lane / 16 → C4 group (4 groups of fp32x4)
    //
    // For 32c: the GROUP_SIZE_4 = 8, so BLOCK_C4 = waves_per_wg * 8.
    // Each group uses 2 waves, so wave_half = wave_within_group % 2.
    // wave_half=0 → K[0:15], wave_half=1 → K[16:31].
    // The result layout is the same as 16c (16x16 output tile per MFMA),
    // but across twice as many waves per group.
    //
    // 3D tile: [BLOCK_Q=16, BLOCK_C4, 4]
    //   X0 = 16 [16]: lane_q → P1 factor 1
    //   X1 = BLOCK_C4 [WAVES_PER_WG, 4]: warp_id → P0, lane_c4 → P1 factor 0
    //   X2 = 4: vectorization → Y0
    // -----------------------------------------------------------------------
    struct Mfma
    {
        static constexpr auto MakeAccTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<16>,
                                   ck_tile::sequence<WAVES_PER_WG, 4>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>>{});
        }
    };

    // -----------------------------------------------------------------------
    // Weight — adds the Fprop LDS read tile distribution.
    //
    // mfma_f32_16x16x32_f16 B operand:
    //   lane_k   = lane % 16 → K-column (outer-product dim of B)
    //   lane_c8  = lane / 16 → C-reduction group (4 groups, each 8 fp16)
    //
    // 3D tile: [block_c, kh*kw, GROUP_SIZE=32]
    //   X0 = block_c [WAVES_PER_WG, 16]: warp_id → P0, lane_k → P1 factor 1
    //   X1 = kh*kw → Y0 (filter positions)
    //   X2 = GROUP_SIZE [4, 8]: lane_c8 → P1 factor 0, sub-channel → Y1
    //
    // P1 merge = {4, 16}: factor 0 = lane/16 → X2 factor 0, factor 1 = lane%16 → X0 factor 1
    // -----------------------------------------------------------------------
    struct Weight : Base::Weight
    {
        static constexpr auto MakeLdsReadTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<WAVES_PER_WG, 16>,
                                   ck_tile::sequence<KH_KW_>,
                                   ck_tile::sequence<4, 8>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<3, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<0, 1>>,
                    ck_tile::sequence<2, 3>,
                    ck_tile::sequence<0, 1>>{});
        }

        // Dgrad distribution: single-warp [32, 16] encoding.
        // Each warp independently reads 32 K_out rows × 16 C columns
        // from the per-warp base pointer.
        //
        // Output (post-transpose): [C=16, K_out=32]
        //   X0 = <16> (C lanes), X1 = <4, 2, 4> (K_out = 4*2*4 = 32)
        //     - X1[0] = 4: P-mapped via lane/16 (selects K_out quarter)
        //     - X1[1] = 2: Y-mapped (2 ds_read calls per filter position)
        //     - X1[2] = 4: Y-mapped (4 fp16 per ds_read, vectorized)
        //   P merge{4,16}: factor 0 (lane/16) → X1[0], factor 1 (lane%16) → X0[0]
        //   Y dims: X1 factor 1 (size 2) and X1 factor 2 (size 4)
        //
        // InputTileDistributionTraits derives the pre-transpose input encoding.
        static constexpr auto MakeLdsReadTileDistributionDgrad()
        {
            using OutputEncode = ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<16>,
                               ck_tile::sequence<4, 2, 4>>,
                ck_tile::tuple<ck_tile::sequence<2, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 0>>,
                ck_tile::sequence<2, 2>,
                ck_tile::sequence<1, 2>>;

            using InputEncode = typename ck_tile::InputTileDistributionTraits<
                OutputEncode, _Float16>::TransposedDstrEncode;

            return ck_tile::make_static_tile_distribution(InputEncode{});
        }
    };

    // -----------------------------------------------------------------------
    // Output — adds the narrow DRAM write tile distribution.
    //
    // 4D tile: [1, BLOCK_Q=16, BLOCK_C4, 4]
    //   X0 = 1 (row) → Y0
    //   X1 = 16 (Q) → P1 factor 1
    //   X2 = BLOCK_C4 [WAVES_PER_WG, 4]: warp_id → P0, lane_c4 → P1 factor 0
    //   X3 = 4 → Y1
    // -----------------------------------------------------------------------
    struct Output : direct_conv::TileConstantsBase<cfg>::Output
    {
        static constexpr auto MakeDramWriteTileDistributionNarrow()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<1>,
                                   ck_tile::sequence<16>,
                                   ck_tile::sequence<WAVES_PER_WG, 4>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<3>, ck_tile::sequence<3, 2>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<1, 4>,
                    ck_tile::sequence<0, 0>>{});
        }
    };
};

// Workgroup-level coordinates derived from blockIdx.
template <Config cfg>
using BlockCoords = direct_conv::BlockCoords<cfg>;

// ===================================================================
// InputLoader32c — shared DRAM→LDS, 32c-specific LDS→register read.
//
// Reuses the shared InputLoader for DRAM→LDS (async buffer_load_lds with
// CK Tile pad/XOR/CyclicShift transforms, OOB checking, double buffering).
// The LDS→register read returns fp16x8_t (8 fp16 = one uint4) instead of
// fp16x4_t, matching the mfma_f32_16x16x32_f16 operand size.
//
// Lane mapping for LDS reads:
//   lane_q  = lane % 16 → spatial position (0..15)
//   lane_c8 = lane / 16 → C8 slice within group (0..3)
//   wave_group = wave / 2 → which group
//   wave_half  = wave % 2 → which half (0: K[0:15], 1: K[16:31])
//
// LDS read offset:
//   (lane_q, wave_group * GROUP_SIZE_8 + wave_half * (GROUP_SIZE_8/2) + lane_c8)
// ===================================================================
template <Config cfg, bool Padded = true>
struct InputLoader32c : direct_conv::InputLoader<TileConstants<cfg>, cfg, ck_tile::fp16x8_t, Padded>
{
    using base = direct_conv::InputLoader<TileConstants<cfg>, cfg, ck_tile::fp16x8_t, Padded>;
    using TC = TileConstants<cfg>;

    template <typename BlockCoords_>
    __device__ InputLoader32c(const BlockCoords_& bc,
                               uint4* input_lds,
                               const _Float16* __restrict__ in,
                               int hi,
                               int wi,
                               int px,
                               int py,
                               int dx,
                               int dy,
                               int sx,
                               int sy,
                               int c_per_group = TC::GROUP_SIZE)
        : base(bc, input_lds, in, hi, wi, px, py, dx, dy, sx, sy, c_per_group, /*don't init MFMA offset in base*/false)
    {
        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int wave = static_cast<int>(threadIdx.x) / WAVE_SIZE;

        const int lane_q  = lane % 16;
        const int lane_c8 = lane / 16;
        const int wave_group = wave / 2;

        // Each group has GROUP_SIZE_8 = 4 uint4 slots.
        // wave_half=0 reads C8 slots [0,1], wave_half=1 reads [2,3].
        // lane_c8 (0..3) selects which of the 4 C8 positions to read.
        // For 32c with mfma K=32: each lane reads 8 fp16 (one uint4).
        // The lane_c8 from the MFMA maps to one of 4 groups of 8 fp16.
        const int c8_pos = wave_group * TC::GROUP_SIZE_8 + lane_c8;

        for(int s = 0; s < cfg.kw; s++)
        {
            int spatial_pos = lane_q + s;
            if constexpr(TC::SWIZZLE_TYPE == SwizzleType::None)
            {
                base::mfma_lds_offsets[s] = spatial_pos * TC::BLOCK_C8 * 8 + c8_pos * 8;
            }
            else
            {
                constexpr auto lds_read_desc = TC::Input::MakeLdsReadDescriptor();
                auto coord = ck_tile::make_tensor_coordinate(
                    lds_read_desc,
                    ck_tile::make_tuple(spatial_pos, c8_pos * 2, 0));
                base::mfma_lds_offsets[s] = coord.get_offset();
            }
        }
    }
};

// ===================================================================
// WeightLoader — shared DRAM→LDS, 32c-specific LDS→register read.
//
// The 32c kernel stores fp16x8_t weights[kh*kw] (one per filter position).
// Dgrad uses shared weight_read_dgrad with WavesPerGroup=2 and 2
// load_tile_transpose reads per filter position (ds_read_b64_tr_b16).
// Fprop reads fp16x8_t directly from LDS using the tile distribution.
// ===================================================================
template <Config cfg>
struct WeightLoader : direct_conv::WeightAccessor8<cfg.kh, cfg.kw>
{
    using TC = TileConstants<cfg>;

    template <bool Padded_ = true, typename BlockCoords_>
    __device__ static void load_to_lds(const BlockCoords_& bc,
                                       uint4* weight_lds,
                                       const _Float16* __restrict__ wei,
                                       int c_per_group,
                                       int k_per_group)
    {
        direct_conv::weight_load_to_lds<TC, cfg, Padded_>(bc, weight_lds, wei, c_per_group, k_per_group);
    }

    // Read weights from LDS into registers (this->weights[]).
    __device__ void read_from_lds(uint4* weight_lds)
    {
        if constexpr(cfg.direction == Direction::Dgrad)
        {
            // Dgrad: transposed LDS reads via load_tile_transpose.
            // WavesPerGroup=2: wave_group selects conv group, wave_half selects C half.
            direct_conv::weight_read_dgrad<TC, cfg.kh, cfg.kw, direct_conv::WeightAccessor8<cfg.kh, cfg.kw>, 2>(
                *this, weight_lds);
        }
        else
        {
            // Fprop: read fp16x8_t from LDS using tile distribution.
            weight_read_fprop_32c<TC>(*this, weight_lds);
        }
    }
};

// Fprop weight read for 32c: loads weight tile and reinterprets as fp16x8_t.
template <typename TC, int KH, int KW>
__device__ void weight_read_fprop_32c(WeightAccessor8<KH, KW>& wa, uint4* weight_lds)
{
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

    // get_as<fp16x8_t>() reinterprets the thread_buffer<_Float16, KH_KW*8>
    // as thread_buffer<fp16x8_t, KH_KW>.
    const auto& vec_buf = weight_tile.get_thread_buffer().template get_as<ck_tile::fp16x8_t>();
    static_for<TC::KH_KW>(
        [&]<int khw>()
        {
            wa.weights[khw] = vec_buf[ck_tile::number<khw>{}];
        });
}

// OutputWriter — direct DRAM writes (RegistersToGlobalMemory epilogue).
template <Config cfg, bool Padded = true>
using OutputWriter = direct_conv::OutputWriter<TileConstants<cfg>, Padded>;

// OutputWriterLds — LDS-staged writes (RegistersToLdsToGlobalMemory).
template <Config cfg, bool Padded = true>
using OutputWriterLds = direct_conv::OutputWriterLds<TileConstants<cfg>, Padded>;

// Main device function.
template <Config cfg, bool Padded = true>
__device__ void ck_tile_conv2d_grouped_32c_fp16_nhwc_impl(const _Float16* __restrict__ in,
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
    using OutputWriterType = std::conditional_t<use_lds_epilogue,
        OutputWriterLds<cfg, Padded>, OutputWriter<cfg, Padded>>;

    direct_conv::grouped_conv_compute_loop<
        TC, cfg, Padded, Mfma16x16x32_32c,
        BlockCoords<cfg>, InputLoader32c<cfg, Padded>, WeightLoader<cfg>, OutputWriterType>(
        in, wei, out, N, groups, c_per_group, k_per_group, hi, wi, ho, wo, py, px);
}

template <Config cfg, bool Padded = true>
__global__ void ck_tile_conv2d_grouped_32c_fp16_nhwc(const _Float16* __restrict__ in,
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
    ck_tile_conv2d_grouped_32c_fp16_nhwc_impl<cfg, Padded>(in, wei, alpha, beta, out,
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
    const bool needs_padding = par.channels_per_group() != configs[0].group_size() ||
                               par.filters_per_group() != configs[0].group_size();

    auto kernel_launch = [&]<size_t I, bool P>()
    {
        auto view = SizeView<configs[I].direction>(par);
        ck_tile_conv2d_grouped_32c_fp16_nhwc<configs[I], P>
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

static bool channels_can_be_padded(const Conv2dParams& par)
{
    int c = par.channels_per_group();
    int k = par.filters_per_group();
    // Only pad to 32 if at least one dimension exceeds the 16c kernel's range.
    return c <= 32 && k <= 32 && (c > 16 || k > 16);
}

constexpr KernelVariant make_variant()
{
    return {
        .is_applicable =
            [](const Conv2dParams& par)
        {
            if(!is_applicable_base(par))
                return false;
            if(!channels_can_be_padded(par))
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

} // namespace ck_tile::direct_conv::grouped_32c_tile::v2
