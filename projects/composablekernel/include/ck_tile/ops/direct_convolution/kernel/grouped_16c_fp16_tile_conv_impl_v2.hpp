// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_kernel_base.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_output_writer.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_conv_compute_loop.hpp"
#include "ck_tile/ops/direct_convolution/utils/transpose_lds_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/mfma.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/tensor/buffer_view.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/load_tile.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile::direct_conv::grouped_16c_tile::v2
{

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

    // Size of the vector loads/stores
    int vector_size = 8;

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
        std::string swz = "no-swizzle";
        if (swizzle_type == SwizzleType::XOR)
            swz = "xor-swizzle";
        else if (swizzle_type == SwizzleType::CyclicShift)
            swz = "cyclicshift-swizzle";

        std::string vector_size_str = "_vec_" + std::to_string(vector_size);
        std::string base = "v2_grouped_16c_" + swz + "_waves_per_wg_" + std::to_string(waves_per_wg) + vector_size_str;
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
     // Small vector load/store configs for padding cases (channels_per_group < 16)
     // where we can't use vectorized accesses without out-of-bounds.
     // Dgrad CyclicShift (indices 76-79)
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 8},
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 4},
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 2},
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
     // Fprop CyclicShift (indices 80-83)
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 8},
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 4},
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 2},
     {.waves_per_wg = 8,
      .swizzle_type = SwizzleType::CyclicShift,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
     // No-swizzle fallback for padding (indices 84-85)
     {.waves_per_wg = 8, .direction = Direction::Dgrad,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
     {.waves_per_wg = 8,
      .epilogue = EpilogueType::RegistersToLdsToGlobalMemory, .vector_size = 1},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
        return false;
    if((par.groups % cfg.waves_per_wg) != 0)
        return false;
    // XOR swizzle constraint: BLOCK_Q must be a multiple of BLOCK_C8 for
    // multi-tile spatial decomposition. BLOCK_C8 = waves_per_wg * 2.
    // BLOCK_Q = 16 is divisible by BLOCK_C8 only when waves_per_wg divides 8
    // (i.e., waves_per_wg ∈ {1,2,4,8}). For other values, XOR is only valid
    // when the output fits in a single spatial tile.
    if(cfg.swizzle_type == SwizzleType::XOR && !xor_config_valid(cfg, par))
        return false;

    const bool padding_needed = par.channels_per_group() != 16 || par.filters_per_group() != 16;
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
//   Mfma::MakeAccTileDistribution       (mfma_f32_16x16x16f16 lane layout)
//   Weight::MakeLdsReadTileDistribution (Fprop weight read from LDS)
//   Output::MakeDramWriteTileDistributionNarrow
// -----------------------------------------------------------------------
template <Config cfg>
struct TileConstants : direct_conv::TileConstantsBase<cfg>
{
    using Base = direct_conv::TileConstantsBase<cfg>;

    // Direct members (not inherited) so nested structs can access them via
    // enclosing-class name lookup without qualifying through the base template.
    static constexpr int WAVES_PER_WG = cfg.waves_per_wg;
    static constexpr int KH_KW_       = cfg.kh * cfg.kw; // alias: Base::KH_KW inaccessible in Mfma

    // -----------------------------------------------------------------------
    // Mfma — tile distribution for mfma_f32_16x16x16f16 operands and results.
    //
    // mfma_f32_16x16x16f16 lane mapping (64-lane wave):
    //   lane_q   = lane % 16 → Q column (16 output cols)
    //   lane_c4  = lane / 16 → C4 group (4 groups of fp32x4)
    //
    // 3D tile: [BLOCK_Q=16, BLOCK_C4, 4]
    //   X0 = 16 [16]: lane_q → P1 factor 1
    //   X1 = BLOCK_C4 [WAVES_PER_WG, 4]: warp_id → P0, lane_c4 → P1 factor 0
    //   X2 = 4: vectorization → Y0
    //
    // P0 (warp_id = WAVES_PER_WG) → X1 factor 0
    // P1 (lane_id = 64, merge {4, 16}):
    //   factor 0 (4 = lane/16) → X1 factor 1
    //   factor 1 (16 = lane%16) → X0 factor 0
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
    // mfma_f32_16x16x16f16 B operand:
    //   lane_k   = lane % 16 → K-column (outer-product dim of B)
    //   lane_c4  = lane / 16 → C-reduction group (4 groups, each 4 fp16)
    //
    // 3D tile: [block_c, kh*kw, GROUP_SIZE=16]
    //   X0 = block_c [WAVES_PER_WG, 16]: warp_id → P0, lane_k → P1 factor 1
    //   X1 = kh*kw → Y0 (filter positions)
    //   X2 = GROUP_SIZE [4, 4]: lane_c4 → P1 factor 0, sub-channel → Y1
    //
    // P1 merge = {4, 16}: factor 0 = lane/16 → X2 factor 0, factor 1 = lane%16 → X0 factor 1
    // No R dimension — all Q positions come from within a single wave.
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
                                   ck_tile::sequence<4, 4>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<3, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<0, 1>>,
                    ck_tile::sequence<2, 3>,
                    ck_tile::sequence<0, 1>>{});
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

// InputLoader — DRAM→LDS async load, double-buffered, MFMA reads.
template <Config cfg, bool Padded = true>
using InputLoader = direct_conv::InputLoader<TileConstants<cfg>, cfg, ck_tile::fp16x4_t, Padded>;

// WeightLoader — async weight loads to LDS, then register reads.
template <Config cfg>
struct WeightLoader : direct_conv::WeightAccessor<cfg.kh, cfg.kw>
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
                this->weights[khw] = weight_lds_fp16.template transpose_get<ck_tile::fp16x4_t>(
                    weight_base + khw * TC::GROUP_SIZE + tr_col, 0, true);
            }
        }
        else
        {
            direct_conv::weight_read_fprop<TC>(*this, weight_lds);
        }
    }
};

// OutputWriter — direct DRAM writes (RegistersToGlobalMemory epilogue).
template <Config cfg, bool Padded = true>
using OutputWriter = direct_conv::OutputWriter<TileConstants<cfg>, Padded>;

// OutputWriterLds — LDS-staged writes (RegistersToLdsToGlobalMemory).
template <Config cfg, bool Padded = true>
using OutputWriterLds = direct_conv::OutputWriterLds<TileConstants<cfg>, Padded>;

// Main device function.
template <Config cfg, bool Padded = true>
__device__ void ck_tile_conv2d_grouped_16c_fp16_nhwc_impl(const _Float16* __restrict__ in,
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
        TC, cfg, Padded, Mfma16x16x16,
        BlockCoords<cfg>, InputLoader<cfg, Padded>, WeightLoader<cfg>, OutputWriterType>(
        in, wei, out, N, groups, c_per_group, k_per_group, hi, wi, ho, wo, py, px);
}

template <Config cfg, bool Padded = true>
__global__ void ck_tile_conv2d_grouped_16c_fp16_nhwc(const _Float16* __restrict__ in,
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
    ck_tile_conv2d_grouped_16c_fp16_nhwc_impl<cfg, Padded>(in, wei, alpha, beta, out,
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
        ck_tile_conv2d_grouped_16c_fp16_nhwc<configs[I], P>
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
    // Only pad to 16 if at least one dimension exceeds the 8c kernel's range.
    return c <= 16 && k <= 16 && (c > 8 || k > 8);
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

} // namespace ck_tile::direct_conv::grouped_16c_tile::v2
