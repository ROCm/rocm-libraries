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

    // Epilogue type - by default skip LDS staging and write directly from registers to global memory.
    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;

    // Size of the vector loads/stores
    // TODO: We should use different load and store vector sizes.
    int vector_size = 8;

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
        std::string swz = "no-swizzle";
        if (swizzle_type == SwizzleType::XOR)
            swz = "xor-swizzle";
        else if (swizzle_type == SwizzleType::CyclicShift)
            swz = "cyclicshift-swizzle";
        
        std::string vector_size_str = "_vec_" + std::to_string(vector_size);
        std::string waves_c64_str = "_waves_c64_" + std::to_string(waves_c64);
        std::string waves_q4_str = "_waves_q4_" + std::to_string(waves_q4);
        std::string base = "v3_grouped_4c_" + swz + waves_c64_str + waves_q4_str + vector_size_str;
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
     // Small vector load/store configs for padding cases (channels_per_group < 4) 
     // where we can't use vectorized accesses without out-of-bounds.
     // The only relevant vector size are 1 and 1 since we need the number of both input and output channels
     // to be divisible by the vector size, and we have at most 4 channels per group.
     // In the K=C=4 case, we are able to use the full 8-wide vector size for better performance.
     {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2},
     {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1},
     {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Fprop,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2},
     {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Fprop,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1},
     {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2},
     {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1},
     {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Fprop,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2},
     {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Fprop,
     .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1},
     // TODO: These configurations produce wrong results.
    //  {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
    //  .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2},
    //  {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
    //  .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1},
    //  {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Fprop,
    //  .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2},
    //  {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Fprop,
    //  .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
        return false;
    if((par.groups % cfg.block_groups()) != 0)
        return false;
    const int out_q = (par.direction == Direction::Dgrad) ? par.w : par.q;
    if(out_q < cfg.block_q() && cfg.waves_q4 > 1)
        return false;
    if(cfg.swizzle_type == SwizzleType::XOR && !xor_config_valid(cfg, par))
        return false;

    const bool padding_needed = par.channels_per_group() != 4 || par.filters_per_group() != 4;
    if (padding_needed && par.channels_per_group() % cfg.vector_size != 0)
        return false;
    if (padding_needed && par.filters_per_group() % cfg.vector_size != 0)
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
//   Mfma::MakeAccTileDistribution       (mfma_f32_4x4x4f16 lane layout)
//   Weight::MakeLdsReadTileDistribution (Fprop weight read from LDS)
//   Output::MakeDramWriteTileDistributionNarrow
// -----------------------------------------------------------------------
template <Config cfg>
struct TileConstants : direct_conv::TileConstantsBase<cfg>
{
    using Base = direct_conv::TileConstantsBase<cfg>;

    // Compile-time aliases for the two wave-dimension sizes used in the
    // tile distributions below. cfg.waves_q4 / cfg.waves_c64 are accessible
    // as template parameters of the enclosing struct, but spelled out here
    // to make the distribution code self-documenting.
    static constexpr int WAVES_Q4  = cfg.waves_q4;
    static constexpr int WAVES_C64 = cfg.waves_c64;

    // -----------------------------------------------------------------------
    // Mfma — tile distribution for mfma_f32_4x4x4f16 operands and results.
    //
    // Maps (P0=warp_id, P1=lane_id) to a 3D tile coordinate
    //   (q_local, c4_local, c_sub) where:
    //   q_local  = warp_q * WARP_Q + lane_col  ∈ [0, block_q)
    //   c4_local = warp_c64 * 16 + lane_batch  ∈ [0, BLOCK_C4)
    //   c_sub    ∈ [0, 4) — vectorization (Y dimension)
    //
    // mfma_f32_4x4x4f16 lane layout (64-lane wave):
    //   lane_col   = (lane % 4)       → Q column within warp (4 output cols)
    //   lane_batch = (lane / 4) % 16  → C4 group within warp (16 groups of 4)
    //
    // P0 merge = {waves_q4, waves_c64}: warp_q → X0 factor 0, warp_c64 → X1 factor 0
    // P1 merge = {16, 4}:               lane_batch → X1 factor 1, lane_col → X0 factor 1
    // Y0 (length 4) → X2 (vectorization)
    // -----------------------------------------------------------------------
    struct Mfma
    {
        static constexpr auto MakeAccTileDistribution()
        {
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
    // Weight — adds the Fprop LDS read tile distribution.
    //
    // P0 merge = {waves_q4, waves_c64}:
    //   factor 0 (waves_q4)  → R dim (replicated across Q-waves)
    //   factor 1 (waves_c64) → X0 factor 0 (K-channel wave group)
    // P1 merge = {16, 4}:
    //   factor 0 (16 = lane_batch) → X0 factor 1
    //   factor 1 (4  = lane_col)   → X0 factor 2
    // Y0 (kh*kw) → X1, Y1 (GROUP_SIZE=4) → X2
    // R = [waves_q4]: all Q-waves read the same weights (replication).
    // -----------------------------------------------------------------------
    struct Weight : Base::Weight
    {
        static constexpr auto MakeLdsReadTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<WAVES_Q4>,
                    ck_tile::tuple<ck_tile::sequence<WAVES_C64, 16, 4>,
                                   ck_tile::sequence<Base::KH_KW>,
                                   ck_tile::sequence<Base::GROUP_SIZE>>,
                    ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<1, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 2>>,
                    ck_tile::sequence<2, 3>,
                    ck_tile::sequence<0, 0>>{});
        }
    };

    // -----------------------------------------------------------------------
    // Output — adds the narrow DRAM write tile distribution.
    //
    // 4D tile: [1, block_q, BLOCK_C4, 4]
    // P0 merge = {waves_q4, waves_c64}:
    //   factor 0 (waves_q4)  → X1 factor 0 (Q wave group)
    //   factor 1 (waves_c64) → X2 factor 0 (C wave group)
    // P1 merge = {16, 4}:
    //   factor 0 (16=batch) → X2 factor 1
    //   factor 1 (4=col)    → X1 factor 1
    // Y0 (1) → X0 (trivial row), Y1 (4) → X3 (vectorization)
    // -----------------------------------------------------------------------
    struct Output : Base::Output
    {
        static constexpr auto MakeDramWriteTileDistributionNarrow()
        {
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
using BlockCoords = direct_conv::BlockCoords<cfg>;

// Handles input loads from global memory into LDS and then into registers.
template <Config cfg>
using InputLoader = direct_conv::InputLoader<TileConstants<cfg>, cfg>;

// Handles weight loading (DRAM → LDS → registers) and provides
// register-resident weight access via inherited WeightAccessor.
template <Config cfg>
struct WeightLoader : direct_conv::WeightAccessor<cfg.kh, cfg.kw>
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

    // Read weights from LDS into registers (this->weights[]).
    __device__ void read_from_lds(uint4* weight_lds)
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
                static_cast<ck_tile::index_t>(TC::Weight::WEIGHT_LDS_SIZE_UINT4 *
                                              (sizeof(uint4) / sizeof(_Float16)))};

            using TransposeLayout = TransposeLDSLayout<4, 4, 16>;
            const int tr_batch    = TransposeLayout::batch(lane);
            const int tr_row      = TransposeLayout::row(lane);
            int filter_local      = wave_c64 * 64 + tr_batch * TC::GROUP_SIZE + tr_row;

            const ck_tile::index_t weight_base =
                filter_local * cfg.kh * cfg.kw * TC::GROUP_SIZE;

            for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
            {
                this->weights[khw] = output_lds_fp16.template transpose_get<ck_tile::fp16x4_t>(
                    weight_base + khw * TC::GROUP_SIZE, 0, true);
            }
        }
        else
        {
            direct_conv::weight_read_fprop<TC>(*this, weight_lds);
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
__device__ void ck_tile_conv2d_grouped_4c_fp16_nhwc_impl(const _Float16* __restrict__ in,
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
        in, wei, out, N, groups, c_per_group, k_per_group, hi, wi, ho, wo, py, px);
}

template <Config cfg>
__global__ void ck_tile_conv2d_grouped_4c_fp16_nhwc(const _Float16* __restrict__ in,
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
    ck_tile_conv2d_grouped_4c_fp16_nhwc_impl<cfg>(in, wei, alpha, beta, out,
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
        ck_tile_conv2d_grouped_4c_fp16_nhwc<configs[I]>
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

static bool channels_can_be_padded(const Conv2dParams& par)
{
    // Check if we can pad the input tensor in the channel dimensions such that we K=C=4 per conv group.
    int c_per_group = par.c_tot / par.groups;
    int k_per_group = par.k_tot / par.groups;
    bool can_pad = c_per_group <= 4 && k_per_group <= 4;
    return can_pad;
}

static bool is_applicable(const Conv2dParams& par)
{
    if(!is_applicable_base(par))
        return false;
    if(!channels_can_be_padded(par))
        return false;
    return true;
}

constexpr KernelVariant make_variant()
{
    return {
        .is_applicable = &is_applicable,
        .config_is_compatible = [](const Conv2dParams& par, int idx)
        { return is_valid_config(par, configs[idx]); },
        .get_launch_params  = &get_launch_params,
        .launch             = &launch,
        .get_workspace_size = [](int, const Conv2dParams&) -> size_t { return 0; },
        .num_configs        = NUM_CONFIGS,
    };
}

} // namespace ck_tile::direct_conv::grouped_4c_tile::v3
