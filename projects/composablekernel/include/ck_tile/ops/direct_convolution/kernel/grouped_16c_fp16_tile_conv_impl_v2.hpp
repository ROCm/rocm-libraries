// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/direct_convolution/utils/transpose_lds_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
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

// Block output is 16 columns wide (fixed by mfma_f32_16x16x16f16 M=16).
constexpr int BLOCK_Q = 16;

// Kernel configuration parameters.
struct Config
{
    int waves_per_wg;

    int kh = 3;
    int kw = 3;

    int group_size = 16;

    int n_fold = 8;

    Direction direction = Direction::Fprop;

    SwizzleType swizzle_type = SwizzleType::None;

    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;

    constexpr int block_c() const { return group_size * waves_per_wg; }

    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    constexpr int block_groups() const { return waves_per_wg; }

    std::string GetName() const
    {
        std::string swz = (swizzle_type == SwizzleType::XOR) ? "swizzleXOR" : "noswizzle";
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
    static constexpr int GROUP_SIZE   = cfg.group_size;   // 16
    static constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4;   // 4
    static constexpr int GROUP_SIZE_8 = GROUP_SIZE / 8;   // 2

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
    static constexpr int NUM_INPUT_LDS_BUFFERS = 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C4 = INPUT_LDS_BUFFER_SIZE_C8 * 2;

    // Tile-level async load constants.
    static constexpr int NUM_WAVES     = cfg.waves_per_wg;
    static constexpr int LANES_PER_ROW = WAVE_SIZE / BLOCK_C8;
    static constexpr int TOTAL_SPATIAL  = cfg.block_size() / BLOCK_C8;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C8   = BLOCK_C8 * TOTAL_SPATIAL;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C4   = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_FP16 = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 8;

    static constexpr int KH_KW = cfg.kh * cfg.kw;

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
        static constexpr auto MakeDistribution()
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
    //   LDS:   4D [1, TOTAL_SPATIAL, BLOCK_C8, 8] in fp16 — staging buffer.
    //   Regs:  3D [BLOCK_W, BLOCK_C4, 4] in fp16 — MFMA A operand.
    // -----------------------------------------------------------------------
    struct Input
    {
        // DRAM descriptor: [hi, wi_padded, BLOCK_C8, 8] with pad on W.
        static CK_TILE_DEVICE auto MakeDramDescriptor(int hi, int wi, int C_total, int px)
        {
            constexpr int right_pad_w = cfg.kw - 1;

            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(hi, wi, ck_tile::number<BLOCK_C8>{}, ck_tile::number<8>{}),
                ck_tile::make_tuple(wi * C_total, C_total, ck_tile::number<8>{}, ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});

            const auto desc_padded = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(ck_tile::make_pass_through_transform(hi),
                                    ck_tile::make_pad_transform(wi, px, right_pad_w),
                                    ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_C8>{}),
                                    ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));

            if constexpr(cfg.swizzle_type == SwizzleType::XOR)
            {
                return ck_tile::transform_tensor_descriptor(
                    desc_padded,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(hi),
                        ck_tile::make_xor_transform(ck_tile::make_tuple(
                            wi + px + right_pad_w, ck_tile::number<BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                        ck_tile::sequence<3>{}));
            }
            else
            {
                return desc_padded;
            }
        }

        // Tile distribution for DRAM async loads.
        // Maps (P0=warp_id, P1=lane_id) to 4D (row, x_local, c8_local, sub).
        static constexpr auto MakeDramDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<1>,
                                   ck_tile::sequence<NUM_WAVES, LANES_PER_ROW>,
                                   ck_tile::sequence<BLOCK_C8>,
                                   ck_tile::sequence<8>>,
                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 3>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<1, 4>,
                    ck_tile::sequence<0, 0>>{});
        }

        // LDS store descriptor: [1, TOTAL_SPATIAL, BLOCK_C8, 8] contiguous.
        static constexpr auto MakeLdsStoreDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<TOTAL_SPATIAL>{},
                                    ck_tile::number<BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<TOTAL_SPATIAL * BLOCK_C8 * 8>{},
                                    ck_tile::number<BLOCK_C8 * 8>{},
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        }

        // LDS read descriptor: [BLOCK_W, BLOCK_C4, 4] for MFMA register reads.
        static constexpr auto MakeLdsReadDescriptor()
        {
            constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<BLOCK_W>{},
                                    ck_tile::number<BLOCK_C8>{},
                                    ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<BLOCK_C8 * 8>{},
                                    ck_tile::number<8>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});

            auto make_desc = [](auto desc_3d) constexpr {
                constexpr auto desc_merged = ck_tile::transform_tensor_descriptor(
                    desc_3d,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_W>{}),
                        ck_tile::make_merge_transform(
                            ck_tile::make_tuple(ck_tile::number<BLOCK_C8>{}, ck_tile::number<8>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

                return ck_tile::transform_tensor_descriptor(
                    desc_merged,
                    ck_tile::make_tuple(
                        ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_W>{}),
                        ck_tile::make_unmerge_transform(
                            ck_tile::make_tuple(ck_tile::number<BLOCK_C4>{}, ck_tile::number<4>{}))),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}));
            };

            if constexpr(cfg.swizzle_type == SwizzleType::XOR)
            {
                constexpr auto desc_xor = ck_tile::transform_tensor_descriptor(
                    desc_raw,
                    ck_tile::make_tuple(
                        ck_tile::make_xor_transform(
                            ck_tile::make_tuple(ck_tile::number<BLOCK_W>{}, ck_tile::number<BLOCK_C8>{})),
                        ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                    ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}));
                return make_desc(desc_xor);
            }
            else
            {
                return make_desc(desc_raw);
            }
        }
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
        // Weight LDS staging: [block_c][kh*kw][GROUP_SIZE] in fp16 units.
        // In uint2 (4 fp16): block_c * kh * kw * GROUP_SIZE / 4 = block_c * kh*kw * GROUP_SIZE_4.
        static constexpr int WEIGHT_LDS_SIZE_UINT2 =
            cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE * GROUP_SIZE_4;
        static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_SIZE_UINT2 / 2;

        static constexpr int NUM_WEIGHT_PASSES =
            (WEIGHT_LDS_SIZE_UINT4 + cfg.block_size() - 1) / cfg.block_size();

        static constexpr int WEIGHT_LDS_PADDED_UINT4 = NUM_WEIGHT_PASSES * cfg.block_size();

        static constexpr int WEIGHT_LDS_READ_K = cfg.block_c();

        // DRAM descriptor: 2D [WEIGHT_LDS_SIZE_UINT4, 8] padded to
        // [WEIGHT_LDS_PADDED_UINT4, 8].
        static constexpr auto MakeDramDescriptor()
        {
            constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<WEIGHT_LDS_SIZE_UINT4>{}, ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<8>{}, ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});

            constexpr int right_pad = WEIGHT_LDS_PADDED_UINT4 - WEIGHT_LDS_SIZE_UINT4;
            constexpr auto desc_padded = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(
                    ck_tile::make_pad_transform(ck_tile::number<WEIGHT_LDS_SIZE_UINT4>{}, 0, right_pad),
                    ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
            return desc_padded;
        }

        // Tile distribution for weight async loads: linear tid → row.
        static constexpr auto MakeDramDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<NUM_WAVES, WAVE_SIZE>,
                                   ck_tile::sequence<8>>,
                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1>>,
                    ck_tile::sequence<2>,
                    ck_tile::sequence<0>>{});
        }

        // LDS store descriptor: [WEIGHT_LDS_PADDED_UINT4, 8] contiguous.
        static constexpr auto MakeLdsStoreDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<WEIGHT_LDS_PADDED_UINT4>{}, ck_tile::number<8>{}),
                ck_tile::make_tuple(ck_tile::number<8>{}, ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        }

        // LDS read descriptor (Fprop): 3D [block_c, kh*kw, GROUP_SIZE] row-major.
        static constexpr auto MakeLdsReadDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<WEIGHT_LDS_READ_K>{},
                                    ck_tile::number<KH_KW>{},
                                    ck_tile::number<GROUP_SIZE>{}),
                ck_tile::make_tuple(ck_tile::number<KH_KW * GROUP_SIZE>{},
                                    ck_tile::number<GROUP_SIZE>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});
        }

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
        static constexpr auto MakeLdsReadDistribution()
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
        static constexpr int OUTPUT_LDS_BUFFER_SIZE = BLOCK_C8 * BLOCK_Q;

        // LDS write descriptor: [BLOCK_Q, BLOCK_C4, 4] row-major.
        static constexpr auto MakeLdsWriteDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<BLOCK_Q>{},
                                    ck_tile::number<BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                ck_tile::make_tuple(ck_tile::number<BLOCK_C4 * 4>{},
                                    ck_tile::number<4>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});
        }

        // LDS read descriptor: [1, BLOCK_Q, BLOCK_C4, 4] row-major.
        static constexpr auto MakeLdsReadDescriptor()
        {
            return ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<BLOCK_Q>{},
                                    ck_tile::number<BLOCK_C4>{},
                                    ck_tile::number<4>{}),
                ck_tile::make_tuple(ck_tile::number<BLOCK_Q * BLOCK_C4 * 4>{},
                                    ck_tile::number<BLOCK_C4 * 4>{},
                                    ck_tile::number<4>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});
        }

        // DRAM descriptor: [ho, wo_padded, BLOCK_C4, 4].
        static CK_TILE_DEVICE auto MakeDramDescriptor(int ho, int wo, int C)
        {
            const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ho, wo, ck_tile::number<BLOCK_C4>{}, ck_tile::number<4>{}),
                ck_tile::make_tuple(wo * C, C, ck_tile::number<4>{}, ck_tile::number<1>{}),
                ck_tile::number<4>{},
                ck_tile::number<1>{});

            constexpr int right_pad_w = BLOCK_Q;
            const auto desc_padded = ck_tile::transform_tensor_descriptor(
                desc_raw,
                ck_tile::make_tuple(ck_tile::make_pass_through_transform(ho),
                                    ck_tile::make_pad_transform(wo, 0, right_pad_w),
                                    ck_tile::make_pass_through_transform(ck_tile::number<BLOCK_C4>{}),
                                    ck_tile::make_pass_through_transform(ck_tile::number<4>{})),
                ck_tile::make_tuple(
                    ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}),
                ck_tile::make_tuple(
                    ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                    ck_tile::sequence<2>{}, ck_tile::sequence<3>{}));

            return desc_padded;
        }

        // Tile distribution for DRAM output writes.
        // 4D: [1, BLOCK_Q, BLOCK_C4, 4]
        //   X0 = 1 (row) → Y0
        //   X1 = 16 (Q) → P1
        //   X2 = BLOCK_C4 [waves_per_wg, 4] → P0, P1
        //   X3 = 4 → Y1
        static constexpr auto MakeDramDistribution()
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
        : C(groups * cfg.group_size), C8(C / 8), K(C)
    {
        const int block_q_n_idx = blockIdx.x;
        block_n     = static_cast<int>(blockIdx.z) * cfg.n_fold + block_q_n_idx % cfg.n_fold;
        block_q     = (block_q_n_idx / cfg.n_fold) * BLOCK_Q;
        block_group = static_cast<int>(blockIdx.y) * cfg.block_groups();
        block_k     = block_group * cfg.group_size;
        block_c8    = block_k / 8;
    }
};

// ===================================================================
// InputLoader — DRAM→LDS async load, double-buffered, MFMA reads.
// ===================================================================
template <Config cfg>
struct InputLoader
{
    using TC = TileConstants<cfg>;

    using InputDramWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<const _Float16*>(nullptr),
            TC::Input::MakeDramDescriptor(int{}, int{}, int{}, int{})),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{},
        TC::Input::MakeDramDistribution()));

    using LdsWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            static_cast<_Float16*>(nullptr),
            TC::Input::MakeLdsStoreDescriptor()),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{}));

    static constexpr auto mfma_desc = TC::Input::MakeLdsReadDescriptor();
    static constexpr auto mfma_dist = TC::Mfma::MakeDistribution();

    using MfmaBuf      = ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>;
    using MfmaViewType = ck_tile::tensor_view<MfmaBuf, ck_tile::remove_cvref_t<decltype(mfma_desc)>>;

    using MfmaWindowType = decltype(ck_tile::make_tile_window(
        MfmaViewType{},
        ck_tile::make_tuple(ck_tile::number<BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0},
        mfma_dist));

    InputDramWindowType input_dram_window;
    LdsWindowType       lds_window_0;
    LdsWindowType       lds_window_1;
    MfmaWindowType      mfma_window_0;
    MfmaWindowType      mfma_window_1;
    uint4* input_lds_ptr;

    __device__ InputLoader(const BlockCoords<cfg>& bc,
                           uint4* input_lds,
                           const _Float16* __restrict__ in,
                           int hi,
                           int wi,
                           int px)
                : input_lds_ptr(input_lds)
    {
        const auto input_dram_desc = TC::Input::MakeDramDescriptor(hi, wi, bc.C, px);
        const auto input_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C + bc.block_k,
            input_dram_desc);

        constexpr auto input_dram_dist = TC::Input::MakeDramDistribution();
        input_dram_window = ck_tile::make_tile_window(
            input_dram_view,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, bc.block_q, 0, 0},
            input_dram_dist);

        constexpr auto lds_store_desc = TC::Input::MakeLdsStoreDescriptor();
        auto lds_view_0 = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(&input_lds[0]), lds_store_desc);
        auto lds_view_1 = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(&input_lds[TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8]),
            lds_store_desc);
        lds_window_0 = ck_tile::make_tile_window(
            lds_view_0,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, 0, 0, 0});
        lds_window_1 = ck_tile::make_tile_window(
            lds_view_1,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, 0, 0, 0});

        auto mfma_buf_0 = MfmaBuf{
            reinterpret_cast<_Float16*>(input_lds_ptr),
            static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16)};
        auto mfma_view_0 = MfmaViewType{mfma_buf_0, mfma_desc};
        mfma_window_0 = ck_tile::make_tile_window(
            mfma_view_0,
            ck_tile::make_tuple(ck_tile::number<BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0},
            mfma_dist);

        auto mfma_buf_1 = MfmaBuf{
            reinterpret_cast<_Float16*>(input_lds_ptr) + TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16,
            static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16)};
        auto mfma_view_1 = MfmaViewType{mfma_buf_1, mfma_desc};
        mfma_window_1 = ck_tile::make_tile_window(
            mfma_view_1,
            ck_tile::make_tuple(ck_tile::number<BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0},
            mfma_dist);
    }

    __device__ void fetch_tile_to_lds(int lds_buffer_index)
    {
        ck_tile::move_tile_window(input_dram_window, {1, 0, 0, 0});
        if(lds_buffer_index == 0)
            ck_tile::async_load_tile(lds_window_0, input_dram_window);
        else
            ck_tile::async_load_tile(lds_window_1, input_dram_window);
    }

    __device__ void prefetch_tile_to_lds(int lds_buffer_index)
    {
        if(lds_buffer_index == 0)
            ck_tile::async_load_tile(lds_window_0, input_dram_window);
        else
            ck_tile::async_load_tile(lds_window_1, input_dram_window);
    }

    __device__ void read_from_lds(ck_tile::fp16x4_t& input_reg, int slice, int lds_buffer_index)
    {
        auto& window = (lds_buffer_index == 0) ? mfma_window_0 : mfma_window_1;
        auto tile = ck_tile::load_tile(window);
        __builtin_memcpy(&input_reg, &tile.get_thread_buffer()(ck_tile::number<0>{}), sizeof(ck_tile::fp16x4_t));
        if(slice < cfg.kw - 1)
        {
            ck_tile::move_tile_window(window, {1, 0, 0});
        }
        else
        {
            ck_tile::move_tile_window(window, {-(cfg.kw - 1), 0, 0});
        }
    }
};

// ===================================================================
// WeightLoader — async weight loads to LDS, then register reads.
// ===================================================================
template <Config cfg>
struct WeightLoader
{
    using TC = TileConstants<cfg>;

    __device__ static void load_to_lds(const BlockCoords<cfg>& bc,
                                       uint4* weight_lds,
                                       const _Float16* __restrict__ wei)
    {
        constexpr auto weight_dram_desc = TC::Weight::MakeDramDescriptor();
        auto weight_dram_buf = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(
            wei + static_cast<size_t>(bc.block_k) * cfg.kh * cfg.kw * TC::GROUP_SIZE,
            static_cast<ck_tile::index_t>(weight_dram_desc.get_element_space_size()));
        auto weight_dram_view =
            ck_tile::tensor_view<remove_cvref_t<decltype(weight_dram_buf)>,
                                remove_cvref_t<decltype(weight_dram_desc)>>{
                weight_dram_buf, weight_dram_desc};

        constexpr auto weight_dram_dist = TC::Weight::MakeDramDistribution();
        auto weight_dram_window = ck_tile::make_tile_window(
            weight_dram_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0},
            weight_dram_dist);

        constexpr auto weight_lds_desc = TC::Weight::MakeLdsStoreDescriptor();
        auto weight_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(weight_lds), weight_lds_desc);
        auto weight_lds_window = ck_tile::make_tile_window(
            weight_lds_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0});

        static_for<TC::Weight::NUM_WEIGHT_PASSES>(
            [&]<int Pass>()
            {
                ck_tile::async_load_tile(weight_lds_window, weight_dram_window);
                if constexpr(Pass < TC::Weight::NUM_WEIGHT_PASSES - 1)
                {
                    ck_tile::move_tile_window(weight_dram_window, {cfg.block_size(), 0});
                    ck_tile::move_tile_window(weight_lds_window, {cfg.block_size(), 0});
                }
            });
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
                static_cast<ck_tile::index_t>(TC::Weight::WEIGHT_LDS_PADDED_UINT4 *
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

            constexpr auto weight_lds_read_dist = TC::Weight::MakeLdsReadDistribution();
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
struct OutputWriter
{
    using TC = TileConstants<cfg>;

    static constexpr auto OutputDramDist = TC::Output::MakeDramDistribution();
    using OutputDstrTensor =
        ck_tile::static_distributed_tensor<_Float16, ck_tile::remove_cvref_t<decltype(OutputDramDist)>>;

    using OutputDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::Output::MakeDramDescriptor(int{}, int{}, int{}))>;
    using OutputDramBuf =
        ck_tile::buffer_view<ck_tile::address_space_enum::global, _Float16, ck_tile::index_t, true>;
    using OutputDramView = ck_tile::tensor_view<OutputDramBuf, OutputDramDesc>;
    using OutputDramWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputDramView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        OutputDramDist))>;

    OutputDramWindow dram_window;
    int last_p_out;

    __device__ OutputWriter(const BlockCoords<cfg>& bc,
                            uint4*,
                            _Float16* __restrict__ out,
                            int ho,
                            int wo)
        : last_p_out(0)
    {
        constexpr auto out_dist = TC::Output::MakeDramDistribution();
        const auto out_desc = TC::Output::MakeDramDescriptor(ho, wo, bc.C);
        auto out_buf = OutputDramBuf{
            out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k,
            static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
        auto out_view = OutputDramView{out_buf, out_desc};

        dram_window = ck_tile::make_tile_window(
            out_view,
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, bc.block_q, 0, 0},
            out_dist);
    }

    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        const auto* fp16_ptr = reinterpret_cast<const _Float16*>(halves);

        OutputDstrTensor output_tile;
        output_tile.get_thread_buffer()(ck_tile::number<0>{}) = fp16_ptr[0];
        output_tile.get_thread_buffer()(ck_tile::number<1>{}) = fp16_ptr[1];
        output_tile.get_thread_buffer()(ck_tile::number<2>{}) = fp16_ptr[2];
        output_tile.get_thread_buffer()(ck_tile::number<3>{}) = fp16_ptr[3];

        ck_tile::move_tile_window(dram_window, {p_out - last_p_out, 0, 0, 0});
        last_p_out = p_out;

        ck_tile::store_tile(dram_window, output_tile);
    }
};

// ===================================================================
// OutputWriterLds — LDS-staged writes (RegistersToLdsToGlobalMemory).
// ===================================================================
template <Config cfg>
struct OutputWriterLds
{
    using TC = TileConstants<cfg>;

    static constexpr auto OutputLdsDist  = TC::Mfma::MakeDistribution();
    static constexpr auto OutputDramDist = TC::Output::MakeDramDistribution();

    using OutputDstrTensor =
        ck_tile::static_distributed_tensor<_Float16, ck_tile::remove_cvref_t<decltype(OutputLdsDist)>>;

    using OutputLdsBuf = ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>;

    using OutputLdsWriteDesc   = ck_tile::remove_cvref_t<decltype(TC::Output::MakeLdsWriteDescriptor())>;
    using OutputLdsWriteView   = ck_tile::tensor_view<OutputLdsBuf, OutputLdsWriteDesc>;
    using OutputLdsWriteWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputLdsWriteView{},
        ck_tile::make_tuple(ck_tile::number<BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0},
        OutputLdsDist))>;

    using OutputLdsReadDesc   = ck_tile::remove_cvref_t<decltype(TC::Output::MakeLdsReadDescriptor())>;
    using OutputLdsReadView   = ck_tile::tensor_view<OutputLdsBuf, OutputLdsReadDesc>;
    using OutputLdsReadWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputLdsReadView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        OutputDramDist))>;

    using OutputDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::Output::MakeDramDescriptor(int{}, int{}, int{}))>;
    using OutputDramBuf =
        ck_tile::buffer_view<ck_tile::address_space_enum::global, _Float16, ck_tile::index_t, true>;
    using OutputDramView = ck_tile::tensor_view<OutputDramBuf, OutputDramDesc>;
    using OutputDramWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputDramView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<BLOCK_Q>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        OutputDramDist))>;

    OutputLdsWriteWindow lds_write_window;
    OutputLdsReadWindow  lds_read_window;
    OutputDramWindow     dram_window;
    int last_p_out;

    __device__ OutputWriterLds(const BlockCoords<cfg>& bc,
                               uint4* output_lds,
                               _Float16* __restrict__ out,
                               int ho,
                               int wo)
        : last_p_out(0)
    {
        auto lds_buf = OutputLdsBuf{
            reinterpret_cast<_Float16*>(output_lds),
            static_cast<ck_tile::index_t>(
                ck_tile::max(TC::Weight::WEIGHT_LDS_PADDED_UINT4, TC::Output::OUTPUT_LDS_BUFFER_SIZE) *
                (sizeof(uint4) / sizeof(_Float16)))};

        constexpr auto lds_write_desc = TC::Output::MakeLdsWriteDescriptor();
        constexpr auto lds_write_dist = TC::Mfma::MakeDistribution();
        auto lds_write_view = OutputLdsWriteView{lds_buf, lds_write_desc};
        lds_write_window = ck_tile::make_tile_window(
            lds_write_view,
            ck_tile::make_tuple(ck_tile::number<BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0},
            lds_write_dist);

        constexpr auto lds_read_desc = TC::Output::MakeLdsReadDescriptor();
        constexpr auto lds_read_dist = TC::Output::MakeDramDistribution();
        auto lds_read_view = OutputLdsReadView{lds_buf, lds_read_desc};
        lds_read_window = ck_tile::make_tile_window(
            lds_read_view,
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0, 0},
            lds_read_dist);

        constexpr auto out_dist = TC::Output::MakeDramDistribution();
        const auto out_desc = TC::Output::MakeDramDescriptor(ho, wo, bc.C);
        auto out_buf = OutputDramBuf{
            out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k,
            static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
        auto out_view = OutputDramView{out_buf, out_desc};

        dram_window = ck_tile::make_tile_window(
            out_view,
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<BLOCK_Q>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, bc.block_q, 0, 0},
            out_dist);
    }

    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        const auto* fp16_ptr = reinterpret_cast<const _Float16*>(halves);

        OutputDstrTensor output_tile;
        output_tile.get_thread_buffer()(ck_tile::number<0>{}) = fp16_ptr[0];
        output_tile.get_thread_buffer()(ck_tile::number<1>{}) = fp16_ptr[1];
        output_tile.get_thread_buffer()(ck_tile::number<2>{}) = fp16_ptr[2];
        output_tile.get_thread_buffer()(ck_tile::number<3>{}) = fp16_ptr[3];

        ck_tile::store_tile(lds_write_window, output_tile);

        ck_tile::s_waitcnt_lgkm<0>();

        const auto lds_tile = ck_tile::load_tile(lds_read_window);

        ck_tile::move_tile_window(dram_window, {p_out - last_p_out, 0, 0, 0});
        last_p_out = p_out;

        ck_tile::store_tile(dram_window, lds_tile);
    }
};

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

    __shared__ uint4 input_lds[TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8];
    static constexpr int OUTPUT_LDS_SIZE = use_lds_epilogue
                                               ? ck_tile::max(TC::Weight::WEIGHT_LDS_PADDED_UINT4,
                                                              TC::Output::OUTPUT_LDS_BUFFER_SIZE)
                                               : TC::Weight::WEIGHT_LDS_PADDED_UINT4;
    __shared__ uint4 output_lds[OUTPUT_LDS_SIZE];

    BlockCoords<cfg> bc(groups);
    if(bc.block_n >= N)
        return;

    InputLoader<cfg> il(bc, input_lds, in, hi, wi, px);
    OutputWriterType ow(bc, output_lds, out, ho, wo);

    fp16x4_t weights_reg[cfg.kh * cfg.kw];
    WeightLoader<cfg>::load_to_lds(bc, output_lds, wei);
    wait_vmcnt<0>();
    __syncthreads();

    WeightLoader<cfg>::read_from_lds(weights_reg, output_lds);
    __syncthreads();

    // Prefetch first input row into LDS buffer 0.
    il.prefetch_tile_to_lds(0);
    wait_vmcnt<0>();
    __syncthreads();

    // Circular accumulator buffer.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    // Main loop: iterate over input rows.
    for(int y_base = 0; y_base + cfg.kh <= hi; y_base += cfg.kh)
    {
        static_for<cfg.kh>(
            [&]<int Y_LOCAL>()
            {
                wait_vmcnt<0>();
                __syncthreads();

                int y = y_base + Y_LOCAL;
                if((y + 1) < hi)
                {
                    il.fetch_tile_to_lds(tic);
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        ck_tile::fp16x4_t input_reg;
                        il.read_from_lds(input_reg, S, toc);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(cfg.direction == Direction::Dgrad)
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                                        weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                                        input_reg,
                                        acc[p_idx],
                                        0, 0, 0);
                                else
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                                        weights_reg[R * cfg.kw + S],
                                        input_reg,
                                        acc[p_idx],
                                        0, 0, 0);
                            });
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_FLUSH], p_out);
                acc[P_FLUSH] = Zero;
            });
    }

    // Remainder loop: hi % kh leftover rows.
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

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        ck_tile::fp16x4_t input_reg;
                        il.read_from_lds(input_reg, S, toc);

                        static_for<cfg.kh>(
                            [&]<int R>()
                            {
                                constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                                if constexpr(cfg.direction == Direction::Dgrad)
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                                        weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                                        input_reg,
                                        acc[p_idx],
                                        0, 0, 0);
                                else
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                                        weights_reg[R * cfg.kw + S],
                                        input_reg,
                                        acc[p_idx],
                                        0, 0, 0);
                            });
                    });

                tic ^= 1;
                toc ^= 1;

                constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
                int p_out = y + py - (cfg.kh - 1);
                if(p_out >= 0 && p_out < ho)
                    ow.flush(acc[P_FLUSH], p_out);
                acc[P_FLUSH] = Zero;
            });
    }

    // Tail flush: output rows not flushed by the main/remainder loops.
    for(int p_out = hi - cfg.kh + 1 + py; p_out < ho; p_out++)
    {
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
