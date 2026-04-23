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

    Direction direction = Direction::Fprop;

    // Swizzle pattern - by default no explicit swizzle.
    SwizzleType swizzle_type = SwizzleType::None;

    EpilogueType epilogue = EpilogueType::RegistersToGlobalMemory;

    // Total number of waves.
    constexpr int num_waves() const { return waves_c64 * waves_q4; }

    // Tile size in the channel dimension: number of input channels processed by one workgroup.
    constexpr int block_c() const { return waves_c64 * 64; }

    // Tile size in the output column dimension.
    constexpr int block_q() const { return waves_q4 * 4; }

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

    // Number of uint4 vectors to store per output row (LDS epilogue path).
    static constexpr int STORE_VECS = cfg.block_q() * BLOCK_C8;

    // LDS double buffering for input loads.
    static constexpr int NUM_INPUT_LDS_BUFFERS    = 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C4 = INPUT_LDS_BUFFER_SIZE_C8 * 2;
    // Output LDS buffer size for LDS epilogue path.
    static constexpr int OUTPUT_LDS_BUFFER_SIZE   = BLOCK_C8 * cfg.block_q();

    // Weight LDS staging: [kh*kw][block_groups][GROUP_SIZE] in uint2 units.
    static constexpr int WEIGHT_LDS_SIZE_UINT2 = cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE;
    static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_SIZE_UINT2 / 2;

    // -----------------------------------------------------------------------
    // LDS descriptor (uint4 units, for MFMA read path).
    //
    // Logical layout: [BLOCK_W, BLOCK_C8] row-major (x = row, c8 = column)
    // Physical layout depends on swizzle_type:
    //   None: plain row-major, offset = x * BLOCK_C8 + c8.
    //   XOR:  xor_t transform, offset = x * BLOCK_C8 + (c8 ^ (x % BLOCK_C8)).
    // -----------------------------------------------------------------------
    static constexpr auto MakeInputLdsBlockDescriptor()
    {
        constexpr auto desc_naive = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<BLOCK_W>{}, ck_tile::number<BLOCK_C8>{}),
            ck_tile::make_tuple(ck_tile::number<BLOCK_C8>{}, ck_tile::number<1>{}));

        if constexpr(cfg.swizzle_type == SwizzleType::XOR)
        {
            return ck_tile::transform_tensor_descriptor(
                desc_naive,
                ck_tile::make_tuple(ck_tile::make_xor_transform(
                    ck_tile::make_tuple(ck_tile::number<BLOCK_W>{}, ck_tile::number<BLOCK_C8>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}));
        }
        else
        {
            return desc_naive;
        }
    }

    // Total channels per block in fp16 elements.
    static constexpr int BLOCK_C = BLOCK_C8 * 8;

    // fp16x4 groups per channel fiber (4 fp16 per group = one MFMA result vector).
    static constexpr int BLOCK_C4 = BLOCK_C / 4;

    // -----------------------------------------------------------------------
    // Global input descriptor: 3D [hi, wi_padded, C] in fp16 elements.
    //
    // Construction chain:
    //   1. Naive [hi, wi, C] with strides [wi*C, C, 1]  (fp16 units)
    //   2. Pad W dimension: [hi, wi + px + (kw-1), C]
    //      - Left pad = px (convolution padding)
    //      - Right pad = kw - 1 (halo past right edge)
    //      - OOB accesses in padded region automatically return zero
    //
    // Used by InputLoader for per-thread async loads with automatic
    // OOB handling via the pad transform.
    // -----------------------------------------------------------------------
    static CK_TILE_DEVICE auto MakeInputGlobalDescriptor(int hi, int wi, int C, int px)
    {
        constexpr int right_pad_w = cfg.kw - 1;

        // C is always a multiple of 8 fp16 (= one uint4).
        // GuaranteedLastDimensionVectorLength = 8 enables 128-bit vectorized loads.
        const auto desc_hwc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(hi, wi, C),
            ck_tile::make_tuple(wi * C, C, ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});

        const auto desc_padded = ck_tile::transform_tensor_descriptor(
            desc_hwc,
            ck_tile::make_tuple(ck_tile::make_pass_through_transform(hi),
                                ck_tile::make_pad_transform(wi, px, right_pad_w),
                                ck_tile::make_pass_through_transform(C)),
            ck_tile::make_tuple(
                ck_tile::sequence<0>{}, ck_tile::sequence<1>{}, ck_tile::sequence<2>{}),
            ck_tile::make_tuple(
                ck_tile::sequence<0>{}, ck_tile::sequence<1>{}, ck_tile::sequence<2>{}));

        return desc_padded;
    }

    // -----------------------------------------------------------------------
    // Tile-level async load constants and descriptors.
    //
    // The tile distribution maps (warp_id, lane_id) to a 4D tile coordinate
    // (row, x_local, c8_local, c_sub) where x_local ∈ [0, TOTAL_SPATIAL),
    // c8_local ∈ [0, BLOCK_C8), c_sub ∈ [0, 8).  TOTAL_SPATIAL > BLOCK_W
    // in general; surplus threads (x_local >= BLOCK_W) produce OOB
    // coordinates that the pad transform suppresses automatically.
    //
    // No swizzle: each thread loads from its natural (x_local, c8_local)
    // coordinate and writes to the corresponding row-major LDS position.
    //
    // Element type: _Float16 (fp16). The 8-element sub-channel dimension
    // is the vectorization dimension (Y), producing 128-bit loads.
    // -----------------------------------------------------------------------
    static constexpr int LANES_PER_ROW = WAVE_SIZE / BLOCK_C8;
    static constexpr int TOTAL_SPATIAL = cfg.block_size() / BLOCK_C8;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C8 = BLOCK_C8 * TOTAL_SPATIAL;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C4 = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 2;
    // Padded LDS buffer size in fp16 elements (each C8 group = 8 fp16).
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_FP16 = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 8;

    // DRAM descriptor for tile-level async loads: [hi, wi_padded, BLOCK_C8, 8]
    // in fp16 units, with pad transform on W.
    // Row dimension allows advancing rows via move_tile_window({1, 0, 0, 0}).
    //
    // Base pointer: in + batch_offset + block_k  (shifted to tile's channel origin).
    //
    // When swizzle_type == XOR, an xor_t transform is applied on dims (1, 2)
    // so that the DRAM read at tile coordinate (x_local, c8_local) fetches
    // channel c8_local ^ (x_local % BLOCK_C8) from global memory.  The LDS
    // store remains linear, so the data arrives in LDS in XOR-swizzled order
    // matching the MFMA read path (MakeInputLdsBlockDescriptor with XOR).
    static CK_TILE_DEVICE auto MakeInputDramDescriptor(int hi, int wi, int C_total, int px)
    {
        constexpr int right_pad_w = cfg.kw - 1;

        // Step 1: Naive [hi, wi, BLOCK_C8, 8] with global strides in fp16 units.
        const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(hi, wi, ck_tile::number<BLOCK_C8>{}, ck_tile::number<8>{}),
            ck_tile::make_tuple(wi * C_total, C_total, ck_tile::number<8>{}, ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});

        // Step 2: Pad W dimension: [hi, wi_padded, BLOCK_C8, 8].
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
            // Step 3: XOR transform on (spatial, channel) dims.
            // Maps upper (x, c8) to lower (x, c8 ^ (x % BLOCK_C8)).
            // This permutes the channel read so data lands in LDS in XOR order.
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

    // LDS store descriptor for async_load_tile: [1, TOTAL_SPATIAL, BLOCK_C8, 8]
    // in fp16 units, simple contiguous layout.
    // Matches the DRAM distribution's 4D X-space (row=1, spatial, channel, sub).
    static constexpr auto MakeInputLdsStoreDescriptor()
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

    // Tile distribution for DRAM async loads.
    // Maps (P0=warp_id, P1=lane_id) to 4D tile coordinate
    //   (row=0, x_local, c8_local, c_sub) where:
    //   x_local = warp_id * LANES_PER_ROW + lane_id / BLOCK_C8
    //   c8_local = lane_id % BLOCK_C8
    //   c_sub ∈ [0, 8) — Y-iteration dimension (vectorization)
    //
    // X0 = 1 (row), Y0 iteration (trivial length 1)
    // X1 = TOTAL_SPATIAL, decomposed as [NUM_WAVES, LANES_PER_ROW]:
    //   factor 0 (NUM_WAVES) = outer spatial (big strides), contributed by P0
    //   factor 1 (LANES_PER_ROW) = inner spatial (small strides), contributed by P1
    //   Unmerge: X1 = factor0 * LANES_PER_ROW + factor1 = warp_id * 4 + lane_id/16
    // X2 = BLOCK_C8, decomposed as [BLOCK_C8]:
    //   P1 contributes BLOCK_C8 factor
    // X3 = 8, decomposed as [8]:
    //   Y1 iteration dimension (128-bit vectorized loads)
    //
    // LDS constraint: global_load_lds writes lane L at M0 + L * 16 bytes.
    // LDS store descriptor is row-major [TOTAL_SPATIAL, BLOCK_C8, 8].
    // For correct placement: X1 must equal warp_id * LANES_PER_ROW + lane_id/BLOCK_C8
    // so that LDS offset = X1 * BLOCK_C8 * 8 + X2 * 8 = (warp*4 + lane/16)*128 + (lane%16)*8
    //                    = warp*512 + lane*8 = M0/elem_size + lane*vec_size.
    static constexpr int NUM_WAVES = cfg.num_waves();
    static constexpr auto MakeInputDramDistribution()
    {
        // RH-major indices: 0=R(empty), 1=X0(row), 2=X1(spatial), 3=X2(channel), 4=X3(sub)
        //
        // X1 unmerge factors: [NUM_WAVES, LANES_PER_ROW]
        //   factor 0 (NUM_WAVES, minor=0): outer spatial
        //   factor 1 (LANES_PER_ROW, minor=1): inner spatial
        //
        // P0 (warp_id, single factor of NUM_WAVES):
        //   maps to X1 factor 0 (NUM_WAVES) → major=2, minor=0
        // P1 (lane_id=64, decomposed as [LANES_PER_ROW, BLOCK_C8]):
        //   Merge low_lengths = {LANES_PER_ROW=4, BLOCK_C8=16}
        //   lane_id / 16 → factor 0 → X1 factor 1 (LANES_PER_ROW) → major=2, minor=1
        //   lane_id % 16 → factor 1 → X2 factor 0 (BLOCK_C8)      → major=3, minor=0
        // Y0 (length 1): maps to X0 H-factor 0 → major=1, minor=0 (trivial row)
        // Y1 (length 8): maps to X3 H-factor 0 → major=4, minor=0 (vectorization)
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

    // -----------------------------------------------------------------------
    // Weight tile-level async load descriptors.
    //
    // Weight data is a contiguous block of WEIGHT_LDS_SIZE_UINT4 uint4 values
    // (= WEIGHT_LDS_SIZE_UINT4 * 8 fp16). We treat it as 2D [rows, 8] in fp16
    // and map all block_size threads linearly: thread tid loads row tid.
    //
    // Rows 0..WEIGHT_LDS_SIZE_UINT4-1 contain valid data; the remaining rows
    // up to block_size-1 are OOB (suppressed by the pad transform on the
    // DRAM descriptor).
    // -----------------------------------------------------------------------

    // Number of async load passes needed to load all weights into LDS.
    // Each pass loads block_size uint4 values. When block_size >= WEIGHT_LDS_SIZE_UINT4,
    // a single pass suffices. Otherwise, multiple passes are needed.
    static constexpr int NUM_WEIGHT_PASSES =
        (WEIGHT_LDS_SIZE_UINT4 + cfg.block_size() - 1) / cfg.block_size();

    // Padded weight LDS size: must accommodate all weight data.
    // Rounded up to a multiple of block_size for uniform multi-pass loading.
    static constexpr int WEIGHT_LDS_PADDED_UINT4 = NUM_WEIGHT_PASSES * cfg.block_size();

    // DRAM descriptor: 2D [WEIGHT_LDS_SIZE_UINT4, 8] padded to
    // [WEIGHT_LDS_PADDED_UINT4, 8].
    // Base pointer: wei + block_k * kh * kw * GROUP_SIZE.
    static constexpr auto MakeWeightDramDescriptor()
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

    // LDS store descriptor: 2D [block_size, 8] contiguous row-major in fp16.
    // Each pass stores block_size rows. Multiple passes write to successive
    // regions in LDS by advancing the window origin.
    static constexpr auto MakeWeightLdsStoreDescriptor()
    {
        // LDS descriptor covers the full padded weight buffer so that
        // multi-pass window advancement stays in-bounds.
        return ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<WEIGHT_LDS_PADDED_UINT4>{}, ck_tile::number<8>{}),
            ck_tile::make_tuple(ck_tile::number<8>{}, ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});
    }

    // Tile distribution for weight async loads.
    // Maps (P0=warp_id, P1=lane_id) to 2D tile coordinate (row, sub):
    //   row = warp_id * WAVE_SIZE + lane_id = tid (linear mapping)
    //   sub ∈ [0, 8) — Y-iteration dimension (128-bit vectorized loads)
    //
    // X0 = block_size, decomposed as [NUM_WAVES, WAVE_SIZE]:
    //   factor 0 (NUM_WAVES, outer): P0 contributes
    //   factor 1 (WAVE_SIZE, inner): P1 contributes
    // X1 = 8 (vectorization, Y dimension)
    static constexpr auto MakeWeightDramDistribution()
    {
        // RH-major indices: 0=R(empty), 1=X0(row), 2=X1(sub)
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

    // -----------------------------------------------------------------------
    // Weight LDS read descriptors (LDS → registers for MFMA B operand).
    //
    // After load_to_lds, the weight data in LDS is a contiguous buffer
    // viewed as 3D [block_c, kh*kw, GROUP_SIZE] in fp16 where:
    //   block_c = waves_c64 * 64 = total K channels in the tile
    //   kh*kw = filter spatial positions
    //   GROUP_SIZE = 4 = C channels per group (fp16x4 vector)
    //
    // The MFMA B operand layout for mfma_f32_4x4x4f16 maps each lane to
    // a specific K channel:
    //   lane_k     = lane % 4
    //   lane_batch = (lane / 4) % 16
    //   filter_local = wave_c64*64 + lane_batch*4 + lane_k
    //
    // Each thread iterates over kh*kw filter positions, reading fp16x4 at each.
    // -----------------------------------------------------------------------

    // Flat K-channel count for the weight LDS read.
    static constexpr int WEIGHT_LDS_READ_K = cfg.block_c(); // waves_c64 * 64

    // LDS read descriptor: 3D [block_c, kh*kw, GROUP_SIZE] row-major in fp16.
    static constexpr auto MakeWeightLdsReadDescriptor()
    {
        return ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<WEIGHT_LDS_READ_K>{},
                                ck_tile::number<cfg.kh * cfg.kw>{},
                                ck_tile::number<GROUP_SIZE>{}),
            ck_tile::make_tuple(ck_tile::number<cfg.kh * cfg.kw * GROUP_SIZE>{},
                                ck_tile::number<GROUP_SIZE>{},
                                ck_tile::number<1>{}),
            ck_tile::number<GROUP_SIZE>{},
            ck_tile::number<1>{});
    }

    // Tile distribution for weight LDS reads (Fprop path).
    //
    // Maps (P0=warp_id, P1=lane_id) to 3D tile coordinate
    //   (filter_local, khw, c_sub) where:
    //   filter_local = wave_c64 * 64 + lane_batch * 4 + lane_k
    //   khw ∈ [0, kh*kw) — Y-iteration dimension (filter positions)
    //   c_sub ∈ [0, 4) — Y-vectorization dimension
    //
    // 3D tile: [block_c, kh*kw, GROUP_SIZE]
    //   X0 = block_c [waves_c64, 16, 4]:
    //     factor 0 (waves_c64, outer): wave's channel group → P0
    //     factor 1 (16, middle): batch within wave → P1
    //     factor 2 (4, inner): K channel within batch → P1
    //   X1 = kh*kw [kh*kw]: filter position → Y0 iteration
    //   X2 = GROUP_SIZE [4]: sub-channel → Y1 vectorization
    //
    // R = [waves_q4]: all Q-waves read the same weights (replication).
    static constexpr auto MakeWeightLdsReadDistribution()
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
                ck_tile::sequence<cfg.waves_q4>,
                ck_tile::tuple<ck_tile::sequence<cfg.waves_c64, 16, 4>,
                               ck_tile::sequence<cfg.kh * cfg.kw>,
                               ck_tile::sequence<GROUP_SIZE>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<1, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 2>>,
                ck_tile::sequence<2, 3>,
                ck_tile::sequence<0, 0>>{});
    }

    // -----------------------------------------------------------------------
    // Output LDS write descriptors (MFMA result → LDS staging).
    // Used by the LDS epilogue path (RegistersToLdsToGlobalMemory).
    //
    // Thread mapping (derived from MFMA result layout):
    //   P0 = warp_id, merge of {waves_q4, waves_c64}
    //   P1 = lane_id ∈ [0,64), merge of {16, 4}
    //   Y0 = 4 (vectorization, the 4 fp16 channels within one group)
    // -----------------------------------------------------------------------

    // LDS descriptor for output staging: [block_q, BLOCK_C4, 4] row-major in fp16.
    static constexpr auto MakeOutputLdsWriteDescriptor()
    {
        return ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<cfg.block_q()>{},
                                ck_tile::number<BLOCK_C4>{},
                                ck_tile::number<4>{}),
            ck_tile::make_tuple(ck_tile::number<BLOCK_C4 * 4>{},
                                ck_tile::number<4>{},
                                ck_tile::number<1>{}),
            ck_tile::number<4>{},
            ck_tile::number<1>{});
    }

    static constexpr auto MakeOutputLdsReadDescriptor()
    {
        // 4D view of the flat LDS buffer, matching the DRAM window shape.
        // The LDS write descriptor is 3D [block_q, BLOCK_C4, 4]; we add a trivial row dimension for the DRAM read.
        return ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<cfg.block_q()>{},
                                ck_tile::number<BLOCK_C4>{},
                                ck_tile::number<4>{}),
            ck_tile::make_tuple(ck_tile::number<cfg.block_q() * BLOCK_C4 * 4>{},
                                ck_tile::number<BLOCK_C4 * 4>{},
                                ck_tile::number<4>{},
                                ck_tile::number<1>{}),
            ck_tile::number<4>{},
            ck_tile::number<1>{});
    }

    // Tile distribution for output LDS writes (MFMA result scatter).
    static constexpr auto MakeMfmaOutputDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<cfg.waves_q4, 4>,
                               ck_tile::sequence<cfg.waves_c64, 16>,
                               ck_tile::sequence<4>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<2, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 1>>,
                ck_tile::sequence<3>,
                ck_tile::sequence<0>>{});
    }

    static constexpr auto MakeMfmaInputDistribution()
    {
        // same [block_q, BLOCK_C4, 4] mapping
        return MakeMfmaOutputDistribution(); 
    }

    // Descriptor for MFMA input reads: [BLOCK_W, BLOCK_C4, 4] in fp16.
    // Built from the raw [BLOCK_W, BLOCK_C8, 8] layout with XOR applied
    // on (x, c8) if cfg.swizzle_type == XOR.
    static constexpr auto MakeMfmaInputLdsReadDescriptor()
    {
        // Raw fp16 layout: [BLOCK_W, BLOCK_C8, 8]
        constexpr auto desc_raw = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<BLOCK_W>{},
                                ck_tile::number<BLOCK_C8>{},
                                ck_tile::number<8>{}),
            ck_tile::make_tuple(ck_tile::number<BLOCK_C8 * 8>{},
                                ck_tile::number<8>{},
                                ck_tile::number<1>{}),
            ck_tile::number<4>{},
            ck_tile::number<1>{});

        // Apply XOR on (x, c8), pass-through on fp16-sub dimension.
        auto make_desc = [](auto desc_3d) constexpr {
            // Merge {c8, 8} → c (size BLOCK_C), then unmerge → {c4, 4}.
            // Converts uint4-granularity to fp16x4-granularity.
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

    // Tile distribution for output LDS reads (LDS staging gather before DRAM store).
    static constexpr auto MakeOutputLdsReadDistribution()
    {
        // Same mapping as DRAM distribution but over 3D LDS tile [block_q, BLOCK_C4, 4],
        // i.e., but without the trivial row dimension.
        // See, MakeOutputDramDistribution().
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<cfg.waves_q4, 4>,
                            ck_tile::sequence<cfg.waves_c64, 16>,
                            ck_tile::sequence<4>>,
                ck_tile::tuple<ck_tile::sequence<2, 3>, ck_tile::sequence<3, 2>>,
                ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 1>>,
                ck_tile::sequence<4>,
                ck_tile::sequence<0>>{});
    }

    // -----------------------------------------------------------------------
    // Direct DRAM output descriptors (no LDS staging).
    //
    // The MFMA result is written directly to global memory via store_tile.
    // Descriptor: 4D [ho, wo_padded, BLOCK_C4, 4] in fp16 with pad on wo.
    // The pad transform handles edge blocks where block_q + Q >= wo by
    // suppressing OOB writes automatically.
    //
    // Distribution: same MFMA mapping as LDS write, with a trivial row dim
    // prepended. Window shape: [1, block_q, BLOCK_C4, 4].
    // -----------------------------------------------------------------------

    // DRAM descriptor for direct output writes.
    // Base pointer: out + batch_offset + block_k (shifted to tile's channel origin).
    static CK_TILE_DEVICE auto MakeOutputDramDescriptor(int ho, int wo, int C)
    {
        // Step 1: Naive [ho, wo, BLOCK_C4, 4] with strides [wo*C, C, 4, 1] in fp16.
        const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ho, wo, ck_tile::number<BLOCK_C4>{}, ck_tile::number<4>{}),
            ck_tile::make_tuple(wo * C, C, ck_tile::number<4>{}, ck_tile::number<1>{}),
            ck_tile::number<4>{},
            ck_tile::number<1>{});

        // Step 2: Pad wo → wo + block_q so any block_q start has room.
        // OOB positions (wo <= pos < wo + block_q) are suppressed by the pad transform.
        constexpr int right_pad_w = cfg.block_q();
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

    // Tile distribution for direct DRAM output writes.
    // Maps MFMA result coordinates to a 4D tile with a trivial row dimension
    // prepended (X0 = 1, for the single output row per flush call).
    //
    // 4D tile: [1, block_q, BLOCK_C4, 4]
    //   X0 = 1 (row, trivial)           — Y0 iteration (length 1)
    //   X1 = block_q [waves_q4, 4]      — Q spatial
    //   X2 = BLOCK_C4 [waves_c64, 16]   — C groups
    //   X3 = 4                           — C sub (vectorization, Y1)
    static constexpr auto MakeOutputDramDistribution()
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
                               ck_tile::sequence<cfg.waves_q4, 4>,
                               ck_tile::sequence<cfg.waves_c64, 16>,
                               ck_tile::sequence<4>>,
                ck_tile::tuple<ck_tile::sequence<2, 3>, ck_tile::sequence<3, 2>>,
                ck_tile::tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<1, 1>>,
                ck_tile::sequence<1, 4>,
                ck_tile::sequence<0, 0>>{});
    }
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
struct InputLoader
{
    using TC = TileConstants<cfg>;

    // Derive the DRAM window type from the factory functions with dummy args.
    using InputDramWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<const _Float16*>(nullptr),
            TC::MakeInputDramDescriptor(int{}, int{}, int{}, int{})),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{},
        TC::MakeInputDramDistribution()));

    // LDS window has no distribution — same descriptor, no distribution arg.
    using LdsWindowType = decltype(ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            static_cast<_Float16*>(nullptr),
            TC::MakeInputLdsStoreDescriptor()),
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        ck_tile::multi_index<4>{}));

    static constexpr auto mfma_desc = TC::MakeMfmaInputLdsReadDescriptor();
    static constexpr auto mfma_dist = TC::MakeMfmaInputDistribution();

    using MfmaBuf  = ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>;
    using MfmaViewType = ck_tile::tensor_view<MfmaBuf, ck_tile::remove_cvref_t<decltype(mfma_desc)>>;

    using MfmaWindowType = decltype(ck_tile::make_tile_window(
        MfmaViewType{},
        ck_tile::make_tuple(ck_tile::number<TC::BLOCK_W - cfg.kw + 1>{},  // = block_q
                                    ck_tile::number<TC::BLOCK_C4>{},
                                    ck_tile::number<4>{}),
        {0,0,0},
        mfma_dist));
    

    // Members
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
        const auto input_dram_desc = TC::MakeInputDramDescriptor(hi, wi, bc.C, px);
        const auto input_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
            in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C + bc.block_k,
            input_dram_desc);

        // DRAM tile window with distribution. Window = [1, TOTAL_SPATIAL, BLOCK_C8, 8].
        constexpr auto input_dram_dist = TC::MakeInputDramDistribution();
        input_dram_window         = ck_tile::make_tile_window(
            input_dram_view,
            ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                                ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
            {0, bc.block_q, 0, 0},
            input_dram_dist);

        // LDS tile windows for double-buffered store (no distribution needed).
        // [1, TOTAL_SPATIAL, BLOCK_C8, 8] in fp16 units.
        constexpr auto lds_store_desc = TC::MakeInputLdsStoreDescriptor();
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

        // Create MFMA windows for the two LDS buffers, to be used for register reads.
        auto mfma_buf_0 = MfmaBuf{
            reinterpret_cast<_Float16*>(input_lds_ptr),
            static_cast<ck_tile::index_t>(TC::INPUT_LDS_BUFFER_SIZE_PADDED_FP16)};
        auto mfma_view_0 = MfmaViewType{mfma_buf_0, mfma_desc};
        mfma_window_0 = ck_tile::make_tile_window(
            mfma_view_0,
            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_W - cfg.kw + 1>{},  // = block_q
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
            ck_tile::make_tuple(ck_tile::number<TC::BLOCK_W - cfg.kw + 1>{},  // = block_q
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


    // Read a given kw slice for this thread from LDS into registers.
    // Must be called for slice = 0, 1, ..., kw-1 in order.
    // Assumes the relevant tile is already loaded and synced in LDS.
    __device__ void read_from_lds(ck_tile::fp16x4_t &input_reg, int slice, int lds_buffer_index)
    {
        auto& window = (lds_buffer_index == 0) ? mfma_window_0 : mfma_window_1;
        auto tile  = ck_tile::load_tile(window);
        __builtin_memcpy(&input_reg, &tile.get_thread_buffer()(ck_tile::number<0>{}), sizeof(ck_tile::fp16x4_t));
        if (slice < cfg.kw - 1)
        {
            ck_tile::move_tile_window(window, {1, 0, 0});
        }
        else
        {
            // Reset to origin so the next row starts at slice 0.
            ck_tile::move_tile_window(window, {-(cfg.kw - 1), 0, 0});
        }
    }
};

// Handles weight reads from LDS into registers.
template <Config cfg>
struct WeightLoader
{
    using TC = TileConstants<cfg>;

    __device__ static void load_to_lds(const BlockCoords<cfg>& bc,
                                       uint4* weight_lds,
                                       const _Float16* __restrict__ wei)
    {
        constexpr auto weight_dram_desc = TC::MakeWeightDramDescriptor();
        auto weight_dram_buf            = ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(
            wei + static_cast<size_t>(bc.block_k) * cfg.kh * cfg.kw * TC::GROUP_SIZE,
            static_cast<ck_tile::index_t>(weight_dram_desc.get_element_space_size()));
        auto weight_dram_view =
            ck_tile::tensor_view<remove_cvref_t<decltype(weight_dram_buf)>,
                                remove_cvref_t<decltype(weight_dram_desc)>>{
                weight_dram_buf, weight_dram_desc};

        constexpr auto weight_dram_dist = TC::MakeWeightDramDistribution();
        auto weight_dram_window         = ck_tile::make_tile_window(
            weight_dram_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0},
            weight_dram_dist);

        constexpr auto weight_lds_desc = TC::MakeWeightLdsStoreDescriptor();
        auto weight_lds_view           = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<_Float16*>(weight_lds), weight_lds_desc);
        auto weight_lds_window = ck_tile::make_tile_window(
            weight_lds_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_size()>{}, ck_tile::number<8>{}),
            {0, 0});

        // Multi-pass weight loading: when the weight data is larger than
        // block_size, we need multiple async loads with advancing offsets.
        // The pad transform on the DRAM descriptor suppresses OOB reads
        // in the final pass.
        static_for<TC::NUM_WEIGHT_PASSES>(
            [&]<int Pass>()
            {
                ck_tile::async_load_tile(weight_lds_window, weight_dram_window);
                if constexpr(Pass < TC::NUM_WEIGHT_PASSES - 1)
                {
                    ck_tile::move_tile_window(weight_dram_window,
                                              {cfg.block_size(), 0});
                    ck_tile::move_tile_window(weight_lds_window,
                                              {cfg.block_size(), 0});
                }
            });
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
                static_cast<ck_tile::index_t>(TC::WEIGHT_LDS_PADDED_UINT4 *
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
            constexpr auto weight_lds_read_desc = TC::MakeWeightLdsReadDescriptor();
            auto weight_lds_view =
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    reinterpret_cast<_Float16*>(weight_lds), weight_lds_read_desc);

            constexpr auto weight_lds_read_dist = TC::MakeWeightLdsReadDistribution();
            auto weight_lds_read_window         = ck_tile::make_tile_window(
                weight_lds_view,
                ck_tile::make_tuple(ck_tile::number<TC::WEIGHT_LDS_READ_K>{},
                                    ck_tile::number<cfg.kh * cfg.kw>{},
                                    ck_tile::number<TC::GROUP_SIZE>{}),
                {0, 0, 0},
                weight_lds_read_dist);

            const auto weight_tile = ck_tile::load_tile(weight_lds_read_window);
            const auto& buf        = weight_tile.get_thread_buffer();

            static_for<cfg.kh * cfg.kw>(
                [&]<int khw>()
                {
                    __builtin_memcpy(&weights_reg[khw],
                                     &buf.get(khw * TC::GROUP_SIZE),
                                     sizeof(fp16x4_t));
                });
        }
    }
};

// Handles writing MFMA accumulator results directly to global memory.
// No LDS staging — store_tile writes directly from registers to DRAM.
// OOB handling is done via the pad transform on the wo dimension.
template <Config cfg>
struct OutputWriter
{
    using TC = TileConstants<cfg>;

    // Output tile distribution and distributed tensor type for direct DRAM writes.
    static constexpr auto OutputDramDist = TC::MakeOutputDramDistribution();
    using OutputDstrTensor =
        ck_tile::static_distributed_tensor<_Float16, ck_tile::remove_cvref_t<decltype(OutputDramDist)>>;

    // DRAM tile window type — derived from MakeOutputDramDescriptor with dummy args.
    using OutputDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::MakeOutputDramDescriptor(int{}, int{}, int{}))>;
    using OutputDramBuf =
        ck_tile::buffer_view<ck_tile::address_space_enum::global, _Float16, ck_tile::index_t, true>;
    using OutputDramView = ck_tile::tensor_view<OutputDramBuf, OutputDramDesc>;
    using OutputDramWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputDramView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<cfg.block_q()>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        OutputDramDist))>;

    OutputDramWindow dram_window;
    int last_p_out; // last row stored (for move_tile_window delta)

    __device__ OutputWriter(const BlockCoords<cfg>& bc,
                            uint4*, // Unused, need to match the signature of OutputWriterLds constructor for epilogue template.
                            _Float16* __restrict__ out,
                            int ho,
                            int wo)
        : last_p_out(0)
    {
        constexpr auto out_dist = TC::MakeOutputDramDistribution();
        const auto out_desc = TC::MakeOutputDramDescriptor(ho, wo, bc.C);
        auto out_buf = OutputDramBuf{
            out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k,
            static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
        auto out_view = OutputDramView{out_buf, out_desc};
        
        dram_window =  ck_tile::make_tile_window(
                  out_view,
                  ck_tile::make_tuple(ck_tile::number<1>{},
                                      ck_tile::number<cfg.block_q()>{},
                                      ck_tile::number<TC::BLOCK_C4>{},
                                      ck_tile::number<4>{}),
                  {0, bc.block_q, 0, 0},
                  out_dist);
    }

    // Convert fp32x4 accumulator to fp16x4 and write directly to global memory.
    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        // 1. Convert fp32→fp16 and pack into distributed tensor.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        const auto* fp16_ptr = reinterpret_cast<const _Float16*>(halves);

        OutputDstrTensor output_tile;
        output_tile.get_thread_buffer()(ck_tile::number<0>{}) = fp16_ptr[0];
        output_tile.get_thread_buffer()(ck_tile::number<1>{}) = fp16_ptr[1];
        output_tile.get_thread_buffer()(ck_tile::number<2>{}) = fp16_ptr[2];
        output_tile.get_thread_buffer()(ck_tile::number<3>{}) = fp16_ptr[3];

        // 2. Move window to current output row.
        ck_tile::move_tile_window(dram_window, {p_out - last_p_out, 0, 0, 0});
        last_p_out = p_out;

        // 3. Direct store to DRAM — pad transform handles OOB.
        ck_tile::store_tile(dram_window, output_tile);
    }
};

// Handles output staging through LDS and writing to global memory.
// Used when epilogue == RegistersToLdsToGlobalMemory.
template <Config cfg>
struct OutputWriterLds
{
    using TC = TileConstants<cfg>;

    // Output tile distribution and distributed tensor type for LDS writes.
    static constexpr auto OutputLdsDist = TC::MakeMfmaOutputDistribution();
    static constexpr auto OutputDramDist = TC::MakeOutputDramDistribution();

    using OutputDstrTensor =
        ck_tile::static_distributed_tensor<_Float16, ck_tile::remove_cvref_t<decltype(OutputLdsDist)>>;

    // LDS buffer - same for read and write
    using OutputLdsBuf    = ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>;

    // LDS write tile window type.
    using OutputLdsWriteDesc   = ck_tile::remove_cvref_t<decltype(TC::MakeOutputLdsWriteDescriptor())>;
    using OutputLdsWriteView   = ck_tile::tensor_view<OutputLdsBuf, OutputLdsWriteDesc>;
    using OutputLdsWriteWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputLdsWriteView{},
        ck_tile::make_tuple(ck_tile::number<cfg.block_q()>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0},
        // This ensures that tensor coordinates are pre-computed at construction time.
        OutputLdsDist))>;

    // Output tile distribution and distributed tensor for LDS reads before DRAM store.
    using OutputLdsReadDesc   = ck_tile::remove_cvref_t<decltype(TC::MakeOutputLdsReadDescriptor())>;
    using OutputLdsReadView   = ck_tile::tensor_view<OutputLdsBuf, OutputLdsReadDesc>;
    using OutputLdsReadWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputLdsReadView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<cfg.block_q()>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        // LDS read must match with DRAM write distribution. 
        // Pass DRAM distribution to pre-compute the mapping from tensor coordinates to thread coordinates.
        OutputDramDist))>; 

    // DRAM tile window type — derived from MakeOutputDramDescriptor with dummy args.
    using OutputDramDesc =
        ck_tile::remove_cvref_t<decltype(TC::MakeOutputDramDescriptor(int{}, int{}, int{}))>;
    using OutputDramBuf =
        ck_tile::buffer_view<ck_tile::address_space_enum::global, _Float16, ck_tile::index_t, true>;
    using OutputDramView = ck_tile::tensor_view<OutputDramBuf, OutputDramDesc>;
    using OutputDramWindow = ck_tile::remove_cvref_t<decltype(ck_tile::make_tile_window(
        OutputDramView{},
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<cfg.block_q()>{},
                            ck_tile::number<TC::BLOCK_C4>{},
                            ck_tile::number<4>{}),
        {0, 0, 0, 0},
        OutputDramDist))>;

    OutputLdsWriteWindow lds_write_window;
    OutputLdsReadWindow lds_read_window;
    OutputDramWindow dram_window;
    int last_p_out;

    __device__ OutputWriterLds(const BlockCoords<cfg>& bc,
                               uint4* output_lds,
                               _Float16* __restrict__ out,
                               int ho,
                               int wo)
        : last_p_out(0)
    {
        // Construct the common LDS buffer for write and read.
        auto lds_buf = OutputLdsBuf{
            reinterpret_cast<_Float16*>(output_lds),
            static_cast<ck_tile::index_t>(
                ck_tile::max(TC::WEIGHT_LDS_PADDED_UINT4, TC::OUTPUT_LDS_BUFFER_SIZE) *
                (sizeof(uint4) / sizeof(_Float16)))};

        // Construct LDS write window for the MFMA output staging to LDS.
        constexpr auto lds_write_desc = TC::MakeOutputLdsWriteDescriptor();
        constexpr auto lds_write_dist = TC::MakeMfmaOutputDistribution();
        auto lds_write_view = OutputLdsWriteView{lds_buf, lds_write_desc};
        lds_write_window = ck_tile::make_tile_window(
            lds_write_view,
            ck_tile::make_tuple(ck_tile::number<cfg.block_q()>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0},
            // If we don't pass the distribution, the coordinate-to-thread mapping will be computed at store time, causing overhead.
            lds_write_dist);

        // Construct LDS read window for staging before DRAM store.
        constexpr auto lds_read_desc = TC::MakeOutputLdsReadDescriptor();
        constexpr auto lds_read_dist = TC::MakeOutputDramDistribution(); // LDS read must match DRAM distribution for correct thread mapping.
        auto lds_read_view = OutputLdsReadView{lds_buf, lds_read_desc};
        lds_read_window = ck_tile::make_tile_window(
            lds_read_view,
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<cfg.block_q()>{},
                                ck_tile::number<TC::BLOCK_C4>{},
                                ck_tile::number<4>{}),
            {0, 0, 0, 0},
            // If we don't pass the distribution, the coordinate-to-thread mapping will be computed at store time, causing overhead.
            lds_read_dist);

        // Construct output DRAM window for global memory writes.
        constexpr auto out_dist = TC::MakeOutputDramDistribution();
        const auto out_desc = TC::MakeOutputDramDescriptor(ho, wo, bc.C);
        auto out_buf = OutputDramBuf{
            out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C + bc.block_k,
            static_cast<ck_tile::index_t>(out_desc.get_element_space_size())};
        auto out_view = OutputDramView{out_buf, out_desc};
        
        dram_window = ck_tile::make_tile_window(
                  out_view,
                  ck_tile::make_tuple(ck_tile::number<1>{},
                                      ck_tile::number<cfg.block_q()>{},
                                      ck_tile::number<TC::BLOCK_C4>{},
                                      ck_tile::number<4>{}),
                  {0, bc.block_q, 0, 0},
                  out_dist);
    }

    // Convert fp32x4 accumulator to fp16x4 and write through LDS to global memory.
    __device__ void flush(fp32x4_t acc_val, int p_out)
    {
        // 1. Convert fp32→fp16 and pack into distributed tensor.
        __half2 halves[2];
        halves[0] = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1] = __float22half2_rn({acc_val[2], acc_val[3]});
        const auto* fp16_ptr = reinterpret_cast<const _Float16*>(halves);

        OutputDstrTensor output_tile;
        output_tile.get_thread_buffer()(ck_tile::number<0>{}) = fp16_ptr[0];
        output_tile.get_thread_buffer()(ck_tile::number<1>{}) = fp16_ptr[1];
        output_tile.get_thread_buffer()(ck_tile::number<2>{}) = fp16_ptr[2];
        output_tile.get_thread_buffer()(ck_tile::number<3>{}) = fp16_ptr[3];

        // 2. Store to LDS via tile window.
        ck_tile::store_tile(lds_write_window, output_tile);

        // 3. Synchronize such that all threads' LDS writes are visible.
        ck_tile::s_waitcnt_lgkm<0>();

        // 4. LDS back to registers for coalesced store to DRAM.
        const auto lds_tile = ck_tile::load_tile(lds_read_window);

        // 5. Move window to current output row.
        ck_tile::move_tile_window(dram_window, {p_out - last_p_out, 0, 0, 0});
        last_p_out = p_out;

        // 6. Store LDS tile to DRAM — pad transform handles OOB.
        ck_tile::store_tile(dram_window, lds_tile);
    }
};

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

    // --- LDS buffers (padded to accommodate the full tile distribution span) ---
    // TODO: Optimize the LDS usage if occupancy get limited by the LDS usage.
    __shared__ uint4 input_lds[TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8];
    // LDS epilogue needs space for output staging; direct DRAM path needs weight-only.
    static constexpr int OUTPUT_LDS_SIZE = use_lds_epilogue
                                               ? ck_tile::max(TC::WEIGHT_LDS_PADDED_UINT4,
                                                              TC::OUTPUT_LDS_BUFFER_SIZE)
                                               : TC::WEIGHT_LDS_PADDED_UINT4;
    __shared__ uint4 output_lds[OUTPUT_LDS_SIZE];

    // --- Coordinate setup ---
    BlockCoords<cfg> bc(groups);
    if(bc.block_n >= N)
        return;
                                             
    InputLoader<cfg> il(bc, input_lds, in, hi, wi, px);
    OutputWriterType ow(bc, output_lds, out, ho, wo);

    fp16x4_t weights_reg[cfg.kh * cfg.kw];
    WeightLoader<cfg>::load_to_lds(bc, output_lds, wei);
    // Wait for weight loads to complete before reading from LDS.
    wait_vmcnt<0>();
    __syncthreads();

    // We can use the output LDS storage for loading the weights as we don't yet compute any output.
    // The output LDS is guaranteed to have sufficient space.
    WeightLoader<cfg>::read_from_lds(weights_reg, output_lds);
    __syncthreads();

    // Prefetch first input row (row 0) into LDS buffer 0.
    il.prefetch_tile_to_lds(0);
    wait_vmcnt<0>();
    __syncthreads();

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
                if((y + 1) < hi)
                {
                    il.fetch_tile_to_lds(tic);
                }

                // Accumulate MFMA products over filter width.
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
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_4x4x4f16(
                                        weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                                        input_reg,
                                        acc[p_idx],
                                        0,
                                        0,
                                        0);
                                else
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_4x4x4f16(
                                        weights_reg[R * cfg.kw + S],
                                        input_reg,
                                        acc[p_idx],
                                        0,
                                        0,
                                        0);
                            });
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
    } // end of the main loop

    // --- Remainder loop: hi % kh leftover rows ---
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
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_4x4x4f16(
                                        weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                                        input_reg,
                                        acc[p_idx],
                                        0,
                                        0,
                                        0);
                                else
                                    acc[p_idx] = __builtin_amdgcn_mfma_f32_4x4x4f16(
                                        weights_reg[R * cfg.kw + S],
                                        input_reg,
                                        acc[p_idx],
                                        0,
                                        0,
                                        0);
                            });
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
