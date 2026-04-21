// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/direct_convolution/utils/matrix_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/types.hpp"
#include "ck_tile/ops/direct_convolution/utils/mathutil.hpp"
#include "ck_tile/ops/direct_convolution/utils/launch_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/kernel_variant.hpp"
#include "ck_tile/ops/direct_convolution/utils/transpose_lds_layout.hpp"
#include "ck_tile/ops/direct_convolution/utils/memory.hpp"
#include "ck_tile/ops/direct_convolution/utils/detail.hpp"
#include "ck_tile/ops/direct_convolution/utils/swizzle.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/tensor/buffer_view.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/load_tile.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <type_traits>
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

    // Swizzle pattern - v2 uses the XOR-based swizzle.
    SwizzleType swizzle_type = SwizzleType::XOR;

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
        return "v3_grouped_4c_swizzleXOR";
    }
};

// All instantiated configurations. The first valid config is expected to be the fastest.
constexpr Config configs[] = {
    {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 8}
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
    return true;
}

inline LaunchParams get_launch_params(int config_idx, const Conv2dParams& par)
{
    const auto& cfg = configs[config_idx];

    // Compute the grid size.
    // For Dgrad the output is the input gradient (width = par.w, not par.q).
    const int out_q    = (cfg.direction == Direction::Dgrad) ? par.w : par.q;
    auto blocks_w      = divup(out_q, cfg.block_q());
    auto blocks_w_n    = blocks_w * cfg.n_fold;
    auto blocks_c      = divup(par.c_tot, cfg.block_c());
    auto blocks_n_fold = divup(par.n, cfg.n_fold);

    LaunchParams launch;
    launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
    launch.block_size = dim3(cfg.block_size(), 1, 1);
    return launch;
}

// Tile constants derived from the kernel configuration.
template <Config cfg>
struct TileConstants
{
    static constexpr int MFMA_M     = 4;
    static constexpr int MFMA_K     = 4;
    static constexpr int MFMA_N     = 4;
    static constexpr int MFMA_BATCH = 16;

    using OperandLayout = MatrixLayout<MFMA_M, MFMA_K, MFMA_BATCH, __half>;
    using ResultLayout  = MatrixLayout<MFMA_N, MFMA_K, MFMA_BATCH, float>;

    static constexpr int GROUP_SIZE   = cfg.channels_per_group; // 4
    static constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4;         // 1

    // Number of input columns loaded by each workgroup (output columns plus halo).
    static constexpr int BLOCK_W = cfg.block_q() + (cfg.kw - 1);

    // uint4 vectors per channel fiber (8 fp16 per uint4).
    static constexpr int BLOCK_C8 = cfg.block_c() / 8;

    // Number of uint4 vectors to store per output row.
    static constexpr int STORE_VECS = cfg.block_q() * BLOCK_C8;

    // LDS double buffering for input loads.
    static constexpr int NUM_INPUT_LDS_BUFFERS    = 2;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C8 = BLOCK_C8 * BLOCK_W;
    static constexpr int INPUT_LDS_BUFFER_SIZE_C4 = INPUT_LDS_BUFFER_SIZE_C8 * 2;
    static constexpr int OUTPUT_LDS_BUFFER_SIZE   = BLOCK_C8 * cfg.block_q();

    // Weight LDS staging: [kh*kw][block_groups][GROUP_SIZE] in uint2 units.
    static constexpr int WEIGHT_LDS_SIZE_UINT2 = cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE;
    static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_SIZE_UINT2 / 2;

    // -----------------------------------------------------------------------
    // LDS descriptor with XOR swizzle (uint4 units, for MFMA read path).
    //
    // Logical layout: [BLOCK_W, BLOCK_C8] row-major (x = row, c8 = column)
    // Physical layout: xor_t transform maps (x, c8) → (x, c8 ^ (x % C8))
    //   which produces flat offset x * C8 + (c8 ^ (x % C8)).
    //
    // This is the CK Tile equivalent of SwizzleXOR::offset_uint4(x, c8).
    // -----------------------------------------------------------------------
    static constexpr auto MakeInputLdsBlockDescriptor()
    {
        constexpr auto desc_naive = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<BLOCK_W>{}, ck_tile::number<BLOCK_C8>{}),
            ck_tile::make_tuple(ck_tile::number<BLOCK_C8>{}, ck_tile::number<1>{}));

        return ck_tile::transform_tensor_descriptor(
            desc_naive,
            ck_tile::make_tuple(ck_tile::make_xor_transform(
                ck_tile::make_tuple(ck_tile::number<BLOCK_W>{}, ck_tile::number<BLOCK_C8>{}))),
            ck_tile::make_tuple(ck_tile::sequence<0, 1>{}),
            ck_tile::make_tuple(ck_tile::sequence<0, 1>{}));
    }

    // Compute uint2 offset from logical (x, c4) using the uint4 descriptor.
    //   uint2_offset = descriptor_offset(x, c8) * 2 + c4 % 2
    // where c8 = c4 / 2.
    static CK_TILE_DEVICE int lds_offset_uint2(int x, int c4)
    {
        constexpr auto desc = MakeInputLdsBlockDescriptor();
        const int c8    = c4 / 2;
        const int c4_lo = c4 % 2;
        auto coord = ck_tile::make_tensor_coordinate(desc, ck_tile::make_tuple(x, c8));
        return static_cast<int>(coord.get_offset()) * 2 + c4_lo;
    }

    // Compute uint4 offset from logical (x, c8) using the descriptor.
    static CK_TILE_DEVICE int lds_offset_uint4(int x, int c8)
    {
        constexpr auto desc = MakeInputLdsBlockDescriptor();
        auto coord = ck_tile::make_tensor_coordinate(desc, ck_tile::make_tuple(x, c8));
        return static_cast<int>(coord.get_offset());
    }

    // Total channels per block in fp16 elements.
    static constexpr int BLOCK_C = BLOCK_C8 * 8;

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

    using Swizzle = SwizzleXOR<cfg.block_c()>;

    // -----------------------------------------------------------------------
    // Tile-level async load constants and descriptors.
    //
    // The tile distribution maps (warp_id, lane_id) to a 3D tile coordinate
    // (x_local, c8_local, c_sub) where x_local ∈ [0, TOTAL_SPATIAL),
    // c8_local ∈ [0, BLOCK_C8), c_sub ∈ [0, 8).  TOTAL_SPATIAL > BLOCK_W
    // in general; surplus threads (x_local >= BLOCK_W) produce OOB
    // coordinates that the pad transform suppresses automatically.
    //
    // The DRAM descriptor applies an XOR transform so that lane
    // (x_local, c8_local) reads from global coordinate
    // (x_local, c8_local ^ (x_local % BLOCK_C8)), producing XOR-swizzled
    // data in LDS that matches the MFMA read path (lds_offset_uint2).
    //
    // Element type: _Float16 (fp16). The 8-element sub-channel dimension
    // is the vectorization dimension (Y), producing 128-bit loads.
    // Row advancement uses the linear offset parameter of
    // async_load_tile_with_offset.
    // -----------------------------------------------------------------------
    static constexpr int LANES_PER_ROW = WAVE_SIZE / BLOCK_C8;
    static constexpr int TOTAL_SPATIAL = cfg.block_size() / BLOCK_C8;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C8 = BLOCK_C8 * TOTAL_SPATIAL;
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_C4 = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 2;
    // Padded LDS buffer size in fp16 elements (each C8 group = 8 fp16).
    static constexpr int INPUT_LDS_BUFFER_SIZE_PADDED_FP16 = INPUT_LDS_BUFFER_SIZE_PADDED_C8 * 8;

    // DRAM descriptor for tile-level async loads: [hi, wi_padded, BLOCK_C8, 8]
    // in fp16 units, with pad transform on W and XOR transform on (W, C8).
    // Row dimension allows advancing rows via move_tile_window({1, 0, 0, 0}).
    //
    // Base pointer: in + batch_offset + block_k  (shifted to tile's channel origin).
    // The XOR works because block_q is always a multiple of BLOCK_C8:
    //   block_q = waves_q4 * 4 * tile_idx, and waves_q4 * 4 >= BLOCK_C8.
    //   So (block_q + x_local) % BLOCK_C8 == x_local % BLOCK_C8.
    static CK_TILE_DEVICE auto MakeInputDramDescriptorXOR(int hi, int wi, int C_total, int px)
    {
        constexpr int right_pad_w = cfg.kw - 1;
        const int wi_padded       = wi + px + right_pad_w;

        // Step 1: Naive [hi, wi, BLOCK_C8, 8] with global strides in fp16 units.
        // stride_sub = 1, stride_c8 = 8, stride_w = C_total, stride_h = wi * C_total.
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

        // Step 3: XOR transform on (W_padded, BLOCK_C8).
        // XOR computes c8_phys = c8 ^ (col_padded % BLOCK_C8).
        // Since block_q % BLOCK_C8 == 0, col_padded % BLOCK_C8 == x_local % BLOCK_C8.
        // Row and sub-channel dimensions pass through unchanged.
        const auto desc_xor = ck_tile::transform_tensor_descriptor(
            desc_padded,
            ck_tile::make_tuple(
                ck_tile::make_pass_through_transform(hi),
                ck_tile::make_xor_transform(
                    ck_tile::make_tuple(wi_padded, ck_tile::number<BLOCK_C8>{})),
                ck_tile::make_pass_through_transform(ck_tile::number<8>{})),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                ck_tile::sequence<3>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{},
                                ck_tile::sequence<3>{}));

        return desc_xor;
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
    // X1 = TOTAL_SPATIAL, decomposed as [LANES_PER_ROW, NUM_WAVES]:
    //   P0 contributes NUM_WAVES factor, P1 contributes LANES_PER_ROW factor
    // X2 = BLOCK_C8, decomposed as [BLOCK_C8]:
    //   P1 contributes BLOCK_C8 factor
    // X3 = 8, decomposed as [8]:
    //   Y1 iteration dimension (128-bit vectorized loads)
    static constexpr int NUM_WAVES = cfg.num_waves();

    static constexpr auto MakeInputDramDistribution()
    {
        // RH-major indices: 0=R(empty), 1=X0(row), 2=X1(spatial), 3=X2(channel), 4=X3(sub)
        // P0 (warp_id=NUM_WAVES): maps to X1 H-factor 1 → major=2, minor=1
        // P1 (lane_id=64=LANES_PER_ROW*BLOCK_C8):
        //   factor 0 (LANES_PER_ROW): maps to X1 H-factor 0 → major=2, minor=0
        //   factor 1 (BLOCK_C8):      maps to X2 H-factor 0 → major=3, minor=0
        // Y0 (length 1): maps to X0 H-factor 0 → major=1, minor=0 (trivial row)
        // Y1 (length 8): maps to X3 H-factor 0 → major=4, minor=0 (vectorization)
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<1>,
                               ck_tile::sequence<LANES_PER_ROW, NUM_WAVES>,
                               ck_tile::sequence<BLOCK_C8>,
                               ck_tile::sequence<8>>,
                ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 3>>,
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<0, 0>>,
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

// Thread-level coordinates derived from threadIdx and MFMA lane mappings.
template <Config cfg>
struct ThreadMapping
{
    using TC = TileConstants<cfg>;

    int tid;
    int wave;
    int lane;
    int wave_c64;
    int wave_q4;

    // MFMA result coordinates for this thread.
    int thread_q;
    int lane_c4;
    int lane_batch;

    __device__ ThreadMapping()
        : tid(threadIdx.x),
          wave(tid / WAVE_SIZE),
          lane(tid % WAVE_SIZE),
          wave_c64(wave % cfg.waves_c64),
          wave_q4(wave / cfg.waves_c64),
          thread_q(wave_q4 * WARP_Q + TC::ResultLayout::outer(lane)),
          lane_c4(TC::ResultLayout::inner(lane) / 4),
          lane_batch(TC::ResultLayout::batch(lane))
    {
    }
};

// Handles weight loads from global memory into LDS and then into registers.
template <Config cfg>
struct WeightLoader
{
    using TC = TileConstants<cfg>;

    // Load weights from global memory into LDS (output_lds is reused for weight staging).
    __device__ static void load_to_lds(int tid,
                                       const BlockCoords<cfg>& bc,
                                       uint4* output_lds,
                                       const _Float16* __restrict__ wei)
    {
        // Weight tensor layout is [K_tot, kh, kw, C_per_group] in fp16 elements.
        // Each thread loads 8 contiguous fp16 (= 1 uint4 = 16 bytes) per iteration.
        // block_k selects the starting output channel for this workgroup.
        constexpr int FP16_PER_UINT4 = 8;
        const int weight_elements    = bc.K * cfg.kh * cfg.kw * TC::GROUP_SIZE;
        const int base_fp16          = bc.block_k * cfg.kh * cfg.kw * TC::GROUP_SIZE;

        auto weight_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            wei,
            ck_tile::make_tuple(weight_elements), // Flat tensor
            ck_tile::make_tuple(ck_tile::number<1>{}), // Stride doesn't matter since we only use linear indexing.
            ck_tile::number<FP16_PER_UINT4>{}); // Vector size of 8 fp16 elements per uint4

        for(int j = tid; j < TC::WEIGHT_LDS_SIZE_UINT4; j += cfg.block_size())
        {
            auto coord = ck_tile::make_tensor_coordinate(
                weight_view.get_tensor_descriptor(),
                ck_tile::make_tuple(base_fp16 + j * FP16_PER_UINT4));
            weight_view.template async_get_vectorized_elements<ck_tile::fp16x8_t, false>(
                reinterpret_cast<CK_TILE_LDS_ADDR _Float16*>(&output_lds[j]),
                coord,
                0,
                ck_tile::bool_constant<false>{});
        }
    }

    // Read weights from LDS into registers after sync.
    __device__ static void read_from_lds(
        fp16x4_t (&weights_reg)[cfg.kh * cfg.kw],
        const ThreadMapping<cfg>& tm,
        ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>&
            output_lds_fp16)
    {
        if constexpr(cfg.direction == Direction::Dgrad)
        {
            using TransposeLayout = TransposeLDSLayout<4, 4, 16>;
            const int tr_batch    = TransposeLayout::batch(tm.lane);
            const int tr_row      = TransposeLayout::row(tm.lane);
            int filter_local      = tm.wave_c64 * 64 + tr_batch * TC::GROUP_SIZE + tr_row;

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
            auto lane_k      = TC::OperandLayout::outer(tm.lane);
            auto lane_batch  = TC::OperandLayout::batch(tm.lane);
            int filter_local = tm.wave_c64 * 64 + lane_batch * TC::GROUP_SIZE + lane_k;

            const ck_tile::index_t weight_base = filter_local * cfg.kh * cfg.kw * 4;
            for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
            {
                weights_reg[khw] = output_lds_fp16.template get<ck_tile::fp16x4_t>(
                    weight_base + khw * 4, 0, true);
            }
        }
    }
};

// Handles output staging through LDS and writing to global memory.
template <Config cfg>
struct OutputWriter
{
    using TC = TileConstants<cfg>;

    // Output tensor descriptor: [ho, wo, C] in _Float16 elements for this batch.
    // Used only for offset computation; the actual store is a flat global write.
    using OutputDesc = decltype(ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(int{}, int{}, int{}),
        ck_tile::make_tuple(int{}, int{}, ck_tile::number<1>{}),
        ck_tile::number<8>{}));

    bool store_active;
    bool store_valid;
    const uint4* load_output_lds;
    _Float16* out_base;   // output base pointer for this batch
    OutputDesc out_desc;  // descriptor for coordinate → offset mapping
    int out_q;            // output column coordinate
    int out_c_fp16;       // output channel coordinate in fp16 units
    int output_lds_offset;

    __device__ OutputWriter(const ThreadMapping<cfg>& tm,
                            const BlockCoords<cfg>& bc,
                            uint4* output_lds,
                            _Float16* __restrict__ out,
                            int ho,
                            int wo)
        : store_active(tm.tid < TC::STORE_VECS),
          store_valid(false),
          load_output_lds(nullptr),
          out_base(out + static_cast<size_t>(bc.block_n) * ho * wo * bc.C),
          out_desc(ck_tile::make_naive_tensor_descriptor(
              ck_tile::make_tuple(ho, wo, bc.C),
              ck_tile::make_tuple(wo * bc.C, bc.C, ck_tile::number<1>{}),
              ck_tile::number<8>{})),
          out_q(0),
          out_c_fp16(0)
    {
        // Pre-compute the output LDS swizzle offset (thread-constant).
        output_lds_offset = TC::lds_offset_uint2(
            tm.thread_q, tm.wave_c64 * 16 + tm.lane_batch * TC::GROUP_SIZE_4 + tm.lane_c4);

        if(store_active)
        {
            const int col = tm.tid / TC::BLOCK_C8;
            const int c8  = tm.tid % TC::BLOCK_C8;
            out_q         = bc.block_q + col;
            out_c_fp16    = (bc.block_c8 + c8) * 8;
            load_output_lds = &output_lds[TC::lds_offset_uint4(col, c8)];
            store_valid     = (out_q < wo);
        }
    }

    // Convert fp32x4 accumulator to fp16x4 and write through LDS to global memory.
    __device__ void flush(
        fp32x4_t acc_val,
        int p_out,
        ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>&
            output_lds_fp16) const
    {
        __half2 halves[2];
        halves[0]    = __float22half2_rn({acc_val[0], acc_val[1]});
        halves[1]    = __float22half2_rn({acc_val[2], acc_val[3]});
        auto out_reg = *reinterpret_cast<const fp16x4_t*>(halves);

        output_lds_fp16.template set<ck_tile::fp16x4_t>(
            output_lds_offset * 4, 0, true, out_reg);

        __syncthreads();

        if(store_valid)
        {
            // Compute the global offset from (p_out, out_q, out_c_fp16) coordinates.
            auto coord = ck_tile::make_tensor_coordinate(
                out_desc, ck_tile::make_tuple(p_out, out_q, out_c_fp16));
            *reinterpret_cast<uint4*>(out_base + coord.get_offset()) = *load_output_lds;
        }
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
    using TC = TileConstants<cfg>;

    // --- LDS buffers (padded to accommodate the full tile distribution span) ---
    __shared__ uint4 input_lds[TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8];
    __shared__ uint4 output_lds[maximum(TC::WEIGHT_LDS_SIZE_UINT4, TC::OUTPUT_LDS_BUFFER_SIZE)];

    auto input_lds_fp16 =
        ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>{
            reinterpret_cast<_Float16*>(input_lds),
            static_cast<ck_tile::index_t>(
                TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_PADDED_C8 *
                (sizeof(uint4) / sizeof(_Float16)))};

    auto output_lds_fp16 =
        ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>{
            reinterpret_cast<_Float16*>(output_lds),
            static_cast<ck_tile::index_t>(
                maximum(TC::WEIGHT_LDS_SIZE_UINT4, TC::OUTPUT_LDS_BUFFER_SIZE) *
                (sizeof(uint4) / sizeof(_Float16)))};

    // --- Coordinate setup ---
    BlockCoords<cfg> bc(groups);
    if(bc.block_n >= N)
        return;

    ThreadMapping<cfg> tm;
    OutputWriter<cfg> ow(tm, bc, output_lds, out, ho, wo);

    // --- Input tile windows for async_load_tile ---
    // DRAM tensor view: fp16 data with XOR-swizzled 4D descriptor [hi, wi_padded, C8, 8].
    // Base pointer shifted to this tile's channel origin.
    const auto input_dram_desc = TC::MakeInputDramDescriptorXOR(hi, wi, bc.C, px);
    const auto input_dram_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
        in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C + bc.block_k,
        input_dram_desc);

    // DRAM tile window with distribution. Window = [1, TOTAL_SPATIAL, BLOCK_C8, 8].
    constexpr auto input_dram_dist = TC::MakeInputDramDistribution();
    auto input_dram_window         = ck_tile::make_tile_window(
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
    auto lds_window_0 = ck_tile::make_tile_window(
        lds_view_0,
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        {0, 0, 0, 0});
    auto lds_window_1 = ck_tile::make_tile_window(
        lds_view_1,
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<TC::TOTAL_SPATIAL>{},
                            ck_tile::number<TC::BLOCK_C8>{}, ck_tile::number<8>{}),
        {0, 0, 0, 0});

    // --- Weight prologue: global → LDS → registers ---
    fp16x4_t weights_reg[cfg.kh * cfg.kw];
    WeightLoader<cfg>::load_to_lds(tm.tid, bc, output_lds, wei);
    // Prefetch first input row (row 0) into LDS buffer 0.
    ck_tile::async_load_tile(lds_window_0, input_dram_window);

    {
        wait_vmcnt<0>();
        __syncthreads();
        WeightLoader<cfg>::read_from_lds(weights_reg, tm, output_lds_fp16);
    }

    // --- Pre-compute per-thread LDS offsets (XOR-swizzled, uint2 units) ---
    int input_lds_offsets[cfg.kw];
    static_for<cfg.kw>(
        [&]<int S>()
        {
            input_lds_offsets[S] = TC::lds_offset_uint2(
                tm.thread_q + S,
                tm.wave_c64 * 16 + tm.lane_batch * TC::GROUP_SIZE_4 + tm.lane_c4);
        });

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
                    ck_tile::move_tile_window(input_dram_window, {1, 0, 0, 0});
                    if(tic == 0)
                        ck_tile::async_load_tile(lds_window_0, input_dram_window);
                    else
                        ck_tile::async_load_tile(lds_window_1, input_dram_window);
                }

                // Accumulate MFMA products over filter width.
                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        auto input_reg = input_lds_fp16.template get<ck_tile::fp16x4_t>(
                            0,
                            (toc * TC::INPUT_LDS_BUFFER_SIZE_PADDED_C4 + input_lds_offsets[S]) * 4,
                            true);

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
                    ow.flush(acc[P_FLUSH], p_out, output_lds_fp16);
                acc[P_FLUSH] = Zero;
            });
    }

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
                    ck_tile::move_tile_window(input_dram_window, {1, 0, 0, 0});
                    if(tic == 0)
                        ck_tile::async_load_tile(lds_window_0, input_dram_window);
                    else
                        ck_tile::async_load_tile(lds_window_1, input_dram_window);
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        fp16x4_t input_reg = input_lds_fp16.template get<ck_tile::fp16x4_t>(
                            0,
                            (toc * TC::INPUT_LDS_BUFFER_SIZE_PADDED_C4 + input_lds_offsets[S]) * 4,
                            true);

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
                    ow.flush(acc[P_FLUSH], p_out, output_lds_fp16);
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
        ow.flush(slot, p_out, output_lds_fp16);
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
