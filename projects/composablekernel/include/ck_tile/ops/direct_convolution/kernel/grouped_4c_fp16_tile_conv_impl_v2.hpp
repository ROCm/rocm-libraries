// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/tensor/tensor_descriptor.hpp"
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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <type_traits>
#include <string>

namespace ck_tile::direct_conv::grouped_4c_tile::v2
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
        return "v2_grouped_4c_swizzleXOR";  
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

    // Size of the vector load
    static constexpr int ELEMENTS_PER_VECTOR_LOAD = 8;

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
    // Global input tensor descriptor: 3D [hi, wi_padded, C] in fp16 elements.
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
            ck_tile::number<ELEMENTS_PER_VECTOR_LOAD>{},
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
    // Global weight descriptor: 1D flat tensor in fp16 elements.
    // Weight tensor layout is [K_tot, kh, kw, C_per_group] in fp16 elements.
    //
    // Used by WeightLoader for per-thread async loads.
    // -----------------------------------------------------------------------
    static CK_TILE_DEVICE auto MakeWeightGlobalDescriptor(int K)
    {
        const int weight_elements    = K * cfg.kh * cfg.kw * GROUP_SIZE;
        const auto desc = make_naive_tensor_descriptor(
            ck_tile::make_tuple(weight_elements),
            ck_tile::make_tuple(ck_tile::number<1>{}), 
            ck_tile::number<ELEMENTS_PER_VECTOR_LOAD>{},
            ck_tile::number<1>{}
        );
        return desc;
    }

    using Swizzle = SwizzleXOR<cfg.block_c()>;
};

// Workgroup-level coordinates derived from blockIdx.
template <Config cfg>
struct BlockCoords
{
    int block_n;      // Batch index for this workgroup (unfolded from blockIdx).
    int block_q;      // Starting output column (W dimension) for this workgroup.
    int block_group;  // Starting convolution group index for this workgroup.
    int block_k;      // Starting output channel index (= block_group * channels_per_group).
    int block_k8;     // block_k expressed in uint4 vector units (block_k / 8).
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
        block_k8    = block_k / 8;
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

// Handles asynchronous input loads from global memory into LDS.
//
// Each thread computes its own (col, c8) coordinate via the XOR swizzle,
// reads from the padded global descriptor, and writes to its sequential
// LDS slot.  The XOR swizzle scrambles the global read addresses so that
// the data lands in LDS in the XOR-swizzled order expected by the MFMA
// read path (lds_offset_uint2).
template <Config cfg>
struct InputLoader
{
    using TC = TileConstants<cfg>;
    using InputTensorView = decltype(ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
        static_cast<const _Float16*>(nullptr),
        TC::MakeInputGlobalDescriptor(int{}, int{}, int{}, int{})));

    InputTensorView input_view;
    uint4* store_input_lds;
    bool load_active;
    int abs_padded_col;  // absolute column in padded space (block_q + tile-local col)
    int abs_c8_fp16;  // absolute channel in fp16 units (block_k + c8_thread * 8)

    __device__ InputLoader(int tid,
                           const BlockCoords<cfg>& bc,
                           uint4* input_lds,
                           const _Float16* __restrict__ in,
                           int hi,
                           int wi,
                           int px)
        : input_view(ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
              in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C,
              TC::MakeInputGlobalDescriptor(hi, wi, bc.C, px))),
          store_input_lds(&input_lds[tid]),
          load_active(tid < TC::BLOCK_W * TC::BLOCK_C8),
          abs_padded_col(bc.block_q + TC::Swizzle::x(tid)),
          abs_c8_fp16(bc.block_k + TC::Swizzle::c8(tid) * 8)
    {
    }

    // Issue an async load for input row y into the specified LDS buffer half.
    // The padded descriptor automatically returns zero for OOB column accesses.
    __device__ void prefetch(int y, int lds_buffer) const
    {
        if(load_active)
        {
            auto coord = ck_tile::make_tensor_coordinate(
                input_view.get_tensor_descriptor(),
                ck_tile::make_tuple(y, abs_padded_col, abs_c8_fp16));
            input_view.template async_get_vectorized_elements<ck_tile::fp16x8_t>(
                reinterpret_cast<CK_TILE_LDS_ADDR _Float16*>(
                    store_input_lds + lds_buffer * TC::INPUT_LDS_BUFFER_SIZE_C8),
                coord,
                0,
                ck_tile::bool_constant<true>{});
        }
    }

    // Issue the initial load for row 0 into LDS buffer 0.
    __device__ void prefetch_first_row() const { prefetch(0, 0); }
};

// Handles weight loads from global memory into LDS and then into registers.
template <Config cfg>
struct WeightLoader
{
    using TC = TileConstants<cfg>;
    using WeightTensorView = decltype(ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
        static_cast<const _Float16*>(nullptr), TC::MakeWeightGlobalDescriptor(int{})));

    WeightTensorView weight_view;
    uint4* store_weight_lds;
    int offset;
    int num_threads_per_block;

    __device__ WeightLoader(int tid,
                           const BlockCoords<cfg>& bc,
                           uint4* weight_lds,
                           const _Float16* __restrict__ wei) 
        // block_k selects the starting output channel for this workgroup.
        : weight_view(ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
              wei + static_cast<size_t>(bc.block_k) * cfg.kh * cfg.kw * TC::GROUP_SIZE,
              TC::MakeWeightGlobalDescriptor(bc.K))),
          store_weight_lds(weight_lds),
          offset(tid),
          num_threads_per_block(cfg.block_size())
    {
    }

    // Load weights from global memory into LDS (output_lds is reused for weight staging).
    __device__ void load_to_lds()
    {
        // Each thread loads 8 contiguous fp16 (= 1 uint4 = 16 bytes) per iteration.
        // All threads in the workgroup (thread block) sweep once over the flat weight tensor.
        // When the first batch is loaded, start loading the next batch of 8 values if the thread
        // maps to a valid LDS address. 
        for(int j = offset; j < TC::WEIGHT_LDS_SIZE_UINT4; j += num_threads_per_block)
        {
            auto coord = ck_tile::make_tensor_coordinate(
                weight_view.get_tensor_descriptor(),
                ck_tile::make_tuple(j * TC::ELEMENTS_PER_VECTOR_LOAD));
            weight_view.template async_get_vectorized_elements<ck_tile::fp16x8_t, false>(
                reinterpret_cast<CK_TILE_LDS_ADDR _Float16*>(&store_weight_lds[j]),
                coord,
                0,
                ck_tile::bool_constant<false>{});
        }
    }

    // Read weights from LDS into registers after sync.
    __device__ static void read_from_lds(
        fp16x4_t (&weights_reg)[cfg.kh * cfg.kw],
        const ThreadMapping<cfg>& tm,
        ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>& output_lds_fp16)
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
            out_c_fp16    = (bc.block_k8 + c8) * 8;
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

    // --- LDS buffers ---
    __shared__ uint4 input_lds[TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8];
    __shared__ uint4 output_lds[maximum(TC::WEIGHT_LDS_SIZE_UINT4, TC::OUTPUT_LDS_BUFFER_SIZE)];

    auto input_lds_fp16 =
        ck_tile::buffer_view<ck_tile::address_space_enum::lds, _Float16, ck_tile::index_t, true>{
            reinterpret_cast<_Float16*>(input_lds),
            static_cast<ck_tile::index_t>(
                TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8 *
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
    InputLoader<cfg> il(tm.tid, bc, input_lds, in, hi, wi, px);
    WeightLoader<cfg> wl(tm.tid, bc, output_lds, wei);
    OutputWriter<cfg> ow(tm, bc, output_lds, out, ho, wo);

    // --- Weight prologue: global → LDS → registers ---
    fp16x4_t weights_reg[cfg.kh * cfg.kw];
    wl.load_to_lds();
    il.prefetch_first_row();

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
                    il.prefetch(y + 1, tic);
                }

                // Accumulate MFMA products over filter width.
                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        auto input_reg = input_lds_fp16.template get<ck_tile::fp16x4_t>(
                            0,
                            (toc * TC::INPUT_LDS_BUFFER_SIZE_C4 + input_lds_offsets[S]) * 4,
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
                    il.prefetch(y + 1, tic);
                }

                static_for<cfg.kw>(
                    [&]<int S>()
                    {
                        fp16x4_t input_reg = input_lds_fp16.template get<ck_tile::fp16x4_t>(
                            0,
                            (toc * TC::INPUT_LDS_BUFFER_SIZE_C4 + input_lds_offsets[S]) * 4,
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

} // namespace ck_tile::direct_conv::grouped_4c_tile::v2
