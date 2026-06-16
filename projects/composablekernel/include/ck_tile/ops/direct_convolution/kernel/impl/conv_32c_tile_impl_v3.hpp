// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CK Tile v3 implementation of non-grouped (standard) convolution with
// cross-wave LDS reduction. Supports two MFMA shapes:
//
//   mfma_f32_16x16x32: 16 spatial x 16 K-output, 32-ch C-reduction
//   mfma_f32_32x32x16: 32 spatial x 32 K-output, 16-ch C-reduction
//
// Splits the C-reduction across waves within the same workgroup. Each wave
// handles one or more channels_per_group C-slices, all producing partial sums for
// the same block_k_size K-channels. An LDS-based cross-wave reduction
// combines the partial sums before output.
//
// Design:
//   - waves_per_group = 1 (each wave is its own C-group)
//   - block_k_size = mfma_n (16 or 32 K-channels)
//   - channels_per_group = mfma_k (32 or 16 C-channels per wave)
//   - block_c = waves_per_wg * channels_per_group
//   - Cross-wave LDS reduction at flush points
//
// Supported: fp16 and bf16, Fprop and Dgrad.

#pragma once

#include "ck_tile/ops/direct_convolution/kernel/impl/grouped_conv_kernel_base.hpp"
#include "ck_tile/ops/direct_convolution/kernel/impl/grouped_conv_input_loader.hpp"
#include "ck_tile/ops/direct_convolution/kernel/impl/grouped_conv_output_writer.hpp"
#include "ck_tile/ops/direct_convolution/kernel/impl/non_grouped_conv_descriptors.hpp"
#include "ck_tile/ops/direct_convolution/kernel/impl/non_grouped_conv_compute_loop_v3.hpp"
#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/mfma.hpp"
#include "ck_tile/ops/direct_convolution/utils/config_map.hpp"
#include "ck_tile/ops/direct_convolution/utils/logging.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/load_tile.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

namespace ck_tile::direct_conv::conv_32c_tile::v3 {

constexpr int WAVE_SIZE = 64;

enum class MfmaShape
{
    M16N16K32,
    // M32N32K16 is retained as an enum value for config compatibility but is no
    // longer supported by the v3 dense kernel (rejected in is_valid_config; the
    // loaders/writers static_assert M16N16K32). No live instance used it and
    // M16N16K32 is the efficient shape.
    M32N32K16
};

// ===================================================================
// Config -- kernel configuration for v3 cross-wave LDS reduction.
//
// Parameters:
//   mfma_shape: M16N16K32
//   waves_per_group() = 1 (each wave is its own C-group)
//   block_groups() = waves_per_wg
//   channels_per_group() = mfma_k (32)
//   block_c() = waves_per_wg * channels_per_group
//   block_k_size() = mfma_m (16 K-output channels; A/C rows)
//   block_q() = mfma_n (16 or 32 spatial positions; B/C columns)
//   c_local_count() = waves_per_wg (one cpg slice per wave)
//
// MFMA dimension convention (matches hardware lane mapping):
//   weight operand = A: 16 (or 32) rows x K reduction
//                    -> row index = mfma_m -> K-output channel (Fprop) or
//                                            C-input channel (Dgrad)
//   input  operand = B: K reduction x 16 (or 32) columns
//                    -> column index = mfma_n -> spatial output position
//   accumulator    = C: rows x columns
//                    -> lane % 16 selects N (column = spatial),
//                      4 values per lane span M (rows = K-output).
// ===================================================================
template <DataType DT = DataType::fp16>
struct Config
{
    static constexpr DataType data_type = DT;
    MfmaShape mfma_shape                = MfmaShape::M16N16K32;
    int waves_per_wg;
    int kh     = 3;
    int kw     = 3;
    int n_fold = 8;

    // Number of channels_per_group C-chunks each wave streams through the
    // same fixed-size LDS buffers. Default 1 reproduces the legacy schedule.
    // For N > 1, block_c = waves_per_wg * N * cpg but LDS sizes stay at the
    // N=1 footprint; chunks ping-pong through the input double-buffer and
    // share the per-wave weight LDS region (loaded sequentially in prologue).
    // Restricted to mfma_shape == M16N16K32 (see static_assert in
    // TileConstants).
    int c_slices_per_wave = 1;

    // Derived from MFMA shape:
    constexpr int mfma_m() const { return (mfma_shape == MfmaShape::M16N16K32) ? 16 : -1; }
    constexpr int mfma_n() const { return (mfma_shape == MfmaShape::M16N16K32) ? 16 : -1; }
    constexpr int mfma_k() const { return (mfma_shape == MfmaShape::M16N16K32) ? 32 : -1; }

    constexpr int channels_per_group() const { return mfma_k(); }
    constexpr int group_size() const { return channels_per_group(); }
    constexpr int waves_per_group() const { return 1; }
    constexpr int block_groups() const { return waves_per_wg; }

    constexpr int num_waves() const { return waves_per_wg; }
    // Channels in a single in-flight LDS chunk = one input double-buffer
    // entry = one prologue iteration's weight LDS region.
    // INVARIANT: block_c() must not scale with c_slices_per_wave -- it is
    // consumed by TileConstantsBase (BLOCK_C8, INPUT_LDS_BUFFER_SIZE_*) and
    // Weight::WEIGHT_LDS_READ_K, which must stay fixed-size when N grows.
    constexpr int block_c() const { return channels_per_group() * block_groups(); }
    // Total C channels covered per workgroup across all chunks.
    constexpr int total_block_c() const { return block_c() * c_slices_per_wave; }
    // Spatial output positions per wave = MFMA N (columns of B/C).
    constexpr int block_q() const { return mfma_n(); }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }

    // K-output channels per wave = MFMA M (rows of A/C).
    // All waves share the same block_k_size K-channels.
    constexpr int block_k_size() const { return mfma_m(); }

    // Total C-sections per workgroup across all waves and chunks.
    constexpr int c_local_count() const { return block_groups() * c_slices_per_wave; }

    Direction direction = Direction::Fprop;

    SwizzleType swizzle_type = SwizzleType::None;
    EpilogueType epilogue    = EpilogueType::RegistersToGlobalMemory;
    int vector_size          = 8;

    std::string GetName() const
    {
        std::string mfma_str = (mfma_shape == MfmaShape::M32N32K16) ? "32x32x16" : "16x16x32";
        std::string swizzle_type_str = "_no_swizzle";
        if(swizzle_type == SwizzleType::CyclicShift)
        {
            swizzle_type_str = "_cyclic_shift_swizzle";
        }
        else if(swizzle_type == SwizzleType::XOR)
        {
            swizzle_type_str = "_xor_swizzle";
        }

        std::string base = "mfma_" + mfma_str + "_waves_per_wg_" + std::to_string(waves_per_wg) +
                           swizzle_type_str + "_cross_wave_lds_reduce";

        std::string epilogue_suffix;
        if(epilogue == EpilogueType::RegistersToGlobalMemory)
        {
            epilogue_suffix = "_direct_dram_epilogue";
        }
        else if(epilogue == EpilogueType::RegistersToLdsToGlobalMemory)
        {
            epilogue_suffix = "_lds_staged_epilogue";
        }

        std::string cspw_suffix = "_cspw_" + std::to_string(c_slices_per_wave);

        return base + epilogue_suffix + cspw_suffix;
    }
};

// ===================================================================
// weight_load_to_lds_kyxc -- load one c_slice of KYXC weights to LDS.
//
// Loads weight[block_k_start : +block_k_size, :, c_slice*cpg : +cpg]
// from KYXC DRAM layout into contiguous [block_k_size, KH_KW, cpg] LDS.
// cpg = channels_per_group (32 for M16N16K32).
// ===================================================================
template <auto cfg, typename ElementType = _Float16>
CK_TILE_DEVICE void weight_load_to_lds_kyxc(uint4* weight_lds,
                                            const ElementType* __restrict__ wei,
                                            int block_k_start,
                                            int c_slice,
                                            int C_total)
{
    constexpr int WEIGHT_K    = cfg.block_k_size();
    constexpr int KH_KW       = cfg.kh * cfg.kw;
    constexpr int C_SLICE     = cfg.channels_per_group();
    constexpr int TOTAL_UINT4 = WEIGHT_K * KH_KW * C_SLICE / 8;
    constexpr int NUM_PASSES  = (TOTAL_UINT4 + cfg.block_size() - 1) / cfg.block_size();

    const int tid      = static_cast<int>(threadIdx.x);
    const int K_stride = KH_KW * C_total;

    for(int pass = 0; pass < NUM_PASSES; pass++)
    {
        int flat_idx = pass * cfg.block_size() + tid;
        if(flat_idx < TOTAL_UINT4)
        {
            // Decompose: LDS layout is [WEIGHT_K, KH_KW, C_SLICE], C innermost.
            // Each uint4 = 8 fp16 values in the C dimension.
            constexpr int C_UINT4 = C_SLICE / 8; // 4
            int c8                = flat_idx % C_UINT4;
            int temp              = flat_idx / C_UINT4;
            int filter            = temp % KH_KW;
            int k                 = temp / KH_KW;

            const ElementType* src = wei + static_cast<size_t>(block_k_start + k) * K_stride +
                                     filter * C_total + c_slice * C_SLICE + c8 * 8;

            weight_lds[flat_idx] = *reinterpret_cast<const uint4*>(src);
        }
    }
}

// ===================================================================
// weight_load_to_lds_kyxc_dgrad -- load one k_slice of KYXC weights
// for Dgrad into LDS.
//
// Loads weight[k_slice_start : +cpg, :, block_c_start : +block_k_size]
// from KYXC DRAM layout into contiguous [cpg_K, KH_KW, block_k_size_C] LDS.
//
// For Dgrad: K dimension = channels_per_group (MFMA reduction),
// C dimension = block_k_size (output channels of input gradient).
// ===================================================================
template <auto cfg, typename ElementType = _Float16>
CK_TILE_DEVICE void weight_load_to_lds_kyxc_dgrad(uint4* weight_lds,
                                                  const ElementType* __restrict__ wei,
                                                  int k_slice_start,
                                                  int block_c_start,
                                                  int C_total)
{
    constexpr int K_SLICE = cfg.channels_per_group();
    constexpr int KH_KW   = cfg.kh * cfg.kw;
    constexpr int BLOCK_C = cfg.block_k_size();
    // Total uint4s = K_SLICE * KH_KW * BLOCK_C / 8 = same as Fprop weight LDS size.
    constexpr int TOTAL_UINT4 = K_SLICE * KH_KW * BLOCK_C / 8;
    constexpr int NUM_PASSES  = (TOTAL_UINT4 + cfg.block_size() - 1) / cfg.block_size();

    const int tid      = static_cast<int>(threadIdx.x);
    const int K_stride = KH_KW * C_total;

    for(int pass = 0; pass < NUM_PASSES; pass++)
    {
        int flat_idx = pass * cfg.block_size() + tid;
        if(flat_idx < TOTAL_UINT4)
        {
            // Decompose: LDS layout is [K_SLICE, KH_KW, BLOCK_C], C innermost.
            constexpr int C_UINT4 = BLOCK_C / 8;
            int c8                = flat_idx % C_UINT4;
            int temp              = flat_idx / C_UINT4;
            int filter            = temp % KH_KW;
            int k                 = temp / KH_KW;

            const ElementType* src = wei + static_cast<size_t>(k_slice_start + k) * K_stride +
                                     filter * C_total + block_c_start + c8 * 8;

            weight_lds[flat_idx] = *reinterpret_cast<const uint4*>(src);
        }
    }
}

// ===================================================================
// is_valid_config -- config compatibility check for v3.
// ===================================================================
template <DataType DT = DataType::fp16>
inline bool is_valid_config(const Conv2dParams& par, const Config<DT>& cfg)
{
    if(par.direction != cfg.direction)
    {
        LogInfo("Direction mismatch: conv direction != config direction, ",
                " conv direction = ",
                std::move(to_string(par.direction)),
                ", config direction = ",
                std::move(to_string(cfg.direction)));
        return false;
    }

    if(par.kh != cfg.kh || par.kw != cfg.kw)
    {
        LogInfo("Kernel size mismatch: conv kh/kw != config kh/kw");
        return false;
    }

    // M32N32K16 was dropped from the v3 dense kernel (no live instance used it;
    // M16N16K32 is the efficient shape). Reject any M32N32K16 config cleanly.
    if(cfg.mfma_shape != MfmaShape::M16N16K32)
    {
        LogInfo("v3 dense kernel supports only MfmaShape::M16N16K32");
        return false;
    }

    // C_in covering rule: the workgroup reduces total_block_c() channels across
    // its waves. With channel padding, the real C_in may be smaller — but it
    // must land in (total_block_c() - cpg, total_block_c()] so that exactly this
    // config minimally covers C_in (the last wave's cpg-slice is the partial
    // one, and no fully-empty wave exists). Phase 2 supports any C_in within
    // that range, including counts that are not multiples of 8 (sub-8): the
    // straddling channel tile is loaded and zeroed at element granularity by
    // the weight loader.
    const int C_in = (cfg.direction == Direction::Dgrad) ? par.k_tot : par.c_tot;
    if(C_in > cfg.total_block_c() || C_in <= cfg.total_block_c() - cfg.channels_per_group())
    {
        LogInfo("Input channel not covered by config: ",
                " C_in = ",
                std::move(std::to_string(C_in)),
                ", config.total_block_c() = ",
                std::move(std::to_string(cfg.total_block_c())));
        return false;
    }

    // The output-channel count (k_tot for Fprop, c_tot for Dgrad) may be any
    // positive value; partial block_k_size tiles — down to sub-8 — are masked
    // at element granularity at output time and need no config restriction.

    // Channel padding splits into two INDEPENDENT cases with different needs:
    //
    //   - Reduction-channel padding (C_in < total_block_c()): correctness comes
    //     from the WEIGHT loader, which zeroes every invalid reduction channel
    //     (garbage_input * 0_weight == 0 in the MFMA) using the GLOBAL channel
    //     index c_slice * cpg + c8*8 (c_slice = CS * waves + wave_id), so it is
    //     already chunk-aware. The input loader additionally masks, per thread
    //     and per chunk CS, the SWIZZLED physical channel-8 block it loads
    //     (global_c8 = CS * BLOCK_C8 + swizzled_c8(global_spatial, logical_c8))
    //     when that block lands at or beyond ceil(C_in/8), purely to avoid
    //     reading garbage. Both the validity check and the weight zeroing derive
    //     from the same global channel coordinate, so they compose with any
    //     swizzle AND any c_slices_per_wave. The only remaining requirement is
    //     the regular {BLOCK_C8, LANES_PER_ROW} lane decomposition, i.e.
    //     64 % BLOCK_C8 == 0.
    //
    //   - Output-channel padding (K_out % block_k_size != 0): handled entirely
    //     by element-precise masking in the output writer plus K-row zeroing in
    //     the weight loader. Both are independent of c_slices_per_wave and the
    //     swizzle, so NO extra restriction applies. When the reduction is not
    //     padded (C_in == total_block_c()) the input-loader channel mask is a
    //     no-op (main_c8 < ceil(C_in/8) always), so swizzled / multi-slice
    //     configs are safe.
    const bool reduction_padded = (C_in != cfg.total_block_c());
    if(reduction_padded)
    {
        const int block_c8 = cfg.block_c() / 8;
        if(64 % block_c8 != 0)
        {
            LogInfo("Reduction-padded config requires 64 %% (block_c/8) == 0");
            return false;
        }
    }

    if(!swizzle_config_valid(cfg, par))
    {
        return false;
    }

    return true;
}

template <auto cfg>
inline LaunchParams get_launch_params(const Conv2dParams& par)
{
    return get_launch_params_non_grouped(cfg, par);
}

// ===================================================================
// TileConstants -- extends TileConstantsBase for v3.
//
// Weight LDS is sized for [block_k_size, KH*KW, cpg] -- one c_slice.
// block_k_size x cpg = 512 for both MFMA shapes.
// ===================================================================
template <auto cfg>
struct TileConstants : direct_conv::TileConstantsBase<cfg>
{
    using Base = direct_conv::TileConstantsBase<cfg>;

    // c_slices_per_wave > 1 is only supported for the 16x16x32 MFMA path
    // for now (the 32x32x16 path stays at N=1).
    static_assert(cfg.c_slices_per_wave >= 1, "c_slices_per_wave must be >= 1");
    static_assert(cfg.c_slices_per_wave == 1 || cfg.mfma_shape == MfmaShape::M16N16K32,
                  "c_slices_per_wave > 1 is only supported for M16N16K32 configs");

    static constexpr int WAVES_PER_WG = cfg.waves_per_wg;
    static constexpr int KH_KW_       = cfg.kh * cfg.kw;

    // Output-channel block size (K rows of the MFMA accumulator). Used by
    // DenseSharedDescriptors<TC>::Output::MakeChannelPadDescriptor.
    static constexpr int BLOCK_K = cfg.block_k_size();

    // Weight LDS sizing for one c_slice: [block_k_size, KH_KW, cpg].
    // Both MFMA shapes: block_k_size * cpg = 512, so 512 * 9 = 4,608 elements.
    static constexpr int WEIGHT_LDS_ELEMENTS =
        cfg.block_k_size() * cfg.kh * cfg.kw * cfg.channels_per_group();
    static constexpr int WEIGHT_LDS_SIZE_UINT4 = WEIGHT_LDS_ELEMENTS / 8;
    // Per-wave parallel loading: each wave owns its own LDS region.
    static constexpr int WEIGHT_LDS_ALL_WAVES = WEIGHT_LDS_SIZE_UINT4 * cfg.waves_per_wg;

    // Mfma distribution -- needed by InputLoader's static type declarations.
    // Not used at runtime (ConvInputLoader passes init_mfma_offsets=false).
    // The distribution must be well-formed for type deduction to compile.
    // Input is the MFMA B operand: spatial position = N (columns of B/C).
    static constexpr int SPATIAL_LANES = cfg.mfma_n();       // 16 or 32
    static constexpr int K_GROUPS      = 64 / SPATIAL_LANES; // 4 or 2
    struct Mfma
    {
        static constexpr auto MakeAccTileDistribution()
        {
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<
                    ck_tile::sequence<>,
                    ck_tile::tuple<ck_tile::sequence<SPATIAL_LANES>,
                                   ck_tile::sequence<WAVES_PER_WG, K_GROUPS>,
                                   ck_tile::sequence<4>>,
                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 1>>,
                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                    ck_tile::sequence<3>,
                    ck_tile::sequence<0>>{});
        }
    };

    // Weight -- only LDS sizing overrides needed. Weight reads use manual
    // addressing (not tile distributions).
    struct Weight : Base::Weight
    {
    };
};

// ===================================================================
// BlockCoords -- reuse non-grouped BlockCoords from v1.
// ===================================================================
template <auto cfg>
using ConvBlockCoordsT = direct_conv::BlockCoordsNonGrouped<cfg>;

// ===================================================================
// ConvInputLoader -- extends InputLoader.
//
// Key properties:
//   1. wave_group = wave (not wave / 2): each wave is its own C-group.
//   2. Supplementary overflow loads for spatial positions beyond
//      DIST_SPATIAL. The tile distribution covers DIST_SPATIAL =
//      NUM_WAVES * LANES_PER_ROW positions (which may be less than
//      TOTAL_SPATIAL when 64 % BLOCK_C8 != 0, e.g. waves=6).
//      BLOCK_W = BLOCK_Q + kw - 1. The extra positions from
//      DIST_SPATIAL to BLOCK_W-1 are loaded by the first
//      OVERFLOW_COUNT threads.
// ===================================================================
template <auto cfg, bool Padded = false>
struct ConvInputLoader
    : direct_conv::InputLoader<
          TileConstants<cfg>,
          cfg,
          std::conditional_t<cfg.data_type == DataType::bf16, ck_tile::bf16x8_t, ck_tile::fp16x8_t>,
          false,
          ToType<cfg.data_type>>
{
    using ElementType = ToType<cfg.data_type>;
    using base        = direct_conv::InputLoader<
               TileConstants<cfg>,
               cfg,
               std::conditional_t<cfg.data_type == DataType::bf16, ck_tile::bf16x8_t, ck_tile::fp16x8_t>,
               false,
               ElementType>;
    using TC         = TileConstants<cfg>;
    using input_type = typename base::input_type;

    // Number of spatial positions actually covered by the tile distribution.
    // The distribution maps NUM_WAVES * LANES_PER_ROW positions. When
    // BLOCK_C8 doesn't divide 64 (waves=3,5,6,7), this is less than
    // TOTAL_SPATIAL because some lanes are excess. The overflow loader must
    // handle all positions from DIST_SPATIAL to BLOCK_W-1.
    static constexpr int DIST_SPATIAL = TC::NUM_WAVES * TC::LANES_PER_ROW;

    // Number of extra tile positions beyond DIST_SPATIAL that must be loaded
    // by the overflow path.
    static constexpr int OVERFLOW_COUNT = (TC::BLOCK_W - DIST_SPATIAL) * TC::BLOCK_C8;

    // Byte offset added to input_voffset / overflow_voffset to point at chunk
    // CS of the current input row (constant across rows). For c_slices_per_wave
    // == 1 this is always zero.
    static constexpr ck_tile::index_t CHUNK_VOFFSET_STRIDE =
        static_cast<ck_tile::index_t>(cfg.waves_per_wg) *
        static_cast<ck_tile::index_t>(cfg.channels_per_group()) *
        static_cast<ck_tile::index_t>(sizeof(ElementType));

    // State for overflow loads.
    ck_tile::index_t overflow_voffset;
    CK_TILE_LDS_ADDR ElementType* overflow_lds_dest;
    ck_tile::index_t overflow_is_valid;
    bool overflow_active;

    // Reduction-channel pad state (only meaningful when Padded). The swizzled
    // channel-8 block this thread loads WITHIN a chunk is pad_*_c8_dram; for
    // chunk CS the global channel-8 index is CS * BLOCK_C8 + pad_*_c8_dram, and
    // the block is masked out when that index reaches pad_c8_ceil = ceil(C_in/8).
    // Evaluated per chunk in prefetch_tile_to_lds so it composes with cspw > 1.
    int pad_main_c8_dram = 0;
    int pad_ov_c8_dram   = 0;
    int pad_c8_ceil      = 0;

    template <typename BlockCoords_>
    CK_TILE_DEVICE ConvInputLoader(const BlockCoords_& bc,
                                   uint4* input_lds,
                                   const ElementType* __restrict__ in,
                                   int hi,
                                   int wi,
                                   int px,
                                   int py,
                                   int dx,
                                   int dy,
                                   int sx,
                                   int sy)
        : base(bc,
               input_lds,
               in,
               hi,
               wi,
               px,
               py,
               dx,
               dy,
               sx,
               sy,
               TC::GROUP_SIZE,
               /*init_mfma_offsets=*/false)
    {
        // The base ctor sizes input_rsrc from the dram descriptor's
        // element-space, which is built with BLOCK_C8 (per-chunk) on the
        // C axis. That clamps the buffer's hardware bounds-check to a
        // single chunk's worth of channels at the last (h, w), so any
        // chunk CS > 0 load that targets the LAST real input column gets
        // zeroed by the OOB clamp. Re-make the rsrc to span the full
        // per-batch input tensor (N batches x hi x wi x C), which is the
        // largest extent any chunk's voffset can reach.
        //
        // We also bypass this re-make when N == 1 (no chunks beyond CS=0,
        // base ctor's sizing is already correct).
        if constexpr(cfg.c_slices_per_wave > 1)
        {
            const ElementType* input_base =
                in + static_cast<size_t>(bc.block_n) * hi * wi * bc.C + bc.block_k;
            const size_t rsrc_bytes = static_cast<size_t>(hi) * wi * bc.C * sizeof(ElementType);
            base::input_rsrc        = ck_tile::make_builtin_buffer_resource(
                input_base, static_cast<uint32_t>(rsrc_bytes));
        }

        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int wave = static_cast<int>(threadIdx.x) / WAVE_SIZE;

        // Input = MFMA B operand: lane % mfma_n -> column (spatial position),
        // lane / mfma_n -> K-reduction group.
        constexpr int MFMA_N = cfg.mfma_n();
        const int lane_q     = lane % MFMA_N;
        const int lane_c8    = lane / MFMA_N;

        // v3: each wave is its own C-group (wave_group = wave, not wave / 2).
        const int wave_group = wave;
        const int c8_pos     = wave_group * TC::GROUP_SIZE_8 + lane_c8;

        // The DRAM swizzle is applied to the global wi_padded coordinate
        // (block_q + spatial_pos), so the inverse used to find the LDS slot must
        // also include block_q. Otherwise the inverse is wrong whenever block_q
        // is not a multiple of BLOCK_C8.
        //
        // Express the inverse swizzle as a descriptor transform: build a
        // [spatial, BLOCK_C8, 8] element-strided LDS view with the inverse
        // swizzle, evaluate at the global spatial index, then subtract the
        // block_q base so the result is the tile-local LDS element offset.
        const int spatial_len = bc.block_q + TC::BLOCK_W;
        const auto lds_read_desc =
            direct_conv::DenseSharedDescriptors<TC>::Input::MakeLdsReadDescriptor(spatial_len);
        const ck_tile::index_t block_q_base =
            static_cast<ck_tile::index_t>(bc.block_q) * TC::BLOCK_C8 * 8;
        ck_tile::static_for<0, cfg.kw, 1>{}([&](auto s_n) {
            constexpr int S = s_n.value;
            int spatial_pos = lane_q + S;
            int global      = bc.block_q + spatial_pos;
            auto coord      = ck_tile::make_tensor_coordinate(
                lds_read_desc, ck_tile::make_multi_index(global, c8_pos, 0));
            base::mfma_lds_offsets[S] = coord.get_offset() - block_q_base;
        });

        // Forward-swizzle descriptor: maps (global_spatial, logical_c8) to the
        // physical channel-8 index that the load actually targets in DRAM. Used
        // both to compute the overflow DRAM offset and to make the reduction-pad
        // validity check swizzle-aware (see below). For SwizzleType::None this is
        // the identity on c8.
        auto fwd_swz_desc =
            direct_conv::DenseSharedDescriptors<TC>::Input::MakeForwardSwizzleDescriptor(
                bc.block_q + TC::BLOCK_W);
        auto swizzled_c8 = [&](int global_spatial, int logical_c8) {
            return static_cast<int>(
                ck_tile::make_tensor_coordinate(
                    fwd_swz_desc, ck_tile::make_multi_index(global_spatial, logical_c8))
                    .get_offset());
        };

        // --- Overflow load setup ---
        // The tile distribution covers DIST_SPATIAL spatial positions, but
        // we need BLOCK_W positions. The first OVERFLOW_COUNT threads each
        // handle one extra (spatial, c8) tile position.
        const int tid   = static_cast<int>(threadIdx.x);
        overflow_active = (tid < OVERFLOW_COUNT);
        if(overflow_active)
        {
            const int ov_spatial = DIST_SPATIAL + tid / TC::BLOCK_C8;
            const int ov_c8      = tid % TC::BLOCK_C8;

            // LDS destination for overflow position.
            auto* lds_base    = reinterpret_cast<CK_TILE_LDS_ADDR ElementType*>(input_lds);
            overflow_lds_dest = lds_base + ov_spatial * TC::BLOCK_C8 * 8 + ov_c8 * 8;

            // DRAM offset: input_x = block_q + ov_spatial - px (padding offset).
            const int input_x = bc.block_q + ov_spatial - px;
            overflow_is_valid = (input_x >= 0 && input_x < wi) ? 1 : 0;
            // Match the DRAM descriptor's global-coordinate swizzle so the
            // overflow DRAM read is consistent with the main load path when
            // block_q is not a multiple of BLOCK_C8.
            const int ov_c8_dram = swizzled_c8(bc.block_q + ov_spatial, ov_c8);
            overflow_voffset = static_cast<ck_tile::index_t>((input_x * bc.C + ov_c8_dram * 8) *
                                                             static_cast<int>(sizeof(ElementType)));

            // Reduction-channel padding: record the swizzled channel-8 block this
            // thread loads WITHIN a chunk (ov_c8_dram). The actual masking is done
            // per chunk CS in prefetch_tile_to_lds, where the global channel-8
            // index CS * BLOCK_C8 + ov_c8_dram is compared against ceil(C_in/8).
            // Computing it per chunk (rather than zeroing once here) is what lets
            // reduction padding compose with c_slices_per_wave > 1. Composing the
            // check on the SWIZZLED block also makes it work with any swizzle.
            //
            // Sub-8 (Phase 2): the partial block that straddles C_in (channels
            // [floor(C_in/8)*8, +8) where some are < C_in and some are not) is
            // still LOADED — its garbage lanes are multiplied by zeroed weights
            // in the MFMA, so they contribute nothing. Only fully-out-of-range
            // blocks (>= ceil(C_in/8)) are masked. The input read is hardware
            // bounds-checked (buffer resource), so loading the partial block is
            // always safe.
            if constexpr(Padded)
                pad_ov_c8_dram = ov_c8_dram;
        }
        else
        {
            overflow_voffset  = 0;
            overflow_lds_dest = nullptr;
            overflow_is_valid = 0;
        }

        // Reduction-channel padding for the main load: record the swizzled
        // channel-8 block this thread loads WITHIN a chunk. With 64 % BLOCK_C8 == 0
        // the tile distribution decomposes lane as {BLOCK_C8, LANES_PER_ROW}, so
        // the thread's logical c8 is tid % BLOCK_C8 and its global spatial position
        // is block_q + tid / BLOCK_C8. The block it physically loads is the
        // swizzled index, so the pad check (done per chunk CS in prefetch) is
        // swizzle-aware and chunk-aware: it masks when CS * BLOCK_C8 + main_c8_dram
        // reaches ceil(C_in/8). Sub-8: keep the straddling partial block (ceil),
        // zeroed by weights.
        if constexpr(Padded)
        {
            const int main_c8  = tid % TC::BLOCK_C8;
            const int main_global = bc.block_q + tid / TC::BLOCK_C8;
            pad_main_c8_dram   = swizzled_c8(main_global, main_c8);
            pad_c8_ceil        = (bc.C + 7) / 8; // ceil(C_in / 8)
        }
    }

    // Override prefetch: base load (16 spatial positions) + overflow load (2 extra).
    //
    // CS selects which C-chunk of the CURRENT input row is loaded. CS = 0
    // matches the legacy path; CS > 0 adds CS * (waves * cpg) channels' worth
    // of bytes to the per-thread DRAM offset, leaving the row position alone.
    // Does NOT advance input_voffset / overflow_voffset.
    template <int CS = 0>
    CK_TILE_DEVICE void prefetch_tile_to_lds(int lds_buffer_index)
    {
        static_assert(CS >= 0 && CS < cfg.c_slices_per_wave, "CS out of range");
        constexpr ck_tile::index_t chunk_off = CS * CHUNK_VOFFSET_STRIDE;

        // Reduction-channel pad masks are per chunk: the global channel-8 block
        // this thread loads for chunk CS is CS * BLOCK_C8 + pad_*_c8_dram, masked
        // out when it reaches pad_c8_ceil = ceil(C_in/8). For Padded==false these
        // collapse to the unmodified base validity.
        ck_tile::index_t main_valid = base::is_valid;
        ck_tile::index_t ov_valid   = overflow_is_valid;
        if constexpr(Padded)
        {
            if(CS * TC::BLOCK_C8 + pad_main_c8_dram >= pad_c8_ceil)
                main_valid = 0;
            if(CS * TC::BLOCK_C8 + pad_ov_c8_dram >= pad_c8_ceil)
                ov_valid = 0;
        }

        if(base::load_active)
        {
            CK_TILE_LDS_ADDR ElementType* lds_dest =
                base::store_input_lds + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;

            buffer_load16_to_lds<ElementType>(
                lds_dest, base::input_rsrc, base::input_voffset + chunk_off, main_valid);
        }

        if(overflow_active)
        {
            CK_TILE_LDS_ADDR ElementType* lds_dest =
                overflow_lds_dest + lds_buffer_index * TC::INPUT_LDS_BUFFER_SIZE_FP16;

            buffer_load16_to_lds<ElementType>(
                lds_dest, base::input_rsrc, overflow_voffset + chunk_off, ov_valid);
        }
    }

    // Override fetch: advance both offsets to the next row, then prefetch
    // chunk CS of that row.
    template <int CS = 0>
    CK_TILE_DEVICE void fetch_tile_to_lds(int lds_buffer_index)
    {
        if(base::load_active)
            base::input_voffset += base::row_stride_bytes;
        if(overflow_active)
            overflow_voffset += base::row_stride_bytes;
        prefetch_tile_to_lds<CS>(lds_buffer_index);
    }
};

// ===================================================================
// WeightLoader -- weight accessor for v3.
//
// Weight LDS layout for one c_slice: [16_K, KH_KW, 32_C].
// All waves read the same weight data from LDS (same 16 K-channels),
// but each wave's MFMA pairs these with a different C-section of input.
//
// The DRAM -> LDS load functions are reused from v1 unchanged.
// ===================================================================
template <auto cfg, bool Padded = false>
struct WeightLoader : direct_conv::WeightAccessor8<
                          cfg.kh,
                          cfg.kw,
                          std::conditional_t<cfg.data_type == DataType::bf16, bf16x8_t, fp16x8_t>,
                          cfg.c_slices_per_wave>
{
    using TC          = TileConstants<cfg>;
    using ElementType = ToType<cfg.data_type>;

    // Load one c_slice of weights from KYXC DRAM into weight LDS (Fprop).
    CK_TILE_DEVICE static void load_kyxc_to_lds(uint4* weight_lds,
                                                const ElementType* __restrict__ wei,
                                                int block_k_start,
                                                int c_slice,
                                                int C_total)
    {
        weight_load_to_lds_kyxc<cfg, ElementType>(weight_lds, wei, block_k_start, c_slice, C_total);
    }

    // Load one k_slice of weights from KYXC DRAM into weight LDS (Dgrad).
    CK_TILE_DEVICE static void load_kyxc_to_lds_dgrad(uint4* weight_lds,
                                                      const ElementType* __restrict__ wei,
                                                      int k_slice_start,
                                                      int block_c_start,
                                                      int C_total)
    {
        weight_load_to_lds_kyxc_dgrad<cfg, ElementType>(
            weight_lds, wei, k_slice_start, block_c_start, C_total);
    }

    // Wave-local Fprop load: only this wave's 64 threads load its own c_slice
    // into its private LDS region (weight_lds + wave_id * WEIGHT_LDS_SIZE_UINT4).
    // All waves call this simultaneously with no synchronization required.
    CK_TILE_DEVICE static void
    load_kyxc_to_lds_wave(uint4* wave_lds, // base of this wave's LDS region
                          const ElementType* __restrict__ wei,
                          int block_k_start,
                          int c_slice,
                          int C_total,
                          int K_total)
    {
        constexpr int WEIGHT_K   = cfg.block_k_size();
        constexpr int KH_KW      = cfg.kh * cfg.kw;
        constexpr int C_SLICE    = cfg.channels_per_group();
        constexpr int TOTAL_U4   = WEIGHT_K * KH_KW * C_SLICE / 8;
        constexpr int NUM_PASSES = (TOTAL_U4 + WAVE_SIZE - 1) / WAVE_SIZE;

        const int lane     = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int K_stride = KH_KW * C_total;

        for(int pass = 0; pass < NUM_PASSES; pass++)
        {
            int flat_idx = pass * WAVE_SIZE + lane;
            if(flat_idx < TOTAL_U4)
            {
                constexpr int C_UINT4 = C_SLICE / 8;
                int c8                = flat_idx % C_UINT4;
                int temp              = flat_idx / C_UINT4;
                int filter            = temp % KH_KW;
                int k                 = temp / KH_KW;

                const ElementType* src = wei + static_cast<size_t>(block_k_start + k) * K_stride +
                                         filter * C_total + c_slice * C_SLICE + c8 * 8;

                // Channel padding.
                //   K dim (output): block_k_start + k selects a full weight row;
                //     a row >= K_total is OOB — zero the whole uint4.
                //   C dim (reduction): c_base..c_base+7 are reduction channels.
                //     For sub-8 (Phase 2) the last uint4 straddles C_total: load
                //     the valid lanes and zero the rest (element-wise), which both
                //     avoids the OOB DRAM read and zeroes the invalid reduction
                //     weights so their MFMA contribution is exactly 0.
                if constexpr(Padded)
                {
                    const int k_global = block_k_start + k;
                    const int c_base   = c_slice * C_SLICE + c8 * 8;
                    if(k_global >= K_total)
                    {
                        wave_lds[flat_idx] = uint4{0, 0, 0, 0};
                        continue;
                    }
                    if(c_base + 8 > C_total)
                    {
                        ElementType vals[8];
                        ck_tile::static_for<0, 8, 1>{}([&](auto j_n) {
                            constexpr int J = j_n.value;
                            vals[J] = (c_base + J < C_total) ? src[J] : ElementType{0};
                        });
                        __builtin_memcpy(&wave_lds[flat_idx], vals, sizeof(uint4));
                        continue;
                    }
                }

                wave_lds[flat_idx] = *reinterpret_cast<const uint4*>(src);
            }
        }
    }

    // Wave-local Dgrad load: only this wave's 64 threads load its own k_slice
    // into its private LDS region.
    CK_TILE_DEVICE static void
    load_kyxc_to_lds_dgrad_wave(uint4* wave_lds, // base of this wave's LDS region
                                const ElementType* __restrict__ wei,
                                int k_slice_start,
                                int block_c_start,
                                int C_total,
                                int K_total)
    {
        constexpr int K_SLICE    = cfg.channels_per_group();
        constexpr int KH_KW      = cfg.kh * cfg.kw;
        constexpr int BLOCK_C    = cfg.block_k_size();
        constexpr int TOTAL_U4   = K_SLICE * KH_KW * BLOCK_C / 8;
        constexpr int NUM_PASSES = (TOTAL_U4 + WAVE_SIZE - 1) / WAVE_SIZE;

        const int lane     = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int K_stride = KH_KW * C_total;

        for(int pass = 0; pass < NUM_PASSES; pass++)
        {
            int flat_idx = pass * WAVE_SIZE + lane;
            if(flat_idx < TOTAL_U4)
            {
                constexpr int C_UINT4 = BLOCK_C / 8;
                int c8                = flat_idx % C_UINT4;
                int temp              = flat_idx / C_UINT4;
                int filter            = temp % KH_KW;
                int k                 = temp / KH_KW;

                const ElementType* src = wei + static_cast<size_t>(k_slice_start + k) * K_stride +
                                         filter * C_total + block_c_start + c8 * 8;

                // Channel padding: the weight tensor is physically
                // [K_total, KH, KW, C_total]. For Dgrad the reduction maps to the
                // weight K dim (k_slice_start+k, a full row) and the output maps
                // to the weight C dim (block_c_start+c8*8, within the uint4).
                //   K dim (reduction): a row >= K_total is OOB and an invalid
                //     reduction channel — zero the whole uint4 (MFMA contribution
                //     0).
                //   C dim (output): for sub-8 (Phase 2) the last uint4 straddles
                //     C_total — load the valid lanes and zero the rest to avoid
                //     the OOB DRAM read (the invalid output lanes are also masked
                //     at write time).
                if constexpr(Padded)
                {
                    const int k_global = k_slice_start + k;
                    const int c_base   = block_c_start + c8 * 8;
                    if(k_global >= K_total)
                    {
                        wave_lds[flat_idx] = uint4{0, 0, 0, 0};
                        continue;
                    }
                    if(c_base + 8 > C_total)
                    {
                        ElementType vals[8];
                        ck_tile::static_for<0, 8, 1>{}([&](auto j_n) {
                            constexpr int J = j_n.value;
                            vals[J] = (c_base + J < C_total) ? src[J] : ElementType{0};
                        });
                        __builtin_memcpy(&wave_lds[flat_idx], vals, sizeof(uint4));
                        continue;
                    }
                }

                wave_lds[flat_idx] = *reinterpret_cast<const uint4*>(src);
            }
        }
    }

    // Read weights from LDS into the register cache slot for chunk CS.
    //
    // Uses manual addressing for both Fprop and Dgrad. The tile distribution
    // approach from v1 is not applicable because v3's weight LDS only holds
    // [16_K, KH_KW, 32_C] (one c_slice), whereas the tile distribution
    // expects [block_c, KH_KW, 32] which assumes the wave dimension
    // partitions the K dimension.
    //
    // All waves read the same weight data (same 16 K-channels). Each wave
    // pairs this with a different C-section of the input during MFMA.
    //
    // For c_slices_per_wave > 1, the prologue calls this once per chunk CS
    // with that chunk's weight data in LDS; the per-chunk register slot is
    // weights[F * N + CS].
    template <int CS = 0>
    CK_TILE_DEVICE void read_from_lds_chunk(uint4* weight_lds)
    {
        static_assert(CS >= 0 && CS < cfg.c_slices_per_wave, "CS out of range");
        constexpr int KH_KW_L = cfg.kh * cfg.kw;
        constexpr int N_      = cfg.c_slices_per_wave;
        const int lane        = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const auto* lds_ptr   = reinterpret_cast<const ElementType*>(weight_lds);
        using VecType         = typename std::remove_reference_t<decltype(*this)>::value_type;

        if constexpr(cfg.direction == Direction::Dgrad)
        {
            // Dgrad: LDS layout is [cpg_K, KH_KW, block_k_size_C].
            // Each thread reads 8 K-reduction values per filter position.
            //
            // MFMA A operand mapping:
            //   k_group = lane / mfma_m -> selects which 8 K-reduction values
            //   c_lane  = lane % mfma_m -> C-output position
            constexpr int BLOCK_C = cfg.block_k_size();
            constexpr int MFMA_M  = cfg.mfma_m();

            const int k_group = lane / MFMA_M;
            const int c_lane  = lane % MFMA_M;

            ck_tile::static_for<0, KH_KW_L, 1>{}([&](auto f_n) {
                constexpr int F = f_n.value;
                ElementType vals[8];
                ck_tile::static_for<0, 8, 1>{}([&](auto j_n) {
                    constexpr int J = j_n.value;
                    int k           = k_group * 8 + J;
                    vals[J]         = lds_ptr[k * KH_KW_L * BLOCK_C + F * BLOCK_C + c_lane];
                });
                __builtin_memcpy(&this->weights[F * N_ + CS], vals, sizeof(VecType));
            });
        }
        else
        {
            // Fprop: LDS layout is [block_k_size_K, KH_KW, cpg_C], C innermost.
            //
            // MFMA A operand mapping (weight is A; row index = M dimension):
            //   k_out = lane % mfma_m -> K-output channel (row of A)
            //   c_grp = lane / mfma_m -> C-reduction group (each has 8 values)
            //
            // Each thread reads 8 C values per filter position:
            //   vals[j] = weight[k_out, f, c_grp*8 + j]
            constexpr int C_SLICE = cfg.channels_per_group();
            constexpr int MFMA_M  = cfg.mfma_m();

            const int k_out = lane % MFMA_M;
            const int c_grp = lane / MFMA_M;

            ck_tile::static_for<0, KH_KW_L, 1>{}([&](auto f_n) {
                constexpr int F = f_n.value;
                ElementType vals[8];
                ck_tile::static_for<0, 8, 1>{}([&](auto j_n) {
                    constexpr int J = j_n.value;
                    vals[J] = lds_ptr[k_out * KH_KW_L * C_SLICE + F * C_SLICE + c_grp * 8 + J];
                });
                __builtin_memcpy(&this->weights[F * N_ + CS], vals, sizeof(VecType));
            });
        }
    }

    // Legacy single-chunk entry: equivalent to read_from_lds_chunk<0>().
    // Used by the v3 single-buffer compute loop and by the N=1 path.
    CK_TILE_DEVICE void read_from_lds(uint4* weight_lds) { read_from_lds_chunk<0>(weight_lds); }
};

// ===================================================================
// OutputWriterV3 -- manual offset computation for v3.
//
// In v3, all waves share the same block_k_size K-channels. Only wave 0
// writes the output after cross-wave LDS reduction.
//
// M16N16K32 only (M32N32K16 was dropped — no live instance used it and the
// efficient shape is M16N16K32): lane % 16 → spatial, lane / 16 → K-group
// (4 groups × 4 K). Single 8B DRAM write per thread (unpadded).
//
// Output-channel padding is expressed as a CK Tile pad transform via
// DenseSharedDescriptors<TC>::Output::MakeChannelPadDescriptor. Each thread
// writes a 4-K group [k_off, k_off+4) whose channels are contiguous and
// increasing; the descriptor's pad validity yields valid_k_count, the number
// of in-range channels in this group (a prefix, since the group is
// contiguous). The hot flush path then writes exactly valid_k_count elements
// with no per-element bound check.
// ===================================================================
template <auto cfg, bool Padded = false>
struct OutputWriterV3
{
    static_assert(cfg.mfma_shape == MfmaShape::M16N16K32,
                  "OutputWriterV3 supports only MfmaShape::M16N16K32");

    using ElementType = ToType<cfg.data_type>;
    using AccType     = fp32x4_t;
    using TC          = TileConstants<cfg>;

    ElementType* output_base;
    ck_tile::index_t output_spatial_offset; // q_pos * K (spatial + batch offset)
    ck_tile::index_t row_stride_elems;
    bool store_valid;

    int k_offset; // single K-offset (4 contiguous K values)

    // Number of in-range output channels in this thread's 4-K group (0..4).
    // Only meaningful when Padded; the unpadded path always writes 4.
    int valid_k_count;

    template <typename BlockCoords_>
    CK_TILE_DEVICE OutputWriterV3(const BlockCoords_& bc,
                                  uint4*, // Unused, matches OutputWriter signature.
                                  ElementType* __restrict__ out,
                                  int ho,
                                  int wo)
    {
        output_base      = out + static_cast<size_t>(bc.block_n) * ho * wo * bc.K + bc.block_k_out;
        row_stride_elems = wo * bc.K;

        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;

        const int q_pos       = bc.block_q + lane % 16;
        k_offset              = (lane / 16) * 4;
        output_spatial_offset = static_cast<ck_tile::index_t>(q_pos) * bc.K + k_offset;
        store_valid           = (q_pos < wo);
        valid_k_count         = 4;

        if constexpr(Padded)
        {
            // Derive per-group output-channel validity from the pad-transform
            // descriptor (computed once, here in the ctor — off the hot path).
            const int k_base = bc.block_k_out + k_offset;
            const auto kdesc = direct_conv::DenseSharedDescriptors<TC>::Output::
                MakeChannelPadDescriptor(bc.K);
            int count = 0;
            ck_tile::static_for<0, 4, 1>{}([&](auto j_n) {
                constexpr int J = j_n.value;
                const auto coord =
                    ck_tile::make_tensor_coordinate(kdesc, ck_tile::make_multi_index(k_base + J));
                if(ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(kdesc, coord))
                    ++count;
            });
            valid_k_count = count;
            store_valid   = store_valid && (count > 0);
        }
    }

    CK_TILE_DEVICE void flush(AccType acc_val, int p_out, int wave_id)
    {
        if(wave_id != 0 || !store_valid)
            return;

        const ck_tile::index_t row_offset = static_cast<ck_tile::index_t>(p_out) * row_stride_elems;
        const ck_tile::index_t store_offset = output_spatial_offset + row_offset;

        if constexpr(Padded)
        {
            // Write the in-range prefix of the 4-K group (valid_k_count elements).
            ck_tile::static_for<0, 4, 1>{}([&](auto j_n) {
                constexpr int J = j_n.value;
                if(J < valid_k_count)
                    output_base[store_offset + J] = static_cast<ElementType>(acc_val[J]);
            });
        }
        else
        {
            // Single 8B write: 4 contiguous K values.
            uint32_t words[2];
            words[0] = ConvertFp32ToVec4<ElementType>::convert(acc_val[0], acc_val[1]);
            words[1] = ConvertFp32ToVec4<ElementType>::convert(acc_val[2], acc_val[3]);
            __builtin_memcpy(output_base + store_offset, words, sizeof(words));
        }
    }
};

// ===================================================================
// OutputWriterV3Lds -- LDS-staged epilogue for 16B DRAM writes.
//
// After cross-wave LDS reduction, wave 0 writes the fp16-converted
// accumulator to a staging LDS buffer. All threads then participate
// in barriers. Active threads read 16B (uint4) from the staging
// buffer and write 16B to DRAM -- doubling throughput vs the 8B
// writes of OutputWriterV3.
//
// The staging buffer reuses the cross-wave reduction LDS region
// (reduce_lds), which is dead after cross_wave_reduce completes.
//
// Staging LDS layout: [BLOCK_Q, BLOCK_K] contiguous fp16.
//   M16N16K32: 16 × 16 = 256 fp16 = 512B = 32 uint4
//
// DRAM store: tid-based linear mapping.
//   Each active thread reads one uint4 (8 fp16) from staging LDS
//   and writes it to DRAM. Active threads: BLOCK_Q × BLOCK_K8.
//     M16N16K32: 16 × 2 = 32 active threads
//
// Output-channel padding is expressed as a CK Tile pad transform via
// DenseSharedDescriptors<TC>::Output::MakeChannelPadDescriptor: each active
// thread writes a contiguous 8-K block, and the descriptor's pad validity
// gives valid_k_count (a prefix count, 0..8) used to bound the flush writes.
//
// M16N16K32 only (M32N32K16 was dropped).
// ===================================================================
template <auto cfg, bool Padded = false>
struct OutputWriterV3Lds
{
    static_assert(cfg.mfma_shape == MfmaShape::M16N16K32,
                  "OutputWriterV3Lds supports only MfmaShape::M16N16K32");

    using ElementType = ToType<cfg.data_type>;
    using AccType     = fp32x4_t;
    using TC          = TileConstants<cfg>;

    static constexpr int BLOCK_K  = cfg.block_k_size(); // 16
    static constexpr int BLOCK_Q_ = cfg.block_q();      // 16
    static constexpr int BLOCK_K8 = BLOCK_K / 8;        // 2

    // Number of 16B stores needed to flush the staging buffer.
    static constexpr int STORE_VECS_V3 = BLOCK_Q_ * BLOCK_K8; // 32

    // Verify staging buffer fits within the cross-wave reduction LDS region.
    static constexpr int ACC_FLOATS    = sizeof(AccType) / sizeof(float);
    static constexpr int STAGING_UINT4 = BLOCK_Q_ * BLOCK_K8;
    static constexpr int REDUCE_UINT4  = cfg.waves_per_wg * 64 * ACC_FLOATS / 4;
    static_assert(STAGING_UINT4 <= REDUCE_UINT4,
                  "Output staging LDS must fit within cross-wave reduction LDS");

    ElementType* output_base;
    ElementType* staging_lds;
    ck_tile::index_t row_stride_elems;

    // Wave 0 LDS write state (MFMA accumulator layout).
    int lds_q_pos;
    int lds_k_offset;

    // Wide store state (tid-based 16B per thread).
    ck_tile::index_t store_lds_elem_offset; // element offset in staging LDS
    ck_tile::index_t store_dram_offset;     // element offset in DRAM (relative to output_base)
    bool store_valid;

    // Number of in-range output channels in this thread's 8-K block (0..8).
    // Only meaningful when Padded; the unpadded path always writes 8.
    int valid_k_count;

    template <typename BlockCoords_>
    CK_TILE_DEVICE OutputWriterV3Lds(const BlockCoords_& bc,
                                     uint4* staging_lds_buf,
                                     ElementType* __restrict__ out,
                                     int ho,
                                     int wo)
    {
        output_base      = out + static_cast<size_t>(bc.block_n) * ho * wo * bc.K + bc.block_k_out;
        staging_lds      = reinterpret_cast<ElementType*>(staging_lds_buf);
        row_stride_elems = wo * bc.K;

        const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
        const int tid  = static_cast<int>(threadIdx.x);

        // --- Wave 0 LDS write state ---
        lds_q_pos    = lane % 16;
        lds_k_offset = (lane / 16) * 4;

        // --- Wide store state (16B per active thread) ---
        valid_k_count = 8;
        if(tid < STORE_VECS_V3)
        {
            const int store_q  = tid / BLOCK_K8;
            const int store_k8 = tid % BLOCK_K8;
            const int global_q = bc.block_q + store_q;

            store_lds_elem_offset = store_q * BLOCK_K + store_k8 * 8;
            store_dram_offset     = static_cast<ck_tile::index_t>(global_q) * bc.K + store_k8 * 8;
            store_valid           = (global_q < wo);

            if constexpr(Padded)
            {
                // Derive the contiguous 8-K block's in-range count from the
                // pad-transform descriptor (computed once, off the hot path).
                const int k_base = bc.block_k_out + store_k8 * 8;
                const auto kdesc = direct_conv::DenseSharedDescriptors<TC>::Output::
                    MakeChannelPadDescriptor(bc.K);
                int count = 0;
                ck_tile::static_for<0, 8, 1>{}([&](auto j_n) {
                    constexpr int J  = j_n.value;
                    const auto coord = ck_tile::make_tensor_coordinate(
                        kdesc, ck_tile::make_multi_index(k_base + J));
                    if(ck_tile::coordinate_has_valid_offset_assuming_top_index_is_valid(kdesc,
                                                                                        coord))
                        ++count;
                });
                valid_k_count = count;
                store_valid   = store_valid && (count > 0);
            }
        }
        else
        {
            store_lds_elem_offset = 0;
            store_dram_offset     = 0;
            store_valid           = false;
        }
    }

    CK_TILE_DEVICE void flush(AccType acc_val, int p_out, int wave_id)
    {
        // Step 1: Wave 0 writes fp16-converted accumulator to staging LDS.
        if(wave_id == 0)
        {
            // Single 8B LDS write: 4 contiguous K values.
            uint32_t words[2];
            words[0] = ConvertFp32ToVec4<ElementType>::convert(acc_val[0], acc_val[1]);
            words[1] = ConvertFp32ToVec4<ElementType>::convert(acc_val[2], acc_val[3]);
            __builtin_memcpy(
                staging_lds + lds_q_pos * BLOCK_K + lds_k_offset, words, sizeof(words));
        }

        // Step 2: Barrier for LDS write visibility.
        __syncthreads();

        // Step 3: Active threads read 16B from staging LDS and write 16B to DRAM.
        if(store_valid)
        {
            const uint4* lds_uint4 = reinterpret_cast<const uint4*>(staging_lds);
            uint4 data             = lds_uint4[store_lds_elem_offset / 8];

            ck_tile::index_t store_offset =
                store_dram_offset + static_cast<ck_tile::index_t>(p_out) * row_stride_elems;
            if constexpr(Padded)
            {
                // Write the in-range prefix of the 8-K block (valid_k_count elems).
                const ElementType* d = reinterpret_cast<const ElementType*>(&data);
                ck_tile::static_for<0, 8, 1>{}([&](auto j_n) {
                    constexpr int J = j_n.value;
                    if(J < valid_k_count)
                        output_base[store_offset + J] = d[J];
                });
            }
            else
            {
                __builtin_memcpy(output_base + store_offset, &data, sizeof(data));
            }
        }

        // Step 4: Barrier to prevent next flush from overwriting staging LDS
        // before all threads finish reading.
        __syncthreads();
    }
};

// ===================================================================
// Kernel entry points.
// ===================================================================
template <auto cfg, bool Padded = false>
CK_TILE_DEVICE void ck_tile_conv2d_32c_nhwc_v3_impl(const ToType<cfg.data_type>* __restrict__ in,
                                                    const ToType<cfg.data_type>* __restrict__ wei,
                                                    double alpha,
                                                    double beta,
                                                    ToType<cfg.data_type>* __restrict__ out,
                                                    int N,
                                                    int C,
                                                    int K,
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
    using TC          = TileConstants<cfg>;
    using ElementType = ToType<cfg.data_type>;

    // Select MFMA functor based on data type. M16N16K32 only (M32N32K16 was
    // dropped — no live instance used it and M16N16K32 is the efficient shape).
    static_assert(cfg.mfma_shape == MfmaShape::M16N16K32,
                  "v3 dense kernel supports only MfmaShape::M16N16K32");
    using MfmaFn =
        std::conditional_t<cfg.data_type == DataType::bf16, Mfma16x16x32_bf16, Mfma16x16x32>;

    // Select output writer based on epilogue type.
    using OutputWriterType =
        std::conditional_t<cfg.epilogue == EpilogueType::RegistersToLdsToGlobalMemory,
                           OutputWriterV3Lds<cfg, Padded>,
                           OutputWriterV3<cfg, Padded>>;

    conv_compute_loop_v3<TC,
                         cfg,
                         MfmaFn,
                         ConvBlockCoordsT<cfg>,
                         ConvInputLoader<cfg, Padded>,
                         WeightLoader<cfg, Padded>,
                         OutputWriterType,
                         ElementType>(in, wei, out, N, C, K, hi, wi, ho, wo, py, px);
}

template <auto cfg, bool Padded = false>
struct Conv32cV3Kernel
{
    static constexpr ck_tile::index_t kBlockSize = cfg.block_size();

    CK_TILE_DEVICE void operator()(const ToType<cfg.data_type>* __restrict__ in,
                                   const ToType<cfg.data_type>* __restrict__ wei,
                                   double alpha,
                                   double beta,
                                   ToType<cfg.data_type>* __restrict__ out,
                                   int N,
                                   int C,
                                   int K,
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
                                   int px) const
    {
        // XOR swizzle bank-conflict avoidance relies on bitwise XOR of the wave index,
        // which only produces a valid permutation when waves_per_wg is a power of 2.
        static_assert(
            cfg.swizzle_type != SwizzleType::XOR ||
                (cfg.waves_per_wg > 0 && (cfg.waves_per_wg & (cfg.waves_per_wg - 1)) == 0),
            "XOR swizzle requires waves_per_wg to be a power of 2");
        ck_tile_conv2d_32c_nhwc_v3_impl<cfg, Padded>(
            in, wei, alpha, beta, out, N, C, K, hi, wi, ho, wo, fy, fx, sy, sx, dy, dx, py, px);
    }
};

// ===================================================================
// is_applicable -- checks whether a Conv2dParams is suitable for v3.
// ===================================================================
template <DataType DT = DataType::fp16>
inline bool is_applicable(const Conv2dParams& par)
{
    if(!is_applicable_base(par))
        return false;

    if(par.in_type != DT || par.wei_type != DT || par.out_type != DT)
    {
        LogInfo("Data type mismatch.");
        return false;
    }

    if(par.groups != 1)
    {
        LogInfo("Grouped convolution not supported by dense kernel");
        return false;
    }

    // Sub-8 channel granularity: the reduction (C_in) and
    // output (K_out) channel counts may be any positive value. Straddling
    // channel tiles are loaded and zeroed at element granularity by the weight
    // loader (reduction) and masked element-wise by the output writer (output).
    // The exact per-config covering requirement (C_in lands in the last
    // cpg-slice) is checked in is_valid_config.
    //   Fprop: C_in = c_tot (reduction),  K_out = k_tot (output)
    //   Dgrad: C_in = k_tot (reduction),  K_out = c_tot (output)
    if(par.direction != Direction::Fprop && par.direction != Direction::Dgrad)
    {
        LogInfo("Unsupported convolution direction (bwd weight).");
        return false;
    }

    return true;
}

// ===================================================================
// launch_kernel -- compile-time config dispatch for v3.
// ===================================================================
template <auto cfg, DataType DT = DataType::fp16>
inline void launch_kernel(const LaunchParams& lp,
                          const Conv2dParams& par,
                          const void* in,
                          const void* wei,
                          void* out,
                          hipStream_t stream)
{
    using ElementType = ToType<DT>;
    auto view         = SizeView<cfg.direction>(par);

    // Determine whether channel padding is required at runtime: the reduction
    // channel count (C_in) is smaller than the config's full total_block_c(),
    // or the output channel count (K_out) is not a multiple of block_k_size.
    const int C_in        = (cfg.direction == Direction::Dgrad) ? par.k_tot : par.c_tot;
    const int K_out       = (cfg.direction == Direction::Dgrad) ? par.c_tot : par.k_tot;
    const bool needs_pad  = (C_in != cfg.total_block_c()) || (K_out % cfg.block_k_size() != 0);

    // The non-grouped v3 kernel is compute-bound and register-hungry.
    // Using the default min-2-blocks/CU caps registers and spills.
    constexpr int MinBlockPerCu = 1;

    auto run = [&](auto padded_const) {
        constexpr bool Padded = decltype(padded_const)::value;
        ck_tile::make_kernel<MinBlockPerCu>(Conv32cV3Kernel<cfg, Padded>{},
                                            lp.grid,
                                            lp.block_size,
                                            lp.dynamic_shared_bytes,
                                            static_cast<const ElementType*>(in),
                                            static_cast<const ElementType*>(wei),
                                            1.0,
                                            0.0,
                                            static_cast<ElementType*>(out),
                                            par.n,
                                            par.c_tot,
                                            par.k_tot,
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
                                            view.pad_w())(ck_tile::stream_config{stream});
    };

    if(needs_pad)
        run(std::true_type{});
    else
        run(std::false_type{});
}

} // namespace ck_tile::direct_conv::conv_32c_tile::v3
