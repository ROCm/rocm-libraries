// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_tuning.hpp"

/**
 * @file
 * The shape and tuning of one MxGemmPipelineAgBgCrCompTDMScheduled instance as plain values, with
 * the rules that derive the rest of it: validation, LDS layout, read and wait counts, store widths.
 * A port of rocKE's gfx1250 MxGemmSpec (instances/gfx1250/mx_gemm.py at commit b32989439dc).
 *
 * Everything is constexpr so the policy evaluates it at compile time from the Problem and the
 * tuning, and the host test can call it on plain specs.
 */

namespace ck_tile {

enum class MxTdmScheduledDtype : int
{
    Fp4 = 0,
    Fp8 = 1,
    Bf8 = 2
};

enum class MxTdmScheduledOutDtype : int
{
    Fp8E4M3 = 0,
    Fp16    = 1,
    Bf16    = 2
};

/// The scaled WMMA atom: 32x16x128 (FP4 only) or 16x16x128.
enum class MxTdmScheduledMma : int
{
    M32N16 = 0,
    M16N16 = 1
};

/// Why a spec does not build, in the order rocKE's validate() checks. Ok when it builds.
enum class MxTdmScheduledReason : int
{
    Ok,
    ShapePositive,
    KScaleBlock,
    OutDtype,
    TileMN,
    Dtype,
    DtypeWidth,
    TileK,
    WaveGroup,
    Mma,
    TileAtoms,
    ScaleSpan,
    KShort,
    LocalPrefetch,
    HalfPlr,
    HalfPlrB,
    HalfBOrder,
    HalfBLate,
    SignalWmmas,
    Gl2Distance,
    ClusterDim,
    ClusterSize,
    ClusterGrid,
    LdsPad,
    Accumulators,
    LsiRepeat,
    LsiBaselineSplit,
    LsiCapacity,
    LdsCapacity
};

/// The short code of a reason, as in the Python validator.
CK_TILE_HOST_DEVICE constexpr const char* mx_tdm_scheduled_reason_code(MxTdmScheduledReason r)
{
    switch(r)
    {
    case MxTdmScheduledReason::Ok: return "ok";
    case MxTdmScheduledReason::ShapePositive: return "shape_positive";
    case MxTdmScheduledReason::KScaleBlock: return "k_scale_block";
    case MxTdmScheduledReason::OutDtype: return "out_dtype";
    case MxTdmScheduledReason::TileMN: return "tile_mn";
    case MxTdmScheduledReason::Dtype: return "dtype";
    case MxTdmScheduledReason::DtypeWidth: return "dtype_width";
    case MxTdmScheduledReason::TileK: return "tile_k";
    case MxTdmScheduledReason::WaveGroup: return "wave_group";
    case MxTdmScheduledReason::Mma: return "mma";
    case MxTdmScheduledReason::TileAtoms: return "tile_atoms";
    case MxTdmScheduledReason::ScaleSpan: return "scale_span";
    case MxTdmScheduledReason::KShort: return "k_short";
    case MxTdmScheduledReason::LocalPrefetch: return "local_prefetch";
    case MxTdmScheduledReason::HalfPlr: return "half_plr";
    case MxTdmScheduledReason::HalfPlrB: return "half_plr_b";
    case MxTdmScheduledReason::HalfBOrder: return "half_b_order";
    case MxTdmScheduledReason::HalfBLate: return "half_b_late";
    case MxTdmScheduledReason::SignalWmmas: return "signal_wmmas";
    case MxTdmScheduledReason::Gl2Distance: return "gl2_distance";
    case MxTdmScheduledReason::ClusterDim: return "cluster_dim";
    case MxTdmScheduledReason::ClusterSize: return "cluster_size";
    case MxTdmScheduledReason::ClusterGrid: return "cluster_grid";
    case MxTdmScheduledReason::LdsPad: return "lds_pad";
    case MxTdmScheduledReason::Accumulators: return "accumulators";
    case MxTdmScheduledReason::LsiRepeat: return "lsi_repeat";
    case MxTdmScheduledReason::LsiBaselineSplit: return "lsi_baseline_split";
    case MxTdmScheduledReason::LsiCapacity: return "lsi_capacity";
    case MxTdmScheduledReason::LdsCapacity: return "lds_capacity";
    }
    return "unknown";
}

/// rocKE's message for a reason, without the values it formats in.
CK_TILE_HOST_DEVICE constexpr const char* mx_tdm_scheduled_reason_message(MxTdmScheduledReason r)
{
    switch(r)
    {
    case MxTdmScheduledReason::Ok: return "ok";
    case MxTdmScheduledReason::ShapePositive: return "M, N and K must be positive";
    case MxTdmScheduledReason::KScaleBlock:
        return "K must be a multiple of 32: each E8M0 scale covers 32 K values";
    case MxTdmScheduledReason::OutDtype: return "out_dtype must be one of fp8e4m3, fp16, bf16";
    case MxTdmScheduledReason::TileMN: return "tile_m and tile_n must be in 16, 32, 64, 128, 256";
    case MxTdmScheduledReason::Dtype: return "a_dtype and b_dtype must be in fp4, fp8, bf8";
    case MxTdmScheduledReason::DtypeWidth:
        return "a_dtype and b_dtype must have the same width (no FP4 x 8-bit)";
    case MxTdmScheduledReason::TileK:
        return "tile_k must be in 128, 256, 512, 1024 for fp4 and 128, 256 for 8-bit types";
    case MxTdmScheduledReason::WaveGroup: return "wave_group must be 2x2, 4x1 or 1x4 (4 waves)";
    case MxTdmScheduledReason::Mma: return "mma must be 32x16 or 16x16 for fp4 and 16x16 for 8-bit";
    case MxTdmScheduledReason::TileAtoms: return "the tile must hold whole WMMA atoms per wave";
    case MxTdmScheduledReason::ScaleSpan:
        return "A and B scale bytes must each be at most 2^31 (i32 offsets)";
    case MxTdmScheduledReason::KShort: return "K must hold at least one full tile";
    case MxTdmScheduledReason::LocalPrefetch: return "local_prefetch must be 0 or 1";
    case MxTdmScheduledReason::HalfPlr: return "half_plr must be 'b' or 'none'";
    case MxTdmScheduledReason::HalfPlrB:
        return "half_plr 'b' needs local_prefetch 1 and an even wave_repeat_n";
    case MxTdmScheduledReason::HalfBOrder: return "half_b_order must be 'ahead' or 'late'";
    case MxTdmScheduledReason::HalfBLate: return "half_b_order 'late' needs half_plr 'b'";
    case MxTdmScheduledReason::SignalWmmas:
        return "cluster_signal_wmmas must be in 0..signal_limit";
    case MxTdmScheduledReason::Gl2Distance:
        return "gl2_prefetch_distance must be 0 (off) or 2..8 tiles";
    case MxTdmScheduledReason::ClusterDim: return "cluster_dim must be a positive (x, y, z) triple";
    case MxTdmScheduledReason::ClusterSize:
        return "cluster_dim must be 2D with at most 16 workgroups";
    case MxTdmScheduledReason::ClusterGrid: return "the grid must be divisible by cluster_dim";
    case MxTdmScheduledReason::LdsPad: return "an LDS pad does not encode as TDM padding";
    case MxTdmScheduledReason::Accumulators: return "the accumulators per lane exceed 1024 VGPRs";
    case MxTdmScheduledReason::LsiRepeat:
        return "segment interleave needs the split operand's wave repeat to cover its wave's rows";
    case MxTdmScheduledReason::LsiBaselineSplit:
        return "segment interleave: the baseline layout already splits B's components across "
               "segments";
    case MxTdmScheduledReason::LsiCapacity:
        return "segment interleave: the aligned layout needs more LDS than fits";
    case MxTdmScheduledReason::LdsCapacity: return "LDS bytes exceed the 327680 B of gfx1250";
    }
    return "unknown";
}

/// pad_bytes of LDS padding after every block_bytes written (TensileLite LdsPad and
/// LdsBlockSizePerPad, in bytes).
struct MxTdmScheduledLdsPad
{
    index_t pad_bytes   = 0;
    index_t block_bytes = 0;

    CK_TILE_HOST_DEVICE constexpr bool on() const { return pad_bytes > 0; }
};

/// One operand's place in an LDS buffer: two TDM components (one per wave pair) of rows rows,
/// component c at base + c * stride. Data pads above 1024 B are TDM iterate mode: iterate_rows
/// rows per iteration, iterations iterate_stride apart (0 when not iterating).
struct MxTdmScheduledOperand
{
    index_t rows           = 0;
    index_t base           = 0;
    index_t stride         = 0;
    index_t pad_bytes      = 0;
    index_t block_bytes    = 0;
    index_t iterate_rows   = 0;
    index_t iterate_stride = 0;

    CK_TILE_HOST_DEVICE constexpr MxTdmScheduledLdsPad pad() const
    {
        return {pad_bytes, block_bytes};
    }
};

/// The two LDS buffers. Buffer 1 starts buffer_stride after buffer 0; buffer_bytes is where buffer
/// 0's last region ends. reason is not Ok when segment interleave does not apply.
struct MxTdmScheduledLdsLayout
{
    MxTdmScheduledReason reason = MxTdmScheduledReason::Ok;
    index_t row_bytes           = 0;
    index_t k_steps             = 0;
    MxTdmScheduledOperand a{};
    MxTdmScheduledOperand b{};
    MxTdmScheduledOperand a_scale{};
    MxTdmScheduledOperand b_scale{};
    index_t buffer_bytes  = 0;
    index_t buffer_stride = 0;
    index_t lds_bytes     = 0;
};

namespace mx_tdm_scheduled {

inline constexpr index_t kAtomK            = 128;
inline constexpr index_t kWave             = 32;
inline constexpr index_t kWaves            = 4;
inline constexpr index_t kBlockK           = 32;
inline constexpr index_t kDefaultSignal    = 6;
inline constexpr index_t kDataPadBytes     = 16;
inline constexpr index_t kBankCycleBytes   = 256;
inline constexpr index_t kMxsBlockBytes    = 256;
inline constexpr index_t kMxsPadBytes      = 16;
inline constexpr index_t kRegionAlignBits  = 2048;
inline constexpr index_t kSegmentBytes     = 1 << 16;
inline constexpr index_t kTdmPadPeriodMax  = 1024;
inline constexpr index_t kTdmPadMaxBytes   = 128 * 4;
inline constexpr index_t kLdsPadStepBytes  = 8;
inline constexpr index_t kDscntMax         = 63;
inline constexpr long_index_t kI32Span     = 1ll << 31;
inline constexpr index_t kGl2LineBytes     = 256;
inline constexpr index_t kLdsCapacityBytes = 327680;
inline constexpr index_t kVgprs            = 1024;
/// Lanes that hold one row of a fragment: 16 for both atoms.
inline constexpr index_t kFragLanes = 16;
/// A lane's K chunk of a row and K step, and the lane groups (lane / 16) side by side.
inline constexpr index_t kChunkBytes = 16;
inline constexpr index_t kLaneGroups = 2;

CK_TILE_HOST_DEVICE constexpr bool is_pow2(long_index_t v) { return v > 0 && (v & (v - 1)) == 0; }

CK_TILE_HOST_DEVICE constexpr long_index_t round_up(long_index_t v, long_index_t m)
{
    return (v + m - 1) / m * m;
}

CK_TILE_HOST_DEVICE constexpr long_index_t ceil_div(long_index_t v, long_index_t d)
{
    return (v + d - 1) / d;
}

/// The smallest power of two at least v (1 for v <= 1).
CK_TILE_HOST_DEVICE constexpr long_index_t pow2_ceil(long_index_t v)
{
    long_index_t p = 1;
    while(p < v)
        p <<= 1;
    return p;
}

CK_TILE_HOST_DEVICE constexpr index_t min_i(index_t a, index_t b) { return a < b ? a : b; }
CK_TILE_HOST_DEVICE constexpr index_t max_i(index_t a, index_t b) { return a > b ? a : b; }

} // namespace mx_tdm_scheduled

/**
 * @brief One comp_tdm_scheduled kernel: problem, tile and tuning. Fields follow rocKE's MxGemmSpec;
 * M, N and K are 0 where only the compile-time part is known (see mx_tdm_scheduled_validate's
 * check_shape).
 */
struct MxTdmScheduledSpec
{
    long_index_t M                        = 0;
    long_index_t N                        = 0;
    long_index_t K                        = 0;
    MxTdmScheduledOutDtype out            = MxTdmScheduledOutDtype::Fp8E4M3;
    index_t tile_m                        = 256;
    index_t tile_n                        = 256;
    index_t tile_k                        = 512;
    index_t warp_m                        = 2;
    index_t warp_n                        = 2;
    index_t cluster_x                     = 4;
    index_t cluster_y                     = 4;
    index_t cluster_z                     = 1;
    index_t gl2_distance                  = 4;
    index_t signal_wmmas_in               = -1;
    index_t local_prefetch                = 1;
    MxTdmScheduledHalfPlr half_plr        = MxTdmScheduledHalfPlr::B;
    MxTdmScheduledHalfBOrder half_b_order = MxTdmScheduledHalfBOrder::Ahead;
    bool lds_segment_interleave           = true;
    MxTdmScheduledMma mma                 = MxTdmScheduledMma::M32N16;
    MxTdmScheduledDtype a_dtype           = MxTdmScheduledDtype::Fp4;
    MxTdmScheduledDtype b_dtype           = MxTdmScheduledDtype::Fp4;

    // The WMMA atom.
    CK_TILE_HOST_DEVICE constexpr index_t atom_m() const
    {
        return mma == MxTdmScheduledMma::M32N16 ? 32 : 16;
    }
    CK_TILE_HOST_DEVICE constexpr index_t atom_n() const { return 16; }
    /// FP32 accumulators per lane of one atom.
    CK_TILE_HOST_DEVICE constexpr index_t c_frag_len() const
    {
        return mma == MxTdmScheduledMma::M32N16 ? 16 : 8;
    }
    /// A rows per lane of one atom (lane % 16, then + 16).
    CK_TILE_HOST_DEVICE constexpr index_t a_rows() const
    {
        return mma == MxTdmScheduledMma::M32N16 ? 2 : 1;
    }
    /// Words of one A or B operand of the WMMA (FP4 fills 8 of the 16x16 atom's 16).
    CK_TILE_HOST_DEVICE constexpr index_t frag_words() const
    {
        return mma == MxTdmScheduledMma::M32N16 ? 8 : 16;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t dtype_bits(MxTdmScheduledDtype t)
    {
        return t == MxTdmScheduledDtype::Fp4 ? 4 : 8;
    }
    CK_TILE_HOST_DEVICE constexpr index_t elem_bits() const { return dtype_bits(a_dtype); }
    CK_TILE_HOST_DEVICE constexpr index_t k_per_byte() const { return 8 / elem_bits(); }
    /// Bytes of one K step of an A or B row.
    CK_TILE_HOST_DEVICE constexpr index_t step_bytes() const
    {
        return mx_tdm_scheduled::kAtomK / k_per_byte();
    }
    CK_TILE_HOST_DEVICE constexpr index_t chunk_bytes() const
    {
        return mx_tdm_scheduled::kChunkBytes;
    }
    /// 16 B K chunks each lane reads of one A or B row and K step.
    CK_TILE_HOST_DEVICE constexpr index_t chunks_per_lane() const
    {
        return step_bytes() / (mx_tdm_scheduled::kLaneGroups * mx_tdm_scheduled::kChunkBytes);
    }
    /// Byte offset, in one K step of a row, of lane group 0's chunk j; lane group g reads
    /// g * chunk_bytes further.
    CK_TILE_HOST_DEVICE constexpr index_t chunk_offset(index_t j) const
    {
        return j * mx_tdm_scheduled::kLaneGroups * mx_tdm_scheduled::kChunkBytes;
    }

    CK_TILE_HOST_DEVICE constexpr index_t wave_repeat_m() const
    {
        return tile_m / (warp_m * atom_m());
    }
    CK_TILE_HOST_DEVICE constexpr index_t wave_repeat_n() const
    {
        return tile_n / (warp_n * atom_n());
    }
    CK_TILE_HOST_DEVICE constexpr index_t block_size() const
    {
        return mx_tdm_scheduled::kWave * mx_tdm_scheduled::kWaves;
    }
    CK_TILE_HOST_DEVICE constexpr index_t k_steps() const
    {
        return tile_k / mx_tdm_scheduled::kAtomK;
    }
    /// K steps of the last, partial tile when tile_k does not divide K, counting a partial last
    /// K step when 128 does not divide K.
    CK_TILE_HOST_DEVICE constexpr index_t tail_steps() const
    {
        return static_cast<index_t>(
            mx_tdm_scheduled::ceil_div(K % tile_k, mx_tdm_scheduled::kAtomK));
    }
    CK_TILE_HOST_DEVICE constexpr bool edge_m() const { return M % tile_m != 0; }
    CK_TILE_HOST_DEVICE constexpr bool edge_n() const { return N % tile_n != 0; }
    /// Bytes of an A or B row.
    CK_TILE_HOST_DEVICE constexpr long_index_t packed_k() const { return K / k_per_byte(); }
    /// A's byte offsets, whole tiles counted, pass i32: address it in i64.
    CK_TILE_HOST_DEVICE constexpr bool wide_a() const
    {
        return mx_tdm_scheduled::round_up(M, tile_m) * packed_k() > mx_tdm_scheduled::kI32Span;
    }
    CK_TILE_HOST_DEVICE constexpr bool wide_b() const
    {
        return mx_tdm_scheduled::round_up(N, tile_n) * packed_k() > mx_tdm_scheduled::kI32Span;
    }
    CK_TILE_HOST_DEVICE constexpr bool wide_d() const
    {
        return mx_tdm_scheduled::round_up(M, tile_m) * mx_tdm_scheduled::round_up(N, tile_n) >
               mx_tdm_scheduled::kI32Span;
    }
    CK_TILE_HOST_DEVICE constexpr index_t out_bytes() const
    {
        return out == MxTdmScheduledOutDtype::Fp8E4M3 ? 1 : 2;
    }
    /// Elements per D store and C load at most: 16 B and the N repeat.
    CK_TILE_HOST_DEVICE constexpr index_t max_store_width() const
    {
        return mx_tdm_scheduled::min_i(wave_repeat_n(), 16 / out_bytes());
    }
    /// Elements per D store and C load: at most 16 B and a divisor of N, so a vector is wholly
    /// inside or outside N and stays aligned.
    CK_TILE_HOST_DEVICE constexpr index_t store_width() const
    {
        index_t width = max_store_width();
        while(width > 1 && N % width)
            width /= 2;
        return width;
    }
    CK_TILE_HOST_DEVICE constexpr bool has_cluster() const
    {
        return !(cluster_x == 1 && cluster_y == 1);
    }
    CK_TILE_HOST_DEVICE constexpr bool half_b() const
    {
        return local_prefetch == 1 && half_plr == MxTdmScheduledHalfPlr::B;
    }
    CK_TILE_HOST_DEVICE constexpr index_t signal_limit() const
    {
        const index_t per_step = wave_repeat_m() * wave_repeat_n();
        return half_b() ? per_step / 2 : per_step;
    }
    CK_TILE_HOST_DEVICE constexpr index_t signal_wmmas() const
    {
        return signal_wmmas_in >= 0
                   ? signal_wmmas_in
                   : mx_tdm_scheduled::min_i(mx_tdm_scheduled::kDefaultSignal, signal_limit());
    }
    CK_TILE_HOST_DEVICE constexpr index_t fp8_convert_width() const
    {
        return wave_repeat_n() % 8 == 0 ? 8 : 4;
    }
    CK_TILE_HOST_DEVICE constexpr index_t a_scale_reads() const
    {
        return mx_tdm_scheduled::max_i(1, wave_repeat_m() / 4);
    }
    CK_TILE_HOST_DEVICE constexpr index_t b_scale_reads() const
    {
        return mx_tdm_scheduled::max_i(1, wave_repeat_n() / 4);
    }
    /// LDS reads of one K step: scales, then one per K chunk of each A row and B fragment.
    CK_TILE_HOST_DEVICE constexpr index_t reads_per_step() const
    {
        return a_scale_reads() + b_scale_reads() +
               chunks_per_lane() * (a_rows() * wave_repeat_m() + wave_repeat_n());
    }
    CK_TILE_HOST_DEVICE constexpr bool half_b_late() const
    {
        return half_b() && half_b_order == MxTdmScheduledHalfBOrder::Late;
    }
    /// LDS reads of half of one K step's B fragments.
    CK_TILE_HOST_DEVICE constexpr index_t b_half_reads() const
    {
        return chunks_per_lane() * (wave_repeat_n() / 2);
    }
    /// LDS reads of one K step's scales and A rows.
    CK_TILE_HOST_DEVICE constexpr index_t head_reads() const
    {
        return reads_per_step() - 2 * b_half_reads();
    }
    /// How many s_wait_dscnt the K loop has per K step: 2 with half-B late, else 1.
    CK_TILE_HOST_DEVICE constexpr index_t num_dscnt_waits() const { return half_b_late() ? 2 : 1; }
    /// The K loop's s_wait_dscnt counts before clamping: the reads issued after the ones each
    /// wait needs.
    CK_TILE_HOST_DEVICE constexpr index_t dscnt_wait(index_t i) const
    {
        if(!local_prefetch)
            return 0;
        if(half_b_late())
            return i == 0 ? b_half_reads() : head_reads();
        return i == 0 ? reads_per_step() : 0;
    }
    /// Reads of the other buffer that may stay in flight at the cluster late wait.
    CK_TILE_HOST_DEVICE constexpr index_t late_pending_reads() const
    {
        if(!local_prefetch)
            return 0;
        return half_b_late() ? head_reads() : reads_per_step();
    }
    /// Workgroups along x (N tiles) and y (M tiles).
    CK_TILE_HOST_DEVICE constexpr long_index_t grid_x() const
    {
        return mx_tdm_scheduled::ceil_div(N, tile_n);
    }
    CK_TILE_HOST_DEVICE constexpr long_index_t grid_y() const
    {
        return mx_tdm_scheduled::ceil_div(M, tile_m);
    }
};

namespace mx_tdm_scheduled {

/// calcLdsBlockSizePerPad for an UnrollMajorLDS operand: the K row, rounded to a bank cycle,
/// times VectorWidth once that is longer.
CK_TILE_HOST_DEVICE constexpr index_t tensile_data_block(index_t k_bytes, index_t vector_width)
{
    index_t block = static_cast<index_t>(round_up(k_bytes, kBankCycleBytes));
    if(k_bytes * vector_width > kBankCycleBytes)
        block = static_cast<index_t>(round_up(k_bytes * vector_width, kBankCycleBytes));
    return block;
}

/// get_mxs_mt_config(MatrixInstK=128, MXBlock=32, VectorWidth).
CK_TILE_HOST_DEVICE constexpr MxTdmScheduledLdsPad tensile_scale_pad(index_t vector_width)
{
    const index_t per_block = (kAtomK / 32) * vector_width;
    if(vector_width < 4 || per_block % 16 || (per_block / 16) & 1)
        return {};
    return {kMxsPadBytes, kMxsBlockBytes};
}

struct DerivedPads
{
    MxTdmScheduledLdsPad a{};
    MxTdmScheduledLdsPad b{};
    MxTdmScheduledLdsPad a_scale{};
    MxTdmScheduledLdsPad b_scale{};
};

/// TensileLite's padding with its A/B relabeled to rocKE B/A and VectorWidth taken as the
/// operand's wave repeat.
CK_TILE_HOST_DEVICE constexpr DerivedPads derived_lds_pads(const MxTdmScheduledSpec& s)
{
    const index_t k_bytes = s.tile_k / s.k_per_byte();
    const index_t rm      = s.wave_repeat_m();
    const index_t rn      = s.wave_repeat_n();
    return {{kDataPadBytes, tensile_data_block(k_bytes, rm)},
            {kDataPadBytes, tensile_data_block(k_bytes, rn)},
            tensile_scale_pad(rm),
            tensile_scale_pad(rn)};
}

struct DataComponent
{
    index_t bytes          = 0;
    index_t iterate_rows   = 0;
    index_t iterate_stride = 0;
};

CK_TILE_HOST_DEVICE constexpr DataComponent
data_component(index_t rows, index_t row_bytes, MxTdmScheduledLdsPad pad)
{
    if(pad.on() && pad.block_bytes > kTdmPadPeriodMax)
    {
        const index_t iterate_rows = pad.block_bytes / row_bytes;
        const index_t stride       = pad.block_bytes + pad.pad_bytes;
        return {static_cast<index_t>(ceil_div(rows, iterate_rows)) * stride, iterate_rows, stride};
    }
    index_t size = rows * row_bytes;
    if(pad.on())
        size += size / pad.block_bytes * pad.pad_bytes;
    return {size, 0, 0};
}

/// calcLdsNumBytesAB for an operand's scales, padded as one tensor.
CK_TILE_HOST_DEVICE constexpr index_t
scale_region_bytes(index_t rows, index_t steps, MxTdmScheduledLdsPad pad)
{
    const index_t size = 2 * rows * steps * 4;
    return pad.on() ? size / pad.block_bytes * (pad.block_bytes + pad.pad_bytes) : size;
}

/// Bases and component strides of A and B and where the scales start. An aligned layout reaches
/// span (scales included), past the baseline span.
struct Placement
{
    MxTdmScheduledReason reason = MxTdmScheduledReason::Ok;
    index_t a_base              = 0;
    index_t a_stride            = 0;
    index_t b_base              = 0;
    index_t b_stride            = 0;
    index_t scale_base          = 0;
    bool aligned                = false;
    index_t span                = 0;
};

/// first at 0 with component stride `stride`, the other after first's component 0 with
/// `second_stride`.
CK_TILE_HOST_DEVICE constexpr Placement place(bool first_is_b,
                                              index_t comp_first,
                                              index_t stride,
                                              index_t second_stride,
                                              index_t scale_base,
                                              bool aligned = false,
                                              index_t span = 0)
{
    Placement p{};
    if(first_is_b)
    {
        p.b_base   = 0;
        p.b_stride = stride;
        p.a_base   = comp_first;
        p.a_stride = second_stride;
    }
    else
    {
        p.a_base   = 0;
        p.a_stride = stride;
        p.b_base   = comp_first;
        p.b_stride = second_stride;
    }
    p.scale_base = scale_base;
    p.aligned    = aligned;
    p.span       = span;
    return p;
}

/// TensileLite's segment_interleave.evaluate with LDSSegmentInterleave 1, VectorWidth equal to
/// the wave repeat and no TDMSplit; `scales` is the aligned size of both scale regions.
///
/// 2x2: A's components go to different segments, as tight [B0][A0][B1][A1], or segment-aligned
/// when B0 A0 fit one segment. With the 32x16 atom an A read would cross a component, so A stays
/// whole: [B0][A0 A1][B1], aligned when that fits one segment. 4x1 and 1x4: only the operand
/// spanning four waves splits, around the other one.
CK_TILE_HOST_DEVICE constexpr Placement interleave_placement(const MxTdmScheduledSpec& s,
                                                             index_t comp_a,
                                                             index_t comp_b,
                                                             index_t data_a,
                                                             index_t data_b,
                                                             index_t scales)
{
    constexpr index_t seg = kSegmentBytes;
    const index_t lanes   = min_i(s.atom_m(), s.atom_n());
    const auto coarse     = [&](bool is_b) {
        const index_t tile   = is_b ? s.tile_n : s.tile_m;
        const index_t waves  = is_b ? s.warp_n : s.warp_m;
        const index_t repeat = is_b ? s.wave_repeat_n() : s.wave_repeat_m();
        return lanes * repeat >= tile / waves;
    };
    Placement fail{};
    if(s.warp_m == 1 || s.warp_n == 1)
    {
        const bool active_b      = s.warp_n > 1;
        const index_t comp_act   = active_b ? comp_b : comp_a;
        const index_t comp_share = active_b ? comp_a : comp_b;
        const index_t data_act   = active_b ? data_b : data_a;
        if(!coarse(active_b))
        {
            fail.reason = MxTdmScheduledReason::LsiRepeat;
            return fail;
        }
        if(active_b && data_b <= seg && comp_b >= seg)
        {
            fail.reason = MxTdmScheduledReason::LsiBaselineSplit;
            return fail;
        }
        const index_t stride = comp_act + 2 * comp_share;
        if(stride / seg > (data_act - 1) / seg)
            return place(active_b, comp_act, stride, comp_share, stride + comp_act);
        const index_t pre  = static_cast<index_t>(round_up(stride, seg));
        const index_t span = pre + comp_act;
        return place(active_b, comp_act, pre, comp_share, span, true, span + scales);
    }
    if(!coarse(true))
    {
        fail.reason = MxTdmScheduledReason::LsiRepeat;
        return fail;
    }
    const index_t a_cols  = s.atom_m() * s.warp_m * s.wave_repeat_m();
    const bool a_readable = coarse(false) || (s.tile_m / 2) % a_cols == 0;
    if(!a_readable)
    {
        const index_t stride = comp_b + 2 * comp_a;
        if(stride >= seg)
            return place(true, comp_b, stride, comp_a, stride + comp_b);
        const index_t pre  = static_cast<index_t>(round_up(stride, seg));
        const index_t span = pre + comp_b;
        return place(true, comp_b, pre, comp_a, span, true, span + scales);
    }
    const index_t pair = comp_a + comp_b;
    if(pair < seg)
    {
        const index_t span = seg + pair;
        return place(true, comp_b, seg, seg, span, true, span + scales);
    }
    return place(true, comp_b, pair, pair, 2 * pair);
}

} // namespace mx_tdm_scheduled

/// The LDS layout of a spec whose tile, types and wave group are valid (rocKE _lds_placement).
/// Baseline (segment interleave off) is [B0 B1][B scales][A scales][A0 A1]; segment interleave
/// places the components so the two LDS read ports read different 64 KiB segments.
CK_TILE_HOST_DEVICE constexpr MxTdmScheduledLdsLayout
mx_tdm_scheduled_lds_layout(const MxTdmScheduledSpec& s)
{
    using namespace mx_tdm_scheduled;
    const DerivedPads pads  = derived_lds_pads(s);
    const index_t row_bytes = s.tile_k / s.k_per_byte();
    const index_t steps     = s.k_steps();
    const index_t a_rows    = s.tile_m / 2;
    const index_t b_rows    = s.tile_n / 2;
    const DataComponent a   = data_component(a_rows, row_bytes, pads.a);
    const DataComponent b   = data_component(b_rows, row_bytes, pads.b);

    const auto scale_component = [&](index_t rows, MxTdmScheduledLdsPad pad) {
        const index_t size = steps * rows * 4;
        return size + (pad.on() ? size / pad.block_bytes * pad.pad_bytes : 0);
    };

    const index_t align = kRegionAlignBits / s.elem_bits();
    const index_t b_scale_region =
        static_cast<index_t>(round_up(scale_region_bytes(b_rows, steps, pads.b_scale), align));
    const index_t a_scale_region =
        static_cast<index_t>(round_up(scale_region_bytes(a_rows, steps, pads.a_scale), align));
    const index_t b_end = static_cast<index_t>(round_up(2 * b.bytes, align));
    Placement baseline{};
    baseline.a_base             = b_end + b_scale_region + a_scale_region;
    baseline.a_stride           = a.bytes;
    baseline.b_base             = 0;
    baseline.b_stride           = b.bytes;
    baseline.scale_base         = b_end;
    const index_t baseline_span = baseline.a_base + 2 * a.bytes;
    const index_t natural_stride =
        baseline.a_base + static_cast<index_t>(round_up(2 * a.bytes, align));

    MxTdmScheduledLdsLayout out{};
    Placement placement = baseline;
    if(s.lds_segment_interleave)
    {
        placement = interleave_placement(s,
                                         a.bytes,
                                         b.bytes,
                                         a_rows * row_bytes,
                                         b_rows * row_bytes,
                                         b_scale_region + a_scale_region);
        if(placement.reason != MxTdmScheduledReason::Ok)
        {
            out.reason = placement.reason;
            return out;
        }
    }
    index_t buffer_stride = 0;
    index_t lds_bytes     = 0;
    if(placement.aligned)
    {
        buffer_stride = static_cast<index_t>(
            pow2_ceil(natural_stride > placement.span ? natural_stride : placement.span));
        if(2ll * buffer_stride > kLdsCapacityBytes)
        {
            out.reason = MxTdmScheduledReason::LsiCapacity;
            return out;
        }
        lds_bytes = max_i(buffer_stride + baseline_span, buffer_stride + placement.span);
    }
    else
    {
        buffer_stride = natural_stride;
        if(natural_stride + pow2_ceil(natural_stride) <= kLdsCapacityBytes)
            buffer_stride = static_cast<index_t>(pow2_ceil(natural_stride));
        lds_bytes = buffer_stride + baseline_span;
    }
    const index_t b_scale_base = placement.scale_base;
    const index_t a_scale_base = b_scale_base + b_scale_region;
    const index_t a_scale_comp = scale_component(a_rows, pads.a_scale);
    const index_t b_scale_comp = scale_component(b_rows, pads.b_scale);

    out.row_bytes = row_bytes;
    out.k_steps   = steps;
    out.a         = {a_rows,
                     placement.a_base,
                     placement.a_stride,
                     pads.a.pad_bytes,
                     pads.a.block_bytes,
                     a.iterate_rows,
                     a.iterate_stride};
    out.b         = {b_rows,
                     placement.b_base,
                     placement.b_stride,
                     pads.b.pad_bytes,
                     pads.b.block_bytes,
                     b.iterate_rows,
                     b.iterate_stride};
    out.a_scale   = {
        a_rows, a_scale_base, a_scale_comp, pads.a_scale.pad_bytes, pads.a_scale.block_bytes, 0, 0};
    out.b_scale = {
        b_rows, b_scale_base, b_scale_comp, pads.b_scale.pad_bytes, pads.b_scale.block_bytes, 0, 0};
    out.buffer_bytes =
        max_i(max_i(placement.a_base + placement.a_stride + a.bytes,
                    placement.b_base + placement.b_stride + b.bytes),
              max_i(a_scale_base + 2 * a_scale_comp, b_scale_base + 2 * b_scale_comp));
    out.buffer_stride = buffer_stride;
    out.lds_bytes     = lds_bytes;
    return out;
}

/// LDS byte offset, in one buffer, of row `row` (0..tile rows) of a data operand.
template <typename T>
CK_TILE_HOST_DEVICE constexpr T
mx_tdm_scheduled_row_offset(const MxTdmScheduledOperand& op, index_t row_bytes, T row)
{
    const T component = row / op.rows;
    const T local     = row % op.rows;
    T linear          = 0;
    if(op.iterate_rows)
    {
        linear = local / op.iterate_rows * op.iterate_stride + local % op.iterate_rows * row_bytes;
    }
    else
    {
        linear = local * row_bytes;
        if(op.pad_bytes > 0)
            linear += linear / op.block_bytes * op.pad_bytes;
    }
    return op.base + component * op.stride + linear;
}

/// LDS byte offset of the four E8M0 scales of row `row` at K step `step`.
template <typename T>
CK_TILE_HOST_DEVICE constexpr T
mx_tdm_scheduled_scale_offset(const MxTdmScheduledOperand& op, T step, T row)
{
    T linear = step * (op.rows * 4) + row % op.rows * 4;
    if(op.pad_bytes > 0)
        linear += linear / op.block_bytes * op.pad_bytes;
    return op.base + row / op.rows * op.stride + linear;
}

namespace mx_tdm_scheduled {

CK_TILE_HOST_DEVICE constexpr MxTdmScheduledReason
pad_reason(MxTdmScheduledLdsPad pad, index_t row_bytes, index_t rows, index_t chunk)
{
    if(pad.pad_bytes < 0 || pad.block_bytes < 0)
        return MxTdmScheduledReason::LdsPad;
    if(!pad.on())
        return MxTdmScheduledReason::Ok;
    if(pad.pad_bytes % kLdsPadStepBytes || pad.pad_bytes > kTdmPadMaxBytes)
        return MxTdmScheduledReason::LdsPad;
    const index_t block = pad.block_bytes;
    if(block <= kTdmPadPeriodMax)
    {
        if(block < 8 || !is_pow2(block))
            return MxTdmScheduledReason::LdsPad;
    }
    else if(row_bytes == 0)
        return MxTdmScheduledReason::LdsPad;
    else if(block % row_bytes || rows % (block / row_bytes))
        return MxTdmScheduledReason::LdsPad;
    if(block % chunk)
        return MxTdmScheduledReason::LdsPad;
    return MxTdmScheduledReason::Ok;
}

CK_TILE_HOST_DEVICE constexpr bool valid_tile_mn(index_t t)
{
    return t == 16 || t == 32 || t == 64 || t == 128 || t == 256;
}

CK_TILE_HOST_DEVICE constexpr bool valid_dtype(MxTdmScheduledDtype t)
{
    return t == MxTdmScheduledDtype::Fp4 || t == MxTdmScheduledDtype::Fp8 ||
           t == MxTdmScheduledDtype::Bf8;
}

} // namespace mx_tdm_scheduled

/**
 * @brief rocKE's validate(): Ok if the spec builds, else the first rule it breaks.
 *
 * @param check_shape  Check the rules that need M, N and K (positive, K a multiple of 32, i32 scale
 *                     offsets, K of at least one tile, grid divisible by the cluster). Off at
 *                     compile time, where only the tile and tuning are known; the kernel checks
 *                     them in IsSupportedArgument.
 */
CK_TILE_HOST_DEVICE constexpr MxTdmScheduledReason
mx_tdm_scheduled_validate(const MxTdmScheduledSpec& s, bool check_shape = true)
{
    using namespace mx_tdm_scheduled;
    using R = MxTdmScheduledReason;
    if(check_shape)
    {
        if(s.M <= 0 || s.N <= 0 || s.K <= 0)
            return R::ShapePositive;
        if(s.K % kBlockK)
            return R::KScaleBlock;
    }
    if(!(s.out == MxTdmScheduledOutDtype::Fp8E4M3 || s.out == MxTdmScheduledOutDtype::Fp16 ||
         s.out == MxTdmScheduledOutDtype::Bf16))
        return R::OutDtype;
    if(!valid_tile_mn(s.tile_m) || !valid_tile_mn(s.tile_n))
        return R::TileMN;
    if(!valid_dtype(s.a_dtype) || !valid_dtype(s.b_dtype))
        return R::Dtype;
    if(MxTdmScheduledSpec::dtype_bits(s.b_dtype) != s.elem_bits())
        return R::DtypeWidth;
    const bool fp4 = s.a_dtype == MxTdmScheduledDtype::Fp4;
    if(!(s.tile_k == 128 || s.tile_k == 256 || (fp4 && (s.tile_k == 512 || s.tile_k == 1024))))
        return R::TileK;
    if(!((s.warp_m == 2 && s.warp_n == 2) || (s.warp_m == 4 && s.warp_n == 1) ||
         (s.warp_m == 1 && s.warp_n == 4)))
        return R::WaveGroup;
    if(!(s.mma == MxTdmScheduledMma::M16N16 || (fp4 && s.mma == MxTdmScheduledMma::M32N16)))
        return R::Mma;
    if(s.tile_m % (s.warp_m * s.atom_m()) || s.tile_n % (s.warp_n * s.atom_n()))
        return R::TileAtoms;
    const index_t rm = s.wave_repeat_m();
    const index_t rn = s.wave_repeat_n();
    if(check_shape)
    {
        // A, B and D offsets switch to i64 past 2^31; the scales' stay i32. A partial last tile's
        // rows reach the next whole tile.
        const long_index_t span_m  = round_up(s.M, s.tile_m);
        const long_index_t span_n  = round_up(s.N, s.tile_n);
        const long_index_t mn_span = span_m > span_n ? span_m : span_n;
        if(mn_span * ceil_div(s.K, kAtomK) * 4 > kI32Span)
            return R::ScaleSpan;
        if(s.K < s.tile_k)
            return R::KShort;
    }
    if(s.local_prefetch != 0 && s.local_prefetch != 1)
        return R::LocalPrefetch;
    if(s.half_plr != MxTdmScheduledHalfPlr::B && s.half_plr != MxTdmScheduledHalfPlr::None)
        return R::HalfPlr;
    if(s.half_plr == MxTdmScheduledHalfPlr::B && (s.local_prefetch != 1 || rn % 2))
        return R::HalfPlrB;
    if(s.half_b_order != MxTdmScheduledHalfBOrder::Ahead &&
       s.half_b_order != MxTdmScheduledHalfBOrder::Late)
        return R::HalfBOrder;
    if(s.half_b_order == MxTdmScheduledHalfBOrder::Late && !s.half_b())
        return R::HalfBLate;
    if(s.signal_wmmas() < 0 || s.signal_wmmas() > s.signal_limit())
        return R::SignalWmmas;
    if(s.gl2_distance != 0 && !(2 <= s.gl2_distance && s.gl2_distance <= 8))
        return R::Gl2Distance;
    if(s.cluster_x < 1 || s.cluster_y < 1 || s.cluster_z < 1)
        return R::ClusterDim;
    if(s.cluster_z != 1 || s.cluster_x * s.cluster_y > 16)
        return R::ClusterSize;
    if(check_shape && (s.grid_x() % s.cluster_x || s.grid_y() % s.cluster_y))
        return R::ClusterGrid;
    const DerivedPads pads  = derived_lds_pads(s);
    const index_t row_bytes = s.tile_k / s.k_per_byte();
    struct PadCheck
    {
        MxTdmScheduledLdsPad pad;
        index_t rows;
        index_t chunk;
        bool data;
    };
    const PadCheck checks[4] = {{pads.a, s.tile_m / 2, 16, true},
                                {pads.b, s.tile_n / 2, 16, true},
                                {pads.a_scale, s.tile_m / 2, min_i(16, 4 * rm), false},
                                {pads.b_scale, s.tile_n / 2, min_i(16, 4 * rn), false}};
    for(const PadCheck& c : checks)
    {
        const R reason = pad_reason(c.pad, c.data ? row_bytes : 0, c.rows, c.chunk);
        if(reason != R::Ok)
            return reason;
        if(c.data && c.pad.on() && c.pad.block_bytes <= kTdmPadPeriodMax &&
           c.pad.block_bytes % row_bytes)
            return R::LdsPad;
    }
    if(rm * rn * s.c_frag_len() > kVgprs)
        return R::Accumulators;
    const MxTdmScheduledLdsLayout layout = mx_tdm_scheduled_lds_layout(s);
    if(layout.reason != R::Ok)
        return layout.reason;
    if(layout.lds_bytes > kLdsCapacityBytes)
        return R::LdsCapacity;
    return R::Ok;
}

// Names (host only).

CK_TILE_HOST const char* mx_tdm_scheduled_dtype_name(MxTdmScheduledDtype t)
{
    switch(t)
    {
    case MxTdmScheduledDtype::Fp4: return "fp4";
    case MxTdmScheduledDtype::Fp8: return "fp8";
    case MxTdmScheduledDtype::Bf8: return "bf8";
    }
    return "unknown";
}

CK_TILE_HOST const char* mx_tdm_scheduled_out_dtype_name(MxTdmScheduledOutDtype t)
{
    switch(t)
    {
    case MxTdmScheduledOutDtype::Fp8E4M3: return "fp8e4m3";
    case MxTdmScheduledOutDtype::Fp16: return "fp16";
    case MxTdmScheduledOutDtype::Bf16: return "bf16";
    }
    return "unknown";
}

/**
 * @brief The part of rocKE's kernel name after the tile: the cluster, then every value that is
 *        not its default (out dtype, wave group, pf, cs, then the lp0 / hplrnone / hblate /
 *        lsi0 / mma16 flags), each with a leading "_".
 *
 * @param with_shape_parts  Include the out dtype, wave group and mma16, which a CK kernel name
 *                          already carries in its tile and type fields.
 */
CK_TILE_HOST std::string mx_tdm_scheduled_tuning_name(const MxTdmScheduledSpec& s,
                                                      bool with_shape_parts = true)
{
    std::string name = "_c" + std::to_string(s.cluster_x) + "x" + std::to_string(s.cluster_y);
    if(with_shape_parts && s.out != MxTdmScheduledOutDtype::Fp8E4M3)
        name += std::string("_") + mx_tdm_scheduled_out_dtype_name(s.out);
    if(with_shape_parts && !(s.warp_m == 2 && s.warp_n == 2))
        name += "_wg" + std::to_string(s.warp_m) + "x" + std::to_string(s.warp_n);
    if(s.gl2_distance != 4)
        name += "_pf" + std::to_string(s.gl2_distance);
    if(s.signal_wmmas_in >= 0)
        name += "_cs" + std::to_string(s.signal_wmmas_in);
    if(s.local_prefetch == 0)
        name += "_lp0";
    if(s.half_plr == MxTdmScheduledHalfPlr::None)
        name += "_hplrnone";
    if(s.half_b_late())
        name += "_hblate";
    if(!s.lds_segment_interleave)
        name += "_lsi0";
    if(with_shape_parts && s.mma == MxTdmScheduledMma::M16N16)
        name += "_mma16";
    return name;
}

/// rocKE's MxGemmSpec.kernel_name() for a spec named `prefix`.
CK_TILE_HOST std::string mx_tdm_scheduled_kernel_name(const MxTdmScheduledSpec& s,
                                                      const std::string& prefix = "mx_gemm")
{
    std::string tag = std::string("mx") + mx_tdm_scheduled_dtype_name(s.a_dtype);
    if(s.b_dtype != s.a_dtype)
        tag += mx_tdm_scheduled_dtype_name(s.b_dtype);
    return prefix + "_" + tag + "_M" + std::to_string(s.M) + "N" + std::to_string(s.N) + "K" +
           std::to_string(s.K) + "_t" + std::to_string(s.tile_m) + "x" + std::to_string(s.tile_n) +
           "x" + std::to_string(s.tile_k) + mx_tdm_scheduled_tuning_name(s);
}

} // namespace ck_tile
