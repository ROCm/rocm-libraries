// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

#include <type_traits>

namespace ck_tile {

/**
 * @brief How the gfx125 padded LDS layout inserts padding along K.
 *
 * The gfx125 GEMM LDS descriptors avoid bank conflicts by padding rather than by an XOR swizzle.
 * The padded layout stores groups of `rows_per_group` K rows back to back and inserts `pad_elems`
 * elements after every group, so the stride between groups is
 * `rows_per_group * KPerBlock + pad_elems` elements. The TDM mover produces the same layout in
 * hardware from a (pad interval, pad amount) pair, so both are derived from one spec.
 *
 * - Auto:        the policy's built-in rule (16 B after every max(256 B, one K row)); this is the
 *                default and keeps existing instances byte-identical.
 * - None:        no padding.
 * - PerRow:      `PadElems` elements after every K row.
 * - PerInterval: `PadElems` elements after every `IntervalElems` elements; `IntervalElems` must
 *                be a whole number of K rows.
 */
enum class LdsPadMode
{
    Auto,
    None,
    PerRow,
    PerInterval
};

/**
 * @brief Compile-time LDS K padding spec for one GEMM operand.
 *
 * @tparam Mode           Padding mode.
 * @tparam PadElems       Padding inserted per interval, in elements (PerRow, PerInterval).
 * @tparam IntervalElems  Padding interval in elements (PerInterval only).
 */
template <LdsPadMode Mode = LdsPadMode::Auto, index_t PadElems = 0, index_t IntervalElems = 0>
struct LdsKPadding
{
    static constexpr LdsPadMode mode        = Mode;
    static constexpr index_t pad_elems      = PadElems;
    static constexpr index_t interval_elems = IntervalElems;
    static constexpr bool is_auto           = Mode == LdsPadMode::Auto;

    static_assert(PadElems >= 0 && IntervalElems >= 0, "LDS padding must be non-negative");
    static_assert(Mode != LdsPadMode::Auto || (PadElems == 0 && IntervalElems == 0),
                  "Auto LDS padding takes no parameters");
    static_assert(Mode != LdsPadMode::None || (PadElems == 0 && IntervalElems == 0),
                  "None LDS padding takes no parameters");
    static_assert(Mode != LdsPadMode::PerRow || (PadElems > 0 && IntervalElems == 0),
                  "PerRow LDS padding needs PadElems > 0 and no interval");
    static_assert(Mode != LdsPadMode::PerInterval || (PadElems > 0 && IntervalElems > 0),
                  "PerInterval LDS padding needs PadElems > 0 and IntervalElems > 0");
};

using LdsKPadAuto = LdsKPadding<LdsPadMode::Auto>;
using LdsKPadNone = LdsKPadding<LdsPadMode::None>;
template <index_t PadElems>
using LdsKPadPerRow = LdsKPadding<LdsPadMode::PerRow, PadElems>;
template <index_t PadElems, index_t IntervalElems>
using LdsKPadPerInterval = LdsKPadding<LdsPadMode::PerInterval, PadElems, IntervalElems>;

/**
 * @brief Attaches LDS K padding specs for A and B to an existing GEMM traits type.
 *
 * The pipeline problem exposes the traits as `Problem::Traits`, so the spec reaches every policy
 * and pipeline through the problem and the policies stay non-template. Traits types without the
 * members resolve to Auto.
 */
template <typename Traits_, typename LdsPadSpecA_, typename LdsPadSpecB_ = LdsPadSpecA_>
struct WithLdsPad : public Traits_
{
    using LdsPadSpecA = LdsPadSpecA_;
    using LdsPadSpecB = LdsPadSpecB_;
};

namespace detail {

template <typename Problem, typename = void>
struct lds_pad_spec_a
{
    using type = LdsKPadAuto;
};

template <typename Problem>
struct lds_pad_spec_a<Problem, std::void_t<typename Problem::Traits::LdsPadSpecA>>
{
    using type = typename Problem::Traits::LdsPadSpecA;
};

template <typename Problem, typename = void>
struct lds_pad_spec_b
{
    using type = LdsKPadAuto;
};

template <typename Problem>
struct lds_pad_spec_b<Problem, std::void_t<typename Problem::Traits::LdsPadSpecB>>
{
    using type = typename Problem::Traits::LdsPadSpecB;
};

CK_TILE_HOST_DEVICE constexpr index_t lds_pad_log2_floor(index_t x)
{
    index_t result = 0;
    while(x > 1)
    {
        x >>= 1;
        ++result;
    }
    return result;
}

} // namespace detail

/// @brief LDS K padding spec of operand A (IsA) or B carried on the problem, Auto by default.
template <typename Problem, bool IsA>
using lds_pad_spec_t = std::conditional_t<IsA,
                                          typename detail::lds_pad_spec_a<Problem>::type,
                                          typename detail::lds_pad_spec_b<Problem>::type>;

/**
 * @brief TDM pad codes for a padded LDS layout: (enable, pad_amount, pad_interval).
 *
 * pad_amount is the pad in dwords minus one (7-bit field), pad_interval is log2 of the interval in
 * dwords minus one (3-bit field). Mirrors detail::make_fmha_bwd_tdm_padding_config in
 * ops/fmha/pipeline/fmha_bwd_tdm_padding.hpp, expressed in bytes so packed types work.
 */
template <index_t IntervalBytes, index_t PadBytes>
CK_TILE_HOST_DEVICE constexpr auto make_tdm_lds_padding_config()
{
    constexpr index_t dword_bytes     = 4;
    constexpr index_t pad_dwords      = PadBytes / dword_bytes;
    constexpr index_t interval_dwords = IntervalBytes / dword_bytes;
    static_assert(PadBytes >= 0 && IntervalBytes > 0);
    static_assert(pad_dwords * dword_bytes == PadBytes, "LDS pad must be a whole number of dwords");
    static_assert(interval_dwords * dword_bytes == IntervalBytes,
                  "LDS pad interval must be a whole number of dwords");
    static_assert(pad_dwords <= 128, "pad_amount must fit its 7-bit biased field");

    if constexpr(pad_dwords == 0)
    {
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }
    else
    {
        static_assert(interval_dwords >= 2 && interval_dwords <= 256 &&
                          (interval_dwords & (interval_dwords - 1)) == 0,
                      "pad_interval must be a power of two in [2, 256] dwords");
        return make_tuple(number<true>{},
                          number<pad_dwords - 1>{},
                          number<detail::lds_pad_log2_floor(interval_dwords) - 1>{});
    }
}

/**
 * @brief Resolves an explicit (non-Auto) LDS K padding spec for one operand.
 *
 * @tparam DataType   LDS element type (packed types are handled through PackedSize).
 * @tparam KPerBlock  K extent of one LDS row, in elements.
 * @tparam Spec       An LdsKPadding spec other than Auto.
 *
 * Auto is resolved by the policies themselves, which keeps the existing layouts unchanged.
 */
template <typename DataType, index_t KPerBlock, typename Spec>
struct ResolvedLdsPad
{
    static_assert(!Spec::is_auto, "Auto LDS padding is resolved by the GEMM policy");

    static constexpr index_t kDwordBytes = 4;
    static constexpr index_t kPackedSize = numeric_traits<remove_cvref_t<DataType>>::PackedSize;

    CK_TILE_HOST_DEVICE static constexpr index_t elems_to_bytes(index_t elems)
    {
        return elems * static_cast<index_t>(sizeof(DataType)) / kPackedSize;
    }

    static constexpr index_t k_row_bytes = elems_to_bytes(KPerBlock);
    static_assert(k_row_bytes % kDwordBytes == 0, "LDS K row must be a whole number of dwords");

    /// K rows stored back to back between two pads.
    static constexpr index_t rows_per_group = [] {
        if constexpr(Spec::mode == LdsPadMode::PerInterval)
        {
            static_assert(Spec::interval_elems % KPerBlock == 0,
                          "PerInterval LDS padding interval must be a whole number of K rows");
            return Spec::interval_elems / KPerBlock;
        }
        else
        {
            return index_t{1};
        }
    }();

    static constexpr index_t pad_elems          = Spec::pad_elems;
    static constexpr index_t interval_elems     = rows_per_group * KPerBlock;
    static constexpr index_t group_stride_elems = interval_elems + pad_elems;

    static constexpr index_t pad_bytes      = elems_to_bytes(pad_elems);
    static constexpr index_t interval_bytes = elems_to_bytes(interval_elems);
    static constexpr index_t pad_dw         = pad_bytes / kDwordBytes;
    static constexpr index_t interval_dw    = interval_bytes / kDwordBytes;

    static_assert(pad_elems * static_cast<index_t>(sizeof(DataType)) % kPackedSize == 0,
                  "LDS pad must be a whole number of bytes");
    // Every group must start 16 B aligned so that ds_load_b128 stays legal.
    static_assert(pad_bytes % 16 == 0, "LDS pad must be a multiple of 16 bytes");

    /// TDM pad codes in the (enable, pad_amount, pad_interval) form of GetLdsPaddingConfig.
    /// The TDM encoding limits (pad <= 128 dwords, interval a power of two in [2, 256] dwords)
    /// are checked here, so they only apply to pipelines that program the TDM.
    CK_TILE_HOST_DEVICE static constexpr auto get_padding_config()
    {
        return make_tdm_lds_padding_config<interval_bytes, pad_bytes>();
    }

    static constexpr bool pad_enable = pad_dw > 0;
};

/**
 * @brief Selects the padded-layout geometry (rows per group, pad elements) for a spec.
 *
 * @tparam MNPerBlock  Number of K rows in the LDS tile (MPerBlock for A, NPerBlock for B).
 *
 * Auto forwards the policy's own values unchanged; explicit specs use ResolvedLdsPad.
 */
template <typename Spec,
          typename DataType,
          index_t MNPerBlock,
          index_t KPerBlock,
          index_t AutoRowsPerGroup,
          index_t AutoPadElems>
struct LdsKPadLayout
{
    using Resolved = ResolvedLdsPad<DataType, KPerBlock, Spec>;

    static constexpr index_t rows_per_group = Resolved::rows_per_group;
    static constexpr index_t pad_elems      = Resolved::pad_elems;

    // The descriptors split the tile into MNPerBlock / rows_per_group groups; a remainder would
    // be silently dropped from the LDS allocation and overlap the next buffer.
    static_assert(MNPerBlock % rows_per_group == 0,
                  "LDS pad group (rows per interval) must divide MPerBlock / NPerBlock");
};

template <typename DataType,
          index_t MNPerBlock,
          index_t KPerBlock,
          index_t AutoRowsPerGroup,
          index_t AutoPadElems>
struct LdsKPadLayout<LdsKPadAuto, DataType, MNPerBlock, KPerBlock, AutoRowsPerGroup, AutoPadElems>
{
    static constexpr index_t rows_per_group = AutoRowsPerGroup;
    static constexpr index_t pad_elems      = AutoPadElems;
};

} // namespace ck_tile
