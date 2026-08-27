// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"

#include "gtest/gtest.h"

namespace {

using QKPad = ck_tile::detail::LdsPaddingConfig<true, 256, 16>;
using VPad  = ck_tile::detail::LdsPaddingConfig<true, 256, 32>;
using NoPad = ck_tile::detail::LdsPaddingConfig<false, 0, 0>;

template <typename DataType, typename Descriptor>
constexpr ck_tile::index_t
byte_offset(const Descriptor& descriptor, ck_tile::index_t row, ck_tile::index_t col)
{
    return descriptor.calculate_offset(ck_tile::make_tuple(row, col)) * sizeof(DataType);
}

template <typename DataType>
using PaddedQDescriptor = decltype(
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<DataType, 128, 128, QKPad, 16>());

struct QTag
{
    using PaddingConfig                                   = QKPad;
    [[maybe_unused]] static constexpr ck_tile::index_t Id = 0;
    [[maybe_unused]] static constexpr bool kTranspose     = false;
};

struct KTag
{
    using PaddingConfig                                   = QKPad;
    [[maybe_unused]] static constexpr ck_tile::index_t Id = 1;
    [[maybe_unused]] static constexpr bool kTranspose     = false;
};

struct VTag
{
    using PaddingConfig                                   = VPad;
    [[maybe_unused]] static constexpr ck_tile::index_t Id = 2;
    [[maybe_unused]] static constexpr bool kTranspose     = true;
};

template <ck_tile::index_t M>
using TestFmhaShape = ck_tile::TileFmhaShape<ck_tile::sequence<M, 64, 32, 128, 32, 128>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              true>;

using TestFmhaTraits =
    ck_tile::TileFmhaTraits<false,
                            false,
                            false,
                            false,
                            false,
                            ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                            false,
                            false,
                            false,
                            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE>;

template <typename DataType, ck_tile::index_t M>
using TestFmhaProblem =
    ck_tile::BlockFmhaPipelineProblem<DataType,
                                      DataType,
                                      DataType,
                                      float,
                                      float,
                                      DataType,
                                      uint8_t,
                                      float,
                                      DataType,
                                      float,
                                      DataType,
                                      TestFmhaShape<M>,
                                      false,
                                      ck_tile::ComposedAttention<0>,
                                      ck_tile::SimplifiedGenericAttentionMask<false>,
                                      false,
                                      TestFmhaTraits>;

static_assert(ck_tile::detail::is_valid_lds_padding_config_v<true, 256, 16>);
static_assert(ck_tile::detail::is_valid_lds_padding_config_v<true, 256, 32>);
static_assert(ck_tile::detail::is_valid_lds_padding_config_v<false, 0, 0>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<false, 256, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 0, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 192, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 2048, 16>);
static_assert(!ck_tile::detail::is_valid_lds_padding_config_v<true, 256, 516>);

static_assert(ck_tile::detail::EncodedTdmPadding<QKPad>::kEnabled);
static_assert(ck_tile::detail::EncodedTdmPadding<QKPad>::kPadInterval == 5);
static_assert(ck_tile::detail::EncodedTdmPadding<QKPad>::kPadAmount == 3);
static_assert(ck_tile::detail::EncodedTdmPadding<VPad>::kPadInterval == 5);
static_assert(ck_tile::detail::EncodedTdmPadding<VPad>::kPadAmount == 7);
static_assert(!ck_tile::detail::EncodedTdmPadding<NoPad>::kEnabled);
static_assert(ck_tile::detail::EncodedTdmPadding<NoPad>::kPadInterval == 0);
static_assert(ck_tile::detail::EncodedTdmPadding<NoPad>::kPadAmount == 0);

constexpr auto q_nopad_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t,
                                                          128,
                                                          128,
                                                          NoPad,
                                                          16>();
static_assert(byte_offset<ck_tile::bf16_t>(q_nopad_bf16_desc, 0, 127) == 254);
static_assert(byte_offset<ck_tile::bf16_t>(q_nopad_bf16_desc, 1, 0) == 256);
static_assert(q_nopad_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 32768);

constexpr auto q_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t,
                                                          128,
                                                          128,
                                                          QKPad,
                                                          16>();
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 0, 0) == 0);
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 0, 127) == 254);
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 1, 0) == 272);
static_assert(byte_offset<ck_tile::bf16_t>(q_bf16_desc, 127, 127) == 34798);
static_assert(q_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 34800);

constexpr auto k_prefill_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t,
                                                          64,
                                                          128,
                                                          QKPad,
                                                          16>();
static_assert(k_prefill_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 17392);

constexpr auto k_decode_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t,
                                                          64,
                                                          32,
                                                          QKPad,
                                                          16>();
static_assert(k_decode_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 4336);

constexpr auto v_bf16_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::bf16_t,
                                                          64,
                                                          128,
                                                          VPad,
                                                          16>();
static_assert(byte_offset<ck_tile::bf16_t>(v_bf16_desc, 1, 0) == 288);
static_assert(v_bf16_desc.get_element_space_size() * sizeof(ck_tile::bf16_t) == 18400);

constexpr auto q_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t,
                                                          128,
                                                          128,
                                                          QKPad,
                                                          16>();
static_assert(byte_offset<ck_tile::half_t>(q_half_desc, 1, 0) == 272);
static_assert(q_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 34800);

constexpr auto k_prefill_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t,
                                                          64,
                                                          128,
                                                          QKPad,
                                                          16>();
static_assert(k_prefill_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 17392);

constexpr auto k_decode_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t,
                                                          64,
                                                          32,
                                                          QKPad,
                                                          16>();
static_assert(k_decode_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 4336);

constexpr auto v_half_desc =
    ck_tile::detail::make_qr_tdm_row_major_lds_descriptor<ck_tile::half_t,
                                                          64,
                                                          128,
                                                          VPad,
                                                          16>();
static_assert(byte_offset<ck_tile::half_t>(v_half_desc, 1, 0) == 288);
static_assert(v_half_desc.get_element_space_size() * sizeof(ck_tile::half_t) == 18400);

static_assert(!ck_tile::is_detected<PaddedQDescriptor, ck_tile::pk_fp4_t>::value);

template <typename DataType>
constexpr bool validate_production_geometries()
{
    using PrefillProblem = TestFmhaProblem<DataType, 128>;
    using DecodeProblem  = TestFmhaProblem<DataType, 64>;

    return ck_tile::detail::validate_qr_tdm_issue_geometry<QTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<KTag, PrefillProblem, true>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<VTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<QTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<KTag, DecodeProblem, false>() &&
           ck_tile::detail::validate_qr_tdm_issue_geometry<VTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<QTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<KTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<VTag, PrefillProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<QTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<KTag, DecodeProblem>() &&
           ck_tile::detail::validate_qr_tdm_reader_segments<VTag, DecodeProblem>();
}

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
static_assert(validate_production_geometries<ck_tile::bf16_t>());
static_assert(validate_production_geometries<ck_tile::half_t>());
#endif

TEST(QrTdmLdsPadding, CompileTimeConfiguration) { SUCCEED(); }

} // namespace
