// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"

#include "gtest/gtest.h"

namespace {

using QKPad = ck_tile::detail::LdsPaddingConfig<true, 256, 16>;
using VPad  = ck_tile::detail::LdsPaddingConfig<true, 256, 32>;
using NoPad = ck_tile::detail::LdsPaddingConfig<false, 0, 0>;

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

TEST(QrTdmLdsPadding, CompileTimeConfiguration) { SUCCEED(); }

} // namespace
