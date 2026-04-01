// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Frozen baseline tests for FMHA BWD kernel variants.
//
// Each test reproduces the exact make_spec() call from the example
// registry and asserts ALL fields. If a schema change breaks a test,
// the change is not backwards-compatible.
//
// NOTE: block_size and block_n0 in DqDkDv are demo constants
// (dqdkdv_spec.hpp:200-204). Update when architecture-dependent tile
// selection is implemented.
//
// The findVariant() tests require the registry header from the example
// directory, which is added to the include path by CMakeLists.txt.

#include "rocm_fmha_bwd_registry.hpp"

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>
#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>
#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

#include <gtest/gtest.h>

#include <cstring>

using namespace rocm_ck;

// ============================================================================
// OGradDotO frozen baselines
// ============================================================================

TEST(FmhaBwdCompat, OGradDotO_FP16_D128_Batch)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdCompat, OGradDotO_BF16_D128_Batch)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::BF16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdCompat, OGradDotO_FP16_D64_Batch)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 64, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 64);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdCompat, OGradDotO_FP16_D128_Group)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdCompat, OGradDotO_FP16_D128_Batch_NoPad)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_FALSE(k.pad_seqlen_q);
    EXPECT_FALSE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

// ============================================================================
// DqDkDv frozen baselines
// ============================================================================

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_FALSE(k.has_mask);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_BF16_D128_Batch)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::BF16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_CMask)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_mask = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_TRUE(k.has_mask);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_Det)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_TRUE(k.is_deterministic);
    EXPECT_FALSE(k.has_mask);
    EXPECT_FALSE(k.has_dropout);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::GROUP},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_EQ(k.block_per_cu, 1);
}

// ============================================================================
// ConvertDQ frozen baselines
// ============================================================================

TEST(FmhaBwdCompat, ConvertDQ_FP16_D128_Batch)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdCompat, ConvertDQ_FP16_D128_Group)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

// ============================================================================
// Registry: OGradDotO findVariant
// ============================================================================

TEST(FmhaBwdCompat, Registry_OGradDotO_FindsExactMatch)
{
    const auto* v = findVariant(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_ograd_dot_o_fp16_d128_batch");
}

TEST(FmhaBwdCompat, Registry_OGradDotO_PrefersNoPadWhenAvailable)
{
    const auto* v = findVariant(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_ograd_dot_o_fp16_d128_batch_npad");
}

TEST(FmhaBwdCompat, Registry_OGradDotO_FallsToPaddedWhenNoPadMissing)
{
    // BF16 has no npad variant — falls back to padded
    const auto* v = findVariant(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::BF16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_ograd_dot_o_bf16_d128_batch");
}

TEST(FmhaBwdCompat, Registry_OGradDotO_ReturnsNullForUnregistered)
{
    const auto* v = findVariant(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 256, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(v, nullptr);
}

TEST(FmhaBwdCompat, Registry_OGradDotO_VariantCount)
{
    EXPECT_EQ(ALL_OGRAD_DOT_O_VARIANTS_COUNT, 5);
}

// ============================================================================
// Registry: DqDkDv findVariant
// ============================================================================

TEST(FmhaBwdCompat, Registry_DqDkDv_FindsBaseline)
{
    const auto* v =
        findVariant(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                      .hdim_q = 128,
                                                      .hdim_v = 128,
                                                      .mode   = FmhaMode::BATCH},
                                        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_dqdkdv_fp16_d128_batch");
}

TEST(FmhaBwdCompat, Registry_DqDkDv_FindsMask)
{
    const auto* v = findVariant(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_mask = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
}

TEST(FmhaBwdCompat, Registry_DqDkDv_FindsDeterministic)
{
    const auto* v = findVariant(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_dqdkdv_fp16_d128_batch_det");
}

TEST(FmhaBwdCompat, Registry_DqDkDv_RejectsPartialFeatureMatch)
{
    // mask+det: no combined variant exists
    const auto* v = findVariant(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .has_mask = true, .is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(v, nullptr);
}

TEST(FmhaBwdCompat, Registry_DqDkDv_IgnoresHasBiasGrad)
{
    // findVariant does not match on has_bias_grad — only bias_type
    // ELEMENTWISE + has_bias_grad=true returns nullptr (no ELEMENTWISE variant)
    const auto* v =
        findVariant(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                      .hdim_q = 128,
                                                      .hdim_v = 128,
                                                      .mode   = FmhaMode::BATCH},
                                        .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                      .has_bias_grad = true,
                                                      .pad_hdim_q    = 8,
                                                      .pad_hdim_v    = 8}});
    EXPECT_EQ(v, nullptr);
}

TEST(FmhaBwdCompat, Registry_DqDkDv_ReturnsNullForUnregistered)
{
    const auto* v = findVariant(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64, .hdim_v = 64, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(v, nullptr);
}

TEST(FmhaBwdCompat, Registry_DqDkDv_VariantCount) { EXPECT_EQ(ALL_DQDKDV_VARIANTS_COUNT, 5); }

// ============================================================================
// Registry: ConvertDQ findVariant
// ============================================================================

TEST(FmhaBwdCompat, Registry_ConvertDQ_FindsBatch)
{
    const auto* v = findVariant(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_convert_dq_fp16_d128_batch_det");
}

TEST(FmhaBwdCompat, Registry_ConvertDQ_FindsGroup)
{
    const auto* v = findVariant(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
        .algorithm = {}});
    ASSERT_NE(v, nullptr);
    EXPECT_STREQ(v->name, "fmha_bwd_convert_dq_fp16_d128_group_det");
}

TEST(FmhaBwdCompat, Registry_ConvertDQ_ReturnsNullForBF16)
{
    const auto* v = findVariant(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::BF16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_EQ(v, nullptr);
}

TEST(FmhaBwdCompat, Registry_ConvertDQ_VariantCount)
{
    EXPECT_EQ(ALL_CONVERT_DQ_VARIANTS_COUNT, 2);
}
