// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/dqdkdv_api.hpp>
#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;
namespace S = fmha_bwd_dqdkdv_slots;

// ============================================================================
// Algorithm defaults
// ============================================================================

TEST(FmhaBwdDqDkDv, AlgorithmDefaults)
{
    constexpr FmhaBwdDQDKDVAlgorithm algo{};
    EXPECT_EQ(algo.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(algo.has_bias_grad);
    EXPECT_FALSE(algo.has_mask);
    EXPECT_FALSE(algo.has_dropout);
    EXPECT_FALSE(algo.is_deterministic);
    EXPECT_EQ(algo.pad_hdim_q, 0);
    EXPECT_EQ(algo.pad_hdim_v, 0);
    EXPECT_EQ(algo.block_per_cu, -1);
}

// ============================================================================
// make_spec happy path
// ============================================================================

TEST(FmhaBwdDqDkDv, MakeSpecBaseline)
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
    EXPECT_EQ(k.block_per_cu, 1); // auto-resolved from -1
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdDqDkDv, MakeSpecBF16)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::BF16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.dtype, DataType::BF16);
}

TEST(FmhaBwdDqDkDv, MakeSpecAllHdimsQ)
{
    constexpr auto k32 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 32,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k64 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 64,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k96 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 96,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k128 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k256 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 256,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k32.hdim_q, 32);
    EXPECT_EQ(k64.hdim_q, 64);
    EXPECT_EQ(k96.hdim_q, 96);
    EXPECT_EQ(k128.hdim_q, 128);
    EXPECT_EQ(k256.hdim_q, 256);
}

TEST(FmhaBwdDqDkDv, MakeSpecAllHdimsV)
{
    constexpr auto k32 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 32,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k64 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 64,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k96 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 96,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k128 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k256 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 256,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k32.hdim_v, 32);
    EXPECT_EQ(k64.hdim_v, 64);
    EXPECT_EQ(k96.hdim_v, 96);
    EXPECT_EQ(k128.hdim_v, 128);
    EXPECT_EQ(k256.hdim_v, 256);
}

TEST(FmhaBwdDqDkDv, MakeSpecWithMask)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_mask = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_TRUE(k.has_mask);
}

TEST(FmhaBwdDqDkDv, MakeSpecWithDropout)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_TRUE(k.has_dropout);
}

TEST(FmhaBwdDqDkDv, MakeSpecDeterministic)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_TRUE(k.is_deterministic);
}

TEST(FmhaBwdDqDkDv, MakeSpecBiasElementwise)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
}

TEST(FmhaBwdDqDkDv, MakeSpecBiasAlibi)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ALIBI, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.bias_type, FmhaBiasType::ALIBI);
}

TEST(FmhaBwdDqDkDv, MakeSpecBiasGrad)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                    .has_bias_grad = true,
                                                    .pad_hdim_q    = 8,
                                                    .pad_hdim_v    = 8}});
    EXPECT_TRUE(k.has_bias_grad);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
}

TEST(FmhaBwdDqDkDv, MakeSpecGroupMode)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::GROUP},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
}

TEST(FmhaBwdDqDkDv, MakeSpecGroupModePartialPadQ)
{
    // GROUP with pad_hdim_q=8, pad_hdim_v=0 is valid (AND condition)
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::GROUP},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 0}});
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 0);
}

TEST(FmhaBwdDqDkDv, MakeSpecGroupModePartialPadV)
{
    // GROUP with pad_hdim_q=0, pad_hdim_v=8 is valid (AND condition)
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::GROUP},
                                      .algorithm = {.pad_hdim_q = 0, .pad_hdim_v = 8}});
    EXPECT_EQ(k.pad_hdim_q, 0);
    EXPECT_EQ(k.pad_hdim_v, 8);
}

TEST(FmhaBwdDqDkDv, MakeSpecAllPadValues)
{
    // All valid pad values: 0, 1, 8
    constexpr auto k0 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 0, .pad_hdim_v = 0}});
    constexpr auto k1 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 1, .pad_hdim_v = 1}});
    constexpr auto k8 =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k0.pad_hdim_q, 0);
    EXPECT_EQ(k1.pad_hdim_q, 1);
    EXPECT_EQ(k8.pad_hdim_q, 8);
}

TEST(FmhaBwdDqDkDv, MakeSpecExplicitBlockPerCu)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8, .block_per_cu = 2}});
    EXPECT_EQ(k.block_per_cu, 2);
}

TEST(FmhaBwdDqDkDv, BlockPerCuAutoResolvesToOne)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}}); // block_per_cu defaults to -1
    EXPECT_EQ(k.block_per_cu, 1);
}

// ============================================================================
// Slot constants
// ============================================================================

TEST(FmhaBwdDqDkDv, TensorSlotIndicesFixed)
{
    EXPECT_EQ(S::Q, 0);
    EXPECT_EQ(S::K, 1);
    EXPECT_EQ(S::V, 2);
    EXPECT_EQ(S::LSE, 3);
    EXPECT_EQ(S::DO, 4);
    EXPECT_EQ(S::D, 5);
    EXPECT_EQ(S::DQ_ACC, 6);
    EXPECT_EQ(S::DK, 7);
    EXPECT_EQ(S::DV, 8);
    EXPECT_EQ(S::BIAS, 9);
    EXPECT_EQ(S::DBIAS, 10);
    EXPECT_EQ(S::RANDVAL, 11);
}

TEST(FmhaBwdDqDkDv, ScalarSlotIndicesFixed)
{
    EXPECT_EQ(S::RAW_SCALE, 0);
    EXPECT_EQ(S::SCALE, 1);
    EXPECT_EQ(S::NUM_HEAD_Q, 2);
    EXPECT_EQ(S::NHEAD_RATIO_QK, 3);
    EXPECT_EQ(S::P_UNDROP, 4);
    EXPECT_EQ(S::RP_UNDROP, 5);
    EXPECT_EQ(S::DROP_SEED, 6);
    EXPECT_EQ(S::DROP_OFFSET, 7);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsPlain)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k), 9);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsWithBias)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                    .has_bias_grad = false,
                                                    .pad_hdim_q    = 8,
                                                    .pad_hdim_v    = 8}});
    EXPECT_EQ(S::requiredTensors(k), 10);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsWithBiasGrad)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                    .has_bias_grad = true,
                                                    .pad_hdim_q    = 8,
                                                    .pad_hdim_v    = 8}});
    EXPECT_EQ(S::requiredTensors(k), 11);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsWithDropout)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k), 12);
}

TEST(FmhaBwdDqDkDv, RequiredScalarsPlain)
{
    constexpr auto k =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredScalars(k), 4);
}

TEST(FmhaBwdDqDkDv, RequiredScalarsWithDropout)
{
    constexpr auto k = make_spec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredScalars(k), 8);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsGroupSameAsBatch)
{
    constexpr auto k_batch =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::BATCH},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k_group =
        make_spec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                    .hdim_q = 128,
                                                    .hdim_v = 128,
                                                    .mode   = FmhaMode::GROUP},
                                      .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k_batch), S::requiredTensors(k_group));
}

// ============================================================================
// Grid size
// ============================================================================

TEST(FmhaBwdDqDkDv, GridSizeBasic)
{
    constexpr auto g = dqdkdv_grid_size(2, 8, 256, 128);
    EXPECT_EQ(g.x, 2u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdDqDkDv, GridSizeCeil)
{
    constexpr auto g = dqdkdv_grid_size(1, 1, 129, 128);
    EXPECT_EQ(g.x, 2u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}
