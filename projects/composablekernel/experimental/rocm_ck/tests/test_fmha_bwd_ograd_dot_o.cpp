// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_api.hpp>
#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;
namespace S = fmha_bwd_ograd_dot_o_slots;

// ============================================================================
// Algorithm defaults
// ============================================================================

TEST(FmhaBwdOGradDotO, AlgorithmDefaults)
{
    constexpr FmhaBwdOGradDotOAlgorithm algo{.pad_seqlen_q = true, .pad_hdim_v = true};
    EXPECT_EQ(algo.block_per_cu, 2);
    EXPECT_EQ(algo.block_size, 64);
}

// ============================================================================
// make_spec happy path
// ============================================================================

TEST(FmhaBwdOGradDotO, MakeSpecFP16Batch)
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

TEST(FmhaBwdOGradDotO, MakeSpecBF16)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::BF16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(k.dtype, DataType::BF16);
}

TEST(FmhaBwdOGradDotO, MakeSpecGroupMode)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.pad_seqlen_q);
}

TEST(FmhaBwdOGradDotO, MakeSpecAllHdims)
{
    // make_spec is consteval — cannot use runtime loop variables.
    // Use separate constexpr variables for each hdim value.
    constexpr auto k32  = make_spec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 32, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k64  = make_spec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 64, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k96  = make_spec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 96, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k128 = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k256 = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 256, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k32.hdim_v, 32);
    EXPECT_EQ(k64.hdim_v, 64);
    EXPECT_EQ(k96.hdim_v, 96);
    EXPECT_EQ(k128.hdim_v, 128);
    EXPECT_EQ(k256.hdim_v, 256);
}

TEST(FmhaBwdOGradDotO, MakeSpecNoPadBatch)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});
    EXPECT_FALSE(k.pad_seqlen_q);
    EXPECT_FALSE(k.pad_hdim_v);
}

TEST(FmhaBwdOGradDotO, MakeSpecCustomBlockPerCu)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true, .block_per_cu = 4}});
    EXPECT_EQ(k.block_per_cu, 4);
}

TEST(FmhaBwdOGradDotO, MakeSpecCustomBlockSize)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true, .block_size = 128}});
    EXPECT_EQ(k.block_size, 128);
}

// ============================================================================
// Slot constants
// ============================================================================

TEST(FmhaBwdOGradDotO, TensorSlotIndicesAreSequential)
{
    EXPECT_EQ(S::O, 0);
    EXPECT_EQ(S::DO, 1);
    EXPECT_EQ(S::D, 2);
    EXPECT_EQ(S::SEQSTART_Q, 3);
    EXPECT_EQ(S::SEQLEN_Q, 4);
}

TEST(FmhaBwdOGradDotO, ScalarSlotIndices) { EXPECT_EQ(S::P_UNDROP, 0); }

TEST(FmhaBwdOGradDotO, RequiredTensorsBatch)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(S::requiredTensors(k), 3);
}

TEST(FmhaBwdOGradDotO, RequiredTensorsGroup)
{
    constexpr auto k = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(S::requiredTensors(k), 5);
}

TEST(FmhaBwdOGradDotO, RequiredScalarsAlways1)
{
    constexpr auto k_batch = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k_group = make_spec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(S::requiredScalars(k_batch), 1);
    EXPECT_EQ(S::requiredScalars(k_group), 1);
}

// ============================================================================
// Grid size
// ============================================================================

TEST(FmhaBwdOGradDotO, GridSizeBasic)
{
    constexpr auto g = ograd_dot_o_grid_size(2, 8, 256, 64);
    EXPECT_EQ(g.x, 4u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdOGradDotO, GridSizeCeil)
{
    constexpr auto g = ograd_dot_o_grid_size(1, 1, 65, 64);
    EXPECT_EQ(g.x, 2u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

TEST(FmhaBwdOGradDotO, GridSizeExact)
{
    constexpr auto g = ograd_dot_o_grid_size(1, 1, 64, 64);
    EXPECT_EQ(g.x, 1u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}
