// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/convert_dq_api.hpp>
#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;
namespace S = fmha_bwd_convert_dq_slots;

// ============================================================================
// Algorithm defaults
// ============================================================================

TEST(FmhaBwdConvertDQ, AlgorithmDefaults)
{
    constexpr FmhaBwdConvertDQAlgorithm algo{};
    EXPECT_TRUE(algo.is_deterministic);
    EXPECT_TRUE(algo.pad_seqlen_q);
    EXPECT_TRUE(algo.pad_hdim_q);
    EXPECT_EQ(algo.block_per_cu, 2);
}

// ============================================================================
// make_spec happy path
// ============================================================================

TEST(FmhaBwdConvertDQ, MakeSpecFP16Batch)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdConvertDQ, MakeSpecBF16)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::BF16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_EQ(k.dtype, DataType::BF16);
}

TEST(FmhaBwdConvertDQ, MakeSpecGroupMode)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}});
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.pad_seqlen_q);
}

TEST(FmhaBwdConvertDQ, MakeSpecAllHdims)
{
    constexpr auto k32  = make_spec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 32, .mode = FmhaMode::BATCH},
         .algorithm = {}});
    constexpr auto k64  = make_spec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 64, .mode = FmhaMode::BATCH},
         .algorithm = {}});
    constexpr auto k96  = make_spec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 96, .mode = FmhaMode::BATCH},
         .algorithm = {}});
    constexpr auto k128 = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    constexpr auto k256 = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 256, .mode = FmhaMode::BATCH},
        .algorithm = {}});

    EXPECT_EQ(k32.hdim_q, 32);
    EXPECT_EQ(k64.hdim_q, 64);
    EXPECT_EQ(k96.hdim_q, 96);
    EXPECT_EQ(k128.hdim_q, 128);
    EXPECT_EQ(k256.hdim_q, 256);
}

TEST(FmhaBwdConvertDQ, MakeSpecNoPadBatch)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_q = false}});
    EXPECT_FALSE(k.pad_seqlen_q);
    EXPECT_FALSE(k.pad_hdim_q);
}

TEST(FmhaBwdConvertDQ, MakeSpecCustomBlockPerCu)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.block_per_cu = 4}});
    EXPECT_EQ(k.block_per_cu, 4);
}

TEST(FmhaBwdConvertDQ, IsDeterministicDefault)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_TRUE(k.is_deterministic);
}

TEST(FmhaBwdConvertDQ, MakeSpecNonDeterministic)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = false}});
    EXPECT_FALSE(k.is_deterministic);
}

// ============================================================================
// Slot constants
// ============================================================================

TEST(FmhaBwdConvertDQ, TensorSlotIndices)
{
    EXPECT_EQ(S::DQ_ACC, 0);
    EXPECT_EQ(S::DQ, 1);
    EXPECT_EQ(S::SEQSTART_Q, 2);
    EXPECT_EQ(S::SEQLEN_Q, 3);
    EXPECT_EQ(S::SEQSTART_K, 4);
    EXPECT_EQ(S::SEQLEN_K, 5);
}

TEST(FmhaBwdConvertDQ, RequiredTensorsBatch)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_EQ(S::requiredTensors(k), 2);
}

TEST(FmhaBwdConvertDQ, RequiredTensorsGroup)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
        .algorithm = {}});
    EXPECT_EQ(S::requiredTensors(k), 6);
}

TEST(FmhaBwdConvertDQ, RequiredScalarsAlways0)
{
    constexpr auto k = make_spec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_EQ(S::requiredScalars(k), 0);
}

// ============================================================================
// Grid size
// ============================================================================

TEST(FmhaBwdConvertDQ, GridSizeBasic)
{
    constexpr auto g = convert_dq_grid_size(2, 8, 256, 64);
    EXPECT_EQ(g.x, 4u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdConvertDQ, GridSizeDefaultTileM0)
{
    // Default tile_m0 = 64
    constexpr auto g = convert_dq_grid_size(1, 1, 128);
    EXPECT_EQ(g.x, 2u);
}
