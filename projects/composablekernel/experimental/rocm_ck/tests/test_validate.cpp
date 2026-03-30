// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/gemm_spec.hpp>
#include <rocm_ck/validate.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// A minimal GemmSpec for testing: 3 physical tensors (A, B, C).
static constexpr auto test_spec = make_spec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{
        .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .mfma_tile = {16, 16, 16}});

// A GemmSpec with a D0 tensor (4 physical tensors: A, B, D, C).
static constexpr auto test_spec_d0 = make_spec(
    Signature{.dtype = DataType::FP16,
              .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                        AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
    GemmAlgorithm{
        .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .mfma_tile = {16, 16, 16}});

// ============================================================================
// validate() — passes when all tensors are filled
// ============================================================================

TEST(Validate, PassesWhenAllTensorsFilled)
{
    int dummy_a = 1, dummy_b = 2, dummy_c = 3;

    Args args{};
    args.tensors[test_spec.lhs().args_slot] = {&dummy_a, make_shape(64, 32), make_strides(32, 1)};
    args.tensors[test_spec.rhs().args_slot] = {&dummy_b, make_shape(32, 64), make_strides(1, 32)};
    args.tensors[test_spec.output().args_slot] = {
        &dummy_c, make_shape(64, 64), make_strides(64, 1)};

    // Should not abort
    validate(args, test_spec);
}

TEST(Validate, PassesWithD0TensorFilled)
{
    int dummy_a = 1, dummy_b = 2, dummy_c = 3, dummy_bias = 4;

    Args args{};
    args.tensors[test_spec_d0.lhs().args_slot] = {
        &dummy_a, make_shape(64, 32), make_strides(32, 1)};
    args.tensors[test_spec_d0.rhs().args_slot] = {
        &dummy_b, make_shape(32, 64), make_strides(1, 32)};
    args.tensors[test_spec_d0.output().args_slot] = {
        &dummy_c, make_shape(64, 64), make_strides(64, 1)};
    args.tensors[test_spec_d0.d0().args_slot] = {
        &dummy_bias, make_shape(64, 64), make_strides(64, 1)};

    // Should not abort
    validate(args, test_spec_d0);
}

// ============================================================================
// validate() — aborts when a tensor is missing
// ============================================================================

TEST(ValidateDeathTest, AbortsOnNullTensorPointer)
{
    int dummy_a = 1, dummy_b = 2;

    Args args{};
    args.tensors[test_spec.lhs().args_slot] = {&dummy_a, make_shape(64, 32), make_strides(32, 1)};
    args.tensors[test_spec.rhs().args_slot] = {&dummy_b, make_shape(32, 64), make_strides(1, 32)};
    // output slot intentionally left null

    EXPECT_DEATH(validate(args, test_spec), "tensor \"C\" \\(slot 2\\) has null pointer");
}

TEST(ValidateDeathTest, AbortsOnMissingD0Tensor)
{
    int dummy_a = 1, dummy_b = 2, dummy_c = 3;

    Args args{};
    args.tensors[test_spec_d0.lhs().args_slot] = {
        &dummy_a, make_shape(64, 32), make_strides(32, 1)};
    args.tensors[test_spec_d0.rhs().args_slot] = {
        &dummy_b, make_shape(32, 64), make_strides(1, 32)};
    args.tensors[test_spec_d0.output().args_slot] = {
        &dummy_c, make_shape(64, 64), make_strides(64, 1)};
    // D0 (bias) slot intentionally left null

    EXPECT_DEATH(validate(args, test_spec_d0), "tensor \"bias\" \\(slot 3\\) has null pointer");
}

TEST(ValidateDeathTest, ReportsFirstMissingTensor)
{
    // All slots null — should report the first one (lhs = "A", slot 0)
    Args args{};

    EXPECT_DEATH(validate(args, test_spec), "tensor \"A\" \\(slot 0\\) has null pointer");
}
