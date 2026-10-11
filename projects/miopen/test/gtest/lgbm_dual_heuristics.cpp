// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU-only unit tests for the dual-heuristics gate (ai::common::PreferLgbm), which
// routes small/mid-size gfx942/gfx950 problems LGBM-first and large ones to TunaNet +
// the two-tower. No GPU required: the gate only reads the problem description and the
// device name.

#include <gtest/gtest.h>

#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/tensor.hpp>

#include <array>
#include <cstdlib>
#include <string>

namespace {

using miopen::ai::common::ConvFlops;
using miopen::ai::common::DefaultLgbmDualThresholds;
using miopen::ai::common::LgbmDualThresholds;
using miopen::ai::common::PreferLgbm;
using miopen::ai::common::PreferLgbmForSize;
using miopen::conv::Direction;

struct Conv2d
{
    int n, c, h, w, k, y, x;
    int stride = 1;
    int pad    = 0;
    int groups = 1;
};

miopen::conv::ProblemDescription
MakeProblem(const Conv2d& p, Direction dir, miopenDataType_t dtype = miopenHalf)
{
    const int out_h = (p.h + 2 * p.pad - p.y) / p.stride + 1;
    const int out_w = (p.w + 2 * p.pad - p.x) / p.stride + 1;
    const miopen::TensorDescriptor x_desc(dtype, {p.n, p.c, p.h, p.w});
    const miopen::TensorDescriptor w_desc(dtype, {p.k, p.c / p.groups, p.y, p.x});
    const miopen::TensorDescriptor y_desc(dtype, {p.n, p.k, out_h, out_w});
    const miopen::ConvolutionDescriptor conv(2,
                                             miopenConvolution,
                                             miopenPaddingDefault,
                                             {p.pad, p.pad},
                                             {p.stride, p.stride},
                                             {1, 1},
                                             {0, 0},
                                             p.groups,
                                             1.0f);
    // The problem description swaps x and y for Backward*.
    const bool fwd = dir == Direction::Forward;
    return {fwd ? x_desc : y_desc, w_desc, fwd ? y_desc : x_desc, conv, dir};
}

double ExpectedFlops(const Conv2d& p)
{
    const double out_h = (p.h + 2 * p.pad - p.y) / p.stride + 1;
    const double out_w = (p.w + 2 * p.pad - p.x) / p.stride + 1;
    return 2.0 * p.n * p.k * out_h * out_w * (static_cast<double>(p.c) / p.groups) * p.y * p.x;
}

constexpr std::array<Direction, 3> kDirections = {
    Direction::Forward, Direction::BackwardData, Direction::BackwardWeights};

TEST(CPU_LgbmDualHeuristics_NONE, ConvFlopsIsDirectionIndependent)
{
    const Conv2d plain{16, 64, 56, 56, 128, 3, 3, 1, 1};
    const Conv2d strided_grouped{8, 256, 28, 28, 512, 3, 3, 2, 1, 32};
    for(const auto& p : {plain, strided_grouped})
    {
        for(const auto dir : kDirections)
        {
            EXPECT_DOUBLE_EQ(ConvFlops(MakeProblem(p, dir)), ExpectedFlops(p))
                << "direction " << static_cast<int>(dir);
        }
    }
}

TEST(CPU_LgbmDualHeuristics_NONE, DefaultThresholdsCoverOnlyGfx942AndGfx950)
{
    const auto gfx942 = DefaultLgbmDualThresholds("gfx942");
    ASSERT_TRUE(gfx942.has_value());
    EXPECT_DOUBLE_EQ(gfx942->fwd_flops, 1e12);
    EXPECT_DOUBLE_EQ(gfx942->bwd_flops, 1e11);
    EXPECT_EQ(gfx942->bwd_max_tensor_bytes, std::size_t{1} << 31);

    const auto gfx950 = DefaultLgbmDualThresholds("gfx950");
    ASSERT_TRUE(gfx950.has_value());
    EXPECT_DOUBLE_EQ(gfx950->fwd_flops, 1e14);
    EXPECT_DOUBLE_EQ(gfx950->bwd_flops, 1e14);

    EXPECT_FALSE(DefaultLgbmDualThresholds("gfx90a").has_value());
    EXPECT_FALSE(DefaultLgbmDualThresholds("gfx1100").has_value());
}

TEST(CPU_LgbmDualHeuristics_NONE, SizeDecision)
{
    const LgbmDualThresholds t{1e12, 1e11, 1000};

    // Forward: FLOP threshold only; the tensor-size guard is backward-only.
    EXPECT_TRUE(PreferLgbmForSize(true, 9.9e11, 0, t));
    EXPECT_FALSE(PreferLgbmForSize(true, 1e12, 0, t));
    EXPECT_TRUE(PreferLgbmForSize(true, 1e3, 5000, t));

    // Backward: its own (lower) FLOP threshold plus the tensor-size guard.
    EXPECT_TRUE(PreferLgbmForSize(false, 9.9e10, 999, t));
    EXPECT_FALSE(PreferLgbmForSize(false, 1e11, 0, t));
    EXPECT_FALSE(PreferLgbmForSize(false, 1e3, 1000, t));

    // bwd_max_tensor_bytes == 0 disables the guard.
    const LgbmDualThresholds no_guard{1e12, 1e11, 0};
    EXPECT_TRUE(PreferLgbmForSize(false, 1e3, std::size_t{1} << 40, no_guard));
}

class CPU_LgbmDualHeuristicsGate_NONE : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // The end-to-end cases assert the built-in defaults.
        for(const char* name : {"MIOPEN_DEBUG_LGBM_ONLY",
                                "MIOPEN_DEBUG_LGBM_DUAL_HEURISTICS",
                                "MIOPEN_DEBUG_LGBM_DUAL_FLOPS_THRESHOLD",
                                "MIOPEN_DEBUG_LGBM_DUAL_BWD_FLOPS_THRESHOLD",
                                "MIOPEN_DEBUG_LGBM_DUAL_BWD_MAX_TENSOR_BYTES"})
        {
            if(std::getenv(name) != nullptr)
                GTEST_SKIP() << name << " is set; default thresholds not in effect";
        }
    }
};

TEST_F(CPU_LgbmDualHeuristicsGate_NONE, RoutesBySizeArchAndDirection)
{
    // ~3.7e9 FLOPs: LGBM-first everywhere the gate applies.
    const Conv2d small{16, 64, 56, 56, 64, 3, 3, 1, 1};
    // ~5e12 FLOPs: above gfx942's thresholds, below gfx950's (every tensor < 2 GiB).
    const Conv2d large{64, 512, 128, 128, 512, 3, 3, 1, 1};
    // ~1.2e11 FLOPs: between gfx942's backward (1e11) and forward (1e12) thresholds.
    const Conv2d mid{32, 256, 56, 56, 256, 3, 3, 1, 1};

    for(const auto dir : kDirections)
    {
        EXPECT_TRUE(PreferLgbm(MakeProblem(small, dir), "gfx942"));
        EXPECT_TRUE(PreferLgbm(MakeProblem(small, dir), "gfx950"));
        EXPECT_FALSE(PreferLgbm(MakeProblem(large, dir), "gfx942"));
        EXPECT_TRUE(PreferLgbm(MakeProblem(large, dir), "gfx950"));
        // Architectures without dual thresholds keep TunaNet + two-tower first.
        EXPECT_FALSE(PreferLgbm(MakeProblem(small, dir), "gfx90a"));
    }

    EXPECT_TRUE(PreferLgbm(MakeProblem(mid, Direction::Forward), "gfx942"));
    EXPECT_FALSE(PreferLgbm(MakeProblem(mid, Direction::BackwardData), "gfx942"));
    EXPECT_FALSE(PreferLgbm(MakeProblem(mid, Direction::BackwardWeights), "gfx942"));
}

TEST_F(CPU_LgbmDualHeuristicsGate_NONE, BackwardLargeTensorGuard)
{
    // 1x1 conv, ~3.4e10 FLOPs, but x is 4 GiB in fp32.
    const Conv2d big_tensor{64, 256, 256, 256, 16, 1, 1};
    const auto fwd = MakeProblem(big_tensor, Direction::Forward, miopenFloat);
    ASSERT_LT(ConvFlops(fwd), 1e11);

    EXPECT_TRUE(PreferLgbm(fwd, "gfx950"));
    EXPECT_FALSE(
        PreferLgbm(MakeProblem(big_tensor, Direction::BackwardData, miopenFloat), "gfx950"));
    EXPECT_FALSE(
        PreferLgbm(MakeProblem(big_tensor, Direction::BackwardWeights, miopenFloat), "gfx950"));
}

TEST_F(CPU_LgbmDualHeuristicsGate_NONE, OnlyMeasuredDataTypes)
{
    const Conv2d small{16, 64, 56, 56, 64, 3, 3, 1, 1};
    for(const auto dtype : {miopenFloat, miopenHalf, miopenBFloat16})
        EXPECT_TRUE(PreferLgbm(MakeProblem(small, Direction::Forward, dtype), "gfx942"));
    EXPECT_FALSE(PreferLgbm(MakeProblem(small, Direction::Forward, miopenInt8), "gfx942"));
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
