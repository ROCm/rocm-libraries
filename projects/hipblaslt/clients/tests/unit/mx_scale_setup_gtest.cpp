// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the descriptor/layout pairing used by hipBLASLt.
// The relative include avoids shadowing the clients' utility.hpp with the
// library's internal header of the same name.

#include "../../../library/src/amd_detail/rocblaslt/src/include/mx_scale_setup.hpp"

#include <gtest/gtest.h>

#include <cstddef>

namespace
{
    // A TN GEMM, the orientation in which K is the fast dim of both A and B, so
    // setMXScaleA/B's K-block padding lands in the leading scale stride where it
    // is easy to read back.
    TensileLite::ContractionProblemGemm mxProblem(size_t M, size_t N, size_t K)
    {
        auto problem = TensileLite::ContractionProblemGemm::GEMM_Strides(
            /*transA*/ true,
            /*transB*/ false,
            rocisa::DataType::Float8,
            rocisa::DataType::Float8,
            rocisa::DataType::BFloat16,
            rocisa::DataType::BFloat16,
            M,
            N,
            K,
            1,
            K,
            K * M,
            K,
            K * N,
            M,
            M * N,
            M,
            M * N,
            0.0);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }
}

// The declaration hipBLASLt owes generateCustomCall arrives with the descriptor,
// for A and for B, and it is not something the problem starts out with.
TEST(HostUnitMXScaleSetup, smoke_PackedDeclarationArrivesWithTheScaleDescriptor)
{
    auto problem = mxProblem(64, 64, 128);
    ASSERT_FALSE(problem.customKernelMXScaleAIsPacked());
    ASSERT_FALSE(problem.customKernelMXScaleBIsPacked());

    rocblaslt::setHostMXScaleA(problem, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx950);
    EXPECT_EQ(problem.mxBlockA(), 32u);
    EXPECT_TRUE(problem.customKernelMXScaleAIsPacked());
    EXPECT_FALSE(problem.customKernelMXScaleBIsPacked());

    auto viaB = mxProblem(64, 64, 128);
    rocblaslt::setHostMXScaleB(viaB, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx950);
    EXPECT_EQ(viaB.mxBlockB(), 32u);
    EXPECT_TRUE(viaB.customKernelMXScaleBIsPacked());
    EXPECT_FALSE(viaB.customKernelMXScaleAIsPacked());
}

// A pre-swizzled operand keeps the padded descriptor layout, and declaring one
// operand never changes what was declared for the other.
TEST(HostUnitMXScaleSetup, smoke_PreSwizzledDeclarationIsPerOperand)
{
    auto swizzledA = mxProblem(64, 64, 128);
    rocblaslt::setHostMXScaleA(swizzledA,
                               rocisa::DataType::E8,
                               32,
                               TensileLite::MXScalePadding::Gfx950,
                               /*packed*/ false);
    rocblaslt::setHostMXScaleB(swizzledA,
                               rocisa::DataType::E8,
                               32,
                               TensileLite::MXScalePadding::Gfx950,
                               /*packed*/ true);
    EXPECT_FALSE(swizzledA.customKernelMXScaleAIsPacked());
    EXPECT_TRUE(swizzledA.customKernelMXScaleBIsPacked());

    auto swizzledB = mxProblem(64, 64, 128);
    rocblaslt::setHostMXScaleA(swizzledB,
                               rocisa::DataType::E8,
                               32,
                               TensileLite::MXScalePadding::Gfx950,
                               /*packed*/ true);
    rocblaslt::setHostMXScaleB(swizzledB,
                               rocisa::DataType::E8,
                               32,
                               TensileLite::MXScalePadding::Gfx950,
                               /*packed*/ false);
    EXPECT_TRUE(swizzledB.customKernelMXScaleAIsPacked());
    EXPECT_FALSE(swizzledB.customKernelMXScaleBIsPacked());
}

// The TensileLite client uses the descriptor's padded layout by default.
TEST(HostUnitMXScaleSetup, smoke_RawSetMXScaleLeavesTheDeclarationUnset)
{
    auto problem = mxProblem(64, 64, 128);
    problem.setMXScaleA(rocisa::DataType::E8, 32, {}, TensileLite::MXScalePadding::Gfx950);
    problem.setMXScaleB(rocisa::DataType::E8, 32, {}, TensileLite::MXScalePadding::Gfx950);

    EXPECT_EQ(problem.mxBlockA(), 32u);
    EXPECT_EQ(problem.mxBlockB(), 32u);
    EXPECT_FALSE(problem.customKernelMXScaleAIsPacked());
    EXPECT_FALSE(problem.customKernelMXScaleBIsPacked());
}

// gfx950 pads ceil(128/32) = 4 K-blocks to 8; gfx1250 leaves this size at 4.
TEST(HostUnitMXScaleSetup, smoke_PaddingArgumentReachesTheDescriptor)
{
    auto padded = mxProblem(64, 64, 128);
    rocblaslt::setHostMXScaleA(padded, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx950);
    rocblaslt::setHostMXScaleB(padded, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx950);
    EXPECT_EQ(padded.mxsa().strides()[1], 8u);
    EXPECT_EQ(padded.mxsb().strides()[1], 8u);

    auto unpadded = mxProblem(64, 64, 128);
    rocblaslt::setHostMXScaleA(unpadded, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx1250);
    rocblaslt::setHostMXScaleB(unpadded, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx1250);
    EXPECT_EQ(unpadded.mxsa().strides()[1], 4u);
    EXPECT_EQ(unpadded.mxsb().strides()[1], 4u);
    EXPECT_TRUE(unpadded.customKernelMXScaleAIsPacked());
    EXPECT_TRUE(unpadded.customKernelMXScaleBIsPacked());
}

// K=96, block 32: ceil(K/32) is 3. Packed keeps 3. Gfx950 rounds it to 8.
// The pre-swizzled call is the one that must not claim the buffer is packed.
TEST(HostUnitMXScaleSetup, smoke_PaddingFollowsTheCallerScaleFormat)
{
    auto packed = mxProblem(64, 64, 96);
    rocblaslt::setHostMXScaleA(packed, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Packed);
    rocblaslt::setHostMXScaleB(packed, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Packed);
    EXPECT_EQ(packed.mxsa().strides()[1], 3u);
    EXPECT_EQ(packed.mxsb().strides()[1], 3u);
    EXPECT_TRUE(packed.customKernelMXScaleAIsPacked());
    EXPECT_TRUE(packed.customKernelMXScaleBIsPacked());

    auto preswizzled = mxProblem(64, 64, 96);
    rocblaslt::setHostMXScaleA(
        preswizzled, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx950, false);
    rocblaslt::setHostMXScaleB(
        preswizzled, rocisa::DataType::E8, 32, TensileLite::MXScalePadding::Gfx950, false);
    EXPECT_EQ(preswizzled.mxsa().strides()[1], 8u);
    EXPECT_EQ(preswizzled.mxsb().strides()[1], 8u);
    EXPECT_FALSE(preswizzled.customKernelMXScaleAIsPacked());
    EXPECT_FALSE(preswizzled.customKernelMXScaleBIsPacked());
}
