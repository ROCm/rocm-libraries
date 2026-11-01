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
    ASSERT_FALSE(problem.customKernelMXScalesArePacked());

    rocblaslt::setHostMXScaleA(problem, rocisa::DataType::E8, 32, /*padFreeDim*/ true);
    EXPECT_EQ(problem.mxBlockA(), 32u);
    EXPECT_TRUE(problem.customKernelMXScalesArePacked());

    auto viaB = mxProblem(64, 64, 128);
    ASSERT_FALSE(viaB.customKernelMXScalesArePacked());
    rocblaslt::setHostMXScaleB(viaB, rocisa::DataType::E8, 32, /*padFreeDim*/ true);
    EXPECT_EQ(viaB.mxBlockB(), 32u);
    EXPECT_TRUE(viaB.customKernelMXScalesArePacked());
}

// The TensileLite client uses the descriptor's padded layout by default.
TEST(HostUnitMXScaleSetup, smoke_RawSetMXScaleLeavesTheDeclarationUnset)
{
    auto problem = mxProblem(64, 64, 128);
    problem.setMXScaleA(rocisa::DataType::E8, 32, {}, /*padScaleTensorFreeDim*/ true);
    problem.setMXScaleB(rocisa::DataType::E8, 32, {}, /*padScaleTensorFreeDim*/ true);

    EXPECT_EQ(problem.mxBlockA(), 32u);
    EXPECT_EQ(problem.mxBlockB(), 32u);
    EXPECT_FALSE(problem.customKernelMXScalesArePacked());
}

// gfx950 pads ceil(128/32) = 4 K-blocks to 8; gfx1250 leaves this size at 4.
TEST(HostUnitMXScaleSetup, smoke_PaddingArgumentReachesTheDescriptor)
{
    auto padded = mxProblem(64, 64, 128);
    rocblaslt::setHostMXScaleA(padded, rocisa::DataType::E8, 32, /*padFreeDim*/ true);
    rocblaslt::setHostMXScaleB(padded, rocisa::DataType::E8, 32, /*padFreeDim*/ true);
    EXPECT_EQ(padded.mxsa().strides()[1], 8u);
    EXPECT_EQ(padded.mxsb().strides()[1], 8u);

    auto unpadded = mxProblem(64, 64, 128);
    rocblaslt::setHostMXScaleA(unpadded, rocisa::DataType::E8, 32, /*padFreeDim*/ false);
    rocblaslt::setHostMXScaleB(unpadded, rocisa::DataType::E8, 32, /*padFreeDim*/ false);
    EXPECT_EQ(unpadded.mxsa().strides()[1], 4u);
    EXPECT_EQ(unpadded.mxsb().strides()[1], 4u);
    EXPECT_TRUE(unpadded.customKernelMXScalesArePacked());
}
