// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for setHostMXScaleA/B (mx_scale_setup.hpp), the pair of
// helpers every MX scale attachment in tensile_host.cpp goes through.
//
// What is being pinned is the pairing, not the arithmetic. ContractionSolution::
// generateCustomCall hands an external custom kernel the packed ceil(K/mxBlock)
// scale pitch only when the problem says customKernelMXScalesArePacked(); the
// flag defaults to false because the TensileLite benchmark client allocates its
// scale buffer from the padded descriptor and hands that same buffer over. So
// hipBLASLt has to turn it on, and if that ever stopped happening the result
// would be silent numerical corruption -- the padded pitch read against a packed
// buffer, which is the failure this helper's PR fixed in the other direction.
//
// ConstructTensileProblem and updateTensileProblem, the two functions that set
// the scales, cannot be driven from here: both live in tensile_host.cpp's
// anonymous namespace, and both call mxScaleTensorNeedsPaddingFreeDim(), which
// calls hipGetDevice -- these tests run with no GPU (see README.md). So the
// declaration is instead made inseparable from the descriptor it describes, in
// the helpers below, and this file pins that. Deleting the setup at either call
// site is caught separately, at compile time, by the -Wunused-variable tripwire
// around padMXScaleTensorFreeDim in tensile_host.cpp.
//
// The header is included by relative path, following rocroller_kernel_match_gtest
// .cpp: putting the library's internal src/include on this target's include path
// shadows the clients' own utility.hpp (see the note in ../src/CMakeLists.txt).

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

// Without the helper the declaration is absent, so the test above is reading the
// helper rather than something setMXScaleA/B or the constructor does anyway. This
// is also the state the TensileLite benchmark client is left in on purpose.
TEST(HostUnitMXScaleSetup, smoke_RawSetMXScaleLeavesTheDeclarationUnset)
{
    auto problem = mxProblem(64, 64, 128);
    problem.setMXScaleA(rocisa::DataType::E8, 32, {}, /*padScaleTensorFreeDim*/ true);
    problem.setMXScaleB(rocisa::DataType::E8, 32, {}, /*padScaleTensorFreeDim*/ true);

    EXPECT_EQ(problem.mxBlockA(), 32u);
    EXPECT_EQ(problem.mxBlockB(), 32u);
    EXPECT_FALSE(problem.customKernelMXScalesArePacked());
}

// The helper must still be a faithful wrapper: the padding argument is the one
// thing it forwards, and it is what makes the descriptor disagree with the packed
// pitch in the first place. ceil(128/32) = 4, padded up to a multiple of 8.
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
