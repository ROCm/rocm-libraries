// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <new>
#include <utility>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <gtest/gtest.h>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm helperProblem()
    {
        auto problem = ContractionProblemGemm::GEMM(
            false, false, 256, 256, 256, 256, 256, 256, 1.0, false, 1);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        return problem;
    }
}

// Every helper kernel a solution launches is marked as one, so the adapter
// looks it up in the helper code object instead of the solution's, which does
// not have it.
TEST(HelperKernelInvocation, BuildersMarkTheirKernelsAsHelpers)
{
    ContractionSolution solution;
    solution.sizeMapping.macroTile   = TensileLite::dim3(128, 128, 1);
    solution.sizeMapping.depthU      = 64;
    auto                     problem = helperProblem();
    ContractionInputs        inputs;
    ContractionGroupedInputs grouped;
    grouped.grouped.push_back(inputs);
    hip::HipAMDGPU  device;
    KernelArguments hostArgs(false);

    const std::pair<char const*, KernelInvocation> kernels[] = {
        {"beta-only", solution.generateBetaOnlyCall<false>(problem, inputs)},
        {"output conversion",
         solution.generateOutputConversionCall<false>(problem, inputs, {}, 1, 0)},
        {"grouped output conversion",
         solution.generateOutputConversionCallGroupedGemm<false>(
             {problem}, grouped, device, hostArgs)},
        {"grouped output conversion from user arguments",
         solution.updateUserArgsOutputConversionCallGroupedGemm<false>(
             {problem}, nullptr, nullptr)},
        {"reduction", solution.generateReductionCall<false>(problem, inputs)},
    };
    for(auto const& [name, kernel] : kernels)
        EXPECT_TRUE(kernel.isHelperKernel) << name;
}

// The grouped output conversion for user arguments launches in one dimension.
// The invocation is built over storage filled with ones bits, which a launch
// size the builder left unset would keep.
TEST(HelperKernelInvocation, UserArgsConversionLaunchesInOneDimension)
{
    ContractionSolution solution;
    auto                problem = helperProblem();

    alignas(KernelInvocation) unsigned char storage[sizeof(KernelInvocation)];
    std::memset(storage, 0xff, sizeof(storage));
    auto* kernel = new(storage) KernelInvocation(
        solution.updateUserArgsOutputConversionCallGroupedGemm<false>({problem}, nullptr, nullptr));
    EXPECT_EQ(kernel->numWorkGroups.y, 1u);
    EXPECT_EQ(kernel->numWorkGroups.z, 1u);
    EXPECT_EQ(kernel->numWorkItems.y, 1u);
    EXPECT_EQ(kernel->numWorkItems.z, 1u);
    kernel->~KernelInvocation();
}
