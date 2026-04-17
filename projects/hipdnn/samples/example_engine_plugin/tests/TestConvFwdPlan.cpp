// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: Second Plan test example. This uses the same testing pattern
// as TestReluPlan.cpp but for a convolution kernel.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "ExampleProviderHandle.hpp"
#include "engines/plans/ConvFwdParams.hpp"
#include "engines/plans/ConvFwdPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

using namespace example_provider;
using ::testing::_;
using ::testing::Return;

class ConvFwdPlanTest : public ::testing::Test
{
protected:
    // Convolution dimensions: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    static constexpr int64_t kInputUid = 1;
    static constexpr int64_t kWeightUid = 2;
    static constexpr int64_t kOutputUid = 3;
    static constexpr int64_t kN = 1;
    static constexpr int64_t kC = 1;
    static constexpr int64_t kH = 4;
    static constexpr int64_t kW = 4;
    static constexpr int64_t kK = 1;
    static constexpr int64_t kR = 3;
    static constexpr int64_t kS = 3;
    static constexpr int64_t kOutH = 2;
    static constexpr int64_t kOutW = 2;
    static constexpr int64_t kPadH = 0;
    static constexpr int64_t kPadW = 0;
    static constexpr int64_t kStrideH = 1;
    static constexpr int64_t kStrideW = 1;
    static constexpr int64_t kBlockSize = 256;

    MockKernelCompiler mockCompiler;
    ExampleProviderHandle handle;

    MockCompiledProgram* rawCompiledProgram = nullptr;
    MockRunnableKernel* rawKernel = nullptr;

    std::unique_ptr<ConvFwdPlan> createAndCompilePlan()
    {
        ConvFwdParams params{kInputUid,
                             kWeightUid,
                             kOutputUid,
                             kN,
                             kC,
                             kH,
                             kW,
                             kK,
                             kR,
                             kS,
                             kOutH,
                             kOutW,
                             kPadH,
                             kPadW,
                             kStrideH,
                             kStrideW,
                             kBlockSize};
        auto plan = std::make_unique<ConvFwdPlan>(std::move(params));

        auto compiledProgram = std::make_unique<MockCompiledProgram>();
        rawCompiledProgram = compiledProgram.get();

        auto kernel = std::make_unique<MockRunnableKernel>();
        rawKernel = kernel.get();

        EXPECT_CALL(mockCompiler, compile("ConvForwardNaive.cpp", _))
            .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

        EXPECT_CALL(*rawCompiledProgram, getRunnableKernel("conv_forward_naive_kernel"))
            .WillOnce(Return(testing::ByMove(std::move(kernel))));

        plan->compile(mockCompiler);
        return plan;
    }
};

TEST_F(ConvFwdPlanTest, GetWorkspaceSize_ReturnsZero)
{
    ConvFwdParams params{kInputUid,
                         kWeightUid,
                         kOutputUid,
                         kN,
                         kC,
                         kH,
                         kW,
                         kK,
                         kR,
                         kS,
                         kOutH,
                         kOutW,
                         kPadH,
                         kPadW,
                         kStrideH,
                         kStrideW,
                         kBlockSize};
    ConvFwdPlan plan{std::move(params)};
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST_F(ConvFwdPlanTest, Compile_CallsCompilerWithCorrectFilename)
{
    ConvFwdParams params{kInputUid,
                         kWeightUid,
                         kOutputUid,
                         kN,
                         kC,
                         kH,
                         kW,
                         kK,
                         kR,
                         kS,
                         kOutH,
                         kOutW,
                         kPadH,
                         kPadW,
                         kStrideH,
                         kStrideW,
                         kBlockSize};
    auto plan = std::make_unique<ConvFwdPlan>(std::move(params));

    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();
    auto kernel = std::make_unique<MockRunnableKernel>();

    // Verify the compiler receives the correct kernel filename with empty options.
    EXPECT_CALL(mockCompiler, compile("ConvForwardNaive.cpp", std::vector<std::string>{}))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

    EXPECT_CALL(*rawProgram, getRunnableKernel("conv_forward_naive_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    plan->compile(mockCompiler);
}

TEST_F(ConvFwdPlanTest, Execute_SetsGridAndBlockSizeAndLaunches)
{
    auto plan = createAndCompilePlan();

    // Total output elements: N*K*outH*outW = 1*1*2*2 = 4
    // blockSize=256, gridSize=ceil(4/256)=1
    EXPECT_CALL(*rawKernel, setBlockSize(256, 1, 1));
    EXPECT_CALL(*rawKernel, setGridSize(1, 1, 1));
    EXPECT_CALL(*rawKernel, launchImpl(nullptr, _));

    std::vector<float> inputData(kN * kC * kH * kW, 1.0f);
    std::vector<float> weightData(kK * kC * kR * kS, 1.0f);
    std::vector<float> outputData(kN * kK * kOutH * kOutW, -999.0f);

    hipdnnPluginDeviceBuffer_t buffers[3];
    buffers[0].uid = kInputUid;
    buffers[0].ptr = inputData.data();
    buffers[1].uid = kWeightUid;
    buffers[1].ptr = weightData.data();
    buffers[2].uid = kOutputUid;
    buffers[2].ptr = outputData.data();

    plan->execute(handle, buffers, 3, nullptr);
}

TEST_F(ConvFwdPlanTest, Execute_MissingBuffer_Throws)
{
    auto plan = createAndCompilePlan();

    std::vector<float> inputData(kN * kC * kH * kW, 1.0f);

    // Only provide input buffer, missing weight and output
    hipdnnPluginDeviceBuffer_t buffers[1];
    buffers[0].uid = kInputUid;
    buffers[0].ptr = inputData.data();

    EXPECT_THROW(plan->execute(handle, buffers, 1, nullptr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
