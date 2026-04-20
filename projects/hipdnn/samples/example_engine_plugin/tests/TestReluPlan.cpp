// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE ADAPTATION: Demonstrates the testing pattern for Plans. Key test categories:
// (1) compile: verify correct kernel filename and function name via mock expectations.
// (2) execute: verify grid/block dimensions and kernel launch.
// (3) Error handling: verify missing buffers throw.
// The mock chain pattern (MockKernelCompiler -> MockCompiledProgram -> MockRunnableKernel) with raw
// pointer retention for EXPECT_CALL is reusable for your Plan tests.
// Pick-and-choose which tests are useful for you situation and adapt as needed.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "ExampleProviderHandle.hpp"
#include "engines/plans/ReluParams.hpp"
#include "engines/plans/ReluPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

using namespace example_provider;
using ::testing::_;
using ::testing::Return;

class ReluPlanTest : public ::testing::Test
{
protected:
    static constexpr int64_t kInputUid = 1;
    static constexpr int64_t kOutputUid = 2;
    static constexpr int64_t kNumElements = 6;
    static constexpr double kNegativeSlope = 0.0;

    MockKernelCompiler mockCompiler;
    ExampleProviderHandle handle;

    // Raw pointers for verification. The plan takes ownership through unique_ptr.
    MockCompiledProgram* rawCompiledProgram = nullptr;
    MockRunnableKernel* rawKernel = nullptr;

    std::unique_ptr<ReluPlan> createAndCompilePlan()
    {
        ReluParams params{kInputUid, kOutputUid, kNumElements, kNegativeSlope};
        auto plan = std::make_unique<ReluPlan>(std::move(params));

        // Set up mock expectations: compiler returns a compiled program
        auto compiledProgram = std::make_unique<MockCompiledProgram>();
        rawCompiledProgram = compiledProgram.get();

        auto kernel = std::make_unique<MockRunnableKernel>();
        rawKernel = kernel.get();

        EXPECT_CALL(mockCompiler, compile("ReluForward.cpp", _))
            .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

        EXPECT_CALL(*rawCompiledProgram, getRunnableKernel("relu_forward_kernel"))
            .WillOnce(Return(testing::ByMove(std::move(kernel))));

        plan->compile(mockCompiler);
        return plan;
    }
};

TEST_F(ReluPlanTest, GetWorkspaceSize_ReturnsZero)
{
    // Workspace size can be checked without compiling
    ReluParams params{kInputUid, kOutputUid, kNumElements, kNegativeSlope};
    ReluPlan plan{std::move(params)};
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST_F(ReluPlanTest, Compile_CallsCompilerWithCorrectFilename)
{
    ReluParams params{kInputUid, kOutputUid, kNumElements, kNegativeSlope};
    auto plan = std::make_unique<ReluPlan>(std::move(params));

    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();

    auto kernel = std::make_unique<MockRunnableKernel>();

    // Verify the compiler receives the correct kernel filename with empty options.
    EXPECT_CALL(mockCompiler, compile("ReluForward.cpp", std::vector<std::string>{}))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

    EXPECT_CALL(*rawProgram, getRunnableKernel("relu_forward_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    plan->compile(mockCompiler);
}

TEST_F(ReluPlanTest, Execute_SetsGridAndBlockSizeAndLaunches)
{
    auto plan = createAndCompilePlan();

    // Expect setBlockSize and setGridSize to be called with correct values
    // blockSize=256, numElements=6, gridSize=ceil(6/256)=1
    EXPECT_CALL(*rawKernel, setBlockSize(256, 1, 1));
    EXPECT_CALL(*rawKernel, setGridSize(1, 1, 1));
    EXPECT_CALL(*rawKernel, launchImpl(nullptr, _));

    std::vector<float> inputData = {-3.0f, -1.0f, 0.0f, 1.0f, 2.5f, -0.5f};
    std::vector<float> outputData(kNumElements, -999.0f);

    hipdnnPluginDeviceBuffer_t buffers[2];
    buffers[0].uid = kInputUid;
    buffers[0].ptr = inputData.data();
    buffers[1].uid = kOutputUid;
    buffers[1].ptr = outputData.data();

    plan->execute(handle, buffers, 2, nullptr);
}

TEST_F(ReluPlanTest, Execute_MissingBuffer_Throws)
{
    auto plan = createAndCompilePlan();

    std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Only provide input buffer, not output
    hipdnnPluginDeviceBuffer_t buffers[1];
    buffers[0].uid = kInputUid;
    buffers[0].ptr = inputData.data();

    EXPECT_THROW(plan->execute(handle, buffers, 1, nullptr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
