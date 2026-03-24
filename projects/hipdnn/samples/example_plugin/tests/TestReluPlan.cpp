// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "ExamplePluginHandle.hpp"
#include "engines/plans/ReluParams.hpp"
#include "engines/plans/ReluPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

using namespace example_plugin;
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
    ExamplePluginHandle handle;

    // Raw pointers for verification -- the plan takes ownership through unique_ptr
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

        EXPECT_CALL(*rawCompiledProgram, getKernel("relu_forward_kernel"))
            .WillOnce(Return(testing::ByMove(std::move(kernel))));

        // Create device properties with test architecture
        hipDeviceProp_t props = {};
        snprintf(props.gcnArchName, sizeof(props.gcnArchName), "%s", "gfx90a:sramecc+:xnack-");

        plan->compile(mockCompiler, props);
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

TEST_F(ReluPlanTest, Compile_CallsCompilerWithCorrectArchitecture)
{
    ReluParams params{kInputUid, kOutputUid, kNumElements, kNegativeSlope};
    auto plan = std::make_unique<ReluPlan>(std::move(params));

    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();

    auto kernel = std::make_unique<MockRunnableKernel>();

    // Verify the compiler receives the correct architecture option
    EXPECT_CALL(mockCompiler,
                compile("ReluForward.cpp", std::vector<std::string>{"--offload-arch=gfx90a"}))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

    EXPECT_CALL(*rawProgram, getKernel("relu_forward_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    hipDeviceProp_t props = {};
    snprintf(props.gcnArchName, sizeof(props.gcnArchName), "%s", "gfx90a:sramecc+:xnack-");

    plan->compile(mockCompiler, props);
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
