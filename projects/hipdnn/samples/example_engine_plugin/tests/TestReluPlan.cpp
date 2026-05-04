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

#include <array>
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
using ::testing::_; // NOLINT(bugprone-reserved-identifier)
using ::testing::Return;

class ReluPlanTest : public ::testing::Test
{
protected:
    static constexpr int64_t INPUT_UID = 1;
    static constexpr int64_t OUTPUT_UID = 2;
    static constexpr int64_t NUM_ELEMENTS = 6;
    static constexpr double NEGATIVE_SLOPE = 0.0;

    MockKernelCompiler _mockCompiler;
    ExampleProviderHandle _handle;

    // Raw pointers for verification. The plan takes ownership through unique_ptr.
    MockCompiledProgram* _rawCompiledProgram = nullptr;
    MockRunnableKernel* _rawKernel = nullptr;

    std::unique_ptr<ReluPlan> createAndCompilePlan()
    {
        const ReluParams params{INPUT_UID, OUTPUT_UID, NUM_ELEMENTS, NEGATIVE_SLOPE};
        auto plan = std::make_unique<ReluPlan>(params);

        // Set up mock expectations: compiler returns a compiled program
        auto compiledProgram = std::make_unique<MockCompiledProgram>();
        _rawCompiledProgram = compiledProgram.get();

        auto kernel = std::make_unique<MockRunnableKernel>();
        _rawKernel = kernel.get();

        EXPECT_CALL(_mockCompiler, compile("ReluForward.cpp", _))
            .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

        EXPECT_CALL(*_rawCompiledProgram, getRunnableKernel("relu_forward_kernel"))
            .WillOnce(Return(testing::ByMove(std::move(kernel))));

        plan->compile(_mockCompiler);
        return plan;
    }
};

TEST_F(ReluPlanTest, GetWorkspaceSize_ReturnsZero)
{
    // Workspace size can be checked without compiling
    const ReluParams params{INPUT_UID, OUTPUT_UID, NUM_ELEMENTS, NEGATIVE_SLOPE};
    const ReluPlan plan{params};
    EXPECT_EQ(plan.getWorkspaceSize(_handle), 0u);
}

TEST_F(ReluPlanTest, Compile_CallsCompilerWithCorrectFilename)
{
    const ReluParams params{INPUT_UID, OUTPUT_UID, NUM_ELEMENTS, NEGATIVE_SLOPE};
    auto plan = std::make_unique<ReluPlan>(params);

    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();

    auto kernel = std::make_unique<MockRunnableKernel>();

    // Verify the compiler receives the correct kernel filename with empty options.
    EXPECT_CALL(_mockCompiler, compile("ReluForward.cpp", std::vector<std::string>{}))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

    EXPECT_CALL(*rawProgram, getRunnableKernel("relu_forward_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    plan->compile(_mockCompiler);
}

TEST_F(ReluPlanTest, Execute_SetsGridAndBlockSizeAndLaunches)
{
    auto plan = createAndCompilePlan();

    // Expect setBlockSize and setGridSize to be called with correct values
    // blockSize=256, numElements=6, gridSize=ceil(6/256)=1
    EXPECT_CALL(*_rawKernel, setBlockSize(256, 1, 1));
    EXPECT_CALL(*_rawKernel, setGridSize(1, 1, 1));
    EXPECT_CALL(*_rawKernel, launchImpl(nullptr, _));

    std::vector<float> inputData = {-3.0f, -1.0f, 0.0f, 1.0f, 2.5f, -0.5f};
    std::vector<float> outputData(NUM_ELEMENTS, -999.0f);

    std::array<hipdnnPluginDeviceBuffer_t, 2> buffers;
    buffers[0].uid = INPUT_UID;
    buffers[0].ptr = inputData.data();
    buffers[1].uid = OUTPUT_UID;
    buffers[1].ptr = outputData.data();

    plan->execute(_handle, buffers.data(), 2, nullptr);
}

TEST_F(ReluPlanTest, Execute_MissingBuffer_Throws)
{
    auto plan = createAndCompilePlan();

    std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Only provide input buffer, not output
    std::array<hipdnnPluginDeviceBuffer_t, 1> buffers;
    buffers[0].uid = INPUT_UID;
    buffers[0].ptr = inputData.data();

    EXPECT_THROW(plan->execute(_handle, buffers.data(), 1, nullptr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
