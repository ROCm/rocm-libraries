// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: Second Plan test example. This uses the same testing pattern
// as TestReluPlan.cpp but for a convolution kernel.

#include <gtest/gtest.h>

#include <array>
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
using ::testing::_; // NOLINT(bugprone-reserved-identifier)
using ::testing::Return;

class ConvFwdPlanTest : public ::testing::Test
{
protected:
    // Convolution dimensions: 1x1x4x4 input, 1x1x3x3 weight -> 1x1x2x2 output
    static constexpr int64_t INPUT_UID = 1;
    static constexpr int64_t WEIGHT_UID = 2;
    static constexpr int64_t OUTPUT_UID = 3;
    static constexpr int64_t N = 1;
    static constexpr int64_t C = 1;
    static constexpr int64_t H = 4;
    static constexpr int64_t W = 4;
    static constexpr int64_t K = 1;
    static constexpr int64_t R = 3;
    static constexpr int64_t S = 3;
    static constexpr int64_t OUT_H = 2;
    static constexpr int64_t OUT_W = 2;
    static constexpr int64_t PAD_H = 0;
    static constexpr int64_t PAD_W = 0;
    static constexpr int64_t STRIDE_H = 1;
    static constexpr int64_t STRIDE_W = 1;
    static constexpr int64_t BLOCK_SIZE = 256;

    MockKernelCompiler _mockCompiler;
    ExampleProviderHandle _handle;

    MockCompiledProgram* _rawCompiledProgram = nullptr;
    MockRunnableKernel* _rawKernel = nullptr;

    std::unique_ptr<ConvFwdPlan> createAndCompilePlan()
    {
        const ConvFwdParams params{INPUT_UID,
                                   WEIGHT_UID,
                                   OUTPUT_UID,
                                   N,
                                   C,
                                   H,
                                   W,
                                   K,
                                   R,
                                   S,
                                   OUT_H,
                                   OUT_W,
                                   PAD_H,
                                   PAD_W,
                                   STRIDE_H,
                                   STRIDE_W,
                                   BLOCK_SIZE};
        auto plan = std::make_unique<ConvFwdPlan>(params);

        auto compiledProgram = std::make_unique<MockCompiledProgram>();
        _rawCompiledProgram = compiledProgram.get();

        auto kernel = std::make_unique<MockRunnableKernel>();
        _rawKernel = kernel.get();

        EXPECT_CALL(_mockCompiler, compile("ConvForwardNaive.cpp", _))
            .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

        EXPECT_CALL(*_rawCompiledProgram, getRunnableKernel("conv_forward_naive_kernel"))
            .WillOnce(Return(testing::ByMove(std::move(kernel))));

        plan->compile(_mockCompiler);
        return plan;
    }
};

TEST_F(ConvFwdPlanTest, GetWorkspaceSize_ReturnsZero)
{
    const ConvFwdParams params{INPUT_UID,
                               WEIGHT_UID,
                               OUTPUT_UID,
                               N,
                               C,
                               H,
                               W,
                               K,
                               R,
                               S,
                               OUT_H,
                               OUT_W,
                               PAD_H,
                               PAD_W,
                               STRIDE_H,
                               STRIDE_W,
                               BLOCK_SIZE};
    const ConvFwdPlan plan{params};
    EXPECT_EQ(plan.getWorkspaceSize(_handle), 0u);
}

TEST_F(ConvFwdPlanTest, Compile_CallsCompilerWithCorrectFilename)
{
    const ConvFwdParams params{INPUT_UID,
                               WEIGHT_UID,
                               OUTPUT_UID,
                               N,
                               C,
                               H,
                               W,
                               K,
                               R,
                               S,
                               OUT_H,
                               OUT_W,
                               PAD_H,
                               PAD_W,
                               STRIDE_H,
                               STRIDE_W,
                               BLOCK_SIZE};
    auto plan = std::make_unique<ConvFwdPlan>(params);

    auto compiledProgram = std::make_unique<MockCompiledProgram>();
    auto* rawProgram = compiledProgram.get();
    auto kernel = std::make_unique<MockRunnableKernel>();

    // Verify the compiler receives the correct kernel filename with empty options.
    EXPECT_CALL(_mockCompiler, compile("ConvForwardNaive.cpp", std::vector<std::string>{}))
        .WillOnce(Return(testing::ByMove(std::move(compiledProgram))));

    EXPECT_CALL(*rawProgram, getRunnableKernel("conv_forward_naive_kernel"))
        .WillOnce(Return(testing::ByMove(std::move(kernel))));

    plan->compile(_mockCompiler);
}

TEST_F(ConvFwdPlanTest, Execute_SetsGridAndBlockSizeAndLaunches)
{
    auto plan = createAndCompilePlan();

    // Total output elements: N*K*outH*outW = 1*1*2*2 = 4
    // blockSize=256, gridSize=ceil(4/256)=1
    EXPECT_CALL(*_rawKernel, setBlockSize(256, 1, 1));
    EXPECT_CALL(*_rawKernel, setGridSize(1, 1, 1));
    EXPECT_CALL(*_rawKernel, launchImpl(nullptr, _));

    std::vector<float> inputData(N * C * H * W, 1.0f);
    std::vector<float> weightData(K * C * R * S, 1.0f);
    std::vector<float> outputData(N * K * OUT_H * OUT_W, -999.0f);

    std::array<hipdnnPluginDeviceBuffer_t, 3> buffers;
    buffers[0].uid = INPUT_UID;
    buffers[0].ptr = inputData.data();
    buffers[1].uid = WEIGHT_UID;
    buffers[1].ptr = weightData.data();
    buffers[2].uid = OUTPUT_UID;
    buffers[2].ptr = outputData.data();

    plan->execute(_handle, buffers.data(), 3, nullptr);
}

TEST_F(ConvFwdPlanTest, Execute_MissingBuffer_Throws)
{
    auto plan = createAndCompilePlan();

    std::vector<float> inputData(N * C * H * W, 1.0f);

    // Only provide input buffer, missing weight and output
    std::array<hipdnnPluginDeviceBuffer_t, 1> buffers;
    buffers[0].uid = INPUT_UID;
    buffers[0].ptr = inputData.data();

    EXPECT_THROW(plan->execute(_handle, buffers.data(), 1, nullptr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
