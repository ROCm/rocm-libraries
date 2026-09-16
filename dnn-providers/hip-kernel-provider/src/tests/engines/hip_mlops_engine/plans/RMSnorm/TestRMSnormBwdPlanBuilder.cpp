// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;
using hipdnn_test_sdk::utilities::MockEngineConfig;

class TestRMSnormBwdPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    RMSnormBwdPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
    Handle _dummyHandle;
    MockEngineConfig _mockEngineConfig;

    void setupMockCompileChain()
    {
        hipDeviceProp_t deviceProps = {};
        deviceProps.multiProcessorCount = 60;
        deviceProps.warpSize = 64;
        std::snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", "gfx942");

        EXPECT_CALL(_mockDevicePropertyProvider, getDeviceProperties())
            .WillOnce(::testing::Return(deviceProps));

        // First mock kernel for BwdData
        auto mockKernel1 = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernel1, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*mockKernel1, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

        // Second mock kernel for BwdWeightBias
        auto mockKernel2 = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernel2, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*mockKernel2, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

        auto mockProgram = std::make_unique<MockCompiledProgram>();
        EXPECT_CALL(*mockProgram, getKernel(::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel1))))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel2))));

        EXPECT_CALL(_mockKernelCompiler, compile(::testing::_, ::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));
    }

    static flatbuffers::FlatBufferBuilder createRMSnormActivationTestGraph(
        bool invertNodes,
        bool disconnectNodes,
        bool nonvirtualActivationOutput,
        bool virtualRMSBwdOutput,
        bool leakyRelu,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode
        = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD)
    {
        const std::vector<int64_t>& strides = {150528, 50176, 224, 1};
        const std::vector<int64_t>& dims = {2, 3, 224, 224};

        flatbuffers::FlatBufferBuilder builder;
        std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
            tensorAttributes;

        std::vector<int64_t> derivedDims(dims);
        derivedDims[0] = 1; // Normalize bias/scale on first axis
        const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
            derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

        // inv_rms stat shape is [N, 1, 1, 1, ...] when scale is [1, C, H, W ..]
        std::vector<int64_t> statDims(dims.size(), 1);
        statDims[0] = dims[0];
        const std::vector<int64_t> statStrides = hipdnn_data_sdk::utilities::generateStrides(
            statDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

        // dy (gradient of output)
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                1,
                "dy",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &strides,
                &dims));

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                2,
                "y",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &strides,
                &dims));

        // x (original input)
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                3,
                "x",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &strides,
                &dims));

        // scale
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                4,
                "scale",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));

        // dx (gradient of input)
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                5,
                "dx",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &strides,
                &dims,
                virtualRMSBwdOutput));

        // dscale (gradient of scale)
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                6,
                "dscale",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims,
                virtualRMSBwdOutput));

        // inv_rms (inverse RMS from forward pass)
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                7,
                "inv_rms",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &statStrides,
                &statDims,
                virtualRMSBwdOutput));

        // dbias (gradient of bias)
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                8,
                "dbias",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                9,
                "dyActiv",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &strides,
                &dims,
                !nonvirtualActivationOutput));

        if(disconnectNodes)
        {
            tensorAttributes.push_back(
                hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                    builder,
                    10,
                    "dyIn",
                    hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                    &strides,
                    &dims));
        }

        auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            leakyRelu ? hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD
                      : pointwiseMode,
            leakyRelu ? std::nullopt : std::make_optional(0.1f),
            leakyRelu ? std::nullopt : std::make_optional(0.5f),
            leakyRelu ? std::make_optional(0.01f) : std::nullopt,
            std::nullopt,
            1, // dy uid
            2, // y uid
            std::nullopt,
            9, // dyActiv uid
            std::nullopt,
            std::nullopt,
            std::nullopt);

        auto rmsnormBwdAttributes
            = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormBackwardAttributes(
                builder,
                disconnectNodes ? 10 : 9, // dyIn uid or dyActiv uid
                3, // x uid
                4, // scale uid
                7, // inv_rms uid
                5, // dx uid
                6, // dscale uid
                flatbuffers::Optional<int64_t>(8) // dbias uid
            );

        std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
        if(invertNodes)
        {
            auto nodeBwd = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "rmsnorm_bwd",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
                rmsnormBwdAttributes.Union());
            nodes.push_back(nodeBwd);
            auto nodePointwise = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "pointwise",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
                pointwiseAttributes.Union());
            nodes.push_back(nodePointwise);
        }
        else
        {
            auto nodePointwise = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "pointwise",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
                pointwiseAttributes.Union());
            auto nodeBwd = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "rmsnorm_bwd",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
                rmsnormBwdAttributes.Union());
            nodes.push_back(nodeBwd);
            nodes.push_back(nodePointwise);
        }

        auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
            builder,
            "test",
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
            hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
            &tensorAttributes,
            &nodes,
            flatbuffers::nullopt,
            false);
        builder.Finish(graphOffset);
        return builder;
    }
};

// ============================================================================
// isApplicable - valid graphs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueForValidSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueForValidDoubleNodeActivationGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1},
        {2, 3, 224, 224},
        /*hasOptionalAttributes=*/true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledActivationGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1},
        {2, 3, 224, 224},
        /*hasOptionalAttributes=*/true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueWithoutOptionalAttributes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsApplicableReturnsTrueWithoutOptionalAttributesActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForBatchnormGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForInvertedNodeOrderActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(true, false, false, false, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForDisconnectedActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, true, false, false, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForNonvirtualActivationOutputActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, false, true, false, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForVirtualRMSBwdOutputActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, false, false, true, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForLeakyReluActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, false, false, false, true);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForUnsupportedActivationModeActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(
        false,
        false,
        false,
        false,
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForNonF32ComputeType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormBwdPlanBuilder, IsNotApplicableForNonF32ComputeTypeActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        true,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, BuildPlanSetsPlanForSingleNodeGraph)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestRMSnormBwdPlanBuilder, BuildPlanSetsPlanForDoubleNodeActivationGraph)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

TEST_F(TestRMSnormBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZeroActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestRMSnormBwdPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestRMSnormBwdPlanBuilder, GetCustomKnobsReturnsEmptyActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
