// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <gtest/gtest.h>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/MockNode.hpp>

using hipdnn_test_sdk::utilities::MockEngineConfig;
using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;

class TestRMSnormPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    RMSnormPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
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

        auto mockKernel = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

        auto mockProgram = std::make_unique<MockCompiledProgram>();
        EXPECT_CALL(*mockProgram, getKernel(::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

        EXPECT_CALL(_mockKernelCompiler, compile(::testing::_, ::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));
    }

    static flatbuffers::FlatBufferBuilder createRMSnormActivationTestGraph(
        bool invertNodes,
        bool disconnectNodes,
        bool nonvirtualRMSOutput,
        bool virtualActivationOutput,
        bool leakyRelu,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode
        = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD)
    {
        const std::vector<int64_t>& strides = {150528, 50176, 224, 1};
        const std::vector<int64_t>& dims = {1, 3, 224, 224};

        flatbuffers::FlatBufferBuilder builder;
        std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
            tensorAttributes;

        std::vector<int64_t> derivedDims(dims);
        derivedDims[0] = 1; // Normalize bias/scale on first axis

        const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
            derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                1,
                "x",
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
                &dims,
                !nonvirtualRMSOutput));

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                3,
                "scale",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));

        // Epsilon (pass-by-value)
        const std::vector<int64_t> passByValueDims = {1};
        const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                4,
                "epsilon",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &passByValueDims,
                &passByValueDims,
                false,
                hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
                builder.CreateStruct(epsilonVal).Union()));

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                5,
                "yActiv",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &strides,
                &dims,
                virtualActivationOutput));

        if(disconnectNodes)
        {
            tensorAttributes.push_back(
                hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                    builder,
                    6,
                    "yActivIn",
                    hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                    &strides,
                    &dims));
        }

        auto rmsnormAttributes
            = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                            1, // x uid
                                                                            3, // scale uid
                                                                            4, // epsilon uid
                                                                            2 // y uid
            );

        auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            leakyRelu ? hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD
                      : pointwiseMode,
            leakyRelu ? std::nullopt : std::make_optional(0.1f),
            leakyRelu ? std::nullopt : std::make_optional(0.5f),
            leakyRelu ? std::make_optional(0.01f) : std::nullopt,
            std::nullopt,
            disconnectNodes ? 6 : 2, // yActivIn uid or y uid
            std::nullopt,
            std::nullopt,
            5, // yActiv uid
            std::nullopt,
            std::nullopt,
            std::nullopt);

        std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
        if(invertNodes)
        {
            auto nodePointwise = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "pointwise",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
                pointwiseAttributes.Union());
            nodes.push_back(nodePointwise);
            auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "rmsnorm",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
                rmsnormAttributes.Union());
            nodes.push_back(node);
        }
        else
        {
            auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "rmsnorm",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
                rmsnormAttributes.Union());
            nodes.push_back(node);
            auto nodePointwise = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
                builder,
                "pointwise",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
                pointwiseAttributes.Union());
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

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsTrueForValidInferenceGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsTrueForValidInferenceActivationGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}
TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledActivationGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForThreeNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForInvertedNodeOrderActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(true, false, false, false, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForDisconnectedActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, true, false, false, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForNonvirtualRMSOutputActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, false, true, false, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForVirtualActivationOutputActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, false, false, true, false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForLeakyReluActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(false, false, false, false, true);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsApplicableReturnsFalseForUnsupportedActivationModeActivationGraph)
{
    auto builder = createRMSnormActivationTestGraph(
        false,
        false,
        false,
        false,
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, BuildPlanSetsPlanForSingleNodeInference)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestRMSnormPlanBuilder, BuildPlanSetsPlanForDoubleNodeInferenceActivation)
{
    setupMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, IsNotApplicableForBatchnormGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsNotApplicableForNonF32ComputeType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestRMSnormPlanBuilder, IsNotApplicableForNonF32ComputeTypeActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

TEST_F(TestRMSnormPlanBuilder, GetMaxWorkspaceSizeReturnsZeroActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestRMSnormPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestRMSnormPlanBuilder, GetCustomKnobsReturnsEmptyActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}
