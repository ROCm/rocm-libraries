// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <gtest/gtest.h>
#include <hipdnn_frontend/Types.hpp>

#include "hip_kernel_provider_common/HipDeviceUtils.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/MockNode.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace hip_kernel_provider::layernorm::test
{

class TestLayernormPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    LayernormPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
    Handle _dummyHandle;
    MockEngineConfig _mockEngineConfig;

    void setupFpropMockCompileChain()
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

    void setupBpropMockCompileChain()
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
        auto mockKernelScaleBias = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernelScaleBias, setBlockSize(::testing::_, ::testing::_, ::testing::_))
            .Times(1);
        EXPECT_CALL(*mockKernelScaleBias, setGridSize(::testing::_, ::testing::_, ::testing::_))
            .Times(1);

        auto mockProgram = std::make_unique<MockCompiledProgram>();
        EXPECT_CALL(*mockProgram, getKernel("LayernormBwd"))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));
        EXPECT_CALL(*mockProgram, getKernel("LayernormBwdScaleBias"))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernelScaleBias))));

        EXPECT_CALL(_mockKernelCompiler, compile(::testing::_, ::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));
    }
};

// ============================================================================
// isApplicable - valid graphs
// ============================================================================

TEST_F(TestLayernormPlanBuilder, IsApplicableReturnsTrueForValidInferenceGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestLayernormPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph(
        {588, 196, 14, 1},
        {1, 3, 14, 14},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestLayernormPlanBuilder, IsApplicableReturnsTrueForValidBpropGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestLayernormPlanBuilder, IsApplicableReturnsFalseForTwoNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    EXPECT_CALL(mockGraph, getGraph()).Times(::testing::AnyNumber());
    // nodeWrappers is only used in an all_of check which will pass when it's empty
    std::vector<std::unique_ptr<INodeWrapper>> nodeWrappers;
    EXPECT_CALL(mockGraph, nodeWrappers()).WillRepeatedly(::testing::ReturnRef(nodeWrappers));

    const bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);

    EXPECT_FALSE(applicable);
}

TEST_F(TestLayernormPlanBuilder, IsNotApplicableForBatchnormGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestLayernormPlanBuilder, IsNotApplicableForNonF32ComputeType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestLayernormPlanBuilder, BuildPlanSetsPlanForSingleFpropNode)
{
    setupFpropMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestLayernormPlanBuilder, BuildPlanSetsPlanForSingleBpropNode)
{
    setupBpropMockCompileChain();

    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestLayernormPlanBuilder, GetMaxWorkspaceSizeFpropReturnsZero)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

TEST_F(TestLayernormPlanBuilder, GetMaxWorkspaceSizeBpropReturnsZero)
{
    SKIP_IF_NO_DEVICES(); // Backward branch of getMaxWorkspaceSize requires a device

    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

TEST_F(TestLayernormPlanBuilder, GetMaxWorkspaceSizeBpropParallelReturnsNonzero)
{
    SKIP_IF_NO_DEVICES(); // Backward branch of getMaxWorkspaceSize requires a device

    // The answer is dependent on the number of multiprocessors
    auto multiProcessorCount = static_cast<int64_t>(
        hip_kernel_provider_common::getDeviceProperties(_dummyHandle.getStream())
            .multiProcessorCount);
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
        {3072 * multiProcessorCount, 1024 * multiProcessorCount, 32, 1},
        {4, 3, 32 * multiProcessorCount, 32});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    // The answer is `sizeof(float) * 2 * innerSize * parallelSize` = `4 * 2 * (32 * 32 * 3 * multiProcessorCount) * 2` = `49152 * multiProcessorCount`
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings),
              49152u * multiProcessorCount);
}

TEST_F(TestLayernormPlanBuilder, GetMaxWorkspaceSizeBpropWithoutOptionalTensorsReturnsNonzero)
{
    SKIP_IF_NO_DEVICES(); // Backward branch of getMaxWorkspaceSize requires a device

    const std::vector<int64_t> strides{150528, 50176, 224, 1};
    const std::vector<int64_t> dims{2, 3, 224, 224};
    auto builderWithStats = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
        strides, dims, true, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    auto builderWithoutStats = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph(
        strides, dims, false, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWithStats(
        builderWithStats.GetBufferPointer(), builderWithStats.GetSize());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWithoutStats(
        builderWithoutStats.GetBufferPointer(), builderWithoutStats.GetSize());
    const Settings settings;

    // Both graphs describe the same shape, so they share whatever parallel reduction buffer this
    // device's multiprocessor count calls for. Their difference is the fallback mean/rstd
    // workspace on its own: `sizeof(float) * 2 * outerSize * stride` = `4 * 2 * 2 * 1` = `16`
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graphWithoutStats, settings)
                  - _planBuilder.getMaxWorkspaceSize(_dummyHandle, graphWithStats, settings),
              16u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestLayernormPlanBuilder, GetCustomKnobsReturnsEmptyFprop)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestLayernormPlanBuilder, GetCustomKnobsReturnsEmptyBprop)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

// ============================================================================
// isApplicable - graphs the kernels cannot serve
// ============================================================================

TEST_F(TestLayernormPlanBuilder, ElementCountAtTheKernelIndexLimit)
{
    constexpr int64_t LAST_FITTING = int64_t{UINT32_MAX} - 1;
    for(const bool backward : {false, true})
    {
        SCOPED_TRACE(backward ? "backward" : "forward");
        for(const auto [columns, expected] : {std::pair{LAST_FITTING, true},
                                              std::pair{int64_t{UINT32_MAX}, false},
                                              std::pair{(int64_t{1} << 32) + 1, false}})
        {
            SCOPED_TRACE(columns);
            const std::vector<int64_t> dims{1, 1, 1, columns};
            const std::vector<int64_t> strides{columns, columns, columns, 1};
            auto builder = backward ? createValidLayernormBwdGraph(strides, dims)
                                    : createValidLayernormFpropGraph(strides, dims);
            const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
            EXPECT_EQ(_planBuilder.isApplicable(_dummyHandle, graph), expected);
        }
    }
}

TEST_F(TestLayernormPlanBuilder, DeclinesAVirtualKernelBuffer)
{
    const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
        {"forward", {"x", "y", "scale", "bias"}},
        {"backward", {"dy", "x", "dx", "scale", "dscale", "dbias", "mean", "inv_variance"}},
    };
    for(const auto& [direction, names] : cases)
    {
        for(const auto& name : names)
        {
            SCOPED_TRACE(direction);
            SCOPED_TRACE(name);
            auto builder
                = mutateGraph(direction == "forward" ? createValidLayernormFpropGraph()
                                                     : createValidLayernormBwdGraph(),
                              [&name](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
                                  for(auto& tensor : graph.tensors)
                                  {
                                      if(tensor->name == name)
                                      {
                                          tensor->virtual_ = true;
                                      }
                                  }
                              });
            const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
            EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
        }
    }
}

TEST_F(TestLayernormPlanBuilder, DeclinesAnEpsilonTheGraphDoesNotResolve)
{
    // The plan resolves epsilon as a scalar operand when it is built: a uid the graph lacks and
    // a non-runtime tensor with no baked value have nothing to resolve.
    for(const bool backward : {false, true})
    {
        SCOPED_TRACE(backward ? "backward" : "forward");
        const auto base = [backward] {
            return backward ? createValidLayernormBwdGraph() : createValidLayernormFpropGraph();
        };
        const auto epsilonUid = [backward](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
            auto* node = graph.nodes.at(0).get();
            return backward ? node->attributes.AsLayernormBackwardAttributes()
                                  ->epsilon_tensor_uid.value()
                            : node->attributes.AsLayernormAttributes()->epsilon_tensor_uid;
        };

        {
            auto builder = base();
            const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
            ASSERT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
        }
        {
            auto builder
                = mutateGraph(base(), [&](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
                      const auto uid = epsilonUid(graph);
                      for(auto& tensor : graph.tensors)
                      {
                          if(tensor->uid == uid)
                          {
                              tensor->value.Reset();
                          }
                      }
                  });
            const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
            EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph)) << "valueless epsilon";
        }
        {
            auto builder
                = mutateGraph(base(), [&](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
                      auto* node = graph.nodes.at(0).get();
                      if(backward)
                      {
                          node->attributes.AsLayernormBackwardAttributes()->epsilon_tensor_uid
                              = 9999;
                      }
                      else
                      {
                          node->attributes.AsLayernormAttributes()->epsilon_tensor_uid = 9999;
                      }
                  });
            const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
            EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph)) << "unknown uid";
        }
    }
}

TEST_F(TestLayernormPlanBuilder, AcceptsABakedOrRuntimeEpsilon)
{
    for(const bool backward : {false, true})
    {
        SCOPED_TRACE(backward ? "backward" : "forward");
        auto builder = mutateGraph(backward ? createValidLayernormBwdGraph()
                                            : createValidLayernormFpropGraph(),
                                   [](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
                                       for(auto& tensor : graph.tensors)
                                       {
                                           if(tensor->name == "epsilon")
                                           {
                                               tensor->value.Reset();
                                               tensor->is_runtime_pass_by_value = true;
                                           }
                                       }
                                   });
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
    }
}

} // namespace hip_kernel_provider::layernorm::test
