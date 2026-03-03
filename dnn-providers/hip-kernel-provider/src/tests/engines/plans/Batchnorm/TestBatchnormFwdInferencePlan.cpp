// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/plans/BatchnormFwdInferencePlan.hpp"
#include "mocks/MockKernelCompiler.hpp"

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;

// ============================================================================
// BatchnormFwdInferenceParams - construction from valid graph data
// ============================================================================

TEST(TestBatchnormFwdInferenceParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    EXPECT_NO_THROW(BatchnormFwdInferenceParams params(attr, graph.getTensorMap()));
}

TEST(TestBatchnormFwdInferenceParams, HasCorrectTensorPointersForSingleNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferenceParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.estMean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
}

// ============================================================================
// BatchnormFwdInferencePlan
// ============================================================================

namespace
{

BatchnormFwdInferencePlan createPlanFromSingleNodeGraph(const IKernelCompiler& kernelCompiler)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormFwdInferenceParams params(attr, graph.getTensorMap());
    return {std::move(params), kernelCompiler};
}

} // namespace

TEST(TestBatchnormFwdInferencePlan, ExecuteWithoutCompileThrows)
{
    MockKernelCompiler mockCompiler;
    auto plan = createPlanFromSingleNodeGraph(mockCompiler);
    HipKernelHandle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdInferencePlan, GetWorkspaceSizeReturnsZero)
{
    MockKernelCompiler mockCompiler;
    auto plan = createPlanFromSingleNodeGraph(mockCompiler);
    HipKernelHandle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}
