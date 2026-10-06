// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuBatchnormBwdPlan.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

namespace
{

constexpr int64_t SCALE_UID = 4;

} // namespace

TEST(TestGpuBatchnormBwdPlanBuilder, PlanConstructionAndApplicability)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({12, 4, 2, 1}, {2, 3, 2, 2});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormBwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT>
        planBuilder;

    EXPECT_TRUE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    auto plan = planBuilder.buildNodePlan(graph, graph.getNode(0));
    const bool hasExpectedType
        = dynamic_cast<GpuBatchnormBwdPlan<float, float, float, float, float, float>*>(plan.get())
          != nullptr;
    EXPECT_TRUE(hasExpectedType);

    auto tensorMap = graph.getTensorMap();
    tensorMap.erase(SCALE_UID);
    EXPECT_FALSE(planBuilder.isApplicable(graph.getNode(0), tensorMap));
}

TEST(TestGpuBatchnormBwdPlan, ExecutorIsApplicable)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({12, 4, 2, 1}, {2, 3, 2, 2});
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;

    EXPECT_TRUE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}
