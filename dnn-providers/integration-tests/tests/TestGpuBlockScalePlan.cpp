// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
using hipdnn_test_sdk::utilities::createValidBlockScaleDequantizeGraph;
using hipdnn_test_sdk::utilities::createValidBlockScaleQuantizeGraph;

TEST(TestGpuBlockScalePlan, QuantizeFloatGraphIsApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph();
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, DequantizeFloatGraphIsApplicable)
{
    auto graph = createValidBlockScaleDequantizeGraph();
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, CompactMxGraphIsApplicable)
{
    auto graph = createValidBlockScaleDequantizeGraph({65536, 2048, 64, 1},
                                                      {2, 32, 32, 64},
                                                      DataType::FP4_E2M1,
                                                      DataType::FLOAT,
                                                      false,
                                                      DataType::FP8_E8M0,
                                                      DataType::FLOAT);
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}
