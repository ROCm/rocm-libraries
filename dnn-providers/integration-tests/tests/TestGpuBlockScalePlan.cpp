// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "RaggedGraphTestUtils.hpp"
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

TEST(TestGpuBlockScalePlan, InvalidScaleDimensionsAreNotApplicable)
{
    auto graph
        = createValidBlockScaleQuantizeGraph({2, 64, 32, 32}, {65536, 1024, 32, 1}, {2, 3, 32, 32});
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, RankAboveLimitIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph({1, 2, 3, 4, 5, 6},
                                                    {720, 360, 120, 30, 6, 1},
                                                    {1, 1, 3, 4, 5, 6},
                                                    {360, 360, 120, 30, 6, 1});
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, UnsupportedCompactTypeCombinationIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                    {65536, 1024, 32, 1},
                                                    {2, 2, 32, 32},
                                                    {2048, 1024, 32, 1},
                                                    32,
                                                    DataType::FLOAT,
                                                    DataType::FP4_E2M1,
                                                    DataType::FLOAT);
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, NonFloatComputeIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                    {65536, 1024, 32, 1},
                                                    {2, 2, 32, 32},
                                                    {2048, 1024, 32, 1},
                                                    32,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::DOUBLE);
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestGpuBlockScalePlan, RaggedTensorIsNotApplicable)
{
    auto graph = createValidBlockScaleQuantizeGraph();
    auto ragged
        = hipdnn_integration_tests::test_utils::markFirstTensorRagged(graph.GetBufferPointer());
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(ragged.data(), ragged.size()));
}
