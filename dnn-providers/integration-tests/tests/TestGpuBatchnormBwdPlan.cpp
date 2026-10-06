// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuBatchnormBwdPlan.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

namespace
{

constexpr int64_t X_UID = 1;
constexpr int64_t DY_UID = 2;
constexpr int64_t DX_UID = 3;
constexpr int64_t SCALE_UID = 4;
constexpr int64_t DSCALE_UID = 5;
constexpr int64_t DBIAS_UID = 6;
constexpr int64_t MEAN_UID = 7;
constexpr int64_t INV_VARIANCE_UID = 8;

void runPlanExecuteVsCpuRef(bool useSavedStatistics)
{
    const std::vector<int64_t> dims{2, 3, 2, 2};
    const std::vector<int64_t> strides{12, 4, 2, 1};
    const std::vector<int64_t> channelDims{1, 3, 1, 1};
    const std::vector<int64_t> channelStrides{3, 1, 1, 1};

    Tensor<float> x(dims, strides);
    Tensor<float> dy(dims, strides);
    Tensor<float> scale(channelDims, channelStrides);
    Tensor<float> mean(channelDims, channelStrides);
    Tensor<float> invVariance(channelDims, channelStrides);
    Tensor<float> gpuDx(dims, strides);
    Tensor<float> gpuDscale(channelDims, channelStrides);
    Tensor<float> gpuDbias(channelDims, channelStrides);
    Tensor<float> cpuDx(dims, strides);
    Tensor<float> cpuDscale(channelDims, channelStrides);
    Tensor<float> cpuDbias(channelDims, channelStrides);

    unsigned int seed = 42;
    x.fillWithRandomValues(-1.0f, 1.0f, seed++);
    dy.fillWithRandomValues(-0.1f, 0.1f, seed++);
    scale.fillWithRandomValues(-0.1f, 0.1f, seed++);
    mean.fillWithRandomValues(-0.1f, 0.1f, seed++);
    invVariance.fillWithRandomValues(1.9f, 2.0f, seed++);

    auto graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        strides, dims, useSavedStatistics);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const auto* attributes = graph.getNode(0).attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attributes, nullptr);

    const auto& tensorMap = graph.getTensorMap();
    GpuBatchnormBwdParams params(*tensorMap.at(DY_UID),
                                 *tensorMap.at(X_UID),
                                 *tensorMap.at(SCALE_UID),
                                 *tensorMap.at(DX_UID),
                                 *tensorMap.at(DSCALE_UID),
                                 *tensorMap.at(DBIAS_UID),
                                 useSavedStatistics ? tensorMap.at(MEAN_UID) : nullptr,
                                 useSavedStatistics ? tensorMap.at(INV_VARIANCE_UID) : nullptr);
    GpuBatchnormBwdPlan<float, float, float, float, float, float> plan(std::move(params));

    hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::backward(
        dy,
        x,
        scale,
        cpuDx,
        cpuDscale,
        cpuDbias,
        useSavedStatistics ? &mean : nullptr,
        useSavedStatistics ? &invVariance : nullptr);

    std::unordered_map<int64_t, void*> variantPack{{X_UID, x.rawDeviceData()},
                                                   {DY_UID, dy.rawDeviceData()},
                                                   {DX_UID, gpuDx.rawDeviceData()},
                                                   {SCALE_UID, scale.rawDeviceData()},
                                                   {DSCALE_UID, gpuDscale.rawDeviceData()},
                                                   {DBIAS_UID, gpuDbias.rawDeviceData()}};
    if(useSavedStatistics)
    {
        variantPack[MEAN_UID] = mean.rawDeviceData();
        variantPack[INV_VARIANCE_UID] = invVariance.rawDeviceData();
    }

    plan.execute(variantPack);
    gpuDx.markDeviceModified();
    gpuDscale.markDeviceModified();
    gpuDbias.markDeviceModified();

    constexpr float TOLERANCE = 1.0e-5f;
    const auto* cpuDxData = static_cast<const float*>(cpuDx.rawHostData());
    const auto* cpuDscaleData = static_cast<const float*>(cpuDscale.rawHostData());
    const auto* cpuDbiasData = static_cast<const float*>(cpuDbias.rawHostData());
    const auto* gpuDxData = static_cast<const float*>(gpuDx.rawHostData());
    const auto* gpuDscaleData = static_cast<const float*>(gpuDscale.rawHostData());
    const auto* gpuDbiasData = static_cast<const float*>(gpuDbias.rawHostData());
    for(size_t i = 0; i < cpuDx.elementCount(); ++i)
    {
        EXPECT_NEAR(cpuDxData[i], gpuDxData[i], TOLERANCE);
    }
    for(size_t i = 0; i < cpuDscale.elementCount(); ++i)
    {
        EXPECT_NEAR(cpuDscaleData[i], gpuDscaleData[i], TOLERANCE);
        EXPECT_NEAR(cpuDbiasData[i], gpuDbiasData[i], TOLERANCE);
    }
}

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

TEST(TestGpuBatchnormBwdPlan, ExecuteWithSavedStatistics)
{
    runPlanExecuteVsCpuRef(true);
}

TEST(TestGpuBatchnormBwdPlan, ExecuteWithoutSavedStatistics)
{
    runPlanExecuteVsCpuRef(false);
}
