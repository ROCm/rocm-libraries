// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <stdexcept>
#include <unordered_map>

#include "BatchnormFwdTrainGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuBatchnormFwdTrainPlan.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_test_sdk::utilities;

TEST(TestGpuBatchnormFwdTrainPlanBuilder, PlanConstruction)
{
    auto builder = createBatchnormFwdTrainGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result = dynamic_cast<GpuBatchnormFwdTrainPlan<float, float, float, float, float>*>(
                            builtPlan.get())
                        != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, PlanConstructionWithSaveStatsTensors)
{
    auto builder = createBatchnormFwdTrainGraph({1, 3, 14, 14}, {588, 196, 14, 1}, true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result = dynamic_cast<GpuBatchnormFwdTrainPlan<float, float, float, float, float>*>(
                            builtPlan.get())
                        != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, PlanConstructionWithRunningStatsTensors)
{
    auto builder = createBatchnormFwdTrainGraph({1, 3, 14, 14}, {588, 196, 14, 1}, false, true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result = dynamic_cast<GpuBatchnormFwdTrainPlan<float, float, float, float, float>*>(
                            builtPlan.get())
                        != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicable)
{
    auto builder = createBatchnormFwdTrainGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuBatchnormFwdTrainPlanBuilder<DataType::HALF,
                                          DataType::HALF,
                                          DataType::HALF,
                                          DataType::HALF,
                                          DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableTrueWhenAllStatsPresent)
{
    auto builder = createBatchnormFwdTrainGraph({1, 3, 14, 14}, {588, 196, 14, 1}, true, true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenInputTypeMismatched)
{
    auto builder = createBatchnormFwdTrainGraph(
        {1, 3, 14, 14}, {588, 196, 14, 1}, false, false, DataType::HALF);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenScaleTypeMismatched)
{
    auto builder = createBatchnormFwdTrainGraph(
        {1, 3, 14, 14}, {588, 196, 14, 1}, false, false, DataType::FLOAT, DataType::BFLOAT16);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenMeanVarianceTypeMismatched)
{
    auto builder = createBatchnormFwdTrainGraph({1, 3, 14, 14},
                                                {588, 196, 14, 1},
                                                false,
                                                true,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::HALF);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenComputeTypeMismatched)
{
    auto builder = createBatchnormFwdTrainGraph({1, 3, 14, 14},
                                                {588, 196, 14, 1},
                                                false,
                                                false,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::DOUBLE);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, BuildNodePlanThrowsForMissingMomentumTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenBiasTensorMissing)
{
    auto builder = createBatchnormFwdTrainGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->bias_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenOutputTensorMissing)
{
    auto builder = createBatchnormFwdTrainGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->y_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenEpsilonTensorMissing)
{
    auto builder = createBatchnormFwdTrainGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->epsilon_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenMomentumTensorIsMissing)
{
    auto builder = createBatchnormFwdTrainGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    ASSERT_TRUE(nodeAttributes->momentum_tensor_uid().has_value());
    tensorMapCopy.erase(nodeAttributes->momentum_tensor_uid().value());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenNotAllSaveStatsTensorsArePresent)
{
    auto builder = createBatchnormFwdTrainGraph({1, 3, 14, 14}, {588, 196, 14, 1}, true, true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    auto tensorMapCopy1 = graph.getTensorMap();
    ASSERT_TRUE(nodeAttributes->mean_tensor_uid().has_value());
    tensorMapCopy1.erase(nodeAttributes->mean_tensor_uid().value());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy1));

    auto tensorMapCopy2 = graph.getTensorMap();
    ASSERT_TRUE(nodeAttributes->inv_variance_tensor_uid().has_value());
    tensorMapCopy2.erase(nodeAttributes->inv_variance_tensor_uid().value());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy2));
}

TEST(TestGpuBatchnormFwdTrainPlanBuilder, IsApplicableFalseWhenNotAllRunStatsTensorsArePresent)
{
    auto builder = createBatchnormFwdTrainGraph({1, 3, 14, 14}, {588, 196, 14, 1}, false, true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainPlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    auto tensorMapCopy1 = graph.getTensorMap();
    ASSERT_TRUE(nodeAttributes->prev_running_variance_tensor_uid().has_value());
    tensorMapCopy1.erase(nodeAttributes->prev_running_variance_tensor_uid().value());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy1));

    auto tensorMapCopy2 = graph.getTensorMap();
    ASSERT_TRUE(nodeAttributes->next_running_mean_tensor_uid().has_value());
    tensorMapCopy2.erase(nodeAttributes->next_running_mean_tensor_uid().value());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy2));

    auto tensorMapCopy3 = graph.getTensorMap();
    ASSERT_TRUE(nodeAttributes->next_running_variance_tensor_uid().has_value());
    tensorMapCopy3.erase(nodeAttributes->next_running_variance_tensor_uid().value());
    ASSERT_TRUE(nodeAttributes->prev_running_mean_tensor_uid().has_value());
    tensorMapCopy3.erase(nodeAttributes->prev_running_mean_tensor_uid().value());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy3));
}

namespace
{

template <typename DType>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& dims, const TensorLayout& layout)
{
    const auto strides = generateStrides(dims, layout.strideOrder);

    auto dataType = nativeTypeToDataType<DType>();

    auto builder = createBatchnormFwdTrainGraph(dims,
                                                strides,
                                                true,
                                                true,
                                                dataType,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                dataType,
                                                DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    const auto& tensorMap = graph.getTensorMap();

    GpuBatchnormFwdTrainParams params(
        *tensorMap.at(nodeAttributes->x_tensor_uid()),
        *tensorMap.at(nodeAttributes->scale_tensor_uid()),
        *tensorMap.at(nodeAttributes->bias_tensor_uid()),
        *tensorMap.at(nodeAttributes->y_tensor_uid()),
        *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
        *tensorMap.at(nodeAttributes->momentum_tensor_uid().value()),
        tensorMap.at(nodeAttributes->mean_tensor_uid().value()),
        tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value()),
        tensorMap.at(nodeAttributes->prev_running_mean_tensor_uid().value()),
        tensorMap.at(nodeAttributes->prev_running_variance_tensor_uid().value()),
        tensorMap.at(nodeAttributes->next_running_mean_tensor_uid().value()),
        tensorMap.at(nodeAttributes->next_running_variance_tensor_uid().value()));

    GpuBatchnormFwdTrainPlan<DType, float, float, DType, float> gpuPlan(std::move(params));

    const auto derivedDims = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
        tensorMap.at(nodeAttributes->scale_tensor_uid())->dims());
    const auto derivedStrides
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
            tensorMap.at(nodeAttributes->scale_tensor_uid())->strides());
    Tensor<DType> inputTensor(dims, strides);
    Tensor<float> scaleTensor(derivedDims, derivedStrides);
    Tensor<float> biasTensor(derivedDims, derivedStrides);
    Tensor<float> prevRunningMeanTensor(derivedDims, derivedStrides);
    Tensor<float> prevRunningVarianceTensor(derivedDims, derivedStrides);

    Tensor<DType> cpuOutput(dims, strides);
    Tensor<DType> gpuOutput(dims, strides);
    Tensor<float> cpuMean(derivedDims, derivedStrides);
    Tensor<float> gpuMean(derivedDims, derivedStrides);
    Tensor<float> cpuInvVariance(derivedDims, derivedStrides);
    Tensor<float> gpuInvVariance(derivedDims, derivedStrides);
    Tensor<float> cpuNextRunningMean(derivedDims, derivedStrides);
    Tensor<float> gpuNextRunningMean(derivedDims, derivedStrides);
    Tensor<float> cpuNextRunningVariance(derivedDims, derivedStrides);
    Tensor<float> gpuNextRunningVariance(derivedDims, derivedStrides);

    const auto seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();
    inputTensor.fillWithRandomValues(static_cast<DType>(-1.0f), static_cast<DType>(1.0f), seed);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 1);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    prevRunningMeanTensor.fillWithRandomValues(-1.0f, 1.0f, seed + 3);
    prevRunningVarianceTensor.fillWithRandomValues(0.1f, 1.0f, seed + 4);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[nodeAttributes->x_tensor_uid()] = inputTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->bias_tensor_uid()] = biasTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->y_tensor_uid()] = gpuOutput.rawDeviceData();
    gpuVariantPack[nodeAttributes->mean_tensor_uid().value()] = gpuMean.rawDeviceData();
    gpuVariantPack[nodeAttributes->inv_variance_tensor_uid().value()]
        = gpuInvVariance.rawDeviceData();
    gpuVariantPack[nodeAttributes->prev_running_mean_tensor_uid().value()]
        = prevRunningMeanTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->prev_running_variance_tensor_uid().value()]
        = prevRunningVarianceTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->next_running_mean_tensor_uid().value()]
        = gpuNextRunningMean.rawDeviceData();
    gpuVariantPack[nodeAttributes->next_running_variance_tensor_uid().value()]
        = gpuNextRunningVariance.rawDeviceData();

    gpuPlan.execute(gpuVariantPack);
    gpuOutput.markDeviceModified();
    gpuMean.markDeviceModified();
    gpuInvVariance.markDeviceModified();
    gpuNextRunningMean.markDeviceModified();
    gpuNextRunningVariance.markDeviceModified();

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[nodeAttributes->x_tensor_uid()] = inputTensor.rawHostData();
    cpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawHostData();
    cpuVariantPack[nodeAttributes->bias_tensor_uid()] = biasTensor.rawHostData();
    cpuVariantPack[nodeAttributes->y_tensor_uid()] = cpuOutput.rawHostData();
    cpuVariantPack[nodeAttributes->mean_tensor_uid().value()] = cpuMean.rawHostData();
    cpuVariantPack[nodeAttributes->inv_variance_tensor_uid().value()]
        = cpuInvVariance.rawHostData();
    cpuVariantPack[nodeAttributes->prev_running_mean_tensor_uid().value()]
        = prevRunningMeanTensor.rawHostData();
    cpuVariantPack[nodeAttributes->prev_running_variance_tensor_uid().value()]
        = prevRunningVarianceTensor.rawHostData();
    cpuVariantPack[nodeAttributes->next_running_mean_tensor_uid().value()]
        = cpuNextRunningMean.rawHostData();
    cpuVariantPack[nodeAttributes->next_running_variance_tensor_uid().value()]
        = cpuNextRunningVariance.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), cpuVariantPack);
    cpuOutput.markHostModified();
    cpuMean.markHostModified();
    cpuInvVariance.markHostModified();
    cpuNextRunningMean.markHostModified();
    cpuNextRunningVariance.markHostModified();

    const auto* cpuOutputData = static_cast<const DType*>(cpuOutput.rawHostData());
    const auto* gpuOutputData = static_cast<const DType*>(gpuOutput.rawHostData());
    for(size_t i = 0; i < cpuOutput.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuOutputData[i]),
                    static_cast<float>(cpuOutputData[i]),
                    batchnorm::getToleranceTraining<DType>())
            << "Mismatch in output Y at index " << i;
    }

    const auto* cpuMeanData = static_cast<const float*>(cpuMean.rawHostData());
    const auto* gpuMeanData = static_cast<const float*>(gpuMean.rawHostData());
    for(size_t i = 0; i < cpuMean.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuMeanData[i]),
                    static_cast<float>(cpuMeanData[i]),
                    batchnorm::getToleranceTraining<float>())
            << "Mismatch in mean at index " << i;
    }

    const auto* cpuInvVarianceData = static_cast<const float*>(cpuInvVariance.rawHostData());
    const auto* gpuInvVarianceData = static_cast<const float*>(gpuInvVariance.rawHostData());
    for(size_t i = 0; i < cpuInvVariance.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuInvVarianceData[i]),
                    static_cast<float>(cpuInvVarianceData[i]),
                    batchnorm::getToleranceTraining<float>())
            << "Mismatch in inverse variance at index " << i;
    }

    const auto* cpuNextRunningMeanData
        = static_cast<const float*>(cpuNextRunningMean.rawHostData());
    const auto* gpuNextRunningMeanData
        = static_cast<const float*>(gpuNextRunningMean.rawHostData());
    for(size_t i = 0; i < cpuNextRunningMean.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuNextRunningMeanData[i]),
                    static_cast<float>(cpuNextRunningMeanData[i]),
                    batchnorm::getToleranceTraining<float>())
            << "Mismatch in next running mean at index " << i;
    }

    const auto* cpuNextRunningVarianceData
        = static_cast<const float*>(cpuNextRunningVariance.rawHostData());
    const auto* gpuNextRunningVarianceData
        = static_cast<const float*>(gpuNextRunningVariance.rawHostData());
    for(size_t i = 0; i < cpuNextRunningVariance.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuNextRunningVarianceData[i]),
                    static_cast<float>(cpuNextRunningVarianceData[i]),
                    batchnorm::getToleranceTraining<float>())
            << "Mismatch in next running variance at index " << i;
    }
}

} // namespace

TEST(TestGpuBatchnormFwdTrainPlanFp32, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float>({1, 64, 56, 56}, TensorLayout::NCHW);
}

TEST(TestGpuBatchnormFwdTrainPlanFp32, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float>({1, 64, 56, 56}, TensorLayout::NHWC);
}

TEST(TestGpuBatchnormFwdTrainPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half>({1, 64, 56, 56}, TensorLayout::NCHW);
}

TEST(TestGpuBatchnormFwdTrainPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half>({1, 64, 56, 56}, TensorLayout::NHWC);
}

TEST(TestGpuBatchnormFwdTrainPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16>({1, 64, 56, 56}, TensorLayout::NCHW);
}

TEST(TestGpuBatchnormFwdTrainPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16>({1, 64, 56, 56}, TensorLayout::NHWC);
}
