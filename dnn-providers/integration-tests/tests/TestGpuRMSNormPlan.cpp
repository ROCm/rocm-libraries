// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "EpsilonTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuRMSNormPlan.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_integration_tests::test_utils;

// =============================================================
// Test GpuRMSNormFwdPlan
// =============================================================

TEST(TestGpuRMSNormFwdPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuRMSNormFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuRMSNormFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormFwdPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicableFalseWhenScaleTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->scale_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicableFalseWhenTensorTypeMismatched)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicableAcceptsEpsilonTypeDifferentFromComputeType)
{
    // Helper uses float for epsilon type, which differs for double type we've passed for compute
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph({150528, 50176, 224, 1},
                                                                       {1, 3, 224, 224},
                                                                       DataType::FLOAT, // input
                                                                       DataType::DOUBLE // compute
    );

    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::DOUBLE>
        planBuilder;

    EXPECT_TRUE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

// =============================================================
// Test GpuRMSNormBwdPlan
// =============================================================

TEST(TestGpuRMSNormBwdPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuRMSNormBwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuRMSNormBwdPlanBuilder<DataType::HALF,
                                   DataType::HALF,
                                   DataType::HALF,
                                   DataType::HALF,
                                   DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormBackwardAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->dy_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormBwdPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableFalseWhenInputTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormBackwardAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableFalseWhenDxTensorTypeMismatched)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        false,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableFalseWhenInvRmsTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormBackwardAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->inv_rms_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormBwdPlanBuilder, BuildNodePlanSucceedsWithOptionalDbias)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));
    const bool result
        = dynamic_cast<GpuRMSNormBwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

namespace
{
template <typename XType, typename ScaleType, typename YType, typename ComputeType>
void runFwdPlanExecuteVsCpuRef(const std::vector<int64_t>& ioDims,
                               const TensorLayout& layout,
                               float tolerance,
                               DataType epsilonDataType = DataType::UNSET)
{
    const auto ioStrides = generateStrides(ioDims, layout.strideOrder);

    constexpr int64_t X_UID = 1;
    constexpr int64_t Y_UID = 2;
    constexpr int64_t SCALE_UID = 3;
    constexpr int64_t EPSILON_UID = 4;
    auto xDataType = nativeTypeToDataType<XType>();
    auto yDataType = nativeTypeToDataType<YType>();
    auto scaleDataType = nativeTypeToDataType<ScaleType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();
    if(epsilonDataType == DataType::UNSET)
    {
        epsilonDataType = computeDataType;
    }

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> derivedDims(ioDims);
    derivedDims[0] = 1; // Normalize bias/scale on first axis

    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, X_UID, "x", xDataType, &ioStrides, &ioDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, Y_UID, "y", yDataType, &ioStrides, &ioDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, SCALE_UID, "scale", scaleDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(
        createEpsilonTensorAttributes(builder, EPSILON_UID, 1e-5, epsilonDataType));

    auto rmsnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                        X_UID, // x uid
                                                                        SCALE_UID, // scale uid
                                                                        EPSILON_UID, // epsilon uid
                                                                        Y_UID // y uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    const auto& tensorMap = graph.getTensorMap();

    GpuRMSNormFwdParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                               *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                               *tensorMap.at(nodeAttributes->y_tensor_uid()),
                               *tensorMap.at(nodeAttributes->epsilon_tensor_uid()));

    GpuRMSNormFwdPlan<XType, ScaleType, YType, ComputeType> gpuPlan(std::move(params));

    Tensor<XType> xTensor(ioDims, ioStrides);
    Tensor<ScaleType> scaleTensor(derivedDims, derivedStrides);
    Tensor<YType> cpuY(ioDims, ioStrides);
    Tensor<YType> gpuY(ioDims, ioStrides);

    constexpr unsigned int SEED = 42;
    xTensor.fillWithRandomValues(static_cast<XType>(-1.0f), static_cast<XType>(1.0f), SEED);
    scaleTensor.fillWithRandomValues(
        static_cast<ScaleType>(-1.0f), static_cast<ScaleType>(1.0f), SEED + 1);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[nodeAttributes->x_tensor_uid()] = xTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->y_tensor_uid()] = gpuY.rawDeviceData();
    gpuPlan.execute(gpuVariantPack);
    gpuY.markDeviceModified();

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[nodeAttributes->x_tensor_uid()] = xTensor.rawHostData();
    cpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawHostData();
    cpuVariantPack[nodeAttributes->y_tensor_uid()] = cpuY.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), cpuVariantPack);
    cpuY.markHostModified();

    const auto* cpuYData = static_cast<const YType*>(cpuY.rawHostData());
    const auto* gpuYData = static_cast<const YType*>(gpuY.rawHostData());
    for(size_t i = 0; i < cpuY.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuYData[i]), static_cast<float>(cpuYData[i]), tolerance)
            << "Mismatch in y at index " << i;
    }
}

template <typename GradOutputType,
          typename InputType,
          typename ScaleType,
          typename GradInputType,
          typename ComputeType>
void runBwdPlanExecuteVsCpuRef(const std::vector<int64_t>& ioDims,
                               const TensorLayout& layout,
                               float tolerance)
{
    std::vector<int64_t> derivedDims(ioDims);
    derivedDims[0] = 1; // Normalize bias/scale on first axis
    std::vector<int64_t> invRMSDims(ioDims.size(), 1);
    invRMSDims[0] = ioDims[0];

    const auto ioStrides = generateStrides(ioDims, layout.strideOrder);
    const auto derivedStrides = generateStrides(derivedDims, layout.strideOrder);
    const auto invRMSStrides = generateStrides(invRMSDims, layout.strideOrder);

    constexpr int64_t DY_UID = 1;
    constexpr int64_t X_UID = 2;
    constexpr int64_t SCALE_UID = 3;
    constexpr int64_t INV_RMS_UID = 4;
    constexpr int64_t DX_UID = 5;
    constexpr int64_t DSCALE_UID = 6;
    constexpr int64_t DBIAS_UID = 7;

    auto dyDataType = nativeTypeToDataType<GradOutputType>();
    auto xDataType = nativeTypeToDataType<InputType>();
    auto scaleDataType = nativeTypeToDataType<ScaleType>();
    auto dxDataType = nativeTypeToDataType<GradInputType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // dy (gradient of output)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, DY_UID, "dy", dyDataType, &ioStrides, &ioDims));

    // x (original input)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, X_UID, "x", xDataType, &ioStrides, &ioDims));

    // scale
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, SCALE_UID, "scale", scaleDataType, &derivedStrides, &derivedDims));

    // dx (gradient of input)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, DX_UID, "dx", dxDataType, &ioStrides, &ioDims));

    // dscale (gradient of scale)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, DSCALE_UID, "dscale", scaleDataType, &derivedStrides, &derivedDims));

    // inv_rms (inverse RMS from forward pass)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, INV_RMS_UID, "inv_rms", computeDataType, &invRMSStrides, &invRMSDims));

    // dbias (gradient of bias)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, DBIAS_UID, "dbias", scaleDataType, &derivedStrides, &derivedDims));

    auto rmsnormBwdAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormBackwardAttributes(
            builder,
            DY_UID, // dy uid
            X_UID, // x uid
            SCALE_UID, // scale uid
            INV_RMS_UID, // inv_rms uid
            DX_UID, // dx uid
            DSCALE_UID, // dscale uid
            DBIAS_UID // dbias_uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm_bwd",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
        rmsnormBwdAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormBackwardAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    const auto& tensorMap = graph.getTensorMap();

    GpuRMSNormBwdParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                               *tensorMap.at(nodeAttributes->x_tensor_uid()),
                               *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                               *tensorMap.at(nodeAttributes->inv_rms_tensor_uid()),
                               *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                               *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                               tensorMap.at(nodeAttributes->dbias_tensor_uid().value()));

    GpuRMSNormBwdPlan<GradOutputType, InputType, ScaleType, GradInputType, ComputeType> gpuPlan(
        std::move(params));

    // Input tensors
    Tensor<GradOutputType> dyTensor(ioDims, ioStrides);
    Tensor<InputType> xTensor(ioDims, ioStrides);
    Tensor<ScaleType> scaleTensor(derivedDims, derivedStrides);
    Tensor<ComputeType> invRMSTensor(invRMSDims, invRMSStrides);

    // CPU output tensors
    Tensor<GradInputType> dxCpuTensor(ioDims, ioStrides);
    Tensor<ScaleType> dCpuScaleTensor(derivedDims, derivedStrides);
    Tensor<ScaleType> dCpuBiasTensor(derivedDims, derivedStrides);

    // GPU output tensors
    Tensor<GradInputType> dxGpuTensor(ioDims, ioStrides);
    Tensor<ScaleType> dGpuScaleTensor(derivedDims, derivedStrides);
    Tensor<ScaleType> dGpuBiasTensor(derivedDims, derivedStrides);

    unsigned int seed = 42;
    dyTensor.fillWithRandomValues(
        static_cast<GradOutputType>(-1.0f), static_cast<GradOutputType>(1.0f), seed++);
    xTensor.fillWithRandomValues(
        static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed++);
    scaleTensor.fillWithRandomValues(
        static_cast<ScaleType>(-1.0f), static_cast<ScaleType>(1.0f), seed++);
    invRMSTensor.fillWithRandomValues(
        static_cast<ComputeType>(.5f), static_cast<ComputeType>(2.0f), seed++);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[nodeAttributes->dy_tensor_uid()] = dyTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->x_tensor_uid()] = xTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->inv_rms_tensor_uid()] = invRMSTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->dx_tensor_uid()] = dxGpuTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->dscale_tensor_uid()] = dGpuScaleTensor.rawDeviceData();
    gpuVariantPack[nodeAttributes->dbias_tensor_uid().value()] = dGpuBiasTensor.rawDeviceData();

    gpuPlan.execute(gpuVariantPack);
    dxGpuTensor.markDeviceModified();
    dGpuScaleTensor.markDeviceModified();
    dGpuBiasTensor.markDeviceModified();

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[nodeAttributes->dy_tensor_uid()] = dyTensor.rawHostData();
    cpuVariantPack[nodeAttributes->x_tensor_uid()] = xTensor.rawHostData();
    cpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawHostData();
    cpuVariantPack[nodeAttributes->inv_rms_tensor_uid()] = invRMSTensor.rawHostData();
    cpuVariantPack[nodeAttributes->dx_tensor_uid()] = dxCpuTensor.rawHostData();
    cpuVariantPack[nodeAttributes->dscale_tensor_uid()] = dCpuScaleTensor.rawHostData();
    cpuVariantPack[nodeAttributes->dbias_tensor_uid().value()] = dCpuBiasTensor.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), cpuVariantPack);
    dxCpuTensor.markHostModified();
    dCpuScaleTensor.markHostModified();
    dCpuBiasTensor.markHostModified();

    const auto* cpuDXData = static_cast<const GradInputType*>(dxCpuTensor.rawHostData());
    const auto* gpuDXData = static_cast<const GradInputType*>(dxGpuTensor.rawHostData());
    for(size_t i = 0; i < dxCpuTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuDXData[i]), static_cast<float>(cpuDXData[i]), tolerance)
            << "Mismatch in dx at index " << i;
    }

    const auto* cpuDScaleData = static_cast<const ScaleType*>(dCpuScaleTensor.rawHostData());
    const auto* gpuDScaleData = static_cast<const ScaleType*>(dGpuScaleTensor.rawHostData());
    for(size_t i = 0; i < dCpuScaleTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(
            static_cast<float>(gpuDScaleData[i]), static_cast<float>(cpuDScaleData[i]), tolerance)
            << "Mismatch in dscale at index " << i;
    }

    const auto* cpuDBiasData = static_cast<const ScaleType*>(dCpuBiasTensor.rawHostData());
    const auto* gpuDBiasData = static_cast<const ScaleType*>(dGpuBiasTensor.rawHostData());
    for(size_t i = 0; i < dCpuBiasTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(
            static_cast<float>(gpuDBiasData[i]), static_cast<float>(cpuDBiasData[i]), tolerance)
            << "Mismatch in dbias at index " << i;
    }
}

} // namespace

// =========================
// FP32 plan execution tests
// =========================

TEST(TestGpuRMSNormFwdPlanFp32, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runFwdPlanExecuteVsCpuRef<float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, rmsnorm::getTolerance<float>());
}

TEST(TestGpuRMSNormFwdPlanFp32, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runFwdPlanExecuteVsCpuRef<float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, rmsnorm::getTolerance<float>());
}

TEST(TestGpuRMSNormFwdPlanFp32, ExecutePlanWithDoubleEpsilon)
{
    SKIP_IF_NO_DEVICES();

    runFwdPlanExecuteVsCpuRef<float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, rmsnorm::getTolerance<float>(), DataType::DOUBLE);
}

TEST(TestGpuRMSNormBwdPlanFp32, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<float, float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, rmsnorm::getTolerance<float>());
}

TEST(TestGpuRMSNormBwdPlanFp32, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<float, float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, rmsnorm::getTolerance<float>());
}

// =========================
// FP16 plan execution tests
// =========================

TEST(TestGpuRMSNormFwdPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runFwdPlanExecuteVsCpuRef<half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, rmsnorm::getTolerance<half>());
}

TEST(TestGpuRMSNormFwdPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runFwdPlanExecuteVsCpuRef<half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, rmsnorm::getTolerance<half>());
}

TEST(TestGpuRMSNormBwdPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<half, half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, rmsnorm::getTolerance<half>());
}

TEST(TestGpuRMSNormBwdPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<half, half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, rmsnorm::getTolerance<half>());
}

// =========================
// BFP16 plan execution tests
// =========================

TEST(TestGpuRMSNormFwdPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runFwdPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, rmsnorm::getTolerance<bfloat16>());
}

TEST(TestGpuRMSNormFwdPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runFwdPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, rmsnorm::getTolerance<bfloat16>());
}

TEST(TestGpuRMSNormBwdPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, rmsnorm::getTolerance<bfloat16>());
}

TEST(TestGpuRMSNormBwdPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, rmsnorm::getTolerance<bfloat16>());
}
