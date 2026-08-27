// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "LayernormGraphUtils.hpp"
#include "LayernormTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormBpropPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestLayernormBpropPlan : public ::testing::Test
{
};

TEST_F(TestLayernormBpropPlan, ExecutePlan)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormBpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormBpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();

    EXPECT_TRUE(attributes.mean_tensor_uid().has_value());
    EXPECT_TRUE(attributes.inv_variance_tensor_uid().has_value());

    const auto& tensorMap = graphWrapper.getTensorMap();
    LayernormBpropParams params(*tensorMap.at(attributes.dy_tensor_uid()),
                                *tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.dx_tensor_uid()),
                                *tensorMap.at(attributes.dscale_tensor_uid()),
                                *tensorMap.at(attributes.dbias_tensor_uid()),
                                normalizedDimCount,
                                tensorMap.at(attributes.mean_tensor_uid().value()),
                                tensorMap.at(attributes.inv_variance_tensor_uid().value()),
                                attributes.epsilon_tensor_uid().has_value()
                                    ? tensorMap.at(attributes.epsilon_tensor_uid().value())
                                    : nullptr);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowDyTensor = createShallowTensor<float>(
        params.dyTensor, directTensorBundle.getTensor(attributes.dy_tensor_uid()).rawHostData());
    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowMeanTensor = createShallowTensor<float>(
        params.meanTensor.value(),
        directTensorBundle.getTensor(attributes.mean_tensor_uid().value()).rawHostData());
    auto shallowInvVarianceTensor = createShallowTensor<float>(
        params.invVarianceTensor.value(),
        directTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value()).rawHostData());
    auto shallowDxTensor = createShallowTensor<float>(
        params.dxTensor, directTensorBundle.getTensor(attributes.dx_tensor_uid()).rawHostData());
    auto shallowDscaleTensor = createShallowTensor<float>(
        params.dscaleTensor,
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()).rawHostData());
    auto shallowDbiasTensor = createShallowTensor<float>(
        params.dbiasTensor,
        directTensorBundle.getTensor(attributes.dbias_tensor_uid()).rawHostData());

    CpuFpReferenceLayernorm::bprop(*shallowDyTensor,
                                   *shallowXTensor,
                                   *shallowScaleTensor,
                                   *shallowDxTensor,
                                   *shallowDscaleTensor,
                                   *shallowDbiasTensor,
                                   hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
                                   shallowMeanTensor.get(),
                                   shallowInvVarianceTensor.get(),
                                   normalizedDimCount);

    LayernormBpropPlan<float, float, float, float, float> bpropPlan(std::move(params));
    bpropPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dx_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dx_tensor_uid())));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()),
        planTensorBundle.getTensor(attributes.dscale_tensor_uid())));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dbias_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dbias_tensor_uid())));
}

TEST_F(TestLayernormBpropPlan, ExecutePlanWithRuntimeEpsilonFromPack)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();

    // Pure runtime pass-by-value epsilon delivered through the variant pack.
    auto runtimeGraph = buildLayernormBpropGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 dims,
                                                 normalizedDimCount,
                                                 TensorLayout::NHWC,
                                                 /*meanInvVar=*/true,
                                                 /*onePadded=*/false,
                                                 /*runtimeEpsilon=*/true);
    auto [runtimeSerialized, runtimeSerErr] = runtimeGraph->to_binary();
    ASSERT_TRUE(runtimeSerErr.is_good()) << runtimeSerErr.get_message();
    const GraphWrapper runtimeWrapper(runtimeSerialized.data(), runtimeSerialized.size());
    const INodeWrapper& runtimeNode = runtimeWrapper.getNodeWrapper(0);
    LayernormBpropTensorBundle runtimeBundle(runtimeNode, runtimeWrapper.getTensorMap(), seed);

    const auto& runtimeAttrs
        = runtimeNode
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();
    ASSERT_TRUE(runtimeAttrs.epsilon_tensor_uid().has_value());
    const auto epsilonHostValue = static_cast<float>(LAYERNORM_DEFAULT_EPSILON);
    *static_cast<float*>(
        runtimeBundle.tensors[runtimeAttrs.epsilon_tensor_uid().value()]->rawHostData())
        = epsilonHostValue;

    const auto& runtimeTensorMap = runtimeWrapper.getTensorMap();
    LayernormBpropParams runtimeParams(
        *runtimeTensorMap.at(runtimeAttrs.dy_tensor_uid()),
        *runtimeTensorMap.at(runtimeAttrs.x_tensor_uid()),
        *runtimeTensorMap.at(runtimeAttrs.scale_tensor_uid()),
        *runtimeTensorMap.at(runtimeAttrs.dx_tensor_uid()),
        *runtimeTensorMap.at(runtimeAttrs.dscale_tensor_uid()),
        *runtimeTensorMap.at(runtimeAttrs.dbias_tensor_uid()),
        normalizedDimCount,
        runtimeTensorMap.at(runtimeAttrs.mean_tensor_uid().value()),
        runtimeTensorMap.at(runtimeAttrs.inv_variance_tensor_uid().value()),
        runtimeTensorMap.at(runtimeAttrs.epsilon_tensor_uid().value()));
    const std::unordered_map<int64_t, void*> runtimeVariantPack = runtimeBundle.toHostVariantPack();
    LayernormBpropPlan<float, float, float, float, float> runtimePlan(std::move(runtimeParams));
    runtimePlan.execute(runtimeVariantPack);

    // Baked-epsilon reference graph with the same seed and equal epsilon value.
    auto bakedGraph = buildLayernormBpropGraph(DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               dims,
                                               normalizedDimCount,
                                               TensorLayout::NHWC,
                                               /*meanInvVar=*/true);
    auto [bakedSerialized, bakedSerErr] = bakedGraph->to_binary();
    ASSERT_TRUE(bakedSerErr.is_good()) << bakedSerErr.get_message();
    const GraphWrapper bakedWrapper(bakedSerialized.data(), bakedSerialized.size());
    const INodeWrapper& bakedNode = bakedWrapper.getNodeWrapper(0);
    LayernormBpropTensorBundle bakedBundle(bakedNode, bakedWrapper.getTensorMap(), seed);

    const auto& bakedAttrs
        = bakedNode
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();
    const auto& bakedTensorMap = bakedWrapper.getTensorMap();
    LayernormBpropParams bakedParams(
        *bakedTensorMap.at(bakedAttrs.dy_tensor_uid()),
        *bakedTensorMap.at(bakedAttrs.x_tensor_uid()),
        *bakedTensorMap.at(bakedAttrs.scale_tensor_uid()),
        *bakedTensorMap.at(bakedAttrs.dx_tensor_uid()),
        *bakedTensorMap.at(bakedAttrs.dscale_tensor_uid()),
        *bakedTensorMap.at(bakedAttrs.dbias_tensor_uid()),
        normalizedDimCount,
        bakedTensorMap.at(bakedAttrs.mean_tensor_uid().value()),
        bakedTensorMap.at(bakedAttrs.inv_variance_tensor_uid().value()),
        bakedTensorMap.at(bakedAttrs.epsilon_tensor_uid().value()));
    const std::unordered_map<int64_t, void*> bakedVariantPack = bakedBundle.toHostVariantPack();
    LayernormBpropPlan<float, float, float, float, float> bakedPlan(std::move(bakedParams));
    bakedPlan.execute(bakedVariantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(bakedBundle.getTensor(bakedAttrs.dx_tensor_uid()),
                                        runtimeBundle.getTensor(runtimeAttrs.dx_tensor_uid())));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(bakedBundle.getTensor(bakedAttrs.dscale_tensor_uid()),
                                        runtimeBundle.getTensor(runtimeAttrs.dscale_tensor_uid())));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(bakedBundle.getTensor(bakedAttrs.dbias_tensor_uid()),
                                        runtimeBundle.getTensor(runtimeAttrs.dbias_tensor_uid())));
}

TEST_F(TestLayernormBpropPlan, ExecutePlanWithoutOptionals)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          false);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormBpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormBpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();

    EXPECT_FALSE(attributes.mean_tensor_uid().has_value());
    EXPECT_FALSE(attributes.inv_variance_tensor_uid().has_value());

    const auto& tensorMap = graphWrapper.getTensorMap();
    LayernormBpropParams params(*tensorMap.at(attributes.dy_tensor_uid()),
                                *tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.dx_tensor_uid()),
                                *tensorMap.at(attributes.dscale_tensor_uid()),
                                *tensorMap.at(attributes.dbias_tensor_uid()),
                                normalizedDimCount,
                                nullptr,
                                nullptr,
                                attributes.epsilon_tensor_uid().has_value()
                                    ? tensorMap.at(attributes.epsilon_tensor_uid().value())
                                    : nullptr);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowDyTensor = createShallowTensor<float>(
        params.dyTensor, directTensorBundle.getTensor(attributes.dy_tensor_uid()).rawHostData());
    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowDxTensor = createShallowTensor<float>(
        params.dxTensor, directTensorBundle.getTensor(attributes.dx_tensor_uid()).rawHostData());
    auto shallowDscaleTensor = createShallowTensor<float>(
        params.dscaleTensor,
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()).rawHostData());
    auto shallowDbiasTensor = createShallowTensor<float>(
        params.dbiasTensor,
        directTensorBundle.getTensor(attributes.dbias_tensor_uid()).rawHostData());

    CpuFpReferenceLayernorm::bprop(
        *shallowDyTensor,
        *shallowXTensor,
        *shallowScaleTensor,
        *shallowDxTensor,
        *shallowDscaleTensor,
        *shallowDbiasTensor,
        hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
        static_cast<const hipdnn_data_sdk::utilities::TensorBase<float>*>(nullptr),
        static_cast<const hipdnn_data_sdk::utilities::TensorBase<float>*>(nullptr),
        normalizedDimCount);

    LayernormBpropPlan<float, float, float, float, float> bpropPlan(std::move(params));
    bpropPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dx_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dx_tensor_uid())));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()),
        planTensorBundle.getTensor(attributes.dscale_tensor_uid())));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dbias_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dbias_tensor_uid())));
}

TEST_F(TestLayernormBpropPlan, ExecutePlanOnePaddedNormalizedDimCount2)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 2;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormBpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormBpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();

    EXPECT_TRUE(attributes.mean_tensor_uid().has_value());
    EXPECT_TRUE(attributes.inv_variance_tensor_uid().has_value());

    const auto& tensorMap = graphWrapper.getTensorMap();
    LayernormBpropParams params(*tensorMap.at(attributes.dy_tensor_uid()),
                                *tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.dx_tensor_uid()),
                                *tensorMap.at(attributes.dscale_tensor_uid()),
                                *tensorMap.at(attributes.dbias_tensor_uid()),
                                normalizedDimCount,
                                tensorMap.at(attributes.mean_tensor_uid().value()),
                                tensorMap.at(attributes.inv_variance_tensor_uid().value()),
                                attributes.epsilon_tensor_uid().has_value()
                                    ? tensorMap.at(attributes.epsilon_tensor_uid().value())
                                    : nullptr);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowDyTensor = createShallowTensor<float>(
        params.dyTensor, directTensorBundle.getTensor(attributes.dy_tensor_uid()).rawHostData());
    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowMeanTensor = createShallowTensor<float>(
        params.meanTensor.value(),
        directTensorBundle.getTensor(attributes.mean_tensor_uid().value()).rawHostData());
    auto shallowInvVarianceTensor = createShallowTensor<float>(
        params.invVarianceTensor.value(),
        directTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value()).rawHostData());
    auto shallowDxTensor = createShallowTensor<float>(
        params.dxTensor, directTensorBundle.getTensor(attributes.dx_tensor_uid()).rawHostData());
    auto shallowDscaleTensor = createShallowTensor<float>(
        params.dscaleTensor,
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()).rawHostData());
    auto shallowDbiasTensor = createShallowTensor<float>(
        params.dbiasTensor,
        directTensorBundle.getTensor(attributes.dbias_tensor_uid()).rawHostData());

    CpuFpReferenceLayernorm::bprop(*shallowDyTensor,
                                   *shallowXTensor,
                                   *shallowScaleTensor,
                                   *shallowDxTensor,
                                   *shallowDscaleTensor,
                                   *shallowDbiasTensor,
                                   hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
                                   shallowMeanTensor.get(),
                                   shallowInvVarianceTensor.get(),
                                   normalizedDimCount);

    LayernormBpropPlan<float, float, float, float, float> bpropPlan(std::move(params));
    bpropPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dx_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dx_tensor_uid())));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()),
        planTensorBundle.getTensor(attributes.dscale_tensor_uid())));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dbias_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dbias_tensor_uid())));
}

TEST_F(TestLayernormBpropPlan, ExecutePlanTrainingPhase)
{
    auto tolerance = layernorm::getTolerance<float>();
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormBpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormBpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    EXPECT_TRUE(attributes.mean_tensor_uid().has_value());
    EXPECT_TRUE(attributes.inv_variance_tensor_uid().has_value());

    LayernormBpropParams params(*tensorMap.at(attributes.dy_tensor_uid()),
                                *tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.dx_tensor_uid()),
                                *tensorMap.at(attributes.dscale_tensor_uid()),
                                *tensorMap.at(attributes.dbias_tensor_uid()),
                                normalizedDimCount,
                                tensorMap.at(attributes.mean_tensor_uid().value()),
                                tensorMap.at(attributes.inv_variance_tensor_uid().value()),
                                attributes.epsilon_tensor_uid().has_value()
                                    ? tensorMap.at(attributes.epsilon_tensor_uid().value())
                                    : nullptr);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowDyTensor = createShallowTensor<float>(
        params.dyTensor, directTensorBundle.getTensor(attributes.dy_tensor_uid()).rawHostData());
    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowMeanTensor = createShallowTensor<float>(
        params.meanTensor.value(),
        directTensorBundle.getTensor(attributes.mean_tensor_uid().value()).rawHostData());
    auto shallowInvVarianceTensor = createShallowTensor<float>(
        params.invVarianceTensor.value(),
        directTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value()).rawHostData());
    auto shallowDxTensor = createShallowTensor<float>(
        params.dxTensor, directTensorBundle.getTensor(attributes.dx_tensor_uid()).rawHostData());
    auto shallowDscaleTensor = createShallowTensor<float>(
        params.dscaleTensor,
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()).rawHostData());
    auto shallowDbiasTensor = createShallowTensor<float>(
        params.dbiasTensor,
        directTensorBundle.getTensor(attributes.dbias_tensor_uid()).rawHostData());

    CpuFpReferenceLayernorm::bprop(*shallowDyTensor,
                                   *shallowXTensor,
                                   *shallowScaleTensor,
                                   *shallowDxTensor,
                                   *shallowDscaleTensor,
                                   *shallowDbiasTensor,
                                   hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
                                   shallowMeanTensor.get(),
                                   shallowInvVarianceTensor.get(),
                                   normalizedDimCount);

    LayernormBpropPlan<float, float, float, float, float> bpropPlan(std::move(params));
    bpropPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dx_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dx_tensor_uid())));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()),
        planTensorBundle.getTensor(attributes.dscale_tensor_uid())));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.getTensor(attributes.dbias_tensor_uid()),
                                        planTensorBundle.getTensor(attributes.dbias_tensor_uid())));

    if(attributes.mean_tensor_uid().has_value())
    {
        EXPECT_TRUE(cpuRefOutputValidation.allClose(
            directTensorBundle.getTensor(attributes.mean_tensor_uid().value()),
            planTensorBundle.getTensor(attributes.mean_tensor_uid().value())));
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        EXPECT_TRUE(cpuRefOutputValidation.allClose(
            directTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value()),
            planTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value())));
    }
}

TEST_F(TestLayernormBpropPlan, ExecutePlanGetOutputTensorIds)
{
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const int64_t normalizedDimCount = 3;
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    LayernormBpropTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    LayernormBpropTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();

    EXPECT_TRUE(attributes.mean_tensor_uid().has_value());
    EXPECT_TRUE(attributes.inv_variance_tensor_uid().has_value());

    LayernormBpropParams params(*tensorMap.at(attributes.dy_tensor_uid()),
                                *tensorMap.at(attributes.x_tensor_uid()),
                                *tensorMap.at(attributes.scale_tensor_uid()),
                                *tensorMap.at(attributes.dx_tensor_uid()),
                                *tensorMap.at(attributes.dscale_tensor_uid()),
                                *tensorMap.at(attributes.dbias_tensor_uid()),
                                normalizedDimCount,
                                tensorMap.at(attributes.mean_tensor_uid().value()),
                                tensorMap.at(attributes.inv_variance_tensor_uid().value()),
                                attributes.epsilon_tensor_uid().has_value()
                                    ? tensorMap.at(attributes.epsilon_tensor_uid().value())
                                    : nullptr);

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowDyTensor = createShallowTensor<float>(
        params.dyTensor, directTensorBundle.getTensor(attributes.dy_tensor_uid()).rawHostData());
    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.getTensor(attributes.x_tensor_uid()).rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.getTensor(attributes.scale_tensor_uid()).rawHostData());
    auto shallowMeanTensor = createShallowTensor<float>(
        params.meanTensor.value(),
        directTensorBundle.getTensor(attributes.mean_tensor_uid().value()).rawHostData());
    auto shallowInvVarianceTensor = createShallowTensor<float>(
        params.invVarianceTensor.value(),
        directTensorBundle.getTensor(attributes.inv_variance_tensor_uid().value()).rawHostData());
    auto shallowDxTensor = createShallowTensor<float>(
        params.dxTensor, directTensorBundle.getTensor(attributes.dx_tensor_uid()).rawHostData());
    auto shallowDscaleTensor = createShallowTensor<float>(
        params.dscaleTensor,
        directTensorBundle.getTensor(attributes.dscale_tensor_uid()).rawHostData());
    auto shallowDbiasTensor = createShallowTensor<float>(
        params.dbiasTensor,
        directTensorBundle.getTensor(attributes.dbias_tensor_uid()).rawHostData());

    CpuFpReferenceLayernorm::bprop(*shallowDyTensor,
                                   *shallowXTensor,
                                   *shallowScaleTensor,
                                   *shallowDxTensor,
                                   *shallowDscaleTensor,
                                   *shallowDbiasTensor,
                                   hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON,
                                   shallowMeanTensor.get(),
                                   shallowInvVarianceTensor.get(),
                                   normalizedDimCount);

    const std::vector<int64_t> expectedIds
        = {params.dxTensor.uid, params.dscaleTensor.uid, params.dbiasTensor.uid};
    const LayernormBpropPlan<float, float, float, float, float> bpropPlan(std::move(params));
    EXPECT_EQ(bpropPlan.getOutputTensorIds(), expectedIds);
}

TEST(TestLayernormBpropPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<LayernormBpropPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestLayernormBpropPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(5);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}

TEST(TestLayernormBpropPlanBuilder, PlanConstructionTrainingPhase)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<LayernormBpropPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestLayernormBpropPlanBuilder, IsApplicableTrainingPhase)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        badMeanTypePlanBuilder;
    EXPECT_FALSE(
        badMeanTypePlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestLayernormBpropPlanBuilder, IsApplicableInvalidComputeData)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    auto graph = buildLayernormBpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        floatPlanBuilder;

    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestLayernormBpropPlanBuilder, IsApplicableAndBuildNodePlanInvalidAttributes)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const int64_t normalizedDimCount = 3;
    // Fprop instead of bprop to get wrong attributes
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          normalizedDimCount,
                                          TensorLayout::NHWC,
                                          true);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const LayernormBpropPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    EXPECT_THROW(floatPlanBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0)),
                 std::runtime_error);
}
