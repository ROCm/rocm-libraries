// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <tuple>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "gpu-ref/BatchnormTestCase.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuBatchnormBwdPlan.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;
using gpu_batchnorm_ref_test::assertAllClose;

namespace
{

constexpr int64_t SCALE_UID = 4;
constexpr int64_t X_UID = 1;
constexpr int64_t DY_UID = 2;
constexpr int64_t DX_UID = 3;
constexpr int64_t DSCALE_UID = 5;
constexpr int64_t DBIAS_UID = 6;
constexpr int64_t MEAN_UID = 7;
constexpr int64_t INV_VARIANCE_UID = 8;

using FloatPlanBuilder = GpuBatchnormBwdPlanBuilder<DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT>;

template <typename Dy, typename X, typename Dx>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& dims,
                            const TensorLayout& layout,
                            bool useSavedStats)
{
    const auto strides = generateStrides(dims, layout.strideOrder);
    auto builder = createValidBatchnormBwdGraph(
        strides, dims, useSavedStats, nativeTypeToDataType<X>(), DataType::FLOAT, DataType::FLOAT);
    auto graph = UnPackGraph(builder.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->uid == DY_UID)
        {
            tensor->data_type = nativeTypeToDataType<Dy>();
        }
        else if(tensor->uid == DX_UID)
        {
            tensor->data_type = nativeTypeToDataType<Dx>();
        }
    }
    builder.Clear();
    builder.Finish(Graph::Pack(builder, graph.get()));
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpuExecutor;
    ASSERT_TRUE(gpuExecutor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));

    const auto affineDims = getDerivedShape(dims);
    const auto affineStrides = generateStrides(affineDims, layout.strideOrder);
    Tensor<Dy> dy(dims, strides);
    Tensor<X> x(dims, strides);
    Tensor<float> scale(affineDims, affineStrides);
    Tensor<float> mean(affineDims, affineStrides);
    Tensor<float> invVariance(affineDims, affineStrides);
    Tensor<Dx> cpuDx(dims, strides);
    Tensor<Dx> gpuDx(dims, strides);
    Tensor<float> cpuDscale(affineDims, affineStrides);
    Tensor<float> gpuDscale(affineDims, affineStrides);
    Tensor<float> cpuDbias(affineDims, affineStrides);
    Tensor<float> gpuDbias(affineDims, affineStrides);

    const auto seed = getGlobalTestSeed();
    dy.fillWithRandomValues(static_cast<Dy>(-1.0f), static_cast<Dy>(1.0f), seed);
    x.fillWithRandomValues(static_cast<X>(-1.0f), static_cast<X>(1.0f), seed + 1);
    scale.fillWithRandomValues(-1.0f, 1.0f, seed + 2);
    mean.fillWithRandomValues(-1.0f, 1.0f, seed + 3);
    invVariance.fillWithRandomValues(0.25f, 2.0f, seed + 4);

    std::unordered_map<int64_t, void*> gpuVariantPack{{DY_UID, dy.rawDeviceData()},
                                                      {X_UID, x.rawDeviceData()},
                                                      {SCALE_UID, scale.rawDeviceData()},
                                                      {DX_UID, gpuDx.rawDeviceData()},
                                                      {DSCALE_UID, gpuDscale.rawDeviceData()},
                                                      {DBIAS_UID, gpuDbias.rawDeviceData()}};
    std::unordered_map<int64_t, void*> cpuVariantPack{{DY_UID, dy.rawHostData()},
                                                      {X_UID, x.rawHostData()},
                                                      {SCALE_UID, scale.rawHostData()},
                                                      {DX_UID, cpuDx.rawHostData()},
                                                      {DSCALE_UID, cpuDscale.rawHostData()},
                                                      {DBIAS_UID, cpuDbias.rawHostData()}};
    if(useSavedStats)
    {
        gpuVariantPack.emplace(MEAN_UID, mean.rawDeviceData());
        gpuVariantPack.emplace(INV_VARIANCE_UID, invVariance.rawDeviceData());
        cpuVariantPack.emplace(MEAN_UID, mean.rawHostData());
        cpuVariantPack.emplace(INV_VARIANCE_UID, invVariance.rawHostData());
    }

    gpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), gpuVariantPack);
    gpuDx.markDeviceModified();
    gpuDscale.markDeviceModified();
    gpuDbias.markDeviceModified();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), cpuVariantPack);
    cpuDx.markHostModified();
    cpuDscale.markHostModified();
    cpuDbias.markHostModified();

    {
        SCOPED_TRACE("dx");
        const auto tolerance = batchnorm::getToleranceBackward<Dx>();
        // Different FP32 reduction orders can straddle a low-precision output rounding boundary.
        const auto relativeTolerance
            = std::max(tolerance, static_cast<float>(std::numeric_limits<Dx>::epsilon()));
        const hipdnn_gpu_ref::GpuFpReferenceValidation<Dx> validator(tolerance, relativeTolerance);
        const bool matches = validator.allClose(cpuDx, gpuDx);
        if(!matches)
        {
            printTensorDiff<Dx>(std::cerr, "dx", cpuDx, gpuDx, tolerance, relativeTolerance);
        }
        EXPECT_TRUE(matches);
    }
    {
        SCOPED_TRACE("dscale");
        assertAllClose(cpuDscale, gpuDscale, batchnorm::getToleranceBackward<float>());
    }
    {
        SCOPED_TRACE("dbias");
        assertAllClose(cpuDbias, gpuDbias, batchnorm::getToleranceBackward<float>());
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

class TestGpuBatchnormBwdPlanRequiredTensor : public testing::TestWithParam<int64_t>
{
};

TEST_P(TestGpuBatchnormBwdPlanRequiredTensor, MissingTensorIsNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1}, {2, 3, 4, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    auto tensorMap = graph.getTensorMap();
    tensorMap.erase(GetParam());

    EXPECT_FALSE(FloatPlanBuilder().isApplicable(graph.getNode(0), tensorMap));
}

INSTANTIATE_TEST_SUITE_P(
    AllOperands,
    TestGpuBatchnormBwdPlanRequiredTensor,
    testing::Values(
        X_UID, DY_UID, DX_UID, SCALE_UID, DSCALE_UID, DBIAS_UID, MEAN_UID, INV_VARIANCE_UID));

TEST_P(TestGpuBatchnormBwdPlanRequiredTensor, RaggedTensorIsNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1}, {2, 3, 4, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper denseGraph(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(FloatPlanBuilder().isApplicable(denseGraph.getNode(0), denseGraph.getTensorMap()));

    auto graph = UnPackGraph(builder.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->uid == GetParam())
        {
            tensor->ragged_offset_tensor_uid = X_UID;
        }
    }
    builder.Clear();
    builder.Finish(Graph::Pack(builder, graph.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper raggedGraph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(
        FloatPlanBuilder().isApplicable(raggedGraph.getNode(0), raggedGraph.getTensorMap()));
}

TEST_P(TestGpuBatchnormBwdPlanRequiredTensor, RuntimePassByValueTensorIsNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1}, {2, 3, 4, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper deviceGraph(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(
        FloatPlanBuilder().isApplicable(deviceGraph.getNode(0), deviceGraph.getTensorMap()));

    auto graph = UnPackGraph(builder.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->uid == GetParam())
        {
            tensor->is_runtime_pass_by_value = true;
        }
    }
    builder.Clear();
    builder.Finish(Graph::Pack(builder, graph.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper passByValueGraph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(FloatPlanBuilder().isApplicable(passByValueGraph.getNode(0),
                                                 passByValueGraph.getTensorMap()));
}

TEST_P(TestGpuBatchnormBwdPlanRequiredTensor, MissingTensorThrowsOnBuild)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1},
                                                {2, 3, 4, 4},
                                                true,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                {},
                                                GetParam());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(FloatPlanBuilder().buildNodePlan(graph, graph.getNode(0)), std::out_of_range);
}

class TestGpuBatchnormBwdPlanRequiredTensorType : public testing::TestWithParam<int64_t>
{
};

TEST_P(TestGpuBatchnormBwdPlanRequiredTensorType, MismatchedTypeIsNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph(
        {48, 16, 4, 1}, {2, 3, 4, 4}, true, DataType::FLOAT, DataType::FLOAT, DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto halfBuilder = createValidBatchnormBwdGraph(
        {48, 16, 4, 1}, {2, 3, 4, 4}, true, DataType::HALF, DataType::HALF, DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper halfGraph(
        halfBuilder.GetBufferPointer(), halfBuilder.GetSize());
    auto tensorMap = graph.getTensorMap();
    tensorMap[GetParam()] = halfGraph.getTensorMap().at(GetParam());

    const GpuBatchnormBwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::HALF,
                                     DataType::FLOAT,
                                     DataType::FLOAT>
        planBuilder;
    ASSERT_TRUE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    EXPECT_FALSE(planBuilder.isApplicable(graph.getNode(0), tensorMap));
}

INSTANTIATE_TEST_SUITE_P(AllRequiredOperands,
                         TestGpuBatchnormBwdPlanRequiredTensorType,
                         testing::Values(X_UID, DY_UID, DX_UID, SCALE_UID, DSCALE_UID, DBIAS_UID));

TEST(TestGpuBatchnormBwdPlanBuilder, OptionalStatisticsTypeMismatchIsNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph(
        {48, 16, 4, 1}, {2, 3, 4, 4}, true, DataType::FLOAT, DataType::FLOAT, DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(FloatPlanBuilder().isApplicable(graph.getNode(0), graph.getTensorMap()));
}

class TestGpuBatchnormBwdPlanSavedStatistic : public testing::TestWithParam<int64_t>
{
};

TEST_P(TestGpuBatchnormBwdPlanSavedStatistic, IncompleteSavedStatisticsAreNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1}, {2, 3, 4, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper completeGraph(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(
        FloatPlanBuilder().isApplicable(completeGraph.getNode(0), completeGraph.getTensorMap()));

    auto graph = UnPackGraph(builder.GetBufferPointer());
    auto* attributes = graph->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(attributes, nullptr);
    if(GetParam() == MEAN_UID)
    {
        attributes->mean_tensor_uid = flatbuffers::nullopt;
    }
    else
    {
        attributes->inv_variance_tensor_uid = flatbuffers::nullopt;
    }
    builder.Clear();
    builder.Finish(Graph::Pack(builder, graph.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper incompleteGraph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(FloatPlanBuilder().isApplicable(incompleteGraph.getNode(0),
                                                 incompleteGraph.getTensorMap()));
}

INSTANTIATE_TEST_SUITE_P(BothStatistics,
                         TestGpuBatchnormBwdPlanSavedStatistic,
                         testing::Values(MEAN_UID, INV_VARIANCE_UID));

TEST(TestGpuBatchnormBwdPlanBuilder, ComputeTypeMismatchIsNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1}, {2, 3, 4, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const GpuBatchnormBwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::DOUBLE>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormBwdPlanBuilder, WrongAttributesTypeIsRejectedAndThrowsOnBuild)
{
    auto builder = createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const FloatPlanBuilder planBuilder;

    EXPECT_FALSE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    EXPECT_THROW(planBuilder.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuBatchnormBwdPlanBuilder, PeerStatisticsAreNotApplicable)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1},
                                                {2, 3, 4, 4},
                                                true,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                {MEAN_UID});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(FloatPlanBuilder().isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormBwdPlanBuilder, UnsupportedMetadataValidationIsDeferredToOperation)
{
    const FloatPlanBuilder planBuilder;

    auto rankBuilder = createValidBatchnormBwdGraph({3, 1}, {2, 3});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper rankGraph(
        rankBuilder.GetBufferPointer(), rankBuilder.GetSize());
    EXPECT_TRUE(planBuilder.isApplicable(rankGraph.getNode(0), rankGraph.getTensorMap()));

    auto unpackedBuilder = createValidBatchnormBwdGraph({24, 8, 1}, {2, 3, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper unpackedGraph(
        unpackedBuilder.GetBufferPointer(), unpackedBuilder.GetSize());
    EXPECT_TRUE(planBuilder.isApplicable(unpackedGraph.getNode(0), unpackedGraph.getTensorMap()));

    auto layoutBuilder = createValidBatchnormBwdGraph({60, 5, 15, 1}, {2, 3, 4, 5});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper layoutGraph(
        layoutBuilder.GetBufferPointer(), layoutBuilder.GetSize());
    EXPECT_TRUE(planBuilder.isApplicable(layoutGraph.getNode(0), layoutGraph.getTensorMap()));
}

TEST(TestGpuBatchnormBwdPlanBuilder, MismatchedShapeValidationIsDeferredToOperation)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1}, {2, 3, 4, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const FloatPlanBuilder planBuilder;

    auto tensorMap = graph.getTensorMap();
    tensorMap[DX_UID] = tensorMap.at(SCALE_UID);
    EXPECT_TRUE(planBuilder.isApplicable(graph.getNode(0), tensorMap));

    tensorMap = graph.getTensorMap();
    tensorMap[DY_UID] = tensorMap.at(SCALE_UID);
    EXPECT_TRUE(planBuilder.isApplicable(graph.getNode(0), tensorMap));

    tensorMap = graph.getTensorMap();
    tensorMap[SCALE_UID] = tensorMap.at(X_UID);
    EXPECT_TRUE(planBuilder.isApplicable(graph.getNode(0), tensorMap));
}

TEST(TestGpuBatchnormBwdPlanBuilder, InconsistentOperandLayoutValidationIsDeferredToOperation)
{
    const std::vector<int64_t> dims{2, 3, 4, 5};
    auto nchwBuilder = createValidBatchnormBwdGraph({60, 20, 5, 1}, dims);
    auto nhwcBuilder = createValidBatchnormBwdGraph({60, 1, 15, 3}, dims);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper nchwGraph(
        nchwBuilder.GetBufferPointer(), nchwBuilder.GetSize());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper nhwcGraph(
        nhwcBuilder.GetBufferPointer(), nhwcBuilder.GetSize());
    auto tensorMap = nchwGraph.getTensorMap();
    tensorMap[DY_UID] = nhwcGraph.getTensorMap().at(DY_UID);

    EXPECT_TRUE(FloatPlanBuilder().isApplicable(nchwGraph.getNode(0), tensorMap));
}

TEST(TestGpuBatchnormBwdPlan, MissingVariantPackTensorsThrow)
{
    auto builder = createValidBatchnormBwdGraph({48, 16, 4, 1}, {2, 3, 4, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    auto plan = FloatPlanBuilder().buildNodePlan(graph, graph.getNode(0));
    std::unordered_map<int64_t, void*> completeVariantPack;
    for(int64_t uid = X_UID; uid <= INV_VARIANCE_UID; ++uid)
    {
        completeVariantPack.emplace(uid, reinterpret_cast<void*>(1));
    }

    for(int64_t missingUid = X_UID; missingUid <= INV_VARIANCE_UID; ++missingUid)
    {
        auto variantPack = completeVariantPack;
        variantPack.erase(missingUid);
        EXPECT_THROW(plan->execute(variantPack), std::out_of_range) << "uid " << missingUid;
    }
}

TEST(TestGpuBatchnormBwdPlan, ExecutorIsApplicable)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({12, 4, 2, 1}, {2, 3, 2, 2});
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;

    EXPECT_TRUE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

template <typename Types>
class TestGpuBatchnormBwdPlanExecute : public testing::Test
{
};

using RegisteredBatchnormBwdTypes = testing::Types<std::tuple<float, float, float>,
                                                   std::tuple<half, half, half>,
                                                   std::tuple<bfloat16, bfloat16, bfloat16>,
                                                   std::tuple<half, half, float>,
                                                   std::tuple<bfloat16, bfloat16, float>,
                                                   std::tuple<float, bfloat16, bfloat16>,
                                                   std::tuple<float, half, half>>;
TYPED_TEST_SUITE(TestGpuBatchnormBwdPlanExecute, RegisteredBatchnormBwdTypes, );

TYPED_TEST(TestGpuBatchnormBwdPlanExecute, ExecutorMatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    using Dy = std::tuple_element_t<0, TypeParam>;
    using X = std::tuple_element_t<1, TypeParam>;
    using Dx = std::tuple_element_t<2, TypeParam>;
    const std::vector<std::vector<int64_t>> shapes{{2, 3, 4, 5}, {4, 5, 17, 19}, {3, 2, 5, 7, 11}};
    for(const auto& dims : shapes)
    {
        SCOPED_TRACE(testing::PrintToString(dims));
        const auto layouts
            = dims.size() == 5 ? std::vector<TensorLayout>{TensorLayout::NCDHW, TensorLayout::NDHWC}
                               : std::vector<TensorLayout>{TensorLayout::NCHW, TensorLayout::NHWC};
        for(const auto& layout : layouts)
        {
            SCOPED_TRACE(testing::PrintToString(layout));
            for(const bool useSavedStats : {false, true})
            {
                SCOPED_TRACE(useSavedStats);
                runPlanExecuteVsCpuRef<Dy, X, Dx>(dims, layout, useSavedStats);
            }
        }
    }
}
