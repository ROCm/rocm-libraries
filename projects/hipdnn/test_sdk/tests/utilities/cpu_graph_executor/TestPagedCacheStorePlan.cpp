// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PagedCacheStorePlan.hpp>

#include <unordered_map>
#include <vector>

using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

template <typename T>
void initTensorValues(TensorAttributesT& tensorAttr,
                      DataType dataType,
                      const Tensor<T>& tensor,
                      int64_t uid)
{
    tensorAttr.data_type = dataType;
    tensorAttr.dims = tensor.dims();
    tensorAttr.strides = tensor.strides();
    tensorAttr.uid = uid;
}

template <typename T>
Tensor<T> createTensor(const std::vector<int64_t>& dims)
{
    return Tensor<T>(dims, generateStrides(dims));
}

using FloatPlanBuilder = PagedCacheStorePlanBuilder<DataType::FLOAT,
                                                    DataType::INT32,
                                                    DataType::FLOAT,
                                                    DataType::FLOAT>;

} // namespace

// Executes through the variant pack with a scale tensor: x [2, 1, 2] slots {1, -1}, scale 2,
// into container [1, 1, 2, 2] prefilled with 5.
TEST(TestPagedCacheStorePlan, ExecutePlanWithScale)
{
    auto x = createTensor<float>({2, 1, 2});
    auto* xData = x.memory().hostData();
    xData[0] = 1.0F;
    xData[1] = 2.0F;
    xData[2] = 3.0F;
    xData[3] = 4.0F;
    auto slots = createTensor<int32_t>({2});
    slots.memory().hostData()[0] = 1;
    slots.memory().hostData()[1] = -1;
    auto container = createTensor<float>({1, 1, 2, 2});
    container.fillTensorWithValue(5.0F);
    auto scale = createTensor<float>({1});
    scale.memory().hostData()[0] = 2.0F;

    PagedCacheStoreParams params;
    initTensorValues(params.xTensor, DataType::FLOAT, x, 1);
    initTensorValues(params.slotMappingTensor, DataType::INT32, slots, 2);
    initTensorValues(params.containerTensor, DataType::FLOAT, container, 3);
    params.scaleTensor = TensorAttributesT{};
    initTensorValues(*params.scaleTensor, DataType::FLOAT, scale, 4);

    PagedCacheStorePlan<float, int32_t, float, float> patient(std::move(params));
    EXPECT_EQ(patient.getOutputTensorIds(), std::vector<int64_t>{3});

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = x.memory().hostData();
    variantPack[2] = slots.memory().hostData();
    variantPack[3] = container.memory().hostData();
    variantPack[4] = scale.memory().hostData();

    patient.execute(variantPack);

    const auto* out = container.memory().hostData();
    EXPECT_EQ(out[0], 5.0F);
    EXPECT_EQ(out[1], 5.0F);
    EXPECT_EQ(out[2], 2.0F);
    EXPECT_EQ(out[3], 4.0F);
}

TEST(TestPagedCacheStorePlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidPagedCacheStoreGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const FloatPlanBuilder floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Mismatched container data type
    const PagedCacheStorePlanBuilder<DataType::FLOAT,
                                     DataType::INT32,
                                     DataType::FP8_E4M3,
                                     DataType::FLOAT>
        fp8ContainerBuilder;
    EXPECT_FALSE(fp8ContainerBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Mismatched slot mapping type
    const PagedCacheStorePlanBuilder<DataType::FLOAT,
                                     DataType::INT64,
                                     DataType::FLOAT,
                                     DataType::FLOAT>
        int64Builder;
    EXPECT_FALSE(int64Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing scale tensor (uid 4) in the tensor map
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(4);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // Incompatible node type
    auto matmulBuilder = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherWrap(matmulBuilder.GetBufferPointer(), matmulBuilder.GetSize());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(otherWrap.getNode(0), otherWrap.getTensorMap()));
}

TEST(TestPagedCacheStorePlanBuilder, BuildNodePlan)
{
    const FloatPlanBuilder patient;

    auto builder = hipdnn_test_sdk::utilities::createValidPagedCacheStoreGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));
    EXPECT_NE((dynamic_cast<PagedCacheStorePlan<float, int32_t, float, float>*>(builtPlan.get())),
              nullptr);

    auto otherBuilder = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_THROW(patient.buildNodePlan(otherWrap, otherWrap.getNode(0)), std::runtime_error);
}
