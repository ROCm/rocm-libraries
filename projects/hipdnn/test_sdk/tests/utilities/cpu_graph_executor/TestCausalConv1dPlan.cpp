// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/CausalConv1dPlan.hpp>

#include <unordered_map>
#include <vector>

using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

template <typename T>
TensorAttributesT tensorAttributes(DataType dataType, const Tensor<T>& tensor, int64_t uid)
{
    TensorAttributesT tensorAttr;
    tensorAttr.data_type = dataType;
    tensorAttr.dims = tensor.dims();
    tensorAttr.strides = tensor.strides();
    tensorAttr.uid = uid;
    return tensorAttr;
}

template <typename T>
Tensor<T> makeConvTensor(const std::vector<int64_t>& dims, const std::vector<T>& values)
{
    Tensor<T> tensor(dims, generateStrides(dims));
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = values[i];
    }
    return tensor;
}

using PoolPlanBuilder
    = CausalConv1dPlanBuilder<DataType::FLOAT, DataType::INT32, DataType::INT32, DataType::FLOAT>;

} // namespace

// Executes through the variant pack with final_state bound to initial_state's buffer.
// C = 1, W = 3, cu_seqlens [0, 1, 3], x [1, 2, 3], weight [1, 10, 100], state pool rows
// {[5, 6], [7, 8], [9, 9]} addressed by state_indices [2, 0]:
//   sequence 0 (row 2): xs [9, 9, 1]       -> y 199,      row 2 becomes [9, 1]
//   sequence 1 (row 0): xs [5, 6, 2, 3]    -> y 265, 326, row 0 becomes [2, 3]
// Row 1 is not addressed and keeps [7, 8].
TEST(TestCausalConv1dPlan, ExecutePlanUpdatesStatePoolInPlace)
{
    auto x = makeConvTensor<float>({3, 1}, {1.0F, 2.0F, 3.0F});
    auto weight = makeConvTensor<float>({1, 3}, {1.0F, 10.0F, 100.0F});
    auto cuSeqlens = makeConvTensor<int32_t>({3}, {0, 1, 3});
    auto y = makeConvTensor<float>({3, 1}, {0.0F, 0.0F, 0.0F});
    auto statePool = makeConvTensor<float>({3, 1, 2}, {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 9.0F});
    auto stateIndices = makeConvTensor<int32_t>({2}, {2, 0});

    CausalConv1dParams params;
    params.xTensor = tensorAttributes(DataType::FLOAT, x, 1);
    params.weightTensor = tensorAttributes(DataType::FLOAT, weight, 2);
    params.cuSeqlensTensor = tensorAttributes(DataType::INT32, cuSeqlens, 3);
    params.yTensor = tensorAttributes(DataType::FLOAT, y, 4);
    params.initialStateTensor = tensorAttributes(DataType::FLOAT, statePool, 5);
    params.stateIndicesTensor = tensorAttributes(DataType::INT32, stateIndices, 6);
    params.finalStateTensor = tensorAttributes(DataType::FLOAT, statePool, 7);

    CausalConv1dPlan<float, int32_t, int32_t, float> patient(std::move(params));
    EXPECT_EQ(patient.getOutputTensorIds(), (std::vector<int64_t>{4, 7}));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = x.memory().hostData();
    variantPack[2] = weight.memory().hostData();
    variantPack[3] = cuSeqlens.memory().hostData();
    variantPack[4] = y.memory().hostData();
    variantPack[5] = statePool.memory().hostData();
    variantPack[6] = stateIndices.memory().hostData();
    variantPack[7] = statePool.memory().hostData();

    patient.execute(variantPack);

    const auto* yOut = y.memory().hostData();
    EXPECT_EQ(yOut[0], 199.0F);
    EXPECT_EQ(yOut[1], 265.0F);
    EXPECT_EQ(yOut[2], 326.0F);

    const std::vector<float> expectedPool = {2.0F, 3.0F, 7.0F, 8.0F, 9.0F, 1.0F};
    const auto* poolOut = statePool.memory().hostData();
    for(size_t i = 0; i < expectedPool.size(); ++i)
    {
        EXPECT_EQ(poolOut[i], expectedPool[i]) << "state pool index " << i;
    }
}

TEST(TestCausalConv1dPlan, OutputTensorIdsWithoutFinalState)
{
    CausalConv1dParams params;
    params.yTensor.uid = 4;
    const CausalConv1dPlan<float, int32_t, int32_t, float> patient(std::move(params));
    EXPECT_EQ(patient.getOutputTensorIds(), std::vector<int64_t>{4});
}

TEST(TestCausalConv1dPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidCausalConv1dGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const PoolPlanBuilder poolPlanBuilder;
    EXPECT_TRUE(poolPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Mismatched compute data type
    const CausalConv1dPlanBuilder<DataType::FLOAT, DataType::INT32, DataType::INT32, DataType::HALF>
        halfComputeBuilder;
    EXPECT_FALSE(halfComputeBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Mismatched cu_seqlens type
    const CausalConv1dPlanBuilder<DataType::FLOAT,
                                  DataType::INT64,
                                  DataType::INT32,
                                  DataType::FLOAT>
        int64SeqlensBuilder;
    EXPECT_FALSE(int64SeqlensBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // A builder without state_indices rejects a graph that has them
    const CausalConv1dPlanBuilder<DataType::FLOAT,
                                  DataType::INT32,
                                  DataType::UNSET,
                                  DataType::FLOAT>
        noIndicesBuilder;
    EXPECT_FALSE(noIndicesBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing final_state tensor (uid 8) in the tensor map
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(8);
    EXPECT_FALSE(poolPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));

    // A graph without state_indices
    auto plainBuilder = hipdnn_test_sdk::utilities::createValidCausalConv1dGraph(false);
    const GraphWrapper plainWrap(plainBuilder.GetBufferPointer(), plainBuilder.GetSize());
    EXPECT_TRUE(noIndicesBuilder.isApplicable(plainWrap.getNode(0), plainWrap.getTensorMap()));
    EXPECT_FALSE(poolPlanBuilder.isApplicable(plainWrap.getNode(0), plainWrap.getTensorMap()));

    // Incompatible node type
    auto otherGraph = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherWrap(otherGraph.GetBufferPointer(), otherGraph.GetSize());
    EXPECT_FALSE(poolPlanBuilder.isApplicable(otherWrap.getNode(0), otherWrap.getTensorMap()));
}

TEST(TestCausalConv1dPlanBuilder, BuildNodePlan)
{
    const PoolPlanBuilder patient;

    auto builder = hipdnn_test_sdk::utilities::createValidCausalConv1dGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));
    auto* typedPlan
        = dynamic_cast<CausalConv1dPlan<float, int32_t, int32_t, float>*>(builtPlan.get());
    ASSERT_NE(typedPlan, nullptr);
    EXPECT_EQ(typedPlan->getOutputTensorIds(), (std::vector<int64_t>{4, 8}));

    auto otherBuilder = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper otherWrap(otherBuilder.GetBufferPointer(), otherBuilder.GetSize());
    EXPECT_THROW(patient.buildNodePlan(otherWrap, otherWrap.getNode(0)), std::runtime_error);
}
