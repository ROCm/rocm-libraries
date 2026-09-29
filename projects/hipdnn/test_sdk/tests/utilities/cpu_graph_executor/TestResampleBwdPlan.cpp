// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <gtest/gtest.h>

#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleBwdPlan.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_test_sdk::utilities;

namespace
{

template <typename Type>
void fillSequential(Tensor<Type>& tensor)
{
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        data[i] = static_cast<Type>(i + 1);
    }
    tensor.memory().markHostModified();
}

} // namespace

TEST(TestResampleBwdPlan, ExecuteMatchesCpuReference)
{
    auto builder = createValidResampleBwdGraph(true, ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attributes = *node.attributes_as_ResampleBwdAttributes();

    Tensor<float> planDy({1, 1, 2, 2});
    Tensor<float> planDx({1, 1, 4, 4});
    Tensor<float> directDy({1, 1, 2, 2});
    Tensor<float> directDx({1, 1, 4, 4});
    fillSequential(planDy);
    fillSequential(directDy);

    ResampleBwdParams params(attributes,
                             *graph.getTensorMap().at(attributes.dy_tensor_uid()),
                             *graph.getTensorMap().at(attributes.dx_tensor_uid()));
    ResampleBwdPlan<float, float, float, int32_t> plan(std::move(params));

    const std::unordered_map<int64_t, void*> variantPack{{1, planDy.memory().hostData()},
                                                         {2, planDx.memory().hostData()}};
    plan.execute(variantPack);

    CpuFpReferenceResampleBwd::backward<float, float, float, int32_t>(
        directDy,
        directDx,
        {0, 0},
        {2, 2},
        {2, 2},
        ResampleMode::AVGPOOL_EXCLUDE_PADDING,
        PaddingMode::ZERO_PAD);

    const CpuFpReferenceValidation<float> validator(0.0f, 0.0f);
    EXPECT_TRUE(validator.allClose(directDx, planDx));
}

TEST(TestResampleBwdPlan, ExecuteReadsIndexTensor)
{
    auto builder = createValidResampleBwdGraph(true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attributes = *node.attributes_as_ResampleBwdAttributes();

    Tensor<float> dy({1, 1, 2, 2});
    Tensor<float> dx({1, 1, 4, 4});
    Tensor<int32_t> index({1, 1, 2, 2});
    fillSequential(dy);

    // Maxpool indices with a 2x2 window on a 4x4 input of linear elements
    index.memory().hostData()[0] = 5;
    index.memory().hostData()[1] = 7;
    index.memory().hostData()[2] = 13;
    index.memory().hostData()[3] = 15;
    index.memory().markHostModified();

    const auto* indexAttributes = graph.getTensorMap().at(attributes.index_tensor_uid().value());
    ResampleBwdParams params(attributes,
                             *graph.getTensorMap().at(attributes.dy_tensor_uid()),
                             *graph.getTensorMap().at(attributes.dx_tensor_uid()),
                             indexAttributes);
    ResampleBwdPlan<float, float, float, int32_t> plan(std::move(params));

    const std::unordered_map<int64_t, void*> variantPack{
        {1, dy.memory().hostData()}, {2, dx.memory().hostData()}, {3, index.memory().hostData()}};
    plan.execute(variantPack);

    // Verify that the dx tensor has been populated correctly
    // since, for maxpool, the dx values should be equal to the
    // corresponding dy values at the indices specified in the
    // index tensor
    EXPECT_EQ(dx.memory().hostData()[5], dy.memory().hostData()[0]);
    EXPECT_EQ(dx.memory().hostData()[7], dy.memory().hostData()[1]);
    EXPECT_EQ(dx.memory().hostData()[13], dy.memory().hostData()[2]);
    EXPECT_EQ(dx.memory().hostData()[15], dy.memory().hostData()[3]);
}

TEST(TestResampleBwdPlanBuilder, PlanConstruction)
{
    auto builder = createValidResampleBwdGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const ResampleBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        planBuilder;

    auto builtPlan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<ResampleBwdPlan<float, float, float, int32_t>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestResampleBwdPlanBuilder, IsApplicableRejectsTypeMismatches)
{
    auto builder = createValidResampleBwdGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    // Dy type mismatch
    const ResampleBwdPlanBuilder<DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        badDyBuilder;
    EXPECT_FALSE(badDyBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Dx type mismatch
    const ResampleBwdPlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::UNSET>
        badDxBuilder;
    EXPECT_FALSE(badDxBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Compute type mismatch
    const ResampleBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::UNSET>
        badComputeBuilder;
    EXPECT_FALSE(
        badComputeBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestResampleBwdGraphBuilder, PreservesIndexTensorPresence)
{
    auto noIndexBuilder = createValidResampleBwdGraph(false);
    const GraphWrapper noIndexGraph(noIndexBuilder.GetBufferPointer(), noIndexBuilder.GetSize());
    const auto& noIndexAttr = *noIndexGraph.getNode(0).attributes_as_ResampleBwdAttributes();
    EXPECT_FALSE(noIndexAttr.index_tensor_uid().has_value());

    auto indexBuilder = createValidResampleBwdGraph(true);
    const GraphWrapper indexGraph(indexBuilder.GetBufferPointer(), indexBuilder.GetSize());
    const auto& indexAttr = *indexGraph.getNode(0).attributes_as_ResampleBwdAttributes();
    ASSERT_TRUE(indexAttr.index_tensor_uid().has_value());
    EXPECT_EQ(indexAttr.index_tensor_uid().value(), 3);
}

TEST(TestResampleBwdPlanBuilder, IsApplicableIndexTensorPresence)
{
    auto builder = createValidResampleBwdGraph(false, ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        noIndexBuilder;
    const ResampleBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32>
        indexBuilder;

    EXPECT_TRUE(noIndexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    EXPECT_FALSE(indexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    auto indexBuilderFbb = createValidResampleBwdGraph(true, ResampleMode::MAXPOOL);
    const GraphWrapper indexGraph(indexBuilderFbb.GetBufferPointer(), indexBuilderFbb.GetSize());
    EXPECT_FALSE(noIndexBuilder.isApplicable(indexGraph.getNode(0), indexGraph.getTensorMap()));
    EXPECT_TRUE(indexBuilder.isApplicable(indexGraph.getNode(0), indexGraph.getTensorMap()));

    auto tensorMapCopy = graph.getTensorMap();
    tensorMapCopy.erase(1);
    EXPECT_FALSE(noIndexBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestResampleBwdPlanBuilder, RejectsUnsupportedResampleModes)
{
    const ResampleBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        planBuilder;

    for(const auto mode : {ResampleMode::NOT_SET, static_cast<ResampleMode>(127)})
    {
        auto builder = createValidResampleBwdGraph(true, mode);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    }
}

TEST(TestResampleBwdPlanBuilder, BuildNodePlan)
{
    auto builder = createValidResampleBwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        planBuilder;
    EXPECT_NO_THROW(planBuilder.buildNodePlan(graph, graph.getNode(0)));

    auto reductionBuilder = createValidReductionGraph();
    const GraphWrapper reductionGraph(reductionBuilder.GetBufferPointer(),
                                      reductionBuilder.GetSize());
    EXPECT_THROW(planBuilder.buildNodePlan(reductionGraph, reductionGraph.getNode(0)),
                 std::runtime_error);
}
