// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleFwdPlan.hpp>

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

TEST(TestResampleFwdPlan, ExecuteMatchesCpuReference)
{
    auto builder = createValidResampleFwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attributes = *node.attributes_as_ResampleFwdAttributes();

    Tensor<float> planX({1, 1, 4, 4});
    Tensor<float> planY({1, 1, 2, 2});
    Tensor<float> directX({1, 1, 4, 4});
    Tensor<float> directY({1, 1, 2, 2});
    fillSequential(planX);
    fillSequential(directX);

    ResampleFwdParams params(attributes,
                             *graph.getTensorMap().at(attributes.x_tensor_uid()),
                             *graph.getTensorMap().at(attributes.y_tensor_uid()));
    ResampleFwdPlan<float, float, float> plan(std::move(params));

    const std::unordered_map<int64_t, void*> variantPack{{1, planX.memory().hostData()},
                                                         {2, planY.memory().hostData()}};
    plan.execute(variantPack);

    CpuFpReferenceResampleFwd::forward<float, float, float>(
        directX, directY, {0, 0}, {2, 2}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD);

    const CpuFpReferenceValidation<float> validator(0.0f, 0.0f);
    EXPECT_TRUE(validator.allClose(directY, planY));
}

TEST(TestResampleFwdPlan, ExecuteWritesGeneratedIndex)
{
    auto builder = createValidResampleFwdGraph(true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attributes = *node.attributes_as_ResampleFwdAttributes();

    Tensor<float> x({1, 1, 4, 4});
    Tensor<float> y({1, 1, 2, 2});
    Tensor<int32_t> index({1, 1, 2, 2});
    fillSequential(x);

    const auto* indexAttributes = graph.getTensorMap().at(attributes.index_tensor_uid().value());
    ResampleFwdParams params(attributes,
                             *graph.getTensorMap().at(attributes.x_tensor_uid()),
                             *graph.getTensorMap().at(attributes.y_tensor_uid()),
                             indexAttributes);
    ResampleFwdPlan<float, float, float, int32_t> plan(std::move(params));

    const std::unordered_map<int64_t, void*> variantPack{
        {1, x.memory().hostData()}, {2, y.memory().hostData()}, {3, index.memory().hostData()}};
    plan.execute(variantPack);

    EXPECT_EQ(index.memory().hostData()[0], 5);
    EXPECT_EQ(index.memory().hostData()[1], 7);
    EXPECT_EQ(index.memory().hostData()[2], 13);
    EXPECT_EQ(index.memory().hostData()[3], 15);
}

TEST(TestResampleFwdGraphBuilder, PreservesOptionalGenerateIndex)
{
    auto absentBuilder = createValidResampleFwdGraph();
    const GraphWrapper absentGraph(absentBuilder.GetBufferPointer(), absentBuilder.GetSize());
    const auto& absentAttributes = *absentGraph.getNode(0).attributes_as_ResampleFwdAttributes();
    EXPECT_FALSE(absentAttributes.generate_index().has_value());

    auto falseBuilder = createValidResampleFwdGraph(false);
    const GraphWrapper falseGraph(falseBuilder.GetBufferPointer(), falseBuilder.GetSize());
    const auto& falseAttributes = *falseGraph.getNode(0).attributes_as_ResampleFwdAttributes();
    ASSERT_TRUE(falseAttributes.generate_index().has_value());
    EXPECT_FALSE(falseAttributes.generate_index().value());

    auto trueBuilder = createValidResampleFwdGraph(true);
    const GraphWrapper trueGraph(trueBuilder.GetBufferPointer(), trueBuilder.GetSize());
    const auto& trueAttributes = *trueGraph.getNode(0).attributes_as_ResampleFwdAttributes();
    ASSERT_TRUE(trueAttributes.generate_index().has_value());
    EXPECT_TRUE(trueAttributes.generate_index().value());
}

TEST(TestResampleFwdPlanBuilder, IsApplicable)
{
    auto builder = createValidResampleFwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        noIndexBuilder;
    const ResampleFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32>
        indexBuilder;

    EXPECT_TRUE(noIndexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    EXPECT_FALSE(indexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    auto indexBuilderFbb = createValidResampleFwdGraph(true);
    const GraphWrapper indexGraph(indexBuilderFbb.GetBufferPointer(), indexBuilderFbb.GetSize());
    EXPECT_FALSE(noIndexBuilder.isApplicable(indexGraph.getNode(0), indexGraph.getTensorMap()));
    EXPECT_TRUE(indexBuilder.isApplicable(indexGraph.getNode(0), indexGraph.getTensorMap()));

    auto tensorMapCopy = graph.getTensorMap();
    tensorMapCopy.erase(1);
    EXPECT_FALSE(noIndexBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestResampleFwdPlanBuilder, RejectsUnsupportedModes)
{
    const ResampleFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        planBuilder;

    for(const auto mode : {ResampleMode::NOT_SET, static_cast<ResampleMode>(127)})
    {
        auto builder = createValidResampleFwdGraph(std::nullopt, mode);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    }
}

TEST(TestResampleFwdPlanBuilder, RejectsUnsupportedPaddingModes)
{
    const ResampleFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        planBuilder;

    for(const auto paddingMode : {PaddingMode::PADDING_NOT_SET, static_cast<PaddingMode>(127)})
    {
        auto builder
            = createValidResampleFwdGraph(std::nullopt, ResampleMode::MAXPOOL, paddingMode);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
    }
}

TEST(TestResampleFwdPlanBuilder, BuildNodePlan)
{
    auto builder = createValidResampleFwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET>
        planBuilder;
    EXPECT_NO_THROW(planBuilder.buildNodePlan(graph, graph.getNode(0)));

    auto reductionBuilder = createValidReductionGraph();
    const GraphWrapper reductionGraph(reductionBuilder.GetBufferPointer(),
                                      reductionBuilder.GetSize());
    EXPECT_THROW(planBuilder.buildNodePlan(reductionGraph, reductionGraph.getNode(0)),
                 std::runtime_error);
}
