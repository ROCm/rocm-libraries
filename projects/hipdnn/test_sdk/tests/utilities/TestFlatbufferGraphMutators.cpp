// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_test_sdk::utilities;

namespace
{

std::unique_ptr<TensorAttributesT> makeTensor(int64_t uid, const std::string& name)
{
    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = uid;
    tensor->name = name;
    tensor->data_type = DataType::FLOAT;
    tensor->dims = {4, 4};
    tensor->strides = {4, 1};
    return tensor;
}

flatbuffers::FlatBufferBuilder pack(const GraphT& graph)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, &graph));
    return builder;
}

// virtual, pass-by-value, constant, device, device: one tensor of each kind the mutators skip,
// followed by two device tensors so "first eligible" is distinguishable from "any eligible".
GraphT makeMixedTensorGraph()
{
    GraphT graph;
    graph.name = "mixed";

    auto virtualTensor = makeTensor(10, "virtual");
    virtualTensor->virtual_ = true;
    graph.tensors.push_back(std::move(virtualTensor));

    auto passByValue = makeTensor(11, "pass_by_value");
    passByValue->is_runtime_pass_by_value = true;
    graph.tensors.push_back(std::move(passByValue));

    auto constant = makeTensor(12, "constant");
    constant->value.Set(Float32Value(1.0f));
    graph.tensors.push_back(std::move(constant));

    graph.tensors.push_back(makeTensor(13, "first_device"));
    graph.tensors.push_back(makeTensor(14, "second_device"));
    return graph;
}

const TensorAttributesT& findTensor(const GraphT& graph, const std::string& name)
{
    for(const auto& tensor : graph.tensors)
    {
        if(tensor->name == name)
        {
            return *tensor;
        }
    }
    throw std::runtime_error("No tensor named " + name);
}

} // namespace

TEST(TestFlatbufferGraphMutators, RaggedTensorMarksFirstNonVirtualTensor)
{
    auto mixed = makeMixedTensorGraph();

    auto builder = makeGraphWithRaggedTensor(pack(mixed));
    const auto graph = UnPackGraph(builder.GetBufferPointer());

    ASSERT_EQ(graph->tensors.size(), mixed.tensors.size());
    EXPECT_FALSE(findTensor(*graph, "virtual").ragged_offset_tensor_uid.has_value());
    // Pass-by-value is still non-virtual, so it is the first eligible tensor.
    EXPECT_EQ(findTensor(*graph, "pass_by_value").ragged_offset_tensor_uid, 14);
    EXPECT_FALSE(findTensor(*graph, "constant").ragged_offset_tensor_uid.has_value());
    EXPECT_FALSE(findTensor(*graph, "first_device").ragged_offset_tensor_uid.has_value());
    EXPECT_FALSE(findTensor(*graph, "second_device").ragged_offset_tensor_uid.has_value());
}

TEST(TestFlatbufferGraphMutators, RaggedTensorThrowsWhenNoTensorIsNonVirtual)
{
    GraphT graph;
    auto virtualTensor = makeTensor(1, "virtual");
    virtualTensor->virtual_ = true;
    graph.tensors.push_back(std::move(virtualTensor));

    auto builder = pack(graph);
    EXPECT_THROW(makeGraphWithRaggedTensor(builder.GetBufferPointer()), std::runtime_error);
    EXPECT_THROW(makeGraphWithRaggedTensor(createEmptyValidGraph()), std::runtime_error);
}

TEST(TestFlatbufferGraphMutators, TensorAlignmentSetsOnlyFirstDeviceTensor)
{
    auto builder = pack(makeMixedTensorGraph());

    auto realigned = makeGraphWithTensorAlignment(builder.GetBufferPointer(), 8);
    const auto graph = UnPackGraph(realigned.GetBufferPointer());

    constexpr int64_t DEFAULT_ALIGNMENT = 16;
    EXPECT_EQ(findTensor(*graph, "virtual").alignment, DEFAULT_ALIGNMENT);
    EXPECT_EQ(findTensor(*graph, "pass_by_value").alignment, DEFAULT_ALIGNMENT);
    EXPECT_EQ(findTensor(*graph, "constant").alignment, DEFAULT_ALIGNMENT);
    EXPECT_EQ(findTensor(*graph, "first_device").alignment, 8);
    EXPECT_EQ(findTensor(*graph, "second_device").alignment, DEFAULT_ALIGNMENT);
}

TEST(TestFlatbufferGraphMutators, TensorAlignmentAppliesToBuiltGraph)
{
    auto realigned = makeGraphWithTensorAlignment(createValidMatmulGraph(), 32);
    const auto graph = UnPackGraph(realigned.GetBufferPointer());

    ASSERT_FALSE(graph->tensors.empty());
    EXPECT_EQ(graph->tensors.front()->alignment, 32);
}

TEST(TestFlatbufferGraphMutators, TensorAlignmentThrowsWithoutDeviceTensor)
{
    auto mixed = makeMixedTensorGraph();
    mixed.tensors.resize(3); // keep only the virtual, pass-by-value and constant tensors

    EXPECT_THROW(makeGraphWithTensorAlignment(pack(mixed), 8), std::runtime_error);
    EXPECT_THROW(makeGraphWithTensorAlignment(createEmptyValidGraph(), 8), std::runtime_error);
}
