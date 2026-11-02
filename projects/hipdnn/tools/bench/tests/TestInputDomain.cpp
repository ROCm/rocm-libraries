// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_bench/InputDomain.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

/// A serialized graph of pointwise nodes, each (mode, input uid, output uid).
std::vector<uint8_t>
    pointwiseGraph(const std::vector<std::tuple<fb::PointwiseMode, int64_t, int64_t>>& nodes)
{
    fb::GraphT graph;
    graph.name = "domain";
    for(const auto& [mode, input, output] : nodes)
    {
        for(const int64_t uid : {input, output})
        {
            auto tensor = std::make_unique<fb::TensorAttributesT>();
            tensor->uid = uid;
            tensor->data_type = fb::DataType::FLOAT;
            tensor->dims = {4};
            tensor->strides = {1};
            graph.tensors.push_back(std::move(tensor));
        }
        fb::PointwiseAttributesT attributes;
        attributes.operation = mode;
        attributes.in_0_tensor_uid = input;
        attributes.out_0_tensor_uid = output;
        auto node = std::make_unique<fb::NodeT>();
        node->attributes.Set(attributes);
        graph.nodes.push_back(std::move(node));
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::CreateGraph(builder, &graph));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

} // namespace

TEST(TestInputDomain, TheInputsOfLogSqrtAndRsqrtAreTheOnesFilledPositive)
{
    const auto bytes = pointwiseGraph({{fb::PointwiseMode::LOG, 1, 2},
                                       {fb::PointwiseMode::SQRT, 3, 4},
                                       {fb::PointwiseMode::RSQRT, 5, 6},
                                       {fb::PointwiseMode::EXP, 7, 8},
                                       {fb::PointwiseMode::TAN, 9, 10}});

    EXPECT_EQ(hipdnn_bench::positiveDomainReads(bytes.data(), bytes.size()),
              (std::set<int64_t>{1, 3, 5}));
}

TEST(TestInputDomain, AGraphWithoutThemNeedsNoPositiveInput)
{
    const auto bytes = pointwiseGraph({{fb::PointwiseMode::RELU_FWD, 1, 2}});

    EXPECT_TRUE(hipdnn_bench::positiveDomainReads(bytes.data(), bytes.size()).empty());
}
