// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_bench/InputDomain.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
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

void addTensor(fb::GraphT& graph,
               int64_t uid,
               std::vector<int64_t> dims,
               fb::DataType type = fb::DataType::INT32,
               std::vector<int64_t> strides = {})
{
    auto tensor = std::make_unique<fb::TensorAttributesT>();
    tensor->uid = uid;
    tensor->data_type = type;
    tensor->dims = std::move(dims);
    tensor->strides = std::move(strides);
    graph.tensors.push_back(std::move(tensor));
}

template <typename TAttributes>
std::vector<uint8_t> serialized(fb::GraphT& graph, TAttributes attributes)
{
    auto node = std::make_unique<fb::NodeT>();
    node->attributes.Set(std::move(attributes));
    graph.nodes.push_back(std::move(node));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::CreateGraph(builder, &graph));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

hipdnn_bench::DeclaredContents declared(const std::vector<uint8_t>& bytes)
{
    return hipdnn_bench::declaredContents(bytes.data(), bytes.size());
}

/// Q [2, 4, 3, 16] and K, V [2, 2, 7, 16]: uids 1-4.
fb::GraphT attentionTensors(int64_t kvBlocks = 2, int64_t pageSize = 7)
{
    fb::GraphT graph;
    addTensor(graph, 1, {2, 4, 3, 16}, fb::DataType::HALF);
    addTensor(graph, 2, {kvBlocks, 2, pageSize, 16}, fb::DataType::HALF);
    addTensor(graph, 3, {kvBlocks, 2, pageSize, 16}, fb::DataType::HALF);
    addTensor(graph, 4, {2, 4, 3, 16}, fb::DataType::HALF);
    return graph;
}

fb::SdpaAttributesT attention()
{
    fb::SdpaAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    return sdpa;
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

// Zeroed timing buffers would run lengths of 0 and a mask keeping nothing; the declared
// problem is every query and key of every batch.
TEST(TestInputDomain, AttentionLengthsAndMaskDeclareTheWholeProblem)
{
    auto graph = attentionTensors();
    addTensor(graph, 5, {2, 1, 1, 1});
    addTensor(graph, 6, {2, 1, 1, 1});
    addTensor(graph, 7, {3}, fb::DataType::INT64);
    addTensor(graph, 8, {3});
    addTensor(graph, 9, {2, 4, 1, 1}, fb::DataType::BOOLEAN);
    auto sdpa = attention();
    sdpa.seq_len_q_tensor_uid = 5;
    sdpa.seq_len_kv_tensor_uid = 6;
    sdpa.cu_seq_len_q_tensor_uid = 7;
    sdpa.cu_seq_len_kv_tensor_uid = 8;
    sdpa.block_mask_tensor_uid = 9;
    sdpa.padding_mask = true;

    EXPECT_EQ(declared(serialized(graph, sdpa)),
              (hipdnn_bench::DeclaredContents{{5, {3, 3}},
                                              {6, {7, 7}},
                                              {7, {0, 3, 6}},
                                              {8, {0, 7, 14}},
                                              {9, std::vector<int64_t>(8, 1)}}));
}

// K and V hold 6 pages of 4 keys; each of 2 sequences reads 3 of them.
TEST(TestInputDomain, APagedCacheGivesEachSequenceItsOwnPagesAndDeclaresItsKeys)
{
    const auto paged = [](int64_t blocks, std::optional<int32_t> maxKeys) {
        auto graph = attentionTensors(blocks, 4);
        addTensor(graph, 6, {2, 1, 1, 1});
        addTensor(graph, 10, {2, 1, 3, 1});
        addTensor(graph, 11, {2, 1, 3, 1});
        auto sdpa = attention();
        sdpa.seq_len_kv_tensor_uid = 6;
        sdpa.page_table_k_tensor_uid = 10;
        sdpa.page_table_v_tensor_uid = 11;
        if(maxKeys)
        {
            sdpa.max_seq_len_kv = *maxKeys;
        }
        return declared(serialized(graph, sdpa));
    };

    const std::vector<int64_t> pages{0, 1, 2, 3, 4, 5};
    EXPECT_EQ(paged(6, std::nullopt),
              (hipdnn_bench::DeclaredContents{{6, {12, 12}}, {10, pages}, {11, pages}}));
    // A stated maximum is the declared keys.
    EXPECT_EQ(paged(6, 10).at(6), (std::vector<int64_t>{10, 10}));
    // A cache too small for one page per entry cannot give each sequence its own.
    const auto crowded = paged(5, std::nullopt);
    EXPECT_EQ(crowded.count(10), 0U);
    EXPECT_EQ(crowded.count(11), 0U);
}

// Offsets count multiples of the multiplier (a token here): batch i starts i batch strides in.
TEST(TestInputDomain, RaggedOffsetsPlaceEachBatchWhereTheDenseLayoutDoes)
{
    const auto ragged = [](int64_t outputBatchStride) {
        fb::GraphT graph;
        // THD: a token is 4 heads x 16 contiguous elements, a batch 3 tokens of Q or 7 of K.
        addTensor(graph, 1, {2, 4, 3, 16}, fb::DataType::HALF, {192, 16, 64, 1});
        addTensor(graph, 2, {2, 2, 7, 16}, fb::DataType::HALF, {224, 16, 32, 1});
        addTensor(graph, 3, {2, 2, 7, 16}, fb::DataType::HALF, {224, 16, 32, 1});
        addTensor(graph, 4, {2, 4, 3, 16}, fb::DataType::HALF, {outputBatchStride, 16, 64, 1});
        addTensor(graph, 20, {3});
        addTensor(graph, 21, {3});
        for(const auto& [uid, offsets, multiplier] : {std::tuple{1, 20, 64},
                                                      std::tuple{4, 20, 64},
                                                      std::tuple{2, 21, 32},
                                                      std::tuple{3, 21, 32}})
        {
            graph.tensors[static_cast<size_t>(uid - 1)]->ragged_offset_tensor_uid = offsets;
            graph.tensors[static_cast<size_t>(uid - 1)]->ragged_offset_multiplier = multiplier;
        }
        return declared(serialized(graph, attention()));
    };

    EXPECT_EQ(ragged(192), (hipdnn_bench::DeclaredContents{{20, {0, 3, 6}}, {21, {0, 7, 14}}}));
    // Q and O share offsets but disagree on where batch 1 starts: neither layout is declared.
    EXPECT_EQ(ragged(256), (hipdnn_bench::DeclaredContents{{21, {0, 7, 14}}}));
}

// Every declared row is routed: the token's rows, or the token index's when gathering.
TEST(TestInputDomain, MoeRoutingSpreadsEveryDeclaredRowOverTheGroups)
{
    const auto routed = [](fb::MoeGroupedMatmulMode mode) {
        fb::GraphT graph;
        addTensor(graph, 1, {1, 32, 64}, fb::DataType::HALF);
        addTensor(graph, 2, {4, 64, 128}, fb::DataType::HALF);
        addTensor(graph, 3, {4, 1, 1});
        addTensor(graph, 4, {1, 64, 1});
        addTensor(graph, 5, {1, 64, 128}, fb::DataType::HALF);
        fb::MoeGroupedMatmulAttributesT moe;
        moe.token_tensor_uid = 1;
        moe.weight_tensor_uid = 2;
        moe.first_token_offset_tensor_uid = 3;
        moe.token_index_tensor_uid = 4;
        moe.output_tensor_uid = 5;
        moe.mode = mode;
        moe.top_k = 2;
        return declared(serialized(graph, moe));
    };

    EXPECT_EQ(routed(fb::MoeGroupedMatmulMode::GATHER),
              (hipdnn_bench::DeclaredContents{{3, {0, 16, 32, 48}}}));
    EXPECT_EQ(routed(fb::MoeGroupedMatmulMode::NONE),
              (hipdnn_bench::DeclaredContents{{3, {0, 8, 16, 24}}}));
}
