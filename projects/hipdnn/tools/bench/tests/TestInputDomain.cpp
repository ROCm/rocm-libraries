// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_bench/InputDomain.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;
using Values = std::map<int64_t, std::vector<int64_t>>;

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
void addNode(fb::GraphT& graph, TAttributes attributes)
{
    auto node = std::make_unique<fb::NodeT>();
    node->attributes.Set(std::move(attributes));
    graph.nodes.push_back(std::move(node));
}

template <typename TAttributes>
hipdnn_bench::DeclaredContents declared(fb::GraphT& graph, TAttributes attributes)
{
    addNode(graph, std::move(attributes));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::CreateGraph(builder, &graph));
    return hipdnn_bench::declaredContents(builder.GetBufferPointer(), builder.GetSize());
}

/// Q [2, 4, 3, 16], K [kBlocks, 2, kPage, 16], V [vBlocks, 2, vPage, 16], O: uids 1-4.
fb::GraphT
    attentionTensors(int64_t kBlocks = 2, int64_t kPage = 7, int64_t vBlocks = 2, int64_t vPage = 7)
{
    fb::GraphT graph;
    addTensor(graph, 1, {2, 4, 3, 16}, fb::DataType::HALF);
    addTensor(graph, 2, {kBlocks, 2, kPage, 16}, fb::DataType::HALF);
    addTensor(graph, 3, {vBlocks, 2, vPage, 16}, fb::DataType::HALF);
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

std::vector<int64_t> ramp(int64_t count, int64_t step = 1)
{
    std::vector<int64_t> values(static_cast<size_t>(count));
    for(size_t index = 0; index < values.size(); ++index)
    {
        values[index] = static_cast<int64_t>(index) * step;
    }
    return values;
}

/// Host memory standing in for the bench's device buffers.
struct HostBuffers
{
    std::map<int64_t, std::vector<uint8_t>> memory;

    void* add(const hipdnn_bench::TensorRequirement& tensor)
    {
        auto& block = memory[tensor.uid];
        block.assign(static_cast<size_t>(tensor.bytes), 0);
        return block.data();
    }

    static bool upload(const hipdnn_bench::TensorRequirement& /*tensor*/,
                       void* pointer,
                       const std::vector<uint8_t>& image)
    {
        std::memcpy(pointer, image.data(), image.size());
        return true;
    }
};

hipdnn_bench::TensorRequirement
    requirement(int64_t uid, int64_t bytes, hipdnn_frontend::DataType type, bool produced = false)
{
    hipdnn_bench::TensorRequirement tensor;
    tensor.uid = uid;
    tensor.name = "t" + std::to_string(uid);
    tensor.bytes = bytes;
    tensor.dataType = type;
    tensor.produced = produced;
    return tensor;
}

std::vector<int32_t> int32s(const std::vector<uint8_t>& bytes)
{
    std::vector<int32_t> values(bytes.size() / sizeof(int32_t));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(int32_t));
    return values;
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
// problem is every query and key of every batch. The block mask is UINT8, one bit per tile
// (SEM-025), so every byte is 0xFF: a value of 1 would keep one tile column in eight.
TEST(TestInputDomain, AttentionLengthsAndMaskDeclareTheWholeProblem)
{
    auto graph = attentionTensors();
    addTensor(graph, 5, {2, 1, 1, 1});
    addTensor(graph, 6, {2, 1, 1, 1});
    addTensor(graph, 7, {3}, fb::DataType::INT64);
    addTensor(graph, 8, {3});
    addTensor(graph, 9, {2, 4, 1, 2}, fb::DataType::UINT8);
    auto sdpa = attention();
    sdpa.seq_len_q_tensor_uid = 5;
    sdpa.seq_len_kv_tensor_uid = 6;
    sdpa.cu_seq_len_q_tensor_uid = 7;
    sdpa.cu_seq_len_kv_tensor_uid = 8;
    sdpa.block_mask_tensor_uid = 9;
    sdpa.padding_mask = true;

    const auto contents = declared(graph, sdpa);
    EXPECT_EQ(contents.values,
              (Values{{5, {3, 3}},
                      {6, {7, 7}},
                      {7, {0, 3, 6}},
                      {8, {0, 7, 14}},
                      {9, std::vector<int64_t>(16, 0xFF)}}));
    EXPECT_TRUE(contents.refused.empty());
}

// A mask of another type has no declared layout: a run reading it fails.
TEST(TestInputDomain, ABlockMaskThatIsNotUint8IsRefused)
{
    auto graph = attentionTensors();
    addTensor(graph, 9, {2, 4, 1, 1}, fb::DataType::BOOLEAN);
    auto sdpa = attention();
    sdpa.block_mask_tensor_uid = 9;

    const auto contents = declared(graph, sdpa);
    EXPECT_EQ(contents.values.count(9), 0U);
    EXPECT_EQ(contents.refused.count(9), 1U);
}

// The SDPA backward declares every query and key, as the forward does.
TEST(TestInputDomain, AttentionBackwardLengthsDeclareTheWholeProblem)
{
    auto graph = attentionTensors();
    addTensor(graph, 5, {2, 1, 1, 1});
    addTensor(graph, 6, {2, 1, 1, 1});
    fb::SdpaBackwardAttributesT backward;
    backward.q_tensor_uid = 1;
    backward.k_tensor_uid = 2;
    backward.v_tensor_uid = 3;
    backward.o_tensor_uid = 4;
    backward.seq_len_q_tensor_uid = 5;
    backward.seq_len_kv_tensor_uid = 6;

    const auto contents = declared(graph, backward);
    EXPECT_EQ(contents.values, (Values{{5, {3, 3}}, {6, {7, 7}}}));
    EXPECT_TRUE(contents.refused.empty());
}

// K and V hold 6 pages of 4 keys; each of 2 sequences reads 3 of them.
TEST(TestInputDomain, APagedCacheGivesEachSequenceItsOwnPagesAndDeclaresItsKeys)
{
    const auto paged = [](int64_t blocks, std::optional<int32_t> maxKeys) {
        auto graph = attentionTensors(blocks, 4, blocks, 4);
        addTensor(graph, 6, {2, 1, 1, 1});
        addTensor(graph, 8, {3});
        addTensor(graph, 10, {2, 1, 3, 1});
        addTensor(graph, 11, {2, 1, 3, 1});
        auto sdpa = attention();
        sdpa.seq_len_kv_tensor_uid = 6;
        sdpa.cu_seq_len_kv_tensor_uid = 8;
        sdpa.page_table_k_tensor_uid = 10;
        sdpa.page_table_v_tensor_uid = 11;
        if(maxKeys)
        {
            sdpa.max_seq_len_kv = *maxKeys;
        }
        return declared(graph, sdpa);
    };

    const auto pages = ramp(6);
    const auto full = paged(6, std::nullopt);
    EXPECT_EQ(full.values, (Values{{6, {12, 12}}, {8, {0, 12, 24}}, {10, pages}, {11, pages}}));
    EXPECT_TRUE(full.refused.empty());
    // A stated maximum is the declared keys.
    const auto stated = paged(6, 10);
    EXPECT_EQ(stated.values.at(6), (std::vector<int64_t>{10, 10}));
    EXPECT_EQ(stated.values.at(8), (std::vector<int64_t>{0, 10, 20}));
    // A cache too small for one page per entry cannot give each sequence its own: refused.
    const auto crowded = paged(5, std::nullopt);
    EXPECT_EQ(crowded.values.count(10), 0U);
    EXPECT_EQ(crowded.refused.count(10), 1U);
    EXPECT_EQ(crowded.refused.count(11), 1U);
}

// The keys are the work model's (keyExtent): the smaller paged capacity, an unpaged side's
// own extent, and never more than a paged side holds.
TEST(TestInputDomain, PagedKeysAreTheOnesEverySideHolds)
{
    // K: 3 pages of 4 (12 keys); V: 5 pages of 2 (10 keys). Each side gets its own table.
    {
        auto graph = attentionTensors(6, 4, 10, 2);
        addTensor(graph, 6, {2, 1, 1, 1});
        addTensor(graph, 10, {2, 1, 3, 1});
        addTensor(graph, 11, {2, 1, 5, 1});
        auto sdpa = attention();
        sdpa.seq_len_kv_tensor_uid = 6;
        sdpa.page_table_k_tensor_uid = 10;
        sdpa.page_table_v_tensor_uid = 11;
        const auto contents = declared(graph, sdpa);
        EXPECT_EQ(contents.values, (Values{{6, {10, 10}}, {10, ramp(6)}, {11, ramp(10)}}));
    }
    // V alone paged, 2 pages of 4: an unpaged K of 8 keys per sequence fits.
    const auto vOnly = [](int64_t keys) {
        auto graph = attentionTensors(2, keys, 4, 4);
        addTensor(graph, 6, {2, 1, 1, 1});
        addTensor(graph, 11, {2, 1, 2, 1});
        auto sdpa = attention();
        sdpa.seq_len_kv_tensor_uid = 6;
        sdpa.page_table_v_tensor_uid = 11;
        return declared(graph, sdpa);
    };
    EXPECT_EQ(vOnly(8).values, (Values{{6, {8, 8}}, {11, ramp(4)}}));
    // K's 9 keys do not fit V's 8: no length is declared that V cannot hold.
    const auto overflow = vOnly(9);
    EXPECT_EQ(overflow.values.count(6), 0U);
    EXPECT_EQ(overflow.refused.count(6), 1U);
}

// Offsets count multiples of the multiplier (a token here): batch i starts i batch strides in.
TEST(TestInputDomain, RaggedOffsetsPlaceEachBatchWhereTheDenseLayoutDoes)
{
    const auto ragged = [](int64_t outputBatchStride, int64_t qMultiplier) {
        fb::GraphT graph;
        // THD: a token is 4 heads x 16 contiguous elements, a batch 3 tokens of Q or 7 of K.
        addTensor(graph, 1, {2, 4, 3, 16}, fb::DataType::HALF, {192, 16, 64, 1});
        addTensor(graph, 2, {2, 2, 7, 16}, fb::DataType::HALF, {224, 16, 32, 1});
        addTensor(graph, 3, {2, 2, 7, 16}, fb::DataType::HALF, {224, 16, 32, 1});
        addTensor(graph, 4, {2, 4, 3, 16}, fb::DataType::HALF, {outputBatchStride, 16, 64, 1});
        addTensor(graph, 20, {3});
        addTensor(graph, 21, {3});
        for(const auto& [uid, offsets, multiplier] : {std::tuple{1, 20, qMultiplier},
                                                      std::tuple{4, 20, qMultiplier},
                                                      std::tuple{2, 21, int64_t{32}},
                                                      std::tuple{3, 21, int64_t{32}}})
        {
            graph.tensors[static_cast<size_t>(uid - 1)]->ragged_offset_tensor_uid = offsets;
            graph.tensors[static_cast<size_t>(uid - 1)]->ragged_offset_multiplier = multiplier;
        }
        return declared(graph, attention());
    };

    const auto tokens = ragged(192, 64);
    EXPECT_EQ(tokens.values, (Values{{20, {0, 3, 6}}, {21, {0, 7, 14}}}));
    EXPECT_TRUE(tokens.refused.empty());
    // The default multiplier counts elements.
    EXPECT_EQ(ragged(192, 1).values.at(20), (std::vector<int64_t>{0, 192, 384}));
    // Q and O share offsets but disagree on where batch 1 starts: refused, so the run fails.
    const auto disagreeing = ragged(256, 64);
    EXPECT_EQ(disagreeing.values, (Values{{21, {0, 7, 14}}}));
    EXPECT_EQ(disagreeing.refused.count(20), 1U);
    // A batch stride the multiplier does not divide has no offsets in its units.
    const auto indivisible = ragged(192, 100);
    EXPECT_EQ(indivisible.values.count(20), 0U);
    EXPECT_EQ(indivisible.refused.count(20), 1U);
}

// A refused uid stays refused when a later user agrees with an earlier one.
TEST(TestInputDomain, ARefusedOperandStaysRefused)
{
    fb::GraphT graph;
    addTensor(graph, 1, {2, 4, 3, 16}, fb::DataType::HALF, {192, 16, 64, 1});
    addTensor(graph, 4, {2, 4, 3, 16}, fb::DataType::HALF, {256, 16, 64, 1});
    addTensor(graph, 5, {2, 4, 3, 16}, fb::DataType::HALF, {192, 16, 64, 1});
    addTensor(graph, 20, {3});
    for(const size_t index : {0U, 1U, 2U})
    {
        graph.tensors[index]->ragged_offset_tensor_uid = 20;
        graph.tensors[index]->ragged_offset_multiplier = 64;
    }
    const auto contents = declared(graph, fb::PointwiseAttributesT{});
    EXPECT_EQ(contents.values.count(20), 0U);
    EXPECT_EQ(contents.refused.count(20), 1U);
}

// A gapped operand gets each value at its stride offset, zeros between.
TEST(TestInputDomain, AGappedOperandGetsItsValuesAtTheirStrideOffsets)
{
    auto graph = attentionTensors();
    addTensor(graph, 6, {2, 1, 1, 1}, fb::DataType::INT32, {4, 1, 1, 1});
    auto sdpa = attention();
    sdpa.seq_len_kv_tensor_uid = 6;

    EXPECT_EQ(declared(graph, sdpa).values, (Values{{6, {7, 0, 0, 0, 7}}}));
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
        return declared(graph, moe).values;
    };

    EXPECT_EQ(routed(fb::MoeGroupedMatmulMode::GATHER), (Values{{3, {0, 16, 32, 48}}}));
    EXPECT_EQ(routed(fb::MoeGroupedMatmulMode::NONE), (Values{{3, {0, 8, 16, 24}}}));

    fb::GraphT graph;
    addTensor(graph, 1, {1, 30, 64}, fb::DataType::HALF);
    addTensor(graph, 2, {1, 30, 128}, fb::DataType::HALF);
    addTensor(graph, 3, {4});
    addTensor(graph, 4, {4, 64, 128}, fb::DataType::HALF);
    fb::MoeGroupedMatmulBwdAttributesT backward;
    backward.token_tensor_uid = 1;
    backward.doutput_tensor_uid = 2;
    backward.first_token_offset_tensor_uid = 3;
    backward.dweight_tensor_uid = 4;
    EXPECT_EQ(declared(graph, backward).values, (Values{{3, {0, 7, 15, 22}}}));
}

// GDN and KDA count every token, so every token is in some sequence: n x T / N.
TEST(TestInputDomain, DeltaRuleSequencesCoverEveryToken)
{
    const auto sequences = [](auto attributes) {
        fb::GraphT graph;
        addTensor(graph, 1, {10, 2, 16}, fb::DataType::HALF);
        addTensor(graph, 6, {4});
        addTensor(graph, 7, {10, 2, 16}, fb::DataType::HALF);
        attributes.q_tensor_uid = 1;
        attributes.cu_seqlens_tensor_uid = 6;
        attributes.o_tensor_uid = 7;
        return declared(graph, attributes).values;
    };

    EXPECT_EQ(sequences(fb::GdnAttributesT{}), (Values{{6, {0, 3, 6, 10}}}));
    EXPECT_EQ(sequences(fb::KdaAttributesT{}), (Values{{6, {0, 3, 6, 10}}}));
}

// The bench's allocation writes the declared inputs, fails on a refused one, and the
// validation fill leaves the declared ones as written.
TEST(TestInputDomain, EveryRunReadsTheDeclaredContents)
{
    using hipdnn_frontend::DataType;
    hipdnn_bench::VariantPackPlan plan;
    plan.tensors = {requirement(1, 8, DataType::INT32),
                    requirement(2, 16, DataType::FLOAT),
                    requirement(3, 8, DataType::INT32, true)};
    hipdnn_bench::DeclaredContents contents;
    contents.values = {{1, {7, 7}}, {3, {5, 5}}};

    HostBuffers buffers;
    std::unordered_map<int64_t, void*> pack;
    ASSERT_TRUE(hipdnn_bench::allocateVariantPack(plan, buffers, pack, contents).is_good());
    EXPECT_EQ(int32s(buffers.memory.at(1)), (std::vector<int32_t>{7, 7}));
    // A result keeps its zeros.
    EXPECT_EQ(int32s(buffers.memory.at(3)), (std::vector<int32_t>{0, 0}));

    std::map<int64_t, std::vector<uint8_t>> fills;
    ASSERT_TRUE(
        hipdnn_bench::fillGraphInputs(plan, pack, buffers, 42, {}, contents, fills).is_good());
    EXPECT_EQ(int32s(buffers.memory.at(1)), (std::vector<int32_t>{7, 7}));
    EXPECT_NE(buffers.memory.at(2), std::vector<uint8_t>(16, 0));

    // Declared contents the buffer cannot hold, and a refused input, fail the run.
    contents.values[1] = {7, 7, 7};
    HostBuffers fresh;
    pack.clear();
    EXPECT_TRUE(hipdnn_bench::allocateVariantPack(plan, fresh, pack, contents).is_bad());
    contents.values.erase(1);
    contents.refused[1] = "why";
    pack.clear();
    const auto refused = hipdnn_bench::allocateVariantPack(plan, fresh, pack, contents);
    ASSERT_TRUE(refused.is_bad());
    EXPECT_NE(refused.get_message().find("'t1': why"), std::string::npos);
}
