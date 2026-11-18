// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/RaggedLayoutTestSupport.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaDensifyTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaFwdSupport.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::ragged_densify_test::tensorOf;

namespace
{

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t RAGGED_OFFSET_Q_UID = 10;
constexpr int64_t RAGGED_OFFSET_KV_UID = 11;
constexpr int64_t CU_SEQ_LEN_UID = 90;

// [B = 1, H = 2, S = 4, D = 8]. Token-major strides are {64, 8, 16, 1}.
const std::vector<int64_t> DIMS = raggedDims(1, 4, 2, 8);

// A fully ragged bf16 graph with the ragged_offset dropped from every primary not in `ragged`.
flatbuffers::DetachedBuffer makeGraph(const std::vector<int64_t>& ragged)
{
    const auto graph = createRaggedSdpaFwdGraph(Q_UID,
                                                K_UID,
                                                V_UID,
                                                O_UID,
                                                RAGGED_OFFSET_Q_UID,
                                                RAGGED_OFFSET_KV_UID,
                                                /*batch=*/1,
                                                DIMS,
                                                DIMS,
                                                DIMS,
                                                DIMS,
                                                DataType::BFLOAT16);
    auto graphT = std::unique_ptr<GraphT>(GetGraph(graph.GetBufferPointer())->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        const bool primary = tensor->uid == Q_UID || tensor->uid == K_UID || tensor->uid == V_UID
                             || tensor->uid == O_UID;
        if(primary && std::find(ragged.begin(), ragged.end(), tensor->uid) == ragged.end())
        {
            tensor->ragged_offset_tensor_uid = flatbuffers::nullopt;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

const SdpaAttributes& sdpaAttributes(const GraphWrapper& graphWrapper)
{
    return *graphWrapper.getNode(0).attributes_as_SdpaAttributes();
}

// A standalone ragged tensor with the given strides and ragged_offset_multiplier.
flatbuffers::FlatBufferBuilder makeRaggedTensor(const std::vector<int64_t>& dims,
                                                const std::vector<int64_t>& strides,
                                                int64_t multiplier)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateTensorAttributesDirect(builder,
                                                Q_UID,
                                                "Q",
                                                DataType::BFLOAT16,
                                                &strides,
                                                &dims,
                                                /*virtual_=*/false,
                                                TensorValue::NONE,
                                                /*value=*/0,
                                                /*is_runtime_pass_by_value=*/false,
                                                RAGGED_OFFSET_Q_UID,
                                                /*alignment=*/16,
                                                multiplier));
    return builder;
}

bool isSupportedLayout(const std::vector<int64_t>& strides, int64_t multiplier)
{
    const auto builder = makeRaggedTensor(DIMS, strides, multiplier);
    return isSupportedRaggedSdpaLayout(
        *flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()));
}

bool isSupportedLayout(const ragged_layout_test::RaggedLayoutCase& layout)
{
    const auto builder = makeRaggedTensor(layout.dims, layout.strides, layout.multiplier);
    return isSupportedRaggedSdpaLayout(
        *flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()));
}

constexpr int64_t STATS_UID = 5;
constexpr int64_t RAGGED_OFFSET_STATS_UID = 12;
constexpr int64_t SEQ_LEN_Q_UID = 30;
constexpr int64_t SEQ_LEN_KV_UID = 31;
const std::vector<int64_t> ROLE_UIDS = {Q_UID, K_UID, V_UID, O_UID, STATS_UID};

// A fully ragged bf16 graph with a ragged Stats. `edit` changes it before serialization.
std::vector<uint8_t> makeStatsGraph(const std::function<void(GraphT&)>& edit = {})
{
    RaggedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID;
    options.statsLayout = RaggedStatsLayout::PACKED;
    options.raggedOffsetStatsUid = RAGGED_OFFSET_STATS_UID;
    const auto graph = createRaggedSdpaFwdGraph(Q_UID,
                                                K_UID,
                                                V_UID,
                                                O_UID,
                                                RAGGED_OFFSET_Q_UID,
                                                RAGGED_OFFSET_KV_UID,
                                                /*batch=*/1,
                                                DIMS,
                                                DIMS,
                                                DIMS,
                                                DIMS,
                                                DataType::BFLOAT16,
                                                options);
    const std::vector<uint8_t> bytes(graph.GetBufferPointer(),
                                     graph.GetBufferPointer() + graph.GetSize());
    return edit ? ragged_densify_test::editGraph(bytes, edit) : bytes;
}

// Adds INT32 [batch, 1, 1, 1] seq_len_q/kv tensors and binds the node to them.
void addSeqLens(GraphT& graph, int64_t batch)
{
    for(const int64_t uid : {SEQ_LEN_Q_UID, SEQ_LEN_KV_UID})
    {
        auto tensor = std::make_unique<TensorAttributesT>();
        tensor->uid = uid;
        tensor->data_type = DataType::INT32;
        tensor->dims = {batch, 1, 1, 1};
        tensor->strides = {1, 1, 1, 1};
        graph.tensors.push_back(std::move(tensor));
    }
    auto* attrs = graph.nodes.front()->attributes.AsSdpaAttributes();
    attrs->seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
    attrs->seq_len_kv_tensor_uid = SEQ_LEN_KV_UID;
}

bool nodeAccepted(const std::vector<uint8_t>& graph)
{
    const GraphWrapper graphWrapper(graph.data(), graph.size());
    return isSupportedRaggedSdpaFwdNode(sdpaAttributes(graphWrapper), graphWrapper.getTensorMap());
}

} // namespace

TEST(TestRaggedSdpaFwdSupport, HasRaggedSdpaPrimaryForEachSingleRaggedPrimary)
{
    for(const auto uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        const auto graph = makeGraph({uid});
        const GraphWrapper graphWrapper(graph.data(), graph.size());
        EXPECT_TRUE(hasRaggedSdpaPrimary(sdpaAttributes(graphWrapper), graphWrapper.getTensorMap()))
            << "ragged uid " << uid;
    }
}

TEST(TestRaggedSdpaFwdSupport, HasRaggedSdpaPrimaryFalseForDenseNode)
{
    const auto graph = makeGraph({});
    const GraphWrapper graphWrapper(graph.data(), graph.size());
    EXPECT_FALSE(hasRaggedSdpaPrimary(sdpaAttributes(graphWrapper), graphWrapper.getTensorMap()));
}

// Only Q is ragged. Without Q in the map, or with a null entry for it, no ragged primary is seen.
TEST(TestRaggedSdpaFwdSupport, HasRaggedSdpaPrimaryIgnoresMissingAndNullUids)
{
    const auto graph = makeGraph({Q_UID});
    const GraphWrapper graphWrapper(graph.data(), graph.size());
    const auto& attributes = sdpaAttributes(graphWrapper);
    ASSERT_TRUE(hasRaggedSdpaPrimary(attributes, graphWrapper.getTensorMap()));

    auto tensorMap = graphWrapper.getTensorMap();
    tensorMap.erase(Q_UID);
    EXPECT_FALSE(hasRaggedSdpaPrimary(attributes, tensorMap));

    tensorMap[Q_UID] = nullptr;
    EXPECT_FALSE(hasRaggedSdpaPrimary(attributes, tensorMap));
}

TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutForTokenMajorStrides)
{
    EXPECT_TRUE(isSupportedLayout(raggedStrides(DIMS), 1));
}

// RFC-0014 requires ragged_offset_multiplier >= 1.
TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutFalseForZeroMultiplier)
{
    EXPECT_FALSE(isSupportedLayout(raggedStrides(DIMS), 0));
}

// Head-major (contiguous BHSD, L6) is a valid ragged layout.
TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutTrueForContiguousBhsdStrides)
{
    EXPECT_TRUE(isSupportedLayout(generateStrides(DIMS), 1));
    EXPECT_TRUE(isSupportedLayout(ragged_layout_test::raggedLayoutCase("L6")));
}

// Strides no origins and lengths could make non-aliasing are declined: a head stride below one
// head's feature span, a negative stride, a zero sequence stride. Layouts whose aliasing depends
// on the origins (L10, L11) are declined when the tensor is built (the container).
TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutFalseForAliasingStrides)
{
    EXPECT_FALSE(isSupportedLayout({64, 4, 16, 1}, 1)); // [1, 2, 4, 8]: H stride 4 < 8
    EXPECT_FALSE(isSupportedLayout({64, 8, -16, 1}, 1));
    EXPECT_FALSE(isSupportedLayout({64, 8, 0, 1}, 1));
}

// The outer of the head and feature strides must reach past the inner axis's whole span,
// (extent - 1) * stride + 1: equal is accepted, one short is declined. Both axis orders.
TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutSpanBoundary)
{
    const auto supported
        = [](const std::vector<int64_t>& dims, const std::vector<int64_t>& strides) {
              const auto builder = makeRaggedTensor(dims, strides, 1);
              return isSupportedRaggedSdpaLayout(
                  *flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()));
          };
    // [1, 2, 4, 8], head outer over a feature span of 8.
    EXPECT_TRUE(supported({1, 2, 4, 8}, {64, 8, 16, 1}));
    EXPECT_FALSE(supported({1, 2, 4, 8}, {64, 7, 16, 1}));
    // [1, 8, 4, 2], feature outer over a head span of 8.
    EXPECT_TRUE(supported({1, 8, 4, 2}, {64, 1, 16, 8}));
    EXPECT_FALSE(supported({1, 8, 4, 2}, {64, 1, 16, 7}));
}

// Every container-valid row of the layout table is accepted.
TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutTrueForContainerLayouts)
{
    for(const auto& layout : ragged_layout_test::raggedLayoutCases())
    {
        if(!layout.rejected)
        {
            EXPECT_TRUE(isSupportedLayout(layout)) << layout.id;
        }
    }
}

// The gate accepts every container layout for every role (Q, K, V, O, Stats), without and with the
// seq_len pair.
TEST(TestRaggedSdpaFwdSupport, GateAcceptsContainerLayoutsForEveryRole)
{
    for(const char* id : {"L2", "L5", "L6", "L7a", "L7c", "L7d", "L8", "L9"})
    {
        const auto& layout = ragged_layout_test::raggedLayoutCase(id);
        for(const int64_t role : ROLE_UIDS)
        {
            for(const bool seqLens : {false, true})
            {
                const auto graph = makeStatsGraph([&](GraphT& g) {
                    auto& tensor = tensorOf(g, role);
                    tensor.dims = layout.dims;
                    tensor.strides = layout.strides;
                    tensor.ragged_offset_multiplier = layout.multiplier;
                    if(seqLens)
                    {
                        addSeqLens(g, tensorOf(g, Q_UID).dims[0]);
                    }
                });
                EXPECT_TRUE(nodeAccepted(graph))
                    << id << " role " << role << (seqLens ? " with seq_len" : "");
            }
        }
    }
}

// seq_len_q/kv replace the old blanket decline: both (INT32, one per batch) are accepted, with or
// without padding_mask; one alone, a wrong type or count, or padding_mask without them is not.
TEST(TestRaggedSdpaFwdSupport, GateAppliesSdpaPaddingRule)
{
    EXPECT_TRUE(nodeAccepted(makeStatsGraph()));
    EXPECT_TRUE(nodeAccepted(makeStatsGraph([](GraphT& g) { addSeqLens(g, 1); })));
    EXPECT_TRUE(nodeAccepted(makeStatsGraph([](GraphT& g) {
        addSeqLens(g, 1);
        g.nodes.front()->attributes.AsSdpaAttributes()->padding_mask = true;
    })));

    EXPECT_FALSE(nodeAccepted(makeStatsGraph(
        [](GraphT& g) { g.nodes.front()->attributes.AsSdpaAttributes()->padding_mask = true; })));
    EXPECT_FALSE(nodeAccepted(makeStatsGraph([](GraphT& g) {
        addSeqLens(g, 1);
        g.nodes.front()->attributes.AsSdpaAttributes()->seq_len_kv_tensor_uid
            = flatbuffers::nullopt;
    })));
    EXPECT_FALSE(nodeAccepted(makeStatsGraph([](GraphT& g) {
        addSeqLens(g, 1);
        tensorOf(g, SEQ_LEN_Q_UID).data_type = DataType::INT64;
    })));
    EXPECT_FALSE(nodeAccepted(makeStatsGraph([](GraphT& g) { addSeqLens(g, 2); })));
}

// The ragged references have no cumulative-length input, so each cu_seq_len side is declined alone
// and together, with and without the seq_len pair that would otherwise be accepted.
TEST(TestRaggedSdpaFwdSupport, GateDeclinesCuSeqLenOnEachSide)
{
    using Edit = std::function<void(SdpaAttributesT&)>;
    const std::vector<Edit> edits
        = {[](SdpaAttributesT& a) { a.cu_seq_len_q_tensor_uid = CU_SEQ_LEN_UID; },
           [](SdpaAttributesT& a) { a.cu_seq_len_kv_tensor_uid = CU_SEQ_LEN_UID; },
           [](SdpaAttributesT& a) {
               a.cu_seq_len_q_tensor_uid = CU_SEQ_LEN_UID;
               a.cu_seq_len_kv_tensor_uid = CU_SEQ_LEN_UID;
           }};

    for(const bool withSeqLens : {false, true})
    {
        ASSERT_TRUE(nodeAccepted(makeStatsGraph([&](GraphT& g) {
            if(withSeqLens)
            {
                addSeqLens(g, 1);
            }
        }))) << "control, seq_lens="
             << withSeqLens;

        for(size_t i = 0; i < edits.size(); ++i)
        {
            EXPECT_FALSE(nodeAccepted(makeStatsGraph([&](GraphT& g) {
                if(withSeqLens)
                {
                    addSeqLens(g, 1);
                }
                edits[i](*g.nodes.front()->attributes.AsSdpaAttributes());
            }))) << "variant "
                 << i << ", seq_lens=" << withSeqLens;
        }
    }
}

// Native ragged Max and Sum_exp stay declined (the executors serve them through densify).
TEST(TestRaggedSdpaFwdSupport, GateDeclinesMaxAndSumExp)
{
    EXPECT_FALSE(nodeAccepted(makeStatsGraph([](GraphT& g) {
        addSeqLens(g, 1);
        g.nodes.front()->attributes.AsSdpaAttributes()->max_tensor_uid = STATS_UID;
    })));
    EXPECT_FALSE(nodeAccepted(makeStatsGraph([](GraphT& g) {
        addSeqLens(g, 1);
        g.nodes.front()->attributes.AsSdpaAttributes()->sum_exp_tensor_uid = STATS_UID;
    })));
}

// RaggedSdpaFwdParams carries the seq_len tensors of the node.
TEST(TestRaggedSdpaFwdSupport, ParamsCarrySeqLenTensors)
{
    const auto graph = makeStatsGraph([](GraphT& g) { addSeqLens(g, 1); });
    const GraphWrapper graphWrapper(graph.data(), graph.size());
    const RaggedSdpaFwdParams params(
        sdpaAttributes(graphWrapper), graphWrapper.getTensorMap(), -1, -1, true);
    ASSERT_TRUE(params.seqLenQTensor.has_value());
    ASSERT_TRUE(params.seqLenKvTensor.has_value());
    EXPECT_EQ(params.seqLenQTensor->uid, SEQ_LEN_Q_UID);
    EXPECT_EQ(params.seqLenKvTensor->uid, SEQ_LEN_KV_UID);

    const auto noSeqLens = makeStatsGraph();
    const GraphWrapper plainWrapper(noSeqLens.data(), noSeqLens.size());
    const RaggedSdpaFwdParams plain(
        sdpaAttributes(plainWrapper), plainWrapper.getTensorMap(), -1, -1, true);
    EXPECT_FALSE(plain.seqLenQTensor.has_value());
    EXPECT_FALSE(plain.seqLenKvTensor.has_value());
}
