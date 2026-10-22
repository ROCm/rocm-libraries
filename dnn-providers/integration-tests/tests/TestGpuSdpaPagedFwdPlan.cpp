// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaFwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaPagedFwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaRaggedFwdPlan.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;

namespace
{

constexpr int64_t Q_UID = 10;
constexpr int64_t K_UID = 11;
constexpr int64_t V_UID = 12;
constexpr int64_t O_UID = 13;
constexpr int64_t STATS_UID = 14;
constexpr int64_t RAGGED_OFFSET_Q_UID = 20;
constexpr int64_t RAGGED_OFFSET_O_UID = 21;
constexpr int64_t PAGE_TABLE_K_UID = 30;
constexpr int64_t PAGE_TABLE_V_UID = 31;
constexpr int64_t SEQ_LEN_KV_UID = 32;
constexpr int64_t SEQ_LEN_Q_UID = 33;
constexpr int64_t SINK_UID = 34;

// Not in the tensor map. Setting an unsupported-mode uid to it is enough to reject the node.
constexpr int64_t UNUSED_UID = 99;

constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 2;
constexpr int64_t SEQ_Q = 4;
constexpr int64_t HEAD_DIM = 16;
constexpr int64_t NUM_BLOCKS = 6;
constexpr int64_t BLOCK_SIZE = 8;
constexpr int32_t MAX_SEQ_LEN_KV = 24; // M = 3
constexpr int64_t MAX_BLOCKS = 3;

const std::vector<int64_t> Q_DIMS = raggedDims(BATCH, SEQ_Q, HEADS, HEAD_DIM);
const std::vector<int64_t> KV_DIMS = {NUM_BLOCKS, HEADS, BLOCK_SIZE, HEAD_DIM};

using Bf16Builder = GpuSdpaPagedFwdPlanBuilder<DataType::BFLOAT16,
                                               DataType::BFLOAT16,
                                               DataType::BFLOAT16,
                                               DataType::BFLOAT16>;
using Fp16Builder
    = GpuSdpaPagedFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>;

flatbuffers::FlatBufferBuilder makePagedGraph(SdpaAttributesT attrs = {},
                                              DataType dataType = DataType::BFLOAT16,
                                              PagedSdpaFwdGraphOptions options = {})
{
    options.attrs = attrs;
    return createPagedSdpaFwdGraph(Q_UID,
                                   K_UID,
                                   V_UID,
                                   O_UID,
                                   PAGE_TABLE_K_UID,
                                   PAGE_TABLE_V_UID,
                                   SEQ_LEN_KV_UID,
                                   MAX_SEQ_LEN_KV,
                                   Q_DIMS,
                                   KV_DIMS,
                                   KV_DIMS,
                                   Q_DIMS,
                                   dataType,
                                   options);
}

PagedSdpaFwdGraphOptions raggedQOptions()
{
    PagedSdpaFwdGraphOptions options;
    options.seqLenQUid = SEQ_LEN_Q_UID;
    options.raggedOffsetQUid = RAGGED_OFFSET_Q_UID;
    options.raggedOffsetOUid = RAGGED_OFFSET_O_UID;
    return options;
}

// Re-serializes a graph after editing its object form.
flatbuffers::FlatBufferBuilder mutateGraph(const flatbuffers::FlatBufferBuilder& source,
                                           const std::function<void(GraphT&)>& edit)
{
    auto graph = UnPackGraph(source.GetBufferPointer());
    edit(*graph);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, graph.get()));
    return builder;
}

TensorAttributesT& tensorWithUid(GraphT& graph, int64_t uid)
{
    for(auto& tensor : graph.tensors)
    {
        if(tensor->uid == uid)
        {
            return *tensor;
        }
    }
    throw std::runtime_error("no tensor with uid " + std::to_string(uid));
}

template <typename Builder>
bool isApplicable(const flatbuffers::FlatBufferBuilder& graphBuilder)
{
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Builder builder;
    return builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
}

// INT32 [B, 1, M, 1] page table with the zero-copy block_table strides [M, M, 1, 1].
Tensor<int32_t> makePageTable(const std::vector<int32_t>& entries)
{
    Tensor<int32_t> table({BATCH, 1, MAX_BLOCKS, 1}, {MAX_BLOCKS, MAX_BLOCKS, 1, 1});
    auto* p = table.memory().hostData();
    for(size_t i = 0; i < entries.size(); ++i)
    {
        p[i] = entries[i];
    }
    table.memory().markHostModified();
    return table;
}

Tensor<int32_t> makeInt32(const std::vector<int64_t>& dims, const std::vector<int32_t>& values)
{
    Tensor<int32_t> t(dims);
    auto* p = t.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        p[i] = values[i];
    }
    t.memory().markHostModified();
    return t;
}

// The diagonal band fpropPaged is called with.
struct ExpectedBand
{
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
};

SdpaAttributesT bottomRightCausal()
{
    SdpaAttributesT attrs;
    attrs.right_bound = 0;
    attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    return attrs;
}

// Runs the bf16 plan for @p attrs and fpropPaged directly with @p band on the same inputs and
// expects identical outputs (same kernel, same arguments), including untouched padding rows.
// @p sinkDataType adds a sink_token of that type (FLOAT or BFLOAT16).
void checkExecuteMatchesDirect(bool raggedQ,
                               SdpaAttributesT attrs = bottomRightCausal(),
                               ExpectedBand band = {-1, 0, false},
                               std::optional<DataType> sinkDataType = std::nullopt)
{
    attrs.attn_scale_value = 0.25f;
    PagedSdpaFwdGraphOptions options = raggedQ ? raggedQOptions() : PagedSdpaFwdGraphOptions{};
    options.statsUid = STATS_UID;
    if(sinkDataType.has_value())
    {
        options.sinkUid = SINK_UID;
        options.sinkDataType = sinkDataType.value();
    }
    auto graphBuilder = makePagedGraph(attrs, DataType::BFLOAT16, options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Bf16Builder builder;
    ASSERT_TRUE(builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<bfloat16> q(Q_DIMS, raggedStrides(Q_DIMS));
    Tensor<bfloat16> k(KV_DIMS, raggedStrides(KV_DIMS));
    Tensor<bfloat16> v(KV_DIMS, raggedStrides(KV_DIMS));
    q.fillTensorWithRandomValues(-1.0f, 1.0f, /*seed=*/11);
    k.fillTensorWithRandomValues(-1.0f, 1.0f, /*seed=*/22);
    v.fillTensorWithRandomValues(-1.0f, 1.0f, /*seed=*/33);

    // Batch 0 uses 3 shuffled blocks, batch 1 uses 1 (unused entries -1).
    auto pageTableK = makePageTable({4, 1, 0, 5, -1, -1});
    auto pageTableV = makePageTable({2, 5, 3, 0, -1, -1});
    auto seqLenKv = makeInt32({BATCH, 1, 1, 1}, {19, 3});
    // Ragged Q: 3 and 1 tokens, with seq_len_q leaving a gap row in batch 0.
    auto seqLenQ = makeInt32({BATCH, 1, 1, 1}, {2, 1});
    const int32_t seqStride = HEADS * HEAD_DIM;
    auto offQ = makeInt32({BATCH + 1, 1, 1, 1}, {0, 3 * seqStride, 4 * seqStride});
    auto offO = makeInt32({BATCH + 1, 1, 1, 1}, {0, 3 * seqStride, 4 * seqStride});
    // Per-head sinks in both supported types; the graph binds the one of sinkDataType.
    Tensor<float> sinkFloat({1, HEADS, 1, 1});
    Tensor<bfloat16> sinkBf16({1, HEADS, 1, 1});
    for(int64_t h = 0; h < HEADS; ++h)
    {
        const float value = 1.5f - 2.0f * static_cast<float>(h);
        sinkFloat.setHostValue(value, 0, h, 0, 0);
        sinkBf16.setHostValue(static_cast<bfloat16>(value), 0, h, 0, 0);
    }
    sinkFloat.memory().markHostModified();
    sinkBf16.memory().markHostModified();
    const bool bf16Sink = sinkDataType == DataType::BFLOAT16;

    const auto makeO = [&]() {
        Tensor<bfloat16> o(Q_DIMS, raggedStrides(Q_DIMS));
        o.fillWithValue(static_cast<bfloat16>(7.0f));
        return o;
    };
    const auto makeLse = [&]() {
        Tensor<float> lse({BATCH, HEADS, SEQ_Q, 1});
        lse.fillWithValue(-99.0f);
        return lse;
    };
    auto oPlan = makeO();
    auto oDirect = makeO();
    auto lsePlan = makeLse();
    auto lseDirect = makeLse();

    std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {STATS_UID, lsePlan.memory().deviceData()},
        {PAGE_TABLE_K_UID, pageTableK.memory().deviceData()},
        {PAGE_TABLE_V_UID, pageTableV.memory().deviceData()},
        {SEQ_LEN_KV_UID, seqLenKv.memory().deviceData()},
    };
    if(raggedQ)
    {
        variantPack.emplace(SEQ_LEN_Q_UID, seqLenQ.memory().deviceData());
        variantPack.emplace(RAGGED_OFFSET_Q_UID, offQ.memory().deviceData());
        variantPack.emplace(RAGGED_OFFSET_O_UID, offO.memory().deviceData());
    }
    if(sinkDataType.has_value())
    {
        variantPack.emplace(
            SINK_UID, bf16Sink ? sinkBf16.memory().deviceData() : sinkFloat.memory().deviceData());
    }
    plan->execute(variantPack);
    oPlan.markDeviceModified();
    lsePlan.markDeviceModified();

    const auto runDirect = [&](auto* sink, auto sinkTag) {
        using SinkT = typename decltype(sinkTag)::type;
        hipdnn_gpu_ref::GpuFpReferenceSdpaRagged::
            fpropPaged<bfloat16, bfloat16, bfloat16, bfloat16, float, SinkT>(
                q,
                k,
                v,
                oDirect,
                raggedQ ? &offQ : nullptr,
                raggedQ ? &offO : nullptr,
                pageTableK,
                pageTableV,
                seqLenKv,
                raggedQ ? &seqLenQ : nullptr,
                MAX_SEQ_LEN_KV,
                0.25f,
                band.leftBound,
                band.rightBound,
                band.topLeftAlignment,
                &lseDirect,
                nullptr,
                sdpaProbabilityMode<bfloat16, bfloat16, bfloat16, bfloat16>(),
                nullptr,
                nullptr,
                nullptr,
                {},
                sink);
    };
    if(!sinkDataType.has_value())
    {
        runDirect(static_cast<Tensor<float>*>(nullptr), SdpaSinkTypeTag<float>{});
    }
    else if(bf16Sink)
    {
        runDirect(&sinkBf16, SdpaSinkTypeTag<bfloat16>{});
    }
    else
    {
        runDirect(&sinkFloat, SdpaSinkTypeTag<float>{});
    }

    const auto* op = oPlan.memory().hostData();
    const auto* od = oDirect.memory().hostData();
    for(size_t i = 0; i < oPlan.elementSpace(); ++i)
    {
        EXPECT_EQ(static_cast<float>(op[i]), static_cast<float>(od[i]))
            << "output mismatch at element " << i;
    }
    const auto* lp = lsePlan.memory().hostData();
    const auto* ld = lseDirect.memory().hostData();
    for(size_t i = 0; i < lsePlan.elementSpace(); ++i)
    {
        EXPECT_EQ(lp[i], ld[i]) << "LSE mismatch at element " << i;
    }
}

} // namespace

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsApplicableForBf16DenseQ)
{
    EXPECT_TRUE(isApplicable<Bf16Builder>(makePagedGraph()));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsApplicableForFp16DenseQ)
{
    const auto graph = makePagedGraph({}, DataType::HALF);
    EXPECT_TRUE(isApplicable<Fp16Builder>(graph));
    EXPECT_FALSE(isApplicable<Bf16Builder>(graph));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsApplicableForRaggedQWithSeqLenQ)
{
    EXPECT_TRUE(
        isApplicable<Bf16Builder>(makePagedGraph({}, DataType::BFLOAT16, raggedQOptions())));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsApplicableForSharedPageTable)
{
    auto graph = createPagedSdpaFwdGraph(Q_UID,
                                         K_UID,
                                         V_UID,
                                         O_UID,
                                         PAGE_TABLE_K_UID,
                                         PAGE_TABLE_K_UID,
                                         SEQ_LEN_KV_UID,
                                         MAX_SEQ_LEN_KV,
                                         Q_DIMS,
                                         KV_DIMS,
                                         KV_DIMS,
                                         Q_DIMS,
                                         DataType::BFLOAT16);
    EXPECT_TRUE(isApplicable<Bf16Builder>(graph));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsApplicableWithPaddingMaskAndLse)
{
    SdpaAttributesT attrs;
    attrs.padding_mask = true;
    PagedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID;
    EXPECT_TRUE(isApplicable<Bf16Builder>(makePagedGraph(attrs, DataType::BFLOAT16, options)));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsNotApplicableForRaggedQWithoutSeqLenQ)
{
    auto options = raggedQOptions();
    options.seqLenQUid.reset();
    EXPECT_FALSE(isApplicable<Bf16Builder>(makePagedGraph({}, DataType::BFLOAT16, options)));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsNotApplicableWhenOnlyQIsRagged)
{
    auto options = raggedQOptions();
    options.raggedOffsetOUid.reset();
    EXPECT_FALSE(isApplicable<Bf16Builder>(makePagedGraph({}, DataType::BFLOAT16, options)));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsNotApplicableForRaggedK)
{
    const auto graph
        = mutateGraph(makePagedGraph({}, DataType::BFLOAT16, raggedQOptions()), [](GraphT& g) {
              tensorWithUid(g, K_UID).ragged_offset_tensor_uid = RAGGED_OFFSET_Q_UID;
          });
    EXPECT_FALSE(isApplicable<Bf16Builder>(graph));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsNotApplicableWithoutRequiredPagedInputs)
{
    const std::vector<std::function<void(SdpaAttributesT&)>> edits = {
        [](SdpaAttributesT& a) { a.page_table_k_tensor_uid = flatbuffers::nullopt; },
        [](SdpaAttributesT& a) { a.page_table_v_tensor_uid = flatbuffers::nullopt; },
        [](SdpaAttributesT& a) { a.seq_len_kv_tensor_uid = flatbuffers::nullopt; },
        [](SdpaAttributesT& a) { a.max_seq_len_kv = flatbuffers::nullopt; },
        [](SdpaAttributesT& a) { a.page_table_v_tensor_uid = UNUSED_UID; },
    };
    for(size_t i = 0; i < edits.size(); ++i)
    {
        const auto graph = mutateGraph(makePagedGraph(), [&](GraphT& g) {
            edits[i](*g.nodes[0]->attributes.AsSdpaAttributes());
        });
        EXPECT_FALSE(isApplicable<Bf16Builder>(graph)) << "edit " << i;
    }
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsNotApplicableForUnsupportedModes)
{
    const std::vector<std::function<void(SdpaAttributesT&)>> edits = {
        [](SdpaAttributesT& a) { a.alibi_mask = true; },
        [](SdpaAttributesT& a) { a.attn_mask_tensor_uid = UNUSED_UID; },
        [](SdpaAttributesT& a) { a.dropout_probability = 0.1f; },
        [](SdpaAttributesT& a) { a.seed_tensor_uid = UNUSED_UID; },
        [](SdpaAttributesT& a) { a.block_mask_tensor_uid = UNUSED_UID; },
        [](SdpaAttributesT& a) { a.max_tensor_uid = UNUSED_UID; },
        [](SdpaAttributesT& a) { a.sum_exp_tensor_uid = UNUSED_UID; },
        [](SdpaAttributesT& a) { a.descale_s_tensor_uid = UNUSED_UID; },
        [](SdpaAttributesT& a) { a.scale_o_tensor_uid = UNUSED_UID; },
    };
    for(size_t i = 0; i < edits.size(); ++i)
    {
        SdpaAttributesT attrs;
        edits[i](attrs);
        EXPECT_FALSE(isApplicable<Bf16Builder>(makePagedGraph(attrs))) << "edit " << i;
    }
}

namespace
{
PagedSdpaFwdGraphOptions sinkOptions(DataType sinkDataType)
{
    PagedSdpaFwdGraphOptions options;
    options.sinkUid = SINK_UID;
    options.sinkDataType = sinkDataType;
    return options;
}
} // namespace

// Attention sinks: [1, H, 1, 1] FLOAT, or the Q data type when HALF/BFLOAT16.
TEST(TestGpuSdpaPagedFwdPlanBuilder, IsApplicableWithSink)
{
    EXPECT_TRUE(isApplicable<Bf16Builder>(
        makePagedGraph({}, DataType::BFLOAT16, sinkOptions(DataType::FLOAT))));
    EXPECT_TRUE(isApplicable<Bf16Builder>(
        makePagedGraph({}, DataType::BFLOAT16, sinkOptions(DataType::BFLOAT16))));
    EXPECT_TRUE(
        isApplicable<Fp16Builder>(makePagedGraph({}, DataType::HALF, sinkOptions(DataType::HALF))));
    auto ragged = raggedQOptions();
    ragged.sinkUid = SINK_UID;
    EXPECT_TRUE(isApplicable<Bf16Builder>(makePagedGraph({}, DataType::BFLOAT16, ragged)));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsNotApplicableWithInvalidSink)
{
    // A sink in another half type than Q, or an integer sink.
    EXPECT_FALSE(isApplicable<Bf16Builder>(
        makePagedGraph({}, DataType::BFLOAT16, sinkOptions(DataType::HALF))));
    EXPECT_FALSE(isApplicable<Bf16Builder>(
        makePagedGraph({}, DataType::BFLOAT16, sinkOptions(DataType::INT32))));
    // One sink per KV head instead of per query head.
    const auto kvHeadSink
        = mutateGraph(makePagedGraph({}, DataType::BFLOAT16, sinkOptions(DataType::FLOAT)),
                      [](GraphT& g) { tensorWithUid(g, SINK_UID).dims = {1, 1, 1, 1}; });
    EXPECT_FALSE(isApplicable<Bf16Builder>(kvHeadSink));
    // A sink uid with no tensor behind it.
    SdpaAttributesT dangling;
    dangling.sink_token_tensor_uid = UNUSED_UID;
    EXPECT_FALSE(isApplicable<Bf16Builder>(makePagedGraph(dangling)));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, IsNotApplicableForNonPagedNodes)
{
    const auto dense = createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, Q_DIMS, Q_DIMS, Q_DIMS, Q_DIMS, DataType::BFLOAT16);
    EXPECT_FALSE(isApplicable<Bf16Builder>(dense));
    const auto ragged = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_O_UID,
                                                 BATCH,
                                                 Q_DIMS,
                                                 Q_DIMS,
                                                 Q_DIMS,
                                                 Q_DIMS,
                                                 DataType::BFLOAT16);
    EXPECT_FALSE(isApplicable<Bf16Builder>(ragged));
}

// The dense and ragged plans must keep declining page tables.
TEST(TestGpuSdpaPagedFwdPlanBuilder, DenseAndRaggedPlansRejectPagedNodes)
{
    using DenseBuilder = GpuSdpaFwdPlanBuilder<DataType::BFLOAT16,
                                               DataType::BFLOAT16,
                                               DataType::BFLOAT16,
                                               DataType::BFLOAT16>;
    using RaggedBuilder = GpuSdpaRaggedFwdPlanBuilder<DataType::BFLOAT16,
                                                      DataType::BFLOAT16,
                                                      DataType::BFLOAT16,
                                                      DataType::BFLOAT16>;
    EXPECT_FALSE(isApplicable<DenseBuilder>(makePagedGraph()));
    EXPECT_FALSE(
        isApplicable<RaggedBuilder>(makePagedGraph({}, DataType::BFLOAT16, raggedQOptions())));
}

TEST(TestGpuSdpaPagedFwdPlanBuilder, ExecutorRoutesPagedNodes)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
    GpuReferenceGraphExecutor executor;
    for(const auto& options : {PagedSdpaFwdGraphOptions{}, raggedQOptions()})
    {
        for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
        {
            auto graph = makePagedGraph({}, dataType, options);
            EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
        }
    }
}

TEST(TestGpuSdpaPagedFwdPlan, ExecuteDenseQMatchesDirectFpropPaged)
{
    SKIP_IF_NO_DEVICES();
    checkExecuteMatchesDirect(/*raggedQ=*/false);
}

TEST(TestGpuSdpaPagedFwdPlan, ExecuteRaggedQMatchesDirectFpropPaged)
{
    SKIP_IF_NO_DEVICES();
    checkExecuteMatchesDirect(/*raggedQ=*/true);
}

// The deprecated causal flags compose with left_bound and diagonal_alignment the way the CPU
// and dense GPU plans compose them (cuDNN's causal sliding window), rather than resetting them.
TEST(TestGpuSdpaPagedFwdPlan, ExecuteComposesTheCausalFlagWithTheBand)
{
    SKIP_IF_NO_DEVICES();
    SdpaAttributesT causalWindow;
    causalWindow.causal_mask = true;
    causalWindow.left_bound = 1;
    checkExecuteMatchesDirect(/*raggedQ=*/false, causalWindow, {1, 0, true});

    SdpaAttributesT causalBottomRight;
    causalBottomRight.causal_mask = true;
    causalBottomRight.left_bound = 2;
    causalBottomRight.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    checkExecuteMatchesDirect(/*raggedQ=*/true, causalBottomRight, {2, 0, false});

    SdpaAttributesT bottomRightFlagWindow;
    bottomRightFlagWindow.causal_mask_bottom_right = true;
    bottomRightFlagWindow.left_bound = 3;
    checkExecuteMatchesDirect(/*raggedQ=*/false, bottomRightFlagWindow, {3, 0, false});
}

TEST(TestGpuSdpaPagedFwdPlan, ExecuteWithSinkMatchesDirectFpropPaged)
{
    SKIP_IF_NO_DEVICES();
    checkExecuteMatchesDirect(
        /*raggedQ=*/false, bottomRightCausal(), {-1, 0, false}, DataType::FLOAT);
    checkExecuteMatchesDirect(
        /*raggedQ=*/true, bottomRightCausal(), {-1, 0, false}, DataType::BFLOAT16);
}
