// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// SdpaFwdPagedPlanBuilder / SdpaFwdPagedPlan: applicability against the dense SdpaFwdPlanBuilder,
// signature-key routing, and execution against a direct CpuFpReferenceSdpaRagged::forwardPaged
// call. Graphs are built at the flatbuffer level so the plan sees exactly the paged attributes.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPagedPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS;
using hipdnn_data_sdk::utilities::ShallowRaggedTensor;
using hipdnn_data_sdk::utilities::ShallowTensor;
using hipdnn_data_sdk::utilities::TensorBase;

namespace
{

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t PAGE_TABLE_K_UID = 5;
constexpr int64_t PAGE_TABLE_V_UID = 6;
constexpr int64_t SEQ_LEN_KV_UID = 7;
constexpr int64_t SEQ_LEN_Q_UID = 8;
constexpr int64_t STATS_UID = 9;
constexpr int64_t RAGGED_OFFSET_UID = 10;
constexpr int64_t MASK_UID = 11;

// B = 2, 2 query heads over 1 KV head, Sq (S_max for a ragged Q) 2, D 8, blocks of 4 keys, a
// pool of 6 blocks, max_seq_len_kv 8 (2 blocks per sequence).
constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 2;
constexpr int64_t KV_HEADS = 1;
constexpr int64_t SEQ_Q = 2;
constexpr int64_t HEAD_DIM = 8;
constexpr int64_t BLOCK_SIZE = 4;
constexpr int64_t NUM_BLOCKS = 6;
constexpr int32_t MAX_SEQ_LEN_KV = 8;
constexpr int64_t MAX_BLOCKS = 2;
constexpr float ATTN_SCALE = 0.35f;

using FloatPagedBuilder
    = SdpaFwdPagedPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>;
using FloatDenseBuilder
    = SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>;

struct PagedGraphOptions
{
    DataType dataType = DataType::FLOAT;
    bool pageTableK = true;
    bool pageTableV = true;
    bool sharedPageTable = false;
    bool seqLenKv = true;
    bool maxSeqLenKv = true;
    bool seqLenQ = false;
    // Q, O (and stats) packed, sharing one token-unit ragged_offset aux.
    bool raggedQ = false;
    bool raggedK = false;
    bool stats = false;
    bool attnMask = false;
    bool paddingMask = false;
};

std::unique_ptr<TensorAttributesT> makeTensor(int64_t uid,
                                              const std::string& name,
                                              DataType dataType,
                                              const std::vector<int64_t>& dims,
                                              const std::vector<int64_t>& strides)
{
    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = uid;
    tensor->name = name;
    tensor->data_type = dataType;
    tensor->dims = dims;
    tensor->strides = strides;
    return tensor;
}

// A Q-shaped tensor [B, H, S, width]: packed BSHD with the shared token aux, or dense.
std::unique_ptr<TensorAttributesT> makeQueryShaped(
    int64_t uid, const std::string& name, DataType dataType, int64_t width, bool ragged)
{
    const std::vector<int64_t> dims = {BATCH, HEADS, SEQ_Q, width};
    if(!ragged)
    {
        return makeTensor(uid, name, dataType, dims, generateStrides(dims));
    }
    auto tensor = makeTensor(uid, name, dataType, dims, raggedStrides(dims));
    tensor->ragged_offset_tensor_uid = RAGGED_OFFSET_UID;
    tensor->ragged_offset_multiplier = HEADS * width; // the aux holds token offsets
    return tensor;
}

flatbuffers::DetachedBuffer makePagedGraph(const PagedGraphOptions& opts)
{
    const std::vector<int64_t> kvDims = {NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM};
    const std::vector<int64_t> tableDims = {BATCH, 1, MAX_BLOCKS, 1};
    const std::vector<int64_t> tableStrides = {MAX_BLOCKS, MAX_BLOCKS, 1, 1};
    const std::vector<int64_t> lengthDims = {BATCH, 1, 1, 1};
    const std::vector<int64_t> auxDims = {BATCH + 1, 1, 1, 1};
    const std::vector<int64_t> maskDims = {BATCH, HEADS, SEQ_Q, MAX_SEQ_LEN_KV};

    GraphT graph;
    graph.name = "SdpaFwdPagedTest";
    graph.compute_data_type = DataType::FLOAT;
    graph.intermediate_data_type = DataType::FLOAT;
    graph.io_data_type = opts.dataType;

    graph.tensors.push_back(makeQueryShaped(Q_UID, "Q", opts.dataType, HEAD_DIM, opts.raggedQ));
    auto k = makeTensor(K_UID, "K", opts.dataType, kvDims, generateStrides(kvDims));
    if(opts.raggedK)
    {
        k->ragged_offset_tensor_uid = RAGGED_OFFSET_UID;
    }
    graph.tensors.push_back(std::move(k));
    graph.tensors.push_back(makeTensor(V_UID, "V", opts.dataType, kvDims, generateStrides(kvDims)));
    graph.tensors.push_back(makeQueryShaped(O_UID, "O", opts.dataType, HEAD_DIM, opts.raggedQ));
    graph.tensors.push_back(
        makeTensor(PAGE_TABLE_K_UID, "PAGE_TABLE_K", DataType::INT32, tableDims, tableStrides));
    graph.tensors.push_back(
        makeTensor(PAGE_TABLE_V_UID, "PAGE_TABLE_V", DataType::INT32, tableDims, tableStrides));
    graph.tensors.push_back(makeTensor(
        SEQ_LEN_KV_UID, "SEQ_LEN_KV", DataType::INT32, lengthDims, generateStrides(lengthDims)));
    graph.tensors.push_back(makeTensor(
        SEQ_LEN_Q_UID, "SEQ_LEN_Q", DataType::INT32, lengthDims, generateStrides(lengthDims)));
    if(opts.stats)
    {
        graph.tensors.push_back(
            makeQueryShaped(STATS_UID, "STATS", DataType::FLOAT, 1, opts.raggedQ));
    }
    if(opts.raggedQ || opts.raggedK)
    {
        graph.tensors.push_back(makeTensor(RAGGED_OFFSET_UID,
                                           "RAGGED_OFFSET",
                                           DataType::INT32,
                                           auxDims,
                                           generateStrides(auxDims)));
    }
    if(opts.attnMask)
    {
        graph.tensors.push_back(
            makeTensor(MASK_UID, "MASK", DataType::FLOAT, maskDims, generateStrides(maskDims)));
    }

    SdpaAttributesT sdpa;
    sdpa.q_tensor_uid = Q_UID;
    sdpa.k_tensor_uid = K_UID;
    sdpa.v_tensor_uid = V_UID;
    sdpa.o_tensor_uid = O_UID;
    if(opts.pageTableK)
    {
        sdpa.page_table_k_tensor_uid = PAGE_TABLE_K_UID;
    }
    if(opts.pageTableV)
    {
        sdpa.page_table_v_tensor_uid = opts.sharedPageTable ? PAGE_TABLE_K_UID : PAGE_TABLE_V_UID;
    }
    if(opts.seqLenKv)
    {
        sdpa.seq_len_kv_tensor_uid = SEQ_LEN_KV_UID;
    }
    if(opts.maxSeqLenKv)
    {
        sdpa.max_seq_len_kv = MAX_SEQ_LEN_KV;
    }
    if(opts.seqLenQ)
    {
        sdpa.seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
    }
    if(opts.stats)
    {
        sdpa.stats_tensor_uid = STATS_UID;
    }
    if(opts.attnMask)
    {
        sdpa.attn_mask_tensor_uid = MASK_UID;
    }
    sdpa.padding_mask = opts.paddingMask;
    sdpa.attn_scale_value = ATTN_SCALE;
    sdpa.right_bound = 0;
    sdpa.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;

    auto node = std::make_unique<NodeT>();
    node->name = "sdpa_fwd_paged";
    node->compute_data_type = DataType::FLOAT;
    node->attributes.Set(sdpa);
    graph.nodes.push_back(std::move(node));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, &graph));
    return builder.Release();
}

bool pagedBuilderApplicable(const PagedGraphOptions& opts)
{
    auto buffer = makePagedGraph(opts);
    const GraphWrapper graph(buffer.data(), buffer.size());
    return FloatPagedBuilder().isApplicable(graph.getNode(0), graph.getTensorMap());
}

bool denseBuilderApplicable(const PagedGraphOptions& opts)
{
    auto buffer = makePagedGraph(opts);
    const GraphWrapper graph(buffer.data(), buffer.size());
    return FloatDenseBuilder().isApplicable(graph.getNode(0), graph.getTensorMap());
}

// Host buffers for a FLOAT paged graph. Q holds 4 tokens either way: dense [2, 2, 2, 8], or
// packed with token boundaries {0, 2, 4}. Batch 1 has one live query row, so its second row
// is padding (dense, with seq_len_q) or a gap row (ragged).
struct PagedData
{
    explicit PagedData(bool raggedQIn)
        : raggedQ(raggedQIn)
    {
        const auto fill = [](std::vector<float>& buf, float base) {
            for(size_t i = 0; i < buf.size(); ++i)
            {
                buf[i] = base + 0.01f * static_cast<float>(i % 97) - 0.5f;
            }
        };
        fill(q, 0.1f);
        fill(k, -0.2f);
        fill(v, 0.3f);
    }

    std::unordered_map<int64_t, void*> variantPack(std::vector<float>& oOut,
                                                   std::vector<float>& lseOut)
    {
        return {{Q_UID, q.data()},
                {K_UID, k.data()},
                {V_UID, v.data()},
                {O_UID, oOut.data()},
                {PAGE_TABLE_K_UID, pageTableK.data()},
                {PAGE_TABLE_V_UID, pageTableV.data()},
                {SEQ_LEN_KV_UID, seqLenKv.data()},
                {SEQ_LEN_Q_UID, seqLenQ.data()},
                {STATS_UID, lseOut.data()},
                {RAGGED_OFFSET_UID, raggedOffset.data()}};
    }

    // A Q-shaped view [B, H, S, width] over buf, matching makeQueryShaped.
    std::unique_ptr<TensorBase<float>> queryShaped(std::vector<float>& buf, int64_t width)
    {
        const std::vector<int64_t> dims = {BATCH, HEADS, SEQ_Q, width};
        if(!raggedQ)
        {
            return std::make_unique<ShallowTensor<float>>(buf.data(), dims, generateStrides(dims));
        }
        const std::shared_ptr<ITensor> aux
            = std::make_shared<ShallowTensor<int32_t>>(raggedOffset.data(),
                                                       std::vector<int64_t>{BATCH + 1, 1, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1, 1});
        return std::make_unique<ShallowRaggedTensor<float>>(
            buf.data(), dims, raggedStrides(dims), SDPA_SEQ_AXIS, aux, std::nullopt, HEADS * width);
    }

    // The reference called directly, as SdpaFwdPagedPlan should call it.
    void runDirect(std::vector<float>& oOut, std::vector<float>* lseOut, bool sharedPageTable)
    {
        const std::vector<int64_t> kvDims = {NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM};
        const std::vector<int64_t> tableDims = {BATCH, 1, MAX_BLOCKS, 1};
        const std::vector<int64_t> tableStrides = {MAX_BLOCKS, MAX_BLOCKS, 1, 1};
        const std::vector<int64_t> lengthDims = {BATCH, 1, 1, 1};

        auto qTensor = queryShaped(q, HEAD_DIM);
        auto oTensor = queryShaped(oOut, HEAD_DIM);
        std::unique_ptr<TensorBase<float>> lseTensor;
        if(lseOut != nullptr)
        {
            lseTensor = queryShaped(*lseOut, 1);
        }
        ShallowTensor<float> kTensor(k.data(), kvDims, generateStrides(kvDims));
        ShallowTensor<float> vTensor(v.data(), kvDims, generateStrides(kvDims));
        ShallowTensor<int32_t> tableK(pageTableK.data(), tableDims, tableStrides);
        ShallowTensor<int32_t> tableV(
            sharedPageTable ? pageTableK.data() : pageTableV.data(), tableDims, tableStrides);
        ShallowTensor<int32_t> lengthsKv(seqLenKv.data(), lengthDims, generateStrides(lengthDims));
        ShallowTensor<int32_t> lengthsQ(seqLenQ.data(), lengthDims, generateStrides(lengthDims));

        CpuFpReferenceSdpaRagged::forwardPaged<float, float, float, float, float>(
            *qTensor,
            kTensor,
            vTensor,
            *oTensor,
            tableK,
            tableV,
            lengthsKv,
            &lengthsQ,
            MAX_SEQ_LEN_KV,
            ATTN_SCALE,
            /*leftBound=*/-1,
            /*rightBound=*/0,
            /*topLeftAlignment=*/false,
            lseTensor.get());
    }

    bool raggedQ;
    std::vector<float> q
        = std::vector<float>(static_cast<size_t>(BATCH * HEADS * SEQ_Q * HEAD_DIM));
    std::vector<float> k
        = std::vector<float>(static_cast<size_t>(NUM_BLOCKS * KV_HEADS * BLOCK_SIZE * HEAD_DIM));
    std::vector<float> v
        = std::vector<float>(static_cast<size_t>(NUM_BLOCKS * KV_HEADS * BLOCK_SIZE * HEAD_DIM));
    // Rows {b0: blocks 4, 1}, {b1: block 2, unused}. V uses other blocks.
    std::vector<int32_t> pageTableK = {4, 1, 2, -1};
    std::vector<int32_t> pageTableV = {0, 5, 3, -1};
    std::vector<int32_t> seqLenKv = {5, 3};
    std::vector<int32_t> seqLenQ = {2, 1};
    std::vector<int32_t> raggedOffset = {0, 2, 4};
};

constexpr float SENTINEL = 7.0f;

std::vector<float> sentinelOutput()
{
    return std::vector<float>(static_cast<size_t>(BATCH * HEADS * SEQ_Q * HEAD_DIM), SENTINEL);
}

std::vector<float> sentinelLse()
{
    return std::vector<float>(static_cast<size_t>(BATCH * HEADS * SEQ_Q), SENTINEL);
}

// Executes the graph through the paged plan builder and compares O and LSE with a direct
// forwardPaged call bit for bit.
void expectPlanMatchesForwardPaged(const PagedGraphOptions& opts)
{
    auto buffer = makePagedGraph(opts);
    const GraphWrapper graph(buffer.data(), buffer.size());
    const FloatPagedBuilder builder;
    ASSERT_TRUE(builder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    PagedData data(opts.raggedQ);
    auto oPlan = sentinelOutput();
    auto lsePlan = sentinelLse();
    auto plan = builder.buildNodePlan(graph, graph.getNode(0));
    plan->execute(data.variantPack(oPlan, lsePlan));

    auto oDirect = sentinelOutput();
    auto lseDirect = sentinelLse();
    data.runDirect(oDirect, opts.stats ? &lseDirect : nullptr, opts.sharedPageTable);

    EXPECT_EQ(oPlan, oDirect);
    EXPECT_EQ(lsePlan, lseDirect);
    // The plan wrote the live rows.
    EXPECT_NE(oPlan, sentinelOutput());
}

} // namespace

TEST(TestSdpaFwdPagedPlan, PagedBuilderAcceptsAndDenseBuilderDeclinesPagedGraph)
{
    const PagedGraphOptions opts;
    EXPECT_TRUE(pagedBuilderApplicable(opts));
    EXPECT_FALSE(denseBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, RaggedQPagedGraph)
{
    PagedGraphOptions opts;
    opts.raggedQ = true;
    opts.seqLenQ = true;
    opts.stats = true;
    EXPECT_TRUE(pagedBuilderApplicable(opts));
    EXPECT_FALSE(denseBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableWithOnePageTable)
{
    PagedGraphOptions onlyK;
    onlyK.pageTableV = false;
    EXPECT_FALSE(pagedBuilderApplicable(onlyK));
    EXPECT_FALSE(denseBuilderApplicable(onlyK));

    PagedGraphOptions onlyV;
    onlyV.pageTableK = false;
    EXPECT_FALSE(pagedBuilderApplicable(onlyV));
    EXPECT_FALSE(denseBuilderApplicable(onlyV));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableWithoutPageTables)
{
    PagedGraphOptions opts;
    opts.pageTableK = false;
    opts.pageTableV = false;
    EXPECT_FALSE(pagedBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableWithoutSeqLenKv)
{
    PagedGraphOptions opts;
    opts.seqLenKv = false;
    EXPECT_FALSE(pagedBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableWithoutMaxSeqLenKv)
{
    PagedGraphOptions opts;
    opts.maxSeqLenKv = false;
    EXPECT_FALSE(pagedBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableForRaggedQWithoutSeqLenQ)
{
    PagedGraphOptions opts;
    opts.raggedQ = true;
    EXPECT_FALSE(pagedBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableForRaggedK)
{
    PagedGraphOptions opts;
    opts.raggedK = true;
    EXPECT_FALSE(pagedBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableWithAttnMask)
{
    PagedGraphOptions opts;
    opts.attnMask = true;
    EXPECT_FALSE(pagedBuilderApplicable(opts));
}

TEST(TestSdpaFwdPagedPlan, NotApplicableForOtherDtype)
{
    auto buffer = makePagedGraph(PagedGraphOptions{});
    const GraphWrapper graph(buffer.data(), buffer.size());
    const SdpaFwdPagedPlanBuilder<DataType::BFLOAT16,
                                  DataType::BFLOAT16,
                                  DataType::BFLOAT16,
                                  DataType::BFLOAT16>
        builder;
    EXPECT_FALSE(builder.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestSdpaFwdPagedPlan, PaddingMaskEitherValue)
{
    PagedGraphOptions opts;
    opts.paddingMask = true;
    EXPECT_TRUE(pagedBuilderApplicable(opts));
}

// The signature key of a paged node finds the paged builder; the dense key keeps the dense one.
TEST(TestSdpaFwdPagedPlan, SignatureKeyRoutesPagedGraphs)
{
    const auto builders = SdpaFwdSignatureKey::getPlanBuilders();

    auto pagedBuffer = makePagedGraph(PagedGraphOptions{});
    const GraphWrapper pagedGraph(pagedBuffer.data(), pagedBuffer.size());
    const SdpaFwdSignatureKey pagedKey(pagedGraph.getNode(0), pagedGraph.getTensorMap());
    EXPECT_TRUE(pagedKey.paged);
    const auto pagedIt = builders.find(pagedKey);
    ASSERT_NE(pagedIt, builders.end());
    EXPECT_NE(dynamic_cast<const FloatPagedBuilder*>(pagedIt->second.get()), nullptr);

    PagedGraphOptions denseOpts;
    denseOpts.pageTableK = false;
    denseOpts.pageTableV = false;
    auto denseBuffer = makePagedGraph(denseOpts);
    const GraphWrapper denseGraph(denseBuffer.data(), denseBuffer.size());
    const SdpaFwdSignatureKey denseKey(denseGraph.getNode(0), denseGraph.getTensorMap());
    EXPECT_FALSE(denseKey.paged);
    const auto denseIt = builders.find(denseKey);
    ASSERT_NE(denseIt, builders.end());
    EXPECT_NE(dynamic_cast<const FloatDenseBuilder*>(denseIt->second.get()), nullptr);
}

TEST(TestSdpaFwdPagedPlan, GraphExecutorAcceptsPagedGraph)
{
    CpuReferenceGraphExecutor executor;

    auto buffer = makePagedGraph(PagedGraphOptions{});
    EXPECT_TRUE(executor.isApplicable(buffer.data(), buffer.size()));

    PagedGraphOptions onlyK;
    onlyK.pageTableV = false;
    auto onlyKBuffer = makePagedGraph(onlyK);
    EXPECT_FALSE(executor.isApplicable(onlyKBuffer.data(), onlyKBuffer.size()));

    PagedGraphOptions masked;
    masked.attnMask = true;
    auto maskedBuffer = makePagedGraph(masked);
    EXPECT_FALSE(executor.isApplicable(maskedBuffer.data(), maskedBuffer.size()));
}

TEST(TestSdpaFwdPagedPlan, ExecuteMatchesForwardPaged)
{
    PagedGraphOptions opts;
    opts.seqLenQ = true;
    opts.stats = true;
    expectPlanMatchesForwardPaged(opts);
}

TEST(TestSdpaFwdPagedPlan, ExecuteSharedPageTableMatchesForwardPaged)
{
    PagedGraphOptions opts;
    opts.seqLenQ = true;
    opts.sharedPageTable = true;
    expectPlanMatchesForwardPaged(opts);
}

TEST(TestSdpaFwdPagedPlan, ExecuteRaggedQMatchesForwardPaged)
{
    PagedGraphOptions opts;
    opts.raggedQ = true;
    opts.seqLenQ = true;
    opts.stats = true;
    expectPlanMatchesForwardPaged(opts);
}

TEST(TestSdpaFwdPagedPlan, GraphExecutorExecuteMatchesForwardPaged)
{
    PagedGraphOptions opts;
    opts.raggedQ = true;
    opts.seqLenQ = true;
    opts.stats = true;
    auto buffer = makePagedGraph(opts);

    PagedData data(/*raggedQIn=*/true);
    auto oExec = sentinelOutput();
    auto lseExec = sentinelLse();
    CpuReferenceGraphExecutor executor;
    executor.execute(buffer.data(), buffer.size(), data.variantPack(oExec, lseExec));

    auto oDirect = sentinelOutput();
    auto lseDirect = sentinelLse();
    data.runDirect(oDirect, &lseDirect, /*sharedPageTable=*/false);

    EXPECT_EQ(oExec, oDirect);
    EXPECT_EQ(lseExec, lseDirect);
}
