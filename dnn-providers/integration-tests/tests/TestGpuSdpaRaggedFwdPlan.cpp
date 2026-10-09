// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
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
constexpr int64_t RAGGED_OFFSET_Q_UID = 20;
constexpr int64_t RAGGED_OFFSET_KV_UID = 21;
constexpr int64_t STATS_UID = 14;
constexpr int64_t RAGGED_OFFSET_STATS_UID = 22;

// Not in the tensor map. Setting an unsupported-mode uid to it is enough to reject the node.
constexpr int64_t UNUSED_UID = 99;

// One batch. The shape doesn't affect applicability.
const std::vector<int64_t> DIMS = raggedDims(1, 8, 2, 16);

using Bf16Builder = GpuSdpaRaggedFwdPlanBuilder<DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16>;

// bf16 ragged graph with optional extra attrs.
flatbuffers::FlatBufferBuilder makeRaggedGraph(SdpaAttributesT attrs = {})
{
    RaggedSdpaFwdGraphOptions options;
    options.attrs = attrs;
    return createRaggedSdpaFwdGraph(Q_UID,
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
}

// INT32 ragged_offset aux [B+1,1,1,1] in element offsets: cumTokens * seqStride.
Tensor<int32_t> makeRaggedOffset(const std::vector<int64_t>& lengths, int64_t seqStride)
{
    Tensor<int32_t> off({static_cast<int64_t>(lengths.size()) + 1, 1, 1, 1});
    auto* p = off.memory().hostData();
    p[0] = 0;
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        p[i + 1] = p[i] + static_cast<int32_t>(lengths[i] * seqStride);
    }
    off.memory().markHostModified();
    return off;
}

using Fp32Builder = GpuSdpaRaggedFwdPlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT>;

// Wraps a borrowed packed host buffer as an RFC-0014 ragged tensor ([B, S, H, D], BSHD_SEQ_AXIS).
ShallowRaggedTensor<float> wrapRagged(float* buf,
                                      const std::vector<int64_t>& dims,
                                      int64_t seqStride,
                                      const std::vector<int64_t>& lengths)
{
    return {buf,
            dims,
            raggedStrides(dims),
            BSHD_SEQ_AXIS,
            makeRaggedOffsetAux(cumTokens(lengths), seqStride)};
}

// Runs the fp32 plan with unequal Q/KV lengths and an LSE in statsLayout, and compares it with
// the CPU ragged reference using the same layout. Both LSE buffers start at a sentinel, so a
// misaddressed or padding row shows up as a mismatch. With tokenOffsets the graph's offset
// tables hold tokens and each tensor's ragged_offset_multiplier is its seq stride (AITER's
// form); the CPU side always uses element offsets.
void checkPlanLseAgainstCpu(RaggedStatsLayout statsLayout, bool tokenOffsets = false)
{
    const std::vector<int64_t> seqQ = {3, 5, 1};
    const std::vector<int64_t> seqKv = {4, 2, 6};
    const int64_t batch = 3;
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const int64_t sMaxQ = 5;
    const int64_t sMaxKv = 6;
    const int64_t totalQ = 9;
    const int64_t seqStride = numHeads * headDim;
    const bool packedStats = statsLayout == RaggedStatsLayout::PACKED;

    const auto qDims = raggedDims(batch, sMaxQ, numHeads, headDim);
    const auto kvDims = raggedDims(batch, sMaxKv, numHeads, headDim);
    const auto lseDims = raggedDims(batch, sMaxQ, numHeads, 1);

    RaggedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID;
    options.statsLayout = statsLayout;
    if(packedStats)
    {
        options.raggedOffsetStatsUid = RAGGED_OFFSET_STATS_UID;
    }
    options.tokenOffsets = tokenOffsets;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qDims,
                                                 kvDims,
                                                 kvDims,
                                                 qDims,
                                                 DataType::FLOAT,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Fp32Builder fp32Builder;
    ASSERT_TRUE(fp32Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp32Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(qDims, raggedStrides(qDims));
    Tensor<float> k(kvDims, raggedStrides(kvDims));
    Tensor<float> v(kvDims, raggedStrides(kvDims));
    q.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/11);
    k.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/22);
    v.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/33);
    auto offQ = makeRaggedOffset(seqQ, tokenOffsets ? 1 : seqStride);
    auto offKv = makeRaggedOffset(seqKv, tokenOffsets ? 1 : seqStride);
    auto offLse = makeRaggedOffset(seqQ, tokenOffsets ? 1 : numHeads); // packed LSE stride is H

    constexpr float SENTINEL = -99.0f;
    const auto makeLse = [&]() {
        Tensor<float> lse
            = packedStats ? Tensor<float>(lseDims, raggedStrides(lseDims)) : Tensor<float>(lseDims);
        lse.fillWithValue(SENTINEL);
        return lse;
    };
    auto lsePlan = makeLse();
    auto lseCpu = makeLse();
    Tensor<float> oPlan(qDims, raggedStrides(qDims));
    Tensor<float> oCpu(qDims, raggedStrides(qDims));

    std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
        {STATS_UID, lsePlan.memory().deviceData()},
    };
    if(packedStats)
    {
        variantPack.emplace(RAGGED_OFFSET_STATS_UID, offLse.memory().deviceData());
    }
    plan->execute(variantPack);
    oPlan.markDeviceModified();
    lsePlan.markDeviceModified();

    {
        auto qR = wrapRagged(q.memory().hostData(), qDims, seqStride, seqQ);
        auto kR = wrapRagged(k.memory().hostData(), kvDims, seqStride, seqKv);
        auto vR = wrapRagged(v.memory().hostData(), kvDims, seqStride, seqKv);
        auto oR = wrapRagged(oCpu.memory().hostData(), qDims, seqStride, seqQ);
        if(packedStats)
        {
            auto lseR = wrapRagged(lseCpu.memory().hostData(), lseDims, numHeads, seqQ);
            CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                qR, kR, vR, oR, std::nullopt, -1, -1, true, &lseR);
        }
        else
        {
            CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                qR, kR, vR, oR, std::nullopt, -1, -1, true, &lseCpu);
        }
    }

    const float tolerance = 1e-4f;
    const auto* oPlanHost = oPlan.memory().hostData();
    const auto* oCpuHost = oCpu.memory().hostData();
    for(int64_t i = 0; i < totalQ * seqStride; ++i) // packed region only
    {
        EXPECT_NEAR(oPlanHost[i], oCpuHost[i], tolerance) << "output mismatch at element " << i;
    }
    const auto* lsePlanHost = lsePlan.memory().hostData();
    const auto* lseCpuHost = lseCpu.memory().hostData();
    for(int64_t i = 0; i < batch * numHeads * sMaxQ; ++i) // whole buffer, incl. padding sentinels
    {
        EXPECT_NEAR(lsePlanHost[i], lseCpuHost[i], tolerance) << "LSE mismatch at element " << i;
    }
}

} // namespace

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsApplicableForBf16RaggedNode)
{
    auto graphBuilder = makeRaggedGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Bf16Builder bf16Builder;
    EXPECT_TRUE(bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForDenseNode)
{
    // No ragged_offset on the primaries, so this node belongs to the dense plan.
    auto graphBuilder = createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, DIMS, DIMS, DIMS, DIMS, DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Bf16Builder bf16Builder;
    EXPECT_FALSE(bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

namespace
{

// Re-serializes a graph with one tensor's dims and strides replaced.
flatbuffers::DetachedBuffer withShape(const void* graphBuffer,
                                      int64_t uid,
                                      const std::vector<int64_t>& dims,
                                      const std::vector<int64_t>& strides)
{
    auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuffer)->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        if(tensor->uid == uid)
        {
            tensor->dims = dims;
            tensor->strides = strides;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

} // namespace

// A primary or packed LSE in the pre-RFC [B, H, S, D] order (BSHD strides) is not ragged-legal
// under RFC-0014, so the plan declines it instead of misreading heads as tokens.
TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForHeadsBeforeSequenceLayout)
{
    const Bf16Builder bf16Builder;
    // DIMS is [1, S = 8, H = 2, D = 16]; the same memory as [B, H, S, D] has strides
    // [256, 16, 32, 1].
    auto graphBuilder = makeRaggedGraph();
    const auto qOld
        = withShape(graphBuilder.GetBufferPointer(), Q_UID, {1, 2, 8, 16}, {256, 16, 32, 1});
    auto qWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(qOld.data(), qOld.size());
    EXPECT_FALSE(bf16Builder.isApplicable(qWrap.getNode(0), qWrap.getTensorMap()));

    RaggedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID;
    options.statsLayout = RaggedStatsLayout::PACKED;
    options.raggedOffsetStatsUid = RAGGED_OFFSET_STATS_UID;
    auto statsGraph = createRaggedSdpaFwdGraph(Q_UID,
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
    auto statsWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        statsGraph.GetBufferPointer(), statsGraph.GetSize());
    ASSERT_TRUE(bf16Builder.isApplicable(statsWrap.getNode(0), statsWrap.getTensorMap()));
    const auto lseOld
        = withShape(statsGraph.GetBufferPointer(), STATS_UID, {1, 2, 8, 1}, {16, 1, 2, 1});
    auto lseWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(lseOld.data(), lseOld.size());
    EXPECT_FALSE(bf16Builder.isApplicable(lseWrap.getNode(0), lseWrap.getTensorMap()));
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForDtypeMismatch)
{
    auto graphBuilder = makeRaggedGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    // A half builder must not be applicable to a bf16 ragged graph.
    const GpuSdpaRaggedFwdPlanBuilder<DataType::HALF,
                                      DataType::HALF,
                                      DataType::HALF,
                                      DataType::HALF>
        halfBuilder;
    EXPECT_FALSE(halfBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // A missing input tensor must make the plan inapplicable.
    const Bf16Builder bf16Builder;
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(K_UID);
    EXPECT_FALSE(bf16Builder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForUnsupportedModes)
{
    const Bf16Builder bf16Builder;

    const auto isApplicableWith = [&](const SdpaAttributesT& attrs) {
        auto graphBuilder = makeRaggedGraph(attrs);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        return bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
    };

    {
        // ragged_offset with seq_len (the padded variant) is not supported.
        SdpaAttributesT attrs;
        attrs.seq_len_q_tensor_uid = UNUSED_UID;
        attrs.seq_len_kv_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.alibi_mask = true;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.padding_mask = true;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // Additive bias is gated off on the ASM v3 path.
        SdpaAttributesT attrs;
        attrs.attn_mask_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.dropout_probability = 0.1F;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.page_table_k_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.block_mask_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // No softmax requant: AITER fp8 forward descales only Q/K/V.
        SdpaAttributesT attrs;
        attrs.descale_s_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // The reference does not produce the running-max softmax stat.
        SdpaAttributesT attrs;
        attrs.max_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, PlanConstruction)
{
    auto graphBuilder = makeRaggedGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Bf16Builder bf16Builder;
    auto builtPlan = bf16Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    // Cast outside EXPECT_NE: the template's commas would split the macro arguments.
    auto* casted
        = dynamic_cast<GpuSdpaRaggedFwdPlan<bfloat16, bfloat16, bfloat16, bfloat16, float>*>(
            builtPlan.get());
    EXPECT_NE(casted, nullptr);
}

// The plan must run the same kernel as a direct fpropRagged call, including the LSE. Equal
// lengths leave no padding, so whole tensors are compared. ExecuteUsesBfloat16ProbabilityMode
// checks the probability mode, which this tolerance cannot tell apart.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteMatchesDirectFpropRaggedBf16)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;

    const int64_t batch = 2;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4; // equal per batch -> no padding
    const int64_t headDim = 16;
    const auto qkvDims = raggedDims(batch, seqLen, numHeads, headDim);
    const auto lseDims = raggedDims(batch, seqLen, numHeads, 1);
    const int64_t seqStride = numHeads * headDim;

    RaggedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID; // default dense stats layout
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 DataType::BFLOAT16,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Bf16Builder bf16Builder;
    auto plan = bf16Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<bfloat16> q(qkvDims, raggedStrides(qkvDims));
    Tensor<bfloat16> k(qkvDims, raggedStrides(qkvDims));
    Tensor<bfloat16> v(qkvDims, raggedStrides(qkvDims));
    q.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/11);
    k.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/22);
    v.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/33);
    auto offQ = makeRaggedOffset({seqLen, seqLen}, seqStride);
    auto offKv = makeRaggedOffset({seqLen, seqLen}, seqStride);

    Tensor<bfloat16> oPlan(qkvDims, raggedStrides(qkvDims));
    Tensor<float> lsePlan(lseDims); // dense, matching the graph's stats strides
    lsePlan.fillWithValue(-987.0f); // sentinel: an unwritten LSE would retain this

    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
        {STATS_UID, lsePlan.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();
    lsePlan.markDeviceModified();

    // Direct reference with the same probability mode the plan selects for all-bf16.
    Tensor<bfloat16> oDirect(qkvDims, raggedStrides(qkvDims));
    Tensor<float> lseDirect(lseDims);
    GpuFpReferenceSdpaRagged::fpropRagged<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        q,
        k,
        v,
        oDirect,
        offQ,
        offKv,
        offKv,
        offQ,
        std::nullopt,
        /*leftBound=*/-1,
        /*rightBound=*/-1,
        /*topLeftAlignment=*/true,
        &lseDirect,
        /*raggedOffsetLse=*/nullptr,
        sdpaProbabilityMode<bfloat16, bfloat16, bfloat16, bfloat16>());

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> oValidation(tolerance, tolerance);
    EXPECT_TRUE(oValidation.allClose(oDirect, oPlan))
        << "Plan output differs from direct fpropRagged output";
    const CpuFpReferenceValidation<float> lseValidation(tolerance, tolerance);
    EXPECT_TRUE(lseValidation.allClose(lseDirect, lsePlan))
        << "Plan LSE differs from direct fpropRagged LSE";
}

// All-bf16 graphs round P to bf16 (RTNE) before P@V, as AITER does. Large V values make that
// rounding visible: the FLOAT and BFLOAT16_RTNE outputs differ, and the plan must match RTNE
// exactly. One query token against four keys, H = D = 1, and the default scale of 1.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteUsesBfloat16ProbabilityMode)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;
    using hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode;

    const auto qDims = raggedDims(1, 1, 1, 1);
    const auto kvDims = raggedDims(1, 4, 1, 1);
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 /*batch=*/1,
                                                 qDims,
                                                 kvDims,
                                                 kvDims,
                                                 qDims,
                                                 DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Bf16Builder bf16Builder;
    auto plan = bf16Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<bfloat16> q(qDims, raggedStrides(qDims));
    Tensor<bfloat16> k(kvDims, raggedStrides(kvDims));
    Tensor<bfloat16> v(kvDims, raggedStrides(kvDims));
    q.memory().hostData()[0] = bfloat16(1.0f);
    const std::vector<float> kValues = {2.0f, 0.0f, 1.0f, 0.0f};
    const std::vector<float> vValues = {1000.0f, -1000.0f, 500.0f, 1000.0f};
    for(size_t i = 0; i < kValues.size(); ++i)
    {
        k.memory().hostData()[i] = bfloat16(kValues[i]);
        v.memory().hostData()[i] = bfloat16(vValues[i]);
    }
    auto offQ = makeRaggedOffset({1}, /*seqStride=*/1);
    auto offKv = makeRaggedOffset({4}, /*seqStride=*/1);

    Tensor<bfloat16> oPlan(qDims, raggedStrides(qDims));
    plan->execute({
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
    });
    oPlan.markDeviceModified();

    const auto direct = [&](SdpaSoftmaxProbabilityMode mode) {
        Tensor<bfloat16> o(qDims, raggedStrides(qDims));
        GpuFpReferenceSdpaRagged::fpropRagged<bfloat16, bfloat16, bfloat16, bfloat16, float>(
            q,
            k,
            v,
            o,
            offQ,
            offKv,
            offKv,
            offQ,
            std::nullopt,
            -1,
            -1,
            true,
            nullptr,
            nullptr,
            mode);
        return static_cast<float>(o.memory().hostData()[0]);
    };
    const float oRtne = direct(SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE);
    const float oFloat = direct(SdpaSoftmaxProbabilityMode::FLOAT);

    ASSERT_NE(oRtne, oFloat);
    EXPECT_EQ(static_cast<float>(oPlan.memory().hostData()[0]), oRtne);
}

// fp8: the plan must pass the Q/K/V descales from the variant pack through to fpropRagged.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteFp8MatchesDirectFpropRagged)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;

    constexpr int64_t DESCALE_Q_UID = 30;
    constexpr int64_t DESCALE_K_UID = 31;
    constexpr int64_t DESCALE_V_UID = 32;

    const int64_t batch = 2;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4; // equal per batch -> no padding
    const int64_t headDim = 128;
    const auto qkvDims = raggedDims(batch, seqLen, numHeads, headDim);
    const int64_t seqStride = numHeads * headDim;

    RaggedSdpaFwdGraphOptions options;
    options.descaleQ = FloatOperandSpec{DESCALE_Q_UID};
    options.descaleK = FloatOperandSpec{DESCALE_K_UID};
    options.descaleV = FloatOperandSpec{DESCALE_V_UID};
    options.oDataType = DataType::BFLOAT16;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 DataType::FP8_E4M3,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaRaggedFwdPlanBuilder<DataType::FP8_E4M3,
                                      DataType::FP8_E4M3,
                                      DataType::FP8_E4M3,
                                      DataType::BFLOAT16>
        fp8Builder;
    ASSERT_TRUE(fp8Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp8Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<fp8_e4m3> q(qkvDims, raggedStrides(qkvDims));
    Tensor<fp8_e4m3> k(qkvDims, raggedStrides(qkvDims));
    Tensor<fp8_e4m3> v(qkvDims, raggedStrides(qkvDims));
    q.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/11);
    k.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/22);
    v.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/33);
    auto offQ = makeRaggedOffset({seqLen, seqLen}, seqStride);
    auto offKv = makeRaggedOffset({seqLen, seqLen}, seqStride);

    Tensor<float> descaleQ({1});
    Tensor<float> descaleK({1});
    Tensor<float> descaleV({1});
    descaleQ.memory().hostData()[0] = 0.5f;
    descaleK.memory().hostData()[0] = 0.25f;
    descaleV.memory().hostData()[0] = 2.0f;
    descaleQ.memory().markHostModified();
    descaleK.memory().markHostModified();
    descaleV.memory().markHostModified();

    Tensor<bfloat16> oPlan(qkvDims, raggedStrides(qkvDims));
    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
        {DESCALE_Q_UID, descaleQ.memory().deviceData()},
        {DESCALE_K_UID, descaleK.memory().deviceData()},
        {DESCALE_V_UID, descaleV.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    Tensor<bfloat16> oDirect(qkvDims, raggedStrides(qkvDims));
    GpuFpReferenceSdpaRagged::fpropRagged<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
        q,
        k,
        v,
        oDirect,
        offQ,
        offKv,
        offKv,
        offQ,
        std::nullopt,
        -1,
        -1,
        true,
        nullptr,
        nullptr,
        hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::FLOAT,
        &descaleQ,
        &descaleK,
        &descaleV);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> validation(tolerance, tolerance);
    EXPECT_TRUE(validation.allClose(oDirect, oPlan))
        << "fp8 plan output differs from direct fpropRagged output";
}

namespace
{

using Fp8Builder = GpuSdpaRaggedFwdPlanBuilder<DataType::FP8_E4M3,
                                               DataType::FP8_E4M3,
                                               DataType::FP8_E4M3,
                                               DataType::BFLOAT16>;

// One fp8 descale. One value is a scalar [1]. B * H_kv values are a per-KV-head [B, H_kv, 1, 1]
// descale, DEVICE storage only.
struct DescaleCase
{
    OperandStorage storage;
    std::vector<float> values;
};

// Runs the fp8 plan with Q/K/V descales in the given storage modes and compares it with a direct
// fpropRagged call on device descales. BAKED descales are left out of the variant pack on purpose.
void checkFp8DescaleStorage(const DescaleCase& qCase,
                            const DescaleCase& kCase,
                            const DescaleCase& vCase)
{
    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;

    constexpr int64_t DESCALE_Q_UID = 30;
    constexpr int64_t DESCALE_K_UID = 31;
    constexpr int64_t DESCALE_V_UID = 32;

    const int64_t batch = 2;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4; // equal per batch -> no padding
    const int64_t headDim = 128;
    const auto qkvDims = raggedDims(batch, seqLen, numHeads, headDim);
    const int64_t seqStride = numHeads * headDim;

    const auto descaleDims = [&](const DescaleCase& c) {
        return c.values.size() == 1 ? std::vector<int64_t>{1}
                                    : std::vector<int64_t>{batch, numHeads, 1, 1};
    };
    const auto spec = [&](int64_t uid, const DescaleCase& c) {
        FloatOperandSpec s;
        s.uid = uid;
        s.storage = c.storage;
        s.bakedValue = c.values.front();
        s.dims = descaleDims(c);
        return s;
    };

    RaggedSdpaFwdGraphOptions options;
    options.descaleQ = spec(DESCALE_Q_UID, qCase);
    options.descaleK = spec(DESCALE_K_UID, kCase);
    options.descaleV = spec(DESCALE_V_UID, vCase);
    options.oDataType = DataType::BFLOAT16;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 DataType::FP8_E4M3,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Fp8Builder fp8Builder;
    ASSERT_TRUE(fp8Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp8Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<fp8_e4m3> q(qkvDims, raggedStrides(qkvDims));
    Tensor<fp8_e4m3> k(qkvDims, raggedStrides(qkvDims));
    Tensor<fp8_e4m3> v(qkvDims, raggedStrides(qkvDims));
    q.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/11);
    k.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/22);
    v.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/33);
    auto offQ = makeRaggedOffset({seqLen, seqLen}, seqStride);
    auto offKv = makeRaggedOffset({seqLen, seqLen}, seqStride);

    // Device copies feed the direct reference and any DEVICE plan operands.
    const auto makeDeviceDescale = [&](const DescaleCase& c) {
        Tensor<float> t(descaleDims(c));
        std::copy(c.values.begin(), c.values.end(), t.memory().hostData());
        t.memory().markHostModified();
        return t;
    };
    auto descaleQ = makeDeviceDescale(qCase);
    auto descaleK = makeDeviceDescale(kCase);
    auto descaleV = makeDeviceDescale(vCase);
    float hostQ = qCase.values.front();
    float hostK = kCase.values.front();
    float hostV = vCase.values.front();

    Tensor<bfloat16> oPlan(qkvDims, raggedStrides(qkvDims));
    std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
    };
    const auto bind = [&](int64_t uid, const DescaleCase& c, Tensor<float>& device, float& host) {
        if(c.storage == OperandStorage::DEVICE)
        {
            variantPack.emplace(uid, device.memory().deviceData());
        }
        else if(c.storage == OperandStorage::RUNTIME_PASS_BY_VALUE)
        {
            variantPack.emplace(uid, &host);
        }
    };
    bind(DESCALE_Q_UID, qCase, descaleQ, hostQ);
    bind(DESCALE_K_UID, kCase, descaleK, hostK);
    bind(DESCALE_V_UID, vCase, descaleV, hostV);
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    Tensor<bfloat16> oDirect(qkvDims, raggedStrides(qkvDims));
    GpuFpReferenceSdpaRagged::fpropRagged<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
        q,
        k,
        v,
        oDirect,
        offQ,
        offKv,
        offKv,
        offQ,
        std::nullopt,
        -1,
        -1,
        true,
        nullptr,
        nullptr,
        hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::FLOAT,
        &descaleQ,
        &descaleK,
        &descaleV);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> validation(tolerance, tolerance);
    EXPECT_TRUE(validation.allClose(oDirect, oPlan))
        << "fp8 plan output differs from direct fpropRagged output";
}

} // namespace

// Reviewer repro: a baked Q descale lives in the graph, not the variant pack.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteFp8BakedDescaleNotInVariantPack)
{
    SKIP_IF_NO_DEVICES();
    checkFp8DescaleStorage({OperandStorage::BAKED, {0.5f}},
                           {OperandStorage::DEVICE, {0.25f}},
                           {OperandStorage::DEVICE, {2.0f}});
}

TEST(TestGpuSdpaRaggedFwdPlan, ExecuteFp8RuntimePassByValueDescales)
{
    SKIP_IF_NO_DEVICES();
    checkFp8DescaleStorage({OperandStorage::RUNTIME_PASS_BY_VALUE, {0.5f}},
                           {OperandStorage::RUNTIME_PASS_BY_VALUE, {0.25f}},
                           {OperandStorage::RUNTIME_PASS_BY_VALUE, {2.0f}});
}

// Every storage mode in one graph, with a per-KV-head [B, H_kv, 1, 1] device V descale.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteFp8MixedDescaleStorage)
{
    SKIP_IF_NO_DEVICES();
    checkFp8DescaleStorage({OperandStorage::BAKED, {0.5f}},
                           {OperandStorage::RUNTIME_PASS_BY_VALUE, {0.25f}},
                           {OperandStorage::DEVICE, {1.5f, 2.0f, 2.5f, 3.0f}});
}

// Host-stored descales (baked or pass-by-value) must be scalars.
TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonScalarHostDescale)
{
    for(const auto storage : {OperandStorage::BAKED, OperandStorage::RUNTIME_PASS_BY_VALUE})
    {
        FloatOperandSpec descale;
        descale.uid = 30;
        descale.storage = storage;
        descale.bakedValue = 0.5f;
        descale.dims = {2, 2, 1, 1};
        RaggedSdpaFwdGraphOptions options;
        options.descaleQ = descale;
        options.descaleK = FloatOperandSpec{31};
        options.descaleV = FloatOperandSpec{32};
        options.oDataType = DataType::BFLOAT16;
        const auto dims = raggedDims(2, 4, 2, 128);
        auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                     K_UID,
                                                     V_UID,
                                                     O_UID,
                                                     RAGGED_OFFSET_Q_UID,
                                                     RAGGED_OFFSET_KV_UID,
                                                     /*batch=*/2,
                                                     dims,
                                                     dims,
                                                     dims,
                                                     dims,
                                                     DataType::FP8_E4M3,
                                                     options);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        const Fp8Builder fp8Builder;
        EXPECT_FALSE(fp8Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }
}

// Dense frontend-default stats with unequal lengths: batch b's LSE rows start at b * Sq_max * H,
// not at its first packed Q token.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteDenseStatsUnequalLengthsMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    checkPlanLseAgainstCpu(RaggedStatsLayout::DENSE);
}

// Packed stats carrying their own ragged_offset aux.
TEST(TestGpuSdpaRaggedFwdPlan, ExecutePackedStatsUnequalLengthsMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    checkPlanLseAgainstCpu(RaggedStatsLayout::PACKED);
}

// AITER's form: token offset tables with ragged_offset_multiplier = seq stride on every ragged
// tensor, including the packed stats.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteTokenOffsetsMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    checkPlanLseAgainstCpu(RaggedStatsLayout::PACKED, /*tokenOffsets=*/true);
}

// RFC-0014 requires ragged_offset_multiplier >= 1; the plan declines anything else.
TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForZeroOffsetMultiplier)
{
    auto graphBuilder = makeRaggedGraph();
    auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuilder.GetBufferPointer())->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        if(tensor->uid == K_UID)
        {
            tensor->ragged_offset_multiplier = 0;
        }
    }
    flatbuffers::FlatBufferBuilder rewritten;
    rewritten.Finish(CreateGraph(rewritten, graphT.get()));
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        rewritten.GetBufferPointer(), rewritten.GetSize());

    const Bf16Builder bf16Builder;
    EXPECT_FALSE(bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

// V has its own ragged_offset (allowed by RFC-0014). V lengths that differ from K's must be
// rejected and matching ones must run.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteRejectsKvSequenceLengthMismatch)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t RAGGED_OFFSET_V_UID = 23;
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const int64_t seqStride = numHeads * headDim;
    const auto dims = raggedDims(2, 2, numHeads, headDim);

    RaggedSdpaFwdGraphOptions options;
    options.raggedOffsetVUid = RAGGED_OFFSET_V_UID;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 /*batch=*/2,
                                                 dims,
                                                 dims,
                                                 dims,
                                                 dims,
                                                 DataType::FLOAT,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Fp32Builder fp32Builder;
    ASSERT_TRUE(fp32Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp32Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    Tensor<float> o(dims, raggedStrides(dims));
    q.fillWithValue(0.0f);
    k.fillWithValue(0.0f);
    v.fillWithValue(1.0f);
    auto offQ = makeRaggedOffset({1, 1}, seqStride);
    auto offK = makeRaggedOffset({2, 1}, seqStride);
    auto offVMismatch = makeRaggedOffset({1, 2}, seqStride);
    auto offVMatch = makeRaggedOffset({2, 1}, seqStride);

    const auto variantPackWithV = [&](Tensor<int32_t>& offV) {
        return std::unordered_map<int64_t, void*>{
            {Q_UID, q.memory().deviceData()},
            {K_UID, k.memory().deviceData()},
            {V_UID, v.memory().deviceData()},
            {O_UID, o.memory().deviceData()},
            {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
            {RAGGED_OFFSET_KV_UID, offK.memory().deviceData()},
            {RAGGED_OFFSET_V_UID, offV.memory().deviceData()},
        };
    };
    EXPECT_THROW(plan->execute(variantPackWithV(offVMismatch)), std::invalid_argument);
    EXPECT_NO_THROW(plan->execute(variantPackWithV(offVMatch)));
}

namespace
{

constexpr int64_t SCALE_UID = 40;

// Where the attention scale of a ragged SDPA graph comes from.
enum class ScaleSource
{
    ATTR_VALUE, // attrs.attn_scale_value (no scale tensor)
    DEVICE_TENSOR, // scale_tensor_uid, device-resident
    BAKED_TENSOR, // scale_tensor_uid, value stored in the graph (absent from the variant pack)
    RUNTIME_PASS_BY_VALUE, // scale_tensor_uid, host pointer in the variant pack
};

std::string scaleSourceName(const ::testing::TestParamInfo<ScaleSource>& info)
{
    switch(info.param)
    {
    case ScaleSource::ATTR_VALUE:
        return "AttrValue";
    case ScaleSource::DEVICE_TENSOR:
        return "DeviceTensor";
    case ScaleSource::BAKED_TENSOR:
        return "BakedTensor";
    case ScaleSource::RUNTIME_PASS_BY_VALUE:
        return "RuntimePassByValue";
    default:
        return "Unknown";
    }
}

class TestGpuSdpaRaggedFwdPlanScale : public ::testing::TestWithParam<ScaleSource>
{
};

} // namespace

// A non-default scale must reach the kernel in every storage mode. Comparing against the
// default-scale output proves the inputs are sensitive to the scale.
TEST_P(TestGpuSdpaRaggedFwdPlanScale, ExecuteHonorsNonDefaultScale)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;

    constexpr float SCALE = 0.9f; // default for D = 16 is 0.25
    const std::vector<int64_t> seqLens = {3, 5};
    const int64_t batch = 2;
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const int64_t seqStride = numHeads * headDim;
    const int64_t totalQ = 8;
    const auto dims = raggedDims(batch, 5, numHeads, headDim);
    const auto source = GetParam();

    RaggedSdpaFwdGraphOptions options;
    if(source == ScaleSource::ATTR_VALUE)
    {
        options.attrs.attn_scale_value = SCALE;
    }
    else
    {
        FloatOperandSpec scale;
        scale.uid = SCALE_UID;
        scale.storage = OperandStorage::RUNTIME_PASS_BY_VALUE;
        if(source == ScaleSource::DEVICE_TENSOR)
        {
            scale.storage = OperandStorage::DEVICE;
        }
        else if(source == ScaleSource::BAKED_TENSOR)
        {
            scale.storage = OperandStorage::BAKED;
        }
        scale.bakedValue = SCALE;
        options.scale = scale;
    }
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 dims,
                                                 dims,
                                                 dims,
                                                 dims,
                                                 DataType::FLOAT,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Fp32Builder fp32Builder;
    ASSERT_TRUE(fp32Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp32Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(dims, raggedStrides(dims));
    Tensor<float> k(dims, raggedStrides(dims));
    Tensor<float> v(dims, raggedStrides(dims));
    q.fillWithRandomValues(-2.0f, 2.0f, /*seed=*/11);
    k.fillWithRandomValues(-2.0f, 2.0f, /*seed=*/22);
    v.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/33);
    auto offQ = makeRaggedOffset(seqLens, seqStride);
    auto offKv = makeRaggedOffset(seqLens, seqStride);

    Tensor<float> scaleDevice({1});
    scaleDevice.fillWithValue(SCALE);
    float scaleHost = SCALE;

    Tensor<float> oPlan(dims, raggedStrides(dims));
    std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
    };
    if(source == ScaleSource::DEVICE_TENSOR)
    {
        variantPack.emplace(SCALE_UID, scaleDevice.memory().deviceData());
    }
    else if(source == ScaleSource::RUNTIME_PASS_BY_VALUE)
    {
        variantPack.emplace(SCALE_UID, &scaleHost);
    }
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    const auto runDirect = [&](std::optional<float> scale) {
        Tensor<float> o(dims, raggedStrides(dims));
        GpuFpReferenceSdpaRagged::fpropRagged<float, float, float, float, float>(
            q, k, v, o, offQ, offKv, offKv, offQ, scale);
        return o;
    };
    auto oExpected = runDirect(SCALE);
    auto oDefault = runDirect(std::nullopt);

    const auto* plan0 = oPlan.memory().hostData();
    const auto* expected = oExpected.memory().hostData();
    const auto* fallback = oDefault.memory().hostData();
    float maxScaleEffect = 0.0f;
    for(int64_t i = 0; i < totalQ * seqStride; ++i) // packed region only
    {
        EXPECT_NEAR(plan0[i], expected[i], 1e-5f) << "output mismatch at element " << i;
        maxScaleEffect = std::max(maxScaleEffect, std::abs(expected[i] - fallback[i]));
    }
    EXPECT_GT(maxScaleEffect, 1e-2f) << "inputs too uniform: the scale does not affect the output";
}

INSTANTIATE_TEST_SUITE_P(ScaleSources,
                         TestGpuSdpaRaggedFwdPlanScale,
                         ::testing::Values(ScaleSource::ATTR_VALUE,
                                           ScaleSource::DEVICE_TENSOR,
                                           ScaleSource::BAKED_TENSOR,
                                           ScaleSource::RUNTIME_PASS_BY_VALUE),
                         scaleSourceName);

// --- Mask attributes through the plan ---

namespace
{

// Runs the fp32 plan on a graph with the given mask attributes and checks it against the CPU
// ragged reference with the bounds and alignment the attributes should resolve to. The lengths
// include Sq > Skv and Sq < Skv batches, so top-left and bottom-right give different outputs.
void checkPlanMaskAgainstCpu(const SdpaAttributesT& attrs,
                             int64_t expectedLeftBound,
                             int64_t expectedRightBound,
                             bool expectedTopLeft)
{
    const std::vector<int64_t> seqQ = {5, 2, 4};
    const std::vector<int64_t> seqKv = {2, 5, 4};
    const int64_t batch = 3;
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const int64_t seqStride = numHeads * headDim;
    const int64_t totalQ = 11;
    const auto qDims = raggedDims(batch, 5, numHeads, headDim);
    const auto kvDims = raggedDims(batch, 5, numHeads, headDim);

    RaggedSdpaFwdGraphOptions options;
    options.attrs = attrs;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qDims,
                                                 kvDims,
                                                 kvDims,
                                                 qDims,
                                                 DataType::FLOAT,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Fp32Builder fp32Builder;
    ASSERT_TRUE(fp32Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp32Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(qDims, raggedStrides(qDims));
    Tensor<float> k(kvDims, raggedStrides(kvDims));
    Tensor<float> v(kvDims, raggedStrides(kvDims));
    q.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/11);
    k.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/22);
    v.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/33);
    auto offQ = makeRaggedOffset(seqQ, seqStride);
    auto offKv = makeRaggedOffset(seqKv, seqStride);
    Tensor<float> oPlan(qDims, raggedStrides(qDims));
    Tensor<float> oCpu(qDims, raggedStrides(qDims));

    plan->execute({
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
    });
    oPlan.markDeviceModified();

    {
        auto qR = wrapRagged(q.memory().hostData(), qDims, seqStride, seqQ);
        auto kR = wrapRagged(k.memory().hostData(), kvDims, seqStride, seqKv);
        auto vR = wrapRagged(v.memory().hostData(), kvDims, seqStride, seqKv);
        auto oR = wrapRagged(oCpu.memory().hostData(), qDims, seqStride, seqQ);
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
            qR, kR, vR, oR, std::nullopt, expectedLeftBound, expectedRightBound, expectedTopLeft);
    }

    const auto* oPlanHost = oPlan.memory().hostData();
    const auto* oCpuHost = oCpu.memory().hostData();
    for(int64_t i = 0; i < totalQ * seqStride; ++i)
    {
        EXPECT_NEAR(oPlanHost[i], oCpuHost[i], 1e-4f) << "output mismatch at element " << i;
    }
}

} // namespace

// Deprecated causal_mask_bottom_right resolves to left = -1, right = 0, bottom-right alignment.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteCausalBottomRightFlagMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    SdpaAttributesT attrs;
    attrs.causal_mask_bottom_right = true;
    checkPlanMaskAgainstCpu(attrs, -1, 0, /*expectedTopLeft=*/false);
}

// Deprecated causal_mask resolves to top-left causal.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteCausalTopLeftFlagMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    SdpaAttributesT attrs;
    attrs.causal_mask = true;
    checkPlanMaskAgainstCpu(attrs, -1, 0, /*expectedTopLeft=*/true);
}

// Explicit bounds with bottom-right diagonal_alignment reach the kernel unchanged.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteBottomRightBandMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    SdpaAttributesT attrs;
    attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    attrs.left_bound = 1;
    attrs.right_bound = 0;
    checkPlanMaskAgainstCpu(attrs, 1, 0, /*expectedTopLeft=*/false);
}

// Both deprecated causal flags, or a bound below -1, are rejected when the plan is built.
TEST(TestGpuSdpaRaggedFwdPlanBuilder, BuildRejectsInvalidMaskAttributes)
{
    const Bf16Builder bf16Builder;

    SdpaAttributesT bothCausal;
    bothCausal.causal_mask = true;
    bothCausal.causal_mask_bottom_right = true;
    auto bothCausalGraph = makeRaggedGraph(bothCausal);
    auto bothCausalWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        bothCausalGraph.GetBufferPointer(), bothCausalGraph.GetSize());
    EXPECT_THROW(bf16Builder.buildNodePlan(bothCausalWrap, bothCausalWrap.getNode(0)),
                 std::invalid_argument);

    SdpaAttributesT badBound;
    badBound.left_bound = -2;
    auto badBoundGraph = makeRaggedGraph(badBound);
    auto badBoundWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        badBoundGraph.GetBufferPointer(), badBoundGraph.GetSize());
    EXPECT_THROW(bf16Builder.buildNodePlan(badBoundWrap, badBoundWrap.getNode(0)),
                 std::invalid_argument);

    SdpaAttributesT badRightBound;
    badRightBound.right_bound = -2;
    auto badRightGraph = makeRaggedGraph(badRightBound);
    auto badRightWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        badRightGraph.GetBufferPointer(), badRightGraph.GetSize());
    EXPECT_THROW(bf16Builder.buildNodePlan(badRightWrap, badRightWrap.getNode(0)),
                 std::invalid_argument);
}
