// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
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

// A uid intentionally absent from the graph's tensor map; used to set unsupported-mode optional
// uids whose mere presence must make the plan inapplicable.
constexpr int64_t UNUSED_UID = 99;

// Packed rank-4 [B=1, H=2, S=8, D=16] for a single batch (shape irrelevant to applicability).
const std::vector<int64_t> DIMS = {1, 2, 8, 16};

using Bf16Builder = GpuSdpaRaggedFwdPlanBuilder<DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16>;

// Build a bf16 ragged graph (ragged_offset on all primaries) with optional extra attrs.
flatbuffers::FlatBufferBuilder makeRaggedGraph(SdpaAttributesT attrs = {})
{
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
                                    attrs);
}

// BSHD-layout element strides for packed rank-4 [B, H, S, D]: seq stride = strides[2] = H*D.
std::vector<int64_t> bshd(const std::vector<int64_t>& dims)
{
    return {dims[1] * dims[2] * dims[3], dims[3], dims[1] * dims[3], 1};
}

// ragged_offset aux [B+1,1,1,1] INT32 = cumTokens * seqStride (element offsets).
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
    // No ragged_offset on the primaries: a dense node belongs to the dense plan, not the ragged one.
    auto graphBuilder = createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, DIMS, DIMS, DIMS, DIMS, DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Bf16Builder bf16Builder;
    EXPECT_FALSE(bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
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
        // The padded seq-lens variant (ragged_offset + seq_len) is out of scope here.
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
        // Softmax/output (re)quantization is unsupported (AITER fp8 fwd descales Q/K/V only).
        SdpaAttributesT attrs;
        attrs.descale_s_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // max_tensor_uid (running max softmax stat) is not produced by the reference.
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

    // Parenthesize the cast: the template's commas would otherwise be parsed as macro args.
    auto* casted
        = dynamic_cast<GpuSdpaRaggedFwdPlan<bfloat16, bfloat16, bfloat16, bfloat16, float>*>(
            builtPlan.get());
    EXPECT_NE(casted, nullptr);
}

// Wiring proof: plan execute() drives the same kernel as a direct fpropRagged call (including the
// bf16 provider probability mode and the LSE output), so identical inputs must produce identical
// output through the graph path. Equal per-batch lengths keep prod(dims) == packed, so the padded
// tensors are fully initialized and can be compared in full.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteMatchesDirectFpropRaggedBf16)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;

    const int64_t batch = 2;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4; // equal per batch -> no padding
    const int64_t headDim = 16;
    const std::vector<int64_t> qkvDims = {batch, numHeads, seqLen, headDim};
    const std::vector<int64_t> lseDims = {batch, numHeads, seqLen, 1};
    const int64_t seqStride = numHeads * headDim;

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
                                                 /*attrs=*/{},
                                                 /*statsUid=*/STATS_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Bf16Builder bf16Builder;
    auto plan = bf16Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<bfloat16> q(qkvDims, bshd(qkvDims));
    Tensor<bfloat16> k(qkvDims, bshd(qkvDims));
    Tensor<bfloat16> v(qkvDims, bshd(qkvDims));
    q.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/11);
    k.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/22);
    v.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/33);
    auto offQ = makeRaggedOffset({seqLen, seqLen}, seqStride);
    auto offKv = makeRaggedOffset({seqLen, seqLen}, seqStride);

    Tensor<bfloat16> oPlan(qkvDims, bshd(qkvDims));
    Tensor<float> lsePlan(lseDims, bshd(lseDims));
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
    Tensor<bfloat16> oDirect(qkvDims, bshd(qkvDims));
    Tensor<float> lseDirect(lseDims, bshd(lseDims));
    GpuFpReferenceSdpaRagged::fpropRagged<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        q,
        k,
        v,
        oDirect,
        offQ,
        offKv,
        std::nullopt,
        /*leftBound=*/-1,
        /*rightBound=*/-1,
        /*topLeftAlignment=*/true,
        &lseDirect,
        sdpaProbabilityMode<bfloat16, bfloat16, bfloat16, bfloat16>());

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> oValidation(tolerance, tolerance);
    EXPECT_TRUE(oValidation.allClose(oDirect, oPlan))
        << "Plan output differs from direct fpropRagged output";
    const CpuFpReferenceValidation<float> lseValidation(tolerance, tolerance);
    EXPECT_TRUE(lseValidation.allClose(lseDirect, lsePlan))
        << "Plan LSE differs from direct fpropRagged LSE";
}

// fp8 graph path: the plan must resolve the fp8 Q/K/V descale tensors from the variant pack and
// hand them to fpropRagged. Validated plan-vs-direct with identical inputs, descale, and mode.
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
    const std::vector<int64_t> qkvDims = {batch, numHeads, seqLen, headDim};
    const int64_t seqStride = numHeads * headDim;

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
                                                 /*attrs=*/{},
                                                 /*statsUid=*/std::nullopt,
                                                 /*descaleQUid=*/DESCALE_Q_UID,
                                                 /*descaleKUid=*/DESCALE_K_UID,
                                                 /*descaleVUid=*/DESCALE_V_UID,
                                                 /*oDataType=*/DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaRaggedFwdPlanBuilder<DataType::FP8_E4M3,
                                      DataType::FP8_E4M3,
                                      DataType::FP8_E4M3,
                                      DataType::BFLOAT16>
        fp8Builder;
    ASSERT_TRUE(fp8Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp8Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<fp8_e4m3> q(qkvDims, bshd(qkvDims));
    Tensor<fp8_e4m3> k(qkvDims, bshd(qkvDims));
    Tensor<fp8_e4m3> v(qkvDims, bshd(qkvDims));
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

    Tensor<bfloat16> oPlan(qkvDims, bshd(qkvDims));
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

    Tensor<bfloat16> oDirect(qkvDims, bshd(qkvDims));
    GpuFpReferenceSdpaRagged::fpropRagged<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
        q,
        k,
        v,
        oDirect,
        offQ,
        offKv,
        std::nullopt,
        -1,
        -1,
        true,
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
