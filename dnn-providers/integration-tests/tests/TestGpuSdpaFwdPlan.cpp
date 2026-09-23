// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaFwdPlan.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

namespace
{

constexpr int64_t Q_UID = 10;
constexpr int64_t K_UID = 11;
constexpr int64_t V_UID = 12;
constexpr int64_t O_UID = 13;

// A uid that is intentionally absent from the graph's tensor map; used to set
// unsupported-mode optional uids whose mere presence must make the plan inapplicable.
constexpr int64_t UNUSED_UID = 99;

// A valid stats/LSE output uid, distinct from the q/k/v/o uids.
constexpr int64_t STATS_UID = 14;

// Plain [B=1, H=2, Sq=8, D=16] SDPA shape (head_dim_v = 16).
const std::vector<int64_t> DIMS = {1, 2, 8, 16};

flatbuffers::FlatBufferBuilder makeGraph(SdpaAttributesT attrs = {},
                                         std::optional<int64_t> statsUid = std::nullopt)
{
    return createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, DIMS, DIMS, DIMS, DIMS, DataType::FLOAT, attrs, statsUid);
}

} // namespace

TEST(TestGpuSdpaFwdPlanBuilder, PlanConstruction)
{
    auto graphBuilder = makeGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<GpuSdpaFwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuSdpaFwdPlanBuilder, IsApplicable)
{
    auto graphBuilder = makeGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // A half builder must not be applicable to a float graph.
    const GpuSdpaFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // A missing input tensor must make the plan inapplicable.
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(K_UID);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuSdpaFwdPlanBuilder, IsNotApplicableForUnsupportedModes)
{
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    const auto isApplicableWith = [&](const SdpaAttributesT& attrs) {
        auto graphBuilder = makeGraph(attrs);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        return floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
    };

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
        SdpaAttributesT attrs;
        attrs.dropout_probability = 0.1F;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.seq_len_q_tensor_uid = UNUSED_UID;
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
        SdpaAttributesT attrs;
        attrs.descale_q_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // max_tensor_uid (running max softmax stat) is not produced by the reference.
        SdpaAttributesT attrs;
        attrs.max_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // sum_exp_tensor_uid (running sum-exp softmax stat) is not produced by the reference.
        SdpaAttributesT attrs;
        attrs.sum_exp_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
}

TEST(TestGpuSdpaFwdPlanBuilder, IsApplicableWithStatsOutput)
{
    // A graph with a valid FLOAT stats/LSE output tensor is supported: the plan must be
    // applicable and buildNodePlan must produce the concrete GpuSdpaFwdPlan.
    auto graphBuilder = makeGraph(/*attrs=*/{}, /*statsUid=*/STATS_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto builtPlan = floatPlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));
    const bool result
        = dynamic_cast<GpuSdpaFwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

// End-to-end device test of the LSE graph path: the graph declares a rank-4 [B, H, Sq, 1]
// stats output, and execute() must squeeze it to rank-3, bind it from the variant pack, and
// have the kernel write LSE into it. This is the path the unit-level LSE tests (which call
// fprop() directly with a rank-3 lse) do not cover. Validates against a direct fprop() call
// with identical inputs; the LSE buffer is pre-filled with a sentinel so an unwritten output
// would not accidentally pass.
TEST(TestGpuSdpaFwdPlanBuilder, ExecuteWritesLseThroughGraph)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_gpu_ref::GpuFpReferenceSdpa;

    const int64_t batch = DIMS[0];
    const int64_t numHeads = DIMS[1];
    const int64_t seqQ = DIMS[2];
    const std::vector<int64_t> lseDims = {batch, numHeads, seqQ};

    auto graphBuilder = makeGraph(/*attrs=*/{}, /*statsUid=*/STATS_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    auto plan = floatPlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(DIMS);
    Tensor<float> k(DIMS);
    Tensor<float> v(DIMS);
    q.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/11);
    k.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/22);
    v.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/33);

    Tensor<float> oPlan(DIMS);
    Tensor<float> lsePlan(lseDims);
    lsePlan.fillWithValue(-987.0f); // sentinel: an unwritten LSE would retain this

    // execute() consumes raw device pointers; deviceData() uploads the host inputs.
    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {STATS_UID, lsePlan.memory().deviceData()},
    };
    plan->execute(variantPack);
    // The plan wrote through its own shallow views; tell our tensors the device is now current.
    oPlan.markDeviceModified();
    lsePlan.markDeviceModified();

    // Reference: the same kernel via a direct fprop() with a rank-3 lse and identical inputs.
    Tensor<float> oRef(DIMS);
    Tensor<float> lseRef(lseDims);
    GpuFpReferenceSdpa::fprop<float, float, float, float, float>(q,
                                                                 k,
                                                                 v,
                                                                 oRef,
                                                                 std::nullopt,
                                                                 /*attnMask=*/nullptr,
                                                                 /*leftBound=*/-1,
                                                                 /*rightBound=*/-1,
                                                                 /*topLeftAlignment=*/true,
                                                                 &lseRef);

    // Same kernel + identical inputs, so plan and direct outputs match within a tight bound.
    const float tolerance = 1e-5f;
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(tolerance,
                                                                                 tolerance);
    EXPECT_TRUE(validation.allClose(oRef, oPlan)) << "Plan output differs from direct fprop output";
    EXPECT_TRUE(validation.allClose(lseRef, lsePlan))
        << "Plan LSE (via squeezed graph stats output) differs from direct fprop LSE";
}

TEST(TestGpuSdpaFwdPlanBuilder, ExecuteUsesBfloat16ProviderProbabilityMode)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_gpu_ref::GpuFpReferenceSdpa;
    using hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode;

    const std::vector<int64_t> qDims = {1, 1, 1, 1};
    const std::vector<int64_t> kvDims = {1, 1, 4, 1};
    const std::vector<int64_t> oDims = {1, 1, 1, 1};
    auto graphBuilder = createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, qDims, kvDims, kvDims, oDims, DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16>
        bfloat16PlanBuilder;
    auto plan = bfloat16PlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<bfloat16> q(qDims);
    Tensor<bfloat16> k(kvDims);
    Tensor<bfloat16> v(kvDims);
    q.memory().hostData()[0] = bfloat16(1.0f);
    k.memory().hostData()[0] = bfloat16(2.0f);
    k.memory().hostData()[1] = bfloat16(0.0f);
    k.memory().hostData()[2] = bfloat16(1.0f);
    k.memory().hostData()[3] = bfloat16(0.0f);
    v.memory().hostData()[0] = bfloat16(1000.0f);
    v.memory().hostData()[1] = bfloat16(-1000.0f);
    v.memory().hostData()[2] = bfloat16(500.0f);
    v.memory().hostData()[3] = bfloat16(1000.0f);

    Tensor<bfloat16> oPlan(oDims);
    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    Tensor<bfloat16> oProviderMode(oDims);
    GpuFpReferenceSdpa::fprop<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        q,
        k,
        v,
        oProviderMode,
        /*attnScaleValue=*/1.0f,
        /*attnMask=*/nullptr,
        /*leftBound=*/-1,
        /*rightBound=*/-1,
        /*topLeftAlignment=*/true,
        /*lse=*/nullptr,
        SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE);

    Tensor<bfloat16> oFloatMode(oDims);
    GpuFpReferenceSdpa::fprop<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        q,
        k,
        v,
        oFloatMode,
        /*attnScaleValue=*/1.0f);

    ASSERT_NE(static_cast<float>(oProviderMode.memory().hostData()[0]),
              static_cast<float>(oFloatMode.memory().hostData()[0]));
    EXPECT_EQ(static_cast<float>(oProviderMode.memory().hostData()[0]),
              static_cast<float>(oPlan.memory().hostData()[0]));
}

// Regression guard for the provider P->bf16 rounding gate. The cast is a property
// of the P@V matmul inputs (Q/K/V), not the output dtype, so the registered
// bf16-in/fp32-out config must still round P before P@V. Compile-time because
// sdpaProbabilityMode is constexpr; no device required.
TEST(TestGpuSdpaFwdPlanBuilder, ProbabilityModeKeyedOnInputsNotOutput)
{
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode;

    static_assert(sdpaProbabilityMode<bfloat16, bfloat16, bfloat16, bfloat16>()
                      == SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE,
                  "all-bf16 must round softmax probabilities to bf16 before P@V");
    static_assert(sdpaProbabilityMode<bfloat16, bfloat16, bfloat16, float>()
                      == SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE,
                  "bf16 inputs with fp32 output must still round P to bf16 before P@V");
    static_assert(sdpaProbabilityMode<float, float, float, float>()
                      == SdpaSoftmaxProbabilityMode::FLOAT,
                  "fp32 inputs must not round the softmax probabilities");
    static_assert(sdpaProbabilityMode<float, bfloat16, bfloat16, bfloat16>()
                      == SdpaSoftmaxProbabilityMode::FLOAT,
                  "a non-bf16 P@V input must not select the bf16 P-storage cast");

    SUCCEED();
}

TEST(TestGpuSdpaFwdPlanBuilder, ThrowsOnBothCausalFlags)
{
    SdpaAttributesT attrs;
    attrs.causal_mask = true;
    attrs.causal_mask_bottom_right = true;

    auto graphBuilder = makeGraph(attrs);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::invalid_argument);
}

TEST(TestGpuSdpaFwdPlanBuilder, ThrowsOnInvalidBounds)
{
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    {
        SdpaAttributesT attrs;
        attrs.left_bound = int64_t{-2};
        auto graphBuilder = makeGraph(attrs);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::invalid_argument);
    }
    {
        SdpaAttributesT attrs;
        attrs.right_bound = int64_t{-5};
        auto graphBuilder = makeGraph(attrs);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::invalid_argument);
    }
}
