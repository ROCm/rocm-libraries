// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FloatOperandTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/SdpaFwdTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "RaggedGraphTestUtils.hpp"
#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaFwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaFwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_gpu_ref::common::gpu_fp_reference_tensor;

using hipdnn_test_sdk::utilities::FloatOperandSpec;
using hipdnn_test_sdk::utilities::OperandStorage;

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

// A valid additive mask uid, distinct from the other uids.
constexpr int64_t MASK_UID = 15;

// Plain [B=1, H=2, Sq=8, D=16] SDPA shape (head_dim_v = 16).
const std::vector<int64_t> DIMS = {1, 2, 8, 16};

// Mask [1, 1, Sq=8, Skv=8], right-aligned and broadcast over batch and head.
const std::vector<int64_t> MASK_DIMS = {1, 1, 8, 8};

flatbuffers::FlatBufferBuilder makeGraph(SdpaAttributesT attrs = {},
                                         std::optional<int64_t> statsUid = std::nullopt)
{
    return createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, DIMS, DIMS, DIMS, DIMS, DataType::FLOAT, attrs, statsUid);
}

// FLOAT Q/K/V/O graph whose mask tensor (MASK_UID) has dtype `maskDataType`.
flatbuffers::FlatBufferBuilder makeGraphWithMask(DataType maskDataType)
{
    return createSdpaFwdGraph(Q_UID,
                              K_UID,
                              V_UID,
                              O_UID,
                              DIMS,
                              DIMS,
                              DIMS,
                              DIMS,
                              DataType::FLOAT,
                              {},
                              std::nullopt,
                              /*scale=*/std::nullopt,
                              MASK_UID,
                              MASK_DIMS,
                              maskDataType);
}

// Non-uniform values exactly representable in float, half and bfloat16: a uniform mask would
// pass by coincidence because softmax is shift-invariant.
template <typename MaskT>
void fillExactlyRepresentableMask(hipdnn_data_sdk::utilities::Tensor<MaskT>& mask)
{
    constexpr std::array<float, 8> VALUES = {-2.0f, -1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 1.0f};
    auto* data = mask.memory().hostData();
    for(size_t i = 0; i < mask.elementCount(); ++i)
    {
        data[i] = static_cast<MaskT>(VALUES[(i * 3 + i / 8) % VALUES.size()]);
    }
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

// The dense plan rejects ragged tensors; ragged nodes go to GpuSdpaRaggedFwdPlan.
TEST(TestGpuSdpaFwdPlanBuilder, IsNotApplicableForRaggedTensors)
{
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    auto graphBuilder = makeGraph();
    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    ASSERT_TRUE(floatPlanBuilder.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged = markFirstTensorRagged(graphBuilder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
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
        // A max_tensor_uid naming no tensor is not applicable (Max itself is supported).
        SdpaAttributesT attrs;
        attrs.max_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // A sum_exp_tensor_uid naming no tensor is not applicable (Sum_exp itself is supported).
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
    using hipdnn_gpu_ref::GpuSdpaFwdOptions;

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
    fillWithRandomValues(q, -1.0f, 1.0f, /*seed=*/11);
    fillWithRandomValues(k, -1.0f, 1.0f, /*seed=*/22);
    fillWithRandomValues(v, -1.0f, 1.0f, /*seed=*/33);

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
    GpuFpReferenceSdpa::fprop<float, float, float, float, float>(
        q, k, v, oRef, GpuSdpaFwdOptions<float>{}.setLse(&lseRef));

    // Same kernel + identical inputs, so plan and direct outputs match within a tight bound.
    const float tolerance = 1e-5f;
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(tolerance,
                                                                                 tolerance);
    EXPECT_TRUE(validation.allClose(oRef, oPlan)) << "Plan output differs from direct fprop output";
    EXPECT_TRUE(validation.allClose(lseRef, lsePlan))
        << "Plan LSE (via squeezed graph stats output) differs from direct fprop LSE";
}

// causal_mask=true plus left_bound=2 is a causal sliding window. The deprecated
// flag fixes the diagonal but must keep the left bound, the same rule as the CPU
// reference, so the plan equals a direct fprop() with (leftBound=2, rightBound=0,
// top-left). left_bound=2 keeps 2 keys per row, the diagonal included, as cuDNN's
// set_sliding_window_length(2) does. With Sq = Skv = 8 that window makes rows 2..7
// lose keys, so plain causal attention would not match.
TEST(TestGpuSdpaFwdPlanBuilder, DeprecatedCausalMaskWithLeftBoundIsSlidingWindow)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_gpu_ref::GpuFpReferenceSdpa;
    using hipdnn_gpu_ref::GpuSdpaFwdOptions;

    SdpaAttributesT attrs;
    attrs.causal_mask = true;
    attrs.left_bound = 2;
    auto graphBuilder = makeGraph(attrs);
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
    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    auto direct = [&](int64_t leftBound) {
        Tensor<float> out(DIMS);
        GpuFpReferenceSdpa::fprop<float, float, float, float, float>(
            q,
            k,
            v,
            out,
            GpuSdpaFwdOptions<float>{}.setBand(leftBound, /*right=*/0, /*topLeft=*/true));
        return out;
    };
    auto oWindow = direct(/*leftBound=*/2);
    auto oCausal = direct(/*leftBound=*/-1);

    const float tolerance = 1e-5f;
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(tolerance,
                                                                                 tolerance);
    EXPECT_TRUE(validation.allClose(oWindow, oPlan))
        << "causal_mask=true with left_bound=2 should run as leftBound=2, rightBound=0";
    // Control: the window must actually change the output.
    EXPECT_FALSE(validation.allClose(oCausal, oPlan))
        << "A left bound of 2 on Sq=Skv=8 should differ from plain causal attention";
}

// The graph plan computes P in fp32 (an independent oracle, SEM-012), not with the provider's
// bf16 P rounding, which stays available only through fprop's explicit probabilityMode.
TEST(TestGpuSdpaFwdPlanBuilder, ExecuteUsesFloatProbabilityNotProviderRounding)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_gpu_ref::GpuFpReferenceSdpa;
    using hipdnn_gpu_ref::GpuSdpaFwdOptions;
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
        GpuSdpaFwdOptions<float>{}.setScale(1.0f).setProbabilityMode(
            SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE));

    Tensor<bfloat16> oFloatMode(oDims);
    GpuFpReferenceSdpa::fprop<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        q, k, v, oFloatMode, GpuSdpaFwdOptions<float>{}.setScale(1.0f));

    ASSERT_NE(static_cast<float>(oProviderMode.memory().hostData()[0]),
              static_cast<float>(oFloatMode.memory().hostData()[0]));
    EXPECT_EQ(static_cast<float>(oFloatMode.memory().hostData()[0]),
              static_cast<float>(oPlan.memory().hostData()[0]));
}

// A pass-by-value scale lives in the variant pack as a host scalar. The plan must apply it, not
// fall back to the unscaled default: with random inputs in [-1, 1] a scale of 2.0 and a scale of
// 1.0 give outputs that differ far beyond the comparison tolerance.
TEST(TestGpuSdpaFwdPlanBuilder, ExecuteAppliesRuntimePassByValueScale)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_gpu_ref::GpuFpReferenceSdpa;
    using hipdnn_gpu_ref::GpuSdpaFwdOptions;

    constexpr int64_t SCALE_UID = 15;
    constexpr float SCALE_VALUE = 2.0f;

    FloatOperandSpec scaleSpec;
    scaleSpec.uid = SCALE_UID;
    scaleSpec.storage = OperandStorage::RUNTIME_PASS_BY_VALUE;

    auto graphBuilder = createSdpaFwdGraph(Q_UID,
                                           K_UID,
                                           V_UID,
                                           O_UID,
                                           DIMS,
                                           DIMS,
                                           DIMS,
                                           DIMS,
                                           DataType::FLOAT,
                                           /*attrs=*/{},
                                           /*statsUid=*/std::nullopt,
                                           scaleSpec);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    ASSERT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = floatPlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(DIMS);
    Tensor<float> k(DIMS);
    Tensor<float> v(DIMS);
    fillWithRandomValues(q, -1.0f, 1.0f, /*seed=*/11);
    fillWithRandomValues(k, -1.0f, 1.0f, /*seed=*/22);
    fillWithRandomValues(v, -1.0f, 1.0f, /*seed=*/33);

    Tensor<float> oPlan(DIMS);
    float hostScale = SCALE_VALUE;

    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {SCALE_UID, &hostScale},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    Tensor<float> oRef(DIMS);
    GpuFpReferenceSdpa::fprop<float, float, float, float, float>(
        q, k, v, oRef, GpuSdpaFwdOptions<float>{}.setScale(SCALE_VALUE));

    const float tolerance = 1e-5f;
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(tolerance,
                                                                                 tolerance);
    EXPECT_TRUE(validation.allClose(oRef, oPlan))
        << "Plan ignored the runtime pass-by-value scale tensor";
}

// The scale may be baked, pass-by-value or device-resident. It must be a single element, and a
// device-resident scale must be FLOAT because it is read back as one float.
TEST(TestGpuSdpaFwdPlanBuilder, IsApplicableOnlyForSingleElementScaleOperands)
{
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    constexpr int64_t SCALE_UID = 15;

    const auto isApplicableWith = [&](const FloatOperandSpec& scale) {
        auto graphBuilder = createSdpaFwdGraph(Q_UID,
                                               K_UID,
                                               V_UID,
                                               O_UID,
                                               DIMS,
                                               DIMS,
                                               DIMS,
                                               DIMS,
                                               DataType::FLOAT,
                                               /*attrs=*/{},
                                               /*statsUid=*/std::nullopt,
                                               scale);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        return floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
    };

    FloatOperandSpec scale;
    scale.uid = SCALE_UID;

    scale.storage = OperandStorage::BAKED;
    scale.bakedValue = 2.0f;
    EXPECT_TRUE(isApplicableWith(scale));

    scale.storage = OperandStorage::RUNTIME_PASS_BY_VALUE;
    EXPECT_TRUE(isApplicableWith(scale));

    scale.storage = OperandStorage::TENSOR;
    EXPECT_TRUE(isApplicableWith(scale));

    scale.dims = {2};
    EXPECT_FALSE(isApplicableWith(scale)) << "a multi-element device scale must be declined";
}

// Regression guard for the provider P->bf16 rounding gate. The cast is a property
// of the P@V matmul inputs (Q/K/V), not the output dtype, so the registered
// bf16-in/fp32-out config must still round P before P@V. Compile-time because
// providerSdpaProbabilityMode is constexpr; no device required.
TEST(TestGpuSdpaFwdPlanBuilder, ProbabilityModeKeyedOnInputsNotOutput)
{
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode;

    static_assert(REFERENCE_SDPA_PROBABILITY_MODE == SdpaSoftmaxProbabilityMode::FLOAT,
                  "the graph plans must use fp32 softmax probabilities");
    static_assert(providerSdpaProbabilityMode<bfloat16, bfloat16, bfloat16, bfloat16>()
                      == SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE,
                  "all-bf16 must round softmax probabilities to bf16 before P@V");
    static_assert(providerSdpaProbabilityMode<bfloat16, bfloat16, bfloat16, float>()
                      == SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE,
                  "bf16 inputs with fp32 output must still round P to bf16 before P@V");
    static_assert(providerSdpaProbabilityMode<float, float, float, float>()
                      == SdpaSoftmaxProbabilityMode::FLOAT,
                  "fp32 inputs must not round the softmax probabilities");
    static_assert(providerSdpaProbabilityMode<float, bfloat16, bfloat16, bfloat16>()
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

// A mask builder takes exactly the graphs whose mask tensor has its dtype; the UNSET builder
// takes exactly the graphs without a mask.
TEST(TestGpuSdpaFwdPlanBuilder, MaskDataTypeSelectsApplicableBuilder)
{
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::UNSET>
        noMaskBuilder;
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT>
        floatMaskBuilder;
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::HALF>
        halfMaskBuilder;
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::BFLOAT16>
        bfloat16MaskBuilder;

    const auto applicability = [&](flatbuffers::FlatBufferBuilder& graphBuilder) {
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        const auto& node = graphWrap.getNode(0);
        const auto& tensorMap = graphWrap.getTensorMap();
        return std::array<bool, 4>{noMaskBuilder.isApplicable(node, tensorMap),
                                   floatMaskBuilder.isApplicable(node, tensorMap),
                                   halfMaskBuilder.isApplicable(node, tensorMap),
                                   bfloat16MaskBuilder.isApplicable(node, tensorMap)};
    };
    using Applicability = std::array<bool, 4>;

    auto noMask = makeGraph();
    EXPECT_EQ(applicability(noMask), (Applicability{true, false, false, false}));

    auto floatMask = makeGraphWithMask(DataType::FLOAT);
    EXPECT_EQ(applicability(floatMask), (Applicability{false, true, false, false}));

    auto halfMask = makeGraphWithMask(DataType::HALF);
    EXPECT_EQ(applicability(halfMask), (Applicability{false, false, true, false}));

    auto bfloat16Mask = makeGraphWithMask(DataType::BFLOAT16);
    EXPECT_EQ(applicability(bfloat16Mask), (Applicability{false, false, false, true}));

    // A mask tensor with dtype UNSET matches no builder's dtype rule.
    auto unsetMask = makeGraphWithMask(DataType::UNSET);
    EXPECT_EQ(applicability(unsetMask), (Applicability{false, false, false, false}));
}

// Registry and applicability through the executor, which is what the verification harness
// calls. None of these cases reach the GPU: plan selection fails or succeeds before execution.
class TestGpuSdpaFwdMaskGraphExecutor : public ::testing::Test
{
protected:
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor _executor;
};

TEST_F(TestGpuSdpaFwdMaskGraphExecutor, NoMaskAndSupportedMaskDataTypesAreApplicable)
{
    auto noMask = makeGraph();
    EXPECT_TRUE(_executor.isApplicable(noMask.GetBufferPointer(), noMask.GetSize()));

    for(const auto maskType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        auto graph = makeGraphWithMask(maskType);
        EXPECT_TRUE(_executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
    }
}

// INT32 maps to no registered key: the registry throws, and the graph is not applicable.
TEST_F(TestGpuSdpaFwdMaskGraphExecutor, UnsupportedMaskDataTypeHasNoPlan)
{
    auto graph = makeGraphWithMask(DataType::INT32);
    EXPECT_FALSE(_executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));

    const std::unordered_map<int64_t, void*> emptyVariantPack;
    try
    {
        _executor.execute(graph.GetBufferPointer(), graph.GetSize(), emptyVariantPack);
        FAIL() << "expected a registry error for an INT32 mask";
    }
    catch(const hipdnn_integration_tests::ReferenceCapabilityError&)
    {
        FAIL() << "an unregistered mask dtype must be the registry error, not a capability error";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("mask=INT32"), std::string::npos) << e.what();
    }
}

// An UNSET-dtype mask maps to the registered no-mask key, whose builder rejects it.
TEST_F(TestGpuSdpaFwdMaskGraphExecutor, UnsetMaskDataTypeIsCapabilityError)
{
    auto graph = makeGraphWithMask(DataType::UNSET);
    EXPECT_FALSE(_executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));

    const std::unordered_map<int64_t, void*> emptyVariantPack;
    EXPECT_THROW(_executor.execute(graph.GetBufferPointer(), graph.GetSize(), emptyVariantPack),
                 hipdnn_integration_tests::ReferenceCapabilityError);
}

TEST_F(TestGpuSdpaFwdMaskGraphExecutor, DanglingMaskUidIsNotApplicable)
{
    auto graph = createSdpaFwdGraph(Q_UID,
                                    K_UID,
                                    V_UID,
                                    O_UID,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    DataType::FLOAT,
                                    {},
                                    std::nullopt,
                                    /*scale=*/std::nullopt,
                                    MASK_UID,
                                    MASK_DIMS,
                                    std::nullopt);
    EXPECT_FALSE(_executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

namespace
{

// Runs the half/bfloat16-mask plan and compares it with CpuFpReferenceSdpa::forward on a mask of
// the same dtype and with the FLOAT-mask plan holding the same values. The latter runs the same
// kernel on identical inputs, so it must be bitwise equal: a mask read with the wrong dtype
// diverges.
template <DataType MaskEnum>
void expectMaskedPlanMatchesReferences()
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_data_sdk::utilities::Tensor;
    using MaskT = hipdnn_test_sdk::utilities::MaskDataTypeToNative<MaskEnum>;

    auto graph = makeGraphWithMask(MaskEnum);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graph.GetBufferPointer(), graph.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                MaskEnum>
        builder;
    auto plan = builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    auto floatGraph = makeGraphWithMask(DataType::FLOAT);
    auto floatGraphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        floatGraph.GetBufferPointer(), floatGraph.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT,
                                DataType::FLOAT>
        floatBuilder;
    auto floatPlan = floatBuilder.buildNodePlan(floatGraphWrap, floatGraphWrap.getNode(0));

    Tensor<float> q(DIMS);
    Tensor<float> k(DIMS);
    Tensor<float> v(DIMS);
    fillWithRandomValues(q, -1.0f, 1.0f, /*seed=*/11);
    fillWithRandomValues(k, -1.0f, 1.0f, /*seed=*/22);
    fillWithRandomValues(v, -1.0f, 1.0f, /*seed=*/33);

    Tensor<MaskT> mask(MASK_DIMS);
    fillExactlyRepresentableMask(mask);
    Tensor<float> floatMask(MASK_DIMS);
    for(size_t i = 0; i < mask.elementCount(); ++i)
    {
        floatMask.memory().hostData()[i] = static_cast<float>(mask.memory().hostData()[i]);
    }

    // The CPU oracle reads host data, so it runs before any device upload.
    Tensor<float> oCpu(DIMS);
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::
        forward<float, float, float, float, float, MaskT>(
            q, k, v, oCpu, hipdnn_test_sdk::utilities::SdpaFwdOptions<MaskT>{}.setMask(&mask));

    Tensor<float> oPlan(DIMS);
    Tensor<float> oFloatPlan(DIMS);
    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {MASK_UID, mask.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    const std::unordered_map<int64_t, void*> floatVariantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oFloatPlan.memory().deviceData()},
        {MASK_UID, floatMask.memory().deviceData()},
    };
    floatPlan->execute(floatVariantPack);
    oFloatPlan.markDeviceModified();

    const float tolerance = 2e-5f;
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(tolerance,
                                                                                 tolerance);
    EXPECT_TRUE(validation.allClose(oCpu, oPlan))
        << "Plan output differs from CpuFpReferenceSdpa::forward with a matching typed mask";

    const auto* actual = oPlan.memory().hostData();
    const auto* expected = oFloatPlan.memory().hostData();
    for(size_t i = 0; i < oPlan.elementCount(); ++i)
    {
        ASSERT_EQ(actual[i], expected[i]) << "at flat index " << i;
    }
}

} // namespace

TEST(TestGpuSdpaFwdPlanBuilder, ExecuteWithHalfMask)
{
    expectMaskedPlanMatchesReferences<DataType::HALF>();
}

TEST(TestGpuSdpaFwdPlanBuilder, ExecuteWithBfloat16Mask)
{
    expectMaskedPlanMatchesReferences<DataType::BFLOAT16>();
}

// fp8, as on the CPU: all three FLOAT Q/K/V descales are required, a paged K needs a scalar
// descale, and the cuDNN sdpa_fp8 scales and amax outputs are one-element FLOAT tensors with
// scale_s and descale_s together, fp8 inputs only and no dropout. Each case changes one field of
// an accepted graph.
TEST(TestGpuSdpaFwdPlanBuilder, Fp8ApplicabilityGatesEveryField)
{
    using namespace hipdnn_test_sdk::utilities::sdpa_graph_test;
    const auto applicable = [](const auto& builder, const SdpaGraphSpec& spec) {
        auto graphBuilder = makeSdpaGraph(spec);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        return builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
    };
    const GpuSdpaFwdPlanBuilder<DataType::FP8_E4M3,
                                DataType::FP8_E4M3,
                                DataType::FP8_E4M3,
                                DataType::BFLOAT16>
        ocp;
    const GpuSdpaFwdPlanBuilder<DataType::FP8_E4M3_FNUZ,
                                DataType::FP8_E4M3_FNUZ,
                                DataType::FP8_E4M3_FNUZ,
                                DataType::BFLOAT16>
        fnuz;
    const GpuSdpaFwdPlanBuilder<DataType::FP8_E4M3,
                                DataType::FP8_E4M3,
                                DataType::FP8_E4M3,
                                DataType::FP8_E4M3>
        ocpToFp8;
    const GpuSdpaFwdPlanBuilder<DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16>
        bf16;

    // Descales.
    EXPECT_TRUE(applicable(ocp, {}));
    SdpaGraphSpec spec;
    spec.fp8Type = DataType::FP8_E4M3_FNUZ;
    EXPECT_TRUE(applicable(fnuz, spec));
    EXPECT_FALSE(applicable(ocp, spec)) << "dtype keys the builder";
    spec = {};
    spec.descales = 2;
    EXPECT_FALSE(applicable(ocp, spec)) << "descale_v missing";
    spec.descales = 0;
    EXPECT_FALSE(applicable(ocp, spec)) << "no descales";
    spec = {};
    spec.pagedK = true;
    EXPECT_TRUE(applicable(ocp, spec)) << "paged K with a scalar descale";
    spec.descaleKDims = {1, 2, 1, 1};
    EXPECT_FALSE(applicable(ocp, spec)) << "paged K with a per-head descale";
    spec = {};
    spec.fp8Type = DataType::BFLOAT16;
    EXPECT_FALSE(applicable(bf16, spec)) << "descales on bf16 inputs";
    spec.descales = 0;
    EXPECT_TRUE(applicable(bf16, spec)) << "the same graph without descales";

    // Softmax / output scaling and amax.
    spec = {};
    spec.withFullScaling();
    EXPECT_TRUE(applicable(ocp, spec));
    spec.oType = DataType::FP8_E4M3;
    EXPECT_TRUE(applicable(ocpToFp8, spec)) << "fp8 output";
    spec = {};
    spec.withFullScaling().scaleS = false;
    EXPECT_FALSE(applicable(ocp, spec)) << "descale_s without scale_s";
    spec = {};
    spec.withFullScaling().descaleS = false;
    EXPECT_FALSE(applicable(ocp, spec)) << "scale_s without descale_s";
    for(auto SdpaGraphSpec::*const field :
        {&SdpaGraphSpec::scaleO, &SdpaGraphSpec::amaxS, &SdpaGraphSpec::amaxO})
    {
        spec = {};
        spec.*field = true;
        EXPECT_TRUE(applicable(ocp, spec)) << "each of scale_o / amax_s / amax_o alone";
    }
    spec = {};
    spec.withFullScaling().scalingType = DataType::HALF;
    EXPECT_FALSE(applicable(ocp, spec)) << "scales are read as float";
    spec = {};
    spec.withFullScaling().scalingDims = {1, 2, 1, 1};
    EXPECT_FALSE(applicable(ocp, spec)) << "scales are single elements";
    spec = {};
    spec.withFullScaling().dropout = true;
    EXPECT_FALSE(applicable(ocp, spec)) << "dropout does not combine with fp8 scaling";
    spec.withFullScaling().dropout = false;
    EXPECT_TRUE(applicable(ocp, spec));
    spec = {};
    spec.fp8Type = DataType::BFLOAT16;
    spec.descales = 0;
    spec.scaleO = true;
    EXPECT_FALSE(applicable(bf16, spec)) << "scale_o on bf16 inputs";
}

// cuDNN's block mask (SEM-025), the same rules as the CPU plan test.
TEST(TestGpuSdpaFwdPlanBuilder, BlockMaskApplicabilityGatesEveryField)
{
    using namespace hipdnn_test_sdk::utilities::sdpa_graph_test;
    const GpuSdpaFwdPlanBuilder<DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16>
        bf16;
    const auto applicable = [&bf16](const SdpaGraphSpec& spec) {
        auto graphBuilder = makeSdpaGraph(spec);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        return bf16.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
    };
    SdpaGraphSpec spec;
    spec.fp8Type = DataType::BFLOAT16;
    spec.descales = 0;
    spec.blockMaskType = DataType::UINT8;
    EXPECT_TRUE(applicable(spec));
    spec.blockMaskType = DataType::INT8;
    EXPECT_FALSE(applicable(spec)) << "UINT8 only";
    spec.blockMaskType = DataType::UINT8;
    for(const auto& dims : std::vector<std::vector<int64_t>>{
            {2, 2, 1, 1}, {1, 1, 1, 1}, {1, 2, 2, 1}, {1, 2, 1, 2}, {1, 2, 1}})
    {
        spec.blockMaskDims = dims;
        EXPECT_FALSE(applicable(spec)) << "dims must be [B, H_q, ceil(Sq/128), ceil(tiles/8)]";
    }
}

namespace
{

constexpr int64_t SEED_UID = 16;
constexpr int64_t OFFSET_UID = 17;
constexpr int64_t RNG_DUMP_UID = 18;
constexpr int64_t SINK_UID = 19;

// Philox keep mask / rng_dump [B, H, Sq, Skv] for the DIMS graph.
const std::vector<int64_t> DROPOUT_DIMS = {1, 2, 8, 8};

struct ExtraTensor
{
    int64_t uid;
    DataType dataType;
    std::vector<int64_t> dims;
};

// The FLOAT DIMS graph with `attrs` plus `extras` (for operands createSdpaFwdGraph has no slot
// for: Philox seed / offset / rng_dump, the attention sink).
flatbuffers::FlatBufferBuilder makeGraphWithExtras(SdpaAttributesT attrs,
                                                   const std::vector<ExtraTensor>& extras)
{
    attrs.q_tensor_uid = Q_UID;
    attrs.k_tensor_uid = K_UID;
    attrs.v_tensor_uid = V_UID;
    attrs.o_tensor_uid = O_UID;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto add = [&](int64_t uid, DataType type, const std::vector<int64_t>& dims) {
        const auto strides = generateStrides(dims);
        tensors.push_back(CreateTensorAttributesDirect(builder, uid, "T", type, &strides, &dims));
    };
    for(const auto uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        add(uid, DataType::FLOAT, DIMS);
    }
    for(const auto& extra : extras)
    {
        add(extra.uid, extra.dataType, extra.dims);
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));
    builder.Finish(CreateGraphDirect(builder,
                                     "SdpaFwdTestGraph",
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     &tensors,
                                     &nodes));
    return builder;
}

} // namespace

// Philox dropout with an rng_dump output (SEM-015): the plan accepts it and applies the dumped
// keep mask with scale 1 / (1 - p), so it matches the CPU executor run on the same graph and the
// same dumped mask.
TEST(TestGpuSdpaFwdPlanBuilder, ExecutePhiloxDropoutMatchesCpuOnDumpedMask)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_data_sdk::utilities::Tensor;

    SdpaAttributesT attrs;
    attrs.dropout_probability = 0.25F;
    attrs.seed_tensor_uid = SEED_UID;
    attrs.offset_tensor_uid = OFFSET_UID;
    attrs.rng_dump_tensor_uid = RNG_DUMP_UID;
    auto graph = makeGraphWithExtras(attrs,
                                     {{SEED_UID, DataType::INT64, {1}},
                                      {OFFSET_UID, DataType::INT64, {1}},
                                      {RNG_DUMP_UID, DataType::FLOAT, DROPOUT_DIMS}});
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graph.GetBufferPointer(), graph.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    ASSERT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = floatPlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(DIMS);
    Tensor<float> k(DIMS);
    Tensor<float> v(DIMS);
    fillWithRandomValues(q, -1.0f, 1.0f, /*seed=*/11);
    fillWithRandomValues(k, -1.0f, 1.0f, /*seed=*/22);
    fillWithRandomValues(v, -1.0f, 1.0f, /*seed=*/33);
    Tensor<float> dumpedMask(DROPOUT_DIMS);
    for(size_t i = 0; i < dumpedMask.elementCount(); ++i)
    {
        dumpedMask.memory().hostData()[i] = ((i * 5 + i / 8) % 4 == 1) ? 0.0f : 1.0f;
    }
    Tensor<int64_t> seed({1});
    Tensor<int64_t> offset({1});
    seed.fillWithValue(1234);
    offset.fillWithValue(0);

    // The CPU oracle reads host data, so it runs before any device upload.
    Tensor<float> oCpu(DIMS);
    const std::unordered_map<int64_t, void*> hostVariantPack{
        {Q_UID, q.memory().hostData()},
        {K_UID, k.memory().hostData()},
        {V_UID, v.memory().hostData()},
        {O_UID, oCpu.memory().hostData()},
        {SEED_UID, seed.memory().hostData()},
        {OFFSET_UID, offset.memory().hostData()},
        {RNG_DUMP_UID, dumpedMask.memory().hostData()},
    };
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
        graph.GetBufferPointer(), graph.GetSize(), hostVariantPack);

    Tensor<float> oPlan(DIMS);
    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {SEED_UID, seed.memory().deviceData()},
        {OFFSET_UID, offset.memory().deviceData()},
        {RNG_DUMP_UID, dumpedMask.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    const float tolerance = 2e-5f;
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(tolerance,
                                                                                 tolerance);
    EXPECT_TRUE(validation.allClose(oCpu, oPlan))
        << "Plan output differs from the CPU executor on the same dumped keep mask";
}

// Attention sink: [1, H, 1, 1] (cuDNN's shape) in FLOAT, BFLOAT16 or HALF for any Q dtype
// (SEM-022), the same rule as the CPU plan; the kernel widens a 16-bit sink exactly. DIMS has H = 2.
TEST(TestGpuSdpaFwdPlanBuilder, SinkApplicabilityFollowsTheSinkContract)
{
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    const auto applicable = [&floatPlanBuilder](DataType type, const std::vector<int64_t>& dims) {
        SdpaAttributesT attrs;
        attrs.sink_token_tensor_uid = SINK_UID;
        auto graph = makeGraphWithExtras(attrs, {{SINK_UID, type, dims}});
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graph.GetBufferPointer(), graph.GetSize());
        return floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
    };
    for(const auto type : {DataType::FLOAT, DataType::BFLOAT16, DataType::HALF})
    {
        EXPECT_TRUE(applicable(type, {1, 2, 1, 1})) << EnumNameDataType(type);
        EXPECT_FALSE(applicable(type, {1, 1, 1, 1})) << "one value per head";
        EXPECT_FALSE(applicable(type, {2})) << "rank 4";
    }
    EXPECT_FALSE(applicable(DataType::DOUBLE, {1, 2, 1, 1})) << "no DOUBLE sink";
    EXPECT_FALSE(applicable(DataType::FP8_E4M3, {1, 2, 1, 1})) << "no FP8 sink";
}

// A BFLOAT16 or HALF sink through the plan (bound in its own dtype) gives, bit for bit, the O of
// the FLOAT sink holding the same values, and matches the CPU executor on the 16-bit graph.
TEST(TestGpuSdpaFwdPlanBuilder, ExecutesSixteenBitSinksAsTheFloatSink)
{
    SKIP_IF_NO_DEVICES();
    using hipdnn_data_sdk::utilities::Tensor;
    const GpuSdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    Tensor<float> q(DIMS);
    Tensor<float> k(DIMS);
    Tensor<float> v(DIMS);
    fillWithRandomValues(q, -1.0f, 1.0f, /*seed=*/41);
    fillWithRandomValues(k, -1.0f, 1.0f, /*seed=*/42);
    fillWithRandomValues(v, -1.0f, 1.0f, /*seed=*/43);
    // Head 0 well above the scores, head 1 a value bf16 rounds (exact in HALF).
    const std::vector<float> values{3.5f, 0.0009765625f + 9.5367431640625e-07f};

    const auto runPlan = [&](DataType sinkType, void* sinkDevice, Tensor<float>& o) {
        SdpaAttributesT attrs;
        attrs.sink_token_tensor_uid = SINK_UID;
        auto graph = makeGraphWithExtras(attrs, {{SINK_UID, sinkType, {1, 2, 1, 1}}});
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graph.GetBufferPointer(), graph.GetSize());
        ASSERT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
        auto plan = floatPlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));
        const std::unordered_map<int64_t, void*> variantPack{{Q_UID, q.memory().deviceData()},
                                                             {K_UID, k.memory().deviceData()},
                                                             {V_UID, v.memory().deviceData()},
                                                             {O_UID, o.memory().deviceData()},
                                                             {SINK_UID, sinkDevice}};
        plan->execute(variantPack);
        o.markDeviceModified();
    };
    const auto check = [&](auto sinkTag, DataType sinkType) {
        using SinkType = decltype(sinkTag);
        Tensor<SinkType> sink16({1, 2, 1, 1});
        Tensor<float> sink32({1, 2, 1, 1});
        for(int64_t h = 0; h < 2; ++h)
        {
            sink16.setHostValue(SinkType(values[static_cast<size_t>(h)]), 0, h, 0, 0);
            sink32.setHostValue(static_cast<float>(sink16.getHostValue(0, h, 0, 0)), 0, h, 0, 0);
        }
        // The CPU executor on the 16-bit graph reads host data, so it runs first.
        SdpaAttributesT attrs;
        attrs.sink_token_tensor_uid = SINK_UID;
        auto graph = makeGraphWithExtras(attrs, {{SINK_UID, sinkType, {1, 2, 1, 1}}});
        Tensor<float> oCpu(DIMS);
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            graph.GetBufferPointer(),
            graph.GetSize(),
            {{Q_UID, q.memory().hostData()},
             {K_UID, k.memory().hostData()},
             {V_UID, v.memory().hostData()},
             {O_UID, oCpu.memory().hostData()},
             {SINK_UID, sink16.memory().hostData()}});
        Tensor<float> o16(DIMS);
        Tensor<float> o32(DIMS);
        runPlan(sinkType, sink16.memory().deviceData(), o16);
        runPlan(DataType::FLOAT, sink32.memory().deviceData(), o32);
        const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> exact(0.0f, 0.0f);
        EXPECT_TRUE(exact.allClose(o32, o16)) << EnumNameDataType(sinkType);
        const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> close(2e-5f, 2e-5f);
        EXPECT_TRUE(close.allClose(oCpu, o16)) << EnumNameDataType(sinkType);
    };
    check(hipdnn_data_sdk::types::bfloat16{}, DataType::BFLOAT16);
    check(hipdnn_data_sdk::types::half{}, DataType::HALF);
}

// A pass-by-value descale's variant-pack slot holds a host pointer and a baked one has no slot,
// so the plan stages both as one value instead of handing the slot to the kernel as device
// memory. Each must give the output of the same descale held in a device tensor.
TEST(TestGpuSdpaFwdPlanDescale, ScalarDescalesMatchDeviceDescales)
{
    SKIP_IF_NO_DEVICES();
    using namespace hipdnn_test_sdk::utilities::sdpa_graph_test;
    using hipdnn_data_sdk::types::bfloat16;
    using Fp8 = hipdnn_data_sdk::types::fp8_e4m3;
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    hipdnn_data_sdk::utilities::Tensor<Fp8> q(dims);
    hipdnn_data_sdk::utilities::Tensor<Fp8> k(dims);
    hipdnn_data_sdk::utilities::Tensor<Fp8> v(dims);
    q.fillWithRandomValues(Fp8(-1.0F), Fp8(1.0F), 121);
    k.fillWithRandomValues(Fp8(-1.0F), Fp8(1.0F), 122);
    v.fillWithRandomValues(Fp8(-1.0F), Fp8(1.0F), 123);
    hipdnn_data_sdk::utilities::Tensor<float> descale({1});
    descale.fillWithValue(0.5F);

    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
    const auto run = [&](OperandStorage storage, hipdnn_data_sdk::utilities::Tensor<bfloat16>& o) {
        SdpaGraphSpec spec;
        spec.descaleStorage = storage;
        spec.bakedDescale = 0.5F;
        auto graph = makeSdpaGraph(spec);
        ASSERT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
        std::unordered_map<int64_t, void*> pack = {{Q, q.memory().deviceData()},
                                                   {K, k.memory().deviceData()},
                                                   {V, v.memory().deviceData()},
                                                   {O, o.memory().deviceData()}};
        for(const int64_t uid : {DESCALE_Q, DESCALE_K, DESCALE_V})
        {
            if(storage == OperandStorage::TENSOR)
            {
                pack[uid] = descale.memory().deviceData();
            }
            else if(storage == OperandStorage::RUNTIME_PASS_BY_VALUE)
            {
                pack[uid] = descale.memory().hostData();
            }
        }
        ASSERT_NO_THROW(executor.execute(graph.GetBufferPointer(), graph.GetSize(), pack));
        o.markDeviceModified();
    };
    hipdnn_data_sdk::utilities::Tensor<bfloat16> oDevice(dims);
    hipdnn_data_sdk::utilities::Tensor<bfloat16> oPassByValue(dims);
    hipdnn_data_sdk::utilities::Tensor<bfloat16> oBaked(dims);
    run(OperandStorage::TENSOR, oDevice);
    run(OperandStorage::RUNTIME_PASS_BY_VALUE, oPassByValue);
    run(OperandStorage::BAKED, oBaked);

    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<bfloat16> validation(0.0F, 0.0F);
    EXPECT_TRUE(validation.allClose(oDevice, oPassByValue));
    EXPECT_TRUE(validation.allClose(oDevice, oBaked));
}

// A pass-by-value descale is staged as one value, so more than one element is declined; the same
// shape held in a tensor (a per-KV-head descale) is accepted.
TEST(TestGpuSdpaFwdPlanDescale, MultiElementPassByValueDescaleIsDeclined)
{
    using namespace hipdnn_test_sdk::utilities::sdpa_graph_test;
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
    SdpaGraphSpec spec;
    spec.descaleKDims = {1, 2, 1, 1};
    auto tensorGraph = makeSdpaGraph(spec);
    EXPECT_TRUE(executor.isApplicable(tensorGraph.GetBufferPointer(), tensorGraph.GetSize()));
    spec.descaleStorage = OperandStorage::RUNTIME_PASS_BY_VALUE;
    auto passByValueGraph = makeSdpaGraph(spec);
    EXPECT_FALSE(
        executor.isApplicable(passByValueGraph.GetBufferPointer(), passByValueGraph.GetSize()));
}
