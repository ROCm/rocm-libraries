// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

/// Build an SDPA forward graph with explicit per-tensor input/output data types and
/// optional per-tensor Q/K/V descale tensors. Used to exercise the FP8 + descale
/// paths of SdpaFwdPlanBuilder/SdpaFwdPlan on the CPU (no GPU kernels involved).
/// Fixed UIDs: Q=1, K=2, V=3, O=4, descaleQ=5, descaleK=6, descaleV=7.
std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildDescaleSdpaFwdGraph(const std::vector<int64_t>& qDims,
                             const std::vector<int64_t>& kDims,
                             const std::vector<int64_t>& vDims,
                             hipdnn_frontend::DataType inputDataType,
                             hipdnn_frontend::DataType outputDataType,
                             bool withDescale,
                             bool withDescaleS = false)
{
    using hipdnn_data_sdk::utilities::generateStrides;
    using hipdnn_frontend::DataType;
    using hipdnn_frontend::graph::Graph;
    using hipdnn_frontend::graph::SdpaAttributes;
    using hipdnn_frontend::graph::TensorAttributes;

    auto makeAttr = [](const std::string& name,
                       DataType dataType,
                       const std::vector<int64_t>& dims,
                       int64_t uid) {
        auto attr = std::make_shared<TensorAttributes>();
        attr->set_name(name)
            .set_data_type(dataType)
            .set_dim(dims)
            .set_stride(generateStrides(dims))
            .set_uid(uid);
        return attr;
    };

    auto graph = std::make_shared<Graph>();
    graph->set_name("SdpaFwdDescaleTest");
    graph->set_io_data_type(inputDataType)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto qAttr = makeAttr("Q", inputDataType, qDims, 1);
    auto kAttr = makeAttr("K", inputDataType, kDims, 2);
    auto vAttr = makeAttr("V", inputDataType, vDims, 3);

    SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("SdpaFwdDescale");
    if(withDescale)
    {
        const std::vector<int64_t> scalarDims = {1, 1, 1, 1};
        sdpaAttrs.set_descale_q(makeAttr("DESCALE_Q", DataType::FLOAT, scalarDims, 5));
        sdpaAttrs.set_descale_k(makeAttr("DESCALE_K", DataType::FLOAT, scalarDims, 6));
        sdpaAttrs.set_descale_v(makeAttr("DESCALE_V", DataType::FLOAT, scalarDims, 7));
        if(withDescaleS)
        {
            // Softmax descale is part of the fp8 contract but unsupported by the v3
            // forward kernel, so the executor must reject graphs that carry it.
            sdpaAttrs.set_descale_s(makeAttr("DESCALE_S", DataType::FLOAT, scalarDims, 8));
        }
    }

    auto [oAttr, statsAttr] = graph->sdpa(qAttr, kAttr, vAttr, sdpaAttrs);
    if(!oAttr->has_uid())
    {
        oAttr->set_uid(4);
    }
    const std::vector<int64_t> oDims = {qDims[0], qDims[1], qDims[2], vDims[3]};
    oAttr->set_data_type(outputDataType)
        .set_dim(oDims)
        .set_stride(generateStrides(oDims))
        .set_is_virtual(false);

    return graph;
}

} // namespace

namespace
{

// Builds a minimal single-node SDPA-forward flatbuffer graph directly. The frontend
// graph builder only emits the O and Stats (LSE) outputs and silently drops any
// max/sum_exp request, so the plan builder's rejection of those softmax-stats outputs
// can only be exercised at the flatbuffer level. `attrs` carries the unsupported field
// under test; the q/k/v/o uids and a packed float [1, 2, 4, 8] shape are filled in here.
//
// When `maskDataType` is engaged (any dtype, including DataType::UNSET), a [1, 1, 4, 4] mask
// tensor of that dtype is added under uid 5 and attrs.attn_mask_tensor_uid is set to 5. When
// disengaged the mask is left as `attrs` has it, so a caller-set attn_mask_tensor_uid dangles
// (no tensor behind it).
// When `seqLens` is engaged, seq_len tensors with that dtype and dims are added under uids 6 (Q)
// and 7 (KV) and wired into attrs; `includeKv = false` adds only the Q one.
// When `sink` is engaged, a sink tensor with that dtype and dims is added under uid 10 (the caller
// sets attrs.sink_token_tensor_uid).
// Any additional tensor (uid, name, dtype, dims); the caller wires the uid into attrs.
struct RawExtraTensor
{
    int64_t uid;
    const char* name;
    DataType dataType;
    std::vector<int64_t> dims;
};

struct RawSinkSpec
{
    DataType dataType;
    std::vector<int64_t> dims;
};

struct RawSeqLenSpec
{
    DataType dataType;
    std::vector<int64_t> dims;
    bool includeKv = true;
};

flatbuffers::FlatBufferBuilder
    makeRawSdpaFwdGraph(SdpaAttributesT attrs,
                        std::optional<DataType> maskDataType = std::nullopt,
                        const std::optional<RawSeqLenSpec>& seqLens = std::nullopt,
                        const std::optional<RawSinkSpec>& sink = std::nullopt,
                        const std::vector<RawExtraTensor>& extras = {})
{
    constexpr int64_t SEQ_LEN_Q_UID = 6;
    constexpr int64_t SEQ_LEN_KV_UID = 7;
    constexpr int64_t Q_UID = 1;
    constexpr int64_t K_UID = 2;
    constexpr int64_t V_UID = 3;
    constexpr int64_t O_UID = 4;
    constexpr int64_t MASK_UID = 5;
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const std::vector<int64_t> maskDims = {1, 1, 4, 4};
    const auto maskStrides = hipdnn_data_sdk::utilities::generateStrides(maskDims);

    attrs.q_tensor_uid = Q_UID;
    attrs.k_tensor_uid = K_UID;
    attrs.v_tensor_uid = V_UID;
    attrs.o_tensor_uid = O_UID;
    if(maskDataType.has_value())
    {
        attrs.attn_mask_tensor_uid = MASK_UID;
    }

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, Q_UID, "Q", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, K_UID, "K", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, V_UID, "V", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, O_UID, "O", DataType::FLOAT, &strides, &dims));
    if(maskDataType.has_value())
    {
        tensors.push_back(CreateTensorAttributesDirect(
            builder, MASK_UID, "Mask", maskDataType.value(), &maskStrides, &maskDims));
    }
    std::vector<int64_t> seqLenStrides;
    if(seqLens.has_value())
    {
        seqLenStrides = hipdnn_data_sdk::utilities::generateStrides(seqLens->dims);
        attrs.seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
        tensors.push_back(CreateTensorAttributesDirect(
            builder, SEQ_LEN_Q_UID, "SeqLenQ", seqLens->dataType, &seqLenStrides, &seqLens->dims));
        if(seqLens->includeKv)
        {
            attrs.seq_len_kv_tensor_uid = SEQ_LEN_KV_UID;
            tensors.push_back(CreateTensorAttributesDirect(builder,
                                                           SEQ_LEN_KV_UID,
                                                           "SeqLenKv",
                                                           seqLens->dataType,
                                                           &seqLenStrides,
                                                           &seqLens->dims));
        }
    }

    std::vector<int64_t> sinkStrides;
    if(sink.has_value())
    {
        sinkStrides = hipdnn_data_sdk::utilities::generateStrides(sink->dims);
        tensors.push_back(CreateTensorAttributesDirect(
            builder, 10, "Sink", sink->dataType, &sinkStrides, &sink->dims));
    }

    std::vector<std::vector<int64_t>> extraStrides;
    extraStrides.reserve(extras.size());
    for(const auto& extra : extras)
    {
        extraStrides.push_back(hipdnn_data_sdk::utilities::generateStrides(extra.dims));
        tensors.push_back(CreateTensorAttributesDirect(
            builder, extra.uid, extra.name, extra.dataType, &extraStrides.back(), &extra.dims));
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));
    auto graph = CreateGraphDirect(builder,
                                   "SdpaFwdRejectGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);
    return builder;
}

} // namespace

TEST(TestSdpaFwdPlan, ExecutePlan)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8] — standard MHA (numHeads == numKvHeads)
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    SdpaFwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                         *tensorMap.at(nodeAttributes->k_tensor_uid()),
                         *tensorMap.at(nodeAttributes->v_tensor_uid()),
                         *tensorMap.at(nodeAttributes->o_tensor_uid()),
                         std::nullopt,
                         /*leftBound=*/-1,
                         /*rightBound=*/-1,
                         /*topLeftAlignment=*/true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();

    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor);

    SdpaFwdPlan<float, float, float, float> patient(std::move(params));
    patient.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor));
}

TEST(TestSdpaFwdPlan, ExecutePlanWithRuntimeScaleFromPack)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    // Pure runtime pass-by-value scale delivered through the variant pack.
    float scaleHostValue = 0.25f;
    auto graphTuple = buildSdpaFwdGraph(planTensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/std::nullopt,
                                        /*rightBound=*/std::nullopt,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                                        /*alibiMask=*/false,
                                        /*generateStats=*/false,
                                        /*runtimeScaleHostPtr=*/&scaleHostValue);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    ASSERT_TRUE(nodeAttributes->scale_tensor_uid().has_value());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        builder;
    auto builtPlan = builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
    builtPlan->execute(variantPack);

    // Direct reference with the same explicit scale value.
    CpuFpReferenceSdpa::forward<float, float, float, float>(
        directTensorBundle.qTensor,
        directTensorBundle.kTensor,
        directTensorBundle.vTensor,
        directTensorBundle.oTensor,
        SdpaFwdOptions<float>{}.setScale(scaleHostValue));

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor));
}

TEST(TestSdpaFwdPlan, ExecutePlanWithCausalMask)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8] with causal mask
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle, DataType::FLOAT, /*causalMask=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    SdpaFwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                         *tensorMap.at(nodeAttributes->k_tensor_uid()),
                         *tensorMap.at(nodeAttributes->v_tensor_uid()),
                         *tensorMap.at(nodeAttributes->o_tensor_uid()),
                         std::nullopt,
                         /*leftBound=*/-1,
                         /*rightBound=*/0,
                         /*topLeftAlignment=*/true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();

    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward<float, float, float, float>(
        directTensorBundle.qTensor,
        directTensorBundle.kTensor,
        directTensorBundle.vTensor,
        directTensorBundle.oTensor,
        SdpaFwdOptions<float>{}.setMask(noMask).setBand(-1, 0, true));

    SdpaFwdPlan<float, float, float, float> patient(std::move(params));
    patient.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor));
}

TEST(TestSdpaFwdPlanBuilder, ExecutePlanWithAsymmetricWindow)
{
    // Regression test for a bug where SdpaFwdPlanBuilder forwarded the same value
    // (left_bound) for both leftBound and rightBound to CpuFpReferenceSdpa::forward,
    // ignoring right_bound. With leftBound == rightBound (e.g. for symmetric windows
    // or causal masks where both are -1/0), that bug is invisible. This test uses an
    // asymmetric window (leftBound=2, rightBound=1) so that any confusion of the two
    // bounds produces a different attention pattern and a divergent output.
    //
    // Compares the output of the dispatched plan (graph → SdpaFwdPlanBuilder →
    // SdpaFwdPlan::execute) against a direct call to CpuFpReferenceSdpa::forward
    // with the same asymmetric window parameters.
    //
    // [B=1, H=2, Sq=4, Skv=4, D=8] with leftBound=2, rightBound=1, TopLeft alignment.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    constexpr int64_t LEFT_BOUND = 2;
    constexpr int64_t RIGHT_BOUND = 1;

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/LEFT_BOUND,
                                        /*rightBound=*/RIGHT_BOUND,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    // Build the plan through SdpaFwdPlanBuilder so the dispatcher's left/right bound
    // extraction is exercised (this is where the original bug lived).
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    auto plan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();
    plan->execute(variantPack);

    // Direct CPU reference with the same asymmetric window parameters.
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward<float, float, float, float>(
        directTensorBundle.qTensor,
        directTensorBundle.kTensor,
        directTensorBundle.vTensor,
        directTensorBundle.oTensor,
        SdpaFwdOptions<float>{}.setMask(noMask).setBand(LEFT_BOUND, RIGHT_BOUND, true));

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor))
        << "Plan output (via SdpaFwdPlanBuilder) does not match direct CpuFpReferenceSdpa "
           "with leftBound="
        << LEFT_BOUND << ", rightBound=" << RIGHT_BOUND
        << ". This indicates the dispatcher is not forwarding the two bounds distinctly.";
}

TEST(TestSdpaFwdPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<SdpaFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestSdpaFwdPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    // Correct data types: applicable
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Mismatched data types: not applicable
    const SdpaFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(
        halfPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Missing tensor in map: not applicable
    auto tensorMapCopy = graphWrapper.getTensorMap();
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    tensorMapCopy.erase(nodeAttributes->k_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsAlibiMask)
{
    // SdpaFwdPlanBuilder does not implement ALiBi positional encoding, so it must
    // refuse to claim nodes that have alibi_mask=true. Otherwise a graph with ALiBi
    // would silently fall through to a CPU plan that ignores the ALiBi attribute
    // and produces wrong results.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/std::nullopt,
                                        /*rightBound=*/std::nullopt,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                                        /*alibiMask=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()))
        << "SdpaFwdPlanBuilder must reject nodes with alibi_mask=true";
}

// Padding (seq_len_q/kv): both INT32 tensors with one value per batch, or neither. padding_mask
// without the tensors is declined (cuDNN requires them); the tensors without padding_mask are
// accepted, as hipDNN graphs omit the flag (register SEM-008). The raw graph has batch 1.
struct FwdPaddingApplicabilityCase
{
    const char* name;
    bool paddingMask;
    std::optional<RawSeqLenSpec> seqLens;
    bool applicable;
};

class TestSdpaFwdPlanPaddingApplicability : public TestWithParam<FwdPaddingApplicabilityCase>
{
};

TEST_P(TestSdpaFwdPlanPaddingApplicability, MatchesContract)
{
    const auto& p = GetParam();
    SdpaAttributesT attrs;
    attrs.padding_mask = p.paddingMask;
    auto graphBuilder = makeRawSdpaFwdGraph(attrs, std::nullopt, p.seqLens);

    CpuReferenceGraphExecutor executor;
    EXPECT_EQ(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()),
              p.applicable);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestSdpaFwdPlanPaddingApplicability,
    Values(FwdPaddingApplicabilityCase{"SeqLensWithoutFlag",
                                       false,
                                       RawSeqLenSpec{DataType::INT32, {1, 1, 1, 1}},
                                       true},
           FwdPaddingApplicabilityCase{
               "SeqLensWithFlag", true, RawSeqLenSpec{DataType::INT32, {1, 1, 1, 1}}, true},
           FwdPaddingApplicabilityCase{"FlagWithoutSeqLens", true, std::nullopt, false},
           FwdPaddingApplicabilityCase{
               "OnlySeqLenQ", false, RawSeqLenSpec{DataType::INT32, {1, 1, 1, 1}, false}, false},
           FwdPaddingApplicabilityCase{
               "FloatSeqLens", false, RawSeqLenSpec{DataType::FLOAT, {1, 1, 1, 1}}, false},
           FwdPaddingApplicabilityCase{
               "WrongElementCount", false, RawSeqLenSpec{DataType::INT32, {1, 2, 1, 1}}, false}),
    [](const TestParamInfo<FwdPaddingApplicabilityCase>& info) {
        return std::string(info.param.name);
    });

// A padded graph runs through the executor and equals the reference called directly.
TEST(TestSdpaFwdPlanPadding, ExecuteMatchesDirectReference)
{
    SdpaAttributesT attrs;
    attrs.padding_mask = true;
    auto graphBuilder
        = makeRawSdpaFwdGraph(attrs, std::nullopt, RawSeqLenSpec{DataType::INT32, {1, 1, 1, 1}});

    const std::vector<int64_t> dims = {1, 2, 4, 8};
    hipdnn_data_sdk::utilities::Tensor<float> q(dims);
    hipdnn_data_sdk::utilities::Tensor<float> k(dims);
    hipdnn_data_sdk::utilities::Tensor<float> v(dims);
    hipdnn_data_sdk::utilities::Tensor<float> o(dims);
    hipdnn_data_sdk::utilities::Tensor<int32_t> seqLenQ({1, 1, 1, 1});
    hipdnn_data_sdk::utilities::Tensor<int32_t> seqLenKv({1, 1, 1, 1});
    q.fillWithRandomValues(-1.0F, 1.0F, 31);
    k.fillWithRandomValues(-1.0F, 1.0F, 32);
    v.fillWithRandomValues(-1.0F, 1.0F, 33);
    seqLenQ.fillWithValue(3);
    seqLenKv.fillWithValue(2);
    o.fillWithValue(std::numeric_limits<float>::quiet_NaN());

    const std::unordered_map<int64_t, void*> variantPack = {{1, q.memory().hostData()},
                                                            {2, k.memory().hostData()},
                                                            {3, v.memory().hostData()},
                                                            {4, o.memory().hostData()},
                                                            {6, seqLenQ.memory().hostData()},
                                                            {7, seqLenKv.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);

    hipdnn_data_sdk::utilities::Tensor<float> oRef(dims);
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(
        q, k, v, oRef, SdpaFwdOptions<float>{}.setMask(noMask).setSeqLens(&seqLenQ, &seqLenKv));

    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(o, oRef));
    // Row 3 is padding: O = 0 (it was NaN before the run).
    EXPECT_EQ(o.getHostValue(0, 1, 3, 0), 0.0F);
}

// Paged KV cache (cuDNN layout). The frontend cannot build these graphs (its node validation
// requires K/V batch == Q batch, but a paged container's first dim is the block count), so they
// are built raw. Q is FLOAT [2, 2, 3, 8]; paged K/V containers are [5, 2, 2, 8] (block size 2);
// page tables are INT32 [2, 1, 3, 1] (capacity 6). UIDs: Q=1 K=2 V=3 O=4 SeqLenQ=6 SeqLenKv=7
// PageTableK=8 PageTableV=9.
namespace
{

struct RawPagedGraphSpec
{
    bool pageK = true;
    bool pageV = true;
    bool seqLens = true;
    DataType pageTableType = DataType::INT32;
    std::optional<int32_t> maxSeqLenKv;
};

flatbuffers::FlatBufferBuilder makeRawPagedSdpaFwdGraph(const RawPagedGraphSpec& spec)
{
    const std::vector<int64_t> qDims = {2, 2, 3, 8};
    const std::vector<int64_t> cDims = {5, 2, 2, 8};
    const std::vector<int64_t> denseKvDims = {2, 2, 6, 8};
    const std::vector<int64_t> lenDims = {2, 1, 1, 1};
    const std::vector<int64_t> tableDims = {2, 1, 3, 1};
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);
    const auto cStrides = hipdnn_data_sdk::utilities::generateStrides(cDims);
    const auto denseKvStrides = hipdnn_data_sdk::utilities::generateStrides(denseKvDims);
    const auto lenStrides = hipdnn_data_sdk::utilities::generateStrides(lenDims);
    const auto tableStrides = hipdnn_data_sdk::utilities::generateStrides(tableDims);

    SdpaAttributesT attrs;
    attrs.q_tensor_uid = 1;
    attrs.k_tensor_uid = 2;
    attrs.v_tensor_uid = 3;
    attrs.o_tensor_uid = 4;
    attrs.padding_mask = spec.seqLens;
    attrs.max_seq_len_kv = spec.maxSeqLenKv;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto add = [&](int64_t uid,
                         const char* name,
                         DataType type,
                         const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& dims) {
        tensors.push_back(CreateTensorAttributesDirect(builder, uid, name, type, &strides, &dims));
    };
    add(1, "Q", DataType::FLOAT, qStrides, qDims);
    add(2,
        "K",
        DataType::FLOAT,
        spec.pageK ? cStrides : denseKvStrides,
        spec.pageK ? cDims : denseKvDims);
    add(3,
        "V",
        DataType::FLOAT,
        spec.pageV ? cStrides : denseKvStrides,
        spec.pageV ? cDims : denseKvDims);
    add(4, "O", DataType::FLOAT, qStrides, qDims);
    if(spec.seqLens)
    {
        attrs.seq_len_q_tensor_uid = 6;
        attrs.seq_len_kv_tensor_uid = 7;
        add(6, "SeqLenQ", DataType::INT32, lenStrides, lenDims);
        add(7, "SeqLenKv", DataType::INT32, lenStrides, lenDims);
    }
    if(spec.pageK)
    {
        attrs.page_table_k_tensor_uid = 8;
        add(8, "PageTableK", spec.pageTableType, tableStrides, tableDims);
    }
    if(spec.pageV)
    {
        attrs.page_table_v_tensor_uid = 9;
        add(9, "PageTableV", spec.pageTableType, tableStrides, tableDims);
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));
    auto graph = CreateGraphDirect(builder,
                                   "SdpaFwdPagedGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);
    return builder;
}

} // namespace

struct PagedApplicabilityCase
{
    const char* name;
    RawPagedGraphSpec spec;
    bool applicable;
};

class TestSdpaFwdPlanPagedApplicability : public TestWithParam<PagedApplicabilityCase>
{
};

TEST_P(TestSdpaFwdPlanPagedApplicability, MatchesContract)
{
    auto graphBuilder = makeRawPagedSdpaFwdGraph(GetParam().spec);
    CpuReferenceGraphExecutor executor;
    EXPECT_EQ(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()),
              GetParam().applicable);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestSdpaFwdPlanPagedApplicability,
    Values(PagedApplicabilityCase{"PagedKv", {}, true},
           PagedApplicabilityCase{
               "PagedKOnly", {true, false, true, DataType::INT32, std::nullopt}, true},
           PagedApplicabilityCase{
               "MaxSeqLenWithinCapacity", {true, true, true, DataType::INT32, 5}, true},
           // cuDNN: paging needs padding.
           PagedApplicabilityCase{
               "WithoutSeqLens", {true, true, false, DataType::INT32, std::nullopt}, false},
           PagedApplicabilityCase{
               "MaxSeqLenBeyondCapacity", {true, true, true, DataType::INT32, 7}, false},
           PagedApplicabilityCase{
               "FloatPageTable", {true, true, true, DataType::FLOAT, std::nullopt}, false},
           // cuDNN: max_seq_len_kv without paging is rejected.
           PagedApplicabilityCase{
               "MaxSeqLenWithoutPaging", {false, false, true, DataType::INT32, 6}, false}),
    [](const TestParamInfo<PagedApplicabilityCase>& info) { return std::string(info.param.name); });

// A paged graph runs through the executor and equals gathering the cache and running the dense
// padded reference directly.
TEST(TestSdpaFwdPlanPaged, ExecuteMatchesGatherThenDense)
{
    auto graphBuilder = makeRawPagedSdpaFwdGraph({});

    hipdnn_data_sdk::utilities::Tensor<float> q({2, 2, 3, 8});
    hipdnn_data_sdk::utilities::Tensor<float> kCache({5, 2, 2, 8});
    hipdnn_data_sdk::utilities::Tensor<float> vCache({5, 2, 2, 8});
    hipdnn_data_sdk::utilities::Tensor<float> o({2, 2, 3, 8});
    hipdnn_data_sdk::utilities::Tensor<int32_t> seqLenQ({2, 1, 1, 1});
    hipdnn_data_sdk::utilities::Tensor<int32_t> seqLenKv({2, 1, 1, 1});
    hipdnn_data_sdk::utilities::Tensor<int32_t> pageTable({2, 1, 3, 1});
    q.fillWithRandomValues(-1.0F, 1.0F, 41);
    kCache.fillWithRandomValues(-1.0F, 1.0F, 42);
    vCache.fillWithRandomValues(-1.0F, 1.0F, 43);
    seqLenQ.setHostValue(3, 0, 0, 0, 0);
    seqLenQ.setHostValue(2, 1, 0, 0, 0);
    seqLenKv.setHostValue(5, 0, 0, 0, 0);
    seqLenKv.setHostValue(2, 1, 0, 0, 0);
    // Batch 0 uses blocks 4, 1, 3; batch 1 uses block 2, then the null block 0.
    const std::vector<int32_t> blocks = {4, 1, 3, 2, 0, 0};
    for(int64_t i = 0; i < 6; ++i)
    {
        pageTable.setHostValue(blocks[static_cast<size_t>(i)], i / 3, 0, i % 3, 0);
    }
    o.fillWithValue(std::numeric_limits<float>::quiet_NaN());

    const std::unordered_map<int64_t, void*> variantPack = {{1, q.memory().hostData()},
                                                            {2, kCache.memory().hostData()},
                                                            {3, vCache.memory().hostData()},
                                                            {4, o.memory().hostData()},
                                                            {6, seqLenQ.memory().hostData()},
                                                            {7, seqLenKv.memory().hostData()},
                                                            {8, pageTable.memory().hostData()},
                                                            {9, pageTable.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);

    auto kDense = CpuFpReferenceSdpa::gatherPagedKv(kCache, pageTable, seqLenKv, 6);
    auto vDense = CpuFpReferenceSdpa::gatherPagedKv(vCache, pageTable, seqLenKv, 6);
    hipdnn_data_sdk::utilities::Tensor<float> oRef({2, 2, 3, 8});
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(
        q,
        kDense,
        vDense,
        oRef,
        SdpaFwdOptions<float>{}.setMask(noMask).setSeqLens(&seqLenQ, &seqLenKv));

    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(o, oRef));
}

// End to end: a paged graph built through the frontend (num_blocks != batch) validates,
// serializes, and runs on the CPU reference with the same result as gather-then-dense.
TEST(TestSdpaFwdPlanPaged, FrontendGraphValidatesAndExecutes)
{
    using hipdnn_frontend::graph::TensorAttributes;
    const auto tensorAttr = [](int64_t uid,
                               const char* name,
                               hipdnn_frontend::DataType type,
                               const std::vector<int64_t>& dims) {
        auto t = std::make_shared<TensorAttributes>();
        t->set_uid(uid).set_name(name).set_data_type(type).set_dim(dims).set_stride(
            hipdnn_data_sdk::utilities::generateStrides(dims));
        return t;
    };
    constexpr auto F32 = hipdnn_frontend::DataType::FLOAT;
    constexpr auto I32 = hipdnn_frontend::DataType::INT32;

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("SdpaFwdPagedFrontend")
        .set_io_data_type(F32)
        .set_compute_data_type(F32)
        .set_intermediate_data_type(F32);
    auto q = tensorAttr(1, "Q", F32, {2, 2, 3, 8});
    auto k = tensorAttr(2, "K", F32, {5, 2, 2, 8});
    auto v = tensorAttr(3, "V", F32, {5, 2, 2, 8});
    auto seqQ = tensorAttr(6, "SeqLenQ", I32, {2, 1, 1, 1});
    auto seqKv = tensorAttr(7, "SeqLenKv", I32, {2, 1, 1, 1});
    auto tableK = tensorAttr(8, "PageTableK", I32, {2, 1, 3, 1});
    auto tableV = tensorAttr(9, "PageTableV", I32, {2, 1, 3, 1});

    hipdnn_frontend::graph::SdpaAttributes attrs;
    attrs.set_name("SdpaFwdPaged")
        .set_padding_mask(true)
        .set_seq_len_q(seqQ)
        .set_seq_len_kv(seqKv)
        .set_paged_attention_k_table(tableK)
        .set_paged_attention_v_table(tableV)
        .set_paged_attention_max_seq_len_kv(6);
    auto [o, stats] = graph->sdpa(q, k, v, attrs);
    o->set_uid(4)
        .set_data_type(F32)
        .set_dim({2, 2, 3, 8})
        .set_stride(hipdnn_data_sdk::utilities::generateStrides({2, 2, 3, 8}));
    o->set_is_virtual(false);

    auto validateErr = graph->validate();
    ASSERT_TRUE(validateErr.is_good()) << validateErr.get_message();
    auto [bin, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    hipdnn_data_sdk::utilities::Tensor<float> qT({2, 2, 3, 8});
    hipdnn_data_sdk::utilities::Tensor<float> kCache({5, 2, 2, 8});
    hipdnn_data_sdk::utilities::Tensor<float> vCache({5, 2, 2, 8});
    hipdnn_data_sdk::utilities::Tensor<float> oT({2, 2, 3, 8});
    hipdnn_data_sdk::utilities::Tensor<int32_t> seqLenQ({2, 1, 1, 1});
    hipdnn_data_sdk::utilities::Tensor<int32_t> seqLenKv({2, 1, 1, 1});
    hipdnn_data_sdk::utilities::Tensor<int32_t> pageTable({2, 1, 3, 1});
    qT.fillWithRandomValues(-1.0F, 1.0F, 51);
    kCache.fillWithRandomValues(-1.0F, 1.0F, 52);
    vCache.fillWithRandomValues(-1.0F, 1.0F, 53);
    seqLenQ.setHostValue(3, 0, 0, 0, 0);
    seqLenQ.setHostValue(1, 1, 0, 0, 0);
    seqLenKv.setHostValue(6, 0, 0, 0, 0);
    seqLenKv.setHostValue(3, 1, 0, 0, 0);
    const std::vector<int32_t> blocks = {3, 0, 4, 2, 1, 0};
    for(int64_t i = 0; i < 6; ++i)
    {
        pageTable.setHostValue(blocks[static_cast<size_t>(i)], i / 3, 0, i % 3, 0);
    }

    const std::unordered_map<int64_t, void*> variantPack = {{1, qT.memory().hostData()},
                                                            {2, kCache.memory().hostData()},
                                                            {3, vCache.memory().hostData()},
                                                            {4, oT.memory().hostData()},
                                                            {6, seqLenQ.memory().hostData()},
                                                            {7, seqLenKv.memory().hostData()},
                                                            {8, pageTable.memory().hostData()},
                                                            {9, pageTable.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(bin.data(), bin.size()));
    executor.execute(bin.data(), bin.size(), variantPack);

    auto kDense = CpuFpReferenceSdpa::gatherPagedKv(kCache, pageTable, seqLenKv, 6);
    auto vDense = CpuFpReferenceSdpa::gatherPagedKv(vCache, pageTable, seqLenKv, 6);
    hipdnn_data_sdk::utilities::Tensor<float> oRef({2, 2, 3, 8});
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(
        qT,
        kDense,
        vDense,
        oRef,
        SdpaFwdOptions<float>{}.setMask(noMask).setSeqLens(&seqLenQ, &seqLenKv));
    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(oT, oRef));
}

// cuDNN's left_bound counts the diagonal key: the shared band rule rejects 0 (no keys kept) and
// values below -1, and passes positive bounds and -1 through unchanged.
TEST(TestSdpaFwdPlanBand, LeftBoundZeroIsRejected)
{
    const auto extract = [](int64_t left) {
        SdpaAttributesT attrs;
        attrs.left_bound = left;
        attrs.right_bound = 0;
        auto graphBuilder = makeRawSdpaFwdGraph(attrs);
        const auto* node = GetGraph(graphBuilder.GetBufferPointer())->nodes()->Get(0);
        return extractDiagonalBandParams(*node->attributes_as_SdpaAttributes(), "test");
    };
    EXPECT_EQ(extract(1).leftBound, 1);
    EXPECT_EQ(extract(-1).leftBound, -1);
    EXPECT_THROW(extract(0), std::invalid_argument);
    EXPECT_THROW(extract(-2), std::invalid_argument);
}

// Attention sink: FLOAT [1, H, 1, 1] (cuDNN). The raw graph has H = 2.
TEST(TestSdpaFwdPlanSink, ApplicabilityFollowsCudnnShape)
{
    const auto applicable = [](DataType type, const std::vector<int64_t>& dims) {
        SdpaAttributesT attrs;
        attrs.sink_token_tensor_uid = 10;
        auto graphBuilder
            = makeRawSdpaFwdGraph(attrs, std::nullopt, std::nullopt, RawSinkSpec{type, dims});
        CpuReferenceGraphExecutor executor;
        return executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    };
    EXPECT_TRUE(applicable(DataType::FLOAT, {1, 2, 1, 1}));
    EXPECT_FALSE(applicable(DataType::HALF, {1, 2, 1, 1}));
    EXPECT_FALSE(applicable(DataType::FLOAT, {1, 1, 1, 1}));
    EXPECT_FALSE(applicable(DataType::FLOAT, {2}));
}

// Dropout: cuDNN's explicit-mask form (FLOAT mask [B|1, H|1, Sq, Skv] + one-element FLOAT
// dropout_scale) is supported; Philox dropout, RNG_DUMP, a mask without a scale, or a non-FLOAT
// mask are declined. The raw graph is [1, 2, 4, 8] with Sq = Skv = 4; mask uid 20, scale uid 21.
TEST(TestSdpaFwdPlanDropout, Applicability)
{
    const auto applicable = [](SdpaAttributesT attrs, const std::vector<RawExtraTensor>& extras) {
        auto graphBuilder
            = makeRawSdpaFwdGraph(attrs, std::nullopt, std::nullopt, std::nullopt, extras);
        CpuReferenceGraphExecutor executor;
        return executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    };
    SdpaAttributesT withMask;
    withMask.dropout_mask_tensor_uid = 20;
    withMask.dropout_scale_tensor_uid = 21;
    const RawExtraTensor mask{20, "DropoutMask", DataType::FLOAT, {1, 1, 4, 4}};
    const RawExtraTensor scale{21, "DropoutScale", DataType::FLOAT, {1}};
    EXPECT_TRUE(applicable(withMask, {mask, scale}));
    EXPECT_TRUE(applicable(withMask, {{20, "DropoutMask", DataType::FLOAT, {1, 2, 4, 4}}, scale}));
    EXPECT_FALSE(applicable(withMask, {{20, "DropoutMask", DataType::HALF, {1, 1, 4, 4}}, scale}));
    EXPECT_FALSE(applicable(withMask, {{20, "DropoutMask", DataType::FLOAT, {1, 1, 4, 3}}, scale}));

    SdpaAttributesT maskOnly;
    maskOnly.dropout_mask_tensor_uid = 20;
    EXPECT_FALSE(applicable(maskOnly, {mask}));

    SdpaAttributesT philox = withMask;
    philox.dropout_probability = 0.1F;
    EXPECT_FALSE(applicable(philox, {mask, scale}));

    SdpaAttributesT rngDump = withMask;
    rngDump.rng_dump_tensor_uid = 22;
    EXPECT_FALSE(applicable(rngDump, {mask, scale}));
}

// A dropout graph runs through the executor and equals the reference called directly.
TEST(TestSdpaFwdPlanDropout, ExecuteMatchesDirectReference)
{
    SdpaAttributesT attrs;
    attrs.dropout_mask_tensor_uid = 20;
    attrs.dropout_scale_tensor_uid = 21;
    auto graphBuilder = makeRawSdpaFwdGraph(attrs,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            {{20, "DropoutMask", DataType::FLOAT, {1, 2, 4, 4}},
                                             {21, "DropoutScale", DataType::FLOAT, {1}}});

    const std::vector<int64_t> dims = {1, 2, 4, 8};
    hipdnn_data_sdk::utilities::Tensor<float> q(dims);
    hipdnn_data_sdk::utilities::Tensor<float> k(dims);
    hipdnn_data_sdk::utilities::Tensor<float> v(dims);
    hipdnn_data_sdk::utilities::Tensor<float> o(dims);
    hipdnn_data_sdk::utilities::Tensor<float> keepMask({1, 2, 4, 4});
    hipdnn_data_sdk::utilities::Tensor<float> scale({1});
    q.fillWithRandomValues(-1.0F, 1.0F, 71);
    k.fillWithRandomValues(-1.0F, 1.0F, 72);
    v.fillWithRandomValues(-1.0F, 1.0F, 73);
    for(size_t i = 0; i < keepMask.elementCount(); ++i)
    {
        keepMask.memory().hostData()[i] = (i % 3 == 0) ? 0.0F : 1.0F;
    }
    scale.fillWithValue(1.5F);

    const std::unordered_map<int64_t, void*> variantPack = {{1, q.memory().hostData()},
                                                            {2, k.memory().hostData()},
                                                            {3, v.memory().hostData()},
                                                            {4, o.memory().hostData()},
                                                            {20, keepMask.memory().hostData()},
                                                            {21, scale.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);

    hipdnn_data_sdk::utilities::Tensor<float> oRef(dims);
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(
        q, k, v, oRef, SdpaFwdOptions<float>{}.setMask(noMask).setDropout(&keepMask, 1.5F));
    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(o, oRef));
}

// Philox dropout is checked through RNG_DUMP (decision C3): with dropout_probability, seed, offset
// and an rng_dump output FLOAT [B, H, Sq, Skv], the reference reads the engine's dumped keep mask
// as an explicit mask with scale 1 / (1 - p). Without rng_dump there is no exact oracle.
// Raw graph [1, 2, 4, 8]; seed uid 23, offset uid 24, rng_dump uid 22.
TEST(TestSdpaFwdPlanDropout, PhiloxNeedsRngDump)
{
    const auto applicable = [](SdpaAttributesT attrs, const std::vector<RawExtraTensor>& extras) {
        auto graphBuilder
            = makeRawSdpaFwdGraph(attrs, std::nullopt, std::nullopt, std::nullopt, extras);
        CpuReferenceGraphExecutor executor;
        return executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    };
    SdpaAttributesT philox;
    philox.dropout_probability = 0.25F;
    philox.seed_tensor_uid = 23;
    philox.offset_tensor_uid = 24;
    const RawExtraTensor seed{23, "Seed", DataType::INT64, {1}};
    const RawExtraTensor offset{24, "Offset", DataType::INT64, {1}};
    const RawExtraTensor dump{22, "RngDump", DataType::FLOAT, {1, 2, 4, 4}};
    EXPECT_FALSE(applicable(philox, {seed, offset}));

    SdpaAttributesT withDump = philox;
    withDump.rng_dump_tensor_uid = 22;
    EXPECT_TRUE(applicable(withDump, {seed, offset, dump}));
    EXPECT_FALSE(
        applicable(withDump, {seed, offset, {22, "RngDump", DataType::FLOAT, {1, 1, 4, 4}}}));
    EXPECT_FALSE(
        applicable(withDump, {seed, offset, {22, "RngDump", DataType::HALF, {1, 2, 4, 4}}}));

    SdpaAttributesT certainDrop = withDump;
    certainDrop.dropout_probability = 1.0F;
    EXPECT_FALSE(applicable(certainDrop, {seed, offset, dump}));
}

TEST(TestSdpaFwdPlanDropout, PhiloxWithRngDumpUsesDumpedMask)
{
    SdpaAttributesT attrs;
    attrs.dropout_probability = 0.25F;
    attrs.seed_tensor_uid = 23;
    attrs.offset_tensor_uid = 24;
    attrs.rng_dump_tensor_uid = 22;
    auto graphBuilder = makeRawSdpaFwdGraph(attrs,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            {{23, "Seed", DataType::INT64, {1}},
                                             {24, "Offset", DataType::INT64, {1}},
                                             {22, "RngDump", DataType::FLOAT, {1, 2, 4, 4}}});

    const std::vector<int64_t> dims = {1, 2, 4, 8};
    hipdnn_data_sdk::utilities::Tensor<float> q(dims);
    hipdnn_data_sdk::utilities::Tensor<float> k(dims);
    hipdnn_data_sdk::utilities::Tensor<float> v(dims);
    hipdnn_data_sdk::utilities::Tensor<float> o(dims);
    hipdnn_data_sdk::utilities::Tensor<float> dumpedMask({1, 2, 4, 4});
    hipdnn_data_sdk::utilities::Tensor<int64_t> seed({1});
    hipdnn_data_sdk::utilities::Tensor<int64_t> offset({1});
    q.fillWithRandomValues(-1.0F, 1.0F, 81);
    k.fillWithRandomValues(-1.0F, 1.0F, 82);
    v.fillWithRandomValues(-1.0F, 1.0F, 83);
    for(size_t i = 0; i < dumpedMask.elementCount(); ++i)
    {
        dumpedMask.memory().hostData()[i] = (i % 4 == 1) ? 0.0F : 1.0F;
    }
    seed.fillWithValue(1234);
    offset.fillWithValue(0);

    const std::unordered_map<int64_t, void*> variantPack = {{1, q.memory().hostData()},
                                                            {2, k.memory().hostData()},
                                                            {3, v.memory().hostData()},
                                                            {4, o.memory().hostData()},
                                                            {22, dumpedMask.memory().hostData()},
                                                            {23, seed.memory().hostData()},
                                                            {24, offset.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);

    hipdnn_data_sdk::utilities::Tensor<float> oRef(dims);
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(
        q,
        k,
        v,
        oRef,
        SdpaFwdOptions<float>{}.setMask(noMask).setDropout(&dumpedMask, 1.0F / 0.75F));
    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(o, oRef));
}

// cuDNN's Max and Sum_exp row statistics are FLOAT outputs [B, H, Sq, 1]; a missing or non-FLOAT
// tensor is not applicable. The raw graph is [1, 2, 4, 8].
TEST(TestSdpaFwdPlanRowStats, ApplicabilityRequiresFloatOutputs)
{
    const auto applicable = [](bool max, DataType type, bool present) {
        SdpaAttributesT attrs;
        std::vector<RawExtraTensor> extras;
        if(max)
        {
            attrs.max_tensor_uid = 50;
        }
        else
        {
            attrs.sum_exp_tensor_uid = 50;
        }
        if(present)
        {
            extras.push_back({50, "stat", type, {1, 2, 4, 1}});
        }
        auto graphBuilder
            = makeRawSdpaFwdGraph(attrs, std::nullopt, std::nullopt, std::nullopt, extras);
        CpuReferenceGraphExecutor executor;
        return executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    };
    for(const bool max : {true, false})
    {
        EXPECT_TRUE(applicable(max, DataType::FLOAT, true)) << max;
        EXPECT_FALSE(applicable(max, DataType::HALF, true)) << max;
        EXPECT_FALSE(applicable(max, DataType::FLOAT, false)) << max;
    }
}

// cuDNN ALiBi is applicable only with top-left causal masking (SEM-021): right bound 0 (directly
// or through causal_mask) and top-left alignment.
TEST(TestSdpaFwdPlanAlibi, ApplicableOnlyWithTopLeftCausal)
{
    const auto applicable = [](SdpaAttributesT attrs) {
        attrs.alibi_mask = true;
        auto graphBuilder = makeRawSdpaFwdGraph(attrs);
        CpuReferenceGraphExecutor executor;
        return executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    };
    SdpaAttributesT causal;
    causal.right_bound = 0;
    EXPECT_TRUE(applicable(causal));
    SdpaAttributesT deprecatedCausal;
    deprecatedCausal.causal_mask = true;
    EXPECT_TRUE(applicable(deprecatedCausal));
    EXPECT_FALSE(applicable(SdpaAttributesT{}));
    SdpaAttributesT window;
    window.right_bound = 2;
    EXPECT_FALSE(applicable(window));
    SdpaAttributesT bottomRight;
    bottomRight.right_bound = 0;
    bottomRight.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_FALSE(applicable(bottomRight));
}

// The executor writes Max and Sum_exp from the same row statistics as the LSE:
// LSE = Max + log(Sum_exp), and both equal the reference called directly.
TEST(TestSdpaFwdPlanRowStats, ExecuteWritesMaxAndSumExpConsistentWithLse)
{
    SdpaAttributesT attrs;
    attrs.stats_tensor_uid = 49;
    attrs.max_tensor_uid = 50;
    attrs.sum_exp_tensor_uid = 51;
    attrs.right_bound = 0;
    auto graphBuilder = makeRawSdpaFwdGraph(attrs,
                                            std::nullopt,
                                            std::nullopt,
                                            std::nullopt,
                                            {{49, "lse", DataType::FLOAT, {1, 2, 4, 1}},
                                             {50, "max", DataType::FLOAT, {1, 2, 4, 1}},
                                             {51, "sum_exp", DataType::FLOAT, {1, 2, 4, 1}}});

    const std::vector<int64_t> dims = {1, 2, 4, 8};
    hipdnn_data_sdk::utilities::Tensor<float> q(dims);
    hipdnn_data_sdk::utilities::Tensor<float> k(dims);
    hipdnn_data_sdk::utilities::Tensor<float> v(dims);
    hipdnn_data_sdk::utilities::Tensor<float> o(dims);
    hipdnn_data_sdk::utilities::Tensor<float> lse({1, 2, 4, 1});
    hipdnn_data_sdk::utilities::Tensor<float> rowMax({1, 2, 4, 1});
    hipdnn_data_sdk::utilities::Tensor<float> rowSumExp({1, 2, 4, 1});
    q.fillWithRandomValues(-1.0F, 1.0F, 61);
    k.fillWithRandomValues(-1.0F, 1.0F, 62);
    v.fillWithRandomValues(-1.0F, 1.0F, 63);
    const std::unordered_map<int64_t, void*> variantPack = {{1, q.memory().hostData()},
                                                            {2, k.memory().hostData()},
                                                            {3, v.memory().hostData()},
                                                            {4, o.memory().hostData()},
                                                            {49, lse.memory().hostData()},
                                                            {50, rowMax.memory().hostData()},
                                                            {51, rowSumExp.memory().hostData()}};
    CpuReferenceGraphExecutor executor;
    executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);

    hipdnn_data_sdk::utilities::Tensor<float> oRef(dims);
    hipdnn_data_sdk::utilities::Tensor<float> maxRef({1, 2, 4, 1});
    hipdnn_data_sdk::utilities::Tensor<float> sumExpRef({1, 2, 4, 1});
    CpuFpReferenceSdpa::forward(
        q,
        k,
        v,
        oRef,
        SdpaFwdOptions<float>{}.setBand(-1, 0, true).setRowStats(&maxRef, &sumExpRef));
    for(int64_t h = 0; h < 2; ++h)
    {
        for(int64_t sq = 0; sq < 4; ++sq)
        {
            const auto m = rowMax.getHostValue(0, h, sq, 0);
            const auto se = rowSumExp.getHostValue(0, h, sq, 0);
            EXPECT_NEAR(m, maxRef.getHostValue(0, h, sq, 0), 1e-6F);
            EXPECT_NEAR(se, sumExpRef.getHostValue(0, h, sq, 0), 1e-5F);
            EXPECT_GE(se, 1.0F); // the max term contributes exp(0)
            EXPECT_NEAR(lse.getHostValue(0, h, sq, 0), m + std::log(se), 1e-5F);
        }
    }
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableAndExecutesWithStatsOutput)
{
    // A graph configured to emit the softmax log-sum-exp (LSE) stats output is supported.
    // The plan must be applicable, build into the concrete SdpaFwdPlan, report the stats uid
    // among its output tensor ids, and write rank-4 [B, H, Sq, 1] LSE values when executed.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/std::nullopt,
                                        /*rightBound=*/std::nullopt,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                                        /*alibiMask=*/false,
                                        /*generateStats=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    ASSERT_TRUE(nodeAttributes->stats_tensor_uid().has_value());
    const int64_t statsUid = nodeAttributes->stats_tensor_uid().value();

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_TRUE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    auto builtPlan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
    auto* concretePlan = dynamic_cast<SdpaFwdPlan<float, float, float, float>*>(builtPlan.get());
    ASSERT_NE(concretePlan, nullptr);

    // The stats uid must be reported as an output so the harness binds and validates it.
    const auto outputIds = concretePlan->getOutputTensorIds();
    EXPECT_NE(std::find(outputIds.begin(), outputIds.end(), statsUid), outputIds.end());

    // Allocate the rank-4 LSE output buffer and bind it before executing.
    hipdnn_data_sdk::utilities::Tensor<float> lseTensor({qDims[0], qDims[1], qDims[2], 1});
    lseTensor.fillWithValue(-1.0f);
    variantPack[statsUid] = lseTensor.memory().hostData();

    builtPlan->execute(variantPack);

    // Independently compute the expected LSE via the direct reference and compare.
    SdpaFwdTensorBundle<float> directBundle(qDims, kDims, vDims, /*seed=*/1);
    hipdnn_data_sdk::utilities::Tensor<float> directLse({qDims[0], qDims[1], qDims[2], 1});
    CpuFpReferenceSdpa::forward<float, float, float, float, float>(
        directBundle.qTensor,
        directBundle.kTensor,
        directBundle.vTensor,
        directBundle.oTensor,
        SdpaFwdOptions<float>{}.setLse(&directLse));

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> lseValidation(tolerance, tolerance);
    EXPECT_TRUE(lseValidation.allClose(directLse, lseTensor))
        << "Plan-produced LSE does not match the direct CpuFpReferenceSdpa LSE output.";
}

TEST(TestSdpaFwdPlanBuilder, DeprecatedCausalMaskMatchesExplicitTopLeftBounds)
{
    // The dispatcher in SdpaFwdPlanBuilder maps the deprecated causal_mask=true flag
    // to (leftBound=-1, rightBound=0, TOP_LEFT). This test verifies that mapping by
    // running two graphs that should produce bit-for-bit identical output:
    //   (a) causal_mask=true                         (deprecated path)
    //   (b) leftBound=-1, rightBound=0, TOP_LEFT     (modern path)
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> deprecatedBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> explicitBundle(qDims, kDims, vDims, seed);

    // (a) Deprecated causal_mask=true
    auto deprecatedGraphTuple = buildSdpaFwdGraph(deprecatedBundle,
                                                  DataType::FLOAT,
                                                  /*causalMask=*/true);
    auto& deprecatedGraph = std::get<0>(deprecatedGraphTuple);
    auto [depBin, depErr] = deprecatedGraph->to_binary();
    ASSERT_TRUE(depErr.is_good()) << depErr.get_message();
    const GraphWrapper depWrapper(depBin.data(), depBin.size());

    // (b) Explicit (leftBound=-1, rightBound=0, TOP_LEFT)
    auto explicitGraphTuple = buildSdpaFwdGraph(explicitBundle,
                                                DataType::FLOAT,
                                                /*causalMask=*/false,
                                                /*causalMaskBottomRight=*/false,
                                                /*leftBound=*/-1,
                                                /*rightBound=*/0,
                                                hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& explicitGraph = std::get<0>(explicitGraphTuple);
    auto [expBin, expErr] = explicitGraph->to_binary();
    ASSERT_TRUE(expErr.is_good()) << expErr.get_message();
    const GraphWrapper expWrapper(expBin.data(), expBin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    // Execute deprecated-path plan
    {
        auto plan = planBuilder.buildNodePlan(depWrapper, depWrapper.getNode(0));
        const auto* attrs = depWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = deprecatedBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = deprecatedBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = deprecatedBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = deprecatedBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Execute explicit-bounds plan
    {
        auto plan = planBuilder.buildNodePlan(expWrapper, expWrapper.getNode(0));
        const auto* attrs = expWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = explicitBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = explicitBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = explicitBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = explicitBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Both code paths feed the same arguments into CpuFpReferenceSdpa::forward, so
    // results must match to within bit-for-bit tolerance.
    const float tolerance = 0.0f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(deprecatedBundle.oTensor, explicitBundle.oTensor))
        << "Deprecated causal_mask=true should produce identical output to "
           "leftBound=-1, rightBound=0, TOP_LEFT alignment.";
}

TEST(TestSdpaFwdPlanBuilder, DeprecatedCausalMaskBottomRightMatchesExplicitBottomRightBounds)
{
    // The dispatcher maps the deprecated causal_mask_bottom_right=true flag to
    // (leftBound=-1, rightBound=0, BOTTOM_RIGHT). Verify by comparing against an
    // explicit bottom-right window configuration. Use Sq != Skv so that TOP_LEFT
    // and BOTTOM_RIGHT alignments produce different output, catching any bug
    // where the dispatcher forgets to set the alignment to BOTTOM_RIGHT.
    const std::vector<int64_t> qDims = {1, 2, 2, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> deprecatedBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> explicitBundle(qDims, kDims, vDims, seed);

    // (a) Deprecated causal_mask_bottom_right=true
    auto deprecatedGraphTuple = buildSdpaFwdGraph(deprecatedBundle,
                                                  DataType::FLOAT,
                                                  /*causalMask=*/false,
                                                  /*causalMaskBottomRight=*/true,
                                                  /*leftBound=*/std::nullopt,
                                                  /*rightBound=*/std::nullopt,
                                                  hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& deprecatedGraph = std::get<0>(deprecatedGraphTuple);
    auto [depBin, depErr] = deprecatedGraph->to_binary();
    ASSERT_TRUE(depErr.is_good()) << depErr.get_message();
    const GraphWrapper depWrapper(depBin.data(), depBin.size());

    // (b) Explicit (leftBound=-1, rightBound=0, BOTTOM_RIGHT)
    auto explicitGraphTuple = buildSdpaFwdGraph(explicitBundle,
                                                DataType::FLOAT,
                                                /*causalMask=*/false,
                                                /*causalMaskBottomRight=*/false,
                                                /*leftBound=*/-1,
                                                /*rightBound=*/0,
                                                hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
    auto& explicitGraph = std::get<0>(explicitGraphTuple);
    auto [expBin, expErr] = explicitGraph->to_binary();
    ASSERT_TRUE(expErr.is_good()) << expErr.get_message();
    const GraphWrapper expWrapper(expBin.data(), expBin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    // Execute deprecated-path plan
    {
        auto plan = planBuilder.buildNodePlan(depWrapper, depWrapper.getNode(0));
        const auto* attrs = depWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = deprecatedBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = deprecatedBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = deprecatedBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = deprecatedBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Execute explicit-bounds plan
    {
        auto plan = planBuilder.buildNodePlan(expWrapper, expWrapper.getNode(0));
        const auto* attrs = expWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = explicitBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = explicitBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = explicitBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = explicitBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    const float tolerance = 0.0f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(deprecatedBundle.oTensor, explicitBundle.oTensor))
        << "Deprecated causal_mask_bottom_right=true should produce identical output to "
           "leftBound=-1, rightBound=0, BOTTOM_RIGHT alignment.";
}

namespace
{

struct MaskSpelling
{
    bool causalMask;
    bool causalMaskBottomRight;
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    hipdnn_frontend::DiagonalAlignment alignment;
};

struct DeprecatedCausalMaskMergeCase
{
    const char* name;
    MaskSpelling deprecated;
    MaskSpelling merged;
};

class TestSdpaFwdPlanBuilderDeprecatedCausalMask
    : public TestWithParam<DeprecatedCausalMaskMergeCase>
{
};

void executeSdpaFwdPlan(SdpaFwdTensorBundle<float>& bundle, const MaskSpelling& mask)
{
    auto graphTuple = buildSdpaFwdGraph(bundle,
                                        DataType::FLOAT,
                                        mask.causalMask,
                                        mask.causalMaskBottomRight,
                                        mask.leftBound,
                                        mask.rightBound,
                                        mask.alignment);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    planBuilder.buildNodePlan(wrapper, wrapper.getNode(0))->execute(variantPack);
}

} // namespace

TEST_P(TestSdpaFwdPlanBuilderDeprecatedCausalMask, MatchesMergedExplicitBounds)
{
    const auto& param = GetParam();
    // Sq != Skv so TOP_LEFT and BOTTOM_RIGHT alignments produce different output.
    const std::vector<int64_t> qDims = {1, 2, 2, 8};
    const std::vector<int64_t> kvDims = {1, 2, 4, 8};
    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> deprecatedBundle(qDims, kvDims, kvDims, seed);
    SdpaFwdTensorBundle<float> mergedBundle(qDims, kvDims, kvDims, seed);

    executeSdpaFwdPlan(deprecatedBundle, param.deprecated);
    executeSdpaFwdPlan(mergedBundle, param.merged);

    const CpuFpReferenceValidation<float> exactValidation(0.0f, 0.0f);
    EXPECT_TRUE(exactValidation.allClose(deprecatedBundle.oTensor, mergedBundle.oTensor));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestSdpaFwdPlanBuilderDeprecatedCausalMask,
    Values(
        DeprecatedCausalMaskMergeCase{
            "CausalWithRightBoundOnly",
            {true, false, std::nullopt, 0, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            {false, false, -1, 0, hipdnn_frontend::DiagonalAlignment::TOP_LEFT}},
        DeprecatedCausalMaskMergeCase{
            "CausalWithUnboundedRight",
            {true, false, std::nullopt, -1, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            {false, false, -1, 0, hipdnn_frontend::DiagonalAlignment::TOP_LEFT}},
        DeprecatedCausalMaskMergeCase{
            "CausalWithLeftBoundOnly",
            {true, false, 1, std::nullopt, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            {false, false, 1, 0, hipdnn_frontend::DiagonalAlignment::TOP_LEFT}},
        DeprecatedCausalMaskMergeCase{
            "CausalWithBottomRightAlignment",
            {true,
             false,
             std::nullopt,
             std::nullopt,
             hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT},
            {false, false, -1, 0, hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT}},
        DeprecatedCausalMaskMergeCase{
            "BottomRightWithRightBoundOnly",
            {false, true, std::nullopt, 1, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            {false, false, -1, 1, hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT}},
        DeprecatedCausalMaskMergeCase{
            "BottomRightWithLeftBoundOnly",
            {false, true, 1, std::nullopt, hipdnn_frontend::DiagonalAlignment::TOP_LEFT},
            {false, false, 1, 0, hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT}}),
    [](const TestParamInfo<DeprecatedCausalMaskMergeCase>& info) {
        return std::string(info.param.name);
    });

TEST(TestSdpaFwdPlanBuilder, BothDeprecatedCausalMasksThrow)
{
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    SdpaFwdTensorBundle<float> bundle(dims, dims, dims, getGlobalTestSeed());

    auto graphTuple = buildSdpaFwdGraph(bundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/true,
                                        /*causalMaskBottomRight=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(wrapper, wrapper.getNode(0)), std::invalid_argument);
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableFp8RequiresDescale)
{
    // FP8 inputs require q/k/v descales (mirrors AITER's TORCH_CHECK). The dispatcher
    // must reject fp8-without-descale and accept fp8-with-descale.
    const std::vector<int64_t> dims = {1, 2, 4, 8};

    const SdpaFwdPlanBuilder<DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::BFLOAT16>
        fp8Builder;

    {
        auto graph = buildDescaleSdpaFwdGraph(dims,
                                              dims,
                                              dims,
                                              hipdnn_frontend::DataType::FP8_E4M3,
                                              hipdnn_frontend::DataType::BFLOAT16,
                                              /*withDescale=*/false);
        auto [bin, err] = graph->to_binary();
        ASSERT_TRUE(err.is_good()) << err.get_message();
        const GraphWrapper wrapper(bin.data(), bin.size());
        EXPECT_FALSE(fp8Builder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
            << "FP8 inputs without descales must be rejected";
    }

    {
        auto graph = buildDescaleSdpaFwdGraph(dims,
                                              dims,
                                              dims,
                                              hipdnn_frontend::DataType::FP8_E4M3,
                                              hipdnn_frontend::DataType::BFLOAT16,
                                              /*withDescale=*/true);
        auto [bin, err] = graph->to_binary();
        ASSERT_TRUE(err.is_good()) << err.get_message();
        const GraphWrapper wrapper(bin.data(), bin.size());
        EXPECT_TRUE(fp8Builder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
            << "FP8 inputs with q/k/v descales must be accepted";
    }
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsDescaleForNonFp8)
{
    // Non-FP8 inputs must not carry descales; the dispatcher rejects such graphs.
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    auto graph = buildDescaleSdpaFwdGraph(dims,
                                          dims,
                                          dims,
                                          hipdnn_frontend::DataType::FLOAT,
                                          hipdnn_frontend::DataType::FLOAT,
                                          /*withDescale=*/true);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatBuilder;
    EXPECT_FALSE(floatBuilder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
        << "Non-FP8 inputs carrying descales must be rejected";
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsFp8WithSoftmaxDescale)
{
    // The v3 forward kernel supports only q/k/v descales. Softmax/output
    // (de)quantization (descale_s, scale_s/o, amax) is not modeled, so an fp8 graph
    // that additionally carries descale_s must be rejected even though q/k/v descales
    // are present.
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    auto graph = buildDescaleSdpaFwdGraph(dims,
                                          dims,
                                          dims,
                                          hipdnn_frontend::DataType::FP8_E4M3,
                                          hipdnn_frontend::DataType::BFLOAT16,
                                          /*withDescale=*/true,
                                          /*withDescaleS=*/true);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());
    const SdpaFwdPlanBuilder<DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::BFLOAT16>
        fp8Builder;
    EXPECT_FALSE(fp8Builder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
        << "fp8 graphs carrying descale_s (softmax quant) must be rejected";
}

TEST(TestSdpaFwdPlan, ExecuteFp8WithDescale)
{
    // Drives the full FP8 path through SdpaFwdPlanBuilder -> SdpaFwdPlan::execute:
    // descale UIDs are read in buildNodePlan and the descale tensors are passed to
    // CpuFpReferenceSdpa. Validates against an independent dequantized-float reference.
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::fp8_e4m3;

    const int64_t batch = 1;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4;
    const int64_t headDim = 8;
    const std::vector<int64_t> dims = {batch, numHeads, seqLen, headDim};

    Tensor<fp8_e4m3> qTensor(dims);
    Tensor<fp8_e4m3> kTensor(dims);
    Tensor<fp8_e4m3> vTensor(dims);
    Tensor<bfloat16> oTensor(dims);

    // Deterministic, exactly-representable fp8 values (small multiples of 1/4).
    auto fillPattern = [&](Tensor<fp8_e4m3>& t, int64_t phase) {
        int64_t flat = 0;
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                for(int64_t s = 0; s < seqLen; ++s)
                {
                    for(int64_t d = 0; d < headDim; ++d)
                    {
                        const float value = static_cast<float>(((flat + phase) % 5) - 2) * 0.25f;
                        t.setHostValue(fp8_e4m3(value), b, h, s, d);
                        ++flat;
                    }
                }
            }
        }
    };
    fillPattern(qTensor, 0);
    fillPattern(kTensor, 1);
    fillPattern(vTensor, 2);

    Tensor<float> descaleQ({1, 1, 1, 1});
    Tensor<float> descaleK({1, 1, 1, 1});
    Tensor<float> descaleV({1, 1, 1, 1});
    descaleQ.fillWithValue(2.0f);
    descaleK.fillWithValue(0.5f);
    descaleV.fillWithValue(1.5f);

    auto graph = buildDescaleSdpaFwdGraph(dims,
                                          dims,
                                          dims,
                                          hipdnn_frontend::DataType::FP8_E4M3,
                                          hipdnn_frontend::DataType::BFLOAT16,
                                          /*withDescale=*/true);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());

    const SdpaFwdPlanBuilder<DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::BFLOAT16>
        planBuilder;
    ASSERT_TRUE(planBuilder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()));
    auto plan = planBuilder.buildNodePlan(wrapper, wrapper.getNode(0));

    const auto* nodeAttributes = wrapper.getNode(0).attributes_as_SdpaAttributes();
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = oTensor.memory().hostData();
    variantPack[nodeAttributes->descale_q_tensor_uid().value()] = descaleQ.memory().hostData();
    variantPack[nodeAttributes->descale_k_tensor_uid().value()] = descaleK.memory().hostData();
    variantPack[nodeAttributes->descale_v_tensor_uid().value()] = descaleV.memory().hostData();
    plan->execute(variantPack);

    // Independent reference: dequantize inputs to float (decode(fp8) * descale) and run
    // the plain (descale-free) reference.
    Tensor<float> qF(dims);
    Tensor<float> kF(dims);
    Tensor<float> vF(dims);
    Tensor<bfloat16> oExpected(dims);
    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            for(int64_t s = 0; s < seqLen; ++s)
            {
                for(int64_t d = 0; d < headDim; ++d)
                {
                    qF.setHostValue(
                        static_cast<float>(qTensor.getHostValue(b, h, s, d)) * 2.0f, b, h, s, d);
                    kF.setHostValue(
                        static_cast<float>(kTensor.getHostValue(b, h, s, d)) * 0.5f, b, h, s, d);
                    vF.setHostValue(
                        static_cast<float>(vTensor.getHostValue(b, h, s, d)) * 1.5f, b, h, s, d);
                }
            }
        }
    }
    CpuFpReferenceSdpa::forward<float, float, float, bfloat16, float>(qF, kF, vF, oExpected);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(oExpected, oTensor))
        << "FP8 plan output does not match the dequantized-input reference";
}

namespace
{

template <class MaskT>
std::vector<uint8_t> serializeSdpaFwdGraph(SdpaFwdTensorBundle<float, MaskT>& bundle,
                                           std::unordered_map<int64_t, void*>* variantPack
                                           = nullptr)
{
    auto graphTuple = buildSdpaFwdGraph(bundle, DataType::FLOAT);
    if(variantPack != nullptr)
    {
        *variantPack = std::get<1>(graphTuple);
    }
    auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    return serializedGraph;
}

// Runs the bundle's graph through the CPU reference executor (key -> registry -> plan).
template <class MaskT>
void executeSdpaFwd(SdpaFwdTensorBundle<float, MaskT>& bundle)
{
    std::unordered_map<int64_t, void*> variantPack;
    auto serializedGraph = serializeSdpaFwdGraph(bundle, &variantPack);
    CpuReferenceGraphExecutor executor;
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);
}

} // namespace

template <class MaskT>
class SdpaFwdPlanMaskDtype : public Test
{
};

using SupportedMaskTypes
    = Types<float, hipdnn_data_sdk::types::half, hipdnn_data_sdk::types::bfloat16>;
TYPED_TEST_SUITE(SdpaFwdPlanMaskDtype, SupportedMaskTypes, );

// A half/bfloat16 mask must give exactly the result of the same graph with a float mask holding
// the same values. Reading the mask buffer as float instead gives a different result.
TYPED_TEST(SdpaFwdPlanMaskDtype, OutputMatchesFloatMask)
{
    const unsigned int seed = getGlobalTestSeed();

    SdpaFwdTensorBundle<float, float> floatMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS, seed);
    emplaceNonUniformMask(floatMaskBundle);
    ASSERT_NO_FATAL_FAILURE(executeSdpaFwd(floatMaskBundle));

    SdpaFwdTensorBundle<float, TypeParam> typedMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS, seed);
    emplaceNonUniformMask(typedMaskBundle);
    ASSERT_NO_FATAL_FAILURE(executeSdpaFwd(typedMaskBundle));

    const CpuFpReferenceValidation<float> exact(EXACT_TOLERANCE, EXACT_TOLERANCE);
    EXPECT_TRUE(exact.allClose(floatMaskBundle.oTensor, typedMaskBundle.oTensor));

    // The mask must actually influence the result, otherwise the comparison above is vacuous.
    SdpaFwdTensorBundle<float, float> noMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS, seed);
    ASSERT_NO_FATAL_FAILURE(executeSdpaFwd(noMaskBundle));
    EXPECT_FALSE(exact.allClose(noMaskBundle.oTensor, typedMaskBundle.oTensor));
}

TYPED_TEST(SdpaFwdPlanMaskDtype, ExecutorAcceptsGraph)
{
    SdpaFwdTensorBundle<float, TypeParam> bundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    emplaceNonUniformMask(bundle);
    auto serializedGraph = serializeSdpaFwdGraph(bundle);

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
}

TEST(TestSdpaFwdPlanMask, ExecutorAcceptsGraphWithoutMask)
{
    SdpaFwdTensorBundle<float> bundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    auto serializedGraph = serializeSdpaFwdGraph(bundle);

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
}

TEST(TestSdpaFwdPlanMask, UnsupportedMaskDtypeHasNoPlan)
{
    SdpaFwdTensorBundle<float, int32_t> bundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    bundle.maskTensor.emplace(std::vector<int64_t>{1, 2, 4, 6});
    std::unordered_map<int64_t, void*> variantPack;
    auto serializedGraph = serializeSdpaFwdGraph(bundle, &variantPack);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));

    try
    {
        executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);
        FAIL() << "An INT32 mask must not resolve to a plan";
    }
    catch(const CpuReferenceNotApplicableError& e)
    {
        FAIL() << "Expected the registry error naming the mask dtype, got not-applicable: "
               << e.what();
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("mask=INT32"), std::string::npos) << e.what();
    }
}

TEST(TestSdpaFwdPlanMask, MaskTensorWithUnsetDtypeIsNotApplicable)
{
    auto graphBuilder = makeRawSdpaFwdGraph(SdpaAttributesT{}, DataType::UNSET);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
    EXPECT_THROW(executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), {}),
                 CpuReferenceNotApplicableError);
}

TEST(TestSdpaFwdPlanMask, DanglingMaskUidIsNotApplicable)
{
    SdpaAttributesT attrs;
    attrs.attn_mask_tensor_uid = 99;
    auto graphBuilder = makeRawSdpaFwdGraph(attrs);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
    EXPECT_THROW(executor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), {}),
                 std::out_of_range);
}

// A builder claims only graphs whose mask dtype is exactly its own, so the no-mask builder never
// reads a mask buffer as its placeholder float type.
TEST(TestSdpaFwdPlanMask, BuilderAcceptsOnlyMatchingMaskDtype)
{
    using hipdnn_data_sdk::types::half;

    SdpaFwdTensorBundle<float, half> halfMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    emplaceNonUniformMask(halfMaskBundle);
    auto halfMaskGraph = serializeSdpaFwdGraph(halfMaskBundle);
    const GraphWrapper halfMaskWrapper(halfMaskGraph.data(), halfMaskGraph.size());

    SdpaFwdTensorBundle<float, float> floatMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    emplaceNonUniformMask(floatMaskBundle);
    auto floatMaskGraph = serializeSdpaFwdGraph(floatMaskBundle);
    const GraphWrapper floatMaskWrapper(floatMaskGraph.data(), floatMaskGraph.size());

    SdpaFwdTensorBundle<float, float> noMaskBundle(MASK_QDIMS, MASK_KDIMS, MASK_VDIMS);
    auto noMaskGraph = serializeSdpaFwdGraph(noMaskBundle);
    const GraphWrapper noMaskWrapper(noMaskGraph.data(), noMaskGraph.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::FLOAT,
                             DataType::HALF>
        halfMaskBuilder;
    EXPECT_TRUE(
        halfMaskBuilder.isApplicable(halfMaskWrapper.getNode(0), halfMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        halfMaskBuilder.isApplicable(floatMaskWrapper.getNode(0), floatMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        halfMaskBuilder.isApplicable(noMaskWrapper.getNode(0), noMaskWrapper.getTensorMap()));

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        noMaskBuilder;
    EXPECT_TRUE(noMaskBuilder.isApplicable(noMaskWrapper.getNode(0), noMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        noMaskBuilder.isApplicable(halfMaskWrapper.getNode(0), halfMaskWrapper.getTensorMap()));
    EXPECT_FALSE(
        noMaskBuilder.isApplicable(floatMaskWrapper.getNode(0), floatMaskWrapper.getTensorMap()));
}
