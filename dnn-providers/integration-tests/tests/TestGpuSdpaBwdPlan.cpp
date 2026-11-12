// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU SDPA backward plan: registry lookup, applicability, and a graph run through the GPU
// reference executor compared with the CPU reference executor on the same graph.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdSignatureKey.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaBwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

// Dims: Q/O/dO/dQ [1, H, Sq, 8]; K/V/dK/dV [1, H, Skv, 8] (H = 2, Sq = 5, Skv = 9 by default);
// stats FLOAT [1, 2, 5] (rank 3, as in the checked-in bundles); bias / dBias [Sq, Skv]. UIDs: Q=1
// K=2 V=3 O=4 dO=5 Stats=6 dQ=7 dK=8 dV=9 Bias=10 dBias=11 DropoutMask=12 DropoutScale=13
// DropoutScaleInv=14 SeqLenQ=15 SeqLenKv=16 Scale=17 Sink=18 dSink=19 Seed=20 Offset=21
// RngDump=22.
struct RawBwdSpec
{
    DataType io = DataType::BFLOAT16;
    int64_t heads = 2;
    int64_t seqQ = 5;
    int64_t seqKv = 9;
    DataType statsType = DataType::FLOAT;
    std::vector<int64_t> statsDims = {1, 2, 5};
    bool bias = true;
    // Bias dtype; defaults to io.
    std::optional<DataType> biasType;
    bool dBias = true;
    // dBias dtype and dims; default to the bias's.
    std::optional<DataType> dBiasType;
    std::optional<std::vector<int64_t>> dBiasDims;
    bool alibi = false;
    // TOP_LEFT instead of BOTTOM_RIGHT alignment (right bound 0 either way).
    bool topLeft = false;
    std::optional<float> dropout;
    // Explicit dropout: FLOAT mask [1, H, Sq, Skv] and a one-element FLOAT dropout_scale.
    bool dropoutMask = false;
    std::optional<std::vector<int64_t>> dropoutScaleInvDims;
    DataType dropoutScaleInvType = DataType::FLOAT;
    bool paddingMask = false;
    // Dims of both INT32 seq_len tensors; absent means no seq_len tensors.
    std::optional<std::vector<int64_t>> seqLenDims;
    // Runtime scale tensor, which replaces attn_scale_value.
    std::optional<std::vector<int64_t>> scaleDims;
    DataType scaleType = DataType::FLOAT;
    // Attention sink [1, H, 1, 1] (FLOAT or the io dtype) and its gradient (the sink's dtype).
    std::optional<DataType> sinkType;
    std::optional<std::vector<int64_t>> sinkDims;
    bool dSink = false;
    std::optional<DataType> dSinkType;
    // Philox seed, offset and an rng_dump output FLOAT [1, H, Sq, Skv] (with `dropout` set).
    bool rngDump = false;
    // cuDNN set_deterministic_algorithm and set_max_total_seq_len_q (a workspace hint).
    bool deterministic = false;
    std::optional<int64_t> maxTotalSeqLenQ;
};

flatbuffers::FlatBufferBuilder makeRawSdpaBwdGraph(const RawBwdSpec& spec)
{
    const std::vector<int64_t> qDims = {1, spec.heads, spec.seqQ, 8};
    const std::vector<int64_t> kDims = {1, spec.heads, spec.seqKv, 8};
    const std::vector<int64_t> biasDims = {spec.seqQ, spec.seqKv};
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);
    const auto kStrides = hipdnn_data_sdk::utilities::generateStrides(kDims);
    const auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(spec.statsDims);
    const auto biasStrides = hipdnn_data_sdk::utilities::generateStrides(biasDims);
    const auto biasType = spec.biasType.value_or(spec.io);
    const auto dBiasDims = spec.dBiasDims.value_or(biasDims);
    const auto dBiasStrides = hipdnn_data_sdk::utilities::generateStrides(dBiasDims);
    const std::vector<int64_t> dropoutMaskDims = {1, spec.heads, spec.seqQ, spec.seqKv};
    const auto dropoutMaskStrides = hipdnn_data_sdk::utilities::generateStrides(dropoutMaskDims);
    const std::vector<int64_t> oneDims = {1};
    const std::vector<int64_t> oneStrides = {1};
    const auto scaleInvDims = spec.dropoutScaleInvDims.value_or(oneDims);
    const auto scaleInvStrides = hipdnn_data_sdk::utilities::generateStrides(scaleInvDims);
    const auto seqLenDims = spec.seqLenDims.value_or(oneDims);
    const auto seqLenStrides = hipdnn_data_sdk::utilities::generateStrides(seqLenDims);
    const auto scaleDims = spec.scaleDims.value_or(oneDims);
    const auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(scaleDims);
    const auto sinkLikeDims = spec.sinkDims.value_or(std::vector<int64_t>{1, spec.heads, 1, 1});
    const auto sinkLikeStrides = hipdnn_data_sdk::utilities::generateStrides(sinkLikeDims);

    SdpaBackwardAttributesT attrs;
    attrs.q_tensor_uid = 1;
    attrs.k_tensor_uid = 2;
    attrs.v_tensor_uid = 3;
    attrs.o_tensor_uid = 4;
    attrs.do_tensor_uid = 5;
    attrs.stats_tensor_uid = 6;
    attrs.dq_tensor_uid = 7;
    attrs.dk_tensor_uid = 8;
    attrs.dv_tensor_uid = 9;
    if(!spec.scaleDims.has_value())
    {
        attrs.attn_scale_value = 0.35F;
    }
    attrs.right_bound = 0;
    attrs.diagonal_alignment
        = spec.topLeft ? DiagonalAlignment::TOP_LEFT : DiagonalAlignment::BOTTOM_RIGHT;
    attrs.alibi_mask = spec.alibi;
    attrs.dropout_probability = spec.dropout;
    attrs.padding_mask = spec.paddingMask;
    attrs.is_deterministic_algorithm = spec.deterministic;
    attrs.max_total_seq_len_q = spec.maxTotalSeqLenQ;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto add = [&](int64_t uid,
                         const char* name,
                         DataType type,
                         const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& dims) {
        tensors.push_back(CreateTensorAttributesDirect(builder, uid, name, type, &strides, &dims));
    };
    add(1, "Q", spec.io, qStrides, qDims);
    add(2, "K", spec.io, kStrides, kDims);
    add(3, "V", spec.io, kStrides, kDims);
    add(4, "O", spec.io, qStrides, qDims);
    add(5, "dO", spec.io, qStrides, qDims);
    add(6, "Stats", spec.statsType, statsStrides, spec.statsDims);
    add(7, "dQ", spec.io, qStrides, qDims);
    add(8, "dK", spec.io, kStrides, kDims);
    add(9, "dV", spec.io, kStrides, kDims);
    if(spec.bias)
    {
        attrs.attn_mask_tensor_uid = 10;
        add(10, "Bias", biasType, biasStrides, biasDims);
    }
    if(spec.dBias)
    {
        attrs.dbias_tensor_uid = 11;
        add(11, "dBias", spec.dBiasType.value_or(biasType), dBiasStrides, dBiasDims);
    }
    if(spec.dropoutMask)
    {
        attrs.dropout_mask_tensor_uid = 12;
        attrs.dropout_scale_tensor_uid = 13;
        add(12, "DropoutMask", DataType::FLOAT, dropoutMaskStrides, dropoutMaskDims);
        add(13, "DropoutScale", DataType::FLOAT, oneStrides, oneDims);
    }
    if(spec.dropoutScaleInvDims.has_value())
    {
        attrs.dropout_scale_inv_tensor_uid = 14;
        add(14, "DropoutScaleInv", spec.dropoutScaleInvType, scaleInvStrides, scaleInvDims);
    }
    if(spec.seqLenDims.has_value())
    {
        attrs.seq_len_q_tensor_uid = 15;
        attrs.seq_len_kv_tensor_uid = 16;
        add(15, "SeqLenQ", DataType::INT32, seqLenStrides, seqLenDims);
        add(16, "SeqLenKv", DataType::INT32, seqLenStrides, seqLenDims);
    }
    if(spec.scaleDims.has_value())
    {
        attrs.scale_tensor_uid = 17;
        add(17, "Scale", spec.scaleType, scaleStrides, scaleDims);
    }
    if(spec.sinkType.has_value())
    {
        attrs.sink_token_tensor_uid = 18;
        add(18, "Sink", *spec.sinkType, sinkLikeStrides, sinkLikeDims);
    }
    if(spec.dSink)
    {
        attrs.dsink_token_tensor_uid = 19;
        add(19,
            "dSink",
            spec.dSinkType.value_or(spec.sinkType.value_or(DataType::FLOAT)),
            sinkLikeStrides,
            sinkLikeDims);
    }
    if(spec.rngDump)
    {
        attrs.seed_tensor_uid = 20;
        attrs.offset_tensor_uid = 21;
        attrs.rng_dump_tensor_uid = 22;
        add(20, "Seed", DataType::INT64, oneStrides, oneDims);
        add(21, "Offset", DataType::INT64, oneStrides, oneDims);
        add(22, "RngDump", DataType::FLOAT, dropoutMaskStrides, dropoutMaskDims);
    }

    auto sdpaAttrs = CreateSdpaBackwardAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaBackwardAttributes,
                                     sdpaAttrs.Union()));
    auto graph = CreateGraphDirect(
        builder, "SdpaBwdGraph", spec.io, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
    builder.Finish(graph);
    return builder;
}

// Whether the registry of SignatureKey (GPU or CPU) has a builder for the graph's dtypes that
// accepts it.
template <typename SignatureKey>
bool applicable(const RawBwdSpec& spec)
{
    auto graph = makeRawSdpaBwdGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graph.GetBufferPointer(), graph.GetSize());
    const auto& node = wrapper.getNode(0);
    const SignatureKey key(node, wrapper.getTensorMap());
    auto builders = SignatureKey::getPlanBuilders();
    const auto it = builders.find(key);
    return it != builders.end() && it->second->isApplicable(node, wrapper.getTensorMap());
}

RawBwdSpec editedSpec(const std::function<void(RawBwdSpec&)>& edit)
{
    RawBwdSpec spec;
    edit(spec);
    return spec;
}

} // namespace

TEST(TestGpuSdpaBwdSignatureKey, EveryCpuDtypeHasABuilder)
{
    const auto builders = GpuSdpaBwdSignatureKey::getPlanBuilders();
    for(const auto io : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto mask :
            {DataType::UNSET, DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
        {
            EXPECT_NE(builders.find(GpuSdpaBwdSignatureKey(io, mask)), builders.end())
                << "io=" << static_cast<int>(io) << " mask=" << static_cast<int>(mask);
        }
    }
}

// The GPU registry accepts exactly the graphs the CPU registry accepts, each with its expected
// verdict.
TEST(TestGpuSdpaBwdPlanBuilder, ApplicabilityMirrorsCpuPlan)
{
    struct Case
    {
        const char* name;
        RawBwdSpec spec;
        bool expected;
    };
    const std::vector<Case> cases = {
        {"bf16 bias + dBias", {}, true},
        {"fp32", editedSpec([](RawBwdSpec& s) { s.io = DataType::FLOAT; }), true},
        {"fp16", editedSpec([](RawBwdSpec& s) { s.io = DataType::HALF; }), true},
        {"rank-4 stats", editedSpec([](RawBwdSpec& s) { s.statsDims = {1, 2, 5, 1}; }), true},
        {"HALF stats", editedSpec([](RawBwdSpec& s) { s.statsType = DataType::HALF; }), false},
        {"stats last dim 2", editedSpec([](RawBwdSpec& s) { s.statsDims = {1, 2, 5, 2}; }), false},
        // Mask dtype selects the builder; dBias follows the mask.
        {"no bias",
         editedSpec([](RawBwdSpec& s) {
             s.bias = false;
             s.dBias = false;
         }),
         true},
        {"FLOAT bias", editedSpec([](RawBwdSpec& s) { s.biasType = DataType::FLOAT; }), true},
        {"HALF bias, bf16 io",
         editedSpec([](RawBwdSpec& s) { s.biasType = DataType::HALF; }),
         true},
        {"INT32 bias", editedSpec([](RawBwdSpec& s) { s.biasType = DataType::INT32; }), false},
        {"dBias without bias", editedSpec([](RawBwdSpec& s) { s.bias = false; }), false},
        {"dBias transposed",
         editedSpec([](RawBwdSpec& s) { s.dBiasDims = std::vector<int64_t>{9, 5}; }),
         false},
        {"dBias rank 4",
         editedSpec([](RawBwdSpec& s) { s.dBiasDims = std::vector<int64_t>{1, 1, 5, 9}; }),
         false},
        {"dBias dtype differs",
         editedSpec([](RawBwdSpec& s) { s.dBiasType = DataType::HALF; }),
         false},
        // ALiBi only with top-left causal masking.
        {"ALiBi bottom-right", editedSpec([](RawBwdSpec& s) { s.alibi = true; }), false},
        {"ALiBi top-left",
         editedSpec([](RawBwdSpec& s) {
             s.alibi = true;
             s.topLeft = true;
         }),
         true},
        // Dropout: explicit mask only; dropout_scale_inv one FLOAT element, with the mask.
        {"Philox dropout", editedSpec([](RawBwdSpec& s) { s.dropout = 0.1F; }), false},
        {"explicit dropout", editedSpec([](RawBwdSpec& s) { s.dropoutMask = true; }), true},
        {"explicit dropout + scale_inv",
         editedSpec([](RawBwdSpec& s) {
             s.dropoutMask = true;
             s.dropoutScaleInvDims = std::vector<int64_t>{1};
         }),
         true},
        {"scale_inv two elements",
         editedSpec([](RawBwdSpec& s) {
             s.dropoutMask = true;
             s.dropoutScaleInvDims = std::vector<int64_t>{2};
         }),
         false},
        {"scale_inv HALF",
         editedSpec([](RawBwdSpec& s) {
             s.dropoutMask = true;
             s.dropoutScaleInvDims = std::vector<int64_t>{1};
             s.dropoutScaleInvType = DataType::HALF;
         }),
         false},
        {"scale_inv without dropout mask",
         editedSpec([](RawBwdSpec& s) { s.dropoutScaleInvDims = std::vector<int64_t>{1}; }),
         false},
        // Padding: both INT32 seq_len tensors with one value per batch, or neither.
        {"padding with seq_len",
         editedSpec([](RawBwdSpec& s) {
             s.paddingMask = true;
             s.seqLenDims = std::vector<int64_t>{1, 1, 1, 1};
         }),
         true},
        {"seq_len without padding flag",
         editedSpec([](RawBwdSpec& s) { s.seqLenDims = std::vector<int64_t>{1, 1, 1, 1}; }),
         true},
        {"padding without seq_len", editedSpec([](RawBwdSpec& s) { s.paddingMask = true; }), false},
        {"seq_len batch 2 for batch 1",
         editedSpec([](RawBwdSpec& s) {
             s.paddingMask = true;
             s.seqLenDims = std::vector<int64_t>{2, 1, 1, 1};
         }),
         false},
        // Runtime scale: one FLOAT element.
        {"scale tensor",
         editedSpec([](RawBwdSpec& s) { s.scaleDims = std::vector<int64_t>{1}; }),
         true},
        {"scale tensor two elements",
         editedSpec([](RawBwdSpec& s) { s.scaleDims = std::vector<int64_t>{2}; }),
         false},
        {"scale tensor HALF",
         editedSpec([](RawBwdSpec& s) {
             s.scaleDims = std::vector<int64_t>{1};
             s.scaleType = DataType::HALF;
         }),
         false},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        const bool gpu = applicable<GpuSdpaBwdSignatureKey>(c.spec);
        EXPECT_EQ(gpu, applicable<hipdnn_test_sdk::detail::SdpaBwdSignatureKey>(c.spec));
        EXPECT_EQ(gpu, c.expected);
    }
}

// Each gate arm of the sink, dSink and rng_dump fields declines on the GPU registry exactly as on
// the CPU registry (the GPU builders call the shared isSupportedSdpaBwdNode; a builder that
// dropped the call would accept a graph the CPU declines).
TEST(TestGpuSdpaBwdPlan, DeclinesLikeCpu)
{
    struct Case
    {
        const char* name;
        RawBwdSpec spec;
        bool expected;
    };
    const auto sink
        = [](DataType type) { return editedSpec([type](RawBwdSpec& s) { s.sinkType = type; }); };
    const auto sinkAndDSink = [](DataType sinkType, DataType dSinkType) {
        return editedSpec([sinkType, dSinkType](RawBwdSpec& s) {
            s.sinkType = sinkType;
            s.dSink = true;
            s.dSinkType = dSinkType;
        });
    };
    const auto philox = [](float p) {
        return editedSpec([p](RawBwdSpec& s) {
            s.dropout = p;
            s.rngDump = true;
        });
    };
    const std::vector<Case> cases = {
        // Sink dtype: FLOAT or the io dtype (bf16 here), nothing else.
        {"sink FLOAT", sink(DataType::FLOAT), true},
        {"sink io dtype", sink(DataType::BFLOAT16), true},
        {"sink other half type", sink(DataType::HALF), false},
        {"sink INT32", sink(DataType::INT32), false},
        {"fp16 graph, fp16 sink",
         editedSpec([](RawBwdSpec& s) {
             s.io = DataType::HALF;
             s.sinkType = DataType::HALF;
         }),
         true},
        {"fp16 graph, bf16 sink",
         editedSpec([](RawBwdSpec& s) {
             s.io = DataType::HALF;
             s.sinkType = DataType::BFLOAT16;
         }),
         false},
        {"fp32 graph, FLOAT sink",
         editedSpec([](RawBwdSpec& s) {
             s.io = DataType::FLOAT;
             s.sinkType = DataType::FLOAT;
         }),
         true},
        // Sink shape [1, H, 1, 1].
        {"sink wrong head count",
         editedSpec([](RawBwdSpec& s) {
             s.sinkType = DataType::FLOAT;
             s.sinkDims = std::vector<int64_t>{1, 3, 1, 1};
         }),
         false},
        {"sink rank 1",
         editedSpec([](RawBwdSpec& s) {
             s.sinkType = DataType::FLOAT;
             s.sinkDims = std::vector<int64_t>{2};
         }),
         false},
        // dSink has the sink's dtype and needs the sink.
        {"dSink FLOAT with FLOAT sink", sinkAndDSink(DataType::FLOAT, DataType::FLOAT), true},
        {"dSink io dtype with io sink", sinkAndDSink(DataType::BFLOAT16, DataType::BFLOAT16), true},
        {"dSink io dtype with FLOAT sink",
         sinkAndDSink(DataType::FLOAT, DataType::BFLOAT16),
         false},
        {"dSink FLOAT with io sink", sinkAndDSink(DataType::BFLOAT16, DataType::FLOAT), false},
        {"dSink without sink", editedSpec([](RawBwdSpec& s) { s.dSink = true; }), false},
        // rng_dump is consumed only as Philox dropout's keep mask.
        {"Philox with rng_dump", philox(0.1F), true},
        {"Philox p = 0 with rng_dump", philox(0.0F), true},
        {"Philox p = 1 with rng_dump", philox(1.0F), false},
        {"rng_dump without dropout probability",
         editedSpec([](RawBwdSpec& s) { s.rngDump = true; }),
         false},
        {"rng_dump with explicit mask",
         editedSpec([](RawBwdSpec& s) {
             s.dropout = 0.1F;
             s.rngDump = true;
             s.dropoutMask = true;
         }),
         false},
        // scale_inv needs an explicit dropout mask; a Philox node carrying one is declined.
        {"Philox with rng_dump and scale_inv",
         editedSpec([](RawBwdSpec& s) {
             s.dropout = 0.1F;
             s.rngDump = true;
             s.dropoutScaleInvDims = std::vector<int64_t>{1};
         }),
         false},
        // Determinism is served (the references are deterministic); max_total_seq_len is a hint.
        {"deterministic", editedSpec([](RawBwdSpec& s) { s.deterministic = true; }), true},
        {"max_total_seq_len hint", editedSpec([](RawBwdSpec& s) { s.maxTotalSeqLenQ = 64; }), true},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        const bool gpu = applicable<GpuSdpaBwdSignatureKey>(c.spec);
        EXPECT_EQ(gpu, applicable<hipdnn_test_sdk::detail::SdpaBwdSignatureKey>(c.spec));
        EXPECT_EQ(gpu, c.expected);
    }
}

// A bf16 graph through the GPU and CPU reference executors gives the same gradients. The spec
// sets the graph; the inputs, forward options and bindings follow from it.
namespace
{

struct BwdExecutorCase
{
    const char* name;
    RawBwdSpec spec;
    // Seeds for Q, K, V, bias and dO are seedBase + 0 .. 4.
    unsigned int seedBase;
};

RawBwdSpec alibiTopLeftSpec()
{
    // Four heads over 8 x 8 give head 0 the slope 1/4, which moves the farthest scores by 1.75.
    RawBwdSpec spec;
    spec.heads = 4;
    spec.seqQ = 8;
    spec.seqKv = 8;
    spec.statsDims = {1, 4, 8};
    spec.bias = false;
    spec.dBias = false;
    spec.alibi = true;
    spec.topLeft = true;
    return spec;
}

} // namespace

class TestGpuSdpaBwdPlanExecutor : public ::testing::TestWithParam<BwdExecutorCase>
{
};

// The GPU plan must bind the node's bias and dBias, and apply its band and ALiBi terms, exactly
// as the CPU plan does.
TEST_P(TestGpuSdpaBwdPlanExecutor, MatchesCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    const RawBwdSpec& spec = GetParam().spec;
    const unsigned int seed = GetParam().seedBase;
    auto graph = makeRawSdpaBwdGraph(spec);

    constexpr int64_t HEAD_DIM = 8;
    const std::vector<int64_t> qDims = {1, spec.heads, spec.seqQ, HEAD_DIM};
    const std::vector<int64_t> kvDims = {1, spec.heads, spec.seqKv, HEAD_DIM};
    Tensor<float> qF(qDims);
    Tensor<float> kF(kvDims);
    Tensor<float> vF(kvDims);
    Tensor<float> oF(qDims);
    Tensor<float> biasF({spec.seqQ, spec.seqKv});
    Tensor<float> lse4({1, spec.heads, spec.seqQ, 1});
    qF.fillWithRandomValues(-1.0F, 1.0F, seed);
    kF.fillWithRandomValues(-1.0F, 1.0F, seed + 1);
    vF.fillWithRandomValues(-1.0F, 1.0F, seed + 2);
    biasF.fillWithRandomValues(-1.0F, 1.0F, seed + 3);
    auto forwardOptions = hipdnn_test_sdk::utilities::SdpaFwdOptions<float>{}
                              .setScale(0.35F)
                              .setBand(-1, 0, spec.topLeft)
                              .setAlibi(spec.alibi)
                              .setLse(&lse4);
    if(spec.bias)
    {
        forwardOptions.setMask(&biasF);
    }
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(qF, kF, vF, oF, forwardOptions);

    const auto toBf16 = [](const Tensor<float>& src) {
        Tensor<bfloat16> dst(src.dims());
        for(size_t i = 0; i < src.elementCount(); ++i)
        {
            dst.memory().hostData()[i] = bfloat16(src.memory().hostData()[i]);
        }
        return dst;
    };
    auto q = toBf16(qF);
    auto k = toBf16(kF);
    auto v = toBf16(vF);
    auto o = toBf16(oF);
    auto bias = toBf16(biasF);
    Tensor<bfloat16> dO(qDims);
    dO.fillWithRandomValues(bfloat16(-1.0F), bfloat16(1.0F), seed + 4);
    Tensor<float> stats(spec.statsDims);
    for(size_t i = 0; i < stats.elementCount(); ++i)
    {
        stats.memory().hostData()[i] = lse4.memory().hostData()[i];
    }

    struct Outputs
    {
        Outputs(const std::vector<int64_t>& qDims,
                const std::vector<int64_t>& kvDims,
                const std::vector<int64_t>& biasDims)
            : dQ(qDims)
            , dK(kvDims)
            , dV(kvDims)
            , dBias(biasDims)
        {
        }
        Tensor<bfloat16> dQ;
        Tensor<bfloat16> dK;
        Tensor<bfloat16> dV;
        Tensor<bfloat16> dBias;
    };
    Outputs cpu(qDims, kvDims, biasF.dims());
    Outputs gpu(qDims, kvDims, biasF.dims());
    for(auto* t : {&gpu.dQ, &gpu.dK, &gpu.dV, &gpu.dBias})
    {
        t->fillWithValue(bfloat16(std::numeric_limits<float>::quiet_NaN()));
    }

    const auto pack = [&](Outputs& out, bool device) {
        const auto ptr = [device](auto& t) -> void* {
            return device ? t.memory().deviceData() : t.memory().hostData();
        };
        std::unordered_map<int64_t, void*> variantPack{{1, ptr(q)},
                                                       {2, ptr(k)},
                                                       {3, ptr(v)},
                                                       {4, ptr(o)},
                                                       {5, ptr(dO)},
                                                       {6, ptr(stats)},
                                                       {7, ptr(out.dQ)},
                                                       {8, ptr(out.dK)},
                                                       {9, ptr(out.dV)}};
        if(spec.bias)
        {
            variantPack[10] = ptr(bias);
        }
        if(spec.dBias)
        {
            variantPack[11] = ptr(out.dBias);
        }
        return variantPack;
    };

    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(cpu, false));
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(gpu, true));
    for(auto* t : {&gpu.dQ, &gpu.dK, &gpu.dV, &gpu.dBias})
    {
        t->memory().markDeviceModified();
    }

    // bf16 outputs: the CPU and GPU accumulate in a different order before the bf16 rounding.
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<bfloat16> validation(2e-2F, 2e-2F);
    EXPECT_TRUE(validation.allClose(cpu.dQ, gpu.dQ));
    EXPECT_TRUE(validation.allClose(cpu.dK, gpu.dK));
    EXPECT_TRUE(validation.allClose(cpu.dV, gpu.dV));
    if(spec.dBias)
    {
        EXPECT_TRUE(validation.allClose(cpu.dBias, gpu.dBias));
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuSdpaBwdPlanExecutor,
                         ::testing::Values(
                             // Bias and dBias with a bottom-right causal band.
                             BwdExecutorCase{"BiasBottomRightCausal", RawBwdSpec{}, 301},
                             // ALiBi with a top-left causal band, so the GPU plan applies ALiBi.
                             BwdExecutorCase{"AlibiTopLeftCausal", alibiTopLeftSpec(), 311}),
                         [](const ::testing::TestParamInfo<BwdExecutorCase>& info) {
                             return std::string(info.param.name);
                         });

namespace
{

using hipdnn_data_sdk::types::half;

constexpr int64_t EXEC_HEAD_DIM = 8;

template <typename Io>
constexpr DataType ioTypeOf()
{
    return std::is_same_v<Io, half> ? DataType::HALF : DataType::BFLOAT16;
}

template <typename Io>
Tensor<Io> roundedTo(const Tensor<float>& src)
{
    Tensor<Io> dst(src.dims());
    for(size_t i = 0; i < src.elementCount(); ++i)
    {
        dst.memory().hostData()[i] = Io(src.memory().hostData()[i]);
    }
    return dst;
}

template <typename T>
std::vector<float> hostFloats(Tensor<T>& tensor)
{
    std::vector<float> values;
    values.reserve(tensor.elementCount());
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        values.push_back(static_cast<float>(tensor.memory().hostData()[i]));
    }
    return values;
}

void expectAllClose(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    float tolerance,
                    const char* what)
{
    ASSERT_EQ(actual.size(), expected.size()) << what;
    for(size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_LE(std::abs(actual[i] - expected[i]), tolerance + tolerance * std::abs(expected[i]))
            << what << "[" << i << "]: " << actual[i] << " vs " << expected[i];
    }
}

struct SinkBwdResult
{
    std::vector<float> dQ;
    std::vector<float> dK;
    std::vector<float> dV;
    std::vector<float> dSink;
};

// Runs one sink backward graph (bf16 or fp16 io, sink and dSink of sinkType) through the GPU or the
// CPU reference executor. The inputs are a function of nothing but Io, so every run sees the same
// ones; the sink holds values exactly representable in Io, so a FLOAT sink and an Io sink are the
// same numbers. O and Stats come from a sink-inclusive forward, as the backward needs.
template <typename Io>
SinkBwdResult runSinkBwd(bool gpu, DataType sinkType)
{
    RawBwdSpec spec;
    spec.io = ioTypeOf<Io>();
    spec.bias = false;
    spec.dBias = false;
    spec.sinkType = sinkType;
    spec.dSink = true;
    auto graph = makeRawSdpaBwdGraph(spec);

    const std::vector<int64_t> qDims = {1, spec.heads, spec.seqQ, EXEC_HEAD_DIM};
    const std::vector<int64_t> kvDims = {1, spec.heads, spec.seqKv, EXEC_HEAD_DIM};
    const std::vector<int64_t> sinkDims = {1, spec.heads, 1, 1};
    Tensor<float> qF(qDims);
    Tensor<float> kF(kvDims);
    Tensor<float> vF(kvDims);
    Tensor<float> oF(qDims);
    Tensor<float> sinkF(sinkDims);
    Tensor<float> lse4({1, spec.heads, spec.seqQ, 1});
    qF.fillWithRandomValues(-1.0F, 1.0F, 401);
    kF.fillWithRandomValues(-1.0F, 1.0F, 402);
    vF.fillWithRandomValues(-1.0F, 1.0F, 403);
    sinkF.fillWithRandomValues(-1.0F, 1.0F, 404);
    for(size_t i = 0; i < sinkF.elementCount(); ++i)
    {
        sinkF.memory().hostData()[i] = static_cast<float>(Io(sinkF.memory().hostData()[i]));
    }
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(
        qF,
        kF,
        vF,
        oF,
        hipdnn_test_sdk::utilities::SdpaFwdOptions<float>{}
            .setScale(0.35F)
            .setBand(-1, 0, false)
            .setSink(&sinkF)
            .setLse(&lse4));

    auto q = roundedTo<Io>(qF);
    auto k = roundedTo<Io>(kF);
    auto v = roundedTo<Io>(vF);
    auto o = roundedTo<Io>(oF);
    Tensor<Io> dO(qDims);
    dO.fillWithRandomValues(Io(-1.0F), Io(1.0F), 405);
    Tensor<float> stats({1, spec.heads, spec.seqQ});
    for(size_t i = 0; i < stats.elementCount(); ++i)
    {
        stats.memory().hostData()[i] = lse4.memory().hostData()[i];
    }
    auto sinkIo = roundedTo<Io>(sinkF);

    Tensor<Io> dQ(qDims);
    Tensor<Io> dK(kvDims);
    Tensor<Io> dV(kvDims);
    Tensor<float> dSinkF(sinkDims);
    Tensor<Io> dSinkIo(sinkDims);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for(auto* t : {&dQ, &dK, &dV, &dSinkIo})
    {
        t->fillWithValue(Io(nan));
    }
    dSinkF.fillWithValue(nan);

    const bool floatSink = sinkType == DataType::FLOAT;
    const auto ptr
        = [gpu](auto& t) -> void* { return gpu ? t.memory().deviceData() : t.memory().hostData(); };
    const std::unordered_map<int64_t, void*> variantPack{
        {1, ptr(q)},
        {2, ptr(k)},
        {3, ptr(v)},
        {4, ptr(o)},
        {5, ptr(dO)},
        {6, ptr(stats)},
        {7, ptr(dQ)},
        {8, ptr(dK)},
        {9, ptr(dV)},
        {18, floatSink ? ptr(sinkF) : ptr(sinkIo)},
        {19, floatSink ? ptr(dSinkF) : ptr(dSinkIo)}};
    if(gpu)
    {
        hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
        executor.execute(graph.GetBufferPointer(), graph.GetSize(), variantPack);
        for(auto* t : {&dQ, &dK, &dV, &dSinkIo})
        {
            t->memory().markDeviceModified();
        }
        dSinkF.memory().markDeviceModified();
    }
    else
    {
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor executor;
        executor.execute(graph.GetBufferPointer(), graph.GetSize(), variantPack);
    }
    return {hostFloats(dQ),
            hostFloats(dK),
            hostFloats(dV),
            floatSink ? hostFloats(dSinkF) : hostFloats(dSinkIo)};
}

// GPU against CPU for one sink dtype: dQ/dK/dV at the bf16/fp16 gradient bar, dSink at dSinkTol
// (D11 bar 2 for FLOAT; the Io dtype's own rounding step otherwise).
template <typename Io>
void expectSinkBwdMatchesCpu(DataType sinkType, float dSinkTol)
{
    SCOPED_TRACE(sinkType == DataType::FLOAT ? "FLOAT sink" : "Io-dtype sink");
    const auto cpu = runSinkBwd<Io>(false, sinkType);
    const auto gpu = runSinkBwd<Io>(true, sinkType);
    expectAllClose(gpu.dQ, cpu.dQ, 2e-2F, "dQ");
    expectAllClose(gpu.dK, cpu.dK, 2e-2F, "dK");
    expectAllClose(gpu.dV, cpu.dV, 2e-2F, "dV");
    expectAllClose(gpu.dSink, cpu.dSink, dSinkTol, "dSink");
    // Not all zero: a kernel that wrote nothing useful (or an all-masked reference) would pass
    // the closeness check against itself.
    EXPECT_TRUE(std::any_of(
        gpu.dSink.begin(), gpu.dSink.end(), [](float value) { return std::abs(value) > 0.0F; }));
}

} // namespace

// dSink on the GPU reference equals the CPU reference's (D11 bar 2: atol = rtol = 1e-4 for a
// FLOAT dSink).
TEST(TestGpuSdpaBwdSinkFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    expectSinkBwdMatchesCpu<bfloat16>(DataType::FLOAT, 1e-4F);
    expectSinkBwdMatchesCpu<half>(DataType::FLOAT, 1e-4F);
}

// A sink and dSink in the graph's io dtype: the kernels read and write them in that dtype.
TEST(TestGpuSdpaBwdSinkQDtype, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    expectSinkBwdMatchesCpu<bfloat16>(DataType::BFLOAT16, 1e-2F);
    expectSinkBwdMatchesCpu<half>(DataType::HALF, 1e-2F);
}

// The same sink values in either dtype give the same gradients, and the io-dtype dSink is the
// FLOAT dSink rounded once.
TEST(TestGpuSdpaBwdSinkQDtype, RoundsFloatResultOnce)
{
    SKIP_IF_NO_DEVICES();
    const auto check = [](const SinkBwdResult& asFloat, const SinkBwdResult& asIo, auto round) {
        EXPECT_EQ(asIo.dQ, asFloat.dQ);
        EXPECT_EQ(asIo.dK, asFloat.dK);
        EXPECT_EQ(asIo.dV, asFloat.dV);
        ASSERT_EQ(asIo.dSink.size(), asFloat.dSink.size());
        for(size_t i = 0; i < asIo.dSink.size(); ++i)
        {
            EXPECT_EQ(asIo.dSink[i], round(asFloat.dSink[i])) << "dSink[" << i << "]";
        }
    };
    check(runSinkBwd<bfloat16>(true, DataType::FLOAT),
          runSinkBwd<bfloat16>(true, DataType::BFLOAT16),
          [](float x) { return static_cast<float>(bfloat16(x)); });
    check(runSinkBwd<half>(true, DataType::FLOAT),
          runSinkBwd<half>(true, DataType::HALF),
          [](float x) { return static_cast<float>(half(x)); });
}

// Philox dropout checked through the backward rng_dump: the plan uses the dumped keep mask M as
// the explicit mask with scale 1 / (1 - p) and scale_inv (1 - p), so it equals the explicit-mask
// graph given those operands (bit for bit: the same kernels run) and differs from no dropout.
TEST(TestGpuSdpaBwdPlanDropout, PhiloxWithRngDumpUsesDumpedMask)
{
    SKIP_IF_NO_DEVICES();
    constexpr float P = 0.25F;
    RawBwdSpec base;
    base.bias = false;
    base.dBias = false;
    const std::vector<int64_t> qDims = {1, base.heads, base.seqQ, EXEC_HEAD_DIM};
    const std::vector<int64_t> kvDims = {1, base.heads, base.seqKv, EXEC_HEAD_DIM};
    const std::vector<int64_t> maskDims = {1, base.heads, base.seqQ, base.seqKv};

    // A host-generated keep mask, and the forward that applied it.
    Tensor<float> mask(maskDims);
    std::mt19937 generator(7);
    std::bernoulli_distribution keep(1.0 - static_cast<double>(P));
    for(size_t i = 0; i < mask.elementCount(); ++i)
    {
        mask.memory().hostData()[i] = keep(generator) ? 1.0F : 0.0F;
    }
    Tensor<float> qF(qDims);
    Tensor<float> kF(kvDims);
    Tensor<float> vF(kvDims);
    Tensor<float> oF(qDims);
    Tensor<float> lse4({1, base.heads, base.seqQ, 1});
    qF.fillWithRandomValues(-1.0F, 1.0F, 501);
    kF.fillWithRandomValues(-1.0F, 1.0F, 502);
    vF.fillWithRandomValues(-1.0F, 1.0F, 503);
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(
        qF,
        kF,
        vF,
        oF,
        hipdnn_test_sdk::utilities::SdpaFwdOptions<float>{}
            .setScale(0.35F)
            .setBand(-1, 0, false)
            .setDropout(&mask, 1.0F / (1.0F - P))
            .setLse(&lse4));
    auto q = roundedTo<bfloat16>(qF);
    auto k = roundedTo<bfloat16>(kF);
    auto v = roundedTo<bfloat16>(vF);
    auto o = roundedTo<bfloat16>(oF);
    Tensor<bfloat16> dO(qDims);
    dO.fillWithRandomValues(bfloat16(-1.0F), bfloat16(1.0F), 504);
    Tensor<float> stats({1, base.heads, base.seqQ});
    for(size_t i = 0; i < stats.elementCount(); ++i)
    {
        stats.memory().hostData()[i] = lse4.memory().hostData()[i];
    }
    Tensor<float> scale({1});
    Tensor<float> scaleInv({1});
    scale.fillWithValue(1.0F / (1.0F - P));
    scaleInv.fillWithValue(1.0F - P);
    Tensor<int64_t> seed({1});
    Tensor<int64_t> offset({1});
    seed.fillWithValue(42);
    offset.fillWithValue(0);

    // The gradients of one graph, concatenated.
    const auto run = [&](const RawBwdSpec& spec, const std::unordered_map<int64_t, void*>& extras) {
        auto graph = makeRawSdpaBwdGraph(spec);
        Tensor<bfloat16> dQ(qDims);
        Tensor<bfloat16> dK(kvDims);
        Tensor<bfloat16> dV(kvDims);
        for(auto* t : {&dQ, &dK, &dV})
        {
            t->fillWithValue(bfloat16(std::numeric_limits<float>::quiet_NaN()));
        }
        std::unordered_map<int64_t, void*> variantPack{{1, q.memory().deviceData()},
                                                       {2, k.memory().deviceData()},
                                                       {3, v.memory().deviceData()},
                                                       {4, o.memory().deviceData()},
                                                       {5, dO.memory().deviceData()},
                                                       {6, stats.memory().deviceData()},
                                                       {7, dQ.memory().deviceData()},
                                                       {8, dK.memory().deviceData()},
                                                       {9, dV.memory().deviceData()}};
        variantPack.insert(extras.begin(), extras.end());
        hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
        executor.execute(graph.GetBufferPointer(), graph.GetSize(), variantPack);
        std::vector<float> gradients;
        for(auto* t : {&dQ, &dK, &dV})
        {
            t->memory().markDeviceModified();
            const auto values = hostFloats(*t);
            gradients.insert(gradients.end(), values.begin(), values.end());
        }
        return gradients;
    };

    auto philoxSpec = base;
    philoxSpec.dropout = P;
    philoxSpec.rngDump = true;
    const auto philox = run(philoxSpec,
                            {{20, seed.memory().deviceData()},
                             {21, offset.memory().deviceData()},
                             {22, mask.memory().deviceData()}});

    auto explicitSpec = base;
    explicitSpec.dropoutMask = true;
    explicitSpec.dropoutScaleInvDims = std::vector<int64_t>{1};
    const auto explicitMask = run(explicitSpec,
                                  {{12, mask.memory().deviceData()},
                                   {13, scale.memory().deviceData()},
                                   {14, scaleInv.memory().deviceData()}});
    const auto noDropout = run(base, {});

    EXPECT_EQ(philox, explicitMask);
    EXPECT_NE(philox, noDropout);
}
