// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU SDPA backward plan: registry lookup, applicability, and a graph run through the GPU
// reference executor compared with the CPU reference executor on the same graph.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
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
// DropoutScaleInv=14 SeqLenQ=15 SeqLenKv=16 Scale=17.
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

// The same bf16 graph through the GPU and CPU reference executors gives the same gradients.
TEST(TestGpuSdpaBwdPlan, ExecutorMatchesCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    auto graph = makeRawSdpaBwdGraph({});

    Tensor<float> qF({1, 2, 5, 8});
    Tensor<float> kF({1, 2, 9, 8});
    Tensor<float> vF({1, 2, 9, 8});
    Tensor<float> oF({1, 2, 5, 8});
    Tensor<float> biasF({5, 9});
    Tensor<float> lse4({1, 2, 5, 1});
    qF.fillWithRandomValues(-1.0F, 1.0F, 301);
    kF.fillWithRandomValues(-1.0F, 1.0F, 302);
    vF.fillWithRandomValues(-1.0F, 1.0F, 303);
    biasF.fillWithRandomValues(-1.0F, 1.0F, 304);
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(
        qF,
        kF,
        vF,
        oF,
        hipdnn_test_sdk::utilities::SdpaFwdOptions<float>{}
            .setScale(0.35F)
            .setMask(&biasF)
            .setBand(-1, 0, false)
            .setLse(&lse4));

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
    Tensor<bfloat16> dO({1, 2, 5, 8});
    dO.fillWithRandomValues(bfloat16(-1.0F), bfloat16(1.0F), 305);
    Tensor<float> stats({1, 2, 5});
    for(size_t i = 0; i < stats.elementCount(); ++i)
    {
        stats.memory().hostData()[i] = lse4.memory().hostData()[i];
    }

    struct Outputs
    {
        Tensor<bfloat16> dQ{{1, 2, 5, 8}};
        Tensor<bfloat16> dK{{1, 2, 9, 8}};
        Tensor<bfloat16> dV{{1, 2, 9, 8}};
        Tensor<bfloat16> dBias{{5, 9}};
    };
    Outputs cpu;
    Outputs gpu;
    for(auto* t : {&gpu.dQ, &gpu.dK, &gpu.dV, &gpu.dBias})
    {
        t->fillWithValue(bfloat16(std::numeric_limits<float>::quiet_NaN()));
    }

    const auto pack = [&](Outputs& out, bool device) {
        const auto ptr = [device](auto& t) -> void* {
            return device ? t.memory().deviceData() : t.memory().hostData();
        };
        return std::unordered_map<int64_t, void*>{{1, ptr(q)},
                                                  {2, ptr(k)},
                                                  {3, ptr(v)},
                                                  {4, ptr(o)},
                                                  {5, ptr(dO)},
                                                  {6, ptr(stats)},
                                                  {7, ptr(out.dQ)},
                                                  {8, ptr(out.dK)},
                                                  {9, ptr(out.dV)},
                                                  {10, ptr(bias)},
                                                  {11, ptr(out.dBias)}};
    };

    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(cpu, false));
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(gpu, true));
    for(auto* t : {&gpu.dQ, &gpu.dK, &gpu.dV, &gpu.dBias})
    {
        t->memory().markDeviceModified();
    }

    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<bfloat16> validation(2e-2F, 2e-2F);
    EXPECT_TRUE(validation.allClose(cpu.dQ, gpu.dQ));
    EXPECT_TRUE(validation.allClose(cpu.dK, gpu.dK));
    EXPECT_TRUE(validation.allClose(cpu.dV, gpu.dV));
    EXPECT_TRUE(validation.allClose(cpu.dBias, gpu.dBias));
}

// An ALiBi graph (top-left causal) through the GPU and CPU reference executors gives the same
// gradients, so the GPU plan applies the node's ALiBi term. Four heads over 8 x 8 give head 0 the
// slope 1/4, which moves the farthest scores by 1.75.
TEST(TestGpuSdpaBwdPlan, AlibiExecutorMatchesCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    RawBwdSpec spec;
    spec.heads = 4;
    spec.seqQ = 8;
    spec.seqKv = 8;
    spec.statsDims = {1, 4, 8};
    spec.bias = false;
    spec.dBias = false;
    spec.alibi = true;
    spec.topLeft = true;
    auto graph = makeRawSdpaBwdGraph(spec);

    const std::vector<int64_t> dims = {1, 4, 8, 8};
    Tensor<float> qF(dims);
    Tensor<float> kF(dims);
    Tensor<float> vF(dims);
    Tensor<float> oF(dims);
    Tensor<float> lse4({1, 4, 8, 1});
    qF.fillWithRandomValues(-1.0F, 1.0F, 311);
    kF.fillWithRandomValues(-1.0F, 1.0F, 312);
    vF.fillWithRandomValues(-1.0F, 1.0F, 313);
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(
        qF,
        kF,
        vF,
        oF,
        hipdnn_test_sdk::utilities::SdpaFwdOptions<float>{}
            .setScale(0.35F)
            .setBand(-1, 0, true)
            .setAlibi(true)
            .setLse(&lse4));

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
    Tensor<bfloat16> dO(dims);
    dO.fillWithRandomValues(bfloat16(-1.0F), bfloat16(1.0F), 315);
    Tensor<float> stats({1, 4, 8});
    for(size_t i = 0; i < stats.elementCount(); ++i)
    {
        stats.memory().hostData()[i] = lse4.memory().hostData()[i];
    }

    struct Outputs
    {
        Tensor<bfloat16> dQ{{1, 4, 8, 8}};
        Tensor<bfloat16> dK{{1, 4, 8, 8}};
        Tensor<bfloat16> dV{{1, 4, 8, 8}};
    };
    Outputs cpu;
    Outputs gpu;
    for(auto* t : {&gpu.dQ, &gpu.dK, &gpu.dV})
    {
        t->fillWithValue(bfloat16(std::numeric_limits<float>::quiet_NaN()));
    }

    const auto pack = [&](Outputs& out, bool device) {
        const auto ptr = [device](auto& t) -> void* {
            return device ? t.memory().deviceData() : t.memory().hostData();
        };
        return std::unordered_map<int64_t, void*>{{1, ptr(q)},
                                                  {2, ptr(k)},
                                                  {3, ptr(v)},
                                                  {4, ptr(o)},
                                                  {5, ptr(dO)},
                                                  {6, ptr(stats)},
                                                  {7, ptr(out.dQ)},
                                                  {8, ptr(out.dK)},
                                                  {9, ptr(out.dV)}};
    };

    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(cpu, false));
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(gpu, true));
    for(auto* t : {&gpu.dQ, &gpu.dK, &gpu.dV})
    {
        t->memory().markDeviceModified();
    }

    // bf16 outputs: the CPU and GPU accumulate in a different order before the bf16 rounding.
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<bfloat16> validation(2e-2F, 2e-2F);
    EXPECT_TRUE(validation.allClose(cpu.dQ, gpu.dQ));
    EXPECT_TRUE(validation.allClose(cpu.dK, gpu.dK));
    EXPECT_TRUE(validation.allClose(cpu.dV, gpu.dV));
}
