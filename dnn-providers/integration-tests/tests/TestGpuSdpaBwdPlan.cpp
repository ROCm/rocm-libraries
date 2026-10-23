// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU SDPA backward plan: registry lookup, applicability, and a graph run through the GPU
// reference executor compared with the CPU reference executor on the same graph.

#include <gtest/gtest.h>

#include <cstdint>
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

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaBwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

// Dims: Q/O/dO/dQ [1, 2, 5, 8]; K/V/dK/dV [1, 2, 9, 8]; stats FLOAT [1, 2, 5] (rank 3, as in the
// checked-in bundles); bias / dBias [5, 9]. UIDs: Q=1 K=2 V=3 O=4 dO=5 Stats=6 dQ=7 dK=8 dV=9
// Bias=10 dBias=11.
struct RawBwdSpec
{
    DataType io = DataType::BFLOAT16;
    DataType statsType = DataType::FLOAT;
    std::vector<int64_t> statsDims = {1, 2, 5};
    bool bias = true;
    bool dBias = true;
    bool alibi = false;
    std::optional<float> dropout;
};

flatbuffers::FlatBufferBuilder makeRawSdpaBwdGraph(const RawBwdSpec& spec)
{
    const std::vector<int64_t> qDims = {1, 2, 5, 8};
    const std::vector<int64_t> kDims = {1, 2, 9, 8};
    const std::vector<int64_t> biasDims = {5, 9};
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);
    const auto kStrides = hipdnn_data_sdk::utilities::generateStrides(kDims);
    const auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(spec.statsDims);
    const auto biasStrides = hipdnn_data_sdk::utilities::generateStrides(biasDims);

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
    attrs.attn_scale_value = 0.35F;
    attrs.right_bound = 0;
    attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    attrs.alibi_mask = spec.alibi;
    attrs.dropout_probability = spec.dropout;

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
        add(10, "Bias", spec.io, biasStrides, biasDims);
    }
    if(spec.dBias)
    {
        attrs.dbias_tensor_uid = 11;
        add(11, "dBias", spec.io, biasStrides, biasDims);
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

bool gpuApplicable(const RawBwdSpec& spec)
{
    auto graph = makeRawSdpaBwdGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graph.GetBufferPointer(), graph.GetSize());
    const auto& node = wrapper.getNode(0);
    const GpuSdpaBwdSignatureKey key(node, wrapper.getTensorMap());
    auto builders = GpuSdpaBwdSignatureKey::getPlanBuilders();
    const auto it = builders.find(key);
    return it != builders.end() && it->second->isApplicable(node, wrapper.getTensorMap());
}

} // namespace

TEST(TestGpuSdpaBwdSignatureKey, EveryCpuDtypeHasAGpuBuilder)
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

TEST(TestGpuSdpaBwdPlanBuilder, ApplicabilityMirrorsCpuPlan)
{
    EXPECT_TRUE(gpuApplicable({}));
    RawBwdSpec rank4;
    rank4.statsDims = {1, 2, 5, 1};
    EXPECT_TRUE(gpuApplicable(rank4));
    RawBwdSpec alibi;
    alibi.alibi = true;
    EXPECT_FALSE(gpuApplicable(alibi));
    RawBwdSpec dropout;
    dropout.dropout = 0.1F;
    EXPECT_FALSE(gpuApplicable(dropout));
    RawBwdSpec halfStats;
    halfStats.statsType = DataType::HALF;
    EXPECT_FALSE(gpuApplicable(halfStats));
    RawBwdSpec badStats;
    badStats.statsDims = {1, 2, 5, 2};
    EXPECT_FALSE(gpuApplicable(badStats));
    RawBwdSpec dBiasWithoutBias;
    dBiasWithoutBias.bias = false;
    EXPECT_FALSE(gpuApplicable(dBiasWithoutBias));
}

// The same bf16 graph through the GPU and CPU reference executors gives the same gradients.
TEST(TestGpuSdpaBwdPlan, GpuExecutorMatchesCpuExecutor)
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
        qF, kF, vF, oF, 0.35F, &biasF, -1, 0, false, &lse4);

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
