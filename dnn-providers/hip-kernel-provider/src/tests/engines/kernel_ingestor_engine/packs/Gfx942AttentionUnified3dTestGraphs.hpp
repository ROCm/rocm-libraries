// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "IngestorPackTestSupport.hpp"

/**
 * @file Gfx942AttentionUnified3dTestGraphs.hpp
 * @brief Decode-shaped SDPA graphs and kernel records for hipkernel:Gfx942AttentionUnified3d
 *        host tests. No device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d
{

using hipdnn_plugin_sdk::ingestor::BoundTokens;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

inline constexpr PackSymbols PACK{
    "hipkernel:Gfx942AttentionUnified3d",
    "hipkernel.gfx942_attention_unified3d.graph_match",
    "",
    "hipkernel.gfx942_attention_unified3d.kernel_match",
    "hipkernel.gfx942_attention_unified3d.score",
    "hipkernel.gfx942_attention_unified3d.dispatch",
    "",
    "",
    "",
};

inline constexpr int64_t Q_UID = 1;
inline constexpr int64_t K_UID = 2;
inline constexpr int64_t V_UID = 3;
inline constexpr int64_t O_UID = 4;
inline constexpr int64_t EXTRA_UID = 99;

/// A gfx942 device as the CPU-only tests see it: the arch, the wave width, and MI300X's
/// 304 CUs (the scorer reads the CU count).
inline hipdnn_plugin_sdk::ingestor::DeviceProperties gfx942Device(int cus = 304)
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = "gfx942";
    properties.warpSize = 64;
    properties.multiProcessorCount = cus;
    return properties;
}

/// Contiguous `[B, Hq, 1, D]` (Q and O).
inline std::vector<int64_t> decodeQoStrides(int64_t heads, int64_t headSize)
{
    return {heads * headSize, headSize, heads * headSize, 1};
}

/// BHSD `[B, Hkv, Skv, D]`.
inline std::vector<int64_t> bhsdStrides(int64_t heads, int64_t seq, int64_t headSize)
{
    return {heads * seq * headSize, seq * headSize, headSize, 1};
}

/// BSHD storage of the logical `[B, Hkv, Skv, D]` view.
inline std::vector<int64_t> bshdStrides(int64_t heads, int64_t seq, int64_t headSize)
{
    return {seq * heads * headSize, headSize, heads * headSize, 1};
}

/// The mask spellings the admission table uses, as hipDNN bounds.
enum class Mask
{
    NO_MASK,
    TOP_LEFT_CAUSAL,
    BOTTOM_RIGHT_CAUSAL,
    WINDOW_128
};

struct GraphSpec
{
    int64_t batch = 4;
    int64_t numQueryHeads = 32;
    int64_t numKvHeads = 8;
    int64_t seqLenQ = 1;
    int64_t seqLenKv = 512;
    int64_t headSize = 128;
    data_objects::DataType dataType = data_objects::DataType::HALF;
    std::optional<data_objects::DataType> oDataType;
    Mask mask = Mask::BOTTOM_RIGHT_CAUSAL;
    std::optional<float> attnScaleValue = 0.08838834764831843F;

    std::optional<std::vector<int64_t>> qDims;
    std::optional<std::vector<int64_t>> kDims;
    std::optional<std::vector<int64_t>> vDims;
    std::optional<std::vector<int64_t>> oDims;
    std::optional<std::vector<int64_t>> qStrides;
    std::optional<std::vector<int64_t>> kStrides;
    std::optional<std::vector<int64_t>> vStrides;
    std::optional<std::vector<int64_t>> oStrides;

    /// (left_bound, right_bound) with BOTTOM_RIGHT alignment, replacing @ref mask's.
    std::optional<std::pair<int64_t, int64_t>> bounds;
    bool alibiMask = false;
    bool paddingMask = false;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;

    /// Adds optional SDPA attributes the fields above do not already set (refusal cases).
    /// Never a field the builder below always writes: a flatbuffer table must not carry a
    /// field twice.
    std::function<void(data_objects::SdpaAttributesBuilder&)> extra;
    bool twoNodes = false;
};

inline flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> qDims = spec.qDims.value_or(
        std::vector<int64_t>{spec.batch, spec.numQueryHeads, spec.seqLenQ, spec.headSize});
    const std::vector<int64_t> kvDims{spec.batch, spec.numKvHeads, spec.seqLenKv, spec.headSize};
    const std::vector<int64_t> kDims = spec.kDims.value_or(kvDims);
    const std::vector<int64_t> vDims = spec.vDims.value_or(kvDims);
    const std::vector<int64_t> oDims = spec.oDims.value_or(qDims);
    const auto qStrides
        = spec.qStrides.value_or(decodeQoStrides(spec.numQueryHeads, spec.headSize));
    const auto kStrides
        = spec.kStrides.value_or(bhsdStrides(spec.numKvHeads, spec.seqLenKv, spec.headSize));
    const auto vStrides
        = spec.vStrides.value_or(bhsdStrides(spec.numKvHeads, spec.seqLenKv, spec.headSize));
    const auto oStrides
        = spec.oStrides.value_or(decodeQoStrides(spec.numQueryHeads, spec.headSize));

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto add = [&](int64_t uid,
                         data_objects::DataType type,
                         const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& dims) {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, uid, nullptr, type, &strides, &dims, false));
    };
    add(Q_UID, spec.dataType, qStrides, qDims);
    add(K_UID, spec.dataType, kStrides, kDims);
    add(V_UID, spec.dataType, vStrides, vDims);
    add(O_UID, spec.oDataType.value_or(spec.dataType), oStrides, oDims);

    const auto attributesFor = [&]() {
        data_objects::SdpaAttributesBuilder attributes(builder);
        attributes.add_q_tensor_uid(Q_UID);
        attributes.add_k_tensor_uid(K_UID);
        attributes.add_v_tensor_uid(V_UID);
        attributes.add_o_tensor_uid(O_UID);
        if(spec.bounds.has_value())
        {
            attributes.add_left_bound(spec.bounds->first);
            attributes.add_right_bound(spec.bounds->second);
            attributes.add_diagonal_alignment(data_objects::DiagonalAlignment::BOTTOM_RIGHT);
        }
        else
        {
            switch(spec.mask)
            {
            case Mask::NO_MASK:
                attributes.add_left_bound(-1);
                attributes.add_right_bound(-1);
                attributes.add_diagonal_alignment(data_objects::DiagonalAlignment::TOP_LEFT);
                break;
            case Mask::TOP_LEFT_CAUSAL:
                attributes.add_left_bound(-1);
                attributes.add_right_bound(0);
                attributes.add_diagonal_alignment(data_objects::DiagonalAlignment::TOP_LEFT);
                break;
            case Mask::BOTTOM_RIGHT_CAUSAL:
                attributes.add_left_bound(-1);
                attributes.add_right_bound(0);
                attributes.add_diagonal_alignment(data_objects::DiagonalAlignment::BOTTOM_RIGHT);
                break;
            case Mask::WINDOW_128:
            default:
                attributes.add_left_bound(128);
                attributes.add_right_bound(0);
                attributes.add_diagonal_alignment(data_objects::DiagonalAlignment::BOTTOM_RIGHT);
                break;
            }
        }
        attributes.add_causal_mask(false);
        attributes.add_causal_mask_bottom_right(false);
        if(spec.attnScaleValue.has_value())
        {
            attributes.add_attn_scale_value(*spec.attnScaleValue);
        }
        attributes.add_alibi_mask(spec.alibiMask);
        attributes.add_padding_mask(spec.paddingMask);
        attributes.add_mma_core_mode(spec.mmaCoreMode);
        attributes.add_implementation(data_objects::AttentionImplementation::AUTO);
        if(spec.extra)
        {
            spec.extra(attributes);
        }
        return attributes.Finish();
    };

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attributesFor().Union()));
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa2",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }

    auto name = builder.CreateString("gfx942_attention_unified3d_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graph(builder);
    graph.add_name(name);
    graph.add_tensors(tensorsVector);
    graph.add_nodes(nodesVector);
    builder.Finish(graph.Finish());
    return builder;
}

inline GraphFixture fixture(const GraphSpec& spec, int cus = 304)
{
    return GraphFixture(buildGraph(spec), gfx942Device(cus));
}

inline bool graphMatches(const GraphSpec& spec)
{
    const auto graph = fixture(spec);
    return matchesGraph(PACK, graph.context()).has_value();
}

/// The KMD fields of one shipped variant. tile_size_override follows rocKE's gfx942 rule
/// unless a case overrides it.
struct KernelSpec
{
    std::string dtype = "FP16";
    int64_t headSize = 128;
    int64_t numQueryHeads = 32;
    int64_t numKvHeads = 8;
    int64_t slidingWindow = 0;
    int64_t blockSize = 16;
    std::optional<int64_t> tileSizeOverride;
    int64_t numSegments = 16;
    unsigned idByte = 0xa1;
};

inline hipdnn_plugin_sdk::ingestor::DescriptorId idEndingIn(unsigned lastByte)
{
    constexpr const char* HEX = "0123456789abcdef";
    std::string text = "00000000-0000-4000-8000-0000000000";
    text.push_back(HEX[(lastByte >> 4U) & 0xFU]);
    text.push_back(HEX[lastByte & 0xFU]);
    return hipdnn_flatbuffers_sdk::utilities::parseUuid(text);
}

inline int64_t resolvedTile(int64_t headSize, int64_t blockSize)
{
    return headSize >= 128 && blockSize >= 32 ? blockSize / 2 : blockSize;
}

inline hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(const KernelSpec& spec)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId = idEndingIn(spec.idByte);
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000094a2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000094a3");
    kernel.name = "attention_unified_3d.test.gfx942";
    kernel.metadata = {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("num_query_heads"), spec.numQueryHeads},
        {std::string("num_kv_heads"), spec.numKvHeads},
        {std::string("sliding_window"), spec.slidingWindow},
        {std::string("block_size"), spec.blockSize},
        {std::string("tile_size_override"),
         spec.tileSizeOverride.value_or(resolvedTile(spec.headSize, spec.blockSize))},
        {std::string("num_segments"), spec.numSegments},
    };
    return kernel;
}

/// graph_match then kernel_match, as a catalog build runs them.
inline bool kernelMatchesGraph(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    const auto graph = fixture(graphSpec);
    const auto bound = matchesGraph(PACK, graph.context());
    if(!bound.has_value())
    {
        return false;
    }
    return matchesKernel(PACK, graph.context(), makeKernel(kernelSpec), *bound);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx942_unified_3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
