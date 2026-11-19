// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/SdpaLayoutTestSupport.hpp"

/**
 * @file UnifiedAttentionTestGraphs.hpp
 * @brief The SDPA graphs the hipkernel:Gfx950AttentionUnified2D and
 *        hipkernel:Gfx950AttentionUnified3D suites build, shared by both engines' test
 *        headers.
 *
 * Both engines match the same single dense SDPA-forward node, so the graph builder, the
 * uids and the mask spellings live here once. Symbols, tokens, candidate records and ABIs
 * are each engine's contract and stay in Gfx950AttentionUnified{2D,3D}TestGraphs.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified_attention
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// Mutually distinct, none a small ordinal: a uid that equalled an axis index could read as
/// correct after a transposition.
inline constexpr int64_t Q_UID = 11;
inline constexpr int64_t K_UID = 22;
inline constexpr int64_t V_UID = 33;
inline constexpr int64_t O_UID = 44;
inline constexpr int64_t EXTRA_UID = 99;
/// The RFC-0014 offsets tensor a ragged operand names.
inline constexpr int64_t RAGGED_OFFSET_UID = 55;

/// The mask kinds in the catalog's `mask_type` numbering.
inline constexpr int64_t NO_MASK = 0;
inline constexpr int64_t TOP_LEFT_CAUSAL = 1;
inline constexpr int64_t BOTTOM_RIGHT_CAUSAL = 2;

inline constexpr float SCALE = 0.08838834764831843F;

/// An SDPA graph, defaulting to one both engines' tests start from: bf16, D128, B2, Hq 8
/// over Hkv 2, Sq @p DefaultSeqLenQ over Skv 256, top-left causal. Fields perturb one rule
/// at a time. Each engine's header names its own default (prefill or decode) as GraphSpec.
template <int64_t DefaultSeqLenQ>
struct BasicGraphSpec
{
    int64_t batch = 2;
    int64_t numQueryHeads = 8;
    int64_t numKvHeads = 2;
    int64_t seqLenQ = DefaultSeqLenQ;
    int64_t seqLenKv = 256;
    int64_t headSize = 128;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    std::optional<data_objects::DataType> kDataType;

    SdpaLayout qLayout = SdpaLayout::BSHD;
    SdpaLayout kLayout = SdpaLayout::BSHD;
    SdpaLayout vLayout = SdpaLayout::BSHD;
    SdpaLayout oLayout = SdpaLayout::BSHD;

    /// Dims of V and O, when they should disagree with the shared values.
    std::optional<int64_t> vSeqLen;
    std::optional<int64_t> oHeads;

    std::optional<int64_t> leftBound = -1;
    std::optional<int64_t> rightBound = 0;
    data_objects::DiagonalAlignment alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    bool causalMaskDeprecated = false;
    bool causalMaskBottomRightDeprecated = false;
    std::optional<float> attnScaleValue = SCALE;
    bool alibiMask = false;
    bool paddingMask = false;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation implementation
        = data_objects::AttentionImplementation::AUTO;

    /// Sets any further attribute on the SDPA node (an optional tensor uid, a probability).
    std::function<void(data_objects::SdpaAttributesBuilder&)> extra;

    /// Operand uid (Q/K/V/O) that carries a ragged offsets tensor, if any.
    std::optional<int64_t> raggedOperand;

    bool twoNodes = false;
    /// One RELU node from Q to O in place of the SDPA node.
    bool pointwiseOnly = false;
};

/// A mask spelled the way hipDNN's frontend spells it.
template <typename Spec>
void setMask(Spec& spec, int64_t maskType)
{
    spec.leftBound = std::nullopt;
    spec.causalMaskDeprecated = false;
    spec.causalMaskBottomRightDeprecated = false;
    if(maskType == NO_MASK)
    {
        spec.rightBound = std::nullopt;
        spec.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    }
    else if(maskType == TOP_LEFT_CAUSAL)
    {
        spec.rightBound = 0;
        spec.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    }
    else
    {
        spec.rightBound = 0;
        spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }
}

template <typename Spec>
flatbuffers::FlatBufferBuilder buildSdpaGraph(const Spec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> qDims{spec.batch, spec.numQueryHeads, spec.seqLenQ, spec.headSize};
    const std::vector<int64_t> kDims{spec.batch, spec.numKvHeads, spec.seqLenKv, spec.headSize};
    const std::vector<int64_t> vDims{
        spec.batch, spec.numKvHeads, spec.vSeqLen.value_or(spec.seqLenKv), spec.headSize};
    const int64_t oHeads = spec.oHeads.value_or(spec.numQueryHeads);
    const std::vector<int64_t> oDims{spec.batch, oHeads, spec.seqLenQ, spec.headSize};

    const auto qStrides
        = sdpaStrides(spec.qLayout, spec.numQueryHeads, spec.seqLenQ, spec.headSize);
    const auto kStrides = sdpaStrides(spec.kLayout, spec.numKvHeads, spec.seqLenKv, spec.headSize);
    const auto vStrides = sdpaStrides(
        spec.vLayout, spec.numKvHeads, spec.vSeqLen.value_or(spec.seqLenKv), spec.headSize);
    const auto oStrides = sdpaStrides(spec.oLayout, oHeads, spec.seqLenQ, spec.headSize);

    const auto tensorFor = [&](int64_t uid,
                               data_objects::DataType dataType,
                               const std::vector<int64_t>& strides,
                               const std::vector<int64_t>& dims) {
        const ::flatbuffers::Optional<int64_t> ragged
            = spec.raggedOperand == uid ? ::flatbuffers::Optional<int64_t>(RAGGED_OFFSET_UID)
                                        : ::flatbuffers::nullopt;
        return data_objects::CreateTensorAttributesDirect(builder,
                                                          uid,
                                                          nullptr,
                                                          dataType,
                                                          &strides,
                                                          &dims,
                                                          false,
                                                          data_objects::TensorValue::NONE,
                                                          0,
                                                          false,
                                                          ragged);
    };

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(tensorFor(Q_UID, spec.dataType, qStrides, qDims));
    tensors.push_back(tensorFor(K_UID, spec.kDataType.value_or(spec.dataType), kStrides, kDims));
    tensors.push_back(tensorFor(V_UID, spec.dataType, vStrides, vDims));
    tensors.push_back(tensorFor(O_UID, spec.dataType, oStrides, oDims));
    if(spec.raggedOperand.has_value())
    {
        const std::vector<int64_t> offsetDims{spec.batch + 1, 1, 1, 1};
        const std::vector<int64_t> offsetStrides{1, 1, 1, 1};
        tensors.push_back(
            tensorFor(RAGGED_OFFSET_UID, data_objects::DataType::INT32, offsetStrides, offsetDims));
    }

    const auto attributesFor = [&]() {
        data_objects::SdpaAttributesBuilder attributesBuilder(builder);
        attributesBuilder.add_q_tensor_uid(Q_UID);
        attributesBuilder.add_k_tensor_uid(K_UID);
        attributesBuilder.add_v_tensor_uid(V_UID);
        attributesBuilder.add_o_tensor_uid(O_UID);
        if(spec.leftBound.has_value())
        {
            attributesBuilder.add_left_bound(*spec.leftBound);
        }
        if(spec.rightBound.has_value())
        {
            attributesBuilder.add_right_bound(*spec.rightBound);
        }
        attributesBuilder.add_diagonal_alignment(spec.alignment);
        attributesBuilder.add_causal_mask(spec.causalMaskDeprecated);
        attributesBuilder.add_causal_mask_bottom_right(spec.causalMaskBottomRightDeprecated);
        if(spec.attnScaleValue.has_value())
        {
            attributesBuilder.add_attn_scale_value(*spec.attnScaleValue);
        }
        attributesBuilder.add_alibi_mask(spec.alibiMask);
        attributesBuilder.add_padding_mask(spec.paddingMask);
        attributesBuilder.add_mma_core_mode(spec.mmaCoreMode);
        attributesBuilder.add_implementation(spec.implementation);
        if(spec.extra)
        {
            spec.extra(attributesBuilder);
        }
        return attributesBuilder.Finish();
    };

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(spec.pointwiseOnly)
    {
        const auto relu
            = data_objects::CreatePointwiseAttributes(builder,
                                                      data_objects::PointwiseMode::RELU_FWD,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      Q_UID,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      O_UID);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "relu",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           relu.Union()));
    }
    else
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa2",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }

    auto name = builder.CreateString("gfx950_attention_unified_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// Keeps a built graph alive for as long as a case reads it, and hands out the MatchContext an
/// engine sees. The context holds references into this object, so it must outlive every use.
template <typename Spec>
class BasicGraph
{
public:
    explicit BasicGraph(const Spec& spec = Spec{})
        : _builder(buildSdpaGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
        _properties.gcnArchName = "gfx950";
        _properties.warpSize = 64;
    }

    BasicGraph(const BasicGraph&) = delete;
    BasicGraph& operator=(const BasicGraph&) = delete;

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
};

/// A descriptor id ending in @p lastByte, ordered by the byte.
inline hipdnn_plugin_sdk::ingestor::DescriptorId idEndingIn(unsigned lastByte)
{
    constexpr const char* HEX = "0123456789abcdef";
    std::string text = "00000000-0000-4000-8000-0000000000";
    text.push_back(HEX[(lastByte >> 4U) & 0xFU]);
    text.push_back(HEX[lastByte & 0xFU]);
    return hipdnn_flatbuffers_sdk::utilities::parseUuid(text);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified_attention

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
