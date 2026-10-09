// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file FlydslSdpaRdnaTestGraphs.hpp
 * @brief What the hipkernel:flydsl_sdpa_rdna suites share: the pack's contract strings and a
 *        graph builder parameterized on everything its graph matcher gates.
 *
 * The strings are restated rather than shared with the pack: they are the contract
 * between the descriptors, the matcher that writes a token and the dispatch that reads
 * it, and a test that shared the constant would keep passing through a rename that broke
 * the contract.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

inline constexpr PackSymbols FLYDSL_SDPA_RDNA{"hipkernel:flydsl_sdpa_rdna",
                                              "hipkernel.flydsl_sdpa_rdna.graph_match",
                                              "",
                                              "hipkernel.flydsl_sdpa_rdna.kernel_match",
                                              "hipkernel.flydsl_sdpa_rdna.score",
                                              "hipkernel.flydsl_sdpa_rdna.dispatch",
                                              "flydsl_sdpa_rdna.q.uid",
                                              "flydsl_sdpa_rdna.k.uid",
                                              "flydsl_sdpa_rdna.o.uid"};

constexpr const char* FLYDSL_SDPA_RDNA_V_TOKEN = "flydsl_sdpa_rdna.v.uid";
constexpr const char* FLYDSL_SDPA_RDNA_CAUSAL_TOKEN = "flydsl_sdpa_rdna.causal";
constexpr const char* FLYDSL_SDPA_RDNA_RIGHT_BOUND_TOKEN = "flydsl_sdpa_rdna.right_bound";
constexpr const char* FLYDSL_SDPA_RDNA_LEFT_BOUND_TOKEN = "flydsl_sdpa_rdna.left_bound";
constexpr const char* FLYDSL_SDPA_RDNA_ALIGN_BOTTOM_RIGHT_TOKEN
    = "flydsl_sdpa_rdna.align_bottom_right";
constexpr const char* FLYDSL_SDPA_RDNA_STATS_TOKEN = "flydsl_sdpa_rdna.stats.uid";
constexpr const char* FLYDSL_SDPA_RDNA_SCALE_SOURCE_TOKEN = "flydsl_sdpa_rdna.scale_source";
constexpr const char* FLYDSL_SDPA_RDNA_SCALE_TOKEN = "flydsl_sdpa_rdna.scale";

constexpr const char* FLYDSL_SDPA_RDNA_DTYPE_FIELD = "dtype";
constexpr const char* FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD = "head_dim";
constexpr const char* FLYDSL_SDPA_RDNA_CAUSAL_FIELD = "causal";
constexpr const char* FLYDSL_SDPA_RDNA_BLOCK_M_FIELD = "block_m";
constexpr const char* FLYDSL_SDPA_RDNA_BLOCK_N_FIELD = "block_n";
constexpr const char* FLYDSL_SDPA_RDNA_DV_SPLIT_FIELD = "dv_split";
constexpr const char* FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD = "has_bias";
constexpr const char* FLYDSL_SDPA_RDNA_BIAS_Q_FIELD = "bias_q";
constexpr const char* FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD = "head_dim_max";
constexpr const char* FLYDSL_SDPA_RDNA_DECODE_FIELD = "decode";
constexpr const char* FLYDSL_SDPA_RDNA_MERGE_SYMBOL_FIELD = "merge_symbol";
constexpr const char* FLYDSL_SDPA_RDNA_HD_PAIR_FIELD = "hd_pair";

/// The stats token's value when the graph asks for no LSE output.
constexpr int64_t FLYDSL_SDPA_RDNA_NO_STATS = -1;

/// The scale-source token's two values, as the pack binds them.
constexpr int64_t FLYDSL_SDPA_RDNA_SCALE_BAKED = 0;
constexpr int64_t FLYDSL_SDPA_RDNA_SCALE_TENSOR = 1;

// Mutually distinct, and none a small ordinal: a uid that happened to equal an axis index
// could read as correct after a transposition.
constexpr int64_t FLYDSL_SDPA_RDNA_Q_UID = 11;
constexpr int64_t FLYDSL_SDPA_RDNA_K_UID = 22;
constexpr int64_t FLYDSL_SDPA_RDNA_V_UID = 33;
constexpr int64_t FLYDSL_SDPA_RDNA_O_UID = 44;
constexpr int64_t FLYDSL_SDPA_RDNA_SCALE_UID = 55;
constexpr int64_t FLYDSL_SDPA_RDNA_BIAS_UID = 66;
constexpr int64_t FLYDSL_SDPA_RDNA_STATS_UID = 77;
constexpr int64_t FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID = 88;

/// Physical arrangement of a (B, H, S, D) logical tensor.
enum class SdpaLayout
{
    /// Token-major, head varying fastest: [B, S, H, D] in memory.
    BSHD,
    /// Head-major: [B, H, S, D] in memory.
    BHSD
};

/// The mask a graph asks for, in the spelling the frontend lowers it to.
enum class SdpaMask
{
    NONE,
    /// right_bound 0, TOP_LEFT alignment.
    TOP_LEFT_CAUSAL,
    /// right_bound 0, BOTTOM_RIGHT alignment.
    BOTTOM_RIGHT_CAUSAL,
    /// left_bound 64 and right_bound 0, TOP_LEFT: a causal sliding window.
    SLIDING_WINDOW
};

/// The softmax-statistics output a graph carries.
enum class SdpaStats
{
    NONE,
    /// A servable LSE output: f32 [B, H, Sq, 1].
    LSE,
    /// The same output typed bf16, which the kernel cannot write.
    LSE_BF16,
    /// An output whose last extent is not 1.
    LSE_WRONG_SHAPE,
    /// generate_stats set with no tensor to write the statistics to.
    REQUESTED_WITHOUT_TENSOR
};

/// How the graph states its softmax scale.
enum class SdpaScale
{
    /// attn_scale_value set on the node.
    VALUE,
    /// Neither a value nor a tensor: the default, 1/sqrt(head_dim).
    ABSENT,
    /// A pass-by-value scalar tensor with a compile-time constant.
    CONSTANT_TENSOR,
    /// A pass-by-value scalar tensor the caller supplies at execute.
    RUNTIME_TENSOR,
    /// An ordinary device tensor: the kernel cannot read a scale through a pointer.
    DEVICE_TENSOR
};

/// Everything the graph matcher gates on, defaulting to a graph the shipped objects serve.
struct SdpaGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    int64_t batch = 2;
    int64_t heads = 4;
    int64_t kvHeads = 4;
    int64_t seqLenQ = 128;
    int64_t seqLenKv = 128;
    int64_t headDim = 64;
    SdpaLayout layout = SdpaLayout::BSHD;
    SdpaMask mask = SdpaMask::NONE;
    SdpaScale scaleKind = SdpaScale::VALUE;
    float scale = 0.125F;

    /// V's head dim, when it should differ from Q/K's.
    std::optional<int64_t> valueHeadDim;
    /// V's head count, when it should differ from K's.
    std::optional<int64_t> valueHeads;
    /// Q, K and V as strided views into one [B, S, 3, H, D] buffer, as a fused QKV
    /// projection lays them out. Needs equal head counts and sequence lengths.
    bool packedQkv = false;
    /// K's dtype, when it should differ from Q's.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> keyDataType;
    /// Q's head-dim stride, when it should not be 1.
    std::optional<int64_t> queryHeadDimStride;

    /// Overrides of the bounds the mask kind sets, for bands and other windows.
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;

    SdpaStats stats = SdpaStats::NONE;
    bool withBias = false;
    /// The bias as the graph declares it: rank 1 to 4, right-aligned to [B, H, Sq, Skv].
    /// Empty dims mean a packed [1, 1, Sq, Skv]; empty strides mean packed row-major.
    std::vector<int64_t> biasDims;
    std::vector<int64_t> biasStrides;
    hipdnn_flatbuffers_sdk::data_objects::DataType biasDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    bool withDropout = false;
    /// Marks Q as ragged (THD), which the kernel would read as dense.
    bool raggedQuery = false;
};

/// The bias's dims and strides as @p spec declares them to the graph.
inline std::pair<std::vector<int64_t>, std::vector<int64_t>>
    sdpaBiasLayout(const SdpaGraphSpec& spec)
{
    auto dims = spec.biasDims.empty() ? std::vector<int64_t>{1, 1, spec.seqLenQ, spec.seqLenKv}
                                      : spec.biasDims;
    auto strides = spec.biasStrides;
    if(strides.empty())
    {
        strides.assign(dims.size(), 1);
        for(size_t axis = dims.size() - 1; axis > 0; --axis)
        {
            strides[axis - 1] = strides[axis] * dims[axis];
        }
    }
    return {dims, strides};
}

/// Element strides of a (B, H, S, D) logical tensor laid out as @p layout.
inline std::vector<int64_t>
    sdpaStrides(SdpaLayout layout, int64_t heads, int64_t sequence, int64_t headDim)
{
    if(layout == SdpaLayout::BSHD)
    {
        return {sequence * heads * headDim, headDim, heads * headDim, 1};
    }
    return {heads * sequence * headDim, sequence * headDim, headDim, 1};
}

/// Element strides of Q, K and V as views into one packed [B, S, 3, H, D] buffer, in the
/// logical (B, H, S, D) order: each is the buffer offset by its third of a token's row.
inline std::vector<int64_t> sdpaPackedQkvStrides(const SdpaGraphSpec& spec)
{
    const int64_t row = 3 * spec.heads * spec.headDim;
    return {spec.seqLenQ * row, spec.headDim, row, 1};
}

/// A single SDPA-forward node built to @p spec.
inline flatbuffers::FlatBufferBuilder buildFlydslSdpaRdnaGraph(const SdpaGraphSpec& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    const int64_t valueHeadDim = spec.valueHeadDim.value_or(spec.headDim);

    const std::vector<int64_t> qDims{spec.batch, spec.heads, spec.seqLenQ, spec.headDim};
    const std::vector<int64_t> kDims{spec.batch, spec.kvHeads, spec.seqLenKv, spec.headDim};
    const int64_t valueHeads = spec.valueHeads.value_or(spec.kvHeads);
    const std::vector<int64_t> vDims{spec.batch, valueHeads, spec.seqLenKv, valueHeadDim};
    const std::vector<int64_t> oDims{spec.batch, spec.heads, spec.seqLenQ, valueHeadDim};

    auto qStrides = spec.packedQkv
                        ? sdpaPackedQkvStrides(spec)
                        : sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, spec.headDim);
    if(spec.queryHeadDimStride.has_value())
    {
        qStrides[3] = *spec.queryHeadDimStride;
    }
    const auto kStrides = spec.packedQkv
                              ? sdpaPackedQkvStrides(spec)
                              : sdpaStrides(spec.layout, spec.kvHeads, spec.seqLenKv, spec.headDim);
    const auto vStrides = spec.packedQkv
                              ? sdpaPackedQkvStrides(spec)
                              : sdpaStrides(spec.layout, valueHeads, spec.seqLenKv, valueHeadDim);
    const auto oStrides = sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, valueHeadDim);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder,
        FLYDSL_SDPA_RDNA_Q_UID,
        "q",
        spec.dataType,
        &qStrides,
        &qDims,
        false,
        data_objects::TensorValue::NONE,
        0,
        false,
        spec.raggedQuery ? flatbuffers::Optional<int64_t>(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID)
                         : flatbuffers::nullopt));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder,
                                                   FLYDSL_SDPA_RDNA_K_UID,
                                                   "k",
                                                   spec.keyDataType.value_or(spec.dataType),
                                                   &kStrides,
                                                   &kDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, FLYDSL_SDPA_RDNA_V_UID, "v", spec.dataType, &vStrides, &vDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, FLYDSL_SDPA_RDNA_O_UID, "o", spec.dataType, &oStrides, &oDims));

    const std::vector<int64_t> scalarDims{1};
    if(spec.scaleKind == SdpaScale::CONSTANT_TENSOR || spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        const bool runtime = spec.scaleKind == SdpaScale::RUNTIME_TENSOR;
        const data_objects::Float32Value scaleValue(spec.scale);
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            FLYDSL_SDPA_RDNA_SCALE_UID,
            "scale",
            data_objects::DataType::FLOAT,
            &scalarDims,
            &scalarDims,
            false,
            runtime ? data_objects::TensorValue::NONE : data_objects::TensorValue::Float32Value,
            runtime ? flatbuffers::Offset<void>(0) : builder.CreateStruct(scaleValue).Union(),
            runtime));
    }
    else if(spec.scaleKind == SdpaScale::DEVICE_TENSOR)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     FLYDSL_SDPA_RDNA_SCALE_UID,
                                                                     "scale",
                                                                     data_objects::DataType::FLOAT,
                                                                     &scalarDims,
                                                                     &scalarDims));
    }

    const auto [biasDims, biasStrides] = sdpaBiasLayout(spec);
    if(spec.withBias)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     FLYDSL_SDPA_RDNA_BIAS_UID,
                                                                     "bias",
                                                                     spec.biasDataType,
                                                                     &biasStrides,
                                                                     &biasDims));
    }
    const int64_t statsLast = spec.stats == SdpaStats::LSE_WRONG_SHAPE ? 2 : 1;
    const std::vector<int64_t> statsDims{spec.batch, spec.heads, spec.seqLenQ, statsLast};
    const std::vector<int64_t> statsStrides{
        spec.heads * spec.seqLenQ * statsLast, spec.seqLenQ * statsLast, statsLast, 1};
    const bool withStatsTensor = spec.stats == SdpaStats::LSE || spec.stats == SdpaStats::LSE_BF16
                                 || spec.stats == SdpaStats::LSE_WRONG_SHAPE;
    if(withStatsTensor)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            FLYDSL_SDPA_RDNA_STATS_UID,
            "stats",
            spec.stats == SdpaStats::LSE_BF16 ? data_objects::DataType::BFLOAT16
                                              : data_objects::DataType::FLOAT,
            &statsStrides,
            &statsDims));
    }

    data_objects::SdpaAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_q_tensor_uid(FLYDSL_SDPA_RDNA_Q_UID);
    attributesBuilder.add_k_tensor_uid(FLYDSL_SDPA_RDNA_K_UID);
    attributesBuilder.add_v_tensor_uid(FLYDSL_SDPA_RDNA_V_UID);
    attributesBuilder.add_o_tensor_uid(FLYDSL_SDPA_RDNA_O_UID);

    // Resolved first and written once: a flatbuffer field added twice is two fields.
    std::optional<int64_t> left;
    std::optional<int64_t> right;
    auto alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    switch(spec.mask)
    {
    case SdpaMask::NONE:
        break;
    case SdpaMask::TOP_LEFT_CAUSAL:
        right = 0;
        break;
    case SdpaMask::BOTTOM_RIGHT_CAUSAL:
        right = 0;
        alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
        break;
    case SdpaMask::SLIDING_WINDOW:
        left = 64;
        right = 0;
        break;
    default:
        break;
    }
    if(spec.leftBound.has_value())
    {
        left = spec.leftBound;
    }
    if(spec.rightBound.has_value())
    {
        right = spec.rightBound;
    }
    if(left.has_value())
    {
        attributesBuilder.add_left_bound(*left);
    }
    if(right.has_value())
    {
        attributesBuilder.add_right_bound(*right);
    }
    attributesBuilder.add_diagonal_alignment(alignment);
    attributesBuilder.add_causal_mask(false);
    attributesBuilder.add_causal_mask_bottom_right(false);

    if(spec.scaleKind == SdpaScale::VALUE)
    {
        attributesBuilder.add_attn_scale_value(spec.scale);
    }
    else if(spec.scaleKind != SdpaScale::ABSENT)
    {
        attributesBuilder.add_scale_tensor_uid(FLYDSL_SDPA_RDNA_SCALE_UID);
    }

    if(spec.withBias)
    {
        attributesBuilder.add_attn_mask_tensor_uid(FLYDSL_SDPA_RDNA_BIAS_UID);
    }
    if(withStatsTensor)
    {
        attributesBuilder.add_generate_stats(true);
        attributesBuilder.add_stats_tensor_uid(FLYDSL_SDPA_RDNA_STATS_UID);
    }
    else if(spec.stats == SdpaStats::REQUESTED_WITHOUT_TENSOR)
    {
        attributesBuilder.add_generate_stats(true);
    }
    if(spec.withDropout)
    {
        attributesBuilder.add_dropout_probability(0.1F);
    }

    attributesBuilder.add_alibi_mask(false);
    attributesBuilder.add_padding_mask(false);
    attributesBuilder.add_mma_core_mode(data_objects::DataType::UNSET);
    attributesBuilder.add_implementation(data_objects::AttentionImplementation::AUTO);
    const auto attributes = attributesBuilder.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attributes.Union()));

    auto name = builder.CreateString("flydsl_sdpa_rdna");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
