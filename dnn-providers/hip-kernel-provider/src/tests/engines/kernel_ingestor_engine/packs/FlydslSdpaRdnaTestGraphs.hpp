// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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

/// AOTriton's FlyDSL gfx1201 kernel's engine (gfx12 only): a kernel family of its own,
/// sharing the RDNA engines' graph matcher and bound tokens.
inline constexpr PackSymbols FLYDSL_SDPA_AIW{"hipkernel:flydsl_sdpa_aiw",
                                             "hipkernel.flydsl_sdpa_aiw.graph_match",
                                             "",
                                             "hipkernel.flydsl_sdpa_aiw.kernel_match",
                                             "hipkernel.flydsl_sdpa_aiw.score",
                                             "hipkernel.flydsl_sdpa_aiw.dispatch",
                                             "flydsl_sdpa_rdna.q.uid",
                                             "flydsl_sdpa_rdna.k.uid",
                                             "flydsl_sdpa_rdna.o.uid"};
constexpr const char* FLYDSL_SDPA_AIW_BIAS_F32_FIELD = "bias_f32";
constexpr const char* FLYDSL_SDPA_AIW_BIAS_KSTRIDED_FIELD = "bias_kstrided";
constexpr const char* FLYDSL_SDPA_AIW_BLOCK_THREADS_FIELD = "block_threads";

/// The split-KV decode kernel's engine: a kernel family of its own, sharing the prefill
/// engine's graph matcher (registered under its own symbol) and bound tokens.
inline constexpr PackSymbols FLYDSL_SDPA_RDNA_DECODE{
    "hipkernel:flydsl_sdpa_rdna_decode",
    "hipkernel.flydsl_sdpa_rdna_decode.graph_match",
    "",
    "hipkernel.flydsl_sdpa_rdna_decode.kernel_match",
    "hipkernel.flydsl_sdpa_rdna_decode.score",
    "hipkernel.flydsl_sdpa_rdna_decode.dispatch",
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
constexpr const char* FLYDSL_SDPA_RDNA_HEAD_DIM_V_FIELD = "head_dim_v";
constexpr const char* FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD = "head_dim_max";
constexpr const char* FLYDSL_SDPA_RDNA_MERGE_SYMBOL_FIELD = "merge_symbol";
constexpr const char* FLYDSL_SDPA_RDNA_HD_PAIR_FIELD = "hd_pair";
constexpr const char* FLYDSL_SDPA_RDNA_HAS_SINK_FIELD = "has_sink";

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
constexpr int64_t FLYDSL_SDPA_RDNA_SINK_UID = 99;
/// An f32 [1] device tensor a graph carries when a case names it from a field the spec has
/// no member for (a page table, a dropout seed, a descale), so the field refers to something.
constexpr int64_t FLYDSL_SDPA_RDNA_AUX_UID = 111;

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
    /// Neither a value nor a tensor: the default, 1.0 (cuDNN's).
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
    /// V's layout, when it should differ from Q's and K's.
    std::optional<SdpaLayout> valueLayout;
    /// Q, K and V as strided views into one [B, S, 3, H, D] buffer, as a fused QKV
    /// projection lays them out. Needs equal head counts and sequence lengths.
    bool packedQkv = false;
    /// K's dtype, when it should differ from Q's.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> keyDataType;
    /// Q's head-dim stride, when it should not be 1.
    std::optional<int64_t> queryHeadDimStride;
    /// V's sequence stride, when it should not be the layout's (e.g. a padded row pitch).
    std::optional<int64_t> valueSequenceStride;
    /// O's batch stride, when it should not be the layout's (e.g. a slice of a larger buffer).
    std::optional<int64_t> outputBatchStride;
    /// The LSE's batch stride, when it should not be the layout's.
    std::optional<int64_t> statsBatchStride;

    /// Overrides of the bounds the mask kind sets, for bands and other windows.
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    /// The deprecated causal booleans, which the graph may set alongside (or instead of)
    /// bounds, and an alignment value to write over the mask kind's.
    bool deprecatedCausalTopLeft = false;
    bool deprecatedCausalBottomRight = false;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment> diagonalAlignment;

    SdpaStats stats = SdpaStats::NONE;
    /// The LSE output laid out [B, Sq, H] rather than packed [B, H, Sq].
    bool statsSequenceMajor = false;
    bool withBias = false;
    /// An attention sink: one logit per query head, in the query's dtype, rank 1 [Hq]
    /// unless sinkRank4 asks for [1, Hq, 1, 1].
    bool withSink = false;
    bool sinkRank4 = false;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> sinkDataType;
    /// The bias as the graph declares it: rank 1 to 4, right-aligned to [B, H, Sq, Skv].
    /// Empty dims mean a packed [1, 1, Sq, Skv]; empty strides mean packed row-major.
    std::vector<int64_t> biasDims;
    std::vector<int64_t> biasStrides;
    hipdnn_flatbuffers_sdk::data_objects::DataType biasDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    /// Every third query row's bias is the dtype's lowest finite value on every key.
    bool biasMasksRows = false;
    bool withDropout = false;
    /// Marks Q as ragged (THD), which the kernel would read as dense.
    bool raggedQuery = false;

    // Every other field a graph can carry, so each has a case that sets it.
    /// The node's compute type.
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    /// A declared base-pointer alignment in bytes per tensor uid, over the schema's 16.
    std::map<int64_t, int64_t> alignment;
    /// Tensors (by uid) declared virtual, or ragged through FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID.
    std::set<int64_t> virtualTensors;
    std::set<int64_t> raggedTensors;
    bool alibiMask = false;
    bool paddingMask = false;
    hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation implementation
        = hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation::AUTO;
    /// generate_stats as written, over what `stats` implies.
    std::optional<bool> generateStats;
    /// The scale tensor's dims and dtype (over [1] and f32), and an attn_scale_value written
    /// alongside the tensor.
    std::vector<int64_t> scaleDims;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> scaleDataType;
    bool scaleValueToo = false;
    /// Writes fields the spec has no member for; each should name FLYDSL_SDPA_RDNA_AUX_UID.
    std::function<void(hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesBuilder&)>
        extraAttributes;
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

/// One tensor of the graph @p spec describes, with the alignment, virtual flag and ragged
/// offset the spec gives its uid.
inline flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>
    sdpaTensor(const SdpaGraphSpec& spec,
               flatbuffers::FlatBufferBuilder& builder,
               int64_t uid,
               const char* name,
               hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
               const std::vector<int64_t>* strides,
               const std::vector<int64_t>* dims,
               bool isVirtual = false,
               hipdnn_flatbuffers_sdk::data_objects::TensorValue valueType
               = hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE,
               flatbuffers::Offset<void> value = 0,
               bool runtime = false,
               flatbuffers::Optional<int64_t> ragged = flatbuffers::nullopt)
{
    if(spec.raggedTensors.count(uid) != 0)
    {
        ragged = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID;
    }
    const auto declared = spec.alignment.find(uid);
    return hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        uid,
        name,
        dataType,
        strides,
        dims,
        isVirtual || spec.virtualTensors.count(uid) != 0,
        valueType,
        value,
        runtime,
        ragged,
        declared == spec.alignment.end() ? 16 : declared->second);
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
    auto vStrides = spec.packedQkv ? sdpaPackedQkvStrides(spec)
                                   : sdpaStrides(spec.valueLayout.value_or(spec.layout),
                                                 valueHeads,
                                                 spec.seqLenKv,
                                                 valueHeadDim);
    if(spec.valueSequenceStride.has_value())
    {
        vStrides[2] = *spec.valueSequenceStride;
    }
    auto oStrides = sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, valueHeadDim);
    if(spec.outputBatchStride.has_value())
    {
        oStrides[0] = *spec.outputBatchStride;
    }

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(sdpaTensor(
        spec,
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
    tensors.push_back(sdpaTensor(spec,
                                 builder,
                                 FLYDSL_SDPA_RDNA_K_UID,
                                 "k",
                                 spec.keyDataType.value_or(spec.dataType),
                                 &kStrides,
                                 &kDims));
    tensors.push_back(
        sdpaTensor(spec, builder, FLYDSL_SDPA_RDNA_V_UID, "v", spec.dataType, &vStrides, &vDims));
    tensors.push_back(
        sdpaTensor(spec, builder, FLYDSL_SDPA_RDNA_O_UID, "o", spec.dataType, &oStrides, &oDims));

    const std::vector<int64_t> scalarDims{1};
    const auto scaleDims = spec.scaleDims.empty() ? scalarDims : spec.scaleDims;
    if(spec.scaleKind == SdpaScale::CONSTANT_TENSOR || spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        const bool runtime = spec.scaleKind == SdpaScale::RUNTIME_TENSOR;
        const data_objects::Float32Value scaleValue(spec.scale);
        tensors.push_back(sdpaTensor(
            spec,
            builder,
            FLYDSL_SDPA_RDNA_SCALE_UID,
            "scale",
            spec.scaleDataType.value_or(data_objects::DataType::FLOAT),
            &scaleDims,
            &scaleDims,
            false,
            runtime ? data_objects::TensorValue::NONE : data_objects::TensorValue::Float32Value,
            runtime ? flatbuffers::Offset<void>(0) : builder.CreateStruct(scaleValue).Union(),
            runtime));
    }
    else if(spec.scaleKind == SdpaScale::DEVICE_TENSOR)
    {
        tensors.push_back(sdpaTensor(spec,
                                     builder,
                                     FLYDSL_SDPA_RDNA_SCALE_UID,
                                     "scale",
                                     data_objects::DataType::FLOAT,
                                     &scalarDims,
                                     &scalarDims));
    }

    if(spec.extraAttributes)
    {
        tensors.push_back(sdpaTensor(spec,
                                     builder,
                                     FLYDSL_SDPA_RDNA_AUX_UID,
                                     "aux",
                                     data_objects::DataType::FLOAT,
                                     &scalarDims,
                                     &scalarDims));
    }

    const auto [biasDims, biasStrides] = sdpaBiasLayout(spec);
    if(spec.withBias)
    {
        tensors.push_back(sdpaTensor(spec,
                                     builder,
                                     FLYDSL_SDPA_RDNA_BIAS_UID,
                                     "bias",
                                     spec.biasDataType,
                                     &biasStrides,
                                     &biasDims));
    }
    const std::vector<int64_t> sinkDims = spec.sinkRank4 ? std::vector<int64_t>{1, spec.heads, 1, 1}
                                                         : std::vector<int64_t>{spec.heads};
    const std::vector<int64_t> sinkStrides
        = spec.sinkRank4 ? std::vector<int64_t>{spec.heads, 1, 1, 1} : std::vector<int64_t>{1};
    if(spec.withSink)
    {
        tensors.push_back(sdpaTensor(spec,
                                     builder,
                                     FLYDSL_SDPA_RDNA_SINK_UID,
                                     "sink",
                                     spec.sinkDataType.value_or(spec.dataType),
                                     &sinkStrides,
                                     &sinkDims));
    }
    const int64_t statsLast = spec.stats == SdpaStats::LSE_WRONG_SHAPE ? 2 : 1;
    const std::vector<int64_t> statsDims{spec.batch, spec.heads, spec.seqLenQ, statsLast};
    std::vector<int64_t> statsStrides
        = spec.statsSequenceMajor
              ? std::vector<int64_t>{spec.heads * spec.seqLenQ, 1, spec.heads, 1}
              : std::vector<int64_t>{
                    spec.heads * spec.seqLenQ * statsLast, spec.seqLenQ * statsLast, statsLast, 1};
    if(spec.statsBatchStride.has_value())
    {
        statsStrides[0] = *spec.statsBatchStride;
    }
    const bool withStatsTensor = spec.stats == SdpaStats::LSE || spec.stats == SdpaStats::LSE_BF16
                                 || spec.stats == SdpaStats::LSE_WRONG_SHAPE;
    if(withStatsTensor)
    {
        tensors.push_back(sdpaTensor(spec,
                                     builder,
                                     FLYDSL_SDPA_RDNA_STATS_UID,
                                     "stats",
                                     spec.stats == SdpaStats::LSE_BF16
                                         ? data_objects::DataType::BFLOAT16
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
    attributesBuilder.add_diagonal_alignment(spec.diagonalAlignment.value_or(alignment));
    attributesBuilder.add_causal_mask(spec.deprecatedCausalTopLeft);
    attributesBuilder.add_causal_mask_bottom_right(spec.deprecatedCausalBottomRight);

    if(spec.scaleKind == SdpaScale::VALUE)
    {
        attributesBuilder.add_attn_scale_value(spec.scale);
    }
    else if(spec.scaleKind != SdpaScale::ABSENT)
    {
        attributesBuilder.add_scale_tensor_uid(FLYDSL_SDPA_RDNA_SCALE_UID);
        if(spec.scaleValueToo)
        {
            attributesBuilder.add_attn_scale_value(spec.scale);
        }
    }

    if(spec.withBias)
    {
        attributesBuilder.add_attn_mask_tensor_uid(FLYDSL_SDPA_RDNA_BIAS_UID);
    }
    if(spec.withSink)
    {
        attributesBuilder.add_sink_token_tensor_uid(FLYDSL_SDPA_RDNA_SINK_UID);
    }
    if(withStatsTensor)
    {
        attributesBuilder.add_generate_stats(spec.generateStats.value_or(true));
        attributesBuilder.add_stats_tensor_uid(FLYDSL_SDPA_RDNA_STATS_UID);
    }
    else if(spec.stats == SdpaStats::REQUESTED_WITHOUT_TENSOR)
    {
        attributesBuilder.add_generate_stats(spec.generateStats.value_or(true));
    }
    else if(spec.generateStats.has_value())
    {
        attributesBuilder.add_generate_stats(*spec.generateStats);
    }
    if(spec.withDropout)
    {
        attributesBuilder.add_dropout_probability(0.1F);
    }

    attributesBuilder.add_alibi_mask(spec.alibiMask);
    attributesBuilder.add_padding_mask(spec.paddingMask);
    attributesBuilder.add_mma_core_mode(spec.mmaCoreMode);
    attributesBuilder.add_implementation(spec.implementation);
    if(spec.extraAttributes)
    {
        spec.extraAttributes(attributesBuilder);
    }
    const auto attributes = attributesBuilder.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa",
                                                   spec.computeDataType,
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

/// One edit to a spec that every FlyDSL SDPA engine must decline, and why.
struct SdpaFieldDecline
{
    std::string name;
    std::function<void(SdpaGraphSpec&)> apply;
    /// For a case that adds a feature to show one of its fields: the feature without the
    /// defect, which must be accepted, so the case cannot pass on the feature alone.
    std::function<void(SdpaGraphSpec&)> control;
};

/// A field set to a value no FlyDSL SDPA engine handles, one case per field (or per value
/// class of a field), applied on top of a spec each engine otherwise accepts. Every field of
/// SdpaAttributes, the node and each tensor appears here or in an engine's own accept and
/// decline tables: a matcher that never reads a field accepts every value of it, which looks
/// exactly like support, so each one is pinned by a case.
inline std::vector<SdpaFieldDecline> sdpaFieldDeclines()
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    using Builder = data_objects::SdpaAttributesBuilder;
    constexpr int64_t AUX = FLYDSL_SDPA_RDNA_AUX_UID;
    std::vector<SdpaFieldDecline> cases;
    const auto field = [&cases](std::string name, const std::function<void(Builder&)>& write) {
        cases.push_back(
            {std::move(name), [write](SdpaGraphSpec& spec) { spec.extraAttributes = write; }, {}});
    };
    // Optional tensors naming features the kernels do not implement.
    field("SeqLenQ", [](Builder& b) { b.add_seq_len_q_tensor_uid(AUX); });
    field("SeqLenKv", [](Builder& b) { b.add_seq_len_kv_tensor_uid(AUX); });
    field("DropoutSeed", [](Builder& b) { b.add_seed_tensor_uid(AUX); });
    field("DropoutOffset", [](Builder& b) { b.add_offset_tensor_uid(AUX); });
    field("DropoutMask", [](Builder& b) { b.add_dropout_mask_tensor_uid(AUX); });
    field("DropoutScale", [](Builder& b) { b.add_dropout_scale_tensor_uid(AUX); });
    field("DropoutProbabilityZero", [](Builder& b) { b.add_dropout_probability(0.0F); });
    field("PageTableK", [](Builder& b) { b.add_page_table_k_tensor_uid(AUX); });
    field("PageTableV", [](Builder& b) { b.add_page_table_v_tensor_uid(AUX); });
    field("MaxSeqLenKv", [](Builder& b) { b.add_max_seq_len_kv(4096); });
    field("BlockMask", [](Builder& b) { b.add_block_mask_tensor_uid(AUX); });
    field("DescaleQ", [](Builder& b) { b.add_descale_q_tensor_uid(AUX); });
    field("DescaleK", [](Builder& b) { b.add_descale_k_tensor_uid(AUX); });
    field("DescaleV", [](Builder& b) { b.add_descale_v_tensor_uid(AUX); });
    field("DescaleS", [](Builder& b) { b.add_descale_s_tensor_uid(AUX); });
    field("ScaleS", [](Builder& b) { b.add_scale_s_tensor_uid(AUX); });
    field("ScaleO", [](Builder& b) { b.add_scale_o_tensor_uid(AUX); });
    field("AmaxS", [](Builder& b) { b.add_amax_s_tensor_uid(AUX); });
    field("AmaxO", [](Builder& b) { b.add_amax_o_tensor_uid(AUX); });
    field("MaxOutput", [](Builder& b) { b.add_max_tensor_uid(AUX); });
    field("SumExpOutput", [](Builder& b) { b.add_sum_exp_tensor_uid(AUX); });
    field("RngDump", [](Builder& b) { b.add_rng_dump_tensor_uid(AUX); });

    const auto edit = [&cases](std::string name,
                               std::function<void(SdpaGraphSpec&)> apply,
                               std::function<void(SdpaGraphSpec&)> control = {}) {
        cases.push_back({std::move(name), std::move(apply), std::move(control)});
    };
    // The features a case adds to reach a field: a bias in the query's dtype (which every
    // engine reads), an LSE, a sink.
    const auto withBias = [](SdpaGraphSpec& s) {
        s.withBias = true;
        s.biasDataType = s.dataType;
    };
    const auto withStats = [](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE; };
    const auto withSink = [](SdpaGraphSpec& s) { s.withSink = true; };
    edit("Dropout", [](SdpaGraphSpec& s) { s.withDropout = true; });
    edit("AlibiMask", [](SdpaGraphSpec& s) { s.alibiMask = true; });
    edit("PaddingMask", [](SdpaGraphSpec& s) { s.paddingMask = true; });
    edit("MmaCoreModeFp8",
         [](SdpaGraphSpec& s) { s.mmaCoreMode = data_objects::DataType::FP8_E4M3; });
    edit("MmaCoreModeInt8", [](SdpaGraphSpec& s) { s.mmaCoreMode = data_objects::DataType::INT8; });
    edit("CompositeImplementation", [](SdpaGraphSpec& s) {
        s.implementation = data_objects::AttentionImplementation::COMPOSITE;
    });
    edit("UnifiedImplementation", [](SdpaGraphSpec& s) {
        s.implementation = data_objects::AttentionImplementation::UNIFIED;
    });
    edit("UnnamedImplementation", [](SdpaGraphSpec& s) {
        s.implementation = static_cast<data_objects::AttentionImplementation>(9);
    });

    // The node: the kernels compute in f32, so a type asking for more, or for integer or
    // FP8 arithmetic, declines (HALF and BFLOAT16 are accepted; see the accept tables).
    edit("ComputeTypeDouble",
         [](SdpaGraphSpec& s) { s.computeDataType = data_objects::DataType::DOUBLE; });
    edit("ComputeTypeInt32",
         [](SdpaGraphSpec& s) { s.computeDataType = data_objects::DataType::INT32; });
    edit("ComputeTypeFp8",
         [](SdpaGraphSpec& s) { s.computeDataType = data_objects::DataType::FP8_E4M3; });

    // Each operand's declared alignment, below what the kernels' accesses assume.
    edit("QAlignedTo8", [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_Q_UID] = 8; });
    edit("KAlignedTo8", [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_K_UID] = 8; });
    edit("VAlignedTo8", [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_V_UID] = 8; });
    edit("OAlignedTo2", [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_O_UID] = 2; });
    edit("QAlignedTo24", [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_Q_UID] = 24; });
    edit(
        "BiasAlignedTo4",
        [withBias](SdpaGraphSpec& s) {
            withBias(s);
            s.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 4;
        },
        withBias);
    edit(
        "StatsAlignedTo2",
        [withStats](SdpaGraphSpec& s) {
            withStats(s);
            s.alignment[FLYDSL_SDPA_RDNA_STATS_UID] = 2;
        },
        withStats);
    edit(
        "SinkAlignedTo1",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.alignment[FLYDSL_SDPA_RDNA_SINK_UID] = 1;
        },
        withSink);

    // Operands that are not plain device tensors.
    for(const auto& [name, uid] : {std::pair{"Q", FLYDSL_SDPA_RDNA_Q_UID},
                                   std::pair{"K", FLYDSL_SDPA_RDNA_K_UID},
                                   std::pair{"V", FLYDSL_SDPA_RDNA_V_UID},
                                   std::pair{"O", FLYDSL_SDPA_RDNA_O_UID}})
    {
        const int64_t operand = uid;
        edit(std::string("Virtual") + name,
             [operand](SdpaGraphSpec& s) { s.virtualTensors.insert(operand); });
        edit(std::string("Ragged") + name,
             [operand](SdpaGraphSpec& s) { s.raggedTensors.insert(operand); });
    }
    for(const auto& [name, uid, feature] :
        {std::tuple{
             "Bias", FLYDSL_SDPA_RDNA_BIAS_UID, std::function<void(SdpaGraphSpec&)>(withBias)},
         std::tuple{
             "Stats", FLYDSL_SDPA_RDNA_STATS_UID, std::function<void(SdpaGraphSpec&)>(withStats)},
         std::tuple{
             "Sink", FLYDSL_SDPA_RDNA_SINK_UID, std::function<void(SdpaGraphSpec&)>(withSink)}})
    {
        const int64_t tensorUid = uid;
        const auto add = feature;
        edit(
            std::string("Ragged") + name,
            [add, tensorUid](SdpaGraphSpec& s) {
                add(s);
                s.raggedTensors.insert(tensorUid);
            },
            add);
        edit(
            std::string("Virtual") + name,
            [add, tensorUid](SdpaGraphSpec& s) {
                add(s);
                s.virtualTensors.insert(tensorUid);
            },
            add);
    }
    edit(
        "SinkInADifferentDtype",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = data_objects::DataType::FLOAT;
        },
        withSink);

    // Outputs whose elements share addresses: concurrent writes to one location race.
    const auto twoBatches = [](SdpaGraphSpec& s) { s.batch = 2; };
    edit(
        "OverlappingOutputBatches",
        [](SdpaGraphSpec& s) {
            s.batch = 2;
            s.outputBatchStride = s.headDim;
        },
        twoBatches);
    edit(
        "OverlappingStatsBatches",
        [withStats](SdpaGraphSpec& s) {
            withStats(s);
            s.batch = 2;
            s.statsBatchStride = 1;
        },
        [withStats](SdpaGraphSpec& s) {
            withStats(s);
            s.batch = 2;
        });

    // Statistics asked for inconsistently.
    edit("StatsRequestedWithoutATensor",
         [](SdpaGraphSpec& s) { s.stats = SdpaStats::REQUESTED_WITHOUT_TENSOR; });
    edit(
        "StatsTensorWithGenerateStatsFalse",
        [](SdpaGraphSpec& s) {
            s.stats = SdpaStats::LSE;
            s.generateStats = false;
        },
        withStats);
    edit("StatsInBfloat16", [](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE_BF16; }, withStats);
    edit(
        "StatsOfTheWrongShape",
        [](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE_WRONG_SHAPE; },
        withStats);

    // The scale: one positive finite host scalar, given one way.
    edit("ScaleOnTheDevice", [](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::DEVICE_TENSOR; });
    const auto constantScale = [](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::CONSTANT_TENSOR; };
    const auto runtimeScale = [](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::RUNTIME_TENSOR; };
    edit(
        "ScaleTensorAndValue",
        [](SdpaGraphSpec& s) {
            s.scaleKind = SdpaScale::CONSTANT_TENSOR;
            s.scaleValueToo = true;
        },
        constantScale);
    edit(
        "ScaleTensorOfFourElements",
        [](SdpaGraphSpec& s) {
            s.scaleKind = SdpaScale::RUNTIME_TENSOR;
            s.scaleDims = {4};
        },
        runtimeScale);
    edit(
        "ScaleTensorPerHead",
        [](SdpaGraphSpec& s) {
            s.scaleKind = SdpaScale::RUNTIME_TENSOR;
            s.scaleDims = {1, s.heads, 1, 1};
        },
        runtimeScale);
    edit(
        "ScaleTensorInInt32",
        [](SdpaGraphSpec& s) {
            s.scaleKind = SdpaScale::RUNTIME_TENSOR;
            s.scaleDataType = data_objects::DataType::INT32;
        },
        runtimeScale);
    for(const auto& [name, value] : {std::pair{"Negative", -0.125F},
                                     std::pair{"Zero", 0.0F},
                                     std::pair{"NaN", std::numeric_limits<float>::quiet_NaN()},
                                     std::pair{"Infinite", std::numeric_limits<float>::infinity()}})
    {
        const float scale = value;
        edit(std::string("ScaleValue") + name, [scale](SdpaGraphSpec& s) {
            s.scaleKind = SdpaScale::VALUE;
            s.scale = scale;
        });
        edit(
            std::string("ConstantScaleTensor") + name,
            [scale](SdpaGraphSpec& s) {
                s.scaleKind = SdpaScale::CONSTANT_TENSOR;
                s.scale = scale;
            },
            constantScale);
    }

    // The mask's own fields.
    edit("BothDeprecatedBooleans", [](SdpaGraphSpec& s) {
        s.deprecatedCausalTopLeft = true;
        s.deprecatedCausalBottomRight = true;
    });
    edit("UnnamedDiagonalAlignment", [](SdpaGraphSpec& s) {
        s.mask = SdpaMask::TOP_LEFT_CAUSAL;
        s.diagonalAlignment = static_cast<data_objects::DiagonalAlignment>(7);
    });
    edit("LeftBoundBelowMinusOne", [](SdpaGraphSpec& s) { s.leftBound = -2; });
    edit("RightBoundBelowMinusOne", [](SdpaGraphSpec& s) { s.rightBound = -2; });

    // Dtypes the kernels have no objects for.
    edit("Fp32", [](SdpaGraphSpec& s) { s.dataType = data_objects::DataType::FLOAT; });
    edit("Fp8", [](SdpaGraphSpec& s) { s.dataType = data_objects::DataType::FP8_E4M3; });
    edit("MixedDtypes", [](SdpaGraphSpec& s) { s.keyDataType = data_objects::DataType::HALF; });
    return cases;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
