// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
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
constexpr const char* FLYDSL_SDPA_AIW_HAS_DROPOUT_FIELD = "has_dropout";

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
constexpr const char* FLYDSL_SDPA_RDNA_ZERO_BIAS_TOKEN = "flydsl_sdpa_rdna.bias.zero";
constexpr const char* FLYDSL_SDPA_RDNA_SCALE_SOURCE_TOKEN = "flydsl_sdpa_rdna.scale_source";
constexpr const char* FLYDSL_SDPA_RDNA_SCALE_TOKEN = "flydsl_sdpa_rdna.scale";
constexpr const char* FLYDSL_SDPA_RDNA_SEQ_LEN_Q_TOKEN = "flydsl_sdpa_rdna.seq_len_q.uid";
constexpr const char* FLYDSL_SDPA_RDNA_SEQ_LEN_KV_TOKEN = "flydsl_sdpa_rdna.seq_len_kv.uid";
constexpr const char* FLYDSL_SDPA_RDNA_DROPOUT_P_TOKEN = "flydsl_sdpa_rdna.dropout_p";
constexpr const char* FLYDSL_SDPA_RDNA_DROPOUT_SEED_TOKEN = "flydsl_sdpa_rdna.dropout_seed.uid";
constexpr const char* FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_TOKEN = "flydsl_sdpa_rdna.dropout_offset.uid";
constexpr const char* FLYDSL_SDPA_RDNA_RNG_DUMP_TOKEN = "flydsl_sdpa_rdna.rng_dump.uid";
constexpr const char* FLYDSL_SDPA_RDNA_SINK_STRIDE_TOKEN = "flydsl_sdpa_rdna.sink.stride";
constexpr const char* FLYDSL_SDPA_RDNA_SINK_DTYPE_TOKEN = "flydsl_sdpa_rdna.sink.dtype";
constexpr const char* FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_TOKEN
    = "flydsl_sdpa_rdna.ragged_offset_q.uid";
constexpr const char* FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_TOKEN
    = "flydsl_sdpa_rdna.ragged_offset_kv.uid";

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
/// A pair tier (RDNA prefill kernel modification 32): V's runtime width apart from Q/K's.
constexpr const char* FLYDSL_SDPA_RDNA_V_MASK_FIELD = "v_mask";
/// The aiw objects' padded-row fill kernel (aiw modification 10).
constexpr const char* FLYDSL_SDPA_AIW_FILL_SYMBOL_FIELD = "fill_symbol";

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
/// The per-batch query and key lengths a padded graph carries.
constexpr int64_t FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID = 122;
constexpr int64_t FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID = 133;
/// The Philox seed and offset of a graph that drops attention probabilities.
constexpr int64_t FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID = 144;
constexpr int64_t FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID = 155;
/// The ragged-offset tables a packed (THD) graph's Q/O and K/V go through.
constexpr int64_t FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID = 166;
constexpr int64_t FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID = 177;
/// The RNG dump (the keep mask) a dropping graph writes out.
constexpr int64_t FLYDSL_SDPA_RDNA_RNG_DUMP_UID = 188;

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
    /// V's and O's dtypes, when they should differ from Q's.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> valueDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> outputDataType;
    /// Q's head-dim stride, when it should not be 1.
    std::optional<int64_t> queryHeadDimStride;
    /// Q's and K's sequence strides, when they should not be the layout's (a padded row pitch).
    std::optional<int64_t> querySequenceStride;
    std::optional<int64_t> keySequenceStride;
    /// V's sequence stride, when it should not be the layout's (e.g. a padded row pitch).
    std::optional<int64_t> valueSequenceStride;
    /// O's batch stride, when it should not be the layout's (e.g. a slice of a larger buffer).
    std::optional<int64_t> outputBatchStride;
    /// O's sequence stride, when it should not be the layout's (e.g. a padded row pitch).
    std::optional<int64_t> outputSequenceStride;
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
    /// An attention sink: one logit per query head, in sinkDataType, rank 1 [Hq] unless
    /// sinkRank4 asks for [1, Hq, 1, 1].
    bool withSink = false;
    bool sinkRank4 = false;
    /// The sink's dims as the graph declares them, packed row-major, over the [Hq] or
    /// [1, Hq, 1, 1] that sinkRank4 picks (graph only).
    std::vector<int64_t> sinkDims;
    /// Every sink logit -inf (the device values; the graph is unchanged).
    bool sinkNegativeInfinity = false;
    /// The sink's element stride across query heads, 1 (packed) when unset. Graph only: the
    /// device harnesses allocate a packed sink.
    std::optional<int64_t> sinkHeadStride;
    /// The sink's dtype, the query's when unset. The device harnesses hold a FLOAT sink in
    /// f32 (cuDNN's and hipDNN's reference's dtype) and any other in the query's dtype.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> sinkDataType;
    /// The sink in the 16-bit dtype the query is not (SEM-022: a bf16 sink with an fp16 query,
    /// or the reverse), whatever sinkDataType says; the device harnesses hold it in that dtype.
    bool sinkInTheOther16BitDtype = false;
    /// The bias as the graph declares it: rank 1 to 4, right-aligned to [B, H, Sq, Skv].
    /// Empty dims mean a packed [1, 1, Sq, Skv]; empty strides mean packed row-major.
    std::vector<int64_t> biasDims;
    std::vector<int64_t> biasStrides;
    hipdnn_flatbuffers_sdk::data_objects::DataType biasDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    /// Every third query row's bias is the dtype's lowest finite value on every key.
    bool biasMasksRows = false;
    /// The first this-many keys of every query row get a -inf bias, as a boolean key-padding
    /// mask cast to the bias writes them (device values, both device harnesses).
    int64_t biasNegativeInfinityLeadingKeys = 0;
    /// Every third query row's bias is -inf on every key, a row no key reaches (device
    /// values, both device harnesses).
    bool biasNegativeInfinityRows = false;
    /// Every third query row from the second gets a NaN bias on every key, so the row's O and
    /// LSE are NaN where it reaches a key (device values, both device harnesses).
    bool biasNanRows = false;
    /// Under padding, every batch's bias at and past its key length is NaN on every row: keys
    /// past a batch's length are masked whatever the bias holds there. The bias must not be
    /// broadcast over the batch (device values, both device harnesses).
    bool biasNanPastKeys = false;
    bool withDropout = false;
    /// Per-batch query and key lengths (one per batch), each a tensor the graph carries when
    /// set: INT32 [B, 1, 1, 1] unless overridden below. With paddingMask, cuDNN's padding.
    std::vector<int32_t> seqLensQ;
    std::vector<int32_t> seqLensKv;
    /// The graph names no seq_len_q tensor, though seqLensQ is set: seq_len_kv alone over packed
    /// sequences (PyTorch's seqused_k, SEM-028), whose query lengths are the packed ones. The
    /// device harnesses' oracles still read seqLensQ, which must then equal packedSeqLensQ.
    bool omitSeqLenQ = false;
    /// The length tensors' dtype, dims and batch stride as the graph declares them (graph only).
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> seqLenDataType;
    std::vector<int64_t> seqLenDims;
    std::optional<int64_t> seqLenBatchStride;
    /// Dropout, cuDNN's first form: a probability with a seed and an offset tensor, each one
    /// value, INT64 [1, 1, 1, 1] on the device unless overridden (pass-by-value through
    /// passByValueTensors). withDropout above is a probability alone.
    std::optional<float> dropoutProbability;
    bool withDropoutSeed = false;
    bool withDropoutOffset = false;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> dropoutSeedDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> dropoutOffsetDataType;
    std::vector<int64_t> dropoutSeedDims;
    /// The seed's and offset's values (device values, or the host value given at execute).
    uint64_t dropoutSeed = 0;
    uint64_t dropoutOffset = 0;
    /// The RNG dump: FLOAT [B, Hq, Sq, Skv] packed unless overridden below, named by
    /// rng_dump_tensor_uid (rngDumpUid when set, to name another tensor or none).
    bool withRngDump = false;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> rngDumpDataType;
    std::vector<int64_t> rngDumpDims;
    std::vector<int64_t> rngDumpStrides;
    std::optional<int64_t> rngDumpUid;
    /// Device values only: V the identity on its first head_dim keys (V[j][e] = 1 iff j == e)
    /// and Q zero, so each O element is one probability: the dropout mask, read off O.
    bool exposeDropoutMask = false;
    /// Device harnesses: check O and the LSE against hipDNN's CPU reference as well as the
    /// harness's own per-batch reference, whatever else the graph has (a dense, undropped,
    /// NaN-free graph, or one padded or packed into slots the reference densifies).
    bool crossCheckWithCpuReference = false;
    /// Marks Q as ragged (THD), which the kernel would read as dense.
    bool raggedQuery = false;
    /// Packed (THD), compact (setPacked): each sequence's query and key token counts. Q and O
    /// are ragged through FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID, K and V through
    /// FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID, each table the cumulative token counts and each
    /// multiplier the tensor's own sequence stride; the LSE packed through Q's table too when
    /// packedStats. The sequence extents stay the longest sequences.
    std::vector<int32_t> packedSeqLensQ;
    std::vector<int32_t> packedSeqLensKv;
    bool packedStats = false;
    /// Under packing, per tensor uid: left dense, the table it is ragged through, and its
    /// multiplier.
    std::set<int64_t> unpackedTensors;
    std::map<int64_t, int64_t> raggedOffsetOf;
    std::map<int64_t, int64_t> raggedMultiplierOf;
    /// Per tensor uid left dense: a ragged offset multiplier set with no ragged offset
    /// (graph schema 1.4.0's field, which no dense tensor can mean).
    std::map<int64_t, int64_t> strayMultiplierOf;
    /// The graph's opt-in to execute-time override shapes (graph schema 1.1.0).
    bool overrideShapeEnabled = false;
    /// The tables' dtype, dims and batch stride, over INT32, [B + 1, 1, 1, 1] and 1.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> raggedOffsetDataType;
    std::vector<int64_t> raggedOffsetDims;
    std::optional<int64_t> raggedOffsetBatchStride;

    bool packed() const
    {
        return !packedSeqLensQ.empty();
    }

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

    // Device values only (the graph is unchanged), for the RDNA device harness.
    /// V = 1 on every key below this one and 0 from it on, every head and column.
    std::optional<int64_t> valueOnesBelowKey;
    /// Every V element multiplied by this (device values only), e.g. into f32's denormal
    /// range (the CDNA harness).
    std::optional<float> valueScale;
    /// Q's first column NaN at this query position, every batch and head.
    std::optional<int64_t> queryNanPosition;
    /// K's first column NaN at this key, every batch and KV head.
    std::optional<int64_t> keyNanPosition;
    /// Packed with per-batch lengths (P3): every K and V element of a slot past its key length
    /// NaN, every head and column, so a kernel that reads a key past seq_len_kv (PyTorch's
    /// `seqused_k`) turns its rows NaN (both device harnesses).
    bool slotTailNan = false;
    /// NaN in every element of Q, K and V a row pitch leaves past the head dim, and O's gaps
    /// checked still unwritten (the RDNA device harness; give pitches past the head dims).
    bool nanGaps = false;
    /// Every sink logit this value (the device values; the graph is unchanged), e.g. one far
    /// above the scores (both device harnesses).
    std::optional<float> sinkValue;
    /// Tensors (by uid) marked runtime pass-by-value: a host scalar supplied at execute, which
    /// no operand but the scale can be.
    std::set<int64_t> passByValueTensors;
};

/// @p spec padded, cuDNN's way: padding_mask with each batch's query and key lengths.
inline void
    setPadding(SdpaGraphSpec& spec, std::vector<int32_t> seqLensQ, std::vector<int32_t> seqLensKv)
{
    spec.paddingMask = true;
    spec.seqLensQ = std::move(seqLensQ);
    spec.seqLensKv = std::move(seqLensKv);
}

/// @p spec packed (THD), compact: sequence b has @p seqLensQ[b] query and @p seqLensKv[b] key
/// tokens, stored end to end, the stats (when asked for) packed alike in the [T, H] layout.
inline void
    setPacked(SdpaGraphSpec& spec, std::vector<int32_t> seqLensQ, std::vector<int32_t> seqLensKv)
{
    spec.packedSeqLensQ = std::move(seqLensQ);
    spec.packedSeqLensKv = std::move(seqLensKv);
    spec.layout = SdpaLayout::BSHD;
    spec.statsSequenceMajor = true;
    spec.packedStats = true;
}

/// @p spec dropping attention probabilities with @p p, from a device INT64 seed and offset.
inline void setDropout(SdpaGraphSpec& spec, float p, uint64_t seed = 1234, uint64_t offset = 77)
{
    spec.dropoutProbability = p;
    spec.withDropoutSeed = true;
    spec.withDropoutOffset = true;
    spec.dropoutSeed = seed;
    spec.dropoutOffset = offset;
}

/// @p spec dropping with @p p and writing its keep mask to an RNG dump.
inline void
    setDropoutWithRngDump(SdpaGraphSpec& spec, float p, uint64_t seed = 1234, uint64_t offset = 77)
{
    setDropout(spec, p, seed, offset);
    spec.withRngDump = true;
}

/// One padded, dropping graph every family's device suite runs under one case name
/// (`WritesTheCrossFamilyPaddedRngDump`): each asserts its dump equals the same host oracle's
/// mask, so the gfx11, gfx12 and CDNA masks of this graph are equal. f16, GQA, bottom-right
/// causal, a full batch, one with odd lengths and one with no queries; a key count not a
/// multiple of 4; a seed past 32 bits and an offset that carries into the counter's high word.
inline SdpaGraphSpec crossFamilyPaddedDropoutSpec()
{
    SdpaGraphSpec spec;
    spec.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    spec.batch = 3;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.headDim = 64;
    spec.seqLenQ = 40;
    spec.seqLenKv = 70;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    setDropoutWithRngDump(spec, 0.25F, 0x0123456789ABCDEFULL, (uint64_t{1} << 32) - 7);
    setPadding(spec, {40, 17, 0}, {70, 33, 70});
    return spec;
}

/// One named device case of a cross-family case list.
struct NamedSdpaSpec
{
    std::string name;
    SdpaGraphSpec spec;
};

/// A left bound with no right one (G2): the case list every family's device suite runs against
/// hipDNN's CPU reference (`AppliesALeftOnlyWindowAsTheCpuReferenceDoes`), so all families
/// agree on it. f16, GQA, the LSE; both corners; fewer, as many and more queries than keys
/// (keyless rows top-left); L at 1 and around a 64-key tile; a window no row's reach covers
/// (no mask); a bias, f32 and in the query's dtype; padding at both corners; packed THD; one
/// query row.
inline std::vector<NamedSdpaSpec> leftOnlyWindowSpecs()
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    std::vector<NamedSdpaSpec> cases;
    const auto base = [](bool bottomRight, int64_t sq, int64_t skv, int64_t left) {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.headDim = 64;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        spec.leftBound = left;
        spec.stats = SdpaStats::LSE;
        spec.crossCheckWithCpuReference = true;
        if(bottomRight)
        {
            spec.diagonalAlignment = DiagonalAlignment::BOTTOM_RIGHT;
        }
        return spec;
    };
    for(const bool bottomRight : {false, true})
    {
        const std::string corner = bottomRight ? "BottomRight" : "TopLeft";
        for(const auto& [sq, skv] : {std::pair<int64_t, int64_t>{70, 200},
                                     std::pair<int64_t, int64_t>{130, 130},
                                     std::pair<int64_t, int64_t>{200, 70}})
        {
            for(const int64_t left : {1, 63, 64, 65})
            {
                cases.push_back({corner + "Sq" + std::to_string(sq) + "Skv" + std::to_string(skv)
                                     + "L" + std::to_string(left),
                                 base(bottomRight, sq, skv, left)});
            }
        }
        cases.push_back({corner + "WiderThanEveryRow", base(bottomRight, 130, 70, 200)});
        auto bias = base(bottomRight, 96, 160, 40);
        bias.withBias = true;
        cases.push_back({corner + "WithAnF32Bias", bias});
        bias.biasDataType = DataType::HALF;
        cases.push_back({corner + "WithAQueryDtypeBias", bias});
        auto padded = base(bottomRight, 130, 150, 48);
        setPadding(padded, {130, 61, 0}, {150, 90, 33});
        padded.batch = 3;
        cases.push_back({corner + "Padded", padded});
    }
    auto packed = base(false, 130, 150, 48);
    packed.batch = 3;
    setPacked(packed, {130, 61, 0}, {150, 90, 33});
    cases.push_back({"Packed", packed});
    cases.push_back({"OneQueryRow", base(true, 1, 300, 64)});
    return cases;
}

/// V with a head count of its own (G3, Hk != Hv): the case list every family's device suite runs
/// against hipDNN's CPU reference (`ServesValueHeadsOfTheirOwnAsTheCpuReferenceDoes`). f16, the
/// LSE; Hq/Hk/Hv of 8/2/4, 8/4/2, 8/1/8, 8/8/1 and 6/3/2; one batch in BHSD (K's and V's strides
/// equal); head dims 64, 128 and 256; Q/K 128 with V 64 and Q/K 192 with V 128; causal; a bias;
/// padding; packed THD; one query row. (A packed QKV view holds one head count for all three.)
inline std::vector<NamedSdpaSpec> valueHeadsSpecs()
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    std::vector<NamedSdpaSpec> cases;
    const auto base = [](int64_t heads, int64_t kvHeads, int64_t valueHeads) {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = heads;
        spec.kvHeads = kvHeads;
        spec.valueHeads = valueHeads;
        spec.headDim = 64;
        spec.seqLenQ = 70;
        spec.seqLenKv = 90;
        spec.stats = SdpaStats::LSE;
        spec.crossCheckWithCpuReference = true;
        return spec;
    };
    for(const auto& [hq, hk, hv] : {std::tuple<int64_t, int64_t, int64_t>{8, 2, 4},
                                    std::tuple<int64_t, int64_t, int64_t>{8, 4, 2},
                                    std::tuple<int64_t, int64_t, int64_t>{8, 1, 8},
                                    std::tuple<int64_t, int64_t, int64_t>{8, 8, 1},
                                    std::tuple<int64_t, int64_t, int64_t>{6, 3, 2}})
    {
        cases.push_back(
            {"Heads" + std::to_string(hq) + "Kv" + std::to_string(hk) + "V" + std::to_string(hv),
             base(hq, hk, hv)});
    }
    auto oneBatch = base(8, 2, 4);
    oneBatch.batch = 1;
    oneBatch.layout = SdpaLayout::BHSD;
    cases.push_back({"OneBatchBhsd", oneBatch});
    for(const int64_t headDim : {128, 256})
    {
        auto wide = base(8, 2, 4);
        wide.headDim = headDim;
        cases.push_back({"HeadDim" + std::to_string(headDim), wide});
    }
    auto pair = base(8, 2, 4);
    pair.headDim = 128;
    pair.valueHeadDim = 64;
    cases.push_back({"QueryKey128Value64", pair});
    auto mla = base(8, 2, 4);
    mla.headDim = 192;
    mla.valueHeadDim = 128;
    cases.push_back({"QueryKey192Value128", mla});
    auto causal = base(8, 4, 2);
    causal.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    cases.push_back({"BottomRightCausal", causal});
    auto bias = base(8, 2, 4);
    bias.withBias = true;
    bias.biasDataType = DataType::HALF;
    bias.seqLenKv = 96;
    cases.push_back({"WithABias", bias});
    auto padded = base(8, 2, 4);
    padded.batch = 3;
    setPadding(padded, {70, 33, 0}, {90, 41, 90});
    padded.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    cases.push_back({"Padded", padded});
    auto packed = base(8, 2, 4);
    packed.batch = 3;
    setPacked(packed, {70, 33, 0}, {90, 41, 90});
    cases.push_back({"Packed", packed});
    auto oneRow = base(8, 2, 4);
    oneRow.seqLenQ = 1;
    oneRow.seqLenKv = 300;
    oneRow.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    cases.push_back({"OneQueryRow", oneRow});
    return cases;
}

/// The parity extras (E1-E6): graphs one family declined that every family serves now, run by
/// every family's device suite against hipDNN's CPU reference
/// (`ServesTheParityExtrasAsTheCpuReferenceDoes`; dropout through the dump, with the bias
/// where there is one). f16, the LSE.
inline std::vector<NamedSdpaSpec> parityExtrasSpecs()
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    std::vector<NamedSdpaSpec> cases;
    const auto base = [](int64_t headDim, int64_t seqLenKv) {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.headDim = headDim;
        spec.seqLenQ = 64;
        spec.seqLenKv = seqLenKv;
        spec.stats = SdpaStats::LSE;
        spec.crossCheckWithCpuReference = true;
        return spec;
    };
    // E3: dropout at head_dim 96.
    auto d96 = base(96, 96);
    d96.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    setDropoutWithRngDump(d96, 0.2F, 0x0123456789ABCDEFULL, 11U);
    cases.push_back({"DropoutAtHeadDim96", d96});
    // E1: dropout with an f32 bias (rows a 32-byte multiple apart, as the aiw objects read).
    auto f32Bias = base(64, 96);
    f32Bias.withBias = true;
    f32Bias.biasDataType = DataType::FLOAT;
    f32Bias.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    setDropoutWithRngDump(f32Bias, 0.25F, 77U, 5U);
    cases.push_back({"DropoutWithAnF32Bias", f32Bias});
    // E1 and E2: dropout with a bias broadcast along the keys (key stride 0).
    auto keyBroadcast = base(64, 96);
    keyBroadcast.withBias = true;
    keyBroadcast.biasDataType = DataType::HALF;
    keyBroadcast.biasDims = {2, 4, 64, 1};
    setDropoutWithRngDump(keyBroadcast, 0.25F, 78U, 6U);
    cases.push_back({"DropoutWithAKeyBroadcastBias", keyBroadcast});
    // E4: a head between 256 and 512 with a bias, and with a zero scale. Eight query rows of a
    // two-head group: gfx11 serves these on its wide decode tier's bias objects, which take
    // one 16-row tile (a longer query declines there, COVERAGE §5.4).
    auto wide = base(384, 96);
    wide.seqLenQ = 8;
    wide.withBias = true;
    wide.biasDataType = DataType::HALF;
    wide.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    cases.push_back({"HeadDim384WithABias", wide});
    auto wideZero = base(384, 96);
    wideZero.seqLenQ = 8;
    wideZero.scale = 0.0F;
    cases.push_back({"HeadDim384WithAZeroScale", wideZero});
    // E5: Q/K and V head dims that differ, with a bias.
    auto pair = base(128, 96);
    pair.valueHeadDim = 64;
    pair.withBias = true;
    pair.biasDataType = DataType::HALF;
    cases.push_back({"QueryKey128Value64WithABias", pair});
    // E6: un-absorbed MLA with a bias, and with a negative scale.
    auto mla = base(192, 96);
    mla.valueHeadDim = 128;
    mla.withBias = true;
    mla.biasDataType = DataType::HALF;
    mla.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    cases.push_back({"QueryKey192Value128WithABias", mla});
    auto mlaNegative = base(192, 96);
    mlaNegative.valueHeadDim = 128;
    mlaNegative.scale = -0.1F;
    cases.push_back({"QueryKey192Value128WithANegativeScale", mlaNegative});
    return cases;
}

/// Packed row-major strides over @p dims, the RNG dump's layout.
inline std::vector<int64_t> rowMajorStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t axis = dims.size(); axis-- > 1;)
    {
        strides[axis - 1] = strides[axis] * dims[axis];
    }
    return strides;
}

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

/// One tensor of the graph @p spec describes, with the alignment, virtual and pass-by-value
/// flags and ragged offset the spec gives its uid.
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
    // Packed: Q, O (and the stats when packed too) through Q's table, K and V through K's,
    // offsets counting tokens.
    int64_t multiplier = 1;
    const bool queryPacked = uid == FLYDSL_SDPA_RDNA_Q_UID || uid == FLYDSL_SDPA_RDNA_O_UID
                             || (uid == FLYDSL_SDPA_RDNA_STATS_UID && spec.packedStats);
    const bool keyPacked = uid == FLYDSL_SDPA_RDNA_K_UID || uid == FLYDSL_SDPA_RDNA_V_UID;
    if(spec.packed() && (queryPacked || keyPacked) && spec.unpackedTensors.count(uid) == 0)
    {
        const auto table = spec.raggedOffsetOf.find(uid);
        const int64_t ownTable = queryPacked ? FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID
                                             : FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID;
        ragged = table != spec.raggedOffsetOf.end() ? table->second : ownTable;
        const auto declaredMultiplier = spec.raggedMultiplierOf.find(uid);
        multiplier = declaredMultiplier != spec.raggedMultiplierOf.end()
                         ? declaredMultiplier->second
                         : (*strides)[2];
    }
    if(const auto stray = spec.strayMultiplierOf.find(uid);
       stray != spec.strayMultiplierOf.end() && !ragged.has_value())
    {
        multiplier = stray->second;
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
        runtime || spec.passByValueTensors.count(uid) != 0,
        ragged,
        declared == spec.alignment.end() ? 16 : declared->second,
        multiplier);
}

/// The sink's dtype as a graph built from @p spec declares it: the 16-bit dtype the query is
/// not when sinkInTheOther16BitDtype, else sinkDataType, else the query's.
inline hipdnn_flatbuffers_sdk::data_objects::DataType sdpaSinkDataType(const SdpaGraphSpec& spec)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    if(spec.sinkInTheOther16BitDtype)
    {
        return spec.dataType == DataType::HALF ? DataType::BFLOAT16 : DataType::HALF;
    }
    return spec.sinkDataType.value_or(spec.dataType);
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
    if(spec.querySequenceStride.has_value())
    {
        qStrides[2] = *spec.querySequenceStride;
    }
    auto kStrides = spec.packedQkv
                        ? sdpaPackedQkvStrides(spec)
                        : sdpaStrides(spec.layout, spec.kvHeads, spec.seqLenKv, spec.headDim);
    if(spec.keySequenceStride.has_value())
    {
        kStrides[2] = *spec.keySequenceStride;
    }
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
    if(spec.outputSequenceStride.has_value())
    {
        oStrides[2] = *spec.outputSequenceStride;
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
    tensors.push_back(sdpaTensor(spec,
                                 builder,
                                 FLYDSL_SDPA_RDNA_V_UID,
                                 "v",
                                 spec.valueDataType.value_or(spec.dataType),
                                 &vStrides,
                                 &vDims));
    tensors.push_back(sdpaTensor(spec,
                                 builder,
                                 FLYDSL_SDPA_RDNA_O_UID,
                                 "o",
                                 spec.outputDataType.value_or(spec.dataType),
                                 &oStrides,
                                 &oDims));

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

    const std::vector<int64_t> seqLenDims
        = spec.seqLenDims.empty() ? std::vector<int64_t>{spec.batch, 1, 1, 1} : spec.seqLenDims;
    std::vector<int64_t> seqLenStrides(seqLenDims.size(), 1);
    if(spec.seqLenBatchStride.has_value())
    {
        seqLenStrides[0] = *spec.seqLenBatchStride;
    }
    for(const auto& [uid, name, lengths] :
        {std::tuple{FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID, "seq_len_q", &spec.seqLensQ},
         std::tuple{FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID, "seq_len_kv", &spec.seqLensKv}})
    {
        if(!lengths->empty() && !(spec.omitSeqLenQ && uid == FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID))
        {
            tensors.push_back(
                sdpaTensor(spec,
                           builder,
                           uid,
                           name,
                           spec.seqLenDataType.value_or(data_objects::DataType::INT32),
                           &seqLenStrides,
                           &seqLenDims));
        }
    }

    // A packed graph's two ragged-offset tables.
    const std::vector<int64_t> raggedOffsetDims
        = spec.raggedOffsetDims.empty() ? std::vector<int64_t>{spec.batch + 1, 1, 1, 1}
                                        : spec.raggedOffsetDims;
    std::vector<int64_t> raggedOffsetStrides(raggedOffsetDims.size(), 1);
    if(spec.raggedOffsetBatchStride.has_value())
    {
        raggedOffsetStrides[0] = *spec.raggedOffsetBatchStride;
    }
    for(const auto& [uid, name] :
        {std::pair{FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID, "ragged_offset_q"},
         std::pair{FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID, "ragged_offset_kv"}})
    {
        if(spec.packed())
        {
            tensors.push_back(
                sdpaTensor(spec,
                           builder,
                           uid,
                           name,
                           spec.raggedOffsetDataType.value_or(data_objects::DataType::INT32),
                           &raggedOffsetStrides,
                           &raggedOffsetDims));
        }
    }

    const std::vector<int64_t> philoxDims = {1, 1, 1, 1};
    const auto seedDims = spec.dropoutSeedDims.empty() ? philoxDims : spec.dropoutSeedDims;
    const std::vector<int64_t> seedStrides(seedDims.size(), 1);
    if(spec.withDropoutSeed)
    {
        tensors.push_back(
            sdpaTensor(spec,
                       builder,
                       FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID,
                       "dropout_seed",
                       spec.dropoutSeedDataType.value_or(data_objects::DataType::INT64),
                       &seedStrides,
                       &seedDims));
    }
    if(spec.withDropoutOffset)
    {
        tensors.push_back(
            sdpaTensor(spec,
                       builder,
                       FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID,
                       "dropout_offset",
                       spec.dropoutOffsetDataType.value_or(data_objects::DataType::INT64),
                       &philoxDims,
                       &philoxDims));
    }
    const auto rngDumpDims
        = spec.rngDumpDims.empty()
              ? std::vector<int64_t>{spec.batch, spec.heads, spec.seqLenQ, spec.seqLenKv}
              : spec.rngDumpDims;
    const auto rngDumpStrides
        = spec.rngDumpStrides.empty() ? rowMajorStrides(rngDumpDims) : spec.rngDumpStrides;
    if(spec.withRngDump)
    {
        tensors.push_back(sdpaTensor(spec,
                                     builder,
                                     FLYDSL_SDPA_RDNA_RNG_DUMP_UID,
                                     "rng_dump",
                                     spec.rngDumpDataType.value_or(data_objects::DataType::FLOAT),
                                     &rngDumpStrides,
                                     &rngDumpDims));
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
    std::vector<int64_t> sinkDims = spec.sinkRank4 ? std::vector<int64_t>{1, spec.heads, 1, 1}
                                                   : std::vector<int64_t>{spec.heads};
    const int64_t sinkHeadStride = spec.sinkHeadStride.value_or(1);
    std::vector<int64_t> sinkStrides
        = spec.sinkRank4 ? std::vector<int64_t>{spec.heads * sinkHeadStride, sinkHeadStride, 1, 1}
                         : std::vector<int64_t>{sinkHeadStride};
    if(!spec.sinkDims.empty())
    {
        sinkDims = spec.sinkDims;
        sinkStrides.assign(sinkDims.size(), 1);
        for(size_t axis = sinkDims.size() - 1; axis > 0; --axis)
        {
            sinkStrides[axis - 1] = sinkStrides[axis] * sinkDims[axis];
        }
    }
    if(spec.withSink)
    {
        tensors.push_back(sdpaTensor(spec,
                                     builder,
                                     FLYDSL_SDPA_RDNA_SINK_UID,
                                     "sink",
                                     sdpaSinkDataType(spec),
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
    if(!spec.seqLensQ.empty() && !spec.omitSeqLenQ)
    {
        attributesBuilder.add_seq_len_q_tensor_uid(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
    }
    if(!spec.seqLensKv.empty())
    {
        attributesBuilder.add_seq_len_kv_tensor_uid(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);
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
    else if(spec.dropoutProbability.has_value())
    {
        attributesBuilder.add_dropout_probability(*spec.dropoutProbability);
    }
    if(spec.withDropoutSeed)
    {
        attributesBuilder.add_seed_tensor_uid(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    }
    if(spec.withDropoutOffset)
    {
        attributesBuilder.add_offset_tensor_uid(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    }
    if(spec.withRngDump || spec.rngDumpUid.has_value())
    {
        attributesBuilder.add_rng_dump_tensor_uid(
            spec.rngDumpUid.value_or(FLYDSL_SDPA_RDNA_RNG_DUMP_UID));
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
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapeEnabled);
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

/// A case that writes one SdpaAttributes field the spec has no member for.
inline SdpaFieldDecline sdpaAttributeField(
    std::string name,
    std::function<void(hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesBuilder&)> write)
{
    return {std::move(name),
            [write = std::move(write)](SdpaGraphSpec& spec) { spec.extraAttributes = write; },
            {}};
}

/// The variable-length fields every FlyDSL SDPA engine declines (FlydslSdpaVarlen.hpp): per-batch
/// lengths, the padding switch and ragged Q/K/V/O. A form an engine comes to serve moves from here
/// to that engine's own accept and decline tables.
inline std::vector<SdpaFieldDecline> sdpaVarlenFieldDeclines()
{
    using Builder = hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesBuilder;
    constexpr int64_t AUX = FLYDSL_SDPA_RDNA_AUX_UID;
    std::vector<SdpaFieldDecline> cases;
    cases.push_back(
        sdpaAttributeField("SeqLenQ", [](Builder& b) { b.add_seq_len_q_tensor_uid(AUX); }));
    cases.push_back(
        sdpaAttributeField("SeqLenKv", [](Builder& b) { b.add_seq_len_kv_tensor_uid(AUX); }));
    cases.push_back({"PaddingMask", [](SdpaGraphSpec& s) { s.paddingMask = true; }, {}});
    for(const auto& [name, uid] : {std::pair{"Q", FLYDSL_SDPA_RDNA_Q_UID},
                                   std::pair{"K", FLYDSL_SDPA_RDNA_K_UID},
                                   std::pair{"V", FLYDSL_SDPA_RDNA_V_UID},
                                   std::pair{"O", FLYDSL_SDPA_RDNA_O_UID}})
    {
        const int64_t operand = uid;
        cases.push_back({std::string("Ragged") + name,
                         [operand](SdpaGraphSpec& s) { s.raggedTensors.insert(operand); },
                         {}});
    }
    return cases;
}

/// The dropout fields every FlyDSL SDPA engine declines (FlydslSdpaDropout.hpp): a probability
/// with no seed or offset, a seed or offset with no probability, the mask and scale form, and an
/// RNG dump with no drop (sdpaRngDumpDeclines() has the dumps of a dropping graph). (A
/// probability of 0 is the undropped result, accepted by the RDNA/aiw matcher.)
inline std::vector<SdpaFieldDecline> sdpaDropoutFieldDeclines()
{
    using Builder = hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesBuilder;
    constexpr int64_t AUX = FLYDSL_SDPA_RDNA_AUX_UID;
    std::vector<SdpaFieldDecline> cases;
    cases.push_back(
        sdpaAttributeField("DropoutSeed", [](Builder& b) { b.add_seed_tensor_uid(AUX); }));
    cases.push_back(
        sdpaAttributeField("DropoutOffset", [](Builder& b) { b.add_offset_tensor_uid(AUX); }));
    cases.push_back(
        sdpaAttributeField("DropoutMask", [](Builder& b) { b.add_dropout_mask_tensor_uid(AUX); }));
    cases.push_back(sdpaAttributeField("DropoutScale",
                                       [](Builder& b) { b.add_dropout_scale_tensor_uid(AUX); }));
    cases.push_back(
        sdpaAttributeField("RngDump", [](Builder& b) { b.add_rng_dump_tensor_uid(AUX); }));
    cases.push_back({"Dropout", [](SdpaGraphSpec& s) { s.withDropout = true; }, {}});
    return cases;
}

/// The RNG dumps of a dropping graph every FlyDSL SDPA engine declines (FlydslSdpaRngDump.hpp
/// servableRngDump), one case per gate, each with the dropping graph and its well-formed dump as
/// the control. The dump with packed operands is servableRngDump's own test: no family serves
/// packed dropout, so no graph reaches that gate.
inline std::vector<SdpaFieldDecline> sdpaRngDumpDeclines()
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    std::vector<SdpaFieldDecline> cases;
    const auto dropsWithDump = [](SdpaGraphSpec& s) { setDropoutWithRngDump(s, 0.2F); };
    const auto dump
        = [&cases, dropsWithDump](std::string name, std::function<void(SdpaGraphSpec&)> defect) {
              cases.push_back({"RngDump" + std::move(name),
                               [dropsWithDump, defect = std::move(defect)](SdpaGraphSpec& s) {
                                   dropsWithDump(s);
                                   defect(s);
                               },
                               dropsWithDump});
          };
    // The uid names no tensor, or one of the node's other tensors.
    dump("NamesNoTensor", [](SdpaGraphSpec& s) {
        s.withRngDump = false;
        s.rngDumpUid = 999;
    });
    // The tensor itself.
    dump("Half", [](SdpaGraphSpec& s) { s.rngDumpDataType = data_objects::DataType::HALF; });
    dump("Virtual",
         [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_RNG_DUMP_UID); });
    dump("PassByValue",
         [](SdpaGraphSpec& s) { s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_RNG_DUMP_UID); });
    dump("Ragged", [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_RNG_DUMP_UID); });
    dump("AlignedTo2", [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_RNG_DUMP_UID] = 2; });
    dump("Rank3",
         [](SdpaGraphSpec& s) { s.rngDumpDims = {s.batch, s.heads, s.seqLenQ * s.seqLenKv}; });
    // Each extent, [B, Hq, Sq, Skv].
    dump("BatchExtent",
         [](SdpaGraphSpec& s) { s.rngDumpDims = {s.batch + 1, s.heads, s.seqLenQ, s.seqLenKv}; });
    dump("HeadExtent",
         [](SdpaGraphSpec& s) { s.rngDumpDims = {s.batch, s.heads + 1, s.seqLenQ, s.seqLenKv}; });
    dump("RowExtent",
         [](SdpaGraphSpec& s) { s.rngDumpDims = {s.batch, s.heads, s.seqLenQ + 1, s.seqLenKv}; });
    dump("KeyExtent",
         [](SdpaGraphSpec& s) { s.rngDumpDims = {s.batch, s.heads, s.seqLenQ, s.seqLenKv + 1}; });
    // Strides other than packed row-major: keys apart, and query-major ([B, Sq, Hq, Skv]).
    dump("KeyStride2", [](SdpaGraphSpec& s) {
        s.rngDumpStrides
            = {2 * s.heads * s.seqLenQ * s.seqLenKv, 2 * s.seqLenQ * s.seqLenKv, 2 * s.seqLenKv, 2};
    });
    dump("QueryMajor", [](SdpaGraphSpec& s) {
        s.rngDumpStrides = {s.heads * s.seqLenQ * s.seqLenKv, s.seqLenKv, s.heads * s.seqLenKv, 1};
    });
    // No drop: a probability of 0 runs the undropping objects, which apply no mask.
    dump("WithoutADrop", [](SdpaGraphSpec& s) { s.dropoutProbability = 0.0F; });
    // The dump is the LSE: with one key per row, [B, Hq, Sq, 1] in f32 is a dump's shape.
    cases.push_back({"RngDumpIsTheLse",
                     [](SdpaGraphSpec& s) {
                         setDropout(s, 0.2F);
                         s.seqLenKv = 1;
                         s.stats = SdpaStats::LSE;
                         s.rngDumpUid = FLYDSL_SDPA_RDNA_STATS_UID;
                     },
                     [](SdpaGraphSpec& s) {
                         setDropoutWithRngDump(s, 0.2F);
                         s.seqLenKv = 1;
                         s.stats = SdpaStats::LSE;
                     }});
    // More draws than the writer's 32-bit grid: 2 x 65536 x 65536 / 4 = 2^31; the control,
    // 4 keys fewer, is 2^31 - 2^17.
    cases.push_back({"RngDumpOverTheWritersGrid",
                     [](SdpaGraphSpec& s) {
                         setDropoutWithRngDump(s, 0.2F);
                         s.batch = 1;
                         s.heads = 2;
                         s.kvHeads = 2;
                         s.seqLenQ = 65536;
                         s.seqLenKv = 65536;
                     },
                     [](SdpaGraphSpec& s) {
                         setDropoutWithRngDump(s, 0.2F);
                         s.batch = 1;
                         s.heads = 2;
                         s.kvHeads = 2;
                         s.seqLenQ = 65536;
                         s.seqLenKv = 65532;
                     }});
    return cases;
}

/// A field set to a value no FlyDSL SDPA engine handles, one case per field (or per value
/// class of a field), applied on top of a spec each engine otherwise accepts. Every field of
/// SdpaAttributes, the node and each tensor appears here or in an engine's own accept and
/// decline tables: a matcher that never reads a field accepts every value of it, which looks
/// exactly like support, so each one is pinned by a case. The variable-length and dropout
/// cases are their own lists, appended here.
inline std::vector<SdpaFieldDecline> sdpaFieldDeclines()
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    using Builder = data_objects::SdpaAttributesBuilder;
    constexpr int64_t AUX = FLYDSL_SDPA_RDNA_AUX_UID;
    std::vector<SdpaFieldDecline> cases;
    const auto field = [&cases](std::string name, const std::function<void(Builder&)>& write) {
        cases.push_back(sdpaAttributeField(std::move(name), write));
    };
    // Optional tensors naming features the kernels do not implement.
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
    // A FLOAT sink, cuDNN's and hipDNN's reference's dtype (kernel modifications 31, 11, 36).
    const auto withFloatSink = [](SdpaGraphSpec& s) {
        s.withSink = true;
        s.sinkDataType = data_objects::DataType::FLOAT;
    };
    edit("AlibiMask", [](SdpaGraphSpec& s) { s.alibiMask = true; });
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

    // The graph schema fields no tensor read covers (usesUnreadSchemaFields): execute-time
    // override shapes (1.1.0), and a ragged offset multiplier on a dense tensor (1.4.0).
    edit("OverrideShapeEnabled", [](SdpaGraphSpec& s) { s.overrideShapeEnabled = true; });
    edit("RaggedMultiplierOnADenseQ", [](SdpaGraphSpec& s) {
        s.strayMultiplierOf[FLYDSL_SDPA_RDNA_Q_UID] = s.heads * s.headDim;
    });

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
    // A FLOAT sink's halves are read at its element, so it needs 4 bytes.
    edit(
        "FloatSinkAlignedTo2",
        [withFloatSink](SdpaGraphSpec& s) {
            withFloatSink(s);
            s.alignment[FLYDSL_SDPA_RDNA_SINK_UID] = 2;
        },
        withFloatSink);

    // Operands that are not plain device tensors.
    for(const auto& [name, uid] : {std::pair{"Q", FLYDSL_SDPA_RDNA_Q_UID},
                                   std::pair{"K", FLYDSL_SDPA_RDNA_K_UID},
                                   std::pair{"V", FLYDSL_SDPA_RDNA_V_UID},
                                   std::pair{"O", FLYDSL_SDPA_RDNA_O_UID}})
    {
        const int64_t operand = uid;
        edit(std::string("Virtual") + name,
             [operand](SdpaGraphSpec& s) { s.virtualTensors.insert(operand); });
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
    // The kernels read a sink in FLOAT or either 16-bit dtype (SEM-022): any other dtype
    // declines (shared servableSink); the other 16-bit dtype is the control.
    edit(
        "SinkInFp8",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = data_objects::DataType::FP8_E4M3;
        },
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = s.dataType == data_objects::DataType::HALF
                                 ? data_objects::DataType::BFLOAT16
                                 : data_objects::DataType::HALF;
        });
    edit(
        "SinkInDouble",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = data_objects::DataType::DOUBLE;
        },
        withFloatSink);
    // The kernels read a head's sink at h * stride 16-bit halves in 32 bits: with 64 heads, a
    // stride of 2^26 puts heads 32 and up past it, and a FLOAT sink's 2^25 (two halves an
    // element) heads 32 and up too (shared servableSink).
    const auto sixtyFourHeadsWithASink = [withSink](SdpaGraphSpec& s) {
        withSink(s);
        s.heads = 64;
        s.kvHeads = 8;
    };
    edit(
        "SinkHeadOffsetsPastInt32",
        [sixtyFourHeadsWithASink](SdpaGraphSpec& s) {
            sixtyFourHeadsWithASink(s);
            s.sinkHeadStride = int64_t{1} << 26;
        },
        sixtyFourHeadsWithASink);
    const auto sixtyFourHeadsWithAFloatSink = [withFloatSink](SdpaGraphSpec& s) {
        withFloatSink(s);
        s.heads = 64;
        s.kvHeads = 8;
    };
    edit(
        "FloatSinkHeadOffsetsPastInt32",
        [sixtyFourHeadsWithAFloatSink](SdpaGraphSpec& s) {
            sixtyFourHeadsWithAFloatSink(s);
            s.sinkHeadStride = int64_t{1} << 25;
        },
        sixtyFourHeadsWithAFloatSink);

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

    // The scale: one finite host scalar at most 2^20 in magnitude, given one way.
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
    // Zero and negative scales are taken (sdpaScaleAccepts); NaN and either infinity are not.
    for(const auto& [name, value] :
        {std::pair{"NaN", std::numeric_limits<float>::quiet_NaN()},
         std::pair{"Infinite", std::numeric_limits<float>::infinity()},
         std::pair{"MinusInfinite", -std::numeric_limits<float>::infinity()},
         // Past the shared bound every family serves (2^20), either side of zero.
         std::pair{"AboveTheBound", 2097152.0F},
         std::pair{"BelowMinusTheBound", -2097152.0F}})
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
    edit("UnnamedDiagonalAlignmentUnderTheDeprecatedBoolean", [](SdpaGraphSpec& s) {
        s.deprecatedCausalTopLeft = true;
        s.diagonalAlignment = static_cast<data_objects::DiagonalAlignment>(7);
    });
    edit("LeftBoundBelowMinusOne", [](SdpaGraphSpec& s) { s.leftBound = -2; });
    edit("LeftBoundZero", [](SdpaGraphSpec& s) { s.leftBound = 0; });
    edit("RightBoundBelowMinusOne", [](SdpaGraphSpec& s) { s.rightBound = -2; });

    // Dtypes the kernels have no objects for (f32: every family but the CDNA generic and RDNA
    // prefill objects, whose suites drop this case and test f32 on their own).
    edit("Fp32", [](SdpaGraphSpec& s) { s.dataType = data_objects::DataType::FLOAT; });
    edit("Fp8", [](SdpaGraphSpec& s) { s.dataType = data_objects::DataType::FP8_E4M3; });
    edit("MixedDtypes", [](SdpaGraphSpec& s) { s.keyDataType = data_objects::DataType::HALF; });
    // V read, or O written, as the query's dtype while the graph states the other 16-bit one.
    edit("ValueInAnotherDtype",
         [](SdpaGraphSpec& s) { s.valueDataType = data_objects::DataType::HALF; });
    edit("OutputInAnotherDtype",
         [](SdpaGraphSpec& s) { s.outputDataType = data_objects::DataType::HALF; });

    // A rank-4 sink is [1, Hq, 1, 1]: an extent past 1 on any other axis is a sink the
    // kernels would read once per head (shared servableSink).
    const auto rank4Sink = [withSink](SdpaGraphSpec& s) {
        withSink(s);
        s.sinkRank4 = true;
    };
    for(const auto& [name, axis] : {std::pair{"Batch", size_t{0}},
                                    std::pair{"Sequence", size_t{2}},
                                    std::pair{"HeadDim", size_t{3}}})
    {
        const size_t wide = axis;
        edit(
            std::string("Rank4SinkWithA") + name + "Extent",
            [rank4Sink, wide](SdpaGraphSpec& s) {
                rank4Sink(s);
                s.sinkDims = {1, s.heads, 1, 1};
                s.sinkDims[wide] = 2;
            },
            rank4Sink);
    }
    for(auto& varlen : sdpaVarlenFieldDeclines())
    {
        cases.push_back(std::move(varlen));
    }
    for(auto& dropout : sdpaDropoutFieldDeclines())
    {
        cases.push_back(std::move(dropout));
    }
    return cases;
}

/// Each operand marked runtime pass-by-value: a host scalar where the kernel takes a device
/// pointer. The scale is the only tensor the FlyDSL SDPA engines read that way, so every
/// family declines each of these. A case with a feature carries the feature without the
/// mark as its control.
inline std::vector<SdpaFieldDecline> sdpaPassByValueDeclines()
{
    std::vector<SdpaFieldDecline> cases;
    const auto byValue
        = [](int64_t uid) { return [uid](SdpaGraphSpec& s) { s.passByValueTensors.insert(uid); }; };
    for(const auto& [name, uid] : {std::pair{"Q", FLYDSL_SDPA_RDNA_Q_UID},
                                   std::pair{"K", FLYDSL_SDPA_RDNA_K_UID},
                                   std::pair{"V", FLYDSL_SDPA_RDNA_V_UID},
                                   std::pair{"O", FLYDSL_SDPA_RDNA_O_UID}})
    {
        cases.push_back({std::string("PassByValue") + name, byValue(uid), {}});
    }
    const auto withFeature = [&cases, &byValue](std::string name,
                                                int64_t uid,
                                                const std::function<void(SdpaGraphSpec&)>& add) {
        const auto mark = byValue(uid);
        cases.push_back({std::move(name),
                         [add, mark](SdpaGraphSpec& s) {
                             add(s);
                             mark(s);
                         },
                         add});
    };
    withFeature("PassByValueBias", FLYDSL_SDPA_RDNA_BIAS_UID, [](SdpaGraphSpec& s) {
        s.withBias = true;
        s.biasDataType = s.dataType;
    });
    withFeature("PassByValueStats", FLYDSL_SDPA_RDNA_STATS_UID, [](SdpaGraphSpec& s) {
        s.stats = SdpaStats::LSE;
    });
    withFeature(
        "PassByValueSink", FLYDSL_SDPA_RDNA_SINK_UID, [](SdpaGraphSpec& s) { s.withSink = true; });
    withFeature("PassByValueFloatSink", FLYDSL_SDPA_RDNA_SINK_UID, [](SdpaGraphSpec& s) {
        s.withSink = true;
        s.sinkDataType = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    });
    return cases;
}

/// A softmax scale every FlyDSL SDPA engine takes (any finite one up to 2^20 in magnitude, as
/// PyTorch and AOTriton take any), stated as a value or a constant tensor, and whether it needs
/// scaled scores (FlydslSdpaShared.hpp needsScaledScores): the engines whose plain objects keep
/// raw scores then run a graph without a bias on their bias objects with a zero bias.
struct SdpaScaleAccept
{
    std::string name;
    SdpaScale kind = SdpaScale::VALUE;
    float scale = 0.0F;
    bool scaledScores = false;
};

inline std::vector<SdpaScaleAccept> sdpaScaleAccepts()
{
    std::vector<SdpaScaleAccept> cases;
    for(const auto& [name, value, scaled] :
        {std::tuple{"Zero", 0.0F, true},
         std::tuple{"NegativeZero", -0.0F, true},
         std::tuple{"Negative", -0.125F, true},
         std::tuple{"MinusTheBound", -1048576.0F, true},
         // Subnormal: its product with log2(e) may flush to zero, as zero's does.
         std::tuple{"Subnormal", 1.0e-40F, true},
         // The least normal scale the plain objects take, and the bound.
         std::tuple{"LeastNormal", std::numeric_limits<float>::min(), false},
         std::tuple{"TheBound", 1048576.0F, false}})
    {
        cases.push_back({std::string("Value") + name, SdpaScale::VALUE, value, scaled});
        cases.push_back(
            {std::string("ConstantTensor") + name, SdpaScale::CONSTANT_TENSOR, value, scaled});
    }
    return cases;
}

/// Whether @p spec states a scale, fixed in the graph, that needs scaled scores: zero,
/// negative or subnormal (FlydslSdpaShared.hpp needsScaledScores). A runtime scale is not known
/// at match, and an absent one is 1.
inline bool sdpaSpecNeedsScaledScores(const SdpaGraphSpec& spec)
{
    if(spec.scaleKind != SdpaScale::VALUE && spec.scaleKind != SdpaScale::CONSTANT_TENSOR)
    {
        return false;
    }
    return std::isfinite(spec.scale) && !(spec.scale >= std::numeric_limits<float>::min());
}

/// Whether the engines whose plain objects keep raw scores run @p spec on their bias objects
/// with a zero bias (FlydslSdpaShared.hpp runsOnAZeroBias).
inline bool sdpaSpecRunsOnAZeroBias(const SdpaGraphSpec& spec)
{
    return !spec.withBias && sdpaSpecNeedsScaledScores(spec);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
