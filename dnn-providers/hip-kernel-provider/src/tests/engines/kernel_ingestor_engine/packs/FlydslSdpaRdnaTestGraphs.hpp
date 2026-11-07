// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
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
#include "engines/kernel_ingestor_engine/packs/SdpaLayoutTestSupport.hpp"

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
    /// The sink's dtype as the graph declares it, FLOAT (cuDNN's and hipDNN's, SEM-022) when
    /// unset. Graph only: the device harnesses hold a FLOAT sink.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> sinkDataType;
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
    /// Device harnesses, packed with per-batch lengths: each sequence's first token in Q's
    /// (and O's, the LSE's) or K's (V's) table, in any order and with gaps (cuDNN's per-batch
    /// origins, RFC 0014), each sequence's packedSeqLens* region starting there; the table's
    /// last entry is then 0, which bounds no sequence. End to end when empty.
    std::vector<int32_t> packedOriginsQ;
    std::vector<int32_t> packedOriginsKv;
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
    /// Every Q, K and V element this value (device values only, both device harnesses), as
    /// AOTriton's test_large_bf16_nan_values fills them (133,120: raw scores near 3e11).
    std::optional<float> inputValue;
    /// Q, K and V drawn from N(0, 1) and rounded to the dtype (device values only, both
    /// device harnesses), as AOTriton's test_sm_scale_magnitude draws them: with scores of
    /// both signs a perturbation of the exponent is not common to a row, so softmax does not
    /// subtract it away.
    bool normalInputs = false;
    /// Q, K and V drawn from U[0, 1) and rounded to the dtype (device values only, both device
    /// harnesses), as AOTriton's test_common_mistakes[one_signed_values] draws them: P is never
    /// negative, so a P rounded toward zero before the second product biases O low wherever V
    /// does not have zero mean, which N(0, 1) values average away (AOTriton's lesson 16).
    bool uniformInputs = false;
    /// When set, O's mean signed error over the rows with keys, relative to the reference's mean
    /// magnitude there, at most this (both device harnesses): a correct rounding is unbiased, a P
    /// truncated to bf16 biases O by about -2^-8.5 relative on one-signed V.
    std::optional<double> meanBiasBound;
    /// Q's first column NaN at this query position, every batch and head.
    std::optional<int64_t> queryNanPosition;
    /// K's first column NaN at this key, every batch and KV head.
    std::optional<int64_t> keyNanPosition;
    /// Packed with per-batch lengths: every K and V element of a slot past its key length
    /// NaN, every head and column, so a kernel that reads a key past seq_len_kv (PyTorch's
    /// `seqused_k`) turns its rows NaN (both device harnesses).
    bool slotTailNan = false;
    /// V at every key no live row of its batch sees (sdpaSomeRowSeesKey: a key the mask or the
    /// window removes for every row, or one past the batch's key length) this value, every head
    /// and column (device values; both device harnesses). NaN or infinity there must not reach
    /// O: the references never read such a key (SEM-027), and neither may a kernel.
    std::optional<float> unseenValue;
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

/// A packed operand's ragged-offset table: @p origins (each sequence's first token, in any
/// order) and a last entry of 0, or, with no origins, the regions in @p slots end to end.
inline std::vector<int32_t> sdpaPackedTable(const std::vector<int32_t>& origins,
                                            const std::vector<int32_t>& slots)
{
    if(!origins.empty())
    {
        auto table = origins;
        table.push_back(0);
        return table;
    }
    std::vector<int32_t> table{0};
    for(const auto slot : slots)
    {
        table.push_back(table.back() + slot);
    }
    return table;
}

/// The tokens a packed operand's sequences span: past the last region of @p table (one entry
/// per sequence, then one more), each @p slots[b] tokens from table[b].
inline int64_t sdpaPackedTokens(const std::vector<int32_t>& table,
                                const std::vector<int32_t>& slots)
{
    int64_t tokens = 0;
    for(size_t b = 0; b < slots.size(); ++b)
    {
        tokens = std::max<int64_t>(tokens, int64_t{table[b]} + slots[b]);
    }
    return tokens;
}

/// The query rows packed sequence @p b owns, the rows its O and LSE hold (modification 37 of
/// the RDNA prefill kernel, 47 of the CDNA generic kernel): its query length, clamped to
/// [0, its slot], under per-batch lengths, else its slot.
inline int64_t sdpaPackedQueryRows(const SdpaGraphSpec& spec, int64_t b)
{
    const auto index = static_cast<size_t>(b);
    const int64_t slot = spec.packedSeqLensQ[index];
    const bool padded = !spec.seqLensQ.empty() && !spec.seqLensKv.empty();
    return padded ? std::clamp<int64_t>(spec.seqLensQ[index], 0, slot) : slot;
}

/// "" when every per-batch length of packed @p spec (clamped to [0, S]) fits its region, else
/// the first that does not: a length is its sequence's token count from its origin
/// (modification 37 of the RDNA prefill kernel, 47 of the CDNA generic kernel), so one past its
/// region would reach another sequence's tokens.
inline std::string sdpaPackedLengthsOutsideRegions(const SdpaGraphSpec& spec)
{
    if(!spec.packed() || spec.seqLensQ.empty() || spec.seqLensKv.empty())
    {
        return "";
    }
    for(size_t b = 0; b < static_cast<size_t>(spec.batch); ++b)
    {
        if(std::clamp<int64_t>(spec.seqLensQ[b], 0, spec.seqLenQ) > spec.packedSeqLensQ[b]
           || std::clamp<int64_t>(spec.seqLensKv[b], 0, spec.seqLenKv) > spec.packedSeqLensKv[b])
        {
            return "batch " + std::to_string(b) + "'s length is past its packed region";
        }
    }
    return "";
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

/// crossFamilyPaddedDropoutSpec()'s problem packed (THD) instead: the same sequences end to
/// end, so the same mask (AOTriton's varlen counters: the sequence's index, positions within
/// it, the graph's extents) and the same dump, which every family must write bit for bit.
inline SdpaGraphSpec crossFamilyPackedDropoutSpec()
{
    auto spec = crossFamilyPaddedDropoutSpec();
    spec.seqLensQ.clear();
    spec.seqLensKv.clear();
    spec.paddingMask = false;
    setPacked(spec, {40, 17, 0}, {70, 33, 70});
    return spec;
}

/// One named device case of a cross-family case list.
struct NamedSdpaSpec
{
    std::string name;
    SdpaGraphSpec spec;
};

/// Packed (THD) sequences that drop, the device cases every family runs (each on its own
/// dropout objects). Mixed lengths with a query-less and a keyless sequence, GQA, both
/// causal corners and none, the undropped LSE, the mask read off O bit for bit (Q zero, V the
/// identity), the RNG dump against the oracle, a seed and offset by value, and one table for
/// every operand. withLengths: cases with per-batch lengths in packed slots, which the aiw
/// objects decline. 16-bit; packedF32DropoutSpecs() has the f32 cases.
inline std::vector<NamedSdpaSpec> packedDropoutSpecs(bool withLengths)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    std::vector<NamedSdpaSpec> cases;
    for(const auto mask :
        {SdpaMask::NONE, SdpaMask::TOP_LEFT_CAUSAL, SdpaMask::BOTTOM_RIGHT_CAUSAL})
    {
        SdpaGraphSpec bits;
        bits.dataType = DataType::HALF;
        bits.batch = 3;
        bits.heads = 2;
        bits.kvHeads = 2;
        bits.headDim = 64;
        bits.seqLenQ = 48;
        bits.seqLenKv = 64;
        bits.mask = mask;
        bits.stats = SdpaStats::LSE;
        setDropout(bits, 0.25F, 0x0123456789ABCDEFULL, (uint64_t{1} << 32) - 7);
        setPacked(bits, {48, 29, 0}, {64, 35, 7});
        bits.exposeDropoutMask = true;
        cases.push_back({"MaskBitForBit" + std::to_string(static_cast<int>(mask)), bits});
    }
    SdpaGraphSpec gqa;
    gqa.batch = 3;
    gqa.headDim = 128;
    gqa.heads = 8;
    gqa.kvHeads = 2;
    gqa.seqLenQ = 150;
    gqa.seqLenKv = 130;
    gqa.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    gqa.stats = SdpaStats::LSE;
    setDropout(gqa, 0.2F, 42U, 5U);
    setPacked(gqa, {150, 0, 77}, {0, 130, 61});
    cases.push_back({"GqaBottomRightWithTheUndroppedLse", gqa});
    cases.push_back({"CrossFamilyRngDump", crossFamilyPackedDropoutSpec()});
    auto byValue = crossFamilyPackedDropoutSpec();
    byValue.mask = SdpaMask::TOP_LEFT_CAUSAL;
    byValue.seqLenKv = 53;
    setPacked(byValue, {40, 1, 25}, {53, 9, 0});
    byValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    byValue.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID);
    cases.push_back({"RngDumpWithASeedAndOffsetByValue", byValue});
    auto oneTable = crossFamilyPackedDropoutSpec();
    oneTable.mask = SdpaMask::TOP_LEFT_CAUSAL;
    oneTable.seqLenKv = oneTable.seqLenQ;
    setPacked(oneTable, {40, 17, 0}, {40, 17, 0});
    oneTable.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    oneTable.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    cases.push_back({"SelfAttentionThroughOneTable", oneTable});
    // AOTriton's test_common_mistakes[one_signed_values]: Q, K and V from U[0, 1), so a P
    // rounded toward zero before the second product biases O low (meanBiasBound), here on the
    // dropout objects under packing; GQA, a keyless sequence, both dtypes.
    for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
    {
        SdpaGraphSpec oneSigned;
        oneSigned.dataType = dataType;
        oneSigned.batch = 3;
        oneSigned.heads = 4;
        oneSigned.kvHeads = 2;
        oneSigned.headDim = 128;
        oneSigned.seqLenQ = 143;
        oneSigned.seqLenKv = 579;
        oneSigned.mask
            = dataType == DataType::HALF ? SdpaMask::BOTTOM_RIGHT_CAUSAL : SdpaMask::NONE;
        oneSigned.stats = SdpaStats::LSE;
        oneSigned.uniformInputs = true;
        oneSigned.meanBiasBound = dataType == DataType::HALF ? 0x1p-13 : 0x1p-11;
        setDropout(oneSigned, 0.2F, 7U, 0U);
        setPacked(oneSigned, {143, 100, 61}, {579, 0, 333});
        cases.push_back(
            {std::string("OneSignedValues") + (dataType == DataType::HALF ? "F16" : "Bf16"),
             oneSigned});
    }
    if(withLengths)
    {
        // Per-batch lengths in the slots; the slot's rows past a length are padded rows.
        auto slots = crossFamilyPackedDropoutSpec();
        setPacked(slots, {40, 30, 10}, {70, 50, 70});
        setPadding(slots, {33, 30, 0}, {70, 17, 64});
        cases.push_back({"PackedSlotsWithLengths", slots});
    }
    return cases;
}

/// packedDropoutSpecs()'s cases in f32, which every family's exact-f32 dropout objects serve
/// (on gfx12 the only RDNA prefill objects that drop): the same sequences, masks, tables and
/// dumps, so a mask drawn at the wrong extents fails here too. The one-signed cases are left
/// out: their bound is on a P rounded to 16 bits, which f32 does not do.
inline std::vector<NamedSdpaSpec> packedF32DropoutSpecs(bool withLengths)
{
    std::vector<NamedSdpaSpec> cases;
    for(auto& [name, spec] : packedDropoutSpecs(withLengths))
    {
        if(spec.uniformInputs)
        {
            continue;
        }
        spec.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        cases.push_back({name + "F32", spec});
    }
    return cases;
}

/// A left bound with no right one: the case list every family's device suite runs against
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

/// Whether a live row of batch @p b sees key @p key under @p spec's mask, by the rule the
/// device harnesses and hipDNN's references apply: a key at or past the batch's key length is
/// seen by none; a causal mask (or a right bound) hides a key past row i's diagonal plus the
/// right bound; a left bound L (64 for SLIDING_WINDOW) keeps the L keys ending at the diagonal.
/// The diagonal is row i + align, align Skv_b - Sq_b bottom-right and 0 top-left.
inline bool sdpaSomeRowSeesKey(const SdpaGraphSpec& spec, int64_t b, int64_t key)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    const auto index = static_cast<size_t>(b);
    const bool packed = spec.packed();
    const bool padded = !spec.seqLensQ.empty() && !spec.seqLensKv.empty();
    const int64_t slotQ = packed ? spec.packedSeqLensQ[index] : spec.seqLenQ;
    const int64_t slotKv = packed ? spec.packedSeqLensKv[index] : spec.seqLenKv;
    const int64_t lengthQ = padded ? std::clamp<int64_t>(spec.seqLensQ[index], 0, slotQ) : slotQ;
    const int64_t lengthKv
        = padded ? std::clamp<int64_t>(spec.seqLensKv[index], 0, slotKv) : slotKv;
    if(key >= lengthKv)
    {
        return false;
    }
    const bool causal = spec.mask != SdpaMask::NONE || spec.deprecatedCausalTopLeft
                        || spec.deprecatedCausalBottomRight || spec.rightBound.has_value();
    const bool bottomRight
        = spec.deprecatedCausalBottomRight
          || spec.diagonalAlignment.value_or(spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL
                                                 ? DiagonalAlignment::BOTTOM_RIGHT
                                                 : DiagonalAlignment::TOP_LEFT)
                 == DiagonalAlignment::BOTTOM_RIGHT;
    auto left = spec.leftBound;
    if(spec.mask == SdpaMask::SLIDING_WINDOW && !left.has_value())
    {
        left = 64;
    }
    const int64_t align = bottomRight ? lengthKv - lengthQ : 0;
    for(int64_t row = 0; row < lengthQ; ++row)
    {
        const bool pastDiagonal = causal && key > row + align + spec.rightBound.value_or(0);
        const bool beforeWindow = left.has_value() && key <= row + align - *left;
        if(!pastDiagonal && !beforeWindow)
        {
            return true;
        }
    }
    return false;
}

/// V at keys no row sees (SEM-027): the case list every family's device suite runs with
/// NaN and with infinity there (`KeepsValuesAtKeysNoRowSeesOutOfO`), each against the harness's
/// reference and hipDNN's CPU reference, which read no such key. f16 (one bf16), GQA, the LSE: a
/// left bound alone (bottom-right, more keys than queries); a right side alone (top-left causal
/// and a band, more keys than queries); both (a sliding window, bottom-right); vLLM's window
/// through its mapping (FlashAttention's window (31, 0) is left_bound 32 and right_bound 0,
/// bottom-right) for a prefill chunk and a decode step; padded and packed at both corners;
/// head_dim 128.
/// @p decodeShapes: only the cases the decode engines take (one to four query rows, no packing).
inline std::vector<NamedSdpaSpec> unseenValueSpecs(bool decodeShapes = false)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    const auto base = [](int64_t sq, int64_t skv) {
        SdpaGraphSpec spec;
        spec.dataType = DataType::HALF;
        spec.batch = 2;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.headDim = 64;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        spec.stats = SdpaStats::LSE;
        spec.crossCheckWithCpuReference = true;
        spec.unseenValue = std::numeric_limits<float>::quiet_NaN();
        return spec;
    };
    std::vector<NamedSdpaSpec> cases;
    if(decodeShapes)
    {
        // vLLM's decode step: one query row, FlashAttention's window (31, 0).
        auto step = base(1, 300);
        step.heads = 8;
        step.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        step.leftBound = 32;
        cases.push_back({"VllmWindowDecodeStep", step});
        auto rightOnly = base(4, 300);
        rightOnly.mask = SdpaMask::TOP_LEFT_CAUSAL;
        cases.push_back({"RightOnlyTopLeft", rightOnly});
        auto padded = step;
        padded.batch = 3;
        setPadding(padded, {1, 1, 0}, {300, 170, 33});
        cases.push_back({"VllmWindowDecodeStepPadded", padded});
        return cases;
    }
    auto leftOnly = base(70, 200);
    leftOnly.leftBound = 40;
    leftOnly.diagonalAlignment = DiagonalAlignment::BOTTOM_RIGHT;
    cases.push_back({"LeftOnlyBottomRight", leftOnly});
    auto rightOnly = base(70, 200);
    rightOnly.mask = SdpaMask::TOP_LEFT_CAUSAL;
    cases.push_back({"RightOnlyTopLeft", rightOnly});
    auto band = rightOnly;
    band.rightBound = 5;
    cases.push_back({"BandTopLeft", band});
    auto both = base(100, 300);
    both.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    both.leftBound = 48;
    cases.push_back({"BothSidesBottomRight", both});
    auto chunk = base(16, 300);
    chunk.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    chunk.leftBound = 32;
    cases.push_back({"VllmWindowPrefillChunk", chunk});
    auto step = chunk;
    step.seqLenQ = 1;
    step.heads = 8;
    cases.push_back({"VllmWindowDecodeStep", step});
    auto padded = both;
    padded.batch = 3;
    setPadding(padded, {100, 41, 0}, {300, 170, 33});
    cases.push_back({"BothSidesPadded", padded});
    auto packed = both;
    packed.batch = 3;
    setPacked(packed, {100, 41, 0}, {300, 170, 33});
    cases.push_back({"BothSidesPacked", packed});
    // Both sides top-left, padded and packed: the aiw objects serve windows under padding and
    // packing only top-left (§5.9).
    auto topLeft = both;
    topLeft.mask = SdpaMask::TOP_LEFT_CAUSAL;
    auto topLeftPadded = topLeft;
    topLeftPadded.batch = 3;
    setPadding(topLeftPadded, {100, 41, 0}, {300, 170, 33});
    cases.push_back({"BothSidesTopLeftPadded", topLeftPadded});
    auto topLeftPacked = topLeft;
    topLeftPacked.batch = 3;
    setPacked(topLeftPacked, {100, 41, 0}, {300, 170, 33});
    cases.push_back({"BothSidesTopLeftPacked", topLeftPacked});
    auto wide = both;
    wide.headDim = 128;
    cases.push_back({"BothSidesHeadDim128", wide});
    auto wideRight = rightOnly;
    wideRight.headDim = 128;
    cases.push_back({"RightOnlyHeadDim128", wideRight});
    auto bf16 = rightOnly;
    bf16.dataType = DataType::BFLOAT16;
    bf16.headDim = 128;
    cases.push_back({"RightOnlyBf16HeadDim128", bf16});
    return cases;
}

/// V with a head count of its own (Hk != Hv): the case list every family's device suite runs
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

/// Graphs one family declined that every family serves now, run by
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
    // Dropout at head_dim 96.
    auto d96 = base(96, 96);
    d96.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    setDropoutWithRngDump(d96, 0.2F, 0x0123456789ABCDEFULL, 11U);
    cases.push_back({"DropoutAtHeadDim96", d96});
    // Dropout with an f32 bias (rows a 32-byte multiple apart, as the aiw objects read).
    auto f32Bias = base(64, 96);
    f32Bias.withBias = true;
    f32Bias.biasDataType = DataType::FLOAT;
    f32Bias.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    setDropoutWithRngDump(f32Bias, 0.25F, 77U, 5U);
    cases.push_back({"DropoutWithAnF32Bias", f32Bias});
    // Dropout with a bias broadcast along the keys (key stride 0, read one key at a time).
    auto keyBroadcast = base(64, 96);
    keyBroadcast.withBias = true;
    keyBroadcast.biasDataType = DataType::HALF;
    keyBroadcast.biasDims = {2, 4, 64, 1};
    setDropoutWithRngDump(keyBroadcast, 0.25F, 78U, 6U);
    cases.push_back({"DropoutWithAKeyBroadcastBias", keyBroadcast});
    // A head between 256 and 512 with a bias, and with a zero scale. Eight query rows of a
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
    // Q/K and V head dims that differ, with a bias.
    auto pair = base(128, 96);
    pair.valueHeadDim = 64;
    pair.withBias = true;
    pair.biasDataType = DataType::HALF;
    cases.push_back({"QueryKey128Value64WithABias", pair});
    // Un-absorbed MLA with a bias, and with a negative scale.
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

/// The sink's dtype as a graph built from @p spec declares it: sinkDataType, else FLOAT.
inline hipdnn_flatbuffers_sdk::data_objects::DataType sdpaSinkDataType(const SdpaGraphSpec& spec)
{
    return spec.sinkDataType.value_or(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
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
        if(!lengths->empty() && (!spec.omitSeqLenQ || uid != FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID))
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

/// @p cases applied over @p base instead of the default spec: each case's edit and its control
/// start from base(), and base() itself (the control of a case without one) must be accepted.
/// For gates whose cases edit the spec in place, to show they hold for a feature they combine
/// with (for example the RNG dump's over packed operands).
inline std::vector<SdpaFieldDecline> sdpaDeclinesOver(std::vector<SdpaFieldDecline> cases,
                                                      const std::function<SdpaGraphSpec()>& base)
{
    for(auto& c : cases)
    {
        c.apply = [base, apply = std::move(c.apply)](SdpaGraphSpec& s) {
            s = base();
            apply(s);
        };
        c.control = [base, control = std::move(c.control)](SdpaGraphSpec& s) {
            s = base();
            if(control)
            {
                control(s);
            }
        };
    }
    return cases;
}

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
/// the control. Each family runs these over dense and over packed (THD) operands
/// (sdpaDeclinesOver): a packed graph's dump is the dense graph's.
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

/// The FlyDSL SDPA engine families whose graph matchers the shared padding, packed and dropout
/// decline tables below run on.
enum class SdpaFamily
{
    RDNA,
    CDNA,
};

/// The cases of a shared decline table, each run on @p base: a family's spec its matcher
/// accepts. A case replaces the spec with @p base and edits it; its control is @p base alone,
/// so a case cannot pass on a base the family declines.
class SdpaBaseDeclines
{
public:
    explicit SdpaBaseDeclines(std::function<SdpaGraphSpec()> base)
        : _base(std::move(base))
    {
    }

    void add(std::string name, std::function<void(SdpaGraphSpec&)> edit)
    {
        _cases.push_back({std::move(name),
                          [base = _base, edit = std::move(edit)](SdpaGraphSpec& s) {
                              s = base();
                              edit(s);
                          },
                          [base = _base](SdpaGraphSpec& s) { s = base(); }});
    }

    std::vector<SdpaFieldDecline> take()
    {
        return std::move(_cases);
    }

private:
    std::function<SdpaGraphSpec()> _base;
    std::vector<SdpaFieldDecline> _cases;
};

/// Each gate on the padding fields (FlydslSdpaVarlen.hpp, and servableVarlenAsReferenceReads's
/// lengths without padding_mask), applied to @p base, a padded graph the family's matcher
/// accepts. Every case runs on every family: the gates are shared.
inline std::vector<SdpaFieldDecline> sdpaPaddingDeclines(const std::function<SdpaGraphSpec()>& base)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    SdpaBaseDeclines table(base);
    const auto add = [&table](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        table.add(std::move(name), std::move(edit));
    };
    add("PaddingMaskWithoutQueryLengths", [](SdpaGraphSpec& s) { s.seqLensQ.clear(); });
    add("PaddingMaskWithoutKeyLengths", [](SdpaGraphSpec& s) { s.seqLensKv.clear(); });
    add("QueryLengthsAlone", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.seqLensKv.clear();
    });
    add("KeyLengthsAlone", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.seqLensQ.clear();
    });
    // Without padding_mask, the lengths' own gates still apply (SEM-008 reads them as
    // padding).
    add("Int64LengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.seqLenDataType = DataType::INT64;
    });
    add("VirtualQueryLengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
    });
    add("VirtualKeyLengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID);
    });
    add("KeyLengthsAlignedTo2WithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.alignment[FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID] = 2;
    });
    add("RaggedQueryWithLengthsWithoutThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = false;
        s.raggedTensors.insert(FLYDSL_SDPA_RDNA_Q_UID);
    });
    add("Int64Lengths", [](SdpaGraphSpec& s) { s.seqLenDataType = DataType::INT64; });
    add("FloatLengths", [](SdpaGraphSpec& s) { s.seqLenDataType = DataType::FLOAT; });
    add("LengthsOfTwoPerBatch", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch, 2, 1, 1}; });
    add("LengthsOfRankOne", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch}; });
    add("LengthsOfRankFive", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch, 1, 1, 1, 1}; });
    add("LengthsForAnotherBatch", [](SdpaGraphSpec& s) { s.seqLenDims = {s.batch + 1, 1, 1, 1}; });
    add("LengthsAtAStrideOfTwo", [](SdpaGraphSpec& s) { s.seqLenBatchStride = 2; });
    add("QueryLengthsPassedByValue",
        [](SdpaGraphSpec& s) { s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID); });
    add("KeyLengthsPassedByValue",
        [](SdpaGraphSpec& s) { s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID); });
    add("VirtualQueryLengths",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID); });
    add("VirtualKeyLengths",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID); });
    add("RaggedQueryLengths",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID); });
    add("RaggedKeyLengths",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID); });
    add("QueryLengthsAlignedTo2",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID] = 2; });
    add("KeyLengthsAlignedTo2",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID] = 2; });
    add("PaddedRaggedQuery",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_Q_UID); });
    return table.take();
}

/// @p s's packed slots padded to the lengths they hold: per-batch lengths equal to the slots.
inline void padThePackedSlots(SdpaGraphSpec& s)
{
    setPadding(s, s.packedSeqLensQ, s.packedSeqLensKv);
}

/// Each gate on packed (THD) sequences (FlydslSdpaVarlen.hpp servablePackedVarlenWithLengths,
/// and each matcher's packed LSE, bias and dropout rules), applied to @p base, a packed graph
/// the family's matcher accepts (packed sequences, or packed sequences that drop). Every case
/// runs on every family: the gates are shared, and both matchers decline a bias over packed
/// sequences.
inline std::vector<SdpaFieldDecline> sdpaPackedDeclines(const std::function<SdpaGraphSpec()>& base)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    SdpaBaseDeclines table(base);
    const auto add = [&table](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        table.add(std::move(name), std::move(edit));
    };
    constexpr int64_t Q_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    constexpr int64_t KV_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID;
    // An operand left dense beside packed ones, counting elements, or through the other table.
    for(const auto& [uid, name] : {std::pair{FLYDSL_SDPA_RDNA_Q_UID, "Query"},
                                   std::pair{FLYDSL_SDPA_RDNA_K_UID, "Key"},
                                   std::pair{FLYDSL_SDPA_RDNA_V_UID, "Value"},
                                   std::pair{FLYDSL_SDPA_RDNA_O_UID, "Output"}})
    {
        const int64_t operand = uid;
        const bool querySide = uid == FLYDSL_SDPA_RDNA_Q_UID || uid == FLYDSL_SDPA_RDNA_O_UID;
        add(std::string("Dense") + name,
            [operand](SdpaGraphSpec& s) { s.unpackedTensors.insert(operand); });
        add(std::string(name) + "CountingElements",
            [operand](SdpaGraphSpec& s) { s.raggedMultiplierOf[operand] = 1; });
        add(std::string(name) + "ThroughTheOtherTable", [operand, querySide](SdpaGraphSpec& s) {
            s.raggedOffsetOf[operand] = querySide ? KV_TABLE : Q_TABLE;
        });
    }
    add("KeyThroughATableTheGraphLacks", [](SdpaGraphSpec& s) {
        s.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID;
        s.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_UID;
    });
    add("HeadMajorPacking", [](SdpaGraphSpec& s) { s.layout = SdpaLayout::BHSD; });
    // The LSE: dense beside packed operands, or packed other than Q is.
    add("DenseLse", [](SdpaGraphSpec& s) { s.packedStats = false; });
    add("LseThroughTheKeyTable",
        [](SdpaGraphSpec& s) { s.raggedOffsetOf[FLYDSL_SDPA_RDNA_STATS_UID] = KV_TABLE; });
    add("LseCountingElements",
        [](SdpaGraphSpec& s) { s.raggedMultiplierOf[FLYDSL_SDPA_RDNA_STATS_UID] = 1; });
    add("HeadMajorLse", [](SdpaGraphSpec& s) { s.statsSequenceMajor = false; });
    // The tables.
    add("Int64Offsets", [](SdpaGraphSpec& s) { s.raggedOffsetDataType = DataType::INT64; });
    add("OffsetsForAnotherBatch",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch, 1, 1, 1}; });
    add("OffsetsOfRankOne", [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1}; });
    add("OffsetsOfTwoPerBatch",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1, 2, 1, 1}; });
    add("OffsetsWithASequenceExtent",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1, 1, 2, 1}; });
    add("OffsetsWithAHeadDimExtent",
        [](SdpaGraphSpec& s) { s.raggedOffsetDims = {s.batch + 1, 1, 1, 2}; });
    add("OffsetsAtAStrideOfTwo", [](SdpaGraphSpec& s) { s.raggedOffsetBatchStride = 2; });
    for(const auto& [uid, name] :
        {std::pair{Q_TABLE, "QueryOffsets"}, std::pair{KV_TABLE, "KeyOffsets"}})
    {
        const int64_t tableUid = uid;
        add(std::string("Virtual") + name,
            [tableUid](SdpaGraphSpec& s) { s.virtualTensors.insert(tableUid); });
        add(std::string(name) + "PassedByValue",
            [tableUid](SdpaGraphSpec& s) { s.passByValueTensors.insert(tableUid); });
        add(std::string("Ragged") + name,
            [tableUid](SdpaGraphSpec& s) { s.raggedTensors.insert(tableUid); });
        add(std::string(name) + "AlignedTo2",
            [tableUid](SdpaGraphSpec& s) { s.alignment[tableUid] = 2; });
    }
    // Per-batch lengths with packing (cuDNN's packed form): both or none, each one the kernels
    // read. One alone, the switch alone, or a length tensor the kernels cannot read is not.
    add("PackedUnderThePaddingMaskAlone", [](SdpaGraphSpec& s) { s.paddingMask = true; });
    add("PackedWithQueryLengths", [](SdpaGraphSpec& s) { s.seqLensQ = s.packedSeqLensQ; });
    add("PackedWithQueryLengthsUnderThePaddingMask", [](SdpaGraphSpec& s) {
        s.paddingMask = true;
        s.seqLensQ = s.packedSeqLensQ;
    });
    add("PackedWithInt64Lengths", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.seqLenDataType = DataType::INT64;
    });
    add("PackedWithLengthsOfAnotherBatch", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.seqLenDims = {s.batch + 1, 1, 1, 1};
    });
    // RFC 0014 reads a length at its batch-axis stride; the kernels read consecutive int32s.
    add("PackedWithLengthsAtAStrideOfTwo", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.seqLenBatchStride = 2;
    });
    add("OffsetsAtAStrideOfTwoWithLengths", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.raggedOffsetBatchStride = 2;
    });
    add("HeadMajorPackingWithLengths", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.layout = SdpaLayout::BHSD;
    });
    add("HeadMajorLseWithLengths", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.statsSequenceMajor = false;
    });
    // Each length tensor is checked on its own.
    for(const auto& [uid, name] : {std::pair{FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID, "Query"},
                                   std::pair{FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID, "Key"}})
    {
        const int64_t length = uid;
        add(std::string("PackedWithAVirtual") + name + "Length", [length](SdpaGraphSpec& s) {
            padThePackedSlots(s);
            s.virtualTensors.insert(length);
        });
    }
    // With both lengths, the packing is held to the same rules as without.
    for(const auto& [uid, name] : {std::pair{FLYDSL_SDPA_RDNA_Q_UID, "Query"},
                                   std::pair{FLYDSL_SDPA_RDNA_K_UID, "Key"},
                                   std::pair{FLYDSL_SDPA_RDNA_V_UID, "Value"},
                                   std::pair{FLYDSL_SDPA_RDNA_O_UID, "Output"}})
    {
        const int64_t operand = uid;
        add(std::string("Dense") + name + "WithLengths", [operand](SdpaGraphSpec& s) {
            padThePackedSlots(s);
            s.unpackedTensors.insert(operand);
        });
    }
    add("Int64OffsetsWithLengths", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.raggedOffsetDataType = DataType::INT64;
    });
    add("PaddedWithOffsetsCountingElements", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.raggedMultiplierOf[FLYDSL_SDPA_RDNA_O_UID] = 1;
    });
    add("PaddedWithKeysCountingElements", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.raggedMultiplierOf[FLYDSL_SDPA_RDNA_K_UID] = 1;
    });
    add("PaddedWithVirtualQueryOffsets", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.virtualTensors.insert(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);
    });
    add("PaddedWithADenseLse", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.packedStats = false;
    });
    // What packed sequences are not served with, with or without per-batch lengths: a bias (in
    // f32 or the query's dtype), laid out by batch and position. Dropout is served (each
    // sequence draws at the graph's extents and at positions within it).
    add("PackedWithABias", [](SdpaGraphSpec& s) { s.withBias = true; });
    add("PackedWithABiasOfTheQueryType", [](SdpaGraphSpec& s) {
        s.withBias = true;
        s.biasDataType = s.dataType;
    });
    add("PackedPaddedWithABias", [](SdpaGraphSpec& s) {
        padThePackedSlots(s);
        s.withBias = true;
        s.biasDataType = s.dataType;
    });
    return table.take();
}

/// Each gate on the dropout fields (FlydslSdpaDropout.hpp, and the matchers' combinations),
/// applied to @p base, a dropping graph @p family's matcher accepts (dense, or packed sequences).
/// Every case runs on every family but those marked onlyFor() with the reason.
inline std::vector<SdpaFieldDecline> sdpaDropoutDeclines(SdpaFamily family,
                                                         const std::function<SdpaGraphSpec()>& base)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using Builder = hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesBuilder;
    SdpaBaseDeclines table(base);
    const auto add = [&table](std::string name, std::function<void(SdpaGraphSpec&)> edit) {
        table.add(std::move(name), std::move(edit));
    };
    const auto field = [&add](std::string name, std::function<void(Builder&)> write) {
        add(std::move(name),
            [write = std::move(write)](SdpaGraphSpec& s) { s.extraAttributes = write; });
    };
    const auto onlyFor = [family](SdpaFamily only, const std::function<void()>& addCases) {
        if(family == only)
        {
            addCases();
        }
    };
    constexpr int64_t AUX = FLYDSL_SDPA_RDNA_AUX_UID;
    field("WithADropoutMask", [](Builder& b) { b.add_dropout_mask_tensor_uid(AUX); });
    field("WithADropoutScale", [](Builder& b) { b.add_dropout_scale_tensor_uid(AUX); });
    // A one-element dump no writer fills; a well-formed one is served (sdpaRngDumpDeclines).
    field("WithAnRngDump", [](Builder& b) { b.add_rng_dump_tensor_uid(AUX); });
    for(const auto& [name, p] :
        {std::pair{"ProbabilityOne", 1.0F},
         std::pair{"ProbabilityAboveOne", 1.5F},
         std::pair{"ProbabilityNegative", -0.1F},
         std::pair{"ProbabilityNaN", std::numeric_limits<float>::quiet_NaN()},
         std::pair{"ProbabilityInfinite", std::numeric_limits<float>::infinity()}})
    {
        const float probability = p;
        add(name, [probability](SdpaGraphSpec& s) { s.dropoutProbability = probability; });
    }
    add("WithoutASeed", [](SdpaGraphSpec& s) { s.withDropoutSeed = false; });
    add("WithoutAnOffset", [](SdpaGraphSpec& s) { s.withDropoutOffset = false; });
    add("SeedAndOffsetWithoutAProbability", [](SdpaGraphSpec& s) { s.dropoutProbability.reset(); });
    add("Int32SeedOnTheDevice", [](SdpaGraphSpec& s) { s.dropoutSeedDataType = DataType::INT32; });
    add("Int32OffsetOnTheDevice",
        [](SdpaGraphSpec& s) { s.dropoutOffsetDataType = DataType::INT32; });
    add("FloatSeedPassedByValue", [](SdpaGraphSpec& s) {
        s.dropoutSeedDataType = DataType::FLOAT;
        s.passByValueTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID);
    });
    add("SeedOfTwoValues", [](SdpaGraphSpec& s) { s.dropoutSeedDims = {1, 1, 1, 2}; });
    add("SeedOfRankOne", [](SdpaGraphSpec& s) { s.dropoutSeedDims = {1}; });
    add("SeedOfRankFive", [](SdpaGraphSpec& s) { s.dropoutSeedDims = {1, 1, 1, 1, 1}; });
    add("VirtualSeed",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID); });
    add("VirtualOffset",
        [](SdpaGraphSpec& s) { s.virtualTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID); });
    add("RaggedSeed",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID); });
    add("RaggedOffset",
        [](SdpaGraphSpec& s) { s.raggedTensors.insert(FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID); });
    add("SeedAlignedTo4",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_DROPOUT_SEED_UID] = 4; });
    add("OffsetAlignedTo4",
        [](SdpaGraphSpec& s) { s.alignment[FLYDSL_SDPA_RDNA_DROPOUT_OFFSET_UID] = 4; });
    // A bad seed declines even where nothing is drawn: the matcher gates what it reads.
    add("Int32DeviceSeedAtProbabilityZero", [](SdpaGraphSpec& s) {
        s.dropoutProbability = 0.0F;
        s.dropoutSeedDataType = DataType::INT32;
    });
    // A bias over packed sequences declines with dropout as without (laid out by batch). Two
    // sequences, each at most the base's extents.
    add("WithABiasOverPackedSequences", [](SdpaGraphSpec& s) {
        s.batch = 2;
        setPacked(
            s, {static_cast<int32_t>(s.seqLenQ), 37}, {100, static_cast<int32_t>(s.seqLenKv)});
        s.withBias = true;
        s.biasDataType = s.dataType;
    });
    // The CDNA dropout objects add no sink, so the CDNA matcher declines a sink with a drop.
    // The RDNA matcher binds the sink of a dropping graph, and its kernel match picks an object
    // that serves both or none.
    onlyFor(SdpaFamily::CDNA, [&add] {
        add("WithASink", [](SdpaGraphSpec& s) { s.withSink = true; });
        add("WithABiasAndASink", [](SdpaGraphSpec& s) {
            s.withBias = true;
            s.biasDataType = s.dataType;
            s.withSink = true;
        });
    });
    return table.take();
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
    // A FLOAT sink, cuDNN's and hipDNN's dtype (SEM-022; kernel modifications 31, 11, 36).
    const auto withSink = [](SdpaGraphSpec& s) { s.withSink = true; };
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
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.alignment[FLYDSL_SDPA_RDNA_SINK_UID] = 2;
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
    // The sink is FLOAT (SEM-022): any other dtype declines (shared servableSinkStride), the
    // 16-bit ones too, though the kernels could read them; the FLOAT sink is the control.
    edit(
        "SinkInFp8",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = data_objects::DataType::FP8_E4M3;
        },
        withSink);
    edit(
        "SinkInDouble",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = data_objects::DataType::DOUBLE;
        },
        withSink);
    edit(
        "SinkInTheQueryDtype",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = s.dataType;
        },
        withSink);
    edit(
        "SinkInTheOther16BitDtype",
        [withSink](SdpaGraphSpec& s) {
            withSink(s);
            s.sinkDataType = s.dataType == data_objects::DataType::HALF
                                 ? data_objects::DataType::BFLOAT16
                                 : data_objects::DataType::HALF;
        },
        withSink);
    // The kernels read a head's sink at h * stride 16-bit halves in 32 bits, two halves a FLOAT
    // element: with 64 heads, a stride of 2^26 puts heads 16 and up past it, and 2^25 heads 32
    // and up (shared servableSinkStride).
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
    edit(
        "FloatSinkHeadOffsetsPastInt32",
        [sixtyFourHeadsWithASink](SdpaGraphSpec& s) {
            sixtyFourHeadsWithASink(s);
            s.sinkHeadStride = int64_t{1} << 25;
        },
        sixtyFourHeadsWithASink);

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

/// O's signed error and the reference's magnitude, summed over the elements of the rows with
/// keys, against SdpaGraphSpec::meanBiasBound (both device harnesses).
struct SdpaMeanBias
{
    double signedErrorSum = 0.0;
    double magnitudeSum = 0.0;

    void add(double actual, double expected)
    {
        signedErrorSum += actual - expected;
        magnitudeSum += std::abs(expected);
    }

    /// Empty when @p bound is unset or held; else what failed.
    std::string failure(const std::optional<double>& bound) const
    {
        if(!bound.has_value())
        {
            return {};
        }
        if(!(magnitudeSum > 0.0))
        {
            return "no element of a row with keys to measure O's bias over";
        }
        const double bias = signedErrorSum / magnitudeSum;
        if(std::abs(bias) <= *bound)
        {
            return {};
        }
        return "O is biased by " + std::to_string(bias) + " of its mean magnitude (bound "
               + std::to_string(*bound)
               + "): something rounds one-signed values in one direction, e.g. a P "
                 "truncated before the second product";
    }
};

/// The value source @p spec's inputValue, normalInputs or uniformInputs asks for, one call per
/// element: the constant, N(0, 1) draws (Box-Muller over a 64-bit LCG seeded with @p seed, so
/// each operand gets its own deterministic stream) or U[0, 1) draws (the same LCG). Empty when
/// the spec asks for none.
inline std::function<float()> sdpaInputValueSource(const SdpaGraphSpec& spec, uint64_t seed)
{
    if(spec.inputValue.has_value())
    {
        const float value = *spec.inputValue;
        return [value]() { return value; };
    }
    if(spec.uniformInputs)
    {
        return [state = seed * 0x9E3779B97F4A7C15ULL + 1ULL]() mutable {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            // 24 random bits in [0, 1): exact in f32.
            return static_cast<float>(state >> 40) * 0x1.0p-24F;
        };
    }
    if(!spec.normalInputs)
    {
        return {};
    }
    struct Normal
    {
        uint64_t state;
        std::optional<float> spare;

        double uniform()
        {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            // 53 random bits in (0, 1]: never 0, so the log below is finite.
            return (static_cast<double>(state >> 11) + 1.0) * 0x1.0p-53;
        }

        float operator()()
        {
            if(spare.has_value())
            {
                const float value = *spare;
                spare.reset();
                return value;
            }
            const double radius = std::sqrt(-2.0 * std::log(uniform()));
            const double angle = 6.283185307179586 * uniform();
            spare = static_cast<float>(radius * std::sin(angle));
            return static_cast<float>(radius * std::cos(angle));
        }
    };
    return Normal{seed * 0x9E3779B97F4A7C15ULL + 1ULL, std::nullopt};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
