// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"

/**
 * @file FlydslSdpaBwdCommon.hpp
 * @brief The graph match every FlyDSL SDPA backward engine shares: one pure function of the
 *        graph that either computes the problem the engines launch or says why it declines.
 *
 * Rules are architecture-neutral. A limit that belongs to one engine (tile routing, merged
 * versus split launches, the bias staging constraints) is the engine's, and the problem
 * records what that engine needs to apply it.
 *
 * Guarded by the ingestor alone: FlydslSdpaShared.hpp sits behind HIPDNN_ENGINE_FLYDSL, so
 * the helpers this match shares with the forward engines are SdpaGraphMatch.hpp's. A match
 * makes no runtime calls and keeps no state; it reads only the graph.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_bwd
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The uid, bound or window a graph leaves unset.
inline constexpr int64_t NO_UID = -1;
inline constexpr int64_t NO_WINDOW = -1;

/// Where the softmax scale comes from: attn_scale_value, or a pass-by-value scale tensor.
enum class ScaleSource : uint8_t
{
    VALUE,
    TENSOR
};

/// Width of the dropout seed and offset tensors; NONE without dropout.
enum class SeedWidth : uint8_t
{
    NONE,
    INT32,
    INT64
};

/// How the dropout seed and offset reach the kernel; they share one form. NONE without
/// dropout.
enum class SeedForm : uint8_t
{
    NONE,
    DEVICE,
    PASS_BY_VALUE
};

/// Element strides of a (B, H, S) tensor's batch, head and sequence axes; 0 on a unit axis,
/// or on a batch or head axis the tensor broadcasts. A multi-row sequence axis never has
/// stride 0 (LY-6).
struct Strides
{
    int64_t batch = 0;
    int64_t head = 0;
    int64_t seq = 0;
};

/// The offset tensor of a ragged tensor: the base of batch b is offset[b] * multiplier
/// elements (cuDNN's Ragged Layout). NO_UID for a tensor that is not ragged, which is
/// addressed at b * its batch stride.
struct RaggedOffset
{
    int64_t uid = NO_UID;
    int64_t multiplier = 0;

    bool ragged() const
    {
        return uid != NO_UID;
    }
};

/// Each tensor's own ragged offset. The stats' offset is in the stats' own units.
struct RaggedOffsets
{
    RaggedOffset q;
    RaggedOffset k;
    RaggedOffset v;
    RaggedOffset o;
    RaggedOffset dO;
    RaggedOffset dQ;
    RaggedOffset dK;
    RaggedOffset dV;
    RaggedOffset stats;

    bool any() const
    {
        return q.ragged() || k.ragged() || v.ragged() || o.ragged() || dO.ragged() || dQ.ragged()
               || dK.ragged() || dV.ragged() || stats.ragged();
    }
};

/// What an engine opts into when it asks for the match.
struct MatchOptions
{
    /// Accept FLOAT (fp32) data tensors, bias and dBias. Off, FLOAT declines [data_type] (DT-2).
    bool acceptFloat = false;
    /// Accept a FLOAT bias (and dBias) beside f16 or bf16 Q, K and V. Off, it declines
    /// [attn_mask] (BI-2).
    bool acceptF32Bias = false;
    /// Accept V and dV with a head count of their own, when Hq is a multiple of both Hk and Hv
    /// and the graph has no bias, is not FLOAT and is not ragged. Off, a V whose heads differ
    /// from K's declines [shape] (SH-1).
    bool acceptValueHeads = false;
};

/// What the split engine opts into: every option.
inline constexpr MatchOptions SPLIT_MATCH_OPTIONS = [] {
    MatchOptions options;
    options.acceptFloat = true;
    options.acceptF32Bias = true;
    options.acceptValueHeads = true;
    return options;
}();

/// The head-dim tile and floor of one shipped band of the f32-bias and `Hk != Hv` rows: a
/// padded row serves head dims above `floor` and at most `tile`; an exact row (floor
/// `tile - 1`) serves the tile itself.
struct HeadDimBand
{
    int64_t floor = 0;
    int64_t tile = 0;
};

/// The bands the shipped f32-bias and `Hk != Hv` rows cover, per architecture family (the
/// split engine's graph_match logs a graph outside them; host test
/// TestFlydslSdpaBwdSplitPacks.cpp derives the bands from the loaded descriptors and compares).
inline constexpr std::array<HeadDimBand, 2> GFX950_PARITY_BANDS{{{63, 64}, {127, 128}}};
inline constexpr std::array<HeadDimBand, 2> GFX11_PARITY_BANDS{{{0, 64}, {0, 128}}};

/// What the graph match computes, once, for every engine that serves the graph.
struct SdpaBwdProblem
{
    /// f16 or bf16 on every tensor; FLOAT when the engine accepts it (MatchOptions).
    data_objects::DataType dtype = data_objects::DataType::UNSET;
    /// Width of one element of dtype: 2 for f16 and bf16, 4 for FLOAT.
    uint32_t elementBytes = 0;
    /// Width of one element of the bias and dBias: the bias's own dtype (4 for a FLOAT bias
    /// beside a 16-bit Q), elementBytes when the graph has no bias.
    uint32_t biasElementBytes = 0;
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
    /// V's and dV's head count: numKvHeads unless the engine opted into acceptValueHeads.
    int64_t numVHeads = 0;
    /// Tensor dimensions.
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t headDimQk = 0;
    int64_t headDimV = 0;
    /// Per operand; the last stride of each is 1.
    Strides q;
    Strides k;
    Strides v;
    Strides o;
    Strides dO;
    Strides dQ;
    Strides dK;
    Strides dV;
    /// Per operand, its ragged offset; the strides above record 0 on the batch axis of a
    /// ragged tensor, which is never read.
    RaggedOffsets ragged;
    /// The stats' (batch, head, sequence) element strides, 0 on a unit axis and on the
    /// batch axis of ragged stats.
    Strides stats;

    /// scaleValue is attn_scale_value, or the baked value of a scale tensor that carries one;
    /// 1.0 when the graph states none. scaleUid is NO_UID unless the source is TENSOR.
    ScaleSource scaleSource = ScaleSource::VALUE;
    float scaleValue = 1.0F;
    int64_t scaleUid = NO_UID;

    /// Key j is hidden when j > i + diagonalOffset (0 top-left, Skv - Sq bottom-right).
    bool masked = false;
    /// The graph's resolved diagonal alignment is BOTTOM_RIGHT, whether or not a bound is set.
    /// Under padding_mask each sequence anchors to its own Skv - Sq, which diagonalOffset (from
    /// the tensor dims) cannot express.
    bool alignBottomRight = false;
    int64_t diagonalOffset = 0;
    /// The sliding window in cuDNN's convention: left_bound L keeps the L keys up to and
    /// including the diagonal, so key j is hidden when j < i + diagonalOffset - L + 1; key j
    /// is hidden when j > i + diagonalOffset + right. NO_WINDOW for an unbounded side.
    int64_t windowLeft = NO_WINDOW;
    int64_t windowRight = NO_WINDOW;

    /// Padded varlen: per-batch lengths, NO_UID when absent.
    bool paddingMask = false;
    int64_t seqLenQUid = NO_UID;
    int64_t seqLenKvUid = NO_UID;

    /// Bias and its gradient, NO_UID when absent. The strides and the declared alignment are
    /// recorded for the engine's staging rules. A broadcast bias needs the dBias reduction.
    int64_t biasUid = NO_UID;
    int64_t dbiasUid = NO_UID;
    Strides biasStrides;
    Strides dbiasStrides;
    int64_t biasAlignmentBytes = 0;
    bool biasBroadcastBatch = false;
    bool biasBroadcastHead = false;

    float dropoutP = 0.0F;
    SeedWidth seedWidth = SeedWidth::NONE;
    SeedForm seedForm = SeedForm::NONE;
    int64_t seedUid = NO_UID;
    int64_t offsetUid = NO_UID;
};

/// Why the graph is not served: a category and the field and value that put it there.
struct Decline
{
    std::string cause;
    std::string detail;
};

namespace detail
{

using hipdnn_plugin_sdk::ingestor::MatchContext;
using Tensor = data_objects::TensorAttributes;
using Verdict = std::optional<Decline>;

inline constexpr uint32_t SDPA_RANK = 4;
inline constexpr uint32_t BATCH_AXIS = 0;
inline constexpr uint32_t HEAD_AXIS = 1;
inline constexpr uint32_t SEQ_AXIS = 2;
inline constexpr uint32_t DIM_AXIS = 3;

using sdpa_graph::INT32_LIMIT;
inline constexpr int64_t STATS_BYTES_PER_ELEMENT = 4; // stats is FLOAT
inline constexpr int64_t INT32_BYTES = 4;
inline constexpr int64_t INT64_BYTES = 8;
inline constexpr int64_t REQUIRED_ALIGNMENT_BYTES = 16;
inline constexpr int64_t MAX_HEAD_DIM = 512;
inline constexpr int64_t VECTOR_ELEMENTS = 8;

using sdpa_graph::checkedProduct;
using sdpa_graph::declaresAlignment;
using sdpa_graph::findTensor;
using sdpa_graph::MAX_SOFTMAX_SCALE;

/// Elements a slab of @p rows + 1 rows at @p rowStride elements reaches, the last @p width
/// elements wide; nullopt when it does not fit in int64_t.
inline std::optional<int64_t> slabElements(int64_t rows, int64_t rowStride, int64_t width)
{
    const auto reach = checkedProduct({rows, rowStride});
    int64_t total = 0;
    if(!reach.has_value() || __builtin_add_overflow(*reach, width, &total))
    {
        return std::nullopt;
    }
    return total;
}

using sdpa_graph::elementsDoNotOverlap;

/// @p value rounded up to a multiple of @p vector elements: the 16-bit vector of 8 by default,
/// 4 for FLOAT.
inline int64_t ceilToVector(int64_t value, int64_t vector = VECTOR_ELEMENTS)
{
    return (value + vector - 1) / vector * vector;
}

/// Rows past a sequence's end that the kernels still form offsets for. The gfx950 buffer
/// descriptors bound a row by `num_records`, but a tile origin carried in `soffset` is outside
/// that range check and the offsets are 32 bits wide, so the padded rows count against the
/// 32-bit byte limit like the real ones. These constants mirror the kernels' padded-row
/// addressing; a kernel change to either bound changes them.
///
/// Query axis: the dK/dV kernel walks an even number of query tiles of at most
/// `DKDV_QUERY_TILE` rows, starting at any tile of the sequence, and prefetches two tiles
/// past the last (`fmha_bwd_dkdv_gfx950.py`: `split_t_end`, the `stage_q_tile` /
/// `stage_do_tile` prefetch at `j + slot + NBUF`, NBUF <= 2). The walk can end one tile past
/// `ceil(S / tile)`, so three tiles of padding follow the last whole tile. The dQ kernels'
/// query tiles (`block_m` 128 on gfx950, 256 on gfx11) end inside that bound.
inline constexpr int64_t DKDV_QUERY_TILE = 64;
inline constexpr int64_t DKDV_QUERY_PAD_TILES = 3;
/// Key axis: the dQ kernel walks an even number of at least four key tiles of 64 rows
/// (`flash_attn_utils.py`: `max_num_tiles`), and the dK/dV and merged kernels own key tiles of
/// at most `KV_TILE` rows (`block_kv`).
inline constexpr int64_t DQ_MIN_KEY_ROWS = 256;
inline constexpr int64_t KV_TILE = 128;

/// @p rows limited to one past INT32_LIMIT. A sequence that long fails every DT-9 slab check
/// whatever its padding, so the padding is added to the limited extent and cannot overflow
/// int64 for a graph-controlled extent.
inline int64_t limitedRows(int64_t rows)
{
    return std::min(rows, INT32_LIMIT + 1);
}

/// Query-axis rows of a sequence of @p rows the kernels address offsets for; INT32_LIMIT + 1
/// plus the padding for a sequence beyond the limit.
inline int64_t addressedQueryRows(int64_t rows)
{
    return ceilToVector(limitedRows(rows), DKDV_QUERY_TILE)
           + DKDV_QUERY_PAD_TILES * DKDV_QUERY_TILE;
}

/// Key-axis rows of a sequence of @p rows the kernels address offsets for; INT32_LIMIT + 1
/// rounded up for a sequence beyond the limit.
inline int64_t addressedKeyRows(int64_t rows)
{
    return std::max(DQ_MIN_KEY_ROWS, ceilToVector(limitedRows(rows), KV_TILE));
}

/// Bytes per element of a data type the matcher serves, 0 for one it does not.
inline uint32_t servedElementBytes(data_objects::DataType dataType, bool acceptFloat)
{
    if(dataType == data_objects::DataType::HALF || dataType == data_objects::DataType::BFLOAT16)
    {
        return 2;
    }
    return acceptFloat && dataType == data_objects::DataType::FLOAT ? 4 : 0;
}

/// Whether every stride of @p tensor is the dense row-major stride of its dims.
inline bool stridesAreDense(const Tensor& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size())
    {
        return false;
    }
    int64_t expected = 1;
    for(flatbuffers::uoffset_t axis = dims->size(); axis > 0; --axis)
    {
        if(strides->Get(axis - 1) != expected)
        {
            return false;
        }
        const auto next = checkedProduct({expected, dims->Get(axis - 1)});
        if(!next.has_value())
        {
            return false;
        }
        expected = *next;
    }
    return true;
}

template <typename Range>
std::string listOf(const Range& values)
{
    std::string text = "[";
    bool first = true;
    for(const auto value : values)
    {
        text += (first ? "" : ",") + std::to_string(value);
        first = false;
    }
    return text + "]";
}

inline std::string typeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

/// A tensor the graph names by role, with the uid it names it by.
struct Operand
{
    const char* label;
    int64_t uid;
    const Tensor* tensor = nullptr;
};

inline std::string describe(const Operand& operand)
{
    return std::string(operand.label) + " (uid " + std::to_string(operand.uid) + ")";
}

inline const flatbuffers::Vector<int64_t>& dimsOf(const Operand& operand)
{
    return *operand.tensor->dims();
}

inline const flatbuffers::Vector<int64_t>& stridesOf(const Operand& operand)
{
    return *operand.tensor->strides();
}

/// Finds @p operand's tensor; one the kernels cannot read through a pointer (absent, virtual,
/// or, unless @p allowPassByValue, pass-by-value) declines under @p cause.
inline Verdict resolveOperand(const MatchContext& context,
                              Operand& operand,
                              const char* cause = "operand",
                              bool allowPassByValue = false)
{
    operand.tensor = findTensor(context, operand.uid);
    if(operand.tensor == nullptr)
    {
        return Decline{cause, describe(operand) + " is not in the graph"};
    }
    if(operand.tensor->virtual_())
    {
        return Decline{cause, describe(operand) + " is virtual"};
    }
    if(!allowPassByValue && hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(operand.tensor))
    {
        return Decline{cause, describe(operand) + " is pass-by-value"};
    }
    return std::nullopt;
}

/// The mask a graph asks for, with the deprecated causal booleans resolved as if their
/// setters ran first (right bound 0 unless right_bound is set).
struct MaskBounds
{
    int64_t left = NO_WINDOW;
    int64_t right = NO_WINDOW;
    bool alignBottomRight = false;
};

class BackwardMatcher
{
public:
    BackwardMatcher(const MatchContext& context,
                    const data_objects::SdpaBackwardAttributes& attributes,
                    const MatchOptions& options)
        : _context(context)
        , _attributes(attributes)
        , _options(options)
    {
    }

    std::variant<SdpaBwdProblem, Decline> run()
    {
        const std::array<Verdict (BackwardMatcher::*)(), 14> steps{
            &BackwardMatcher::resolveOperands,
            &BackwardMatcher::checkUidAliasing,
            &BackwardMatcher::checkShapes,
            &BackwardMatcher::checkDataTypes,
            &BackwardMatcher::checkHeads,
            &BackwardMatcher::checkRagged,
            &BackwardMatcher::checkAlignment,
            &BackwardMatcher::checkLayouts,
            &BackwardMatcher::checkStats,
            &BackwardMatcher::checkMask,
            &BackwardMatcher::checkVarlen,
            &BackwardMatcher::checkBias,
            &BackwardMatcher::checkDropout,
            &BackwardMatcher::checkScale};
        for(const auto step : steps)
        {
            if(auto verdict = (this->*step)())
            {
                return *std::move(verdict);
            }
        }
        return _problem;
    }

private:
    static constexpr size_t Q = 0;
    static constexpr size_t K = 1;
    static constexpr size_t V = 2;
    static constexpr size_t O = 3;
    static constexpr size_t DO = 4;
    static constexpr size_t DQ = 5;
    static constexpr size_t DK = 6;
    static constexpr size_t DV = 7;

    const MatchContext& _context;
    const data_objects::SdpaBackwardAttributes& _attributes;
    const MatchOptions _options;
    SdpaBwdProblem _problem;
    std::array<Operand, 8> _operands{};
    Operand _stats{"stats", NO_UID};
    /// The head dim each operand carries: Qk for Q, K, dQ, dK; V's for V, O, dO, dV.
    std::array<int64_t, 8> _headDims{};

    /// Elements of one 16-byte vector in the problem's dtype: 8 for f16 and bf16, 4 for FLOAT.
    int64_t vectorElements() const
    {
        return REQUIRED_ALIGNMENT_BYTES / static_cast<int64_t>(_problem.elementBytes);
    }

    /// Elements of one 16-byte vector in the bias's dtype: 8 for a 16-bit bias, 4 for FLOAT.
    int64_t biasVectorElements() const
    {
        return REQUIRED_ALIGNMENT_BYTES / static_cast<int64_t>(_problem.biasElementBytes);
    }

    int64_t dim(size_t operand, uint32_t axis) const
    {
        return dimsOf(_operands[operand]).Get(axis);
    }

    /// ST-4: an operand uid that is missing, virtual or pass-by-value declines [operand].
    Verdict resolveOperands()
    {
        const auto& a = _attributes;
        _operands = {{{"Q", a.q_tensor_uid()},
                      {"K", a.k_tensor_uid()},
                      {"V", a.v_tensor_uid()},
                      {"O", a.o_tensor_uid()},
                      {"dO", a.do_tensor_uid()},
                      {"dQ", a.dq_tensor_uid()},
                      {"dK", a.dk_tensor_uid()},
                      {"dV", a.dv_tensor_uid()}}};
        _stats = Operand{"stats", a.stats_tensor_uid()};
        for(auto& operand : _operands)
        {
            if(auto verdict = resolveOperand(_context, operand))
            {
                return verdict;
            }
        }
        return resolveOperand(_context, _stats);
    }

    /// ST-5: an output (dQ, dK, dV, dBias) must not share its uid with any other role; inputs
    /// may share among themselves.
    Verdict checkUidAliasing()
    {
        const auto& a = _attributes;
        struct Role
        {
            const char* label;
            int64_t uid;
            bool isOutput;
        };
        std::vector<Role> roles{{"Q", a.q_tensor_uid(), false},
                                {"K", a.k_tensor_uid(), false},
                                {"V", a.v_tensor_uid(), false},
                                {"O", a.o_tensor_uid(), false},
                                {"dO", a.do_tensor_uid(), false},
                                {"stats", a.stats_tensor_uid(), false},
                                {"dQ", a.dq_tensor_uid(), true},
                                {"dK", a.dk_tensor_uid(), true},
                                {"dV", a.dv_tensor_uid(), true}};
        const std::array<std::pair<const char*, flatbuffers::Optional<int64_t>>, 9> optional
            = {{{"scale", a.scale_tensor_uid()},
                {"bias", a.attn_mask_tensor_uid()},
                {"seq_len_q", a.seq_len_q_tensor_uid()},
                {"seq_len_kv", a.seq_len_kv_tensor_uid()},
                {"seed", a.seed_tensor_uid()},
                {"offset", a.offset_tensor_uid()},
                {"dropout_mask", a.dropout_mask_tensor_uid()},
                {"dropout_scale", a.dropout_scale_tensor_uid()},
                {"dropout_scale_inv", a.dropout_scale_inv_tensor_uid()}}};
        for(const auto& [label, uid] : optional)
        {
            if(uid.has_value())
            {
                roles.push_back({label, uid.value(), false});
            }
        }
        if(a.dbias_tensor_uid().has_value())
        {
            roles.push_back({"dBias", a.dbias_tensor_uid().value(), true});
        }
        for(size_t first = 0; first < roles.size(); ++first)
        {
            for(size_t second = first + 1; second < roles.size(); ++second)
            {
                const auto& x = roles[first];
                const auto& y = roles[second];
                if(x.uid == y.uid && (x.isOutput || y.isOutput))
                {
                    const auto& output = x.isOutput ? x : y;
                    const auto& other = x.isOutput ? y : x;
                    return Decline{"operand",
                                   std::string(output.label) + " (uid " + std::to_string(output.uid)
                                       + ") shares its uid with " + other.label};
                }
            }
        }
        return std::nullopt;
    }

    static Verdict expectDims(const Operand& operand,
                              const std::array<int64_t, 4>& expected,
                              std::string_view why)
    {
        const auto& dims = dimsOf(operand);
        if(!std::equal(dims.begin(), dims.end(), expected.begin()))
        {
            return Decline{"shape",
                           describe(operand) + " dims " + listOf(dims) + " != " + listOf(expected)
                               + " (" + std::string(why) + ")"};
        }
        return std::nullopt;
    }

    /// SH-1: dQ, dK and dV match Q, K and V; O and dO are [B, Hq, Sq, Dv]; V's (and dV's)
    /// heads equal K's, or are V's own when the engine opts in (MatchOptions::acceptValueHeads;
    /// checkHeads bounds them); stats is [B, Hq, Sq, 1]; every extent is positive.
    Verdict checkShapes()
    {
        const auto all = [this](auto&& visit) -> Verdict {
            for(const auto& operand : _operands)
            {
                if(auto verdict = visit(operand))
                {
                    return verdict;
                }
            }
            return visit(_stats);
        };
        if(auto verdict = all([](const Operand& operand) -> Verdict {
               const auto* dims = operand.tensor->dims();
               const auto* strides = operand.tensor->strides();
               if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
                  || strides->size() != SDPA_RANK)
               {
                   return Decline{"shape",
                                  describe(operand) + " is not rank 4 with rank-4 strides"};
               }
               for(const auto extent : *dims)
               {
                   if(extent <= 0)
                   {
                       return Decline{"shape",
                                      describe(operand) + " has a non-positive extent in dims "
                                          + listOf(*dims)};
                   }
               }
               return std::nullopt;
           }))
        {
            return verdict;
        }

        const int64_t batch = dim(Q, BATCH_AXIS);
        const int64_t heads = dim(Q, HEAD_AXIS);
        const int64_t seqQ = dim(Q, SEQ_AXIS);
        const int64_t dimQk = dim(Q, DIM_AXIS);
        const int64_t kvHeads = dim(K, HEAD_AXIS);
        const int64_t seqKv = dim(K, SEQ_AXIS);
        const int64_t dimV = dim(V, DIM_AXIS);
        const int64_t valueHeads = _options.acceptValueHeads ? dim(V, HEAD_AXIS) : kvHeads;

        const std::array<int64_t, 4> q{batch, heads, seqQ, dimQk};
        const std::array<int64_t, 4> k{batch, kvHeads, seqKv, dimQk};
        const std::array<int64_t, 4> v{batch, valueHeads, seqKv, dimV};
        const std::array<int64_t, 4> o{batch, heads, seqQ, dimV};
        const std::array<int64_t, 4> stats{batch, heads, seqQ, 1};
        const auto checkAll = [&]() -> Verdict {
            if(auto verdict = expectDims(_operands[K], k, "batch and head dim follow Q"))
            {
                return verdict;
            }
            if(auto verdict
               = expectDims(_operands[V],
                            v,
                            _options.acceptValueHeads
                                ? "batch follows Q; sequence follows K; heads are V's own"
                                : "batch follows Q; heads and sequence follow K"))
            {
                return verdict;
            }
            if(auto verdict = expectDims(_operands[O], o, "O is [B, Hq, Sq, Dv]"))
            {
                return verdict;
            }
            if(auto verdict = expectDims(_operands[DO], o, "dO is [B, Hq, Sq, Dv]"))
            {
                return verdict;
            }
            if(auto verdict = expectDims(_operands[DQ], q, "dQ has Q's dims"))
            {
                return verdict;
            }
            if(auto verdict = expectDims(_operands[DK], k, "dK has K's dims"))
            {
                return verdict;
            }
            if(auto verdict = expectDims(_operands[DV], v, "dV has V's dims"))
            {
                return verdict;
            }
            return expectDims(_stats, stats, "stats is [B, Hq, Sq, 1]");
        };
        if(auto verdict = checkAll())
        {
            return verdict;
        }

        _problem.batch = batch;
        _problem.numHeads = heads;
        _problem.numKvHeads = kvHeads;
        _problem.numVHeads = valueHeads;
        _problem.seqLenQ = seqQ;
        _problem.seqLenKv = seqKv;
        _problem.headDimQk = dimQk;
        _problem.headDimV = dimV;
        _headDims = {dimQk, dimQk, dimV, dimV, dimV, dimQk, dimQk, dimV};
        return std::nullopt;
    }

    /// DT-1: one dtype, f16 or bf16, on all eight tensors. DT-2: FLOAT is served only when the
    /// engine opts in (MatchOptions::acceptFloat), under the same uniformity; otherwise, like
    /// DT-3 (FP8), it declines [data_type] through the same check.
    Verdict checkDataTypes()
    {
        const auto dtype = _operands[Q].tensor->data_type();
        const uint32_t elementBytes = servedElementBytes(dtype, _options.acceptFloat);
        if(elementBytes == 0)
        {
            return Decline{"data_type",
                           describe(_operands[Q]) + " is " + typeName(dtype) + ", not "
                               + (_options.acceptFloat ? "f16/bf16/FLOAT" : "f16/bf16")};
        }
        for(const auto& operand : _operands)
        {
            if(operand.tensor->data_type() != dtype)
            {
                return Decline{"data_type",
                               describe(operand) + " is " + typeName(operand.tensor->data_type())
                                   + ", but Q is " + typeName(dtype)};
            }
        }
        if(_stats.tensor->data_type() != data_objects::DataType::FLOAT)
        {
            return Decline{"layout",
                           describe(_stats) + " is " + typeName(_stats.tensor->data_type())
                               + ", not FLOAT"};
        }
        _problem.dtype = dtype;
        _problem.elementBytes = elementBytes;
        _problem.biasElementBytes = elementBytes;
        return std::nullopt;
    }

    /// DT-4: max(Dqk, Dv) above MAX_HEAD_DIM declines [head_size]. DT-5: unequal Dqk and Dv are
    /// accepted; tile routing is the engine's. DT-6: any group size is accepted, a head count
    /// that is not a multiple of the key/value heads declines [gqa]. SH-1: V heads that differ
    /// from K's (only under MatchOptions::acceptValueHeads) need Hq to be a multiple of Hv as
    /// well, [gqa] otherwise, and decline [shape] under FLOAT; checkRagged and checkBias hold
    /// the ragged and bias halves of the rule.
    // Every step shares one signature for the table in run(); this one reads only.
    // NOLINTNEXTLINE(readability-make-member-function-const)
    Verdict checkHeads()
    {
        if(std::max(_problem.headDimQk, _problem.headDimV) > MAX_HEAD_DIM)
        {
            return Decline{"head_size",
                           "head dims Qk " + std::to_string(_problem.headDimQk) + ", V "
                               + std::to_string(_problem.headDimV) + " exceed "
                               + std::to_string(MAX_HEAD_DIM)};
        }
        if(_problem.numHeads % _problem.numKvHeads != 0)
        {
            return Decline{"gqa",
                           "query heads " + std::to_string(_problem.numHeads)
                               + " are not a multiple of key/value heads "
                               + std::to_string(_problem.numKvHeads)};
        }
        if(_problem.numHeads % _problem.numVHeads != 0)
        {
            return Decline{"gqa",
                           "query heads " + std::to_string(_problem.numHeads)
                               + " are not a multiple of value heads "
                               + std::to_string(_problem.numVHeads)};
        }
        if(_problem.numVHeads != _problem.numKvHeads
           && _problem.dtype == data_objects::DataType::FLOAT)
        {
            return Decline{"shape",
                           "value heads " + std::to_string(_problem.numVHeads)
                               + " differ from key heads " + std::to_string(_problem.numKvHeads)
                               + " under FLOAT"};
        }
        return std::nullopt;
    }

    /// LY-1: every operand declares the 16-byte alignment the kernels assume.
    Verdict checkAlignment()
    {
        for(const auto& operand : _operands)
        {
            if(!declaresAlignment(*operand.tensor, REQUIRED_ALIGNMENT_BYTES))
            {
                return Decline{"operand",
                               describe(operand) + " declares alignment "
                                   + std::to_string(operand.tensor->alignment())
                                   + " bytes, below the " + std::to_string(REQUIRED_ALIGNMENT_BYTES)
                                   + " the kernels assume"};
            }
        }
        return std::nullopt;
    }

    /// The strides of the (B, H, S) axes of @p operand, 0 on a unit axis.
    static Strides recordStrides(const Operand& operand, bool ragged = false)
    {
        const auto& dims = dimsOf(operand);
        const auto& strides = stridesOf(operand);
        const auto stride
            = [&](uint32_t axis) { return dims.Get(axis) > 1 ? strides.Get(axis) : int64_t{0}; };
        return Strides{
            ragged ? int64_t{0} : stride(BATCH_AXIS), stride(HEAD_AXIS), stride(SEQ_AXIS)};
    }

    /// DT-9: whether a per-(batch, head) slab of @p rows + 1 rows at @p rowStride elements, the
    /// last @p width elements wide, of elements @p elementBytes wide, fits the kernels' 32-bit
    /// byte offsets. Callers pass the rows the kernels address (addressedQueryRows,
    /// addressedKeyRows), padding included. DT-7: this is the only bound on sequence length
    /// and tensor size; a tensor larger than 4 GiB in total is served when every slab fits.
    static bool slabFits(int64_t rows, int64_t rowStride, int64_t width, uint32_t elementBytes)
    {
        const auto reach = checkedProduct({rows, rowStride});
        if(!reach.has_value() || *reach > INT32_LIMIT)
        {
            return false;
        }
        return (*reach + width) * static_cast<int64_t>(elementBytes) <= INT32_LIMIT;
    }

    /// LY-2: the last stride is 1, the others are independent. LY-3: every stride of an axis of
    /// more than one element is a non-negative multiple of 8 elements (4 for FLOAT). LY-6: a sequence axis of
    /// more than one row has a non-zero stride on every operand (padding rows read zero through
    /// the row bound, and an aliased row would feed them data). DT-8: a head dim that is not a
    /// multiple of 8 (4 for FLOAT) needs a smallest stride of at least ceil8(D) (ceil4(D)).
    /// DT-9: the per-head slab,
    /// padding rows included, fits. LY-5: output elements do not overlap.
    Verdict checkTensorLayout(
        const Operand& operand, int64_t headDim, bool isOutput, bool ragged, bool queryAxis)
    {
        const auto& dims = dimsOf(operand);
        const auto& strides = stridesOf(operand);
        if(strides.Get(DIM_AXIS) != 1)
        {
            return Decline{"layout",
                           describe(operand) + " last stride is "
                               + std::to_string(strides.Get(DIM_AXIS)) + ", not 1"};
        }
        int64_t smallest = std::numeric_limits<int64_t>::max();
        for(uint32_t axis = BATCH_AXIS; axis <= SEQ_AXIS; ++axis)
        {
            if(dims.Get(axis) == 1)
            {
                continue;
            }
            // A ragged tensor's batches sit at offset * multiplier; its batch stride is never read.
            if(ragged && axis == BATCH_AXIS)
            {
                continue;
            }
            const int64_t stride = strides.Get(axis);
            if(stride < 0 || stride % vectorElements() != 0)
            {
                return Decline{"layout",
                               describe(operand) + " stride " + std::to_string(stride) + " on axis "
                                   + std::to_string(axis) + " (extent "
                                   + std::to_string(dims.Get(axis))
                                   + ") is not a non-negative multiple of "
                                   + std::to_string(vectorElements()) + " elements"};
            }
            if(axis == SEQ_AXIS && stride == 0)
            {
                return Decline{"layout",
                               describe(operand) + " sequence axis (extent "
                                   + std::to_string(dims.Get(axis)) + ") has stride 0; rows of a "
                                   + "multi-row sequence axis must not alias"};
            }
            if(stride > 0)
            {
                smallest = std::min(smallest, stride);
            }
        }
        const int64_t padded = ceilToVector(headDim, vectorElements());
        if(headDim % vectorElements() != 0
           && (smallest == std::numeric_limits<int64_t>::max() || smallest < padded))
        {
            return Decline{"layout",
                           describe(operand) + " head dim " + std::to_string(headDim)
                               + " is not a multiple of " + std::to_string(vectorElements())
                               + " and its smallest row stride "
                               + (smallest == std::numeric_limits<int64_t>::max()
                                      ? std::string("does not exist")
                                      : std::to_string(smallest))
                               + " is below the padded width " + std::to_string(padded)};
        }
        const int64_t addressed = queryAxis ? addressedQueryRows(dims.Get(SEQ_AXIS))
                                            : addressedKeyRows(dims.Get(SEQ_AXIS));
        if(!slabFits(addressed - 1, strides.Get(SEQ_AXIS), padded, _problem.elementBytes))
        {
            return Decline{"layout",
                           describe(operand) + " per-head slab of " + std::to_string(addressed)
                               + " addressed rows (" + std::to_string(dims.Get(SEQ_AXIS))
                               + " plus the kernels' padding) at stride "
                               + std::to_string(strides.Get(SEQ_AXIS))
                               + " exceeds the 32-bit byte offset limit"};
        }
        if(isOutput && !elementsDoNotOverlap(*operand.tensor, ragged))
        {
            return Decline{"layout",
                           describe(operand) + " elements overlap (dims " + listOf(dims)
                               + ", strides " + listOf(strides) + ")"};
        }
        return std::nullopt;
    }

    std::array<RaggedOffset*, 8> raggedSlots()
    {
        return {&_problem.ragged.q,
                &_problem.ragged.k,
                &_problem.ragged.v,
                &_problem.ragged.o,
                &_problem.ragged.dO,
                &_problem.ragged.dQ,
                &_problem.ragged.dK,
                &_problem.ragged.dV};
    }

    /// @p owner's ragged offset, when it has one: a device INT32 [B + 1, 1, 1, 1] tensor with
    /// dense strides and a declared alignment of at least 4 bytes, and a multiplier of at least
    /// 1. Where the offsets point, their monotonicity and the extent they address are the
    /// caller's preconditions; the base of batch b is offset[b] * multiplier elements.
    Verdict checkOffsetTensor(const Operand& owner, RaggedOffset& out)
    {
        const auto uid = optionalLong(owner.tensor->ragged_offset_tensor_uid());
        if(!uid.has_value())
        {
            return std::nullopt;
        }
        Operand offset{"ragged_offset", *uid};
        if(auto verdict = resolveOperand(_context, offset, "varlen"))
        {
            return verdict;
        }
        const std::string of = " of " + describe(owner);
        if(offset.tensor->data_type() != data_objects::DataType::INT32)
        {
            return Decline{"varlen",
                           describe(offset) + of + " is " + typeName(offset.tensor->data_type())
                               + ", not INT32"};
        }
        const auto* dims = offset.tensor->dims();
        const auto* strides = offset.tensor->strides();
        const std::array<int64_t, 4> expected{_problem.batch + 1, 1, 1, 1};
        if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
           || strides->size() != SDPA_RANK
           || !std::equal(dims->begin(), dims->end(), expected.begin()))
        {
            return Decline{"varlen",
                           describe(offset) + of + " dims "
                               + (dims == nullptr ? std::string("[]") : listOf(*dims))
                               + " != " + listOf(expected)};
        }
        if(auto verdict = checkScalarMemory(offset, INT32_BYTES, "varlen"))
        {
            return verdict;
        }
        const int64_t multiplier = owner.tensor->ragged_offset_multiplier();
        if(multiplier < 1)
        {
            return Decline{"varlen",
                           describe(owner) + " ragged_offset_multiplier "
                               + std::to_string(multiplier) + " is below 1"};
        }
        out = RaggedOffset{*uid, multiplier};
        return std::nullopt;
    }

    /// PV-3: Q, K, V, O, dO, dQ, dK and dV may each be ragged, with an offset tensor and
    /// multiplier of its own (checkOffsetTensor); independently of one another, and beside
    /// dense tensors. PV-8: the stats may carry one too (checkStats pairs it with Q); a ragged
    /// bias or dBias declines (checkBias). The padding_mask rule is checkVarlen's. SH-1: V heads
    /// that differ from K's decline [shape] when any tensor is ragged.
    Verdict checkRagged()
    {
        const auto slots = raggedSlots();
        for(size_t index = 0; index < _operands.size(); ++index)
        {
            if(auto verdict = checkOffsetTensor(_operands[index], *slots[index]))
            {
                return verdict;
            }
        }
        if(auto verdict = checkOffsetTensor(_stats, _problem.ragged.stats))
        {
            return verdict;
        }
        if(_problem.numVHeads != _problem.numKvHeads && _problem.ragged.any())
        {
            return Decline{"shape",
                           "value heads " + std::to_string(_problem.numVHeads)
                               + " differ from key heads " + std::to_string(_problem.numKvHeads)
                               + " with a ragged tensor"};
        }
        return std::nullopt;
    }

    /// LY-4: dQ, dK and dV may share one buffer and Q, K and V may be packed views; aliasing
    /// between tensors is not visible here, only overlap within one output (LY-5).
    Verdict checkLayouts()
    {
        const std::array<Strides*, 8> slots{&_problem.q,
                                            &_problem.k,
                                            &_problem.v,
                                            &_problem.o,
                                            &_problem.dO,
                                            &_problem.dQ,
                                            &_problem.dK,
                                            &_problem.dV};
        const auto raggedOf = raggedSlots();
        for(size_t index = 0; index < _operands.size(); ++index)
        {
            const bool isOutput = index == DQ || index == DK || index == DV;
            const bool ragged = raggedOf[index]->ragged();
            const bool queryAxis = index == Q || index == O || index == DO || index == DQ;
            if(auto verdict
               = checkTensorLayout(_operands[index], _headDims[index], isOutput, ragged, queryAxis))
            {
                return verdict;
            }
            *slots[index] = recordStrides(_operands[index], ragged);
        }
        return std::nullopt;
    }

    /// SS-3: stats declares the 16-byte alignment the kernels assume for every data tensor
    /// (the forward's check at the f32 element size is weaker). SS-4: stats is ragged exactly
    /// when Q is. SS-1: dense stats, with a Q tensor that is not ragged, is served through its
    /// actual (B, H, S) strides: every non-unit axis has a non-negative stride, no two elements
    /// share an address (a zero stride on an axis of more than one element is such an overlap),
    /// and one head's rows, the addressed query rows (padding included) * s_token FLOAT elements,
    /// lie inside the 32-bit byte range the gfx950 per-head row descriptor addresses. Padded and non-canonical strides are
    /// served; unit axes are exempt. SS-5: ragged stats, with its own offset, is packed over
    /// tokens: HT has sequence stride 1 and a positive head stride (the total token count, a
    /// caller precondition), TH has head stride 1 and sequence stride H; its batch stride is
    /// never read. The kernels read the stats through its own strides, so any other ragged
    /// layout declines.
    Verdict checkStats()
    {
        if(!declaresAlignment(*_stats.tensor, REQUIRED_ALIGNMENT_BYTES))
        {
            return Decline{"operand",
                           describe(_stats) + " declares alignment "
                               + std::to_string(_stats.tensor->alignment()) + " bytes, below the "
                               + std::to_string(REQUIRED_ALIGNMENT_BYTES) + " the kernels assume"};
        }
        const bool statsRagged = _problem.ragged.stats.ragged();
        if(statsRagged != _problem.ragged.q.ragged())
        {
            return Decline{"varlen",
                           statsRagged ? describe(_stats) + " has a ragged offset but Q does not"
                                       : describe(_stats)
                                             + " is dense but Q is ragged: stats must be packed "
                                               "with its own ragged offset"};
        }
        const auto& strides = stridesOf(_stats);
        const int64_t heads = _problem.numHeads;
        const int64_t seq = _problem.seqLenQ;
        if(statsRagged)
        {
            const int64_t headStride = strides.Get(HEAD_AXIS);
            const int64_t seqStride = strides.Get(SEQ_AXIS);
            const bool packedHt = (seq == 1 || seqStride == 1) && (heads == 1 || headStride >= 1);
            const bool packedTh
                = (heads == 1 || headStride == 1) && (seq == 1 || seqStride == heads);
            if(!packedHt && !packedTh)
            {
                return Decline{"layout",
                               describe(_stats) + " strides " + listOf(strides)
                                   + " are neither packed (H, T) nor packed (T, H)"};
            }
            _problem.stats = recordStrides(_stats, true);
            return std::nullopt;
        }
        const auto& dims = dimsOf(_stats);
        for(const uint32_t axis : {BATCH_AXIS, HEAD_AXIS, SEQ_AXIS})
        {
            if(dims.Get(axis) > 1 && strides.Get(axis) < 0)
            {
                return Decline{"layout",
                               describe(_stats) + " strides " + listOf(strides)
                                   + " have a negative stride on a non-unit axis"};
            }
        }
        if(!elementsDoNotOverlap(*_stats.tensor))
        {
            return Decline{"layout",
                           describe(_stats) + " elements overlap (dims " + listOf(dims)
                               + ", strides " + listOf(strides) + ")"};
        }
        const Strides recorded = recordStrides(_stats);
        const auto rowSpan
            = checkedProduct({addressedQueryRows(seq), recorded.seq, STATS_BYTES_PER_ELEMENT});
        if(!rowSpan.has_value() || *rowSpan > INT32_LIMIT)
        {
            return Decline{"layout",
                           describe(_stats) + " strides " + listOf(strides)
                               + " put one head's rows beyond the 32-bit byte offset limit"};
        }
        _problem.stats = recordStrides(_stats);
        return std::nullopt;
    }

    static std::optional<int64_t> optionalLong(const flatbuffers::Optional<int64_t>& value)
    {
        return value;
    }

    /// MW-1: both deprecated causal flags decline. MW-2: a flag implies right bound 0 unless
    /// right_bound is set. MW-3: a bound below -1 declines. MW-4: only TOP_LEFT and BOTTOM_RIGHT
    /// alignments are served. MW-5, MW-6, MW-7, MW-8: windows are accepted in cuDNN's
    /// convention. MW-9: every left_bound of 0 declines, whatever the right bound.
    /// MW-11: dropout with a bottom-right causal mask is accepted. MW-12: ALiBi declines.
    /// MW-13: a window with bottom-right alignment under padding_mask declines. MW-10 (a mask
    /// with bias) is checkBias's.
    Verdict checkMask()
    {
        const auto& a = _attributes;
        if(a.alibi_mask())
        {
            return Decline{"alibi_mask", "alibi_mask is set"};
        }
        const bool topLeftFlag = a.causal_mask();
        const bool bottomRightFlag = a.causal_mask_bottom_right();
        if(topLeftFlag && bottomRightFlag)
        {
            return Decline{"mask", "both causal_mask and causal_mask_bottom_right set"};
        }
        MaskBounds bounds;
        bounds.left = optionalLong(a.left_bound()).value_or(NO_WINDOW);
        bounds.right = optionalLong(a.right_bound()).value_or(NO_WINDOW);
        if(bounds.left < NO_WINDOW)
        {
            return Decline{"mask", "left_bound " + std::to_string(bounds.left) + " is below -1"};
        }
        if(bounds.right < NO_WINDOW)
        {
            return Decline{"mask", "right_bound " + std::to_string(bounds.right) + " is below -1"};
        }
        if((topLeftFlag || bottomRightFlag) && bounds.right == NO_WINDOW)
        {
            bounds.right = 0;
        }
        const auto alignment = a.diagonal_alignment();
        if(alignment != data_objects::DiagonalAlignment::TOP_LEFT
           && alignment != data_objects::DiagonalAlignment::BOTTOM_RIGHT)
        {
            return Decline{"mask",
                           "diagonal_alignment " + std::to_string(static_cast<int>(alignment))
                               + " is neither TOP_LEFT nor BOTTOM_RIGHT"};
        }
        bounds.alignBottomRight
            = bottomRightFlag || alignment == data_objects::DiagonalAlignment::BOTTOM_RIGHT;
        if(bounds.left == 0)
        {
            return Decline{"mask",
                           "left_bound 0 is not a valid window; a left bound is at least 1"};
        }
        const bool windowed = bounds.left != NO_WINDOW || bounds.right > 0;
        if(a.padding_mask() && bounds.alignBottomRight && windowed)
        {
            return Decline{"mask",
                           "window (left_bound " + std::to_string(bounds.left) + ", right bound "
                               + std::to_string(bounds.right)
                               + ") with BOTTOM_RIGHT alignment under padding_mask"};
        }
        _problem.masked = bounds.left != NO_WINDOW || bounds.right != NO_WINDOW;
        _problem.alignBottomRight = bounds.alignBottomRight;
        _problem.diagonalOffset
            = bounds.alignBottomRight ? _problem.seqLenKv - _problem.seqLenQ : 0;
        _problem.windowLeft = bounds.left;
        _problem.windowRight = bounds.right;
        return std::nullopt;
    }

    /// A device integer tensor the kernels read one element of per batch (PV-11, DR-8): dense
    /// strides and a declared alignment of at least @p elementBytes, else a decline under
    /// @p cause.
    static Verdict
        checkScalarMemory(const Operand& operand, int64_t elementBytes, const char* cause)
    {
        if(!stridesAreDense(*operand.tensor))
        {
            return Decline{cause,
                           describe(operand) + " strides " + listOf(stridesOf(operand))
                               + " are not dense for dims " + listOf(dimsOf(operand))};
        }
        if(!declaresAlignment(*operand.tensor, elementBytes))
        {
            return Decline{cause,
                           describe(operand) + " declares alignment "
                               + std::to_string(operand.tensor->alignment())
                               + " bytes, below its element size " + std::to_string(elementBytes)};
        }
        return std::nullopt;
    }

    /// PV-4, PV-7: a seq_len tensor is a device INT32 [B, 1, 1, 1] tensor without a ragged
    /// offset. PV-11: it is dense and declares an alignment of at least its element size.
    Verdict checkSeqLen(const char* label, int64_t uid)
    {
        Operand operand{label, uid};
        if(auto verdict = resolveOperand(_context, operand, "varlen"))
        {
            return verdict;
        }
        if(operand.tensor->ragged_offset_tensor_uid().has_value())
        {
            return Decline{"varlen", describe(operand) + " has a ragged offset tensor"};
        }
        if(operand.tensor->data_type() != data_objects::DataType::INT32)
        {
            return Decline{"varlen",
                           describe(operand) + " is " + typeName(operand.tensor->data_type())
                               + ", not INT32"};
        }
        const auto* dims = operand.tensor->dims();
        const auto* strides = operand.tensor->strides();
        const std::array<int64_t, 4> expected{_problem.batch, 1, 1, 1};
        if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
           || strides->size() != SDPA_RANK
           || !std::equal(dims->begin(), dims->end(), expected.begin()))
        {
            return Decline{"varlen",
                           describe(operand) + " dims "
                               + (dims == nullptr ? std::string("[]") : listOf(*dims))
                               + " != " + listOf(expected)};
        }
        if(_problem.batch > 1 && strides->Get(BATCH_AXIS) != 1)
        {
            return Decline{"varlen",
                           describe(operand) + " batch stride "
                               + std::to_string(strides->Get(BATCH_AXIS)) + " is not 1"};
        }
        return checkScalarMemory(operand, INT32_BYTES, "varlen");
    }

    /// PV-1: padding_mask with both seq_len tensors is accepted. PV-2: the flag without
    /// tensors, tensors without the flag, or one length tensor decline. PV-3: a ragged tensor
    /// (checkRagged) needs padding_mask with both seq_len tensors: the lengths are the
    /// sequences' extents.
    Verdict checkVarlen()
    {
        const auto& a = _attributes;
        const auto lengthQ = optionalLong(a.seq_len_q_tensor_uid());
        const auto lengthKv = optionalLong(a.seq_len_kv_tensor_uid());
        if(lengthQ.has_value() != lengthKv.has_value())
        {
            return Decline{"varlen", "only one of seq_len_q and seq_len_kv is set"};
        }
        if(a.padding_mask() != lengthQ.has_value())
        {
            return Decline{"varlen",
                           a.padding_mask() ? "padding_mask is set without seq_len tensors"
                                            : "seq_len tensors are set without padding_mask"};
        }
        if(!a.padding_mask())
        {
            if(_problem.ragged.any())
            {
                return Decline{"varlen",
                               "a tensor has a ragged offset but padding_mask and the seq_len "
                               "tensors are not set"};
            }
            return std::nullopt;
        }
        if(auto verdict = checkSeqLen("seq_len_q", *lengthQ))
        {
            return verdict;
        }
        if(auto verdict = checkSeqLen("seq_len_kv", *lengthKv))
        {
            return verdict;
        }
        _problem.paddingMask = true;
        _problem.seqLenQUid = *lengthQ;
        _problem.seqLenKvUid = *lengthKv;
        return std::nullopt;
    }

    /// The bias, right-aligned to (B, Hq, Sq, Skv); the first three axes may broadcast.
    struct BiasView
    {
        std::array<int64_t, 4> dims{};
        std::array<int64_t, 4> strides{};
    };

    static Verdict checkSharedBiasLayout(const Operand& operand)
    {
        const auto& dims = dimsOf(operand);
        const auto& strides = stridesOf(operand);
        if(!declaresAlignment(*operand.tensor, REQUIRED_ALIGNMENT_BYTES))
        {
            return Decline{"operand",
                           describe(operand) + " declares alignment "
                               + std::to_string(operand.tensor->alignment()) + " bytes, below the "
                               + std::to_string(REQUIRED_ALIGNMENT_BYTES) + " the kernels assume"};
        }
        if(strides.Get(strides.size() - 1) != 1)
        {
            return Decline{"layout",
                           describe(operand) + " last stride is "
                               + std::to_string(strides.Get(strides.size() - 1)) + ", not 1"};
        }
        for(flatbuffers::uoffset_t axis = 0; axis < dims.size(); ++axis)
        {
            if(dims.Get(axis) > 1 && strides.Get(axis) < 0)
            {
                return Decline{"layout",
                               describe(operand) + " has a negative stride in " + listOf(strides)};
            }
        }
        return std::nullopt;
    }

    /// BI-6: dBias without bias declines. BI-1: a bias of rank 1..4 right-aligned to
    /// (B, Hq, Sq, Skv), whose first three axes may broadcast. BI-2: the bias dtype is Q's, or
    /// FLOAT beside a 16-bit Q when the engine opts in (MatchOptions::acceptF32Bias); every bias
    /// bound below counts the bias's own element width (SdpaBwdProblem::biasElementBytes).
    /// BI-3: a bias broadcast over batch or head with dBias is accepted. BI-4: dBias with a
    /// row-broadcast bias declines. BI-5, BI-7: dBias has the bias's dims and dtype. PV-8: a
    /// ragged offset on bias or dBias declines. PV-10: bias with padding_mask declines. MW-10:
    /// bias with a mask declines. LY-1, DT-9: bias and dBias alignment and slab bounds. BI-9:
    /// every non-unit stride of dBias other than the last is a multiple of 16 bytes (8 elements
    /// of a 16-bit dBias, 4 of a FLOAT one), as LY-3 asks of the other outputs. SH-1: a bias
    /// beside V heads that differ from K's declines [shape].
    Verdict checkBias()
    {
        const auto& a = _attributes;
        const auto biasUid = optionalLong(a.attn_mask_tensor_uid());
        const auto dbiasUid = optionalLong(a.dbias_tensor_uid());
        if(dbiasUid.has_value() && !biasUid.has_value())
        {
            return Decline{"dbias", "dBias (uid " + std::to_string(*dbiasUid) + ") without a bias"};
        }
        if(!biasUid.has_value())
        {
            return std::nullopt;
        }
        if(_problem.numVHeads != _problem.numKvHeads)
        {
            return Decline{"shape",
                           "value heads " + std::to_string(_problem.numVHeads)
                               + " differ from key heads " + std::to_string(_problem.numKvHeads)
                               + " with a bias"};
        }

        Operand bias{"bias", *biasUid};
        if(auto verdict = resolveOperand(_context, bias))
        {
            return verdict;
        }
        if(bias.tensor->ragged_offset_tensor_uid().has_value())
        {
            return Decline{"varlen", describe(bias) + " has a ragged offset tensor"};
        }
        if(bias.tensor->dims() == nullptr || bias.tensor->strides() == nullptr
           || bias.tensor->dims()->size() != bias.tensor->strides()->size())
        {
            return Decline{"attn_mask", describe(bias) + " has no dims or mismatched strides"};
        }
        const auto biasType = bias.tensor->data_type();
        const bool floatBesideHalf = _options.acceptF32Bias
                                     && biasType == data_objects::DataType::FLOAT
                                     && _problem.elementBytes == 2;
        if(biasType != _problem.dtype && !floatBesideHalf)
        {
            return Decline{"attn_mask",
                           describe(bias) + " is " + typeName(biasType) + ", but Q is "
                               + typeName(_problem.dtype)
                               + (_options.acceptF32Bias
                                          || biasType != data_objects::DataType::FLOAT
                                          || _problem.elementBytes != 2
                                      ? std::string()
                                      : std::string("; this engine does not accept an f32 bias "
                                                    "beside a 16-bit Q"))};
        }
        _problem.biasElementBytes = servedElementBytes(biasType, true);
        const auto& biasDims = dimsOf(bias);
        const auto& biasStrides = stridesOf(bias);
        const std::array<int64_t, 4> target{
            _problem.batch, _problem.numHeads, _problem.seqLenQ, _problem.seqLenKv};
        BiasView view;
        if(biasDims.empty() || biasDims.size() > SDPA_RANK)
        {
            return Decline{"attn_mask",
                           describe(bias) + " rank " + std::to_string(biasDims.size())
                               + " is not in 1..4"};
        }
        const uint32_t lead = SDPA_RANK - biasDims.size();
        for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
        {
            view.dims[axis] = axis < lead ? 1 : biasDims.Get(axis - lead);
            view.strides[axis] = axis < lead ? 0 : biasStrides.Get(axis - lead);
            const bool mayBroadcast = axis != DIM_AXIS;
            if(view.dims[axis] != target[axis] && (!mayBroadcast || view.dims[axis] != 1))
            {
                return Decline{"attn_mask",
                               describe(bias) + " dims " + listOf(biasDims)
                                   + " do not broadcast to " + listOf(target)};
            }
        }
        if(auto verdict = checkSharedBiasLayout(bias))
        {
            return verdict;
        }
        const auto biasSlabRows
            = view.dims[SEQ_AXIS] > 1 ? addressedQueryRows(_problem.seqLenQ) - 1 : int64_t{0};
        const auto biasSlabWidth = addressedKeyRows(_problem.seqLenKv);
        if(!slabFits(
               biasSlabRows, view.strides[SEQ_AXIS], biasSlabWidth, _problem.biasElementBytes))
        {
            return Decline{"layout",
                           describe(bias) + " per-head slab at row stride "
                               + std::to_string(view.strides[SEQ_AXIS])
                               + " exceeds the 32-bit byte offset limit"};
        }
        if(a.padding_mask())
        {
            return Decline{"varlen", describe(bias) + " with padding_mask"};
        }
        if(_problem.masked)
        {
            return Decline{"mask", "a causal or window mask with " + describe(bias)};
        }
        const auto stridesOfView
            = [&](const std::array<int64_t, 4>& dims, const std::array<int64_t, 4>& strides) {
                  const auto stride
                      = [&](uint32_t axis) { return dims[axis] > 1 ? strides[axis] : int64_t{0}; };
                  return Strides{stride(BATCH_AXIS), stride(HEAD_AXIS), stride(SEQ_AXIS)};
              };
        _problem.biasUid = *biasUid;
        _problem.biasStrides = stridesOfView(view.dims, view.strides);
        _problem.biasAlignmentBytes = bias.tensor->alignment();
        _problem.biasBroadcastBatch = view.dims[BATCH_AXIS] == 1 && _problem.batch > 1;
        _problem.biasBroadcastHead = view.dims[HEAD_AXIS] == 1 && _problem.numHeads > 1;

        if(!dbiasUid.has_value())
        {
            return std::nullopt;
        }
        Operand dbias{"dBias", *dbiasUid};
        if(auto verdict = resolveOperand(_context, dbias))
        {
            return verdict;
        }
        if(dbias.tensor->ragged_offset_tensor_uid().has_value())
        {
            return Decline{"varlen", describe(dbias) + " has a ragged offset tensor"};
        }
        if(dbias.tensor->dims() == nullptr || dbias.tensor->strides() == nullptr
           || dbias.tensor->dims()->size() != dbias.tensor->strides()->size())
        {
            return Decline{"dbias", describe(dbias) + " has no dims or mismatched strides"};
        }
        if(dbias.tensor->data_type() != bias.tensor->data_type())
        {
            return Decline{"dbias",
                           describe(dbias) + " is " + typeName(dbias.tensor->data_type())
                               + ", but bias is " + typeName(bias.tensor->data_type())};
        }
        const auto& dbiasDims = dimsOf(dbias);
        if(!std::equal(dbiasDims.begin(), dbiasDims.end(), biasDims.begin(), biasDims.end()))
        {
            return Decline{"dbias",
                           "dBias dims " + listOf(dbiasDims) + " != bias dims " + listOf(biasDims)};
        }
        if(view.dims[SEQ_AXIS] == 1 && _problem.seqLenQ > 1)
        {
            return Decline{"dbias", "dBias with a row-broadcast bias of dims " + listOf(biasDims)};
        }
        if(auto verdict = checkSharedBiasLayout(dbias))
        {
            return verdict;
        }
        const auto& dbiasStrides = stridesOf(dbias);
        for(flatbuffers::uoffset_t axis = 0; axis + 1 < dbiasDims.size(); ++axis)
        {
            const int64_t stride = dbiasStrides.Get(axis);
            if(dbiasDims.Get(axis) > 1 && stride % biasVectorElements() != 0)
            {
                return Decline{"layout",
                               describe(dbias) + " stride " + std::to_string(stride) + " on axis "
                                   + std::to_string(axis) + " (extent "
                                   + std::to_string(dbiasDims.Get(axis))
                                   + ") is not a non-negative multiple of "
                                   + std::to_string(biasVectorElements()) + " elements"};
            }
        }
        BiasView dbiasView;
        for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
        {
            dbiasView.dims[axis] = view.dims[axis];
            dbiasView.strides[axis] = axis < lead ? 0 : dbiasStrides.Get(axis - lead);
        }
        if(!slabFits(
               biasSlabRows, dbiasView.strides[SEQ_AXIS], biasSlabWidth, _problem.biasElementBytes))
        {
            return Decline{"layout",
                           describe(dbias) + " per-head slab at row stride "
                               + std::to_string(dbiasView.strides[SEQ_AXIS])
                               + " exceeds the 32-bit byte offset limit"};
        }
        if(!elementsDoNotOverlap(*dbias.tensor))
        {
            return Decline{"layout",
                           describe(dbias) + " elements overlap (dims " + listOf(dbiasDims)
                               + ", strides " + listOf(dbiasStrides) + ")"};
        }
        _problem.dbiasUid = *dbiasUid;
        _problem.dbiasStrides = stridesOfView(dbiasView.dims, dbiasView.strides);
        return std::nullopt;
    }

    /// DR-4: a seed and offset tensor is INT64 or INT32 and the width is recorded, except that a
    /// device INT32 tensor declines, as the forward engine's matcher declines it: the kernels
    /// read 8 bytes at a device seed or offset. A pass-by-value INT32 is accepted and
    /// sign-extended to 64 bits on the host. DR-6: a device tensor or a pass-by-value one is
    /// accepted, of any rank whose every extent is 1. DR-8: a device tensor is dense and
    /// declares an alignment of at least its element size; a pass-by-value one has no memory.
    Verdict checkSeedTensor(Operand& operand, SeedWidth& width, SeedForm& form)
    {
        if(auto verdict = resolveOperand(_context, operand, "dropout", true))
        {
            return verdict;
        }
        if(operand.tensor->ragged_offset_tensor_uid().has_value())
        {
            return Decline{"varlen", describe(operand) + " has a ragged offset tensor"};
        }
        const auto* dims = operand.tensor->dims();
        // A scalar of any rank: every extent is 1 (the frontend's constant form is [1]).
        if(dims == nullptr || dims->empty()
           || std::any_of(dims->begin(), dims->end(), [](int64_t extent) { return extent != 1; }))
        {
            return Decline{"dropout",
                           describe(operand) + " dims "
                               + (dims == nullptr ? std::string("[]") : listOf(*dims))
                               + " are not a scalar (every extent 1)"};
        }
        switch(operand.tensor->data_type())
        {
        case data_objects::DataType::INT32:
            width = SeedWidth::INT32;
            break;
        case data_objects::DataType::INT64:
            width = SeedWidth::INT64;
            break;
        default:
            return Decline{"dropout",
                           describe(operand) + " is " + typeName(operand.tensor->data_type())
                               + ", not INT32 or INT64"};
        }
        form = hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(operand.tensor)
                   ? SeedForm::PASS_BY_VALUE
                   : SeedForm::DEVICE;
        if(form == SeedForm::DEVICE)
        {
            if(width == SeedWidth::INT32)
            {
                return Decline{"dropout",
                               describe(operand)
                                   + " is a device INT32 tensor; the kernels read 8 bytes there, "
                                     "so a device seed or offset must be INT64"};
            }
            return checkScalarMemory(operand, INT64_BYTES, "dropout");
        }
        return std::nullopt;
    }

    /// DR-1: probability 0 or unset is off, (0, 1) is on. DR-2: p outside [0, 1) or NaN
    /// declines. DR-3: p in (0, 1) needs both seed and offset; a seed or offset with no
    /// probability declines. DR-6: seed and offset share one width and one form. DR-7:
    /// dropout_mask, dropout_scale and dropout_scale_inv tensors decline. DR-9: probability 0
    /// is no dropout whether or not a seed or offset is named; a named one is gated as at
    /// p > 0 (DR-4, DR-6 per tensor, DR-8) and not read, and the two need not agree.
    Verdict checkDropout()
    {
        const auto& a = _attributes;
        if(a.dropout_mask_tensor_uid().has_value())
        {
            return Decline{"dropout", "dropout_mask_tensor_uid is set"};
        }
        if(a.dropout_scale_tensor_uid().has_value())
        {
            return Decline{"dropout", "dropout_scale_tensor_uid is set"};
        }
        if(a.dropout_scale_inv_tensor_uid().has_value())
        {
            return Decline{"dropout", "dropout_scale_inv_tensor_uid is set"};
        }
        const float p
            = a.dropout_probability().has_value() ? a.dropout_probability().value() : 0.0F;
        // Written so a NaN fails.
        if(std::isnan(p) || p < 0.0F || p >= 1.0F)
        {
            return Decline{"dropout",
                           "dropout_probability " + std::to_string(p) + " is outside [0, 1)"};
        }
        const auto seedUid = optionalLong(a.seed_tensor_uid());
        const auto offsetUid = optionalLong(a.offset_tensor_uid());
        if(p == 0.0F)
        {
            // A seed or offset without any probability is half-stated.
            if(!a.dropout_probability().has_value()
               && (seedUid.has_value() || offsetUid.has_value()))
            {
                return Decline{"dropout",
                               "seed or offset tensor set without dropout_probability > 0"};
            }
            // Probability 0 drops nothing, so the plain objects run and nothing is read from a
            // seed or offset; a named one is still gated as the kernels would read it.
            for(auto [name, uid] : {std::pair<const char*, std::optional<int64_t>>{"seed", seedUid},
                                    {"offset", offsetUid}})
            {
                if(!uid.has_value())
                {
                    continue;
                }
                Operand operand{name, *uid};
                SeedWidth width = SeedWidth::NONE;
                SeedForm form = SeedForm::NONE;
                if(auto verdict = checkSeedTensor(operand, width, form))
                {
                    return verdict;
                }
            }
            return std::nullopt;
        }
        if(!seedUid.has_value() || !offsetUid.has_value())
        {
            return Decline{"dropout",
                           "dropout_probability " + std::to_string(p)
                               + " needs both seed and offset tensors"};
        }
        Operand seed{"seed", *seedUid};
        Operand offset{"offset", *offsetUid};
        SeedWidth seedWidth = SeedWidth::NONE;
        SeedWidth offsetWidth = SeedWidth::NONE;
        SeedForm seedForm = SeedForm::NONE;
        SeedForm offsetForm = SeedForm::NONE;
        if(auto verdict = checkSeedTensor(seed, seedWidth, seedForm))
        {
            return verdict;
        }
        if(auto verdict = checkSeedTensor(offset, offsetWidth, offsetForm))
        {
            return verdict;
        }
        if(seedWidth != offsetWidth)
        {
            return Decline{"dropout",
                           describe(seed) + " is " + typeName(seed.tensor->data_type()) + " but "
                               + describe(offset) + " is " + typeName(offset.tensor->data_type())};
        }
        if(seedForm != offsetForm)
        {
            return Decline{
                "dropout",
                "seed and offset differ: one is a device tensor, the other pass-by-value"};
        }
        _problem.dropoutP = p;
        _problem.seedWidth = seedWidth;
        _problem.seedForm = seedForm;
        _problem.seedUid = *seedUid;
        _problem.offsetUid = *offsetUid;
        return std::nullopt;
    }

    static bool isServableScaleValue(double value)
    {
        return std::isfinite(value) && std::fabs(value) <= MAX_SOFTMAX_SCALE;
    }

    Verdict checkScaleTensor(int64_t uid)
    {
        Operand scale{"scale", uid};
        scale.tensor = findTensor(_context, uid);
        if(scale.tensor == nullptr)
        {
            return Decline{"scale", describe(scale) + " is not in the graph"};
        }
        if(!hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scale.tensor))
        {
            return Decline{"scale", describe(scale) + " is a device tensor, not pass-by-value"};
        }
        if(scale.tensor->ragged_offset_tensor_uid().has_value())
        {
            return Decline{"scale", describe(scale) + " has a ragged offset tensor"};
        }
        const auto* dims = scale.tensor->dims();
        if(dims == nullptr
           || std::any_of(dims->begin(), dims->end(), [](int64_t extent) { return extent != 1; }))
        {
            return Decline{"scale",
                           describe(scale) + " dims "
                               + (dims == nullptr ? std::string("[]") : listOf(*dims))
                               + " are not all 1"};
        }
        const auto dataType = scale.tensor->data_type();
        if(dataType != data_objects::DataType::FLOAT && dataType != data_objects::DataType::HALF
           && dataType != data_objects::DataType::BFLOAT16
           && dataType != data_objects::DataType::DOUBLE)
        {
            return Decline{"scale",
                           describe(scale) + " is " + typeName(dataType)
                               + ", not a floating-point type"};
        }
        try
        {
            const auto operand
                = hipdnn_plugin_sdk::makeScalarOperand(_context.graph.getTensorMap(), uid, "scale");
            if(!operand.isRuntimeUserSupplied)
            {
                const double value = hipdnn_plugin_sdk::toDouble(operand.bakedDefault);
                if(!isServableScaleValue(value))
                {
                    return Decline{"scale",
                                   describe(scale) + " value " + std::to_string(value)
                                       + " is non-finite or exceeds 2^20 in magnitude"};
                }
                _problem.scaleValue = static_cast<float>(value);
            }
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return Decline{"scale", describe(scale) + " value is unreadable"};
        }
        _problem.scaleSource = ScaleSource::TENSOR;
        _problem.scaleUid = uid;
        return std::nullopt;
    }

    /// SC-1: attn_scale_value or a pass-by-value scale tensor, 1.0 when absent. SC-2: a device
    /// scale tensor declines. SC-3: both set declines. SC-4: any finite scale up to the bound,
    /// zero and negative included. SC-5: non-finite or above 2^20 in magnitude declines.
    Verdict checkScale()
    {
        const auto& a = _attributes;
        const auto scaleUid = optionalLong(a.scale_tensor_uid());
        if(a.attn_scale_value().has_value() && scaleUid.has_value())
        {
            return Decline{"scale", "both attn_scale_value and scale_tensor_uid set"};
        }
        if(scaleUid.has_value())
        {
            return checkScaleTensor(*scaleUid);
        }
        if(a.attn_scale_value().has_value())
        {
            const float value = a.attn_scale_value().value();
            if(!isServableScaleValue(static_cast<double>(value)))
            {
                return Decline{"scale",
                               "attn_scale_value " + std::to_string(value)
                                   + " is non-finite or exceeds 2^20 in magnitude"};
            }
            _problem.scaleValue = value;
        }
        return std::nullopt;
    }
};

} // namespace detail

/**
 * @brief Matches a single SdpaBackwardAttributes graph, returning the problem every engine
 *        shares or the reason the graph is declined. @p options are what the calling engine
 *        opts into; the default is the 16-bit match.
 *
 * Decline causes: node, operand, data_type, head_size, gqa, shape, layout, mask, alibi_mask,
 * varlen, attn_mask, dbias, dropout, scale. The detail names the field and its value.
 */
inline std::variant<SdpaBwdProblem, Decline>
    matchSdpaBackward(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      const MatchOptions& options = {})
{
    // ST-1: a graph that is not exactly one SdpaBackwardAttributes node declines [node].
    if(context.graph.nodeCount() != 1)
    {
        return Decline{"node",
                       "graph has " + std::to_string(context.graph.nodeCount())
                           + " nodes, expected 1"};
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaBackwardAttributes)
    {
        return Decline{"node", "node is not an SDPA backward node"};
    }
    // ST-2: an override-shape graph declines [node].
    // Launch arguments are bound from the graph's shapes at prepare, so an override-shape
    // graph, whose shapes are replaced at execute, is declined.
    if(context.graph.getGraph().is_override_shape_enabled())
    {
        return Decline{"node", "graph has override shapes enabled"};
    }
    // ST-3: the compute type is one of UNSET, FLOAT, HALF, BF16.
    // The kernels accumulate in f32 whatever the compute type; DOUBLE asks for more and an
    // integer or FP8 compute type for a different computation.
    const auto computeType = node.computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT
       && computeType != data_objects::DataType::HALF
       && computeType != data_objects::DataType::BFLOAT16)
    {
        return Decline{"node", "node compute type is " + detail::typeName(computeType)};
    }
    detail::BackwardMatcher matcher(
        context, node.attributesAs<data_objects::SdpaBackwardAttributes>(), options);
    return matcher.run();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_bwd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
