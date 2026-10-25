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
 * the helpers this match needs are restated here rather than included. A match makes no
 * runtime calls and keeps no state; it reads only the graph.
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

/// How the f32 statistics are packed: (B*H, Sq) or (B, Sq, H).
enum class StatsLayout : uint8_t
{
    HT,
    TH
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

/// What an engine opts into when it asks for the match.
struct MatchOptions
{
    /// Accept FLOAT (fp32) data tensors, bias and dBias. Off, FLOAT declines [data_type] (DT-2).
    bool acceptFloat = false;
};

/// What the graph match computes, once, for every engine that serves the graph.
struct SdpaBwdProblem
{
    /// f16 or bf16 on every tensor; FLOAT when the engine accepts it (MatchOptions).
    data_objects::DataType dtype = data_objects::DataType::UNSET;
    /// Width of one element of dtype: 2 for f16 and bf16, 4 for FLOAT.
    uint32_t elementBytes = 0;
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
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

    StatsLayout statsLayout = StatsLayout::HT;
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

inline constexpr int64_t INT32_LIMIT = 2147483647LL;
inline constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and f16 only
inline constexpr int64_t STATS_BYTES_PER_ELEMENT = 4; // stats is FLOAT
inline constexpr int64_t INT32_BYTES = 4;
inline constexpr int64_t INT64_BYTES = 8;
inline constexpr int64_t REQUIRED_ALIGNMENT_BYTES = 16;
inline constexpr int64_t MAX_HEAD_DIM = 512;
inline constexpr int64_t VECTOR_ELEMENTS = 8;

/// The largest softmax scale magnitude the kernels serve (the forward's bound).
inline constexpr double MAX_SOFTMAX_SCALE = 1048576.0; // 2^20

inline const Tensor* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// Whether @p tensor promises a base pointer aligned to @p bytes.
inline bool declaresAlignment(const Tensor& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// The product of @p factors, or nullopt when it does not fit in int64_t.
inline std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for(const int64_t factor : factors)
    {
        if(__builtin_mul_overflow(product, factor, &product))
        {
            return std::nullopt;
        }
    }
    return product;
}

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

/// Whether no two elements of @p tensor share an address: its axes of more than one element,
/// ordered by stride, must each step past everything the axes inside it span. Conservative:
/// a layout interleaving two axes inside one another's gaps declines too.
inline bool elementsDoNotOverlap(const Tensor& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size())
    {
        return false;
    }
    std::vector<std::pair<int64_t, int64_t>> axes; // (stride, extent)
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) > 1)
        {
            axes.emplace_back(strides->Get(axis), dims->Get(axis));
        }
    }
    std::sort(axes.begin(), axes.end());
    int64_t span = 1; // elements the inner axes reach, from the first
    for(const auto& [stride, extent] : axes)
    {
        if(stride < span)
        {
            return false;
        }
        const auto reach = checkedProduct({extent - 1, stride});
        if(!reach.has_value() || *reach > std::numeric_limits<int64_t>::max() - span)
        {
            return false;
        }
        span += *reach;
    }
    return true;
}

/// @p value rounded up to a multiple of @p vector elements: the 16-bit vector of 8 by default,
/// 4 for FLOAT.
inline int64_t ceilToVector(int64_t value, int64_t vector = VECTOR_ELEMENTS)
{
    return (value + vector - 1) / vector * vector;
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
        const std::array<Verdict (BackwardMatcher::*)(), 13> steps{
            &BackwardMatcher::resolveOperands,
            &BackwardMatcher::checkUidAliasing,
            &BackwardMatcher::checkShapes,
            &BackwardMatcher::checkDataTypes,
            &BackwardMatcher::checkHeads,
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

    /// SH-1: dQ, dK and dV match Q, K and V; O and dO are [B, Hq, Sq, Dv]; V's heads equal
    /// K's; stats is [B, Hq, Sq, 1]; every extent is positive.
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

        const std::array<int64_t, 4> q{batch, heads, seqQ, dimQk};
        const std::array<int64_t, 4> k{batch, kvHeads, seqKv, dimQk};
        const std::array<int64_t, 4> v{batch, kvHeads, seqKv, dimV};
        const std::array<int64_t, 4> o{batch, heads, seqQ, dimV};
        const std::array<int64_t, 4> stats{batch, heads, seqQ, 1};
        const auto checkAll = [&]() -> Verdict {
            if(auto verdict = expectDims(_operands[K], k, "batch and head dim follow Q"))
            {
                return verdict;
            }
            if(auto verdict
               = expectDims(_operands[V], v, "batch follows Q; heads and sequence follow K"))
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
        return std::nullopt;
    }

    /// DT-4: max(Dqk, Dv) above MAX_HEAD_DIM declines [head_size]. DT-5: unequal Dqk and Dv are
    /// accepted; tile routing is the engine's. DT-6: any group size is accepted, a head count
    /// that is not a multiple of the key/value heads declines [gqa].
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
        // V's head count equals K's (checkShapes), so one remainder covers both.
        if(_problem.numHeads % _problem.numKvHeads != 0)
        {
            return Decline{"gqa",
                           "query heads " + std::to_string(_problem.numHeads)
                               + " are not a multiple of key/value heads "
                               + std::to_string(_problem.numKvHeads)};
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
    static Strides recordStrides(const Operand& operand)
    {
        const auto& dims = dimsOf(operand);
        const auto& strides = stridesOf(operand);
        const auto stride
            = [&](uint32_t axis) { return dims.Get(axis) > 1 ? strides.Get(axis) : int64_t{0}; };
        return Strides{stride(BATCH_AXIS), stride(HEAD_AXIS), stride(SEQ_AXIS)};
    }

    /// DT-9: whether a per-(batch, head) slab of @p rows + 1 rows at @p rowStride elements, the
    /// last @p width elements wide, fits the kernels' 32-bit byte offsets at the problem's
    /// element width. DT-7: this is the
    /// only bound on sequence length and tensor size; a tensor larger than 4 GiB in total is
    /// served when every slab fits.
    bool slabFits(int64_t rows, int64_t rowStride, int64_t width) const
    {
        const auto reach = checkedProduct({rows, rowStride});
        if(!reach.has_value() || *reach > INT32_LIMIT)
        {
            return false;
        }
        return (*reach + width) * static_cast<int64_t>(_problem.elementBytes) <= INT32_LIMIT;
    }

    /// LY-2: the last stride is 1, the others are independent. LY-3: every stride of an axis of
    /// more than one element is a non-negative multiple of 8 elements (4 for FLOAT). LY-6: a sequence axis of
    /// more than one row has a non-zero stride on every operand (padding rows read zero through
    /// the row bound, and an aliased row would feed them data). DT-8: a head dim that is not a
    /// multiple of 8 (4 for FLOAT) needs a smallest stride of at least ceil8(D) (ceil4(D)).
    /// DT-9: the per-head slab
    /// fits. LY-5: output elements do not overlap.
    Verdict checkTensorLayout(const Operand& operand, int64_t headDim, bool isOutput)
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
        const int64_t rows = dims.Get(SEQ_AXIS) - 1;
        if(!slabFits(rows, strides.Get(SEQ_AXIS), padded))
        {
            return Decline{"layout",
                           describe(operand) + " per-head slab of " + std::to_string(rows + 1)
                               + " rows at stride " + std::to_string(strides.Get(SEQ_AXIS))
                               + " exceeds the 32-bit byte offset limit"};
        }
        if(isOutput && !elementsDoNotOverlap(*operand.tensor))
        {
            return Decline{"layout",
                           describe(operand) + " elements overlap (dims " + listOf(dims)
                               + ", strides " + listOf(strides) + ")"};
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
        for(size_t index = 0; index < _operands.size(); ++index)
        {
            const bool isOutput = index == DQ || index == DK || index == DV;
            if(auto verdict = checkTensorLayout(_operands[index], _headDims[index], isOutput))
            {
                return verdict;
            }
            *slots[index] = recordStrides(_operands[index]);
        }
        return std::nullopt;
    }

    /// SS-3: stats declares the 16-byte alignment the kernels assume for every data tensor
    /// (the forward's check at the f32 element size is weaker). SS-1: stats is packed (B*H, Sq)
    /// (HT) or packed (B, Sq, H) (TH); the kernels take no stats strides, so any other layout
    /// declines. Both packed layouts address every element once, so SS-1 already excludes
    /// overlapping stats. The kernels also address one batch's H*Sq FLOAT rows through a 32-bit
    /// byte range, so a batch whose stats span exceeds it declines.
    Verdict checkStats()
    {
        if(!declaresAlignment(*_stats.tensor, REQUIRED_ALIGNMENT_BYTES))
        {
            return Decline{"operand",
                           describe(_stats) + " declares alignment "
                               + std::to_string(_stats.tensor->alignment()) + " bytes, below the "
                               + std::to_string(REQUIRED_ALIGNMENT_BYTES) + " the kernels assume"};
        }
        const auto& strides = stridesOf(_stats);
        const int64_t batch = _problem.batch;
        const int64_t heads = _problem.numHeads;
        const int64_t seq = _problem.seqLenQ;
        const auto packed = checkedProduct({heads, seq});
        if(packed.has_value() && *packed > INT32_LIMIT / STATS_BYTES_PER_ELEMENT)
        {
            return Decline{"layout",
                           describe(_stats) + " spans " + std::to_string(*packed)
                               + " rows per batch, over the 32-bit byte offset limit"};
        }
        const auto matches = [&](int64_t batchStride, int64_t headStride, int64_t seqStride) {
            return (batch == 1 || strides.Get(BATCH_AXIS) == batchStride)
                   && (heads == 1 || strides.Get(HEAD_AXIS) == headStride)
                   && (seq == 1 || strides.Get(SEQ_AXIS) == seqStride);
        };
        if(packed.has_value() && matches(*packed, seq, 1))
        {
            _problem.statsLayout = StatsLayout::HT;
        }
        else if(packed.has_value() && matches(*packed, 1, heads))
        {
            _problem.statsLayout = StatsLayout::TH;
        }
        else
        {
            return Decline{"layout",
                           describe(_stats) + " strides " + listOf(strides)
                               + " are neither packed (B*H, Sq) nor packed (B, Sq, H)"};
        }
        return std::nullopt;
    }

    static std::optional<int64_t> optionalLong(const flatbuffers::Optional<int64_t>& value)
    {
        return value;
    }

    /// MW-1: both deprecated causal flags decline. MW-2: a flag implies right bound 0 unless
    /// right_bound is set. MW-3: a bound below -1 declines. MW-4: only TOP_LEFT and BOTTOM_RIGHT
    /// alignments are served. MW-5, MW-6, MW-7, MW-8, MW-9a: windows are accepted in cuDNN's
    /// convention. MW-9b, MW-9c: left_bound 0 with a right bound of 0 or unbounded declines.
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
        if(bounds.left == 0 && bounds.right == 0)
        {
            return Decline{"mask", "left_bound 0 with right bound 0 selects an empty band"};
        }
        if(bounds.left == 0 && bounds.right == NO_WINDOW)
        {
            return Decline{
                "mask",
                "left_bound 0 with an unbounded right bound keeps only keys after the diagonal"};
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
    /// tensors, tensors without the flag, or one length tensor decline. PV-3, PV-8: a ragged
    /// offset on any operand or stats declines.
    Verdict checkVarlen()
    {
        const auto& a = _attributes;
        for(const auto& operand : _operands)
        {
            if(operand.tensor->ragged_offset_tensor_uid().has_value())
            {
                return Decline{"varlen", describe(operand) + " has a ragged offset tensor"};
            }
        }
        if(_stats.tensor->ragged_offset_tensor_uid().has_value())
        {
            return Decline{"varlen", describe(_stats) + " has a ragged offset tensor"};
        }
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
    /// (B, Hq, Sq, Skv), whose first three axes may broadcast. BI-2: the bias dtype is Q's.
    /// BI-3: a bias broadcast over batch or head with dBias is accepted. BI-4: dBias with a
    /// row-broadcast bias declines. BI-5, BI-7: dBias has the bias's dims and dtype. PV-8: a
    /// ragged offset on bias or dBias declines. PV-10: bias with padding_mask declines. MW-10:
    /// bias with a mask declines. LY-1, DT-9: bias and dBias alignment and slab bounds. BI-9:
    /// every non-unit stride of dBias other than the last is a multiple of 8 elements (4 for
    /// FLOAT), as LY-3 asks of the other outputs.
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
        if(bias.tensor->data_type() != _problem.dtype)
        {
            return Decline{"attn_mask",
                           describe(bias) + " is " + typeName(bias.tensor->data_type())
                               + ", but Q is " + typeName(_problem.dtype)};
        }
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
        const auto biasSlabRows = view.dims[SEQ_AXIS] > 1 ? _problem.seqLenQ - 1 : int64_t{0};
        if(!slabFits(biasSlabRows, view.strides[SEQ_AXIS], _problem.seqLenKv))
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
            if(dbiasDims.Get(axis) > 1 && stride % vectorElements() != 0)
            {
                return Decline{"layout",
                               describe(dbias) + " stride " + std::to_string(stride) + " on axis "
                                   + std::to_string(axis) + " (extent "
                                   + std::to_string(dbiasDims.Get(axis))
                                   + ") is not a non-negative multiple of "
                                   + std::to_string(vectorElements()) + " elements"};
            }
        }
        BiasView dbiasView;
        for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
        {
            dbiasView.dims[axis] = view.dims[axis];
            dbiasView.strides[axis] = axis < lead ? 0 : dbiasStrides.Get(axis - lead);
        }
        if(!slabFits(biasSlabRows, dbiasView.strides[SEQ_AXIS], _problem.seqLenKv))
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

    /// DR-4: INT32 and INT64 seed and offset tensors are accepted; the width is recorded. DR-6:
    /// a device tensor or a pass-by-value one is accepted, of any rank whose every extent is 1. DR-8: a device tensor is dense and
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
            return checkScalarMemory(
                operand, width == SeedWidth::INT32 ? INT32_BYTES : INT64_BYTES, "dropout");
        }
        return std::nullopt;
    }

    /// DR-1: probability 0 or unset is off, (0, 1) is on. DR-2: p outside [0, 1) or NaN
    /// declines. DR-3: p in (0, 1) needs both seed and offset, and either one without p > 0
    /// declines. DR-6: seed and offset share one width and one form. DR-7: dropout_mask,
    /// dropout_scale and dropout_scale_inv tensors decline.
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
            if(seedUid.has_value() || offsetUid.has_value())
            {
                return Decline{"dropout",
                               "seed or offset tensor set without dropout_probability > 0"};
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
