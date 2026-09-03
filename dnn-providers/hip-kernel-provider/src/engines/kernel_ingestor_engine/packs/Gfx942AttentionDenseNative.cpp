// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseGeometry.hpp"

/**
 * @file Gfx942AttentionDenseNative.cpp
 * @brief The hipkernel:Gfx942AttentionDense engine's native half: matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * The kernel is `rocke/library/kernels/gfx942/attention_dense.py`'s
 * `build_attention_dense`, packaged (kind: rocke -> kpack) for gfx942 only. Its
 * applicability rules live only in that Python; this file is where they become
 * enforceable. The derivations behind each check are recorded in `graph_contract.md`
 * (hipDNN's side) and `mining.md` (the kernel's side) at the repository root.
 *
 * Three facts drive almost every check below, and each one is a silent wrong answer
 * if missed:
 *
 *  1. **The kernel is BSHD; hipDNN dims are always (B, H, S, D) with layout in the
 *     strides.** `_build_attention_dense_single_buffer` computes
 *     `stride_q_tok = Hq * D` / `stride_k_tok = Hkv * D` (attention_dense.py:1155-1156)
 *     and takes NO stride arguments (`attention_dense_signature`, :1819). A BHSD graph
 *     is therefore indexed as if it were BSHD: in-bounds reads of the wrong elements,
 *     no fault. Every shipped quick/SdpaFwd bundle is BHSD, so this check is the
 *     difference between declining them and returning garbage for them.
 *
 *  2. **Every shape field is baked into the emitted binary**, including `batch`: the
 *     K/V buffer resources are sized `B * Skv * Hkv * D * 2` at build time
 *     (attention_dense.py:1264-1265), so a larger batch reads zero-fill past the
 *     bound rather than faulting. Hence strict equality on batch/seqlens/head counts
 *     in kernelMatches.
 *
 *  3. **hipDNN has no `causal` boolean.** Causality is derived from the deprecated
 *     `causal_mask` / `causal_mask_bottom_right` pair, which take precedence when set,
 *     otherwise from (`left_bound`, `right_bound`, `diagonal_alignment`). The
 *     canonical derivation is `asm_sdpa_engine/plans/SdpaPlanUtils.hpp::getMaskType`
 *     and is reproduced here rather than linked, because that header belongs to a
 *     different engine's plan layer. Reading only the deprecated booleans computes
 *     "not causal" for every shipped causal bundle -- they all leave both false and
 *     express causality as left_bound=-1, right_bound=0.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx942_attention_dense.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx942_attention_dense.dispatch";

// KMD fields this engine varies along.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view SEQLEN_Q_FIELD = "seqlen_q";
constexpr std::string_view SEQLEN_KV_FIELD = "seqlen_kv";
constexpr std::string_view BATCH_FIELD = "batch";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view BLOCK_N_FIELD = "block_n";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
// The persistent grid-stride variant launches a DIFFERENT grid shape, so the launch
// path must know which variant it holds. dispatch/attention/gfx942.py::_dense_spec
// turns it on when `nqb * Hq * B >= num_persistent`; the kernel bakes that choice into
// the binary, and attention_dense_grid() then returns (num_persistent, 1, 1) instead of
// the default 3-D grid. Without these two fields the engine cannot tell the two apart
// and launches every variant on the default grid -- which is exactly "the grid and the
// body's query tiling disagree ... writes some rows twice and others never".
constexpr std::string_view PERSISTENT_FIELD = "persistent";
constexpr std::string_view NUM_PERSISTENT_FIELD = "num_persistent";

constexpr std::string_view Q_TOKEN = "attention_dense.q.uid";
constexpr std::string_view K_TOKEN = "attention_dense.k.uid";
constexpr std::string_view V_TOKEN = "attention_dense.v.uid";
constexpr std::string_view O_TOKEN = "attention_dense.o.uid";
/// The mask type graphMatches derived, so kernelMatches does not re-derive it.
constexpr std::string_view CAUSAL_TOKEN = "attention_dense.causal";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "attention_dense.scale_bits";

// The problem geometry `problemFor` already derives, which is what a UHD actually ranks
// on. Without these the only bindable graph facts were four tensor uids -- high-cardinality
// handles that are actively harmful as model inputs -- plus causal and a float bit pattern,
// so no model could see the SHAPE at all and every kernel scored alike on every problem.
//
// The names restate the KMD field spellings above deliberately: the kernel varies along
// `batch`/`seqlen_q`/..., and a UHD ranks by comparing the problem against those very
// fields, so one vocabulary on both sides spares the model a translation table that could
// drift from either.
//
// The op root is `attention_dense`, extending the `attention_dense.q.uid` spelling this
// pack already ships rather than standing a second convention beside it. RFC 0019 §13.6
// writes the same fields under an `sdpa_fwd` root; reconciling the two roots is RFC 0020's
// naming question, and resolution is a flat lookup into BoundTokens at either depth.
constexpr std::string_view BATCH_TOKEN = "attention_dense.batch";
constexpr std::string_view SEQLEN_Q_TOKEN = "attention_dense.seqlen_q";
constexpr std::string_view SEQLEN_KV_TOKEN = "attention_dense.seqlen_kv";
constexpr std::string_view NUM_QUERY_HEADS_TOKEN = "attention_dense.num_query_heads";
constexpr std::string_view NUM_KV_HEADS_TOKEN = "attention_dense.num_kv_heads";
constexpr std::string_view HEAD_SIZE_TOKEN = "attention_dense.head_size";
/// One dtype, not four: the matcher refuses a graph whose operands disagree (step 5), so a
/// per-tensor spelling would publish three columns that are copies of the first -- constant
/// features that cost a small-data model and state nothing the first does not.
constexpr std::string_view DTYPE_TOKEN = "attention_dense.dtype";

// RFC 0019 §13.6's precomputed op-intrinsic cost fields, `$<op>.flops` and `$<op>.bytes`,
// which a features_signature divides into arithmetic intensity. They are facts about the
// op, identical for every engine implementing it, so they belong to the binding layer
// rather than to any one UHD.
constexpr std::string_view FLOPS_TOKEN = "attention_dense.flops";
constexpr std::string_view BYTES_TOKEN = "attention_dense.bytes";
/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of the
/// memory layout, which lives in the strides -- see SdpaAttributes.hpp's class doc.
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

/// Unbounded, in the left_bound/right_bound convention shared with the reference
/// executor (GpuRefSdpaFwd.cpp:129,141 treat a negative bound as "no clamp").
constexpr int64_t UNBOUNDED = -1;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids and derived scalars a matched dense-attention graph binds.
struct AttentionDenseBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    float scale = 0.0F;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors.
struct AttentionDenseProblem
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node this engine's matchers read, or nullptr if the graph is not a single
/// SDPA-forward node.
const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaAttributes)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::SdpaAttributes>();
}

/// Total over an UNVALIDATED graph: rank, stride/dim agreement and positive extents,
/// checked before anything indexes an axis. A caller can present a tensor the frontend
/// would have rejected, and every predicate below dereferences dims/strides.
bool isWellFormedOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }

    // Positive extents. supports_attention_dense checks this explicitly
    // (attention_dense.py:900-909) because Python's `%` is sign-following, so a zero
    // or negative extent passes every divisibility rule -- and Hq == 0 emits
    // `sdiv i32 %hq, 0` into the kernel.
    for(const auto dim : *dims)
    {
        if(dim <= 0)
        {
            return false;
        }
    }

    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/**
 * @brief Is this tensor's memory BSHD -- token-major, head varying fastest?
 *
 * The kernel bakes this layout: `stride_q_tok = Hq * D` (attention_dense.py:1155)
 * and there are no stride kernargs, so a differently-strided tensor is read as if it
 * were this one. Q/O use the Hq form, K/V the Hkv form; both are `H * D` in terms of
 * that tensor's own head count, so one predicate serves all four.
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the
 * extent is 1, so no address depends on it and a producer may declare anything there.
 * A single-head tensor is byte-identically BSHD and BHSD while the two spellings
 * disagree on strides[H] -- a strict compare would decline a graph the kernel serves
 * perfectly, and graph_match returning nullopt empties the WHOLE engine catalog.
 */
bool hasBshdStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, int64_t expected) {
        return dims->Get(axis) == 1 || strides->Get(axis) == expected;
    };

    return axisOk(BATCH_AXIS, sequence * heads * headSize) && axisOk(HEAD_AXIS, headSize)
           && axisOk(SEQ_AXIS, heads * headSize) && axisOk(HEAD_SIZE_AXIS, 1);
}

/// Mask classification, mirroring asm_sdpa_engine/plans/SdpaPlanUtils.hpp::getMaskType.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    SLIDING_WINDOW = 3
};

/**
 * @brief Which mask the graph is asking for.
 *
 * A REAL BOUND WINS OVER THE DEPRECATED BOOLEANS. The booleans only distinguish
 * top-left from bottom-right; they cannot express a window, so a graph that sets one
 * AND carries a bound is asking for a windowed mask and must be reported as such.
 *
 * This ordering is load-bearing rather than stylistic. Returning on the boolean first
 * -- which is what this function used to do -- served a causal graph with
 * `left_bound = 128` as plain causal: the window was silently discarded and the kernel
 * attended the whole causal triangle instead of the band. Five gpt_oss graphs in
 * hipDNN's own corpus are exactly that shape (`causal_mask = true`, `left_bound = 128`)
 * and were served, wrongly, before this was fixed. Nothing downstream catches it,
 * because SLIDING_WINDOW is declined at the switch below while TOP_LEFT_CAUSAL is
 * served -- so the bug turns a decline into a wrong answer.
 *
 * Both-deprecated-set is a caller error the frontend owns; here it is simply declined
 * rather than thrown, because graph_match must be total over an unvalidated graph.
 */
std::optional<MaskType> maskTypeFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();

    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }

    const int64_t left
        = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    const int64_t right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;

    // A bounded left edge is a window whatever the booleans say. `right` is not
    // consulted here: right == 0 is the causal spelling and right > 0 is a lookahead
    // band, and both are covered by the trio logic below.
    if(left != UNBOUNDED)
    {
        return MaskType::SLIDING_WINDOW;
    }

    if(topLeftDeprecated)
    {
        return MaskType::TOP_LEFT_CAUSAL;
    }
    if(bottomRightDeprecated)
    {
        return MaskType::BOTTOM_RIGHT_CAUSAL;
    }

    if(right == UNBOUNDED)
    {
        return MaskType::NO_MASK;
    }
    if(right == 0)
    {
        return attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT
                   ? MaskType::BOTTOM_RIGHT_CAUSAL
                   : MaskType::TOP_LEFT_CAUSAL;
    }
    return MaskType::SLIDING_WINDOW;
}

/// The kernel's dtype spelling for a graph dtype, or nullopt for one it cannot be
/// built for. One vocabulary: the KMD carries the hipDNN enum name, so that is what
/// the comparison uses -- `supports_attention_dense` gates dtype to bf16/fp16
/// (attention_dense.py:591, 870).
std::optional<std::string> supportedDataTypeName(data_objects::DataType dataType)
{
    if(dataType == data_objects::DataType::BFLOAT16)
    {
        return std::string("BF16");
    }
    if(dataType == data_objects::DataType::HALF)
    {
        return std::string("FP16");
    }
    return std::nullopt;
}

/// The graph's shape, read from Q and K. Callers must have validated both operands.
AttentionDenseProblem problemFor(const data_objects::TensorAttributes& q,
                                 const data_objects::TensorAttributes& k)
{
    AttentionDenseProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numQueryHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headSize = q.dims()->Get(HEAD_SIZE_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

/// The two runtime facts about a dtype the BINDING publishes.
///
/// `spelling` is what `to_string(DataType)` in hipdnn_frontend/Types.hpp answers -- the
/// only vocabulary a `.dtype` binding may hold, and the one CategoricalEncoding.hpp
/// encodes. It is restated here rather than called because this provider does not link the
/// frontend. It is emphatically NOT `supportedDataTypeName`'s vocabulary above: that one
/// answers the KMD's spelling ("BF16") so kernelMatches can compare against a kernel's
/// baked metadata, and crossing the two would hand CategoricalEncoding.hpp a string it
/// refuses as unknown -- losing the feature rather than warning.
///
/// An empty `spelling` is `to_string`'s "unknown" fallthrough, and `bytes == 0` is a width
/// this pack will not state: both make the dependent token absent instead of silently
/// wrong, which is what RFC 0019 §13.6 requires of a cost field with no exact form. The
/// sub-byte types are the second kind -- their footprint is a property of the packing, not
/// of the element, so no per-element byte count exists to publish. Neither arm is reachable
/// through THIS matcher, which declines every dtype but bf16/fp16 before it binds anything;
/// the table is written whole regardless, so that widening that gate cannot silently start
/// publishing a fabricated width.
///
/// This is the THIRD verbatim copy of this table (ConvNative.cpp, PointwiseNative.cpp hold
/// the other two). Two were a coincidence; three are a pattern, and this is the moment to
/// stop: the next pack that needs it must lift all three into a header the packs share
/// rather than add a fourth. It is duplicated once more here only because whether that
/// header exists is a design call this pack does not own, and inventing it mid-change would
/// leave two answers standing at once.
struct DataTypeFacts
{
    std::string_view spelling;
    int64_t bytes;
};

DataTypeFacts dataTypeFacts(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return {"fp32", 4};
    case data_objects::DataType::HALF:
        return {"fp16", 2};
    case data_objects::DataType::BFLOAT16:
        return {"bf16", 2};
    case data_objects::DataType::DOUBLE:
        return {"fp64", 8};
    case data_objects::DataType::UINT8:
        return {"uint8", 1};
    case data_objects::DataType::INT32:
        return {"int32", 4};
    case data_objects::DataType::INT8:
        return {"int8", 1};
    case data_objects::DataType::FP8_E4M3:
        return {"fp8_e4m3", 1};
    case data_objects::DataType::FP8_E5M2:
        return {"fp8_e5m2", 1};
    case data_objects::DataType::FP8_E8M0:
        return {"fp8_e8m0", 1};
    case data_objects::DataType::FP8_E4M3_FNUZ:
        return {"fp8_e4m3_fnuz", 1};
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return {"fp8_e5m2_fnuz", 1};
    case data_objects::DataType::INT64:
        return {"int64", 8};
    case data_objects::DataType::BOOLEAN:
        return {"boolean", 1};
    case data_objects::DataType::FP4_E2M1:
        return {"fp4_e2m1", 0};
    case data_objects::DataType::FP6_E2M3:
        return {"fp6_e2m3", 0};
    case data_objects::DataType::FP6_E3M2:
        return {"fp6_e3m2", 0};
    case data_objects::DataType::INT4:
        return {"int4", 0};
    case data_objects::DataType::UNSET:
    default:
        return {{}, 0};
    }
}

/// @p left * @p right, or nullopt if the product would overflow or either factor is not
/// positive. Dims arrive from an unvalidated graph and nothing upstream bounds their
/// product, so an unchecked multiply would publish a wrapped, negative "cost" as a model
/// feature -- the silently-wrong value RFC 0019 §13.6 rules out.
std::optional<int64_t> checkedMultiply(int64_t left, int64_t right)
{
    if(left <= 0 || right <= 0 || left > std::numeric_limits<int64_t>::max() / right)
    {
        return std::nullopt;
    }
    return left * right;
}

std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t result = 1;
    for(const auto factor : factors)
    {
        const auto next = checkedMultiply(result, factor);
        if(!next.has_value())
        {
            return std::nullopt;
        }
        result = *next;
    }
    return result;
}

std::optional<int64_t> elementCount(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    int64_t count = 1;
    for(const auto dim : *dims)
    {
        const auto next = checkedMultiply(count, dim);
        if(!next.has_value())
        {
            return std::nullopt;
        }
        count = *next;
    }
    return count;
}

/// Bytes moved by the op: the sum over Q/K/V/O of element_count x **that tensor's own**
/// dtype size. Reading one operand's dtype and applying it to all four is the bug RFC 0019
/// §13.6's caveat names -- wrong the moment a graph is mixed-precision. This matcher
/// refuses mixed dtypes today, so the per-tensor sum is the FORM rather than yet a
/// difference; a pack that later admits fp8-in/fp16-out inherits the right answer.
///
/// nullopt when any operand has no statable element width, so the token is absent rather
/// than short by one tensor.
std::optional<int64_t>
    movedBytes(std::initializer_list<const data_objects::TensorAttributes*> operands)
{
    int64_t total = 0;
    for(const auto* operand : operands)
    {
        const auto count = elementCount(*operand);
        const auto width = dataTypeFacts(operand->data_type()).bytes;
        if(!count.has_value() || width == 0)
        {
            return std::nullopt;
        }
        const auto operandBytes = checkedMultiply(*count, width);
        if(!operandBytes.has_value()
           || total > std::numeric_limits<int64_t>::max() - *operandBytes)
        {
            return std::nullopt;
        }
        total += *operandBytes;
    }
    return total;
}

/**
 * @brief The FLOPs this op performs, under the project's settled counting convention.
 *
 * CAUSAL-EFFECTIVE, NOT DENSE. RFC 0019 §13.6 settles the convention as whatever
 * `attention_flops` in rocKE's
 * `optimization/utilities/tools/stage1_benchmark/_ua_shape_utils.py:363` computes, which is
 * also aiter's and Triton's. For a square causal problem that is about HALF the dense
 * `4 * B * Hq * Sq * Skv * D`. The choice is load-bearing rather than cosmetic: this field
 * is the numerator of the arithmetic intensity a UHD ranks on and the denominator of every
 * TFLOPS number derived from it, so counting densely here while the training corpus counts
 * effectively in Python is train/serve skew inside a feature, not a rounding difference.
 *
 * The two branches:
 *  - `causal == 0`: the effective KV length is the full `seqLenKv`, and the expression
 *    reduces exactly to the dense `4 * B * Sq * Skv * Hq * D`.
 *  - `causal != 0`: with a contiguous query suffix the average effective KV length per
 *    query is `Skv - Sq + (Sq + 1) / 2`, so the MACs per sequence are
 *    `Sq * Skv - Sq * (Sq - 1) / 2` -- the dense rectangle less the triangle the mask
 *    discards. Written as that subtraction rather than as the reference's average because
 *    `Sq * (Sq - 1) / 2` is exact in integers while the average is a half-integer; the two
 *    agree term for term, and the half-integer form would need a float round-trip to
 *    reproduce the reference's per-sequence `int()` truncation.
 *
 * The `Sq > 2 * Skv + 1` arm reproduces the reference's `avg_eff < 0` fallback
 * (`avg_eff = kv_len / 2`) rather than improving on it, degenerate though that value is:
 * agreeing with the corpus generator is the point of adopting its convention, and a count
 * that diverges only in a corner is the harder disagreement to ever notice.
 *
 * nullopt on overflow -- §13.6's "absent beats silently wrong", because a wrapped negative
 * would not be caught downstream; it would become a model feature.
 */
std::optional<int64_t> attentionFlopsFor(const AttentionDenseProblem& problem, int64_t causal)
{
    const auto dense = checkedMultiply(problem.seqLenQ, problem.seqLenKv);
    if(!dense.has_value())
    {
        return std::nullopt;
    }

    int64_t macsPerSequence = *dense;
    if(causal != 0)
    {
        // Guarded on `> 1` because Sq == 1 masks nothing out and checkedMultiply refuses
        // the zero factor that Sq - 1 would then be.
        int64_t maskedOut = 0;
        if(problem.seqLenQ > 1)
        {
            if(problem.seqLenQ > std::numeric_limits<int64_t>::max() / (problem.seqLenQ - 1))
            {
                return std::nullopt;
            }
            maskedOut = problem.seqLenQ * (problem.seqLenQ - 1) / 2;
        }
        macsPerSequence = *dense >= maskedOut ? *dense - maskedOut : *dense / 2;
    }

    // 4 = two flops per multiply-accumulate, over the two GEMMs (QK^T and PV). Every
    // sequence in the batch has the same shape here -- varlen is declined at the matcher --
    // so the reference's per-sequence sum collapses to a multiply by `batch`.
    return checkedProduct(
        {4, problem.batch, macsPerSequence, problem.numQueryHeads, problem.headSize});
}
/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * The body is `mining.md`'s rejection checklist in its severity order: silent wrong
 * answers first (layout, unhonoured modes), then faults, then declined features.
 * Checks that compare the graph against a SPECIFIC kernel's baked constants are NOT
 * here -- no candidate exists yet -- they are in kernelMatches.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips
 *          EVERY remaining pack, not just this one
 *          (KernelIngestorStateManager.hpp:450-455).
 */
std::optional<BoundTokens> gfx942AttentionDenseGraphMatches(const MatchContext& context)
{
    // --- 1. Node shape. One SDPA-forward node; this engine serves a whole graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // --- 2. Operands. Q/K/V/O are the four the ABI has slots for
    // (attention_dense_signature, attention_dense.py:1819-1839).
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }

    // --- 3. Total predicates, before anything indexes an axis. O is included for
    // well-formedness only; its LAYOUT is checked in prepare(), not here, because the
    // frontend infers its shape and it is not reliably populated at match time.
    if(!isWellFormedOperand(*q) || !isWellFormedOperand(*k) || !isWellFormedOperand(*v)
       || !isWellFormedOperand(*o))
    {
        return std::nullopt;
    }

    // --- 4. Layout. Tier 1: the failure is wrong elements in bounds, no fault.
    if(!hasBshdStrides(*q) || !hasBshdStrides(*k) || !hasBshdStrides(*v))
    {
        return std::nullopt;
    }

    // --- 5. Cross-tensor consistency.
    const auto problem = problemFor(*q, *k);

    // One dtype across every operand: the builder takes a single `spec.dtype` and
    // types q/k/v/o pointers with it (attention_dense.py:1167-1177).
    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    if(!supportedDataTypeName(problem.dataType).has_value())
    {
        return std::nullopt;
    }

    // V shares K's base and stride in the builder, so it must share K's shape exactly.
    if(v->dims()->Get(BATCH_AXIS) != problem.batch
       || v->dims()->Get(HEAD_AXIS) != problem.numKvHeads
       || v->dims()->Get(SEQ_AXIS) != problem.seqLenKv
       || v->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    // K must agree with Q on batch, and on head size: the kernel has ONE head_size,
    // while hipDNN permits D_qk != D_v.
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    // O is Q's shape: the epilogue reuses q_base and stride_q_tok verbatim
    // (attention_dense.py:1708-1713).
    if(o->dims()->Get(BATCH_AXIS) != problem.batch
       || o->dims()->Get(HEAD_AXIS) != problem.numQueryHeads
       || o->dims()->Get(SEQ_AXIS) != problem.seqLenQ
       || o->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }

    // GQA: the kernel derives its group size by integer division, `gqa = Hq // Hkv`
    // (attention_dense.py:1154), so a non-divisible pair silently drops heads.
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }

    // --- 6. 32-bit addressing. Every offset is built from IRBuilder add/mul, which
    // lower to `add nsw`/`mul nsw` i32: signed overflow is UB, so LLVM may poison the
    // whole address chain rather than merely read the wrong place. Mirrors
    // supports_attention_dense:1067-1084. Note the two bounds have DIFFERENT units --
    // K/V is bytes, Q/O is elements -- exactly as the Python has them.
    constexpr int64_t INT32_LIMIT = 2147483648LL; // 2^31
    constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 are the only dtypes here
    if(problem.batch * problem.seqLenKv * problem.numKvHeads * problem.headSize * BYTES_PER_ELEMENT
       >= INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(problem.batch * problem.seqLenQ * problem.numQueryHeads * problem.headSize >= INT32_LIMIT)
    {
        return std::nullopt;
    }

    // --- 7. The mask. hipDNN has no `causal` boolean; see maskTypeFor.
    const auto mask = maskTypeFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }
    int64_t causal = 0;
    switch(*mask)
    {
    case MaskType::NO_MASK:
        causal = 0;
        break;
    case MaskType::TOP_LEFT_CAUSAL:
        causal = 1;
        break;
    case MaskType::BOTTOM_RIGHT_CAUSAL:
        // The kernel's causal clamp is TOP-LEFT: its KV-loop bound is derived from the
        // query-block index with no (Skv - Sq) offset term, i.e. the reference's
        // `windowOffset == 0` case (GpuRefSdpaFwd.cpp:92). Bottom-right differs from
        // top-left only by that offset, so the two coincide EXACTLY when Sq == Skv --
        // and every shipped quick/SdpaFwd causal bundle sets BOTTOM_RIGHT at Sq == Skv.
        // Declining it outright would decline all of them; serving it at Sq != Skv
        // would be a silent wrong answer. Hence the shape-conditional accept.
        if(problem.seqLenQ != problem.seqLenKv)
        {
            return std::nullopt;
        }
        causal = 1;
        break;
    case MaskType::SLIDING_WINDOW:
    default:
        // supports_attention_dense rejects sliding_window unconditionally
        // (attention_dense.py:922). `default` is required by -Wswitch-default and
        // doubles as the safe verdict for any mask kind added to the enum later:
        // an unrecognised mask is declined, never served as if it were dense.
        return std::nullopt;
    }

    // --- 8. Every optional attribute this kernel cannot honour, declined explicitly.
    // Worked from sdpa_attributes.fbs field by field: an UNCHECKED field is accepted
    // and then silently not performed. Several features appear under more than one
    // spelling and rejecting only one admits graphs carrying the rest.

    // Additive attention bias: no such input in the 5-slot ABI.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 KERNARG
    // (attention_dense.py:1837), so there is no pointer slot for a scale tensor.
    if(attributes.scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // varlen, both spellings.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Dropout: five spellings of one feature.
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt;
    }
    // Paged KV. supports_attention_dense never inspects `spec.paged`; the base spec's
    // validators reject every paged configuration reachable here (paged requires
    // sliding_window > 0, which is then rejected unconditionally), so the graph side is
    // the only place this can be caught.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        return std::nullopt;
    }
    // Block-sparse and attention sinks.
    if(attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // FP8 quantization: six scale UIDs plus two amax outputs.
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Auxiliary softmax outputs. `generate_stats` and `stats_tensor_uid` are two
    // spellings of one request; `max` and `sum_exp` are two more outputs the kernel
    // never writes. Note generate_stats is optional<bool>, so an explicit `false` is
    // fine and only a `true` is declined.
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        return std::nullopt;
    }
    // ALiBi slopes and padding masks: neither has a code path in the body.
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt;
    }
    // mma_core_mode overrides the MFMA accumulate type. The builder emits an f32
    // accumulator unconditionally, so UNSET (schema default) and an explicit FLOAT are
    // both inert; any other request names a core mode the builder cannot parameterise.
    // Written as an allow-list rather than `!= UNSET`: every shipped SdpaFwd bundle
    // sets "float", so the naive form silently declines all of them.
    if(attributes.mma_core_mode() != data_objects::DataType::UNSET
       && attributes.mma_core_mode() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint. AUTO leaves the choice to the
    // provider; a named strategy is a request for a specific backend shape this pack
    // does not implement, and honouring it is not the same as ignoring it.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // The softmax scale is a REQUIRED launch argument with no default. The schema
    // marks it optional and the mathematically obvious 1/sqrt(D) is exactly the guess
    // native-pack.md warns against: it would silently override whatever the frontend's
    // omission meant. Require presence, decline absence.
    if(!attributes.attn_scale_value().has_value())
    {
        return std::nullopt;
    }
    const float scale = attributes.attn_scale_value().value();

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(CAUSAL_TOKEN)] = causal;
    // BoundTokens carry int64_t, so the scale travels as its IEEE-754 bit pattern and
    // is reassembled in prepare(). A float rounded through an integer would not
    // round-trip.
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);

    // The geometry, without which a UHD sees no shape at all and can only rank on the
    // tensor uids above. Everything here is what problemFor already read off the graph;
    // dispatch does not consume any of it, the model does.
    bound[std::string(BATCH_TOKEN)] = problem.batch;
    bound[std::string(SEQLEN_Q_TOKEN)] = problem.seqLenQ;
    bound[std::string(SEQLEN_KV_TOKEN)] = problem.seqLenKv;
    bound[std::string(NUM_QUERY_HEADS_TOKEN)] = problem.numQueryHeads;
    bound[std::string(NUM_KV_HEADS_TOKEN)] = problem.numKvHeads;
    bound[std::string(HEAD_SIZE_TOKEN)] = problem.headSize;

    // dtype binds as the runtime spelling STRING, never a pre-encoded number: the integer
    // code space is CategoricalEncoding.hpp's and is applied downstream by the feature
    // extractor, so a number here would freeze that code space inside the matcher and let
    // the two drift apart silently.
    const auto facts = dataTypeFacts(problem.dataType);
    if(!facts.spelling.empty())
    {
        bound[std::string(DTYPE_TOKEN)] = std::string(facts.spelling);
    }

    const auto flops = attentionFlopsFor(problem, causal);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }

    const auto bytes = movedBytes({q, k, v, o});
    if(bytes.has_value())
    {
        bound[std::string(BYTES_TOKEN)] = *bytes;
    }
    return bound;
}

/// Re-reads the bindings a match established. Throws rather than returns, because
/// reaching dispatch without them is an internal inconsistency, not a graph the engine
/// declines.
AttentionDenseBinding attentionDenseBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "gfx942 attention_dense dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    AttentionDenseBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.causal = read(CAUSAL_TOKEN);

    const auto scaleBits = static_cast<int32_t>(read(SCALE_BITS_TOKEN));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit?
 *
 * Strict equality on every shape field, because this kernel is fully
 * shape-specialized: batch, both sequence lengths and both head counts are compiled
 * into the emitted code (the K/V buffer-resource extent at attention_dense.py:1264-1265,
 * the KV-loop trip count `n_ktiles = Skv // BN` at :1236, and the grid at :1791). A
 * graph differing on any of them is served zero-fill or a truncated loop, silently.
 *
 * block_m / block_n are the knob-selection rows: the graph supplies the sequence
 * lengths, the variant supplies the tile, and the test is per-candidate so a different
 * tile can serve what this one cannot.
 */
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return false;
    }
    const auto& attributes = *attributesPtr;

    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    if(q == nullptr || k == nullptr)
    {
        return false;
    }
    const auto problem = problemFor(*q, *k);

    const auto dataTypeName = supportedDataTypeName(problem.dataType);
    if(!dataTypeName.has_value()
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dataTypeName)
    {
        return false;
    }

    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };

    if(intField(HEAD_SIZE_FIELD) != problem.headSize
       || intField(NUM_QUERY_HEADS_FIELD) != problem.numQueryHeads
       || intField(NUM_KV_HEADS_FIELD) != problem.numKvHeads
       || intField(SEQLEN_Q_FIELD) != problem.seqLenQ
       || intField(SEQLEN_KV_FIELD) != problem.seqLenKv || intField(BATCH_FIELD) != problem.batch)
    {
        return false;
    }

    // The mask the graph derived to, against the mask this variant was compiled for.
    const auto causal = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CAUSAL_TOKEN);
    if(!causal.has_value() || intField(CAUSAL_FIELD) != *causal)
    {
        return false;
    }

    // Knob-selection rows. Implied by the shape equalities above for the variants this
    // engine ships, but stated because they are properties of the TILE, not the shape:
    // a future variant set with a shape range would need exactly these.
    const int64_t blockM = intField(BLOCK_M_FIELD);
    const int64_t blockN = intField(BLOCK_N_FIELD);
    return blockM > 0 && blockN > 0 && problem.seqLenQ % blockM == 0
           && problem.seqLenKv % blockN == 0;
}

/**
 * @brief Ranks the candidates that survived kernelMatches. Higher wins.
 *
 * Every shape and capability field is pinned by kernelMatches, so exactly one axis is
 * still free: `block_n`, the KV tile width. A larger tile amortises the per-tile
 * barrier, LDS publication and loop overhead across more keys, so it is ranked higher.
 *
 * This is a HEURISTIC placeholder, deliberately ranking one real knob rather than
 * returning a constant (which would make ranking arbitrary and hide
 * mis-specialization). What would replace it: measured per-config timings, or a UHD
 * model. The kernel's own module docstring records that block_n=32 is proven-negative
 * for fp16 D128 and only part-dependently positive for bf16 D128, which is consistent
 * with preferring 64 -- but it is a measured claim this function does not encode.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_N_FIELD)));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything launch() needs, resolved once and owning
/// nothing that points back into the MatchContext or BoundTokens it came from.
class PreparedGfx942AttentionDense : public PreparedDispatch
{
public:
    PreparedGfx942AttentionDense(std::unique_ptr<compilation::ICompiledProgram> program,
                                 std::unique_ptr<compilation::IRunnableKernel> kernel,
                                 AttentionDenseBinding binding)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const AttentionDenseBinding& binding() const
    {
        return _binding;
    }

private:
    // The runnable kernel is a VIEW into its program's module; both are held for the
    // plan's lifetime or the kernel dangles.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    AttentionDenseBinding _binding;
};

/**
 * @brief The native dispatch behind this engine's UDD: sizes, prepares and launches.
 *
 * Split per RFC 0017 §8.5: everything graph- or kernel-derived resolves at prepare(),
 * so launch() only resolves device buffers and launches, and nothing mutates once
 * prepared.
 */
class Gfx942AttentionDenseDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    Gfx942AttentionDenseDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                        const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Zero, and that is a real answer rather than a stub: the kernel's only scratch is
    /// LDS (K_lds + V_lds, sized at build time by `_lds_bytes` and bounded by the
    /// 64 KB gfx942 capacity) plus registers. It allocates no global scratch, and the
    /// 5-slot ABI has no workspace pointer to hand one to.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = attentionDenseBinding(bound);

        // Deferred from graph_match: the output tensor's shape is inferred by the
        // frontend and is not reliably populated during matching, so requiring its
        // layout there would decline graphs this engine serves. By prepare() it is
        // real, and O is addressed with q_base/stride_q_tok verbatim -- a non-BSHD O
        // writes the right bytes to the wrong places, silently.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx942 attention_dense: the output tensor is not dense BSHD; the kernel "
                "bakes that layout and takes no stride arguments");
        }

        // KernelCompileOptions dereferences the tensor it is handed UNCONDITIONALLY:
        // addDataTypeAndLayoutOptions() calls `tensorAttrs->data_type()` with no null
        // check, and isChannelLastLayout() THROWS for any 4D stride order that is
        // neither NCHW nor NHWC. BSHD attention memory is neither, so passing the real
        // query tensor throws at prepare() time -- and passing `nullptr` segfaults.
        // "Layout-neutral stand-in" means a minimal REAL tensor, not the absence of
        // one; reading it the other way cost a device run.
        //
        // Hence a 1x1x1x1 tensor with NCHW-ordered strides, which classifies cleanly.
        // Safe on this path and only this path: every kernel in this pack is KPACK
        // (kernel_source.kind: rocke, lowered by hkp_pack at build time), and
        // buildIngestorKernelCode deliberately does not consult `options` on the KPACK
        // branch -- a kpack blob's build defines were baked at pack time. An
        // EMBEDDED_SOURCE kernel in this pack would need a real answer here.
        flatbuffers::FlatBufferBuilder standInBuilder;
        {
            const std::vector<int64_t> unitDims{1, 1, 1, 1};
            const std::vector<int64_t> unitStrides{1, 1, 1, 1};
            standInBuilder.Finish(
                data_objects::CreateTensorAttributesDirect(standInBuilder,
                                                           0,
                                                           nullptr,
                                                           data_objects::DataType::FLOAT,
                                                           &unitStrides,
                                                           &unitDims,
                                                           false));
        }
        const auto* standIn = flatbuffers::GetRoot<data_objects::TensorAttributes>(
            standInBuilder.GetBufferPointer());
        const compilation::KernelCompileOptions options(standIn,
                                                        context.deviceProperties.gcnArchName);

        auto code
            = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options);

        // Geometry, restated from the builder's own helpers, INCLUDING the persistent
        // branch. The arithmetic and its guards live in attentionDenseGeometry() so a
        // test can reach them without a device: this correspondence is unchecked by
        // the build, the packer and the validator, it fails silently rather than
        // loudly, and it has already shipped two defects. Every term comes from the
        // KMD, so a variant launches the geometry it was actually compiled for.
        const auto geometry
            = attentionDenseGeometry(kernel.getIntMetadata(std::string(BLOCK_M_FIELD)),
                                     kernel.getIntMetadata(std::string(SEQLEN_Q_FIELD)),
                                     kernel.getIntMetadata(std::string(NUM_QUERY_HEADS_FIELD)),
                                     kernel.getIntMetadata(std::string(BATCH_FIELD)),
                                     kernel.getIntMetadata(std::string(PERSISTENT_FIELD)),
                                     kernel.getIntMetadata(std::string(NUM_PERSISTENT_FIELD)),
                                     toString(kernel.kernelId));

        code.kernel->setBlockSize(geometry.blockX, 1, 1);
        code.kernel->setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);

        return std::make_unique<PreparedGfx942AttentionDense>(
            std::move(code.program), std::move(code.kernel), binding);
    }

    /// The ABI is `attention_dense_signature` (attention_dense.py:1819-1839):
    /// `(q_ptr, k_ptr, v_ptr, o_ptr, scale)`, in that order. It is UNCONDITIONAL --
    /// five slots for every spec this engine ships, with no optional pointers appended
    /// under a feature flag, because varlen/paged/sinks are all declined. That order is
    /// a hand-maintained contract with the Python, unchecked by the type system.
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedDense = dynamic_cast<const PreparedGfx942AttentionDense&>(prepared);
        const auto& binding = preparedDense.binding();

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);

        preparedDense.kernel().launch(
            handle.getStream(), q.ptr, k.ptr, v.ptr, o.ptr, binding.scale);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& gfx942AttentionDenseKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx942AttentionDenseModuleCache()
{
    gfx942AttentionDenseKpackModuleCache().clear();
}

namespace
{

/// This engine's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so it
/// (and the compiler and loader it holds) must outlive every Container.
const Gfx942AttentionDenseDispatchHandler& gfx942AttentionDenseDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx942AttentionDenseKpackModuleCache());
    static const Gfx942AttentionDenseDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                       s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerGfx942AttentionDenseSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx942AttentionDenseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx942AttentionDenseDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
