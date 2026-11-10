// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionDenseCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"
#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"

/**
 * @file AttentionDenseNative.cpp
 * @brief The native half hipkernel:Gfx942AttentionDense and hipkernel:Gfx950AttentionDense
 *        share: graph and kernel matching, the kernel signature, dispatch, and the saved
 *        launch values.
 *
 * The kernels are `rocke/library/kernels/gfx942/attention_dense.py` and
 * `rocke/library/kernels/gfx950/attention_dense.py`'s `build_attention_dense`, packaged
 * (kind: rocke -> kpack) for their own arch only. Their applicability rules live only in
 * that Python; this file is where they become enforceable. Each engine's
 * `Gfx9xxAttentionDenseNative.cpp` binds these functions to its AttentionDenseArch and
 * registers them under its own symbols.
 *
 * Every KMD field read here is one both engine manifests declare, except kv_segments,
 * which only gfx942's declares (KV_SEGMENTS_FIELD).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::attention_dense
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

// The KMD fields both engines' schemas carry and this file reads. The KMD carries only
// what varies between candidates; a variant that moved a field absent from it would
// collide on the catalog key and the loader would drop one. The metadata's batch and
// sequence lengths are canonical build inputs of a shape-generic kernel and are never
// read here.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view RAGGED_FIELD = "ragged";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
constexpr std::string_view BLOCK_N_FIELD = "block_n";
// A split-KV build multiplies the grid's x extent by kv_segments and takes a workspace and
// a combine launch this dispatch does not make. The gfx942 KMD declares it; the gfx950 KMD
// does not and every gfx950 variant is built at 1, so an absent field reads as 1, the one
// value served.
constexpr std::string_view KV_SEGMENTS_FIELD = "kv_segments";

// sdpa_graph's single-node SDPA-forward helpers this pack reads the graph with.
using sdpa_graph::BATCH_AXIS;
using sdpa_graph::computesInFloat;
using sdpa_graph::describeOperand;
using sdpa_graph::findTensor;
using sdpa_graph::hasBshdStrides;
using sdpa_graph::hasSdpaForwardNode;
using sdpa_graph::HEAD_AXIS;
using sdpa_graph::HEAD_SIZE_AXIS;
using sdpa_graph::problemFor;
using sdpa_graph::SDPA_RANK;
using sdpa_graph::sdpaNode;
using sdpa_graph::SEQ_AXIS;

// Every product of graph-controlled extents goes through this: an unchecked product is
// signed overflow. Every factor here is an extent isWellFormedOperand already held positive.
using pack_features::checkedProduct;

/// The base alignment every dense operand must declare. Both arches' builders declare the
/// q/k/v/o kernel parameters `align=16` (rocke/library/kernels/gfx9{42,50}/attention_dense.py),
/// which lowers to a literal `align 16` LLVM parameter attribute, so the promise is made about
/// the base pointer itself and does not depend on which kernel variant dispatch picks. The
/// backend enforces only the alignment a graph declares, so a graph declaring less would hand
/// the kernel a pointer it has already told LLVM cannot arrive: that is UB, not a fault, so it
/// has to be refused here rather than left to misbehave. 16 is also the schema default, so only
/// a graph that explicitly under-declares is turned away.
constexpr int64_t OPERAND_ALIGNMENT_BYTES = 16;

/// sdpa_graph::isWellFormedOperand, and no ragged offset tensor: a ragged tensor packs the
/// sequences back to back, and the dense kernels index the buffer as (B, S, H, D), so they
/// would read and write past each sequence. The dense packs' own check.
bool isWellFormedOperand(const data_objects::TensorAttributes& tensor)
{
    return sdpa_graph::isWellFormedOperand(tensor)
           && !tensor.ragged_offset_tensor_uid().has_value();
}

// The softmax scales both kernels compute correctly: a chosen margin, not a hardware limit,
// and the range both rocKE runners check (check_dense_scale). gfx950 takes the row max on
// unscaled scores (valid only for scale > 0), folds the scale into an fma whose rounding
// residue grows with the scale, and masks raw scores with a power-of-two sentinel. gfx942
// multiplies the f32 QK accumulator by scale * log2(e), then masks with the finite sentinel
// -1e30 (its attention_dense.py qk_scale, do_mask): a visible score below -1e30 loses the
// row max to a masked key, and a product past FLT_MAX makes s - m = inf - inf = NaN. At 2^4 the
// multiplier is 16 * log2(e) ~= 23.08, and fp16 inputs at D <= 128 give
// |q.k| <= 128 * 65504^2 ~= 5.5e11, so every score stays within ~1.3e13, far inside -1e30.
// bf16 shares f32's exponent range, so no scale bound covers every bf16 input, but a score
// reaches -1e30 only past |q.k| ~= 4.3e28 and FLT_MAX only past ~1.5e37. NaN, ±inf, zero
// and negative scales are outside the range.
bool isSupportedScale(float scale)
{
    return !std::isnan(scale) && scale >= 0x1p-64F && scale <= 0x1p4F;
}

/// The graph decline and saved-plan refusal text for a scale isSupportedScale rejects.
constexpr std::string_view UNSUPPORTED_SCALE_REASON = "is outside the supported [2^-64, 2^4]";

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations
// available for the graph", so each decline logs its cause at INFO. graph_match runs on
// every catalog miss, so every piece of that work, these helpers included, sits inside
// HIPDNN_PLUGIN_LOG_INFO's argument: the macro tests the level first, and with INFO off
// nothing below is called and no message is built.
// ---------------------------------------------------------------------------

/// The decline head of @p arch's engine for @p cause. A function rather than a braced
/// DeclinedBy, so the logging macro does not split its argument at the comma.
sdpa_graph::DeclinedBy declined(const AttentionDenseArch& arch, std::string_view cause)
{
    return sdpa_graph::DeclinedBy{arch.engineName, cause};
}

/// Why @p tensor is not one the kernel can address, with the strides the kernel bakes for
/// its dims. Checks the rank itself rather than trusting that the caller validated it.
std::string notBshdReason(std::string_view role, const data_objects::TensorAttributes& tensor)
{
    std::ostringstream out;
    out << describeOperand(role, tensor) << " is not dense BSHD (token-major, head fastest)";

    const auto dims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(tensor.dims());
    if(dims.size() != SDPA_RANK)
    {
        return out.str();
    }
    const int64_t heads = dims[HEAD_AXIS];
    const int64_t sequence = dims[SEQ_AXIS];
    const int64_t headSize = dims[HEAD_SIZE_AXIS];
    const auto show = [](std::optional<int64_t> value) {
        return value.has_value() ? std::to_string(*value) : std::string("overflow");
    };
    out << "; the kernel bakes strides [" << show(checkedProduct({sequence, heads, headSize}))
        << ", " << headSize << ", " << show(checkedProduct({heads, headSize})) << ", 1]"
        << " (unit-extent axes exempt) and takes no stride arguments";
    return out.str();
}

/// The kernel's dtype spelling for a graph dtype, or nullopt for one it cannot be built for.
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

/// Whether a top-left causal window of @p slidingWindow keys leaves the last query rows no
/// key: row q sees (q - W, q], and q - W + 1 >= seqlen_kv once seqlen_q - seqlen_kv >= W.
/// Only the kernel's LSE path zeroes such rows; otherwise every score of such a row is the
/// mask value, so it averages V over the tiles it visits. A zero window (no window) leaves
/// none.
bool windowLeavesRowsWithoutKeys(int64_t seqLenQ, int64_t seqLenKv, int64_t slidingWindow)
{
    return slidingWindow > 0 && seqLenQ - seqLenKv >= slidingWindow;
}

/// head_size is 64 or 128 (AttentionDenseSpec.__post_init__).
bool isSupportedHeadSize(int64_t headSize)
{
    return headSize == 64 || headSize == 128;
}

// The kernel addresses K/V with 32-bit byte offsets and Q/O with 32-bit element offsets. A
// product too large for int64_t is past the limit, so it fails like one that fits and
// exceeds it.
constexpr int64_t OFFSET_LIMIT = 2147483648LL; // 2^31
constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 only

/// Whether K and V are each below 2^31 bytes.
bool kvFitByteOffsets(const AttentionDenseProblem& problem)
{
    const auto kvBytes = checkedProduct(
        {problem.batch, problem.seqLenKv, problem.numKvHeads, problem.headSize, BYTES_PER_ELEMENT});
    return kvBytes.has_value() && *kvBytes < OFFSET_LIMIT;
}

/// Whether Q and O each have fewer than 2^31 elements.
bool qoFitElementOffsets(const AttentionDenseProblem& problem)
{
    const auto qElements
        = checkedProduct({problem.batch, problem.seqLenQ, problem.numQueryHeads, problem.headSize});
    return qElements.has_value() && *qElements < OFFSET_LIMIT;
}

/// The stats tensor's trailing unit axis: [B, H, S_q, 1].
constexpr uint32_t STATS_UNIT_AXIS = 3;

/**
 * @brief Is this the stats tensor the kernel writes: rank 4 with dims [B, Hq, Sq, 1] and
 *        packed strides [Hq * Sq, Sq, 1, any], not virtual, ragged or pass-by-value?
 *
 * The kernel writes row (b, h, s) at element (b * Hq + h) * Sq + s of a FLOAT buffer and
 * takes no stride arguments, so any other layout is written as if it were this one. As in
 * hasBshdStrides, a unit-extent axis is exempt: its stride multiplies an index that is
 * always 0. The data type is checked separately.
 */
bool isServableStatsTensor(const data_objects::TensorAttributes& stats,
                           const AttentionDenseProblem& problem)
{
    const auto* dims = stats.dims();
    const auto* strides = stats.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK || stats.virtual_()
       || stats.ragged_offset_tensor_uid().has_value()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&stats))
    {
        return false;
    }
    if(dims->Get(BATCH_AXIS) != problem.batch || dims->Get(HEAD_AXIS) != problem.numQueryHeads
       || dims->Get(SEQ_AXIS) != problem.seqLenQ || dims->Get(STATS_UNIT_AXIS) != 1)
    {
        return false;
    }

    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1 || (expected.has_value() && strides->Get(axis) == *expected);
    };
    return axisOk(BATCH_AXIS, checkedProduct({problem.numQueryHeads, problem.seqLenQ}))
           && axisOk(HEAD_AXIS, problem.seqLenQ) && axisOk(SEQ_AXIS, 1);
}

/// The mask the kernel is built for: causal or not, and the window in keys (0 = none).
struct AttentionDenseMask
{
    int64_t causal = 0;
    int64_t slidingWindow = 0;
};

/**
 * @brief The mask the graph asks for, as the kernel spells it, or nullopt for one the
 *        kernel cannot compute.
 *
 * Deprecated causal booleans are merged with the bounds and alignment by
 * asm_sdpa_engine::plan_utils::resolveDiagonalBand, then classified by getMaskType
 * without makeCoveringBoundsUnbound (the Python shape miner does not unbound covering
 * windows either). Both booleans set, invalid bounds, or any band other than unmasked,
 * causal or (where the arch serves one) a causal window, is declined.
 *
 * The kernel's causal clamp and window are anchored TOP-LEFT: key k is visible to query q
 * iff `q - sliding_window < k <= q`. The hipDNN reference keeps k iff
 * `q - left_bound < k <= q + right_bound` (SEM-001), so a causal window of left_bound L
 * is the kernel's sliding_window L. resolveDiagonalBand hands back L - 1, the keys left of
 * the diagonal, and rejects L == 0. Bottom-right alignment coincides with top-left exactly
 * when seqlen_q == seqlen_kv, and is declined otherwise: the kernels build a bottom-right
 * causal mask (causal_bottom_right), but no shipped variant carries one, and neither
 * builds a bottom-right window. A band with a right bound other than 0 is not causal and
 * is declined; this pack ships no lookahead or non-causal window.
 *
 * An arch that serves no window (servesSlidingWindow() false) declines every windowed
 * band: its catalog carries no non-zero sliding_window.
 *
 * A top-left window that leaves the last query rows no key (windowLeavesRowsWithoutKeys)
 * is declined. This restates supports_attention_dense's row rule without its sink and LSE
 * exemptions: both packs decline sinks, and a graph that binds stats is declined here too,
 * because the rows such a window leaves keyless are graded only against an infinite LSE,
 * which no device-site validator compares.
 */
std::optional<AttentionDenseMask> maskFor(const data_objects::SdpaAttributes& attributes,
                                          const AttentionDenseProblem& problem,
                                          const AttentionDenseArch& arch)
{
    using asm_sdpa_engine::plan_utils::getMaskType;
    using asm_sdpa_engine::plan_utils::resolveDiagonalBand;
    using PlanMaskType = asm_sdpa_engine::plan_utils::MaskType;

    asm_sdpa_engine::plan_utils::DiagonalBand band{};
    PlanMaskType planMask{};
    try
    {
        band = resolveDiagonalBand(attributes);
        planMask = getMaskType(band.leftBound, band.rightBound, band.alignment);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mask") << e.what());
        return std::nullopt;
    }

    const bool sameLengths = problem.seqLenQ == problem.seqLenKv;
    const bool bottomRight = band.alignment == data_objects::DiagonalAlignment::BOTTOM_RIGHT;

    switch(planMask)
    {
    case PlanMaskType::NO_MASK:
        return AttentionDenseMask{0, 0};
    case PlanMaskType::TOP_LEFT_CAUSAL:
        return AttentionDenseMask{1, 0};
    case PlanMaskType::BOTTOM_RIGHT_CAUSAL:
        if(!sameLengths)
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mask")
                                   << "bottom-right causal needs seqlen_q == seqlen_kv, got "
                                   << problem.seqLenQ << " and " << problem.seqLenKv
                                   << "; not yet supported by this pack, whose variants are "
                                      "all top-left");
            return std::nullopt;
        }
        return AttentionDenseMask{1, 0};
    case PlanMaskType::SLIDING_WINDOW:
    {
        if(!arch.servesSlidingWindow())
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mask")
                                   << "left_bound " << attributes.left_bound().value_or(-1)
                                   << ", right_bound " << attributes.right_bound().value_or(-1)
                                   << " is a window; the kernel is causal or unmasked only and "
                                      "no shipped variant carries a sliding_window");
            return std::nullopt;
        }
        if(band.rightBound != 0 || band.leftBound < 0)
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mask")
                                   << "left_bound " << band.leftBound << ", right_bound "
                                   << band.rightBound
                                   << " is not a causal window; the gfx942 and gfx950 kernels "
                                      "implement right_bound 0 windows only");
            return std::nullopt;
        }
        if(bottomRight && !sameLengths)
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mask")
                                   << "a bottom-right window needs seqlen_q == seqlen_kv, got "
                                   << problem.seqLenQ << " and " << problem.seqLenKv
                                   << "; the kernel builds top-left windows only");
            return std::nullopt;
        }
        const int64_t window = band.leftBound + 1;
        if(windowLeavesRowsWithoutKeys(problem.seqLenQ, problem.seqLenKv, window))
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mask")
                                   << "a " << window << "-key window over seqlen_q "
                                   << problem.seqLenQ << " and seqlen_kv " << problem.seqLenKv
                                   << " leaves the last query rows no key; not yet supported "
                                      "by this pack (only the kernel's LSE path zeroes such "
                                      "rows)");
            return std::nullopt;
        }
        return AttentionDenseMask{1, window};
    }
    default:
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mask") << "mask type " << static_cast<int>(planMask)
                                                      << " is not supported");
        return std::nullopt;
    }
}

/**
 * @brief The stats output the graph binds -- its uid, or ATTENTION_DENSE_NO_STATS for
 *        none -- or nullopt for a stats request the kernel cannot honour, logged.
 *
 * For an arch that serves stats. The rule is the FlyDSL packs': a stats_tensor_uid is
 * served unless generate_stats is explicitly false, because the references write the LSE
 * whenever the uid is set; generate_stats true without a uid names no tensor to write and
 * is declined (stats_flag). The tensor must be FLOAT (stats_dtype) and laid out as
 * isServableStatsTensor describes (stats_layout). Max, Sum_exp and the dropout RNG dump
 * are outputs the kernel cannot write (stats_rowstats).
 */
std::optional<int64_t> statsFor(const MatchContext& context,
                                const data_objects::SdpaAttributes& attributes,
                                const AttentionDenseProblem& problem,
                                const AttentionDenseArch& arch)
{
    if(attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "stats_rowstats")
                               << "the Max, Sum_exp and dropout RNG dump outputs are not "
                                  "supported; the kernel writes the LSE (stats) only");
        return std::nullopt;
    }

    const auto generateStats = attributes.generate_stats();
    if(!attributes.stats_tensor_uid().has_value())
    {
        if(generateStats.has_value() && generateStats.value())
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "stats_flag")
                                   << "generate_stats is true, but the node names no stats "
                                      "tensor to write the LSE to");
            return std::nullopt;
        }
        return ATTENTION_DENSE_NO_STATS;
    }

    const int64_t statsUid = attributes.stats_tensor_uid().value();
    if(generateStats.has_value() && !generateStats.value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "stats_flag")
                               << "the node names stats tensor uid " << statsUid
                               << " but sets generate_stats to false");
        return std::nullopt;
    }
    // A negative uid would read back as ATTENTION_DENSE_NO_STATS, or be refused on restore.
    const auto* stats = statsUid < 0 ? nullptr : findTensor(context, statsUid);
    if(stats == nullptr)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "stats_layout")
                               << "the node names stats tensor uid " << statsUid
                               << ", which is negative or names no tensor of the graph");
        return std::nullopt;
    }
    if(stats->data_type() != data_objects::DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "stats_dtype")
                               << describeOperand("stats", *stats) << " is " << stats->data_type()
                               << "; the kernel writes the LSE as FLOAT (a stats tensor whose "
                                  "data type is left unset takes the graph's io data type)");
        return std::nullopt;
    }
    if(!isServableStatsTensor(*stats, problem))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "stats_layout")
                               << describeOperand("stats", *stats)
                               << " is not the tensor the kernel writes: rank 4, dims [B, Hq, "
                                  "Sq, 1] = ["
                               << problem.batch << ", " << problem.numQueryHeads << ", "
                               << problem.seqLenQ
                               << ", 1], packed strides [Hq * Sq, Sq, 1, any] (unit-extent "
                                  "axes exempt), not virtual, ragged or pass-by-value; the "
                                  "kernel takes no stride arguments");
        return std::nullopt;
    }
    return statsUid;
}

} // namespace

/**
 * Each decline of a graph with an SDPA-forward node logs its cause at INFO (see
 * sdpa_graph::DeclinedBy); any other graph declines silently. The message is built inside
 * the logging macro, so with INFO off a decline costs the check and nothing more. Both
 * arches run the same checks in the same order, so a graph with several problems reports
 * the same first cause on either, except that a stats output is a decline only on an arch
 * that does not serve stats (gfx950), which reports it where gfx942 binds it.
 */
std::optional<BoundTokens> graphMatches(const MatchContext& context, const AttentionDenseArch& arch)
{
    // --- 1. Node shape. One SDPA-forward node; this engine serves a whole graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        if(HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED() && hasSdpaForwardNode(context))
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "node")
                                   << "the graph is not a single SDPA-forward node ("
                                   << context.graph.nodeCount() << " node(s))");
        }
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // --- 2. Operands. Q/K/V/O are the four the shipped ABI has pointers for.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "operand")
                               << "the SDPA node names a Q, K, V or O uid the graph has no "
                                  "tensor for");
        return std::nullopt;
    }

    // --- 3. Total predicates, before anything indexes an axis.
    for(const auto& [role, tensor] :
        {std::pair{"Q", q}, std::pair{"K", k}, std::pair{"V", v}, std::pair{"O", o}})
    {
        if(!isWellFormedOperand(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "operand")
                                   << describeOperand(role, *tensor)
                                   << " must be a rank-4 (B, H, S, D) tensor with positive "
                                      "extents, not virtual, not ragged and not pass-by-value");
            return std::nullopt;
        }
        if(!sdpa_graph::declaresAlignment(*tensor, OPERAND_ALIGNMENT_BYTES))
        {
            HIPDNN_PLUGIN_LOG_INFO(
                declined(arch, "alignment")
                << describeOperand(role, *tensor) << " declares " << tensor->alignment()
                << "-byte alignment, which does not guarantee the " << OPERAND_ALIGNMENT_BYTES
                << " these kernels declare on their pointer parameters");
            return std::nullopt;
        }
    }

    // --- 4. Layout. Tier 1: the failure is wrong elements in bounds, no fault.
    //
    // O is held to the same rule at §5, so that an output whose extents disagree declines
    // on the disagreement instead.
    for(const auto& [role, tensor] : {std::pair{"Q", q}, std::pair{"K", k}, std::pair{"V", v}})
    {
        if(!hasBshdStrides(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO(declined(arch, "layout") << notBshdReason(role, *tensor));
            return std::nullopt;
        }
    }

    // --- 5. Cross-tensor consistency.
    const auto problem = problemFor(*q, *k);

    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "data_type")
                               << "Q, K, V and O must share one data type, got " << problem.dataType
                               << ", " << k->data_type() << ", " << v->data_type() << ", "
                               << o->data_type());
        return std::nullopt;
    }
    if(!supportedDataTypeName(problem.dataType).has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "data_type")
                               << "data type " << problem.dataType
                               << " is not supported; the kernel is built for BFLOAT16 and HALF");
        return std::nullopt;
    }

    // V shares K's base and stride in the builder, so it must share K's shape exactly.
    if(v->dims()->Get(BATCH_AXIS) != problem.batch
       || v->dims()->Get(HEAD_AXIS) != problem.numKvHeads
       || v->dims()->Get(SEQ_AXIS) != problem.seqLenKv
       || v->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "shape")
                               << describeOperand("V", *v)
                               << " must have K's batch, heads and sequence and Q's head size; "
                               << describeOperand("K", *k));
        return std::nullopt;
    }
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "shape")
                               << describeOperand("K", *k) << " must have Q's batch and head size; "
                               << describeOperand("Q", *q));
        return std::nullopt;
    }
    // O is Q's shape: the epilogue reuses the query base and stride verbatim.
    if(o->dims()->Get(BATCH_AXIS) != problem.batch
       || o->dims()->Get(HEAD_AXIS) != problem.numQueryHeads
       || o->dims()->Get(SEQ_AXIS) != problem.seqLenQ
       || o->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "shape")
                               << describeOperand("O", *o) << " must have Q's dims; "
                               << describeOperand("Q", *q));
        return std::nullopt;
    }
    if(!hasBshdStrides(*o))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "layout") << notBshdReason("O", *o));
        return std::nullopt;
    }

    // GQA: the kernel derives its group size by integer division, so a non-divisible
    // pair silently drops heads.
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "gqa")
                               << "query heads " << problem.numQueryHeads
                               << " must be a multiple of key/value heads " << problem.numKvHeads);
        return std::nullopt;
    }

    if(!isSupportedHeadSize(problem.headSize))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "head_size")
                               << "head size " << problem.headSize
                               << " is not supported; the kernel is built for 64 and 128");
        return std::nullopt;
    }

    // --- 6. 32-bit addressing. K/V bound is bytes, Q/O is elements.
    if(!kvFitByteOffsets(problem))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "size_limit")
                               << "K or V is 2^31 bytes or larger; the kernel addresses K/V "
                                  "with 32-bit byte offsets");
        return std::nullopt;
    }
    if(!qoFitElementOffsets(problem))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "size_limit")
                               << "Q or O has 2^31 elements or more; the kernel addresses them "
                                  "with 32-bit element offsets");
        return std::nullopt;
    }

    // --- 7. The mask. hipDNN has no `causal` boolean; see maskFor, which logs its own
    // declines. The window is bound even when it is zero: kernelMatches compares it
    // against the candidate's sliding_window field, so a variant built with a window is
    // never matched by a graph that asks for a different one or none.
    const auto mask = maskFor(attributes, problem, arch);
    if(!mask.has_value())
    {
        return std::nullopt;
    }

    // --- 8. Every optional attribute this kernel cannot honour, declined explicitly.

    // Additive attention bias.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "attn_mask")
                               << "an additive attention mask (attn_mask tensor) is not "
                                  "supported");
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 kernarg.
    if(attributes.scale_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "scale_tensor")
                               << "a scale tensor is not supported; the kernel takes the scale "
                                  "as a host value (attn_scale_value)");
        return std::nullopt;
    }
    // varlen, both spellings.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "varlen")
                               << "variable sequence lengths (seq_len_q/seq_len_kv tensors) are "
                                  "not implemented in the kernel");
        return std::nullopt;
    }
    // Dropout.
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "dropout") << "dropout is not supported");
        return std::nullopt;
    }
    // Paged KV.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "paged_kv")
                               << "paged K/V (page tables or max_seq_len_kv) is not "
                                  "implemented in the kernel");
        return std::nullopt;
    }
    // Block-sparse, and attention SINKS.
    if(attributes.block_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "block_mask") << "a block mask is not supported");
        return std::nullopt;
    }
    if(attributes.sink_token_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "sinks")
                               << "attention sinks (sink_token tensor) are not yet supported "
                                  "by this pack (a follow-up)");
        return std::nullopt;
    }
    // FP8 quantization.
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "fp8")
                               << "FP8 scaling tensors (descale, scale or amax) are not "
                                  "supported");
        return std::nullopt;
    }
    // Auxiliary softmax outputs. An arch that serves stats binds the LSE output (statsFor
    // logs its own declines); any other declines every one of them. For the latter,
    // generate_stats is optional<bool> and an explicit false with no stats uid is fine.
    int64_t statsUid = ATTENTION_DENSE_NO_STATS;
    if(arch.servesStats())
    {
        const auto stats = statsFor(context, attributes, problem, arch);
        if(!stats.has_value())
        {
            return std::nullopt;
        }
        statsUid = *stats;
    }
    else if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
            || attributes.sum_exp_tensor_uid().has_value()
            || attributes.rng_dump_tensor_uid().has_value()
            || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "stats")
                               << "softmax statistics outputs (stats, max, sum_exp, rng_dump or "
                                  "generate_stats) are not supported");
        return std::nullopt;
    }
    // ALiBi slopes and padding masks.
    if(attributes.alibi_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "alibi_mask") << "alibi_mask is not supported");
        return std::nullopt;
    }
    if(attributes.padding_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "padding_mask") << "padding_mask is not supported");
        return std::nullopt;
    }
    // mma_core_mode is the MMA operand precision. This kernel's MFMA operands are the
    // graph's own fp16/bf16 inputs, so UNSET (the provider's choice), HALF and BFLOAT16
    // describe what it runs. The mode is not compared with the graph dtype: HALF is
    // accepted on bf16 graphs too, because the cuDNN-compat shim writes HALF whenever the
    // caller leaves the field unset. An allow-list, so an enum value added later declines
    // until judged.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "mma_core_mode")
                               << "mma_core_mode " << mmaCoreMode
                               << " is not supported; the kernel runs HALF or BFLOAT16 MFMA");
        return std::nullopt;
    }
    // The kernel accumulates and keeps the softmax statistics in FLOAT.
    if(!computesInFloat(context))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "compute_data_type")
                               << "compute data type "
                               << context.graph.getNodeWrapper(0).computeDataType()
                               << " is not supported; the kernel accumulates in FLOAT");
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint. AUTO leaves the choice to the provider.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "implementation")
                               << "implementation hint " << attributes.implementation()
                               << " is not supported; only AUTO is");
        return std::nullopt;
    }

    // The softmax scale is an f32 launch argument. An absent attn_scale_value means 1.0,
    // cuDNN's default: its SDPA node multiplies by attn_scale only when one is set. It is
    // resolved here, once, and prepare() launches with the bound value.
    const float scale = attributes.attn_scale_value().value_or(1.0F);
    if(!isSupportedScale(scale))
    {
        HIPDNN_PLUGIN_LOG_INFO(declined(arch, "scale")
                               << "attn_scale " << scale << " " << UNSUPPORTED_SCALE_REASON);
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(arch.tokens.q)] = attributes.q_tensor_uid();
    bound[std::string(arch.tokens.k)] = attributes.k_tensor_uid();
    bound[std::string(arch.tokens.v)] = attributes.v_tensor_uid();
    bound[std::string(arch.tokens.o)] = attributes.o_tensor_uid();
    bound[std::string(arch.tokens.causal)] = mask->causal;
    bound[std::string(arch.tokens.slidingWindow)] = mask->slidingWindow;
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(arch.tokens.scaleBits)] = static_cast<int64_t>(scaleBits);
    if(arch.servesStats())
    {
        bound[std::string(arch.tokens.stats)] = statsUid;
    }
    return bound;
}

namespace
{

/// Re-reads the bindings a match established.
AttentionDenseBinding attentionDenseBinding(const BoundTokens& bound,
                                            const AttentionDenseArch& arch)
{
    const auto read = [&bound, &arch](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string(arch.label) + " attention_dense dispatch is missing bound token '"
                    + std::string(token) + "', or it does not hold an integer");
        }
        return *value;
    };

    AttentionDenseBinding binding;
    binding.q = read(arch.tokens.q);
    binding.k = read(arch.tokens.k);
    binding.v = read(arch.tokens.v);
    binding.o = read(arch.tokens.o);
    binding.causal = read(arch.tokens.causal);
    binding.slidingWindow = read(arch.tokens.slidingWindow);

    const auto scaleBits = static_cast<int32_t>(read(arch.tokens.scaleBits));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    if(arch.servesStats())
    {
        binding.stats = read(arch.tokens.stats);
    }
    return binding;
}

bool tileDivides(const AttentionDenseTile& tile, const AttentionDenseProblem& problem)
{
    return problem.seqLenQ % tile.blockM == 0 && problem.seqLenKv % tile.blockN == 0;
}

} // namespace

std::optional<int64_t> integerMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

std::optional<std::string> stringMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

AttentionDenseGridShape gridShape(const AttentionDenseLaunchInputs& launch)
{
    AttentionDenseGridShape shape;
    shape.batch = launch.problem.batch;
    shape.seqLenQ = launch.problem.seqLenQ;
    shape.seqLenKv = launch.problem.seqLenKv;
    shape.numQueryHeads = launch.problem.numQueryHeads;
    shape.numKvHeads = launch.problem.numKvHeads;
    shape.blockM = launch.blockM;
    shape.causal = launch.binding.causal != 0;
    shape.slidingWindow = launch.binding.slidingWindow;
    return shape;
}

std::optional<AttentionDenseTile> candidateTile(const KernelDefinition& kernel,
                                                const AttentionDenseArch& arch)
{
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto blockN = integerMetadata(kernel, BLOCK_N_FIELD);
    if(!headSize.has_value() || !blockM.has_value() || !blockN.has_value()
       || !arch.isSupportedTile(*headSize, *blockM, *blockN))
    {
        return std::nullopt;
    }
    return AttentionDenseTile{*blockM, *blockN};
}

/**
 * Only shape-generic candidates are served. A ragged build bakes its exact shape and pads
 * boundary tiles on-chip; neither catalog ships one, and one arriving from elsewhere is
 * declined even at its authored shape rather than matched on metadata equality. So is a
 * split-KV build (kv_segments other than 1), whose grid and combine launch dispatch does
 * not reproduce.
 *
 * A served candidate receives batch, both sequence lengths and both head counts as
 * runtime kernel arguments, so one binary serves any valid shape for the same
 * dtype/head size/mask/window; the metadata's shape fields are canonical build inputs and
 * are not compared. An arch whose catalog lists each head configuration as its own entry
 * (matchesHeadCounts) still compares the head counts.
 *
 * Divisibility is against the candidate's OWN tile -- `Sq % block_m == 0` and
 * `Skv % block_n == 0` -- so one graph can admit some tiles of a cohort and not others.
 * So is the sliding-window zero-trip guard of an arch that serves windows, which reads the
 * same tile.
 *
 * On an arch that serves stats, LSE is baked at compile time: a candidate that writes it
 * takes a trailing lse_ptr and serves exactly the graphs that bind a stats output, and
 * one that does not serves exactly the graphs that bind none.
 *
 * On an arch that serves grids, the candidate's launch grid must be legal at the graph's
 * shape (AttentionDensePersistent.hpp gridIllegality): hq_minor_swz at a head count that
 * is not a multiple of the XCD count, or a persistent decode past its fast-division bound,
 * is declined so another grid serves the graph.
 */
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel,
                   const AttentionDenseArch& arch)
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

    // Before any comparison that could divide by it.
    const auto tile = candidateTile(kernel, arch);
    if(!tile.has_value())
    {
        return false;
    }

    const auto ragged = integerMetadata(kernel, RAGGED_FIELD);
    if(!ragged.has_value() || *ragged != 0)
    {
        return false;
    }

    const auto kvSegments = kernel.metadata.find(std::string(KV_SEGMENTS_FIELD));
    if(kvSegments != kernel.metadata.end())
    {
        const auto* segments = std::get_if<int64_t>(&kvSegments->second);
        if(segments == nullptr || *segments != 1)
        {
            return false;
        }
    }

    const auto dataTypeName = supportedDataTypeName(problem.dataType);
    if(!dataTypeName.has_value()
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dataTypeName)
    {
        return false;
    }

    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };

    if(intField(HEAD_SIZE_FIELD) != problem.headSize)
    {
        return false;
    }
    if(arch.matchesHeadCounts
       && (intField(NUM_QUERY_HEADS_FIELD) != problem.numQueryHeads
           || intField(NUM_KV_HEADS_FIELD) != problem.numKvHeads))
    {
        return false;
    }

    const auto causal = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, arch.tokens.causal);
    if(!causal.has_value() || intField(CAUSAL_FIELD) != *causal)
    {
        return false;
    }

    // Sliding window: the KV-loop bound (and, where served, the window mask) is baked at
    // compile time.
    const auto slidingWindow
        = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, arch.tokens.slidingWindow);
    if(!slidingWindow.has_value() || intField(SLIDING_WINDOW_FIELD) != *slidingWindow)
    {
        return false;
    }

    if(arch.servesStats())
    {
        const auto stats = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, arch.tokens.stats);
        const auto writesLse = arch.candidateWritesLse(kernel);
        if(!stats.has_value() || !writesLse.has_value()
           || *writesLse != (*stats != ATTENTION_DENSE_NO_STATS))
        {
            return false;
        }
    }

    if(!tileDivides(*tile, problem)
       || (arch.servesSlidingWindow()
           && !arch.windowReachesKeys(
               tile->blockM, tile->blockN, problem.seqLenQ, problem.seqLenKv, *slidingWindow)))
    {
        return false;
    }

    if(arch.servesGrids())
    {
        const auto grid = arch.candidateGrid(kernel);
        if(!grid.has_value())
        {
            return false;
        }
        AttentionDenseLaunchInputs launch;
        launch.problem = problem;
        launch.blockM = tile->blockM;
        launch.binding.causal = *causal;
        launch.binding.slidingWindow = *slidingWindow;
        if(gridIllegality(*grid, gridShape(launch)).has_value())
        {
            return false;
        }
    }
    return true;
}

namespace
{

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The saved stats uid as restore's selector reads it, before any value is checked. An
/// absent or non-integer value selects the argument list without lse_ptr; the reader
/// refuses it afterwards.
int64_t savedStatsUidSelector(const SavedLaunchInputs& inputs)
{
    const auto found = inputs.values.find(std::string(STATS_UID_VALUE));
    const auto* uid = found == inputs.values.end() ? nullptr : std::get_if<int64_t>(&found->second);
    return uid == nullptr ? ATTENTION_DENSE_NO_STATS : *uid;
}

/// The saved grid as restore's selector reads it, before any value is checked: the
/// persistent grid at the saved decode when `grid_order` is `persistent` and
/// `persist_decode` names a runtime decode, else the qb_minor grid (every non-persistent
/// order takes the same arguments). Anything malformed is refused by the reader afterwards.
AttentionDenseGrid savedGridSelector(const SavedLaunchInputs& inputs)
{
    const auto savedString = [&inputs](std::string_view name) -> std::string {
        const auto found = inputs.values.find(std::string(name));
        const auto* value
            = found == inputs.values.end() ? nullptr : std::get_if<std::string>(&found->second);
        return value == nullptr ? std::string() : *value;
    };
    AttentionDenseGrid grid;
    const auto decode = parsePersistDecode(savedString(PERSIST_DECODE_VALUE));
    if(parseGridOrder(savedString(GRID_ORDER_VALUE)) == AttentionDenseGridOrder::PERSISTENT
       && decode.has_value())
    {
        grid.order = AttentionDenseGridOrder::PERSISTENT;
        grid.decode = *decode;
    }
    return grid;
}

/// The kernel signature a contract's launch passes, for a kernel on @p grid that writes LSE
/// (@p writesLse) or not.
///
/// RUNTIME_HEAD_COUNTS is the form every kernel in both catalogs takes; BAKED_HEAD_COUNTS is
/// the 8-argument form of kernels built before the head counts became runtime arguments. A
/// kernel on the persistent grid (gfx942 only) takes a `<d>_magic`, `<d>_shift` i32 pair per
/// fast-division divisor of its decode after the shape (runtime_kernarg_fields), 14 or 16
/// arguments; a non-persistent order takes none. An emit_lse kernel (gfx942 only) takes one
/// more argument, lse_ptr, last. use_sinks=False everywhere, so there is no sink_ptr slot.
///
/// NAMES ARE LOAD-BEARING. requireSignatureMatch compares kind and size always, but names
/// only when BOTH sides carry one. Kind and size alone cannot tell the four pointers
/// apart, nor `scale` (f32) from `batch` (i32) -- both are by_value/4 -- so without names
/// an operand permutation passes the check and the kernel reads the wrong buffers with no
/// error and no status code. hkp_pack lowers the recorded names out of the rocKE builder's
/// own parameter list, so keep these spellings identical to the Python
/// (kernels/gfx9xx/attention_dense.py, the attention_dense_signature parameter list); a
/// divergence here fails every dispatch rather than silently weakening the check.
///
/// Offsets mirror the packed kernarg layout. They are not compared -- they exist so the
/// mismatch diagnostic prints the real layout beside the recorded one instead of ten
/// zeroes that read as data.
std::vector<KernelArgument>
    kernelSignature(AttentionDenseAbi abi, const AttentionDenseGrid& grid, bool writesLse)
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));

    std::vector<KernelArgument> signature{KernelArgument{"global_buffer", PTR, 0, "q_ptr"},
                                          KernelArgument{"global_buffer", PTR, 8, "k_ptr"},
                                          KernelArgument{"global_buffer", PTR, 16, "v_ptr"},
                                          KernelArgument{"global_buffer", PTR, 24, "o_ptr"},
                                          KernelArgument{"by_value", F32, 32, "scale"},
                                          KernelArgument{"by_value", I32, 36, "batch"},
                                          KernelArgument{"by_value", I32, 40, "seqlen_q"},
                                          KernelArgument{"by_value", I32, 44, "seqlen_kv"}};
    if(abi == AttentionDenseAbi::RUNTIME_HEAD_COUNTS)
    {
        signature.push_back(KernelArgument{"by_value", I32, 48, "num_query_heads"});
        signature.push_back(KernelArgument{"by_value", I32, 52, "num_kv_heads"});
    }
    if(grid.persistent())
    {
        for(const auto divisor : fastDivDivisors(grid.decode))
        {
            for(const char* part : {"_magic", "_shift"})
            {
                const uint32_t end = signature.back().offset + signature.back().size;
                signature.push_back(
                    KernelArgument{"by_value", I32, end, std::string(divisorName(divisor)) + part});
            }
        }
    }
    if(writesLse)
    {
        // At the next pointer-aligned offset: 56 after num_kv_heads, 72 or 80 after two or
        // three fast-division pairs.
        const uint32_t end = signature.back().offset + signature.back().size;
        signature.push_back(
            KernelArgument{"global_buffer", PTR, (end + PTR - 1) / PTR * PTR, "lse_ptr"});
    }
    return signature;
}

/// The compiled kernel plus everything launch() needs, owning nothing that points back
/// into the MatchContext or BoundTokens it came from.
class PreparedAttentionDense : public IngestorPreparedDispatch
{
public:
    PreparedAttentionDense(IngestorKernelCode code,
                           AttentionDenseLaunchInputs launch,
                           AttentionDenseContract contract)
        : _code(std::move(code))
        , _launch(launch)
        , _contract(contract)
        , _fastDiv(fastDivKernargs(launch.grid, gridShape(launch)))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const AttentionDenseLaunchInputs& launchInputs() const
    {
        return _launch;
    }

    /// The contract the code object was checked against, so launch() passes the kernel
    /// arguments that code object takes and a save stores the plan under its name.
    const AttentionDenseContract& contract() const
    {
        return _contract;
    }

    /// The persistent grid's fast-division kernargs, computed from the launch shape; empty
    /// off it.
    const std::vector<AttentionDenseFastDivPair>& fastDiv() const
    {
        return _fastDiv;
    }

private:
    IngestorKernelCode _code;
    AttentionDenseLaunchInputs _launch;
    AttentionDenseContract _contract;
    std::vector<AttentionDenseFastDivPair> _fastDiv;
};

// Applies the launch geometry to `code` and builds the prepared dispatch. A plan built
// from a graph and a plan restored from a saved payload both come through here.
// `kernelName` is only for the diagnostic.
//
// Grid from the SELECTED CANDIDATE'S block_m and launch grid and the GRAPH PROBLEM, not
// from descriptor shape metadata: that carries canonical build inputs (B=1, Sq=Skv=512),
// not runtime constraints. block_n does not enter the launch. The arch's geometry validates
// block_m and sizes the CTA and the qb_minor grid; another order is sized by
// AttentionDensePersistent.hpp, the persistent grid being exactly num_persistent CTAs.
std::unique_ptr<PreparedDispatch>
    makePreparedAttentionDense(IngestorKernelCode code,
                               const AttentionDenseLaunchInputs& launch,
                               const std::string& kernelName,
                               const AttentionDenseArch& arch,
                               const AttentionDenseContract& contract)
{
    const auto& problem = launch.problem;
    const auto geometry = arch.geometry(
        launch.blockM, problem.seqLenQ, problem.numQueryHeads, problem.batch, kernelName);

    code.setBlockSize(geometry.blockX, 1, 1);
    if(launch.grid.order == AttentionDenseGridOrder::QB_MINOR)
    {
        code.setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);
    }
    else
    {
        const auto shape = gridShape(launch);
        const auto illegal = gridIllegality(launch.grid, shape);
        if(illegal.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string(arch.label) + " attention_dense: kernel '" + kernelName
                    + "' cannot launch its " + std::string(gridOrderName(launch.grid.order))
                    + " grid over this shape: " + *illegal);
        }
        const auto dims = gridDims(launch.grid, shape);
        for(const int64_t dim : dims)
        {
            if(dim <= 0 || dim > std::numeric_limits<int32_t>::max())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string(arch.label) + " attention_dense: kernel '" + kernelName
                        + "' has a grid dimension of " + std::to_string(dim) + " over this shape");
            }
        }
        code.setGridSize(static_cast<unsigned>(dims[0]),
                         static_cast<unsigned>(dims[1]),
                         static_cast<unsigned>(dims[2]));
    }

    return std::make_unique<PreparedAttentionDense>(std::move(code), launch, contract);
}

class AttentionDenseDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    AttentionDenseDispatchHandler(const compilation::KpackKernelLoader& kpackLoader,
                                  const AttentionDenseArch& arch,
                                  const AttentionDenseContract& contract)
        : _kpackLoader(kpackLoader)
        , _arch(arch)
        , _contract(contract)
    {
    }

    /// Zero: the kernel's only scratch is LDS and registers; no global scratch, and
    /// neither ABI has a workspace pointer.
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
        const auto binding = attentionDenseBinding(bound, _arch);

        // graph_match is the gate for this: a non-BSHD output declines there and never
        // reaches prepare(). This re-check is defence in depth for a caller that reaches
        // the handler without having matched, which is why it faults rather than declines
        // -- by this point the engine has been chosen and there is no one left to defer to.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string(_arch.label)
                    + " attention_dense: the output tensor is not dense BSHD; the kernel "
                      "bakes that layout and takes no stride arguments");
        }

        // kernel_match declines a candidate without a supported tile, so this is the same
        // defence in depth: the grid and CTA below come from block_m, and there is no
        // tile to substitute that would not launch this binary with another's geometry.
        const auto tile = candidateTile(kernel, _arch);
        if(!tile.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string(_arch.label) + " attention_dense: kernel '" + toString(kernel.kernelId)
                    + "' declares no supported block_m/block_n tile");
        }

        // kernel_match declines a candidate without a grid the arch launches; the same
        // defence in depth, since the grid decides the CTA count and the kernel arguments.
        AttentionDenseGrid grid;
        if(_arch.servesGrids())
        {
            const auto candidateGrid = _arch.candidateGrid(kernel);
            if(!candidateGrid.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string(_arch.label) + " attention_dense: kernel '"
                        + toString(kernel.kernelId)
                        + "' declares no launch grid the engine "
                          "launches");
            }
            grid = *candidateGrid;
        }

        const auto* q = findTensor(context, binding.q);
        const auto* k = findTensor(context, binding.k);
        const auto problem = problemFor(*q, *k);

        // The same defence in depth for the stats output: graph_match binds only one the
        // kernel can write, and the kernel takes no stride arguments for it.
        const bool writesLse = binding.stats != ATTENTION_DENSE_NO_STATS;
        if(writesLse)
        {
            const auto* stats = findTensor(context, binding.stats);
            if(stats == nullptr || stats->data_type() != data_objects::DataType::FLOAT
               || !isServableStatsTensor(*stats, problem))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string(_arch.label)
                        + " attention_dense: the stats tensor is not the packed [B, Hq, Sq, 1] "
                          "FLOAT tensor the kernel writes the LSE to");
            }
        }

        auto code = buildIngestorKernelCode(
            _kpackLoader, context, kernel, kernelSignature(_contract.abi, grid, writesLse));

        return makePreparedAttentionDense(
            std::move(code),
            AttentionDenseLaunchInputs{binding, problem, tile->blockM, grid},
            toString(kernel.kernelId),
            _arch,
            _contract);
    }

    std::unique_ptr<PreparedDispatch> restoreLaunch(const SavedLaunchInputs& inputs,
                                                    SavedKernelCode code,
                                                    int deviceOrdinal) const override
    {
        const std::string kernelName = toString(code.kernelId);
        // The saved code object must take this contract's arguments (checked first, inside
        // restoreIngestorLaunch), and its values must be saved under this contract's name.
        const auto readThisContract = [this](const SavedLaunchInputs& saved) {
            requireLaunchContract(saved, _contract.symbol);
            return readLaunchInputs(saved, _arch);
        };
        // Which of the contract's argument lists the saved kernel must take is selected by
        // the saved stats uid and grid, read before restoreIngestorLaunch checks anything. A
        // kernel whose arguments disagree with them -- an LSE kernel saved without a stats
        // output, a persistent kernel saved under another decode or as a grid, or the
        // reverse -- is then refused by the signature check.
        const bool writesLse
            = _arch.servesStats() && savedStatsUidSelector(inputs) != ATTENTION_DENSE_NO_STATS;
        const AttentionDenseGrid grid
            = _arch.servesGrids() ? savedGridSelector(inputs) : AttentionDenseGrid{};
        auto restored = restoreIngestorLaunch<AttentionDenseLaunchInputs>(
            inputs,
            std::move(code),
            kernelSignature(_contract.abi, grid, writesLse),
            readThisContract,
            deviceOrdinal);
        return makePreparedAttentionDense(
            std::move(restored.code), restored.launch, kernelName, _arch, _contract);
    }

    std::optional<SavedLaunchInputs>
        saveLaunchInputs(const PreparedDispatch& prepared) const override
    {
        const auto& preparedDense = dynamic_cast<const PreparedAttentionDense&>(prepared);
        const auto& launch = preparedDense.launchInputs();
        return SavedLaunchInputs{
            std::string(preparedDense.contract().symbol),
            launchValues(launch.binding, launch.problem, launch.blockM, _arch, launch.grid)};
    }

    bool supportsSaving() const override
    {
        return true;
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedDense = dynamic_cast<const PreparedAttentionDense&>(prepared);
        const auto& binding = preparedDense.launchInputs().binding;

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);

        const auto& launchInputs = preparedDense.launchInputs();
        const auto& p = launchInputs.problem;
        auto& kernel = preparedDense.kernelForStream(handle.getStream());

        // Exactly the arguments the code object was checked against (kernelSignature): a
        // BAKED_HEAD_COUNTS code object bakes the head counts, a RUNTIME_HEAD_COUNTS one
        // reads them after seqlen_kv, a persistent one then takes its fast-division pairs,
        // and one that writes LSE takes lse_ptr last. Only gfx950 has a BAKED_HEAD_COUNTS
        // contract, and it binds no stats output and launches no persistent grid.
        std::array<void*, 4> pointers{q.ptr, k.ptr, v.ptr, o.ptr};
        float scale = binding.scale;
        std::array<int32_t, 5> shape{static_cast<int32_t>(p.batch),
                                     static_cast<int32_t>(p.seqLenQ),
                                     static_cast<int32_t>(p.seqLenKv),
                                     static_cast<int32_t>(p.numQueryHeads),
                                     static_cast<int32_t>(p.numKvHeads)};
        auto fastDiv = preparedDense.fastDiv();
        const bool writesLse = binding.stats != ATTENTION_DENSE_NO_STATS;
        void* lse = writesLse ? hipdnn_plugin_sdk::findDeviceBuffer(
                                    binding.stats, deviceBuffers, numDeviceBuffers)
                                    .ptr
                              : nullptr;

        const bool headCounts
            = preparedDense.contract().abi == AttentionDenseAbi::RUNTIME_HEAD_COUNTS;
        std::vector<void*> params;
        params.reserve(pointers.size() + 1 + shape.size() + 2 * fastDiv.size() + 1);
        for(auto& pointer : pointers)
        {
            params.push_back(static_cast<void*>(&pointer));
        }
        params.push_back(&scale);
        for(size_t index = 0; index < (headCounts ? shape.size() : 3); ++index)
        {
            params.push_back(&shape.at(index));
        }
        for(auto& pair : fastDiv)
        {
            params.push_back(&pair.magic);
            params.push_back(&pair.shift);
        }
        if(writesLse)
        {
            params.push_back(static_cast<void*>(&lse));
        }
        // The count is the signature's by construction; checked so a skew can never launch.
        const auto expected
            = kernelSignature(preparedDense.contract().abi, launchInputs.grid, writesLse).size();
        if(params.size() != expected)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string(_arch.label) + " attention_dense: the launch packs "
                    + std::to_string(params.size()) + " kernel arguments, the signature "
                    + std::to_string(expected));
        }
        kernel.launchWithParams(handle.getStream(), params.data());
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
    const AttentionDenseArch& _arch;
    AttentionDenseContract _contract;
};

// Refuses a saved launch value of the dense-attention contract.
[[noreturn]] void refuseAttentionDenseLaunchValue(const SavedLaunchInputs& inputs,
                                                  std::string_view name,
                                                  const std::string& reason)
{
    serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                      "launch value '" + std::string(name) + "' of dispatch '"
                                          + inputs.dispatchSymbol + "' " + reason);
}

// Refuses a saved causal flag or window outside maskFor's rules, without asking whether the
// arch serves a window (see readLaunchInputs for gfx950's exception). Both are baked into the
// code object and the launch does not read them, so these are checks of the plan's integrity,
// not of the launch.
void requireServedSavedMask(const SavedLaunchInputs& inputs,
                            const AttentionDenseLaunchInputs& launch)
{
    const auto& binding = launch.binding;
    const auto& problem = launch.problem;
    if(binding.causal != 0 && binding.causal != 1)
    {
        refuseAttentionDenseLaunchValue(inputs,
                                        CAUSAL_VALUE,
                                        "must be 0 or 1; the plan holds "
                                            + std::to_string(binding.causal));
    }
    if(binding.slidingWindow < 0)
    {
        refuseAttentionDenseLaunchValue(inputs,
                                        SLIDING_WINDOW_VALUE,
                                        "must not be negative; the plan holds "
                                            + std::to_string(binding.slidingWindow));
    }
    if(binding.slidingWindow > 0 && binding.causal != 1)
    {
        refuseAttentionDenseLaunchValue(
            inputs,
            SLIDING_WINDOW_VALUE,
            "must be 0 unless 'causal' is 1; the kernel's window is causal");
    }
    if(windowLeavesRowsWithoutKeys(problem.seqLenQ, problem.seqLenKv, binding.slidingWindow))
    {
        refuseAttentionDenseLaunchValue(inputs,
                                        SLIDING_WINDOW_VALUE,
                                        "leaves the last query rows no key over seqlen_q "
                                            + std::to_string(problem.seqLenQ) + " and seqlen_kv "
                                            + std::to_string(problem.seqLenKv)
                                            + "; not yet supported by this pack (only the "
                                              "kernel's LSE path zeroes such rows)");
    }
}

// Refuses a saved shape graphMatches or kernelMatches never binds, in their order. Each
// would launch out of bounds, except head_size: the launch does not read it, but the size
// limits are computed from it. block_m must already be a tile the arch builds.
void requireServedSavedShape(const SavedLaunchInputs& inputs,
                             const AttentionDenseLaunchInputs& launch)
{
    const auto& problem = launch.problem;
    if(problem.numQueryHeads % problem.numKvHeads != 0)
    {
        refuseAttentionDenseLaunchValue(
            inputs,
            NUM_QUERY_HEADS_VALUE,
            "must be a multiple of 'num_kv_heads' " + std::to_string(problem.numKvHeads)
                + "; the plan holds " + std::to_string(problem.numQueryHeads)
                + ", and the kernel derives the kv head by integer division");
    }
    if(!isSupportedHeadSize(problem.headSize))
    {
        refuseAttentionDenseLaunchValue(inputs,
                                        HEAD_SIZE_VALUE,
                                        "must be 64 or 128; the plan holds "
                                            + std::to_string(problem.headSize));
    }
    if(!kvFitByteOffsets(problem))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values 'batch', 'seqlen_kv', 'num_kv_heads' and 'head_size' of dispatch '"
                + inputs.dispatchSymbol
                + "' give K and V of 2^31 bytes or more; the kernel addresses them with 32-bit "
                  "byte offsets");
    }
    if(!qoFitElementOffsets(problem))
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values 'batch', 'seqlen_q', 'num_query_heads' and 'head_size' of dispatch '"
                + inputs.dispatchSymbol
                + "' give Q and O of 2^31 elements or more; the kernel addresses them with "
                  "32-bit element offsets");
    }
    if(problem.seqLenQ % launch.blockM != 0)
    {
        refuseAttentionDenseLaunchValue(
            inputs,
            SEQLEN_Q_VALUE,
            "must be a multiple of 'block_m' " + std::to_string(launch.blockM) + "; the plan holds "
                + std::to_string(problem.seqLenQ)
                + ", and the kernel reads Q and writes O without a bounds check");
    }
}

// Reads the saved grid of an arch that serves grids, and refuses one the kernel match never
// admits at the saved shape. The shape must already have been validated.
AttentionDenseGrid readSavedGrid(const SavedLaunchInputs& inputs,
                                 const AttentionDenseLaunchInputs& launch)
{
    constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();
    const std::string orderName = requireLaunchString(inputs, GRID_ORDER_VALUE);
    const auto order = parseGridOrder(orderName);
    if(!order.has_value())
    {
        refuseAttentionDenseLaunchValue(inputs,
                                        GRID_ORDER_VALUE,
                                        "must be qb_minor, bt_hkv_minor, hq_minor_swz or "
                                        "persistent; the plan holds '"
                                            + orderName + "'");
    }
    AttentionDenseGrid grid;
    grid.order = *order;
    const std::string decodeName = requireLaunchString(inputs, PERSIST_DECODE_VALUE);
    if(grid.persistent())
    {
        const auto decode = parsePersistDecode(decodeName);
        if(!decode.has_value())
        {
            refuseAttentionDenseLaunchValue(
                inputs,
                PERSIST_DECODE_VALUE,
                "must name a decode the persistent kernel reads the shape for at run time "
                "(qb_major, hkv_major, bt_hkv_minor or hq_minor_swz); the plan holds '"
                    + decodeName + "'");
        }
        grid.decode = *decode;
        grid.numPersistent = requireLaunchInt(inputs, NUM_PERSISTENT_VALUE, 1, INT32_LIMIT);
    }
    else
    {
        if(!decodeName.empty())
        {
            refuseAttentionDenseLaunchValue(inputs,
                                            PERSIST_DECODE_VALUE,
                                            "must be empty off the persistent grid; the plan "
                                            "holds '"
                                                + decodeName + "'");
        }
        static_cast<void>(requireLaunchInt(inputs, NUM_PERSISTENT_VALUE, 0, 0));
    }
    AttentionDenseLaunchInputs withGrid = launch;
    withGrid.grid = grid;
    const auto illegal = gridIllegality(grid, gridShape(withGrid));
    if(illegal.has_value())
    {
        refuseAttentionDenseLaunchValue(inputs, GRID_ORDER_VALUE, "cannot launch: " + *illegal);
    }
    return grid;
}

} // namespace

std::unique_ptr<IKernelDispatchHandler<Handle>>
    makeDispatchHandler(const compilation::KpackKernelLoader& kpackLoader,
                        const AttentionDenseArch& arch,
                        const AttentionDenseContract& contract)
{
    return std::make_unique<AttentionDenseDispatchHandler>(kpackLoader, arch, contract);
}

MetadataValues launchValues(const AttentionDenseBinding& binding,
                            const AttentionDenseProblem& problem,
                            int64_t blockM,
                            const AttentionDenseArch& arch,
                            const AttentionDenseGrid& grid)
{
    MetadataValues values{{std::string(Q_UID_VALUE), binding.q},
                          {std::string(K_UID_VALUE), binding.k},
                          {std::string(V_UID_VALUE), binding.v},
                          {std::string(O_UID_VALUE), binding.o},
                          {std::string(CAUSAL_VALUE), binding.causal},
                          {std::string(SLIDING_WINDOW_VALUE), binding.slidingWindow},
                          {std::string(BATCH_VALUE), problem.batch},
                          {std::string(SEQLEN_Q_VALUE), problem.seqLenQ},
                          {std::string(SEQLEN_KV_VALUE), problem.seqLenKv},
                          {std::string(NUM_QUERY_HEADS_VALUE), problem.numQueryHeads},
                          {std::string(NUM_KV_HEADS_VALUE), problem.numKvHeads},
                          {std::string(HEAD_SIZE_VALUE), problem.headSize},
                          {std::string(BLOCK_M_VALUE), blockM},
                          {std::string(SCALE_VALUE), static_cast<double>(binding.scale)},
                          {std::string(DATA_TYPE_VALUE),
                           std::string(data_objects::EnumNameDataType(problem.dataType))},
                          {std::string(STRIDE_LAYOUT_VALUE), std::string(BSHD_LAYOUT)}};
    if(arch.servesStats())
    {
        values.emplace(std::string(STATS_UID_VALUE), binding.stats);
    }
    if(arch.servesGrids())
    {
        values.emplace(std::string(GRID_ORDER_VALUE), std::string(gridOrderName(grid.order)));
        values.emplace(std::string(PERSIST_DECODE_VALUE),
                       grid.persistent() ? std::string(persistDecodeName(grid.decode))
                                         : std::string());
        values.emplace(std::string(NUM_PERSISTENT_VALUE),
                       grid.persistent() ? grid.numPersistent : int64_t{0});
    }
    return values;
}

AttentionDenseLaunchInputs readLaunchInputs(const SavedLaunchInputs& inputs,
                                            const AttentionDenseArch& arch)
{
    // Every contract of an arch shares this value set; anything else, including the
    // unversioned name, is refused against the current contract.
    const bool legacy = !arch.legacyContract.symbol.empty()
                        && inputs.dispatchSymbol == arch.legacyContract.symbol;
    requireLaunchContract(inputs,
                          legacy ? arch.legacyContract.symbol : arch.currentContract.symbol);
    // The sixteen values both arches share; an arch that serves stats also saves the
    // stats uid.
    std::vector<std::string_view> names{Q_UID_VALUE,
                                        K_UID_VALUE,
                                        V_UID_VALUE,
                                        O_UID_VALUE,
                                        CAUSAL_VALUE,
                                        SLIDING_WINDOW_VALUE,
                                        BATCH_VALUE,
                                        SEQLEN_Q_VALUE,
                                        SEQLEN_KV_VALUE,
                                        NUM_QUERY_HEADS_VALUE,
                                        NUM_KV_HEADS_VALUE,
                                        HEAD_SIZE_VALUE,
                                        BLOCK_M_VALUE,
                                        SCALE_VALUE,
                                        DATA_TYPE_VALUE,
                                        STRIDE_LAYOUT_VALUE};
    if(arch.servesStats())
    {
        names.push_back(STATS_UID_VALUE);
    }
    if(arch.servesGrids())
    {
        names.push_back(GRID_ORDER_VALUE);
        names.push_back(PERSIST_DECODE_VALUE);
        names.push_back(NUM_PERSISTENT_VALUE);
    }
    requireOnlyLaunchValues(inputs, names);

    constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();

    AttentionDenseLaunchInputs launch;
    auto& binding = launch.binding;
    auto& problem = launch.problem;
    binding.q = requireLaunchInt(inputs, Q_UID_VALUE);
    binding.k = requireLaunchInt(inputs, K_UID_VALUE);
    binding.v = requireLaunchInt(inputs, V_UID_VALUE);
    binding.o = requireLaunchInt(inputs, O_UID_VALUE);
    binding.causal = requireLaunchInt(inputs, CAUSAL_VALUE);
    binding.slidingWindow = requireLaunchInt(inputs, SLIDING_WINDOW_VALUE);
    // A uid, or ATTENTION_DENSE_NO_STATS: graph_match binds no negative uid.
    if(arch.servesStats())
    {
        binding.stats = requireLaunchInt(inputs, STATS_UID_VALUE, ATTENTION_DENSE_NO_STATS);
    }

    // The kernel takes these three as int32_t arguments.
    problem.batch = requireLaunchInt(inputs, BATCH_VALUE, 1, INT32_LIMIT);
    problem.seqLenQ = requireLaunchInt(inputs, SEQLEN_Q_VALUE, 1, INT32_LIMIT);
    problem.seqLenKv = requireLaunchInt(inputs, SEQLEN_KV_VALUE, 1, INT32_LIMIT);
    // num_query_heads is a grid dimension.
    problem.numQueryHeads = requireLaunchInt(inputs, NUM_QUERY_HEADS_VALUE, 1, INT32_LIMIT);
    problem.numKvHeads = requireLaunchInt(inputs, NUM_KV_HEADS_VALUE, 1, INT32_LIMIT);
    problem.headSize = requireLaunchInt(inputs, HEAD_SIZE_VALUE, 1, INT32_LIMIT);
    launch.blockM = requireLaunchInt(inputs, BLOCK_M_VALUE);

    requireServedSavedMask(inputs, launch);

    // The kernel takes the scale as a float. A double that a float cannot hold exactly was
    // not written from one.
    const double scale = requireLaunchDouble(inputs, SCALE_VALUE);
    if(!std::isfinite(scale)
       || std::fabs(scale) > static_cast<double>(std::numeric_limits<float>::max())
       || static_cast<double>(static_cast<float>(scale)) != scale)
    {
        refuseAttentionDenseLaunchValue(
            inputs, SCALE_VALUE, "must be a finite double that a float holds exactly");
    }
    binding.scale = static_cast<float>(scale);
    if(!isSupportedScale(binding.scale))
    {
        refuseAttentionDenseLaunchValue(inputs, SCALE_VALUE, std::string(UNSUPPORTED_SCALE_REASON));
    }

    const std::string dataTypeName = requireLaunchString(inputs, DATA_TYPE_VALUE);
    bool dataTypeFound = false;
    for(const auto dataType : data_objects::EnumValuesDataType())
    {
        if(dataTypeName == data_objects::EnumNameDataType(dataType))
        {
            problem.dataType = dataType;
            dataTypeFound = true;
            break;
        }
    }
    if(!dataTypeFound || !supportedDataTypeName(problem.dataType).has_value())
    {
        refuseAttentionDenseLaunchValue(
            inputs,
            DATA_TYPE_VALUE,
            "must name a data type the kernel is built for (BFLOAT16 or HALF); the plan "
            "holds '"
                + dataTypeName + "'");
    }

    // The kernel takes no stride arguments and reads every operand as dense BSHD.
    const std::string strideLayout = requireLaunchString(inputs, STRIDE_LAYOUT_VALUE);
    if(strideLayout != BSHD_LAYOUT)
    {
        refuseAttentionDenseLaunchValue(inputs,
                                        STRIDE_LAYOUT_VALUE,
                                        "must be '" + std::string(BSHD_LAYOUT)
                                            + "'; the plan holds '" + strideLayout + "'");
    }

    try
    {
        static_cast<void>(arch.geometry(
            launch.blockM, problem.seqLenQ, problem.numQueryHeads, problem.batch, "saved plan"));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        if(error.getStatus() != HIPDNN_PLUGIN_STATUS_BAD_PARAM)
        {
            throw;
        }
        refuseAttentionDenseLaunchValue(
            inputs, BLOCK_M_VALUE, "is not a tile the kernel is built with: " + error.getMessage());
    }

    requireServedSavedShape(inputs, launch);
    if(arch.servesGrids())
    {
        launch.grid = readSavedGrid(inputs, launch);
    }
    return launch;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::attention_dense

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
