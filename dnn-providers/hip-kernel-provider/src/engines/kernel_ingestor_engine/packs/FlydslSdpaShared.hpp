// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
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

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/**
 * @file FlydslSdpaShared.hpp
 * @brief The graph contract every FlyDSL SDPA engine applies identically, RDNA and CDNA:
 *        operand lookup, the node and compute-type checks, the softmax-scale tensor, output
 *        aliasing, the dtype vocabulary, metadata reads, the attention sink's stride and the
 *        mask's bounds.
 *
 * FlydslSdpaRdnaCommon.hpp and FlydslSdpaCdnaCommon.hpp each pull this namespace into their
 * own, so a rule both families must apply lives here once and cannot drift between them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// hipDNN SDPA tensors are rank 4 in LOGICAL order (B, H, S, D) whatever the layout.
inline constexpr uint32_t SDPA_RANK = 4;
inline constexpr uint32_t BATCH_AXIS = 0;
inline constexpr uint32_t HEAD_AXIS = 1;
inline constexpr uint32_t SEQ_AXIS = 2;
inline constexpr uint32_t HEAD_DIM_AXIS = 3;

inline constexpr int64_t INT32_LIMIT = 2147483647LL;
inline constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and f16 only

/// A bound the graph leaves open (hipDNN's sentinel for left_bound / right_bound).
inline constexpr int64_t UNBOUNDED = -1;

/// The stats, bias and sink tokens' value when the graph has no such tensor.
inline constexpr int64_t NO_STATS = -1;
inline constexpr int64_t NO_BIAS = -1;
inline constexpr int64_t NO_SINK = -1;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

inline const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node the SDPA matchers read, or nullptr if the graph isn't a single SDPA-forward node.
inline const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
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

/// Whether @p tensor promises a base pointer aligned to @p bytes. The backend enforces the
/// alignment a graph declares and no more, so a kernel that assumes a wider one must read the
/// declaration: a smaller one would hand it a pointer its vector accesses cannot take.
inline bool declaresAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// Whether the kernels' arithmetic meets the node's compute type. They compute the softmax,
/// accumulate and keep the statistics in f32: at least the precision FLOAT, HALF or BFLOAT16
/// asks for, and UNSET leaves it to the engine. DOUBLE asks for more, and an integer or FP8
/// compute type for a different computation, so those decline.
inline bool computesInFloat(const MatchContext& context)
{
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    return computeType == data_objects::DataType::UNSET
           || computeType == data_objects::DataType::FLOAT
           || computeType == data_objects::DataType::HALF
           || computeType == data_objects::DataType::BFLOAT16;
}

/**
 * @brief Whether the graph uses a schema field no SDPA matcher reads where it reads a tensor,
 *        each declined: overridable shapes (graph schema 1.1.0), whose execute-time shapes no
 *        matcher has seen; and a ragged offset multiplier (1.4.0) on a tensor with no ragged
 *        offset, which no dense tensor can mean.
 *
 * The engines claim every graph schema through 1.4.0 (their UEDs' `sdk_version`), so every
 * field up to it is gated: these two here, and the rest where each tensor is read --
 * pass-by-value (1.2.0), ragged offsets and declared alignments (1.3.0), and a ragged
 * tensor's multiplier (1.4.0, isPackedThrough).
 */
inline bool usesUnreadSchemaFields(const MatchContext& context)
{
    if(context.graph.getGraph().is_override_shape_enabled())
    {
        return true;
    }
    for(const auto& entry : context.graph.getTensorMap())
    {
        const auto* tensor = entry.second;
        if(tensor != nullptr && !tensor->ragged_offset_tensor_uid().has_value()
           && tensor->ragged_offset_multiplier() != 1)
        {
            return true;
        }
    }
    return false;
}

/// The largest softmax scale magnitude every FlyDSL SDPA kernel serves. Real graphs use
/// 1/sqrt(d) or a temperature near 1; far above this each family overflows f32 at a different
/// point (the CDNA causal objects' running-max floor times scale * log2(e) passes FLT_MAX near
/// 2.4e8), so one bound here makes all of them decline alike rather than diverge.
inline constexpr double MAX_SOFTMAX_SCALE = 1048576.0; // 2^20

/// Whether @p value is a softmax scale the engines take: finite and at most MAX_SOFTMAX_SCALE
/// in magnitude, zero and negative included (NaN and infinity fail), as PyTorch and AOTriton
/// take any scale. The objects that keep their running max over unscaled scores serve only
/// rawScoresServe's; the others run on the scaled-score ones (needsScaledScores).
inline bool isServableScale(double value)
{
    return value >= -MAX_SOFTMAX_SCALE && value <= MAX_SOFTMAX_SCALE;
}

/// Whether the objects whose softmax keeps its running max over unscaled scores, and folds
/// the scale into the exponent (the RDNA prefill and decode objects and the CDNA objects
/// without a bias), compute softmax(scale * S) at @p value: a positive scale whose product
/// with log2(e) stays a normal f32. A negative one reverses the scores' order, so the max of
/// the unscaled ones is the scaled ones' least, and a masked score (-inf) times it is +inf;
/// at zero, or a scale subnormal enough to flush to it, the masked score times it is NaN.
inline bool rawScoresServe(double value)
{
    return value >= static_cast<double>(std::numeric_limits<float>::min())
           && value <= MAX_SOFTMAX_SCALE;
}

/// Whether a scale the engines take (isServableScale) must run on objects that scale their
/// scores before the softmax's max (each bias object, and every aiw object), not on the
/// raw-score ones (rawScoresServe): zero, negative (-0 included) or subnormal.
inline bool needsScaledScores(double value)
{
    return isServableScale(value) && !rawScoresServe(value);
}

/// The value of the pass-by-value scale tensor @p uid when the graph fixes it, nullopt when it
/// is supplied at execute (or is not a scalar operand at all).
inline std::optional<double> constantScaleTensorValue(const MatchContext& context, int64_t uid)
{
    try
    {
        const auto operand
            = hipdnn_plugin_sdk::makeScalarOperand(context.graph.getTensorMap(), uid, "scale");
        if(operand.isRuntimeUserSupplied)
        {
            return std::nullopt;
        }
        return hipdnn_plugin_sdk::toDouble(operand.bakedDefault);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return std::nullopt;
    }
}

/// The bytes of the zero bias the raw-score objects' bias twins read when a graph without a
/// bias needs scaled scores (needsScaledScores): one row of @p seqLenKv zeros in the query's
/// 16-bit dtype, as a [1, 1, 1, Skv] bias broadcast over B, H and Sq, padded to 16 bytes, the
/// alignment every bias object reads. An f32 graph's strided objects read it as one f32 at
/// stride 0 on every axis (CDNA), within those 16 bytes.
inline size_t zeroBiasBytes(int64_t seqLenKv)
{
    const auto bytes = static_cast<size_t>(std::max<int64_t>(seqLenKv, 1)) * BYTES_PER_ELEMENT;
    return (bytes + 15) / 16 * 16;
}

/// Where in the workspace the zero bias starts: after the @p used bytes a launch already
/// takes, rounded up to 16 bytes.
inline size_t zeroBiasOffset(size_t used)
{
    return (used + 15) / 16 * 16;
}

/// Zeroes the zero bias, zeroBiasBytes(@p seqLenKv) at @p offset in @p workspace, on @p stream
/// ahead of the launch that reads it, and returns its address. A memset on the launch stream,
/// so the launch stays capturable into a graph. @p engine names the engine in an error.
inline void* stageZeroBias(
    void* workspace, size_t offset, int64_t seqLenKv, hipStream_t stream, std::string_view engine)
{
    if(workspace == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string(engine)
                + ": a scale that needs scaled scores needs the engine's workspace");
    }
    void* bias = static_cast<char*>(workspace) + offset;
    if(hipMemsetAsync(bias, 0, zeroBiasBytes(seqLenKv), stream) != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(engine) + ": could not zero the zero bias in the workspace");
    }
    return bias;
}

/// Whether @p uid names a softmax scale the kernels can take: a host scalar (every dim 1, so
/// a per-head or per-row scale is not read as one value) of a floating-point type, and, when
/// its value is in the graph rather than supplied at execute, one isServableScale accepts.
/// A runtime value is checked at launch.
inline bool isServableScaleTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr || tensor->dims() == nullptr
       || !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
       || tensor->ragged_offset_tensor_uid().has_value())
    {
        return false;
    }
    for(const auto extent : *tensor->dims())
    {
        if(extent != 1)
        {
            return false;
        }
    }
    const auto dataType = tensor->data_type();
    if(dataType != data_objects::DataType::FLOAT && dataType != data_objects::DataType::HALF
       && dataType != data_objects::DataType::BFLOAT16
       && dataType != data_objects::DataType::DOUBLE)
    {
        return false;
    }
    try
    {
        const auto operand
            = hipdnn_plugin_sdk::makeScalarOperand(context.graph.getTensorMap(), uid, "scale");
        if(operand.isRuntimeUserSupplied)
        {
            return true;
        }
        return isServableScale(hipdnn_plugin_sdk::toDouble(operand.bakedDefault));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return false;
    }
}

/**
 * @brief Whether a graph runs on the zero bias: it has no bias of its own, and the scale the
 *        match fixes (attn_scale_value, its 1.0 default, or a pass-by-value tensor's constant
 *        value) needs scaled scores (needsScaledScores).
 *
 * The raw-score objects (rawScoresServe) cannot take such a scale, and their bias twins add
 * the bias to scores they have scaled first, as the reference does: a zero bias makes those
 * twins compute exactly the unbiased attention at any finite scale, so an engine whose plain
 * objects keep raw scores runs the graph there, reading a [1, 1, 1, Skv] zero bias in the
 * query's dtype from its workspace (zeroBiasBytes). A graph's own bias already runs there. A
 * scale supplied at execute is not known at match: the plain objects take it, and refuse one
 * they cannot serve at launch. Called once the scale has passed the graph's scale checks.
 */
inline bool runsOnAZeroBias(const MatchContext& context,
                            const data_objects::SdpaAttributes& attributes)
{
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return false;
    }
    if(attributes.scale_tensor_uid().has_value())
    {
        const auto value = constantScaleTensorValue(context, *attributes.scale_tensor_uid());
        return value.has_value() && needsScaledScores(*value);
    }
    return needsScaledScores(static_cast<double>(attributes.attn_scale_value().value_or(1.0F)));
}

/// The product of @p factors, or nullopt when it does not fit in int64_t. Every product of
/// graph-controlled extents goes through this: an unchecked one is signed overflow, and a
/// wrapped value can pass a bound it should fail.
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

/// Whether no two elements of @p tensor share an address, which an output needs: concurrent
/// workgroups writing one location race. Its axes of more than one element, ordered by
/// stride, must each step past everything the axes inside it span. Conservative: a layout
/// interleaving two axes inside one another's gaps declines too.
inline bool elementsDoNotOverlap(const data_objects::TensorAttributes& tensor)
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

/// The kernel metadata spelling of a graph dtype.
inline std::optional<std::string> kernelDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::HALF:
        return std::string("f16");
    default:
        return std::nullopt;
    }
}

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total, unlike KernelDefinition::getIntMetadata: a malformed record declines
/// rather than throwing out of a match.
inline std::optional<int64_t> integerMetadata(const KernelDefinition& kernel,
                                              std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* integer = std::get_if<int64_t>(&*value);
    return integer == nullptr ? std::nullopt : std::optional<int64_t>(*integer);
}

/// The string counterpart of integerMetadata, equally total.
inline std::optional<std::string> stringMetadata(const KernelDefinition& kernel,
                                                 std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* text = std::get_if<std::string>(&*value);
    return text == nullptr ? std::nullopt : std::optional<std::string>(*text);
}

/// The dtype of a servable attention sink relative to the query's (SEM-022): the query's own
/// 16-bit dtype, FLOAT (cuDNN's), or the 16-bit dtype the query is not (a model's bf16 sink
/// with an fp16 query, or the reverse). Bound as each family's sink-dtype token.
enum class SinkDtype : int64_t
{
    QUERY = 0,
    FLOAT = 1,
    OTHER_16BIT = 2,
};

/// A servable attention sink: its head stride in elements of its own dtype, and that dtype.
struct ServableSink
{
    int64_t stride = 0;
    SinkDtype dtype = SinkDtype::QUERY;
};

/// The 16-bit float dtype that is not dataType (HALF <-> BFLOAT16), or UNSET for any other.
inline data_objects::DataType otherSixteenBitDtype(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::HALF:
        return data_objects::DataType::BFLOAT16;
    case data_objects::DataType::BFLOAT16:
        return data_objects::DataType::HALF;
    default:
        return data_objects::DataType::UNSET;
    }
}

/**
 * @brief The attention sink as the kernels read it, or nullopt for a sink they cannot read.
 *
 * One logit per query head, FLOAT, in the query's dtype, or (a 16-bit query) in the other
 * 16-bit dtype (SEM-022), as a rank-1 [Hq] tensor or a rank-4 [1, Hq, 1, 1] one; a device
 * operand aligned to its element, its head stride positive unless there is one head, and
 * every head's offset in 16-bit units inside int32: the kernels read the logit as 16-bit
 * halves at h * stride, scaled to the dtype's width, in 32 bits, and widen it exactly
 * (kernel modifications 31 and 33, 11 and 12, 36 and 41).
 */
inline std::optional<ServableSink> servableSink(const MatchContext& context,
                                                int64_t uid,
                                                int64_t numHeads,
                                                data_objects::DataType dataType)
{
    const auto* sink = findTensor(context, uid);
    if(sink == nullptr || sink->dims() == nullptr || sink->strides() == nullptr || sink->virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(sink)
       || sink->ragged_offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    SinkDtype sinkDtype = SinkDtype::QUERY;
    if(sink->data_type() == data_objects::DataType::FLOAT)
    {
        sinkDtype = SinkDtype::FLOAT;
    }
    else if(sink->data_type() == otherSixteenBitDtype(dataType))
    {
        sinkDtype = SinkDtype::OTHER_16BIT;
    }
    else if(sink->data_type() != dataType)
    {
        return std::nullopt;
    }
    const bool isFloat = sinkDtype == SinkDtype::FLOAT;
    // Each element is one 16-bit half, or two for FLOAT.
    const int64_t halves = isFloat ? 2 : 1;
    if(!declaresAlignment(*sink, BYTES_PER_ELEMENT * halves))
    {
        return std::nullopt;
    }
    const auto* dims = sink->dims();
    uint32_t headAxis = 0;
    if(dims->size() == SDPA_RANK)
    {
        if(dims->Get(BATCH_AXIS) != 1 || dims->Get(SEQ_AXIS) != 1 || dims->Get(HEAD_DIM_AXIS) != 1)
        {
            return std::nullopt;
        }
        headAxis = HEAD_AXIS;
    }
    else if(dims->size() != 1)
    {
        return std::nullopt;
    }
    if(dims->Get(headAxis) != numHeads || sink->strides()->size() != dims->size())
    {
        return std::nullopt;
    }
    const int64_t stride = numHeads == 1 ? 0 : sink->strides()->Get(headAxis);
    if(stride < 0
       || (numHeads > 1 && (stride == 0 || stride > INT32_LIMIT / ((numHeads - 1) * halves))))
    {
        return std::nullopt;
    }
    return ServableSink{stride, sinkDtype};
}

/// The RDNA and aiw kernels' sink_stride argument: the head stride in elements, with bit
/// SINK_FLOAT_STRIDE_BIT set for a FLOAT sink (kernel modifications 31 and 11) and bit
/// SINK_OTHER_16BIT_STRIDE_BIT for a 16-bit sink in the dtype the query is not
/// (modifications 33 and 12); neither for a sink in the query's dtype.
inline constexpr int64_t SINK_FLOAT_STRIDE_BIT = 32;
inline constexpr int64_t SINK_OTHER_16BIT_STRIDE_BIT = 33;

inline int64_t rdnaSinkStrideArgument(int64_t stride, SinkDtype dtype)
{
    if(dtype == SinkDtype::FLOAT)
    {
        return stride | (int64_t{1} << SINK_FLOAT_STRIDE_BIT);
    }
    if(dtype == SinkDtype::OTHER_16BIT)
    {
        return stride | (int64_t{1} << SINK_OTHER_16BIT_STRIDE_BIT);
    }
    return stride;
}

/// A graph's mask bounds in the kernels' terms: key j is hidden from query row i when
/// j > i + align + right or j < i + align - left, with align 0 top-left and Skv - Sq
/// bottom-right; UNBOUNDED for a side with no bound. left is the FlashAttention
/// window_size_left, one less than the graph's left_bound (see resolveMaskBounds).
struct MaskBounds
{
    int64_t left = UNBOUNDED;
    int64_t right = UNBOUNDED;
    bool alignBottomRight = false;
};

/**
 * @brief The graph's mask bounds with the deprecated causal booleans resolved, or nullopt
 *        for a mask no FlyDSL kernel applies.
 *
 * The booleans act as if their setter ran first, as cuDNN's do and the reference resolves
 * them: right 0, a right_bound the graph also sets wins, and a left_bound is kept (a sliding
 * window, as gpt-oss and Mistral graphs set it). causal_mask keeps the stored alignment, so
 * diagonal_alignment refines it; causal_mask_bottom_right is always bottom-right. Both
 * booleans together, a bound below -1, and an alignment the schema does not name decline.
 *
 * The graph's left_bound follows cuDNN: left_bound = L keeps the L keys ending at the
 * diagonal, diagonal included, so the returned left is L - 1. left_bound = 0 keeps no key at
 * or before the diagonal, and L - 1 would read it as unbounded, so it declines.
 */
inline std::optional<MaskBounds> resolveMaskBounds(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();
    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }
    MaskBounds bounds;
    bounds.left = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    bounds.right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;
    if(bounds.left < UNBOUNDED || bounds.right < UNBOUNDED || bounds.left == 0)
    {
        return std::nullopt;
    }
    if(bounds.left != UNBOUNDED)
    {
        bounds.left -= 1;
    }
    if((topLeftDeprecated || bottomRightDeprecated) && bounds.right == UNBOUNDED)
    {
        bounds.right = 0;
    }
    const auto alignment = attributes.diagonal_alignment();
    if(alignment != data_objects::DiagonalAlignment::TOP_LEFT
       && alignment != data_objects::DiagonalAlignment::BOTTOM_RIGHT)
    {
        return std::nullopt;
    }
    bounds.alignBottomRight
        = bottomRightDeprecated || alignment == data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    return bounds;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The tensor a matched graph names by @p uid. Matching established it exists, so a missing
/// one is an internal error, reported under @p engine.
inline const data_objects::TensorAttributes&
    requireSdpaTensor(const MatchContext& context, int64_t uid, std::string_view engine)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "matched " + std::string(engine)
                                                           + " graph has no tensor for uid "
                                                           + std::to_string(uid));
    }
    return *tensor;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
