// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
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

/// The largest softmax scale every FlyDSL SDPA kernel serves. Real graphs use 1/sqrt(d) or a
/// temperature near 1; far above this each family overflows f32 at a different point (the
/// CDNA causal objects' running-max floor times scale * log2(e) passes FLT_MAX near 2.4e8),
/// so one bound here makes all of them decline alike rather than diverge.
inline constexpr double MAX_SOFTMAX_SCALE = 1048576.0; // 2^20

/// Whether @p value is a softmax scale the kernels can take: positive and at most
/// MAX_SOFTMAX_SCALE (NaN and infinity fail both). Positive because the CDNA kernels take the
/// running max over unscaled scores, which orders them as the scaled ones only for a
/// positive scale.
inline bool isServableScale(double value)
{
    return value > 0.0 && value <= MAX_SOFTMAX_SCALE;
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

/// A servable attention sink: its head stride in elements of its own dtype, and whether that
/// dtype is FLOAT (cuDNN's and hipDNN's reference's) rather than the query's.
struct ServableSink
{
    int64_t stride = 0;
    bool isFloat = false;
};

/**
 * @brief The attention sink as the kernels read it, or nullopt for a sink they cannot read.
 *
 * One logit per query head, FLOAT or in the query's dtype, as a rank-1 [Hq] tensor or a
 * rank-4 [1, Hq, 1, 1] one; a device operand aligned to its element, its head stride
 * positive unless there is one head, and every head's offset in 16-bit units inside int32:
 * the kernels read the logit as 16-bit halves at h * stride, scaled to the dtype's width, in
 * 32 bits (kernel modifications 31, 11 and 36).
 */
inline std::optional<ServableSink> servableSink(const MatchContext& context,
                                                int64_t uid,
                                                int64_t numHeads,
                                                data_objects::DataType dataType)
{
    const auto* sink = findTensor(context, uid);
    if(sink == nullptr || sink->dims() == nullptr || sink->strides() == nullptr
       || (sink->data_type() != dataType && sink->data_type() != data_objects::DataType::FLOAT)
       || sink->virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(sink)
       || sink->ragged_offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    const bool isFloat = sink->data_type() == data_objects::DataType::FLOAT;
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
    return ServableSink{stride, isFloat};
}

/// The RDNA and aiw kernels' sink_stride argument: the head stride in elements, with bit
/// SINK_FLOAT_STRIDE_BIT set for a FLOAT sink (kernel modifications 31 and 11).
inline constexpr int64_t SINK_FLOAT_STRIDE_BIT = 32;

inline int64_t rdnaSinkStrideArgument(int64_t stride, bool isFloat)
{
    return stride | (isFloat ? int64_t{1} << SINK_FLOAT_STRIDE_BIT : int64_t{0});
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
