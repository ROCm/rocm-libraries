// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

/**
 * @file FlydslPointwiseCommon.hpp
 * @brief The flyDSL pointwise pack's host-side rules: which operations the shipped objects
 *        compute, and how a graph's output and operands become the kernel's four collapsed
 *        axes and per-operand element strides. Kept apart from the native file so the
 *        collapse can be tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_pointwise
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The axes the kernel decomposes an output index over.
constexpr size_t KERNEL_AXES = 4;

/// The kernel indexes output elements with int32 thread ids and, on its dense path, adds a
/// vector's worth of elements past the last thread; the bound leaves room for both.
constexpr int64_t MAX_ELEMENTS = std::numeric_limits<int32_t>::max() - (int64_t{1} << 16);

/// Element offsets are int64 in the kernel; a span past this is refused rather than risk
/// overflow in the products below.
constexpr int64_t MAX_SPAN = int64_t{1} << 62;

/// The operation spelling the kernel metadata uses, or nullopt for an operation no object
/// computes.
inline std::optional<std::string_view> operationName(data_objects::PointwiseMode mode)
{
    switch(mode)
    {
    case data_objects::PointwiseMode::SIGMOID_FWD:
        return "sigmoid";
    case data_objects::PointwiseMode::SWISH_FWD:
        return "swish";
    case data_objects::PointwiseMode::EXP:
        return "exp";
    case data_objects::PointwiseMode::SOFTPLUS_FWD:
        return "softplus";
    case data_objects::PointwiseMode::NEG:
        return "neg";
    case data_objects::PointwiseMode::IDENTITY:
        return "identity";
    case data_objects::PointwiseMode::ADD:
        return "add";
    case data_objects::PointwiseMode::SUB:
        return "sub";
    case data_objects::PointwiseMode::MUL:
        return "mul";
    case data_objects::PointwiseMode::DIV:
        return "div";
    default:
        return std::nullopt;
    }
}

inline bool isBinary(data_objects::PointwiseMode mode)
{
    return mode == data_objects::PointwiseMode::ADD || mode == data_objects::PointwiseMode::SUB
           || mode == data_objects::PointwiseMode::MUL || mode == data_objects::PointwiseMode::DIV;
}

/// One operand as the collapse reads it: its extents and element strides, outermost first.
struct OperandShape
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

/// The launch geometry of one graph: the output collapsed to KERNEL_AXES axes (padded with
/// leading unit axes), and each operand's element stride along every one of them -- 0 where
/// the operand broadcasts. A unary graph's `b` is all zero.
struct PointwiseGeometry
{
    int64_t elements = 0;
    std::array<int64_t, KERNEL_AXES> extents{};
    std::array<int64_t, KERNEL_AXES> outStrides{};
    std::array<int64_t, KERNEL_AXES> aStrides{};
    std::array<int64_t, KERNEL_AXES> bStrides{};
    /// The output and every input are one dense run of `elements`: the kernel's vector path.
    bool dense = false;
};

namespace detail
{

/// @p input's strides over the output's axes, cuDNN's broadcast rule: the shapes are aligned
/// at the innermost axis, a missing leading axis or an extent of 1 broadcasts (stride 0), and
/// any other extent must equal the output's. nullopt for an input that does not broadcast
/// to the output, or that has a negative stride on an axis it steps along.
inline std::optional<std::vector<int64_t>> broadcastStrides(const OperandShape& input,
                                                            const std::vector<int64_t>& outDims)
{
    const auto rank = outDims.size();
    if(input.dims.size() > rank || input.strides.size() != input.dims.size())
    {
        return std::nullopt;
    }
    const auto offset = rank - input.dims.size();
    std::vector<int64_t> strides(rank, 0);
    for(size_t axis = 0; axis < input.dims.size(); ++axis)
    {
        const auto extent = input.dims[axis];
        if(extent == 1)
        {
            continue;
        }
        if(extent != outDims[offset + axis] || input.strides[axis] < 0)
        {
            return std::nullopt;
        }
        strides[offset + axis] = input.strides[axis];
    }
    return strides;
}

/// Whether the output's elements are distinct addresses: every axis it steps along has a
/// positive stride, and, taken in stride order, each stride clears the whole span of the
/// axes below it. Two threads storing to one element would be a race.
inline bool isNonOverlapping(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    std::vector<std::pair<int64_t, int64_t>> axes; // (stride, extent)
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        if(dims[axis] == 1)
        {
            continue;
        }
        if(strides[axis] <= 0)
        {
            return false;
        }
        axes.emplace_back(strides[axis], dims[axis]);
    }
    std::sort(axes.begin(), axes.end());
    int64_t span = 1;
    for(const auto& [stride, extent] : axes)
    {
        if(stride < span || stride > MAX_SPAN / extent)
        {
            return false;
        }
        span = stride * extent;
    }
    return true;
}

/// One past the largest element offset @p strides reach over @p dims, or nullopt past MAX_SPAN.
inline std::optional<int64_t> span(const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides)
{
    int64_t last = 0;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        const auto step = dims[axis] - 1;
        if(step != 0 && strides[axis] > (MAX_SPAN - last) / step)
        {
            return std::nullopt;
        }
        last += step * strides[axis];
    }
    return last + 1;
}

} // namespace detail

/**
 * @brief The kernel's view of a graph, or nullopt for one it cannot address.
 *
 * Axes of output extent 1 are dropped, then each axis is merged into the one inside it
 * wherever every operand steps across the pair as one run (its outer stride is its inner
 * stride times the inner extent; two broadcast axes merge too). What remains must be at
 * most KERNEL_AXES axes: [T, H] + [1, H], [1, T, k, H] * [1, T, k, 1] and BSHD views of
 * a [B, H, S, D] output all fit, and any packed layout collapses to one axis.
 *
 * @param b nullptr for a unary graph.
 */
inline std::optional<PointwiseGeometry>
    collapse(const OperandShape& out, const OperandShape& a, const OperandShape* b)
{
    const auto& outDims = out.dims;
    if(outDims.empty() || out.strides.size() != outDims.size())
    {
        return std::nullopt;
    }
    int64_t elements = 1;
    for(const auto extent : outDims)
    {
        if(extent <= 0 || elements > MAX_ELEMENTS / extent)
        {
            return std::nullopt;
        }
        elements *= extent;
    }
    if(!detail::isNonOverlapping(outDims, out.strides))
    {
        return std::nullopt;
    }

    const auto aStrides = detail::broadcastStrides(a, outDims);
    const auto bStrides
        = b != nullptr
              ? detail::broadcastStrides(*b, outDims)
              : std::optional<std::vector<int64_t>>(std::vector<int64_t>(outDims.size(), 0));
    if(!aStrides.has_value() || !bStrides.has_value())
    {
        return std::nullopt;
    }
    if(!detail::span(outDims, *aStrides).has_value()
       || !detail::span(outDims, *bStrides).has_value())
    {
        return std::nullopt;
    }

    // Innermost first while merging; reversed into the kernel's outermost-first order below.
    struct Axis
    {
        int64_t extent;
        int64_t out;
        int64_t a;
        int64_t b;
    };
    std::vector<Axis> merged;
    for(size_t index = outDims.size(); index-- > 0;)
    {
        const Axis axis{outDims[index], out.strides[index], (*aStrides)[index], (*bStrides)[index]};
        if(axis.extent == 1)
        {
            continue;
        }
        if(!merged.empty())
        {
            auto& inner = merged.back();
            if(axis.out == inner.out * inner.extent && axis.a == inner.a * inner.extent
               && axis.b == inner.b * inner.extent)
            {
                inner.extent *= axis.extent;
                continue;
            }
        }
        merged.push_back(axis);
    }
    if(merged.size() > KERNEL_AXES)
    {
        return std::nullopt;
    }

    PointwiseGeometry geometry;
    geometry.elements = elements;
    geometry.extents.fill(1);
    for(size_t index = 0; index < merged.size(); ++index)
    {
        const auto slot = KERNEL_AXES - 1 - index;
        geometry.extents[slot] = merged[index].extent;
        geometry.outStrides[slot] = merged[index].out;
        geometry.aStrides[slot] = merged[index].a;
        geometry.bStrides[slot] = merged[index].b;
    }
    // A single element is one unit-stride run like any other.
    const auto last = KERNEL_AXES - 1;
    if(merged.empty())
    {
        geometry.outStrides[last] = 1;
        geometry.aStrides[last] = 1;
        geometry.bStrides[last] = b == nullptr ? 0 : 1;
    }
    // One run with unit strides everywhere. A unary graph's second input is never read, so
    // only the first input and the output decide.
    geometry.dense = merged.size() <= 1 && geometry.outStrides[last] == 1
                     && geometry.aStrides[last] == 1
                     && (b == nullptr || geometry.bStrides[last] == 1);
    return geometry;
}

/// @p tensor's dims and strides, or nullopt when either is missing or they disagree in rank:
/// the matcher runs on unvalidated graphs.
inline std::optional<OperandShape> operandShape(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != strides->size() || dims->empty())
    {
        return std::nullopt;
    }
    return OperandShape{std::vector<int64_t>(dims->begin(), dims->end()),
                        std::vector<int64_t>(strides->begin(), strides->end())};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_pointwise

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
