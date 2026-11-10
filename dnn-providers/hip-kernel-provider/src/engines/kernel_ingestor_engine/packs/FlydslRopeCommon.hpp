// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/rope_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

/**
 * @file FlydslRopeCommon.hpp
 * @brief The flyDSL RoPE pack's reading of a graph: the rank-4 problem its kernel is
 *        launched with, or nullopt for any graph that kernel would not compute exactly.
 *
 * Shared by the matcher and the dispatch in FlydslRopeNative.cpp and by the tests, so the
 * shape the matcher approves and the arguments the launch passes are one computation.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_rope
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The kernel's rank: x is addressed as [d0, d1, d2, D], lower ranks padded with leading
/// unit axes.
constexpr size_t KERNEL_RANK = 4;

/// Grid limits: one workgroup per row on grid (d2, d1, d0).
constexpr int64_t MAX_GRID_X = std::numeric_limits<int32_t>::max();
constexpr int64_t MAX_GRID_YZ = 65535;

/// Element offsets the kernel forms are int64; every operand's furthest element must be
/// addressable with headroom for the row and column terms summed separately.
constexpr int64_t MAX_ELEMENT_OFFSET = std::numeric_limits<int64_t>::max() / 4;

/// Everything the kernel is launched with besides the pointers and the scale.
struct RopeProblem
{
    /// x's extents padded to rank 4; dims[3] is the head dim D.
    std::array<int64_t, KERNEL_RANK> dims{};
    /// Element strides, padded alike. An axis the kernel never steps along (extent 1 in x,
    /// or a broadcast freqs axis) carries 0, so a stride the graph left arbitrary there is
    /// never multiplied in.
    std::array<int64_t, KERNEL_RANK> xStrides{};
    std::array<int64_t, KERNEL_RANK> freqsStrides{};
    std::array<int64_t, KERNEL_RANK> yStrides{};
    int32_t headDim = 0;
    int32_t ropeDim = 0;
    int32_t ropeOffset = 0;
};

namespace detail
{

/// The furthest element @p dims and @p strides reach, or nullopt past MAX_ELEMENT_OFFSET.
inline std::optional<int64_t> furthestElement(const std::array<int64_t, KERNEL_RANK>& dims,
                                              const std::array<int64_t, KERNEL_RANK>& strides)
{
    int64_t furthest = 0;
    for(size_t axis = 0; axis < KERNEL_RANK; ++axis)
    {
        int64_t term = 0;
        if(__builtin_mul_overflow(dims[axis] - 1, strides[axis], &term)
           || __builtin_add_overflow(furthest, term, &furthest) || furthest > MAX_ELEMENT_OFFSET)
        {
            return std::nullopt;
        }
    }
    return furthest;
}

} // namespace detail

/**
 * @brief The kernel's problem for a RoPE node over @p x, @p freqs and @p y, or nullopt
 *        when the kernel does not compute it exactly.
 *
 * Mirrors CpuFpReferenceRope's validation (r = rope_dim or D, even and in (0, D];
 * o = rope_offset_ext or D - r, with [o, o + r) inside D; freqs of x's rank, its last dim
 * r and every other x's or 1; y of x's dims), then adds the kernel's own limits: rank 2-4,
 * the grid, int32 shape arguments, int64 offsets, non-negative strides, and a y with no
 * zero stride on an axis it spans (two rows written to one place).
 *
 * Runs on an unvalidated graph, so it must be total.
 */
inline std::optional<RopeProblem> resolveRopeProblem(const data_objects::RopeAttributes& attributes,
                                                     const data_objects::TensorAttributes& x,
                                                     const data_objects::TensorAttributes& freqs,
                                                     const data_objects::TensorAttributes& y)
{
    const auto* xDims = x.dims();
    const auto* xStrides = x.strides();
    const auto* freqsDims = freqs.dims();
    const auto* freqsStrides = freqs.strides();
    const auto* yDims = y.dims();
    const auto* yStrides = y.strides();
    if(xDims == nullptr || xStrides == nullptr || freqsDims == nullptr || freqsStrides == nullptr
       || yDims == nullptr || yStrides == nullptr)
    {
        return std::nullopt;
    }
    const auto rank = static_cast<size_t>(xDims->size());
    if(rank < 2 || rank > KERNEL_RANK || xStrides->size() != rank || freqsDims->size() != rank
       || freqsStrides->size() != rank || yDims->size() != rank || yStrides->size() != rank)
    {
        return std::nullopt;
    }

    const int64_t headDim = xDims->Get(static_cast<flatbuffers::uoffset_t>(rank - 1));
    if(headDim <= 0 || headDim > std::numeric_limits<int32_t>::max() || attributes.rope_dim() < 0)
    {
        return std::nullopt;
    }
    const int64_t ropeDim = attributes.rope_dim() == 0 ? headDim : attributes.rope_dim();
    if(ropeDim > headDim || ropeDim % 2 != 0)
    {
        return std::nullopt;
    }
    const auto offsetExt = attributes.rope_offset_ext();
    const int64_t ropeOffset = offsetExt.has_value() ? *offsetExt : headDim - ropeDim;
    if(ropeOffset < 0 || ropeOffset > headDim - ropeDim)
    {
        return std::nullopt;
    }

    RopeProblem problem;
    problem.dims.fill(1);
    const size_t pad = KERNEL_RANK - rank;
    for(size_t axis = 0; axis < rank; ++axis)
    {
        const auto index = static_cast<flatbuffers::uoffset_t>(axis);
        const int64_t extent = xDims->Get(index);
        const int64_t freqsExtent = freqsDims->Get(index);
        const bool last = axis + 1 == rank;
        if(extent <= 0 || yDims->Get(index) != extent
           || (last ? freqsExtent != ropeDim : (freqsExtent != extent && freqsExtent != 1)))
        {
            return std::nullopt;
        }
        const int64_t xStride = xStrides->Get(index);
        const int64_t freqsStride = freqsStrides->Get(index);
        const int64_t yStride = yStrides->Get(index);
        if(xStride < 0 || freqsStride < 0 || yStride < 0 || (extent > 1 && yStride == 0))
        {
            return std::nullopt;
        }
        const auto slot = pad + axis;
        problem.dims[slot] = extent;
        problem.xStrides[slot] = extent == 1 ? 0 : xStride;
        problem.yStrides[slot] = extent == 1 ? 0 : yStride;
        problem.freqsStrides[slot] = freqsExtent == 1 ? 0 : freqsStride;
    }

    if(problem.dims[2] > MAX_GRID_X || problem.dims[1] > MAX_GRID_YZ
       || problem.dims[0] > MAX_GRID_YZ)
    {
        return std::nullopt;
    }
    auto freqsDimsPadded = problem.dims;
    freqsDimsPadded[KERNEL_RANK - 1] = ropeDim;
    if(!detail::furthestElement(problem.dims, problem.xStrides).has_value()
       || !detail::furthestElement(problem.dims, problem.yStrides).has_value()
       || !detail::furthestElement(freqsDimsPadded, problem.freqsStrides).has_value())
    {
        return std::nullopt;
    }

    problem.headDim = static_cast<int32_t>(headDim);
    problem.ropeDim = static_cast<int32_t>(ropeDim);
    problem.ropeOffset = static_cast<int32_t>(ropeOffset);
    return problem;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_rope

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
