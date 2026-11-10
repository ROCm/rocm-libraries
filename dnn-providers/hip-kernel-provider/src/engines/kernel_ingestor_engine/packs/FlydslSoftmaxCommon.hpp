// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormCommon.hpp"

/**
 * @file FlydslSoftmaxCommon.hpp
 * @brief The flyDSL softmax pack's reading of a graph: the [rows, columns] problem its
 *        kernel is launched with, or nullopt for any graph that kernel would not compute
 *        exactly, and how many rows one launch may cover.
 *
 * Shared by the matcher and the dispatch in FlydslSoftmaxNative.cpp and by the tests, so
 * the problem the matcher approves and the arguments the launch passes are one computation.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_softmax
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// `N: 0` in a UKD's metadata: the object reads the row width from p's descriptor.
constexpr int64_t GENERIC_N_SENTINEL = 0;
/// `n_max: 0`: the object serves every row width.
constexpr int64_t UNBOUNDED_N_MAX = 0;

/// The bytes one launch may address through one operand: the kernel views p and s as
/// buffers with 32-bit byte offsets. A larger graph runs as several launches, each over a
/// slice of whole rows (rowsPerLaunch()).
constexpr int64_t MAX_LAUNCH_BYTES = flydsl_norm::MAX_OPERAND_BYTES;

/// One operand seen as rows of contiguous elements: every axis but the last collapses to
/// `rows`, uniformly `rowStride` elements apart.
struct RowLayout
{
    int64_t rows = 0;
    int64_t columns = 0;
    int64_t rowStride = 0;
};

/**
 * @brief @p tensor as rows of its last axis, or nullopt when its layout is not that.
 *
 * The last axis must be contiguous (stride 1; any stride when it has one element): the
 * kernel's innermost stride is a compile-time 1. The leading axes must collapse to one
 * row index with one stride: each non-unit axis's stride is the next non-unit axis's
 * stride times that axis's extent, and the innermost of them steps at least a whole row,
 * so rows never overlap. An axis of extent 1 is never stepped along, so its stride is not
 * read. Runs on an unvalidated graph, so it is total.
 */
inline std::optional<RowLayout> rowLayout(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->empty() || strides->size() != dims->size())
    {
        return std::nullopt;
    }
    const auto last = static_cast<flatbuffers::uoffset_t>(dims->size() - 1);
    RowLayout layout;
    layout.columns = dims->Get(last);
    if(layout.columns <= 0 || (layout.columns != 1 && strides->Get(last) != 1))
    {
        return std::nullopt;
    }
    layout.rows = 1;
    layout.rowStride = layout.columns;

    // Elements a launch could address, kept far below int64 so offset arithmetic in the
    // dispatch cannot overflow.
    constexpr int64_t MAX_ELEMENTS = std::numeric_limits<int64_t>::max() / 16;
    int64_t expected = 0; // The stride the next non-unit axis outward must have; 0: none yet.
    for(auto axis = static_cast<int64_t>(last) - 1; axis >= 0; --axis)
    {
        const auto index = static_cast<flatbuffers::uoffset_t>(axis);
        const int64_t extent = dims->Get(index);
        const int64_t stride = strides->Get(index);
        if(extent <= 0)
        {
            return std::nullopt;
        }
        if(extent == 1)
        {
            continue;
        }
        if(expected == 0)
        {
            if(stride < layout.columns)
            {
                return std::nullopt;
            }
            layout.rowStride = stride;
        }
        else if(stride != expected)
        {
            return std::nullopt;
        }
        if(stride > MAX_ELEMENTS / extent || layout.rows > MAX_ELEMENTS / extent)
        {
            return std::nullopt;
        }
        expected = stride * extent;
        layout.rows *= extent;
    }
    return layout;
}

/// Everything the kernel is launched with besides the pointers.
struct SoftmaxProblem
{
    int64_t rows = 0;
    int64_t columns = 0;
    /// Elements between consecutive rows of p and of s.
    int64_t pRowStride = 0;
    int64_t sRowStride = 0;
    data_objects::DataType pType = data_objects::DataType::UNSET;
    data_objects::DataType sType = data_objects::DataType::UNSET;
    bool storeStats = false;
    /// Stats in log2 units: the kernel multiplies the natural-log LSE by 1 / ln 2.
    bool statsLog2 = false;
};

/// Whether @p tensor declares a base alignment of at least, and a multiple of, @p bytes.
inline bool declaresAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/**
 * @brief The kernel's problem for a softmax node over @p p, @p s and optionally
 *        @p stats, or nullopt when the kernel does not compute it exactly.
 *
 * Mirrors CpuFpReferenceSoftmax's validation (s of p's dims; stats of p's dims with the
 * last 1), then adds the kernel's own limits: p and s each in bf16, f16 or f32 (any pair),
 * rows of the last axis (rowLayout()), each declaring its element alignment; stats f32,
 * packed and 4-byte aligned, one per row; one row of p and of s within a launch's 32-bit
 * byte offsets.
 */
inline std::optional<SoftmaxProblem>
    resolveSoftmaxProblem(const data_objects::SoftmaxAttributes& attributes,
                          const data_objects::TensorAttributes& p,
                          const data_objects::TensorAttributes& s,
                          const data_objects::TensorAttributes* stats)
{
    if(!flydsl_norm::isSupportedDataType(p.data_type())
       || !flydsl_norm::isSupportedDataType(s.data_type()))
    {
        return std::nullopt;
    }
    const auto pLayout = rowLayout(p);
    const auto sLayout = rowLayout(s);
    if(!pLayout.has_value() || !sLayout.has_value() || !flydsl_norm::sameShape(p, s))
    {
        return std::nullopt;
    }
    const auto pBytes = flydsl_norm::elementBytes(p.data_type());
    const auto sBytes = flydsl_norm::elementBytes(s.data_type());
    if(!declaresAlignment(p, pBytes) || !declaresAlignment(s, sBytes)
       || pLayout->columns > MAX_LAUNCH_BYTES / std::max(pBytes, sBytes)
       || pLayout->rows > std::numeric_limits<int32_t>::max())
    {
        return std::nullopt;
    }

    if(stats != nullptr)
    {
        const auto* statsDims = stats->dims();
        const auto* statsStrides = stats->strides();
        const auto* pDims = p.dims();
        if(statsDims == nullptr || statsStrides == nullptr
           || statsStrides->size() != statsDims->size() || statsDims->size() != pDims->size()
           || stats->data_type() != data_objects::DataType::FLOAT || !declaresAlignment(*stats, 4))
        {
            return std::nullopt;
        }
        for(flatbuffers::uoffset_t axis = 0; axis < statsDims->size(); ++axis)
        {
            const bool last = axis + 1 == statsDims->size();
            if(statsDims->Get(axis) != (last ? 1 : pDims->Get(axis)))
            {
                return std::nullopt;
            }
        }
        // One f32 per row, consecutive rows adjacent: the kernel's 1-D stats view.
        if(!flydsl_norm::isPackedRowMajor(*stats))
        {
            return std::nullopt;
        }
    }

    SoftmaxProblem problem;
    problem.rows = pLayout->rows;
    problem.columns = pLayout->columns;
    problem.pRowStride = pLayout->rowStride;
    problem.sRowStride = sLayout->rowStride;
    problem.pType = p.data_type();
    problem.sType = s.data_type();
    problem.storeStats = stats != nullptr;
    problem.statsLog2 = attributes.stats_use_log2();
    return problem;
}

/// The rows one launch covers: as many as keep every byte of p and of s it addresses
/// within MAX_LAUNCH_BYTES, a multiple of @p rowsPerBlock when more than one block's
/// worth, and never more than the problem has. At least 1: resolveSoftmaxProblem() bounds
/// one row.
inline int64_t rowsPerLaunch(const SoftmaxProblem& problem, int64_t rowsPerBlock)
{
    const auto fit = [&problem](int64_t rowStride, int64_t bytes) {
        // ((k - 1) * rowStride + columns) * bytes <= MAX_LAUNCH_BYTES.
        return 1 + (MAX_LAUNCH_BYTES / bytes - problem.columns) / rowStride;
    };
    auto rows = std::min({problem.rows,
                          fit(problem.pRowStride, flydsl_norm::elementBytes(problem.pType)),
                          fit(problem.sRowStride, flydsl_norm::elementBytes(problem.sType))});
    if(rowsPerBlock > 0 && rows > rowsPerBlock && rows < problem.rows)
    {
        rows -= rows % rowsPerBlock;
    }
    return std::max<int64_t>(rows, 1);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_softmax

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
