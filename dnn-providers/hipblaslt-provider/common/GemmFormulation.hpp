// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "common/GemmCall.hpp"

namespace hipblaslt_plugin
{

/// A batch of row-major-indexed matrices laid over a tensor: element (b, i, j) of the view is at
/// offset + b * batchStride + i * rowStride + j * colStride of the tensor's buffer.
struct MatrixView
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t rowStride = 0;
    int64_t colStride = 0;
    /// 0 broadcasts one matrix to every batch entry; never 0 for the output.
    int64_t batchStride = 0;
    int64_t offset = 0;
};

/**
 * @brief The column-major GEMM that computes c = a b for every entry of a batch, or nullopt when
 *        a strided-batched BLAS GEMM cannot address the views.
 *
 * Each view must be unit-stride along its rows or its columns, with a leading dimension that
 * keeps its columns (or rows) apart. A row-major c is produced as c^T = b^T a^T, a column-major
 * c directly. a and b share an element type, c is the same or FLOAT. Batches of c may not
 * overlap. Leading dimensions (so m) must fit 32 bits, and k too unless c is FLOAT; a larger
 * n or k is left to fitLibraryLimits.
 */
std::optional<GemmCall>
    gemmForViews(const MatrixView& a, const MatrixView& b, const MatrixView& c, int64_t batchCount);

/// The largest batch stride, in elements, a strided-batched call passes on: rocBLAS's and
/// hipBLASLt's kernels take strides as 32-bit arguments (hipBLASLt declines a wider one; the
/// rocBLAS Tensile path has no check and would truncate it).
constexpr int64_t MAX_BATCH_STRIDE = std::numeric_limits<int32_t>::max();

/// The largest n or k one call passes on, a multiple of 2^10 under the libraries' 32 bits.
constexpr int64_t MAX_GEMM_EXTENT = int64_t{1} << 30;

/**
 * @brief @p calls split until every one fits what rocBLAS and hipBLASLt take, in order.
 *
 * A batched call whose batch stride exceeds MAX_BATCH_STRIDE becomes one call per matrix; an n
 * past MAX_GEMM_EXTENT becomes chunks of columns writing disjoint parts of d; a k past it (only
 * into fp32 d: gemmForViews) becomes chunks the later of which add to the first. m needs no
 * split: d's leading dimension bounds it.
 * Applied by the GEMM plans, so an image over 8 GiB of fp32, or a GEMM over 2^31 positions,
 * runs instead of reaching a library that cannot address it.
 */
std::vector<GemmCall> fitLibraryLimits(std::vector<GemmCall> calls);

} // namespace hipblaslt_plugin
