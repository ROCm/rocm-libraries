// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
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

/// The bytes of the operand a reduction walks past which rocBLAS's Tensile path splits the
/// reduction itself (k times the leading dimension k walks, times a's element size): it keeps
/// its kernels' 32-bit offsets in range by chunking k and adding each chunk into d.
constexpr int64_t ROCBLAS_REDUCTION_BYTES = std::numeric_limits<int32_t>::max();

/**
 * @brief Whether rocBLAS would split @p call's reduction into chunks summed in a non-fp32 d.
 *
 * Each chunk's partial sum is rounded to d's type before the next adds to it, so a half or
 * bf16 d is rounded once per chunk instead of once. No split of m, n or the batch helps: the
 * stride k walks is a leading dimension, fixed by the operand's layout. The rocBLAS plans
 * decline such a call (hipBLASLt does not split k).
 */
bool rocblasRoundsReductionChunks(const GemmCall& call);

/**
 * @brief Whether rocBLAS on @p gcnArchName would run @p call as its small-m transposed GEMV
 *        kernel (rocblas_gemvtsm_kernel) with bf16 inputs.
 *
 * rocBLAS runs a GEMM whose m or n is 1 as a GEMV on gfx906/908/90a, gfx10 and gfx11 (on gfx11
 * always, elsewhere by per-type rules), and a transposed GEMV of at most 64 reduction rows over
 * more than 8 batch entries as rocblas_gemvtsm_kernel. That kernel multiplies its fp32 x by the
 * bf16 matrix through rocblas_bfloat16's operator*(float, rocblas_bfloat16), which rounds every
 * product to bf16 before the fp32 sum: a grouped convolution with one output channel per group
 * (K/G = 1) in bf16 is off by up to a bf16 step of each product. The rocBLAS plans decline such
 * a call; fp16 is unaffected (its products stay fp32), and so is every other GEMV kernel.
 */
bool rocblasGemvRoundsBf16Products(const GemmCall& call, std::string_view gcnArchName);

} // namespace hipblaslt_plugin
