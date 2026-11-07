// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

namespace hipblaslt_plugin
{

/// One operand of a column-major GEMM: the tensor it reads or writes and how to walk it.
struct GemmOperand
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    /// Elements between consecutive columns of the column-major matrix the GEMM sees.
    int64_t leadingDimension = 0;
    /// Elements between consecutive matrices of the batch; 0 broadcasts one matrix to all.
    int64_t batchStride = 0;
    /// Elements from the start of the tensor's buffer to the first matrix.
    int64_t offset = 0;
};

/// Bytes per element of the element types the GEMMs move.
inline int64_t elementBytes(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    return dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT ? 4 : 2;
}

/// @p base advanced to @p operand's first matrix.
inline const void* operandAddress(const void* base, const GemmOperand& operand)
{
    return static_cast<const char*>(base) + operand.offset * elementBytes(operand.dataType);
}

inline void* operandAddress(void* base, const GemmOperand& operand)
{
    return static_cast<char*>(base) + operand.offset * elementBytes(operand.dataType);
}

/**
 * @brief A strided-batched GEMM in BLAS (column-major) terms, independent of the library
 *        that executes it.
 *
 * For each of `batchCount` matrices: d(m x n) = op(a)(m x k) * op(b)(k x n), accumulated in
 * fp32 and written once -- or, with `accumulate`, added to what d already holds (beta = 1).
 * `d` stands in for `c` too, so it carries the output. The GEMM plans split a call until every
 * size fits the 32-bit ones the BLAS libraries take (fitLibraryLimits).
 */
struct GemmCall
{
    bool transposeA = false;
    bool transposeB = false;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t batchCount = 1;
    GemmOperand a;
    GemmOperand b;
    GemmOperand d;
    /// d += product (beta 1) rather than d = product (beta 0): sums a reduction split across
    /// several calls, such as backward weights over slices of images.
    bool accumulate = false;
    /// fp32 operands may multiply as TF32 where the library and GPU can, and as fp32 where they
    /// cannot. A permission, never a guarantee: nothing reports which ran. No effect on fp16 or
    /// bf16 operands.
    bool tf32 = false;
};

} // namespace hipblaslt_plugin
