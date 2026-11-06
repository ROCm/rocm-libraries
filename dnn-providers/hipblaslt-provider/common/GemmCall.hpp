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
};

/**
 * @brief A strided-batched GEMM in BLAS (column-major) terms, independent of the library
 *        that executes it.
 *
 * For each of `batchCount` matrices: d(m x n) = op(a)(m x k) * op(b)(k x n), accumulated in
 * fp32 and written once. `d` is written in place of `c` (beta = 0), so it carries the output.
 * Every value is validated to fit the 32-bit sizes the BLAS libraries take.
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
};

} // namespace hipblaslt_plugin
