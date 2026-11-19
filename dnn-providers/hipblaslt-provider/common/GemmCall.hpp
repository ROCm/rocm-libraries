// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>

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
 * @brief A fused convolution epilogue a backend applies in its own GEMM kernel: one of
 *        hipBLASLt's native epilogues. A bias in it is a vector along m, so it is set only on
 *        calls whose rows are the output channels. Computed in fp32 before the GEMM's one
 *        rounding of d.
 */
struct NativeEpilogue
{
    /// The hipblasLtEpilogue_t value.
    int mode = 0;
    /// The activation's arguments (clamp bounds).
    float act0 = 0.0F;
    float act1 = 0.0F;
    /// The per-channel bias vector along m; none without one.
    std::optional<int64_t> biasUid;
    hipdnn_flatbuffers_sdk::data_objects::DataType biasType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
};

/**
 * @brief A strided-batched GEMM in BLAS (column-major) terms, independent of the library
 *        that executes it.
 *
 * For each of `batchCount` matrices: d(m x n) = op(a)(m x k) * op(b)(k x n), accumulated in
 * fp32 and written once -- or, with `accumulate`, added to what d already holds (beta = 1), or,
 * with `c`, added to c (beta = 1). Without `c`, `d` stands in for c, so it carries the output.
 * The GEMM plans split a call until every size fits the 32-bit ones the BLAS libraries take
 * (fitLibraryLimits).
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
    /// A fused epilogue the backend applies to d (hipBLASLt only); unset for every unfused call.
    std::optional<NativeEpilogue> epilogue;
    /// An addend of d's shape and type the product is added to, in fp32 before the epilogue
    /// and the one rounding (beta = 1); d is then written, never read. A fused epilogue's
    /// per-channel bias where the channels are d's columns: one matrix of the bias broadcast
    /// along the rows, batch stride 0. Never with `accumulate`.
    std::optional<GemmOperand> c;
};

} // namespace hipblaslt_plugin
