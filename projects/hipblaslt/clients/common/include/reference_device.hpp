// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_arguments.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <string>

/*!\file
 * \brief Naive HIP reference GEMM for correctness verification (the compare step
 *        lives in gpu_compare.hpp). Used only for testing/benchmark verification;
 *        not part of the GPU library.
 *
 * One thread per output element, implemented independently of TensileLite so it
 * can catch kernel bugs. f64 accumulates in double, int8 in int32; every other
 * supported type accumulates in float. Selected via the `check_ref` argument /
 * `--check_ref` bench flag.
 */

/// True when `arg` describes a matmul the GPU reference path currently supports
/// (plain GEMM with f32/f16/bf16/OCP-fp8/OCP-bf8 inputs and f32/f16/bf16 C/D on
/// compute 32F, all-f64 on compute 64F, int8 in / int32 out on compute 32I, or
/// all-complex float/double on compute 32F/64F; default epilogue, strided batch).
/// Non-MX scalar/vector scaleA/B, scaleAlphaVec, scaleC, and scaleD are supported
/// on the float-scale computes (32F and the fast-32F variants); 16F-compute
/// scaling is still deferred, as are bias/aux epilogues. MX/block A/B scaling is
/// supported by having the caller pass the host float dequant (refA/refB, scales
/// baked in) as f32 inputs, so the block-scaled side may carry any OCP MX narrow
/// type (fp8/bf8/fp4/fp6/bf6); MX does not combine with C/D scaling or alpha
/// vector here. A and B may carry different float-class inputs
/// (f32/f16/bf16/fp8/bf8); f64, int8, and complex remain same-type. On false,
/// `reason` is filled with the first unsupported feature encountered.
bool gpu_ref_supported(const Arguments& arg, std::string& reason);

/// Compute D_gold on the device. f64 (tD == HIP_R_64F) accumulates in double,
/// int32 (tD == HIP_R_32I) in int32, complex (tD == HIP_C_32F/HIP_C_64F) in
/// complex<float>/complex<double>; every other type accumulates in float. On the
/// float-accumulate path A and B may carry different float-class input types
/// (tA != tB, e.g. OCP f8 x bf8); f64, int8, and complex require matching A/B.
/// conjA/conjB apply conjugation (op == HIPBLAS_OP_C) on top of the transpose,
/// and alphai/betai carry the imaginary parts of alpha/beta (0 on real paths).
///
/// The trailing scale arguments apply only on the float-accumulate path (ignored
/// by the f64/int8/complex paths, which the gate guarantees are unscaled):
/// D = scaleD * (alpha * scaleA * scaleAlphaVec * scaleB * op(A)op(B) + beta * C),
/// where the caller has folded scaleC into beta. dScaleA/dScaleAlphaVec index the
/// M row and dScaleB the N column; scalar scales pass scaleAIsVec/scaleBIsVec ==
/// false. A null scale pointer means no scaling (factor 1). The scale buffers are
/// float (the compute type on this path). All pointers are device pointers.
/// Column-major, with
/// the same transpose/leading-dim/batch-stride conventions as cblas_gemm().
void run_reference_gemm_device(bool        transA_is_n,
                               bool        transB_is_n,
                               bool        conjA,
                               bool        conjB,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               double      alpha,
                               double      alphai,
                               double      beta,
                               double      betai,
                               const void* dA,
                               hipDataType tA,
                               int64_t     lda,
                               int64_t     strideA,
                               const void* dB,
                               hipDataType tB,
                               int64_t     ldb,
                               int64_t     strideB,
                               const void* dC,
                               hipDataType tC,
                               int64_t     ldc,
                               int64_t     strideC,
                               void*       dDgold,
                               hipDataType tD,
                               int64_t     ldd,
                               int64_t     strideD,
                               int32_t     batchCount,
                               hipStream_t stream,
                               const void* dScaleA        = nullptr,
                               bool        scaleAIsVec    = false,
                               const void* dScaleB        = nullptr,
                               bool        scaleBIsVec    = false,
                               const void* dScaleAlphaVec = nullptr,
                               double      scaleD         = 1.0);
