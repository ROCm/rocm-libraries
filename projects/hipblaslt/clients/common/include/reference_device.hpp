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
 * can catch kernel bugs. All supported types accumulate in float. Selected via
 * the `check_ref` argument / `--check_ref` bench flag.
 */

/// True when `arg` describes a matmul the GPU reference path currently supports:
/// plain GEMM with matching f32/f16/bf16 A/B and f32/f16/bf16 C/D on compute 32F,
/// default epilogue, strided batch, no scaling. On false, `reason` is filled with
/// the first unsupported feature encountered.
bool gpu_ref_supported(const Arguments& arg, std::string& reason);

/// Compute D_gold on the device: D = alpha * op(A)op(B) + beta * C, accumulated
/// in float. A/B carry matching f32/f16/bf16 input types; C/D are f32/f16/bf16.
/// Column-major, with the same transpose/leading-dim/batch-stride conventions as
/// cblas_gemm(). All pointers are device pointers. Returns false (after logging)
/// if the launch hits a HIP error, so the caller can fail loudly.
bool run_reference_gemm_device(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               double      alpha,
                               double      beta,
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
                               hipStream_t stream);
