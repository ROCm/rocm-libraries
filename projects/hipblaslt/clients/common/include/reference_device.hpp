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
 * One thread per output element, float accumulate, implemented independently of
 * TensileLite so it can catch kernel bugs. Selected via the `check_ref` argument
 * / `--check_ref` bench flag.
 */

/// True when `arg` describes a matmul the GPU reference path currently supports
/// (plain f32/f16 GEMM, default epilogue, no scaling/bias/aux, strided batch).
/// On false, `reason` is filled with the first unsupported feature encountered.
bool gpu_ref_supported(const Arguments& arg, std::string& reason);

/// Compute D_gold = alpha * op(A) * op(B) + beta * C on the device, accumulating
/// in float. All pointers are device pointers. Column-major, with the same
/// transpose/leading-dim/batch-stride conventions as cblas_gemm().
void run_reference_gemm_device(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               float       alpha,
                               float       beta,
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
