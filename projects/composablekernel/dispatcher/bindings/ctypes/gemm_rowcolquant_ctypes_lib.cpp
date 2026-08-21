// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm RowColQuant ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_rowcolquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, QDataType, AccDataType.
 *
 * Direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly; no dispatcher registry is used.
 *
 * RowColQuant = per-row scale of A (AQ, shape [M, 1], broadcast over N) plus
 * per-column scale of B (BQ, shape [1, N], broadcast over M). Both scale tensors
 * are AccDataType (float). There is NO quant-group size; QK_A == QK_B == 1.
 *
 * Shared infrastructure (memory management, arch validation, timing, init /
 * cleanup, C API boilerplate) lives in quant_bridge_common.hpp.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>

#include "quant_bridge_common.hpp"

extern "C" {

QUANT_BRIDGE_C_API()

/**
 * Run RowColQuant GEMM:
 *   C[M,N] = (A[M,K] * AQ[M,1]) @ (B[K,N] * BQ[1,N])
 * with AQ a per-row scale of A (M floats) and BQ a per-column scale of B (N
 * floats). A, B, AQ, BQ, C are host pointers; device memory is managed
 * internally. time_ms is an optional output. Returns 0 on success.
 */
int dispatcher_run_rowcolquant_gemm(const void* A,
                                    const void* B,
                                    const void* AQ,
                                    const void* BQ,
                                    void* C,
                                    int64_t M,
                                    int64_t N,
                                    int64_t K,
                                    int64_t stride_A,
                                    int64_t stride_B,
                                    int64_t stride_C,
                                    int k_batch,
                                    float* time_ms)
{
    using namespace quant_bridge;
    const char* kFn = "dispatcher_run_rowcolquant_gemm";

    if(!check_initialized(kFn, g_initialized) || !check_non_null(kFn, {A, B, AQ, BQ, C}) ||
       !check_positive_dims(kFn, {M, N, K}) || !validate_supported_arch(kFn))
        return -1;

    // Only packed (contiguous) layouts are supported: buffers are M*K, K*N, M, N, M*N.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << kFn << ": non-packed strides are not supported. Expected stride_A=" << K
                  << " stride_B=" << K << " stride_C=" << N << ", got stride_A=" << stride_A
                  << " stride_B=" << stride_B << " stride_C=" << stride_C << "\n";
        return -1;
    }

    DeviceBuffer<ADataType> A_dev;
    DeviceBuffer<BDataType> B_dev;
    DeviceBuffer<QDataType> AQ_dev;
    DeviceBuffer<QDataType> BQ_dev;
    DeviceBuffer<CDataType> C_dev;
    BRIDGE_HIP_CHECK(kFn, A_dev.allocate(elements_to_bytes<ADataType>(M * K)));
    BRIDGE_HIP_CHECK(kFn, B_dev.allocate(elements_to_bytes<BDataType>(K * N)));
    BRIDGE_HIP_CHECK(kFn, AQ_dev.allocate(elements_to_bytes<QDataType>(M)));
    BRIDGE_HIP_CHECK(kFn, BQ_dev.allocate(elements_to_bytes<QDataType>(N)));
    BRIDGE_HIP_CHECK(kFn, C_dev.allocate(elements_to_bytes<CDataType>(M * N)));

    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(kFn,
                     hipMemcpy(AQ_dev, AQ, elements_to_bytes<QDataType>(M), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(kFn,
                     hipMemcpy(BQ_dev, BQ, elements_to_bytes<QDataType>(N), hipMemcpyHostToDevice));
    BRIDGE_HIP_CHECK(kFn, hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // AQ is the per-row scale ([M,1], stride_AQ=1); BQ is the per-column scale
    // ([1,N], stride_BQ=1); QK_A == QK_B == 1 (no quant groups). Mirrors the
    // RowColQuant branch of run_gemm_quant_example.inc.
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;
    args.bq_ptr    = BQ_dev;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = 1;
    args.QK_B      = 1;
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = 1;
    args.stride_BQ = 1;

    return launch_and_copyback<SelectedKernel, CDataType>(
        kFn, args, C, C_dev, static_cast<std::size_t>(M) * N, time_ms);
}

} // extern "C"
