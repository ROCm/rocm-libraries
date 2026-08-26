// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm RowColQuant ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE
 *   grouped_gemm_rowcolquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, AQDataType, BQDataType, AccDataType.
 *
 * Direct launch -- SelectedKernel::launch(vector<QuantGroupedGemmHostArgs>,
 * stream_config, kargs_ptr, preprocess) is called directly. No dispatcher
 * registry is used: RowColQuant grouped kernels take QuantGroupedGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature
 * used by the dispatcher's registry backend.
 *
 * RowColQuant = per-row scale of A (AQ, [M, 1]) plus per-column scale of B
 * (BQ, [1, N]). There is no quant group size. The whole run() body is
 * quant_bridge::run_scalar_quant_grouped_gemm(), shared verbatim with grouped
 * tensorquant; this file holds the exported entry point and the two op-specific
 * values (the scale extents and the scale stride written into the host args).
 *
 * Each call launches a single problem (num_groups=1). The "grouped" in the name
 * refers to the QuantGroupedGemmHostArgs kernel contract, not multi-group
 * batching by this ABI.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include <cstdint>

#include "quant_bridge_common.hpp"

extern "C" {

QUANT_BRIDGE_C_API()

/**
 * Run RowColQuant Grouped GEMM:
 *   C[M,N] = (A[M,K] * AQ[M,1]) @ (B[K,N] * BQ[1,N])
 *
 * A, B, AQ, BQ, C are host pointers to flat packed arrays; device memory is
 * managed internally.
 *
 * Parameters:
 *   M, N, K              - matrix dimensions (single problem)
 *   stride_A / stride_B  - leading dims (row-major A: K; column-major B: K)
 *   stride_C             - leading dim of C (row-major: N)
 *   stride_AQ, stride_BQ - must be 1. Present for ABI symmetry with the other
 *                          quant ops; the kernel hardwires the scale strides and
 *                          never reads these. Other values are rejected rather
 *                          than silently ignored.
 *   QK_A / QK_B          - number of AQ / BQ elements (== M and == N).
 *   k_batch              - split-K factor (1 = no split)
 *   time_ms              - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_grouped_rowcolquant_gemm(const void* A,
                                            const void* B,
                                            const void* AQ,
                                            const void* BQ,
                                            void* C,
                                            int64_t M,
                                            int64_t N,
                                            int64_t K,
                                            int64_t stride_A,
                                            int64_t stride_B,
                                            int64_t stride_AQ,
                                            int64_t stride_BQ,
                                            int64_t stride_C,
                                            int64_t QK_A,
                                            int64_t QK_B,
                                            int k_batch,
                                            float* time_ms)
{
    // RowColQuant carries one scale per A row and one per B column, so the scale
    // buffers are M and N elements. The 0 written into the host args' stride_AQ /
    // stride_BQ preserves this bridge's existing value; the kernel builds its
    // AQ/BQ views with literal strides and does not read the field.
    return quant_bridge::run_scalar_quant_grouped_gemm<SelectedKernel,
                                                       ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AQDataType,
                                                       BQDataType>(
        "dispatcher_run_grouped_rowcolquant_gemm",
        bridge_initialized(),
        A,
        B,
        AQ,
        BQ,
        C,
        M,
        N,
        K,
        stride_A,
        stride_B,
        stride_AQ,
        stride_BQ,
        stride_C,
        QK_A,
        QK_B,
        /*expected_qk_a=*/M,
        /*expected_qk_b=*/N,
        /*args_scale_stride=*/0,
        k_batch,
        time_ms);
}

} // extern "C"
