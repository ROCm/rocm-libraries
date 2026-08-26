// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm TensorQuant ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE
 *   grouped_gemm_tensorquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, AQDataType, BQDataType, AccDataType.
 *
 * Direct launch -- SelectedKernel::launch(vector<QuantGroupedGemmHostArgs>,
 * stream_config, kargs_ptr, preprocess) is called directly. No dispatcher
 * registry is used: TensorQuant grouped kernels take QuantGroupedGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature
 * used by the dispatcher's registry backend.
 *
 * TensorQuant = one scalar scale for all of A and one for all of B (QK_A=QK_B=1),
 * unlike RowColQuant which uses per-row A and per-column B scales. Neither has a
 * quant group size, so the whole run() body is
 * quant_bridge::run_scalar_quant_grouped_gemm(), shared verbatim with grouped
 * rowcolquant; this file holds the exported entry point and the two op-specific
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
 * Run TensorQuant Grouped GEMM:
 *   C[M,N] = (scale_A * A[M,K]) @ (scale_B * B[K,N])
 *
 * A, B, AQ, BQ, C are host pointers; device memory is managed internally.
 *
 * Parameters:
 *   M, N, K              - matrix dimensions (single problem)
 *   stride_A / stride_B  - leading dims (row-major A: K; column-major B: K)
 *   stride_C             - leading dim of C (row-major: N)
 *   stride_AQ, stride_BQ - leading dims of the scale buffers; must be 1.
 *   QK_A / QK_B          - number of AQ / BQ elements; must be 1.
 *   k_batch              - split-K factor (1 = no split)
 *   time_ms              - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_grouped_tensorquant_gemm(const void* A,
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
    // One scale covers the whole tensor, so each scale buffer holds exactly one
    // element and the host args carry a scale stride of 1. Under
    // QuantType::TensorQuant the kernel simply dereferences aq_ptr / bq_ptr and
    // does not read the stride; the 1 is kept so the host args stay meaningful.
    return quant_bridge::run_scalar_quant_grouped_gemm<SelectedKernel,
                                                       ADataType,
                                                       BDataType,
                                                       CDataType,
                                                       AQDataType,
                                                       BQDataType>(
        "dispatcher_run_grouped_tensorquant_gemm",
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
        /*expected_qk_a=*/1,
        /*expected_qk_b=*/1,
        /*args_scale_stride=*/1,
        k_batch,
        time_ms);
}

} // extern "C"
