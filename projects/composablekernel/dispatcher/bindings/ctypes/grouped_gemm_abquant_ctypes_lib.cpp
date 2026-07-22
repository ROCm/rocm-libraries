// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm ABQuant Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration.
 * Kernel header is force-included at compile time via:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_abquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, QDataType, AccDataType
 *   AQuantGroupSize, BQuantGroupSize
 *
 * ABQuant: both A-side and B-side quantization active.
 *   AQ[ceil(M/aM), ceil(K/aK)] - A-side scale tensor (RowMajor)
 *   BQ[ceil(K/bK), ceil(N/bN)] - B-side scale tensor (RowMajor)
 *   Constraint: AQuantGroupSize::kK == BQuantGroupSize::kK
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType (shared AQ/BQ type), AccDataType,
//          AQuantGroupSize, BQuantGroupSize, SelectedKernel, KERNEL_NAME

#ifndef GFX_ARCH
#define GFX_ARCH "gfx950"
#endif

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

static std::shared_ptr<Dispatcher> g_dispatcher = nullptr;
static bool g_initialized                       = false;

#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t _err = (call);                                                              \
        if(_err != hipSuccess)                                                                 \
        {                                                                                      \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                     \
            return -1;                                                                         \
        }                                                                                      \
    }

extern "C" {

/**
 * Initialize dispatcher -- must be called before dispatcher_run_abquant_gemm.
 * Returns 0 on success, -1 on error.
 */
int dispatcher_initialize()
{
    if(g_initialized)
        return 0;

    KernelKey key;
    key.signature.dtype_a             = DataType::FP8;
    key.signature.dtype_b             = DataType::FP8;
    key.signature.dtype_c             = DataType::FP16;
    key.signature.dtype_acc           = DataType::FP32;
    key.signature.layout_a            = LayoutTag::RowMajor;
    key.signature.layout_b            = LayoutTag::ColMajor;
    key.signature.layout_c            = LayoutTag::RowMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape = {
        SelectedKernel::TileM, SelectedKernel::TileN, SelectedKernel::TileK};
    key.algorithm.wave_shape = {
        SelectedKernel::WarpM, SelectedKernel::WarpN, SelectedKernel::WarpK};
    key.algorithm.warp_tile_shape = {
        SelectedKernel::WarpTileM, SelectedKernel::WarpTileN, SelectedKernel::WarpTileK};
    key.algorithm.pipeline        = Pipeline::CompV3;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = SelectedKernel::BlockSize;
    key.algorithm.double_buffer   = SelectedKernel::DoubleSmemBuffer;
    key.algorithm.persistent      = false;
    key.algorithm.preshuffle      = SelectedKernel::PreshuffleB;
    key.algorithm.transpose_c     = SelectedKernel::TransposeC;
    key.algorithm.num_wave_groups = 1;
    key.gfx_arch                  = GFX_ARCH;

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, KERNEL_NAME);

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);

    g_dispatcher  = std::make_shared<Dispatcher>();
    g_initialized = true;
    return 0;
}

/**
 * Run ABQuantGrouped GEMM:
 *   C[M,N] = dequant(A[M,K], AQ[ceil(M/aM), ceil(K/aK)]) @ dequant(B[K,N], BQ[ceil(K/bK),
 * ceil(N/bN)])
 *
 * A, B, AQ, BQ, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A, B, AQ, BQ, C - host data pointers
 *   M, N, K          - matrix dimensions
 *   stride_A         - leading dim of A (row-major: K)
 *   stride_B         - leading dim of B (col-major: K)
 *   stride_AQ        - leading dim of AQ (row-major: ceil(K/aK))
 *   stride_BQ        - leading dim of BQ (row-major: ceil(N/bN))
 *   stride_C         - leading dim of C  (row-major: N)
 *   QK_A             - ceil(K / aquant_group_k)
 *   QM_A             - ceil(M / aquant_group_m)  (typically == M when aM=1)
 *   QK_B             - ceil(K / bquant_group_k)
 *   QN_B             - ceil(N / bquant_group_n)
 *   k_batch          - split-K factor (1 = no split)
 *   time_ms          - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_abquant_gemm(const void* A,
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
                                int64_t QM_A,
                                int64_t QK_B,
                                int64_t QN_B,
                                int k_batch,
                                float* time_ms)
{
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_abquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !BQ || !C)
    {
        std::cerr << "dispatcher_run_abquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_A <= 0 || QM_A <= 0 || QK_B <= 0 || QN_B <= 0)
    {
        std::cerr << "dispatcher_run_abquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Validate AQ dimensions
    {
        const int64_t exp_QK_A =
            (K + static_cast<int64_t>(AQuantGroupSize::kK) - 1) / AQuantGroupSize::kK;
        const int64_t exp_QM_A =
            (M + static_cast<int64_t>(AQuantGroupSize::kM) - 1) / AQuantGroupSize::kM;
        if(QK_A != exp_QK_A || QM_A != exp_QM_A)
        {
            std::cerr << "dispatcher_run_abquant_gemm: QK_A/QM_A mismatch. " << "Got (" << QK_A
                      << ", " << QM_A << "), " << "expected (" << exp_QK_A << ", " << exp_QM_A
                      << ")\n";
            return -1;
        }
    }

    // Validate BQ dimensions
    {
        const int64_t exp_QK_B =
            (K + static_cast<int64_t>(BQuantGroupSize::kK) - 1) / BQuantGroupSize::kK;
        const int64_t exp_QN_B =
            (N + static_cast<int64_t>(BQuantGroupSize::kN) - 1) / BQuantGroupSize::kN;
        if(QK_B != exp_QK_B || QN_B != exp_QN_B)
        {
            std::cerr << "dispatcher_run_abquant_gemm: QK_B/QN_B mismatch. " << "Got (" << QK_B
                      << ", " << QN_B << "), " << "expected (" << exp_QK_B << ", " << exp_QN_B
                      << ")\n";
            return -1;
        }
    }

    // Only packed (contiguous) layouts are supported.
    // AQ is RowMajor [QM_A, QK_A]: stride_AQ == QK_A
    // BQ is RowMajor [QK_B, QN_B]: stride_BQ == QN_B
    if(stride_A != K || stride_B != K || stride_AQ != QK_A || stride_BQ != QN_B || stride_C != N)
    {
        std::cerr << "dispatcher_run_abquant_gemm: non-packed strides not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_AQ=" << QK_A
                  << " stride_BQ=" << QN_B << " stride_C=" << N << "\n";
        return -1;
    }

    const ADataType* A_host  = static_cast<const ADataType*>(A);
    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);
    CDataType* C_host        = static_cast<CDataType*>(C);

    ADataType* A_dev  = nullptr;
    BDataType* B_dev  = nullptr;
    QDataType* AQ_dev = nullptr;
    QDataType* BQ_dev = nullptr;
    CDataType* C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(AQ_dev)
            (void)hipFree(AQ_dev);
        if(BQ_dev)
            (void)hipFree(BQ_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev, M * K * sizeof(ADataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&B_dev, K * N * sizeof(BDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&AQ_dev, QM_A * QK_A * sizeof(QDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&BQ_dev, QK_B * QN_B * sizeof(QDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&C_dev, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(hipMemcpy(A_dev, A_host, M * K * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, K * N * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(AQ_dev, AQ_host, QM_A * QK_A * sizeof(QDataType), hipMemcpyHostToDevice) !=
       hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(BQ_dev, BQ_host, QK_B * QN_B * sizeof(QDataType), hipMemcpyHostToDevice) !=
       hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemset(C_dev, 0, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    // Build QuantGemmHostArgs: both AQ and BQ active
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev; // A-side scale: active
    args.bq_ptr    = BQ_dev; // B-side scale: active
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = static_cast<ck_tile::index_t>(QK_A);
    args.QK_B      = static_cast<ck_tile::index_t>(QK_B);
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = static_cast<ck_tile::index_t>(stride_AQ);
    args.stride_BQ = static_cast<ck_tile::index_t>(stride_BQ);

    ck_tile::stream_config stream_cfg{nullptr, false, 0, 0, 1, false, false, 1};

    float exec_time = SelectedKernel::launch(args, stream_cfg);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_abquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    if(hipMemcpy(C_host, C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
}

/**
 * Return the compile-time KERNEL_NAME of the force-included kernel.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Initialize dispatcher (alias for consistency).
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Number of kernels registered (always 1 for single-kernel-per-.so model).
 */
int dispatcher_get_kernel_count() { return static_cast<int>(Registry::instance().size()); }

/**
 * Release dispatcher resources.
 */
void dispatcher_cleanup()
{
    g_dispatcher.reset();
    g_initialized = false;
}

} // extern "C"
