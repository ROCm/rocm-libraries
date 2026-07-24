// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * AQuant (A-only quantized) GEMM ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_aquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, QDataType, AccDataType, QuantGroupSize
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used: AQuant kernels take QuantGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature used by
 * the dispatcher's registry backend.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 *
 * AQuant vs BQuant: here the *A* matrix is the quantized operand.  The scale
 * tensor AQ has shape [M, QK_A] (QK_A = ceil(K/gK)), aq_ptr is set and bq_ptr is
 * nullptr.  Its leading dimension follows AQLayout: row-major (rcr/rrr/crr) uses
 * stride_AQ=QK_A, while the ccr layout is column-major and uses stride_AQ=M
 * (see run_gemm_quant_example.inc get_default_stride, ~line 528).  For pk_int4 A the
 * raw values are permuted via permute_vectors_i4x4_b before the device copy (mirrors
 * run_gemm_quant_example.inc), and APreshuffleQuant kernels shuffle AQ via shuffle_aq
 * (row-major only -- ccr is excluded from the preshufflequant path by Old-TE).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType, AccDataType,
//          QuantGroupSize, SelectedKernel, KERNEL_NAME

// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// GPU architecture is derived from the running device at launch time (see the
// runtime check in dispatcher_run_aquant_gemm) rather than assumed at compile
// time -- do not hardcode a default architecture here.

static bool g_initialized = false;

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
 * Initialize the ctypes lib. Must be called before dispatcher_run_aquant_gemm.
 *
 * This library uses a single-kernel-per-.so model: SelectedKernel is
 * force-included at compile time and invoked directly via SelectedKernel::launch().
 * No dispatcher registry is involved -- AQuant kernels require QuantGemmHostArgs
 * which is incompatible with the GeneratedTileKernelInstance::run() signature that
 * the dispatcher's registry backend uses.
 *
 * Returns 0 on success.
 */
int dispatcher_initialize()
{
    if(g_initialized)
        return 0;
    g_initialized = true;
    return 0;
}

/**
 * Run AQuantGrouped GEMM: C[M,N] = dequant(A[M,K], AQ[M, ceil(K/gK)]) @ B[K,N]
 *
 * A, AQ, B, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A, AQ, B, C  - host data pointers
 *   M, N, K      - matrix dimensions
 *   stride_A     - leading dimension of A (row-major: K; col-major: M)
 *   stride_AQ    - leading dimension of AQ (row-major: QK_A; col-major/ccr: M)
 *   stride_B     - leading dimension of B (col-major: K; row-major: N)
 *   stride_C     - leading dimension of C (row-major: N)
 *   QK_A         - number of K-groups = ceil(K / quant_group_k)
 *   k_batch      - split-K factor (1 = no split)
 *   time_ms      - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_aquant_gemm(const void* A,
                               const void* AQ,
                               const void* B,
                               void* C,
                               int64_t M,
                               int64_t N,
                               int64_t K,
                               int64_t stride_A,
                               int64_t stride_AQ,
                               int64_t stride_B,
                               int64_t stride_C,
                               int64_t QK_A,
                               int k_batch,
                               float* time_ms)
{
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_aquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !AQ || !B || !C)
    {
        std::cerr << "dispatcher_run_aquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_A <= 0)
    {
        std::cerr << "dispatcher_run_aquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Derive the GPU architecture from the running device (do not assume one at
    // compile time) and reject unsupported archs, per review feedback.
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_aquant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0 &&
           arch.rfind("gfx90a", 0) != 0)
        {
            std::cerr << "dispatcher_run_aquant_gemm: unsupported GPU architecture '" << arch
                      << "' (supported: gfx90a, gfx942, gfx950)\n";
            return -1;
        }
    }

    // Validate that the caller's QK_A matches the compile-time quant group size baked
    // into this .so.  A mismatch means the AQ device buffer would be allocated with the
    // wrong size while the kernel indexes it with different strides.
    {
        const int64_t expected_QK_A =
            (K + static_cast<int64_t>(QuantGroupSize::kK) - 1) / QuantGroupSize::kK;
        if(QK_A != expected_QK_A)
        {
            std::cerr << "dispatcher_run_aquant_gemm: QK_A mismatch. Got " << QK_A << ", expected "
                      << expected_QK_A << " for K=" << K
                      << " with QuantGroupSize kK=" << QuantGroupSize::kK << "\n";
            return -1;
        }
    }

    // This implementation only supports packed (contiguous) layouts.  The expected
    // leading dimensions depend on the compile-time A/B/AQ layouts baked into the kernel.
    //   A  (ALayout) : row-major -> stride_A=K ; col-major -> stride_A=M
    //   B  (BLayout) : row-major -> stride_B=N ; col-major -> stride_B=K
    //   AQ (AQLayout): scale tensor [M, QK_A]. row-major -> stride_AQ=QK_A ;
    //                  col-major (ccr) -> stride_AQ=M.  Mirrors Old-TE
    //                  get_default_stride(M, QK_A, 0, is_row_major(aq_layout)) in
    //                  run_gemm_quant_example.inc (~line 528): the column-major branch
    //                  returns the row count M, not the K-group count QK_A.
    //   C  (CLayout) : row-major [M, N] -> stride_C=N (CLayout is always RowMajor)
    {
        constexpr bool a_row  = std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>;
        constexpr bool b_row  = std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
        constexpr bool aq_row = std::is_same_v<AQLayout, ck_tile::tensor_layout::gemm::RowMajor>;
        const int64_t exp_stride_A  = a_row ? K : M;
        const int64_t exp_stride_B  = b_row ? N : K;
        const int64_t exp_stride_AQ = aq_row ? QK_A : M;
        const int64_t exp_stride_C  = N;
        if(stride_A != exp_stride_A || stride_B != exp_stride_B || stride_AQ != exp_stride_AQ ||
           stride_C != exp_stride_C)
        {
            std::cerr << "dispatcher_run_aquant_gemm: non-packed strides are not supported. "
                      << "Expected stride_A=" << exp_stride_A << " stride_AQ=" << exp_stride_AQ
                      << " stride_B=" << exp_stride_B << " stride_C=" << exp_stride_C
                      << ", got stride_A=" << stride_A << " stride_AQ=" << stride_AQ
                      << " stride_B=" << stride_B << " stride_C=" << stride_C << "\n";
            return -1;
        }
    }

    const ADataType* A_host  = static_cast<const ADataType*>(A);
    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);
    const BDataType* B_host  = static_cast<const BDataType*>(B);
    CDataType* C_host        = static_cast<CDataType*>(C);

    ADataType* A_dev  = nullptr;
    QDataType* AQ_dev = nullptr;
    BDataType* B_dev  = nullptr;
    CDataType* C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(AQ_dev)
            (void)hipFree(AQ_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    // Allocate device buffers.
    // A may be a packed type (pk_int4_t): 2 logical values per byte.
    // elements_to_bytes<T>(n) handles the packed case via numeric_traits::PackedSize.
    if(hipMalloc(&A_dev, elements_to_bytes<ADataType>(M * K)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&AQ_dev, elements_to_bytes<QDataType>(M * QK_A)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&B_dev, elements_to_bytes<BDataType>(K * N)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&C_dev, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    // Copy A to device.  For pk_int4 A the raw i4x4 values must be permuted for the
    // device implementation -- mirrors run_gemm_quant_example.inc:758-763.
    if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
    {
        constexpr bool a_is_row = std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>;
        ck_tile::HostTensor<ADataType> a_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(M),
                                            static_cast<int>(K),
                                            static_cast<int>(stride_A),
                                            ck_tile::bool_constant<a_is_row>{}));
        std::copy(A_host, A_host + M * K, a_h.begin());
        ck_tile::permute_vectors_i4x4_b(a_h);
        if(hipMemcpy(
               A_dev, a_h.data(), elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    else
    {
        if(hipMemcpy(A_dev, A_host, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup();
            return -1;
        }
    }

    // Apply AQ preshuffle when required -- mirrors run_gemm_quant_example.inc:746-751.
    // APreshuffleQuant reorders AQ in host memory (shuffle_aq) before the device copy
    // so the kernel finds the scale values in the interleaved layout it expects.
    if constexpr(SelectedKernel::APreshuffleQuant)
    {
        // shuffle_aq assumes a row-major AQ descriptor.  This holds because Old-TE
        // rejects the ccr (column-major AQ) layout for the preshufflequant path, so
        // every APreshuffleQuant kernel this .so can host has AQLayout == RowMajor.
        // The assert is gated on APreshuffleQuant so it is only evaluated for kernels
        // that actually take this branch (decode-path ccr kernels are unaffected).
        static_assert(!SelectedKernel::APreshuffleQuant ||
                          std::is_same_v<AQLayout, ck_tile::tensor_layout::gemm::RowMajor>,
                      "APreshuffleQuant requires a row-major AQ layout (ccr is excluded "
                      "from the preshufflequant path); shuffle_aq below assumes row-major");
        constexpr int block_aq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(QuantGroupSize::kK);
        ck_tile::HostTensor<QDataType> aq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(M),
                                            static_cast<int>(QK_A),
                                            static_cast<int>(QK_A),
                                            ck_tile::bool_constant<true>{} /*row-major*/));
        std::copy(AQ_host, AQ_host + M * QK_A, aq_h.begin());
        auto aq_shuffled = ck_tile::shuffle_aq(&aq_h, block_aq_k);
        if(hipMemcpy(AQ_dev,
                     aq_shuffled.data(),
                     elements_to_bytes<QDataType>(M * QK_A),
                     hipMemcpyHostToDevice) != hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    else
    {
        if(hipMemcpy(
               AQ_dev, AQ_host, elements_to_bytes<QDataType>(M * QK_A), hipMemcpyHostToDevice) !=
           hipSuccess)
        {
            cleanup();
            return -1;
        }
    }
    if(hipMemcpy(B_dev, B_host, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice) !=
       hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    // Build QuantGemmHostArgs (bq_ptr = nullptr, QK_B = 0, stride_BQ = 0 for AQuant-only)
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;
    args.bq_ptr    = nullptr;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = static_cast<ck_tile::index_t>(QK_A);
    args.QK_B      = 0;
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = static_cast<ck_tile::index_t>(stride_AQ);
    args.stride_BQ = 0;

    const bool do_time = (time_ms != nullptr);
    // When timing is requested use GPU timer with warmup (cold_niters=3, nrepeat=10).
    // Otherwise run once with no overhead.
    ck_tile::stream_config stream_cfg{
        nullptr,          // stream_id_
        do_time,          // time_kernel_
        0,                // log_level_
        do_time ? 3 : 0,  // cold_niters_
        do_time ? 10 : 1, // nrepeat_
        do_time,          // is_gpu_timer_
        false,            // flush_cache_
        1,                // rotating_count_
    };

    float exec_time = SelectedKernel::launch(args, stream_cfg);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_aquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back
    if(hipMemcpy(C_host, C_dev, elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost) !=
       hipSuccess)
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
 * Initialize dispatcher (alias kept for consistency with gemm_ctypes_lib).
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Number of kernels in this .so (always 1: the force-included SelectedKernel).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Release resources.
 */
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
