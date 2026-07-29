// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm TensorQuant ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_tensor_quant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, QDataType, AccDataType, QuantGroupSize
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used: TensorQuant kernels take
 * QuantGemmHostArgs, which is incompatible with the GeneratedTileKernelInstance::run()
 * signature used by the dispatcher's registry backend.
 *
 * TensorQuant semantics (matches Old-TE gemm_quant_tensor.cpp):
 *   C[M,N] = (aq_scalar * bq_scalar) * (A[M,K] @ B[K,N])
 * where aq_scalar and bq_scalar are single per-tensor float scales. Both aq_ptr
 * and bq_ptr point at exactly ONE float; QK_A=QK_B=1 and stride_AQ=stride_BQ=1.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

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
// runtime check in dispatcher_run_tensor_quant_gemm) rather than assumed at
// compile time -- do not hardcode a default architecture here.

static bool g_initialized = false;

#define HIP_CHECK(call)                                                                         \
    {                                                                                           \
        hipError_t _err = (call);                                                               \
        if(_err != hipSuccess)                                                                  \
        {                                                                                       \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                      \
            cleanup();                                                                          \
            return -1;                                                                          \
        }                                                                                       \
    }

extern "C" {

/**
 * Initialize the ctypes lib. Must be called before dispatcher_run_tensor_quant_gemm.
 *
 * This library uses a single-kernel-per-.so model: SelectedKernel is
 * force-included at compile time and invoked directly via SelectedKernel::launch().
 * No dispatcher registry is involved -- TensorQuant kernels require
 * QuantGemmHostArgs which is incompatible with the GeneratedTileKernelInstance::run()
 * signature that the dispatcher's registry backend uses.
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
 * Run TensorQuant GEMM:
 *   C[M,N] = (AQ * BQ) * (A[M,K] @ B[K,N])
 * with AQ and BQ single per-tensor float scales.
 *
 * A, B, AQ, BQ, C are host pointers. This function manages device memory
 * internally.
 *
 * Parameters:
 *   A, B, C   - host data pointers (A row-major [M,K], B col-major [K,N], C row-major [M,N])
 *   AQ, BQ    - host pointers to a single float scale each
 *   M, N, K   - matrix dimensions
 *   stride_A  - leading dimension of A (row-major: K)
 *   stride_B  - leading dimension of B (col-major: K)
 *   stride_C  - leading dimension of C (row-major: N)
 *   k_batch   - split-K factor (1 = no split)
 *   time_ms   - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_tensor_quant_gemm(const void* A,
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
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_tensor_quant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !BQ || !C)
    {
        std::cerr << "dispatcher_run_tensor_quant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "dispatcher_run_tensor_quant_gemm: invalid dimensions\n";
        return -1;
    }

    // Derive the GPU architecture from the running device (do not assume one at
    // compile time) and reject unsupported archs, per review feedback.
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_tensor_quant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        // TensorQuant fp8/bf8 supported on gfx942 and gfx950 (where Old-TE runs).
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0)
        {
            std::cerr << "dispatcher_run_tensor_quant_gemm: unsupported GPU architecture '" << arch
                      << "' (supported: gfx942, gfx950)\n";
            return -1;
        }
    }

    // This implementation only supports packed (contiguous) layouts.
    // Device buffers are allocated and copied as M*K, K*N, M*N packed arrays.
    // Non-packed strides would cause the kernel to index into a differently-sized
    // buffer, producing incorrect results or out-of-bounds accesses.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << "dispatcher_run_tensor_quant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_C=" << N
                  << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_C=" << stride_C << "\n";
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

    // Allocate device buffers. AQ/BQ are single scalar scales.
    HIP_CHECK(hipMalloc(&A_dev,  elements_to_bytes<ADataType>(M * K)));
    HIP_CHECK(hipMalloc(&B_dev,  elements_to_bytes<BDataType>(K * N)));
    HIP_CHECK(hipMalloc(&AQ_dev, elements_to_bytes<QDataType>(1)));
    HIP_CHECK(hipMalloc(&BQ_dev, elements_to_bytes<QDataType>(1)));
    HIP_CHECK(hipMalloc(&C_dev,  elements_to_bytes<CDataType>(M * N)));

    // Copy inputs to device
    HIP_CHECK(hipMemcpy(A_dev,  A_host,  elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev,  B_host,  elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(AQ_dev, AQ_host, elements_to_bytes<QDataType>(1),     hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(BQ_dev, BQ_host, elements_to_bytes<QDataType>(1),     hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // Build QuantGemmHostArgs. TensorQuant reads *aq_ptr and *bq_ptr as single
    // scalar scales, so QK_A=QK_B=1 and stride_AQ=stride_BQ=1 (matches
    // run_gemm_quant_example.inc TensorQuant branch).
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
        std::cerr << "dispatcher_run_tensor_quant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back
    HIP_CHECK(hipMemcpy(C_host, C_dev, elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost));

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
