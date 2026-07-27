// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * AQuant GEMM ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_aquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, AQDataType, AccDataType, QuantGroupSize, GroupSizeK
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used: AQuant kernels take QuantGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature used by
 * the dispatcher's registry backend.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 *
 * NOTE: Do NOT include dispatcher/include here — it pulls in generated_tile_backend.hpp
 * which conflicts with QuantGemmHostArgs. Include only ck_tile/host headers.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>

#include "ck_tile/host/tensor_shuffle_utils.hpp"

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, AQDataType, AccDataType,
//          QuantGroupSize, GroupSizeK, SelectedKernel, KERNEL_NAME

// Compute byte count for N logical elements of type T.
// Packed types (pk_int4_t, pk_fp4_t) have PackedSize=2 so two logical values
// occupy one byte; for all other types PackedSize=1.
template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

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
 * Run AQuant GEMM: C[M,N] = dequant(A[M,K], AQ[M, QK_A]) @ B[K,N]
 *
 * A, AQ, B, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A           - host A matrix (quantized)
 *   AQ          - host A scale tensor [M, QK_A], row-major, dtype=float32
 *   B           - host B matrix (not quantized)
 *   C           - host output C matrix [M, N], row-major
 *   M, N, K     - matrix dimensions
 *   stride_A    - leading dimension of A (row-major: K; col-major: M)
 *   stride_AQ   - leading dimension of AQ (row-major: QK_A = ceil(K/group_k))
 *   stride_B    - leading dimension of B (col-major: K; row-major: N)
 *   stride_C    - leading dimension of C (row-major: N)
 *   QK_A        - number of K-groups = ceil(K / quant_group_k)
 *   k_batch     - split-K factor (1 = no split)
 *   time_ms     - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_aquant_gemm(const void* A,
                               const void* AQ,
                               const void* B,
                               void*       C,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               int64_t     stride_A,
                               int64_t     stride_AQ,
                               int64_t     stride_B,
                               int64_t     stride_C,
                               int64_t     QK_A,
                               int         k_batch,
                               float*      time_ms)
{
#ifndef CK_TILE_SINGLE_KERNEL_INCLUDE
    std::cerr << "dispatcher_run_aquant_gemm: library built without a kernel; unsupported\n";
    (void)A; (void)AQ; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)stride_A; (void)stride_AQ; (void)stride_B; (void)stride_C;
    (void)QK_A; (void)k_batch; (void)time_ms;
    return -2;
#else
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

    // Derive the GPU architecture from the running device and reject unsupported archs.
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

    // Validate QK_A matches the compile-time quant group size baked into this .so.
    {
        const int64_t expected_QK_A =
            (K + static_cast<int64_t>(GroupSizeK) - 1) / static_cast<int64_t>(GroupSizeK);
        if(QK_A != expected_QK_A)
        {
            std::cerr << "dispatcher_run_aquant_gemm: QK_A mismatch. Got " << QK_A
                      << ", expected " << expected_QK_A
                      << " for K=" << K << " GroupSizeK=" << GroupSizeK << "\n";
            return -1;
        }
    }

    // Only packed (contiguous) strides are supported.
    // For row-major A: stride = K; for col-major A: stride = M.
    // stride_AQ = QK_A (row-major AQ [M, QK_A]).
    // stride_B col-major: stride = K; stride_C row-major: stride = N.
    const bool a_is_col_major = (stride_A == M);
    const int64_t expected_stride_A  = a_is_col_major ? M : K;
    const int64_t expected_stride_AQ = QK_A;
    const int64_t expected_stride_B  = K;
    const int64_t expected_stride_C  = N;
    if(stride_A != expected_stride_A || stride_AQ != expected_stride_AQ ||
       stride_B != expected_stride_B || stride_C != expected_stride_C)
    {
        std::cerr << "dispatcher_run_aquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << expected_stride_A
                  << " stride_AQ=" << expected_stride_AQ
                  << " stride_B=" << expected_stride_B
                  << " stride_C=" << expected_stride_C
                  << ", got stride_A=" << stride_A
                  << " stride_AQ=" << stride_AQ
                  << " stride_B=" << stride_B
                  << " stride_C=" << stride_C << "\n";
        return -1;
    }

    ADataType*  A_dev  = nullptr;
    AQDataType* AQ_dev = nullptr;
    BDataType*  B_dev  = nullptr;
    CDataType*  C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)  (void)hipFree(A_dev);
        if(AQ_dev) (void)hipFree(AQ_dev);
        if(B_dev)  (void)hipFree(B_dev);
        if(C_dev)  (void)hipFree(C_dev);
    };

    // Allocate device buffers.
    // A may be a packed type (pk_int4_t): elements_to_bytes handles the PackedSize.
    if(hipMalloc(&A_dev,  elements_to_bytes<ADataType>(M * K)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&AQ_dev, elements_to_bytes<AQDataType>(M * QK_A)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&B_dev,  elements_to_bytes<BDataType>(K * N)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&C_dev,  elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    // Copy inputs to device.
    if(hipMemcpy(A_dev, static_cast<const ADataType*>(A),
                 elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }

    // AQ preshuffle: when APreshuffleQuant=true, shuffle the AQ scale tensor on host
    // before device copy (mirrors bquant's shuffle_bq pattern).
    if constexpr(SelectedKernel::APreshuffleQuant)
    {
        constexpr int block_aq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(GroupSizeK);
        ck_tile::HostTensor<AQDataType> aq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(M),
                                            static_cast<int>(QK_A),
                                            static_cast<int>(QK_A),
                                            ck_tile::bool_constant<true>{} /*row-major*/));
        std::copy(static_cast<const AQDataType*>(AQ),
                  static_cast<const AQDataType*>(AQ) + M * QK_A,
                  aq_h.begin());
        auto aq_shuffled = ck_tile::shuffle_aq(&aq_h, block_aq_k);
        if(hipMemcpy(AQ_dev,
                     aq_shuffled.data(),
                     elements_to_bytes<AQDataType>(M * QK_A),
                     hipMemcpyHostToDevice) != hipSuccess)
        { cleanup(); return -1; }
    }
    else
    {
        if(hipMemcpy(AQ_dev, static_cast<const AQDataType*>(AQ),
                     elements_to_bytes<AQDataType>(M * QK_A),
                     hipMemcpyHostToDevice) != hipSuccess)
        { cleanup(); return -1; }
    }

    if(hipMemcpy(B_dev, static_cast<const BDataType*>(B),
                 elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }

    if(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

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
    ck_tile::stream_config stream_cfg{
        nullptr,
        do_time,
        0,
        do_time ? 3 : 0,
        do_time ? 10 : 1,
        do_time,
        false,
        1,
    };

    float exec_time = SelectedKernel::launch(args, stream_cfg);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_aquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back to host.
    if(hipMemcpy(static_cast<CDataType*>(C), C_dev,
                 elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost) != hipSuccess)
    { cleanup(); return -1; }

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
}

/**
 * Return the compile-time KERNEL_NAME of the force-included kernel.
 */
const char* dispatcher_get_kernel_name()
{
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
    return KERNEL_NAME;
#else
    return "";
#endif
}

/**
 * Initialize dispatcher (alias for consistency with other ctypes libs).
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
