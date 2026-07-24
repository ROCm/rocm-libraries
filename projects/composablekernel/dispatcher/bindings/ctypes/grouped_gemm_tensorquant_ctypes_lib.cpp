// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm TensorQuant ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_tensorquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType
 *
 * Design: direct launch -- SelectedKernel::launch(vector<QuantGroupedGemmHostArgs>, stream_config, kargs_ptr)
 * is called directly. No dispatcher registry is used: TensorQuant kernels take
 * QuantGroupedGemmHostArgs, which is incompatible with the GeneratedTileKernelInstance::run()
 * signature used by the dispatcher's registry backend.
 *
 * TensorQuant uses a single scalar scale per entire tensor (QK_A=1, QK_B=1), unlike
 * RowColQuant which uses per-row A scales and per-column B scales.
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
// Defines: ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType,
//          SelectedKernel, KERNEL_NAME

// Compute the byte count for N logical elements of type T.
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
 * Initialize the ctypes lib. Must be called before dispatcher_run_tensorquant_gemm.
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
 * Run TensorQuant Grouped GEMM: C[M,N] = (scale_A * A[M,K]) @ (scale_B * B[K,N])
 *
 * A, B, AQ, BQ, C are host pointers to flat packed arrays.
 * TensorQuant: AQ is a single scalar per A tensor, BQ is a single scalar per B tensor.
 *
 * Parameters:
 *   A, B, AQ, BQ, C  - host data pointers
 *   M, N, K          - matrix dimensions
 *   stride_A         - leading dimension of A (row-major: K)
 *   stride_B         - leading dimension of B (col-major: K)
 *   stride_AQ        - leading dimension of AQ (1 for tensor-wise scale)
 *   stride_BQ        - leading dimension of BQ (1 for tensor-wise scale)
 *   stride_C         - leading dimension of C (row-major: N)
 *   k_batch          - split-K factor (1 = no split)
 *   time_ms          - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_tensorquant_gemm(const void* A,
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
                                    int k_batch,
                                    float* time_ms)
{
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !BQ || !C)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Derive the GPU architecture from the running device and reject unsupported archs.
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_tensorquant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0 &&
           arch.rfind("gfx90a", 0) != 0)
        {
            std::cerr << "dispatcher_run_tensorquant_gemm: unsupported GPU architecture '" << arch
                      << "' (supported: gfx90a, gfx942, gfx950)\n";
            return -1;
        }
    }

    // Only packed (contiguous) layouts are supported.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_C=" << N
                  << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const ADataType*  A_host  = static_cast<const ADataType*>(A);
    const BDataType*  B_host  = static_cast<const BDataType*>(B);
    const AQDataType* AQ_host = static_cast<const AQDataType*>(AQ);
    const BQDataType* BQ_host = static_cast<const BQDataType*>(BQ);
    CDataType*        C_host  = static_cast<CDataType*>(C);

    ADataType*  A_dev     = nullptr;
    BDataType*  B_dev     = nullptr;
    AQDataType* AQ_dev    = nullptr;
    BQDataType* BQ_dev    = nullptr;
    CDataType*  C_dev     = nullptr;
    void*       kargs_dev = nullptr;

    auto cleanup = [&]() {
        if(A_dev)     (void)hipFree(A_dev);
        if(B_dev)     (void)hipFree(B_dev);
        if(AQ_dev)    (void)hipFree(AQ_dev);
        if(BQ_dev)    (void)hipFree(BQ_dev);
        if(C_dev)     (void)hipFree(C_dev);
        if(kargs_dev) (void)hipFree(kargs_dev);
    };

    // Allocate device buffers.
    if(hipMalloc(&A_dev, elements_to_bytes<ADataType>(M * K)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&B_dev, elements_to_bytes<BDataType>(K * N)) != hipSuccess)
    { cleanup(); return -1; }
    // TensorQuant: single scalar scale per tensor — 1 element each
    if(hipMalloc(&AQ_dev, elements_to_bytes<AQDataType>(1)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&BQ_dev, elements_to_bytes<BQDataType>(1)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&C_dev, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    // Allocate kargs device buffer for grouped GEMM kernel args (1 group)
    if(hipMalloc(&kargs_dev, sizeof(ck_tile::QuantGemmTransKernelArg)) != hipSuccess)
    { cleanup(); return -1; }

    // Copy inputs to device
    if(hipMemcpy(A_dev, A_host, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(B_dev, B_host, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(AQ_dev, AQ_host, elements_to_bytes<AQDataType>(1), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(BQ_dev, BQ_host, elements_to_bytes<BQDataType>(1), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    // Build QuantGroupedGemmHostArgs for single-group launch.
    // TensorQuant: QK_A=1 (one scale for whole A tensor), QK_B=1 (one scale for whole B tensor).
    // stride_AQ=1 (single element), stride_BQ=1 (single element).
    ck_tile::QuantGroupedGemmHostArgs args(
        A_dev,
        B_dev,
        C_dev,
        AQ_dev,
        BQ_dev,
        static_cast<ck_tile::index_t>(k_batch),
        static_cast<ck_tile::index_t>(M),
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(1), // QK_A: tensor-wise = 1
        static_cast<ck_tile::index_t>(1), // QK_B: tensor-wise = 1
        static_cast<ck_tile::index_t>(stride_A),
        static_cast<ck_tile::index_t>(stride_B),
        static_cast<ck_tile::index_t>(stride_C),
        static_cast<ck_tile::index_t>(1), // stride_AQ: tensor-wise
        static_cast<ck_tile::index_t>(1)  // stride_BQ: tensor-wise
    );

    const std::vector<ck_tile::QuantGroupedGemmHostArgs> gemm_descs = {args};

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

    float exec_time = SelectedKernel::launch(gemm_descs, stream_cfg, kargs_dev);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back
    if(hipMemcpy(C_host, C_dev, elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost) != hipSuccess)
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
 * Initialize dispatcher (alias kept for consistency).
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
