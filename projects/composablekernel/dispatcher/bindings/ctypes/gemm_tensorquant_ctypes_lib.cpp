// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * TensorQuant GEMM ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_tensorquant_ctypes_lib.cpp
 *
 * Force-include defines: SelectedKernel, KERNEL_NAME,
 *   ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType
 *
 * TensorQuant specifics:
 *   - AQ and BQ are single scalar values (float32), shared across the entire tensor
 *   - QK_A=1, QK_B=1: single K-group for both (entire K dimension is one group)
 *   - stride_AQ=1, stride_BQ=1: trivial strides for scalars
 *   - No preshuffle (APreshuffleQuant=false, BPreshuffleQuant=false)
 *   - Smallest possible scale buffers: 1 element each
 *
 * NOTE: Do NOT include dispatcher/include — conflicts with QuantGemmHostArgs.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

static bool g_initialized = false;

extern "C" {

int dispatcher_initialize()
{
    if(g_initialized) return 0;
    g_initialized = true;
    return 0;
}

/**
 * Run TensorQuant GEMM: C[M,N] = (A[M,K] * AQ_scalar) @ (B[K,N] * BQ_scalar)
 *
 * A           - host A matrix (quantized), row-major [M, K]
 * AQ_scalar   - pointer to single float32 — scalar A quantization factor
 * B           - host B matrix (quantized), col-major [K, N]
 * BQ_scalar   - pointer to single float32 — scalar B quantization factor
 * C           - host output C [M, N], row-major
 * stride_A    - leading dim of A (row-major: K)
 * stride_B    - leading dim of B (col-major: K)
 * stride_C    - N (row-major C)
 * k_batch     - split-K factor (1 = no split)
 * time_ms     - output execution time (may be NULL)
 */
int dispatcher_run_tensorquant_gemm(const void* A,
                                    const void* AQ_scalar,
                                    const void* B,
                                    const void* BQ_scalar,
                                    void*       C,
                                    int64_t     M,
                                    int64_t     N,
                                    int64_t     K,
                                    int64_t     stride_A,
                                    int64_t     stride_B,
                                    int64_t     stride_C,
                                    int         k_batch,
                                    float*      time_ms)
{
#ifndef CK_TILE_SINGLE_KERNEL_INCLUDE
    std::cerr << "dispatcher_run_tensorquant_gemm: library built without kernel\n";
    (void)A; (void)AQ_scalar; (void)B; (void)BQ_scalar; (void)C;
    (void)M; (void)N; (void)K;
    (void)stride_A; (void)stride_B; (void)stride_C;
    (void)k_batch; (void)time_ms;
    return -2;
#else
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !AQ_scalar || !B || !BQ_scalar || !C)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Runtime arch check
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
            std::cerr << "dispatcher_run_tensorquant_gemm: unsupported architecture '" << arch << "'\n";
            return -1;
        }
    }

    // Validate packed strides
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: non-packed strides not supported\n";
        return -1;
    }

    ADataType*  A_dev  = nullptr;
    AQDataType* AQ_dev = nullptr;
    BDataType*  B_dev  = nullptr;
    BQDataType* BQ_dev = nullptr;
    CDataType*  C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)  (void)hipFree(A_dev);
        if(AQ_dev) (void)hipFree(AQ_dev);
        if(B_dev)  (void)hipFree(B_dev);
        if(BQ_dev) (void)hipFree(BQ_dev);
        if(C_dev)  (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev,  elements_to_bytes<ADataType>(M * K)) != hipSuccess)
    { cleanup(); return -1; }
    // AQ and BQ are single scalar values (1 element each)
    if(hipMalloc(&AQ_dev, sizeof(AQDataType)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&B_dev,  elements_to_bytes<BDataType>(K * N)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&BQ_dev, sizeof(BQDataType)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&C_dev,  elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    if(hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(AQ_dev, AQ_scalar, sizeof(AQDataType), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(BQ_dev, BQ_scalar, sizeof(BQDataType), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    // Build QuantGemmHostArgs for TensorQuant.
    // QK_A=1, QK_B=1: single K-group (entire K range is one group).
    // stride_AQ=1, stride_BQ=1: trivial strides for scalar values.
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
    args.QK_A      = 1;  // single A scale
    args.QK_B      = 1;  // single B scale
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = 1;  // trivial stride for scalar AQ
    args.stride_BQ = 1;  // trivial stride for scalar BQ

    const bool do_time = (time_ms != nullptr);
    ck_tile::stream_config stream_cfg{
        nullptr, do_time, 0, do_time ? 3 : 0, do_time ? 10 : 1, do_time, false, 1,
    };

    float exec_time;
    try
    {
        exec_time = SelectedKernel::launch(args, stream_cfg);
    }
    catch(const std::exception& e)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: kernel threw: " << e.what() << "\n";
        cleanup();
        return -2;
    }

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_tensorquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    if(hipMemcpy(C, C_dev, elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost) != hipSuccess)
    { cleanup(); return -1; }

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
}

const char* dispatcher_get_kernel_name()
{
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
    return KERNEL_NAME;
#else
    return "";
#endif
}

int dispatcher_init() { return dispatcher_initialize(); }
int dispatcher_get_kernel_count() { return 1; }
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
