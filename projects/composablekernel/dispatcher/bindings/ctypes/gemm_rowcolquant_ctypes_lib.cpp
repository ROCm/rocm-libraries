// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * RowColQuant GEMM ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_rowcolquant_ctypes_lib.cpp
 *
 * Force-include defines: SelectedKernel, KERNEL_NAME,
 *   ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType
 *
 * RowColQuant specifics:
 *   - AQ is a row-scale vector [M, 1]: QK_A = 1, stride_AQ = 1
 *   - BQ is a col-scale vector [1, N]: QK_B = 1, stride_BQ = N
 *   - No preshuffle (APreshuffleQuant=false, BPreshuffleQuant=false)
 *   - No QuantGroupSize validation (scales are per-row/per-col, not blocked)
 *
 * NOTE: Do NOT include dispatcher/include here — conflicts with QuantGemmHostArgs.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

// Only pull in CK host headers needed for template helpers
// (no dispatcher includes to avoid QuantGemmHostArgs conflict)

template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    using namespace ck_tile;
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
 * Run RowColQuant GEMM: C[M,N] = (A[M,K] * AQ[M,1]) @ (B[K,N] * BQ[1,N])
 *
 * A           - host A matrix (quantized), row-major [M, K]
 * AQ          - host A row-scale vector [M, 1], dtype=float32, stride=1
 * B           - host B matrix (quantized), col-major [K, N]
 * BQ          - host B col-scale vector [1, N], dtype=float32, stride=N
 * C           - host output C [M, N], row-major
 * stride_A    - leading dim of A (row-major: K)
 * stride_B    - leading dim of B (col-major: K)
 * stride_C    - N (row-major C)
 * k_batch     - split-K factor (1 = no split)
 * time_ms     - output execution time (may be NULL)
 *
 * AQ and BQ strides are fixed internally: stride_AQ=1, stride_BQ=N.
 */
int dispatcher_run_rowcolquant_gemm(const void* A,
                                    const void* AQ,
                                    const void* B,
                                    const void* BQ,
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
    std::cerr << "dispatcher_run_rowcolquant_gemm: library built without kernel\n";
    (void)A; (void)AQ; (void)B; (void)BQ; (void)C;
    (void)M; (void)N; (void)K;
    (void)stride_A; (void)stride_B; (void)stride_C;
    (void)k_batch; (void)time_ms;
    return -2;
#else
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !AQ || !B || !BQ || !C)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Runtime arch check
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_rowcolquant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0 &&
           arch.rfind("gfx90a", 0) != 0)
        {
            std::cerr << "dispatcher_run_rowcolquant_gemm: unsupported GPU architecture '"
                      << arch << "'\n";
            return -1;
        }
    }

    // Validate packed strides:
    // A row-major [M,K] -> stride=K; B col-major [K,N] -> stride=K; C row-major -> stride=N
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: non-packed strides not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_C=" << N
                  << ", got " << stride_A << " " << stride_B << " " << stride_C << "\n";
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

    // A may be a packed type (pk_int4_t etc.)
    if(hipMalloc(&A_dev,  elements_to_bytes<ADataType>(M * K)) != hipSuccess)
    { cleanup(); return -1; }
    // AQ: row-scale [M, 1] — M float32 values
    if(hipMalloc(&AQ_dev, sizeof(AQDataType) * M) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&B_dev,  elements_to_bytes<BDataType>(K * N)) != hipSuccess)
    { cleanup(); return -1; }
    // BQ: col-scale [1, N] — N float32 values
    if(hipMalloc(&BQ_dev, sizeof(BQDataType) * N) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&C_dev,  elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    if(hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(AQ_dev, AQ, sizeof(AQDataType) * M, hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemcpy(BQ_dev, BQ, sizeof(BQDataType) * N, hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    // Build QuantGemmHostArgs for RowColQuant.
    // QK_A=1, QK_B=1: each row has one A-scale; each col has one B-scale.
    // stride_AQ=1: AQ [M,1] row-major -> leading dim = 1.
    // stride_BQ=N: BQ [1,N] row-major -> leading dim = N.
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
    args.QK_A      = 1;  // AQ [M, 1]
    args.QK_B      = 1;  // BQ [1, N]
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = 1;  // AQ [M,1] row-major: stride = 1
    args.stride_BQ = static_cast<ck_tile::index_t>(N);  // BQ [1,N] row-major: stride = N

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
        std::cerr << "dispatcher_run_rowcolquant_gemm: kernel threw: " << e.what() << "\n";
        cleanup();
        return -2;
    }

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: kernel reported unsupported args\n";
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
