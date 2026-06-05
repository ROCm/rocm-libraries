// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Stream-K GEMM Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration for the STREAM-K GEMM variant.
 * Kernel header included via -include at compile time.
 *
 * Stream-K is a single GEMM (one A/B/C, one M/N/K) like regular GEMM, so this
 * lib keeps the exact same C ABI as gemm_ctypes_lib.cpp -- ``dispatcher_run_gemm``
 * takes host A/B/C and M/N/K. The difference is internal: the generated launch
 * has a Stream-K-specific signature
 *
 *   static float launch(const ck_tile::StreamKHostArgs& args, const stream_config& stream);
 *
 * which allocates the reduction workspace internally (DeviceMem) and uses the
 * Atomic reduction strategy. The single-problem registry path
 * (g_dispatcher->run / GemmHostArgs) and the generated_tile_backend wrapper both
 * hard-code the plain GemmHostArgs launch, so this lib bypasses the registry and
 * calls SelectedKernel::launch(args, stream) directly, reporting the kernel name
 * from the compile-time KERNEL_NAME macro.
 *
 * Because the C ABI matches the regular lib, the Python side reuses
 * GemmDispatcherLib / GpuGemmRunner unchanged -- only the .so internals differ.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_streamk_gemm.so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_gemm(...)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

// Kernel header included via -include compiler flag (with CK_TILE_SINGLE_KERNEL_INCLUDE).
// Defines: ADataType, BDataType, CDataType, AccDataType, SelectedKernel, KERNEL_NAME
// and transitively brings in ck_tile::StreamKHostArgs and ck_tile::stream_config.

// GPU architecture - can be overridden via -DGFX_ARCH="gfx90a" at compile time
#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

static bool g_initialized = false;

extern "C" {

/**
 * Initialize the stream-k GEMM library.
 *
 * The stream-k path does not use the dispatcher/registry (it launches the
 * force-included kernel directly), so this is a lightweight no-op kept for ABI
 * parity with the regular GEMM lib. Returns 0 on success.
 */
int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}

/**
 * Initialize dispatcher (alias)
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Run a Stream-K GEMM on GPU by launching the force-included kernel directly.
 *
 * hipMalloc A/B/C, copy A and B host->device, memset C (the Atomic reduction
 * strategy accumulates into C, so it must start zeroed), build a
 * ck_tile::StreamKHostArgs with rcr default strides (stride_A=K, stride_B=K,
 * stride_C=N) and launch. The launch allocates the reduction workspace
 * internally and resets C between timed iterations. C is then copied back.
 *
 * Layout contract (rcr): A row-major MxK, B col-major KxN, C row-major MxN.
 *
 * Returns: 0 on success, -1 on HIP error / generic throw, -2 if the kernel
 * reports the arguments are unsupported.
 */
int dispatcher_run_gemm(
    const void* A, const void* B, void* C, int64_t M, int64_t N, int64_t K, float* time_ms)
{
    if(!g_initialized || !A || !B || !C || M <= 0 || N <= 0 || K <= 0)
    {
        return -1;
    }

    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host       = static_cast<CDataType*>(C);

    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;

    auto cleanup_gpu_mem = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev, M * K * sizeof(ADataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&B_dev, K * N * sizeof(BDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMalloc(&C_dev, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    if(hipMemcpy(A_dev, A_host, M * K * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, K * N * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }
    if(hipMemset(C_dev, 0, M * N * sizeof(CDataType)) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    // rcr default strides: A row-major (stride=K), B col-major (stride=K),
    // C row-major (stride=N). k_batch is fixed to 1 inside StreamKHostArgs.
    ck_tile::StreamKHostArgs args(static_cast<const void*>(A_dev),
                                  static_cast<const void*>(B_dev),
                                  static_cast<void*>(C_dev),
                                  static_cast<ck_tile::index_t>(M),
                                  static_cast<ck_tile::index_t>(N),
                                  static_cast<ck_tile::index_t>(K),
                                  /*stride_A=*/static_cast<ck_tile::index_t>(K),
                                  /*stride_B=*/static_cast<ck_tile::index_t>(K),
                                  /*stride_C=*/static_cast<ck_tile::index_t>(N));

    ck_tile::stream_config stream_cfg;
    stream_cfg.stream_id_      = nullptr;
    stream_cfg.time_kernel_    = true;
    stream_cfg.log_level_      = 0;
    stream_cfg.cold_niters_    = 3;
    stream_cfg.nrepeat_        = 10;
    stream_cfg.is_gpu_timer_   = true;
    stream_cfg.flush_cache_    = false;
    stream_cfg.rotating_count_ = 1;

    float exec_time = 0.0f;
    try
    {
        exec_time = SelectedKernel::launch(args, stream_cfg);
    }
    catch(const std::exception& e)
    {
        cleanup_gpu_mem();
        if(std::string(e.what()).find("not supported") != std::string::npos)
        {
            if(time_ms)
            {
                *time_ms = -1.0f;
            }
            return -2; // Arguments not supported by this kernel
        }
        return -1;
    }

    if(hipMemcpy(C_host, C_dev, M * N * sizeof(CDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup_gpu_mem();
        return -1;
    }

    if(time_ms)
    {
        *time_ms = exec_time;
    }

    cleanup_gpu_mem();
    return 0;
}

/**
 * Get kernel information (legacy single-kernel ABI).
 *
 * Returns the compile-time KERNEL_NAME of the force-included kernel header.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Get the name of the kernel at a given registry index (multi-kernel ABI).
 *
 * Each stream-k .so force-includes exactly one kernel header, so index 0 reports
 * KERNEL_NAME and any other index is out of range. Mirrors the regular GEMM lib's
 * name ABI so the Python bridge can use the same name-lookup path.
 * Returns 0 on success, -1 on bad args or out-of-range index.
 */
int dispatcher_get_kernel_name_at(int index, char* buffer, int buffer_size)
{
    if(!buffer || buffer_size <= 0 || index != 0)
    {
        return -1;
    }

    std::strncpy(buffer, KERNEL_NAME, static_cast<size_t>(buffer_size) - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

/**
 * Get the number of kernels in this .so (always 1 for the stream-k single-include lib).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Cleanup library resources (no-op; kept for ABI parity).
 */
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
