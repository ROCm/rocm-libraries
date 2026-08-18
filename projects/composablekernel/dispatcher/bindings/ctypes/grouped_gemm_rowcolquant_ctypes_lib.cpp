// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * GroupedGemm RowColQuant ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE
 * grouped_gemm_rowcolquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType
 *
 * Design: direct launch -- SelectedKernel::launch(vector<QuantGroupedGemmHostArgs>, stream_config,
 * kargs_ptr) is called directly. No dispatcher registry is used: RowColQuant kernels take
 * QuantGroupedGemmHostArgs, which is incompatible with the GeneratedTileKernelInstance::run()
 * signature used by the dispatcher's registry backend.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 * For grouped GEMM, all problem groups share A, B, AQ, BQ, C in a flat host buffer;
 * the caller specifies per-group M/N/K and the library packs them into the device.
 */

#include <hip/hip_runtime.h>
#include <atomic>
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

// HIP_CHECK calls cleanup() which must be a lambda in scope at every call site.
// All uses of this macro are inside dispatcher_run_rowcolquant_gemm, after the
// lambda is defined.
#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t _err = (call);                                                              \
        if(_err != hipSuccess)                                                                 \
        {                                                                                      \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                     \
            cleanup();                                                                         \
            return -1;                                                                         \
        }                                                                                      \
    }

static std::atomic<int> g_ref_count{0};

extern "C" {

/**
 * Initialize the ctypes lib. Must be called before dispatcher_run_rowcolquant_gemm.
 * Returns 0 on success.
 */
int dispatcher_initialize()
{
    int dev = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
    {
        std::cerr << "dispatcher_initialize: could not query device architecture\n";
        return -1;
    }
    // GFX_ARCH is injected at compile time by CMake (e.g. "gfx942" or "gfx950").
    // Validate that the runtime device matches the compiled kernel architecture so
    // that we don't attempt to launch a kernel image on a mismatched device.
    // gfx90a is intentionally excluded: fp8/bf8 CompV3 kernels require native FP8
    // hardware which gfx90a lacks (produces NaN without -DCK_USE_OCP_FP8).
    const std::string arch(props.gcnArchName);
    const std::string compiled_arch(GFX_ARCH);
    if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0 &&
       arch.rfind("gfx1250", 0) != 0)
    {
        std::cerr << "dispatcher_initialize: unsupported GPU architecture '" << arch
                  << "' (supported: gfx942, gfx950, gfx1250; fp8/bf8 kernels require native FP8 "
                     "hardware)\n";
        return -1;
    }
    if(arch.rfind(compiled_arch, 0) != 0)
    {
        std::cerr << "dispatcher_initialize: runtime device architecture '" << arch
                  << "' does not match compile-time GFX_ARCH '" << compiled_arch
                  << "'; this .so was compiled for a different device\n";
        return -1;
    }
    // Increment the reference count. Use fetch_add with release so the
    // device-property checks above are visible to any thread that later
    // reads g_ref_count with acquire ordering.
    g_ref_count.fetch_add(1, std::memory_order_release);
    return 0;
}

/**
 * Run RowColQuant Grouped GEMM: C[M,N] = dequant(A[M,K], AQ[M,1]) @ dequant(B[K,N], BQ[1,N])
 *
 * A, B, AQ, BQ, C are host pointers to flat packed arrays.
 * For a single-problem (num_groups=1) call, this is equivalent to a standard GEMM with
 * per-row A scales and per-column B scales.
 *
 * Parameters:
 *   A, B, AQ, BQ, C  - host data pointers (flat packed arrays)
 *   M, N, K          - matrix dimensions (single problem)
 *   stride_A         - leading dimension of A (row-major: K)
 *   stride_B         - leading dimension of B (col-major: K)
 *   stride_AQ        - ignored; present for ABI symmetry with other quant ops.
 *                      The kernel uses broadcast stride=0 for AQ and BQ.
 *   stride_BQ        - ignored; see stride_AQ.
 *   stride_C         - leading dimension of C (row-major: N)
 *   QK_A             - number of AQ elements (== M); used only for buffer sizing.
 *   QK_B             - number of BQ elements (== N); used only for buffer sizing.
 *   k_batch          - split-K factor (1 = no split)
 *   time_ms          - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_rowcolquant_gemm(const void* A,
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
    // acquire: synchronise with the release fetch_add in dispatcher_initialize so
    // that all device-property checks performed there are visible here.
    if(g_ref_count.load(std::memory_order_acquire) <= 0)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !BQ || !C)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: invalid dimensions\n";
        return -1;
    }
    if(k_batch <= 0)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: k_batch must be >= 1, got " << k_batch
                  << " (k_batch is used as a divisor in split-K)\n";
        return -1;
    }
    // RowColQuant kernel indexes M AQ values (per-row) and N BQ values (per-col) directly.
    // Smaller counts cause device out-of-bounds reads; QK_A must equal M, QK_B must equal N.
    if(QK_A != M)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: QK_A must equal M (" << M
                  << ") for RowColQuant; got QK_A=" << QK_A << "\n";
        return -1;
    }
    if(QK_B != N)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: QK_B must equal N (" << N
                  << ") for RowColQuant; got QK_B=" << QK_B << "\n";
        return -1;
    }

    // Only packed (contiguous) layouts are supported for A, B, C.
    // stride_AQ and stride_BQ are unused here; the kernel uses broadcast strides (0).
    //
    // B layout (rcr = row-major A, column-major B, row-major C):
    //   B is stored column-major (Fortran order), shape [K, N].
    //   The leading dimension of a column-major [K, N] matrix is K (the number of
    //   rows), so stride_B == K for a packed column-major B.  This is NOT the same
    //   as row-major stride which would be N.  A row-major (C-contiguous) B passed
    //   with stride_B=K would cause the kernel to read the wrong elements.
    if(stride_A != K || stride_B != K || stride_C != N)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_C=" << N
                  << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const ADataType* A_host   = static_cast<const ADataType*>(A);
    const BDataType* B_host   = static_cast<const BDataType*>(B);
    const AQDataType* AQ_host = static_cast<const AQDataType*>(AQ);
    const BQDataType* BQ_host = static_cast<const BQDataType*>(BQ);
    CDataType* C_host         = static_cast<CDataType*>(C);

    ADataType* A_dev   = nullptr;
    BDataType* B_dev   = nullptr;
    AQDataType* AQ_dev = nullptr;
    BQDataType* BQ_dev = nullptr;
    CDataType* C_dev   = nullptr;
    void* kargs_dev    = nullptr;

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
        if(kargs_dev)
            (void)hipFree(kargs_dev);
    };

    // Allocate device buffers.
    HIP_CHECK(hipMalloc(&A_dev, elements_to_bytes<ADataType>(M * K)));
    HIP_CHECK(hipMalloc(&B_dev, elements_to_bytes<BDataType>(K * N)));
    // AQ: per-row scale [M, 1] -- QK_A rows, 1 col
    HIP_CHECK(hipMalloc(&AQ_dev, elements_to_bytes<AQDataType>(QK_A)));
    // BQ: per-col scale [1, N] -- 1 row, QK_B cols
    HIP_CHECK(hipMalloc(&BQ_dev, elements_to_bytes<BQDataType>(QK_B)));
    HIP_CHECK(hipMalloc(&C_dev, elements_to_bytes<CDataType>(M * N)));

    // Allocate kargs device buffer for grouped GEMM kernel args (1 group)
    HIP_CHECK(hipMalloc(&kargs_dev, sizeof(ck_tile::QuantGemmTransKernelArg)));

    // Copy inputs to device
    HIP_CHECK(hipMemcpy(A_dev, A_host, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(AQ_dev, AQ_host, elements_to_bytes<AQDataType>(QK_A), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(BQ_dev, BQ_host, elements_to_bytes<BQDataType>(QK_B), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // Build QuantGroupedGemmHostArgs for single-group launch.
    // The RowColQuant kernel treats AQ as a 1-D per-row vector and BQ as a 1-D per-col
    // vector. It uses broadcast strides (stride=0) to index them -- i.e. scale[0] is
    // reused across all columns (AQ) / all rows (BQ). QK_A and QK_B are ignored by the
    // kernel; M and N govern the loop bounds directly. Passing any non-zero stride causes
    // the kernel to step past the end of the scale buffer, producing garbage output.
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
        static_cast<ck_tile::index_t>(1), // QK_A: unused
        static_cast<ck_tile::index_t>(1), // QK_B: unused
        static_cast<ck_tile::index_t>(stride_A),
        static_cast<ck_tile::index_t>(stride_B),
        static_cast<ck_tile::index_t>(stride_C),
        static_cast<ck_tile::index_t>(0),  // stride_AQ: broadcast
        static_cast<ck_tile::index_t>(0)); // stride_BQ: broadcast

    const std::vector<ck_tile::QuantGroupedGemmHostArgs> gemm_descs = {args};

    const bool do_time = (time_ms != nullptr);
    // stream_config fields (positional): stream_id, time_kernel, log_level,
    //   cold_niters, nrepeat, do_log_perf, use_gpu_timer, rotating_count
    ck_tile::stream_config stream_cfg{
        nullptr,          // stream_id
        do_time,          // time_kernel
        0,                // log_level
        do_time ? 3 : 0,  // cold_niters
        do_time ? 10 : 1, // nrepeat
        do_time,          // do_log_perf
        false,            // use_gpu_timer
        1,                // rotating_count
    };

    float exec_time = -1.0f;
    try
    {
        exec_time = SelectedKernel::launch(gemm_descs, stream_cfg, kargs_dev);
    }
    catch(const std::exception& e)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: kernel launch threw: " << e.what() << "\n";
        cleanup();
        return -3;
    }
    catch(...)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: kernel launch threw unknown exception\n";
        cleanup();
        return -3;
    }

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_rowcolquant_gemm: kernel reported unsupported args\n";
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
 * Number of kernels in this .so (always 1: the force-included SelectedKernel).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Decrement the initialisation reference count. When it reaches zero the library
 * is considered uninitialised and the next call to dispatcher_run_rowcolquant_gemm
 * will fail until dispatcher_initialize() is called again.
 *
 * Using a reference count instead of a boolean allows multiple independent Python
 * wrappers to share the same loaded .so without one wrapper's destructor
 * invalidating another live wrapper.
 *
 * This function does not free any GPU memory or unload the library; those are
 * managed per-call inside dispatcher_run_rowcolquant_gemm.
 */
void dispatcher_cleanup()
{
    // Only decrement if already positive to guard against unpaired cleanup calls.
    int prev = g_ref_count.load(std::memory_order_relaxed);
    while(prev > 0 && !g_ref_count.compare_exchange_weak(
                          prev, prev - 1, std::memory_order_release, std::memory_order_relaxed))
        ; // retry on CAS failure
}

} // extern "C"
