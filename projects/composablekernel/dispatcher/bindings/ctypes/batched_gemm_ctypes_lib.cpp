// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Batched GEMM Dispatcher ctypes Library
 *
 * Provides a C API for Python ctypes integration for the BATCHED GEMM bridge.
 *
 * Unlike gemm_ctypes_lib.cpp (single-problem GEMM), batched GEMM has a
 * divergent ABI: it carries a batch dimension with per-batch strides. The
 * registry / Dispatcher::run() path only knows the single-problem
 * (A, B, C, M, N, K) signature, so this library BYPASSES the registry and
 * launches the force-included kernel directly via
 * ``SelectedKernel::launch(ck_tile::BatchedGemmHostArgs{...}, stream)`` --
 * the same launch entry the Tile Engine batched_gemm benchmark uses. This
 * mirrors the registry-bypass pattern used by the stream-K bridge.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libbatched_gemm_....so")
 *   lib.dispatcher_init()
 *   lib.dispatcher_run_batched(A, B, C, M, N, K, batch_count,
 *                              stride_A, stride_B, stride_C,
 *                              batch_stride_A, batch_stride_B, batch_stride_C,
 *                              &time_ms)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

// Kernel header included via -include compiler flag under
// CK_TILE_SINGLE_KERNEL_INCLUDE. Defines: ADataType, BDataType, CDataType,
// AccDataType, ALayout, BLayout, CLayout, SelectedKernel, KERNEL_NAME.

// GPU architecture - can be overridden via -DGFX_ARCH="gfx90a" at compile time
#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

namespace {

// The batched bridge is single-kernel-per-.so: the force-included header fully
// determines the kernel. There is no registry to initialize, but the init entry
// is kept for ABI symmetry with the single-problem GEMM library so the Python
// runner can call initialize() uniformly.
bool g_initialized = false;

// Default (contiguous) stride for a row/col-major operand, matching
// ck_tile::get_default_stride used by the Tile Engine profiler: a value of 0
// means "packed", so derive it from the problem shape.
inline std::int64_t
default_stride(std::int64_t rows, std::int64_t cols, std::int64_t provided, bool row_major)
{
    if(provided > 0)
    {
        return provided;
    }
    return row_major ? cols : rows;
}

// Per-operand layout is emitted by the codegen as single-char strings
// GEMM_KEY_LAYOUT_A/B/C ("r"/"c"). The generated header only exports the data
// types (not the layout aliases) into the global namespace, so we read layout
// from the macros rather than std::is_same_v<ALayout, ...>.
inline bool operand_is_row_major(const char* layout_char) { return layout_char[0] == 'r'; }

#ifdef GEMM_KEY_LAYOUT_A
constexpr const char* kLayoutA = GEMM_KEY_LAYOUT_A;
constexpr const char* kLayoutB = GEMM_KEY_LAYOUT_B;
constexpr const char* kLayoutC = GEMM_KEY_LAYOUT_C;
#else
constexpr const char* kLayoutA = "r";
constexpr const char* kLayoutB = "c";
constexpr const char* kLayoutC = "r";
#endif

} // namespace

extern "C" {

/**
 * Initialize the library. No registry is used for batched GEMM (the kernel is
 * force-included), so this simply flips a flag. Returns 0 on success.
 */
int dispatcher_initialize()
{
    g_initialized = true;
    return 0;
}

int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Report the compile-time kernel name of the force-included batched kernel.
 * The batched bridge is always one kernel per .so.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Multi-kernel ABI shim: the batched .so exposes exactly one kernel, so index 0
 * returns KERNEL_NAME and every other index fails. Mirrors the single-problem
 * library so the shared Python wrapper can query names uniformly.
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

int dispatcher_get_kernel_count() { return 1; }

/**
 * Run a batched GEMM on the GPU via the force-included kernel.
 *
 * Takes HOST pointers and manages GPU memory internally (hipMalloc/hipMemcpy/
 * hipFree), matching the single-problem GEMM ABI. The per-batch strides let the
 * caller lay out A/B/C as [batch_count, rows, cols] tensors; a stride argument
 * of 0 falls back to the packed/default stride.
 *
 * Returns: 0 on success, -1 on any HIP/launch error.
 */
int dispatcher_run_batched(const void* A,
                           const void* B,
                           void* C,
                           std::int64_t M,
                           std::int64_t N,
                           std::int64_t K,
                           std::int64_t batch_count,
                           std::int64_t stride_A,
                           std::int64_t stride_B,
                           std::int64_t stride_C,
                           std::int64_t batch_stride_A,
                           std::int64_t batch_stride_B,
                           std::int64_t batch_stride_C,
                           float* time_ms)
{
    if(!g_initialized || !A || !B || !C || M <= 0 || N <= 0 || K <= 0 || batch_count <= 0)
    {
        if(time_ms)
        {
            *time_ms = -1.0f;
        }
        return -1;
    }

    // Resolve strides: 0 -> packed default (matches TE profiler /
    // ck_tile::get_default_stride behaviour), respecting each operand's own
    // row/col-major layout. For rcr this yields stride_A=K, stride_B=K,
    // stride_C=N.
    const std::int64_t sa  = default_stride(M, K, stride_A, operand_is_row_major(kLayoutA));
    const std::int64_t sb  = default_stride(K, N, stride_B, operand_is_row_major(kLayoutB));
    const std::int64_t sc  = default_stride(M, N, stride_C, operand_is_row_major(kLayoutC));
    const std::int64_t bsa = batch_stride_A > 0 ? batch_stride_A : M * K;
    const std::int64_t bsb = batch_stride_B > 0 ? batch_stride_B : K * N;
    const std::int64_t bsc = batch_stride_C > 0 ? batch_stride_C : M * N;

    // Total element counts across all batches.
    const std::int64_t a_elems = batch_stride_A > 0 ? bsa * batch_count : M * K * batch_count;
    const std::int64_t b_elems = batch_stride_B > 0 ? bsb * batch_count : K * N * batch_count;
    const std::int64_t c_elems = batch_stride_C > 0 ? bsc * batch_count : M * N * batch_count;

    const ADataType* A_host = static_cast<const ADataType*>(A);
    const BDataType* B_host = static_cast<const BDataType*>(B);
    CDataType* C_host       = static_cast<CDataType*>(C);

    ADataType* A_dev = nullptr;
    BDataType* B_dev = nullptr;
    CDataType* C_dev = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev, a_elems * sizeof(ADataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&B_dev, b_elems * sizeof(BDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMalloc(&C_dev, c_elems * sizeof(CDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(hipMemcpy(A_dev, A_host, a_elems * sizeof(ADataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemcpy(B_dev, B_host, b_elems * sizeof(BDataType), hipMemcpyHostToDevice) != hipSuccess)
    {
        cleanup();
        return -1;
    }
    if(hipMemset(C_dev, 0, c_elems * sizeof(CDataType)) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    float exec_time = -1.0f;
    try
    {
        // k_batch (split-K) is fixed to 1 for the batched bridge, matching the
        // Tile Engine batched_gemm default.
        ck_tile::BatchedGemmHostArgs args{A_dev,
                                          B_dev,
                                          C_dev,
                                          /*k_batch=*/1,
                                          static_cast<ck_tile::index_t>(M),
                                          static_cast<ck_tile::index_t>(N),
                                          static_cast<ck_tile::index_t>(K),
                                          static_cast<ck_tile::index_t>(sa),
                                          static_cast<ck_tile::index_t>(sb),
                                          static_cast<ck_tile::index_t>(sc),
                                          static_cast<ck_tile::index_t>(bsa),
                                          static_cast<ck_tile::index_t>(bsb),
                                          static_cast<ck_tile::index_t>(bsc),
                                          static_cast<ck_tile::index_t>(batch_count)};

        const ck_tile::stream_config stream{nullptr, /*time_kernel=*/true};
        exec_time = SelectedKernel::launch(args, stream);
    }
    catch(...)
    {
        cleanup();
        return -1;
    }

    if(hipMemcpy(C_host, C_dev, c_elems * sizeof(CDataType), hipMemcpyDeviceToHost) != hipSuccess)
    {
        cleanup();
        return -1;
    }

    if(time_ms)
    {
        *time_ms = exec_time;
    }

    cleanup();
    return 0;
}

void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
