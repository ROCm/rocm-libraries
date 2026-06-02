// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file dispatcher_capi.h
 * @brief T2.2 — Multi-kernel C API for the CK Tile GEMM Dispatcher.
 *
 * Design goals
 * ============
 * 1. Flat C linkage (extern "C") so Python ctypes / cffi can load the shared
 *    library without a C++ ABI shim.
 * 2. All kernels registered at library-load time via the codegen-emitted master
 *    header (dispatcher_wrappers/register_all_kernels.hpp).
 * 3. Callers enumerate kernels by name (string) for debuggability, or by
 *    integer handle for performance.
 * 4. Memory ownership: all device buffers are owned by the caller.  The library
 *    never allocates GPU memory.
 * 5. Error propagation: every function returns DispatcherStatus.  Zero is
 *    success; negative values are error codes (see enum below).
 *
 * Usage sketch (Python ctypes)
 * ============================
 * @code
 *   import ctypes, numpy as np
 *   lib = ctypes.CDLL("./libdispatcher_gemm.so")
 *   count = ctypes.c_int(0)
 *   lib.dispatcher_kernel_count(ctypes.byref(count))
 *   names = (ctypes.c_char_p * count.value)()
 *   lib.dispatcher_kernel_names(names, count.value)
 *   handle = ctypes.c_int(-1)
 *   lib.dispatcher_kernel_by_name(names[0], ctypes.byref(handle))
 *   # ... allocate hip buffers, then:
 *   lib.dispatcher_run_gemm(handle, m, n, k, a_ptr, b_ptr, c_ptr,
 *                           k, k, n, 1, stream, ctypes.byref(elapsed_ms))
 * @endcode
 *
 * Build
 * =====
 * @code
 *   hipcc -fPIC -shared -o libdispatcher_gemm.so \
 *         dispatcher_capi.cpp \
 *         -I<ck_include_root> \
 *         -include <output_dir>/<kernel_set>/dispatcher_wrappers/register_all_kernels.hpp
 * @endcode
 *
 * The master header registers all compiled kernels at static-init time, so the
 * library contains every kernel listed in the generated dispatcher_wrappers/
 * directory.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// -------------------------------------------------------------------------- //
// Status codes
// -------------------------------------------------------------------------- //

typedef int DispatcherStatus;

#define DISPATCHER_OK              0    /**< Success */
#define DISPATCHER_ERR_NOT_FOUND  -1   /**< No kernel matches the selector */
#define DISPATCHER_ERR_INVALID    -2   /**< Bad argument (null pointer, bad dims) */
#define DISPATCHER_ERR_LAUNCH     -3   /**< HIP kernel launch failed */
#define DISPATCHER_ERR_OOM        -4   /**< hipMalloc failed inside the wrapper */

// -------------------------------------------------------------------------- //
// Kernel enumeration
// -------------------------------------------------------------------------- //

/**
 * @brief Return the total number of registered GEMM kernels.
 *
 * @param[out] count Set to the number of kernels in the registry.
 * @return DISPATCHER_OK on success.
 */
DispatcherStatus dispatcher_kernel_count(int* count);

/**
 * @brief Fill @p names with null-terminated kernel identifier strings.
 *
 * Identifiers are the output of KernelKey::encode_identifier() and match the
 * filenames emitted by unified_gemm_codegen.py.  The pointers returned are
 * owned by the library and remain valid for the lifetime of the process.
 *
 * @param[out] names  Caller-allocated array of at least @p max_names pointers.
 * @param[in]  max_names  Maximum entries to fill.
 * @return Number of names written (may be less than the registry size if
 *         max_names is smaller), or a negative DispatcherStatus on error.
 */
int dispatcher_kernel_names(const char** names, int max_names);

/**
 * @brief Look up a kernel by its identifier string.
 *
 * @param[in]  name    Null-terminated kernel identifier.
 * @param[out] handle  Integer handle valid for dispatcher_run_gemm().
 * @return DISPATCHER_OK or DISPATCHER_ERR_NOT_FOUND.
 */
DispatcherStatus dispatcher_kernel_by_name(const char* name, int* handle);

/**
 * @brief Return the identifier string for a handle obtained from
 *        dispatcher_kernel_by_name() or dispatcher_kernel_names().
 *
 * @param[in]  handle  Integer kernel handle.
 * @param[out] name    Set to a library-owned string pointer.
 * @return DISPATCHER_OK or DISPATCHER_ERR_NOT_FOUND.
 */
DispatcherStatus dispatcher_kernel_name_from_handle(int handle, const char** name);

// -------------------------------------------------------------------------- //
// GEMM execution
// -------------------------------------------------------------------------- //

/**
 * @brief Run a registered GEMM kernel on device buffers.
 *
 * Layout: row-column-row (rcr) — A is row-major (M×K), B is column-major
 * (K×N), C is row-major (M×N).  Strides are in units of elements.
 *
 * All pointers must be valid HIP device pointers allocated by the caller.
 * The library never allocates or frees GPU memory.
 *
 * @param[in]  handle      Kernel handle from dispatcher_kernel_by_name().
 * @param[in]  M, N, K     Problem dimensions.
 * @param[in]  a           Device pointer to A (fp16/bf16/fp8/int8, MxK row-major).
 * @param[in]  b           Device pointer to B (same dtype, KxN col-major).
 * @param[out] c           Device pointer to C (output dtype, MxN row-major).
 * @param[in]  stride_a    Row stride of A in elements (typically K for row-major).
 * @param[in]  stride_b    Col stride of B in elements (typically K for col-major).
 * @param[in]  stride_c    Row stride of C in elements (typically N for row-major).
 * @param[in]  split_k     Split-K factor (1 = no split-K).
 * @param[in]  stream      HIP stream to launch on (NULL = default stream).
 * @param[out] elapsed_ms  If non-NULL, GPU-timed kernel duration in milliseconds.
 * @return DISPATCHER_OK, DISPATCHER_ERR_NOT_FOUND, DISPATCHER_ERR_LAUNCH, or
 *         DISPATCHER_ERR_INVALID.
 */
DispatcherStatus dispatcher_run_gemm(
    int          handle,
    int          M,
    int          N,
    int          K,
    const void*  a,
    const void*  b,
    void*        c,
    int          stride_a,
    int          stride_b,
    int          stride_c,
    int          split_k,
    void*        stream,       /**< hipStream_t, passed as void* for C linkage */
    float*       elapsed_ms
);

/**
 * @brief Query whether a given (handle, M, N, K) combination is supported.
 *
 * Returns DISPATCHER_OK if the kernel's supports() predicate accepts the
 * problem, DISPATCHER_ERR_INVALID otherwise.  Callers should check this before
 * dispatcher_run_gemm() if they want a clean error rather than a launch failure.
 *
 * @param[in] handle  Kernel handle.
 * @param[in] M, N, K Problem dimensions.
 */
DispatcherStatus dispatcher_supports(int handle, int M, int N, int K);

// -------------------------------------------------------------------------- //
// Library lifecycle
// -------------------------------------------------------------------------- //

/**
 * @brief Return the library version string (e.g., "1.0.0-muozturk/dispatcher-te-parity").
 */
const char* dispatcher_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif
