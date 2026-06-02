// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file dispatcher_capi.cpp
 * @brief T2.2 — Multi-kernel C API implementation for the CK Tile GEMM Dispatcher.
 *
 * This translation unit implements the flat extern "C" interface declared in
 * dispatcher_capi.h.  It is compiled into libdispatcher_gemm.so alongside the
 * codegen-emitted master registration header (register_all_kernels.hpp), which
 * registers every kernel into a process-global registry at static-init time.
 *
 * Build
 * =====
 *   hipcc -fPIC -shared -std=c++17 \
 *         -o libdispatcher_gemm.so \
 *         dispatcher_capi.cpp \
 *         -I<composablekernel/include> \
 *         -include <output_dir>/<kernel_set>/dispatcher_wrappers/register_all_kernels.hpp
 *
 * The -include flag forces the registration header to be compiled into this TU so
 * all static KernelRegistry::Register<> objects are instantiated before main().
 *
 * Design notes
 * ============
 * The registry is a process-global std::vector<KernelEntry> (not a map) so that
 * integer handles are stable: handle = index into the vector.  The vector is
 * populated by static initializers, so it is complete before any C API function
 * is called.
 *
 * Memory ownership: all device buffers are caller-owned.  This library never
 * calls hipMalloc / hipFree for data buffers.  The only internal allocation is the
 * kernel registry itself, which lives for the process lifetime.
 *
 * Thread safety: reads from the registry (enumerate / look up) are safe from any
 * thread once static initialization completes.  Concurrent writes to the same C
 * tensor from multiple threads are the caller's responsibility.
 */

#include "dispatcher_capi.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

// =========================================================================== //
// Internal kernel registry
// =========================================================================== //

namespace {

/**
 * @brief Per-kernel record stored in the global registry.
 *
 * The run_fn wraps a concrete SelectedKernel::launch() call with the stream
 * timing enabled.  It is a std::function so the lambda can capture the kernel
 * type by value without exposing C++ templates through the C API boundary.
 */
struct KernelEntry
{
    std::string name;

    // run_fn: execute C = A*B and return elapsed_ms. Returns non-zero HIP error on
    // failure.
    std::function<int(int M, int N, int K,
                      const void* a, const void* b, void* c,
                      int stride_a, int stride_b, int stride_c,
                      int split_k, hipStream_t stream, float* elapsed_ms)>
        run_fn;

    // supports_fn: return true if the kernel's supports() predicate accepts (M,N,K).
    std::function<bool(int M, int N, int K)> supports_fn;
};

// Process-global registry.  Kernels are appended by Register() below during
// static initialization (before main).
std::vector<KernelEntry>& registry()
{
    static std::vector<KernelEntry> r;
    return r;
}

} // namespace


// =========================================================================== //
// Registration helper (called by the codegen-emitted master header)
// =========================================================================== //

namespace ck_tile { namespace dispatcher {

/**
 * @brief Register one kernel with the global C API registry.
 *
 * Called exactly once per kernel type at static-init time, typically from
 * register_all_kernels.hpp's anonymous-namespace initializers.
 *
 * @tparam Kernel  A SelectedKernel struct exposing:
 *   - static float launch(const GemmHostArgs&, const stream_config&)
 *   - static bool  supports(int M, int N, int K)
 *   - static constexpr const char* name() [or KERNEL_NAME macro equivalent]
 */
template <typename Kernel>
struct Register
{
    explicit Register(const char* kernel_name)
    {
        KernelEntry entry;
        entry.name = kernel_name ? kernel_name : "(unnamed)";

        entry.run_fn = [](int M, int N, int K,
                          const void* a, const void* b, void* c,
                          int stride_a, int stride_b, int stride_c,
                          int split_k, hipStream_t stream, float* elapsed_ms) -> int
        {
            ck_tile::GemmHostArgs args(
                a, b, c,
                /*k_batch=*/split_k,
                M, N, K,
                stride_a, stride_b, stride_c);

            ck_tile::stream_config sc{};
            sc.stream_id_    = stream;
            sc.time_kernel_  = (elapsed_ms != nullptr);
            sc.nrepeat_      = 20;

            try
            {
                float t = Kernel::launch(args, sc);
                if(elapsed_ms)
                    *elapsed_ms = t;
                return 0;
            }
            catch(const std::exception&)
            {
                return DISPATCHER_ERR_LAUNCH;
            }
        };

        entry.supports_fn = [](int M, int N, int K) -> bool
        {
            return Kernel::supports(M, N, K);
        };

        registry().push_back(std::move(entry));
    }
};

}} // namespace ck_tile::dispatcher


// =========================================================================== //
// C API implementation
// =========================================================================== //

extern "C" {

DispatcherStatus dispatcher_kernel_count(int* count)
{
    if(!count)
        return DISPATCHER_ERR_INVALID;
    *count = static_cast<int>(registry().size());
    return DISPATCHER_OK;
}

int dispatcher_kernel_names(const char** names, int max_names)
{
    if(!names || max_names < 0)
        return DISPATCHER_ERR_INVALID;
    const auto& r = registry();
    const int n   = static_cast<int>(r.size());
    const int out = std::min(n, max_names);
    for(int i = 0; i < out; ++i)
        names[i] = r[static_cast<size_t>(i)].name.c_str();
    return out;
}

DispatcherStatus dispatcher_kernel_by_name(const char* name, int* handle)
{
    if(!name || !handle)
        return DISPATCHER_ERR_INVALID;
    const auto& r = registry();
    for(int i = 0; i < static_cast<int>(r.size()); ++i)
    {
        if(r[static_cast<size_t>(i)].name == name)
        {
            *handle = i;
            return DISPATCHER_OK;
        }
    }
    return DISPATCHER_ERR_NOT_FOUND;
}

DispatcherStatus dispatcher_kernel_name_from_handle(int handle, const char** name)
{
    if(!name)
        return DISPATCHER_ERR_INVALID;
    const auto& r = registry();
    if(handle < 0 || handle >= static_cast<int>(r.size()))
        return DISPATCHER_ERR_NOT_FOUND;
    *name = r[static_cast<size_t>(handle)].name.c_str();
    return DISPATCHER_OK;
}

DispatcherStatus dispatcher_supports(int handle, int M, int N, int K)
{
    const auto& r = registry();
    if(handle < 0 || handle >= static_cast<int>(r.size()))
        return DISPATCHER_ERR_NOT_FOUND;
    return r[static_cast<size_t>(handle)].supports_fn(M, N, K)
               ? DISPATCHER_OK
               : DISPATCHER_ERR_INVALID;
}

DispatcherStatus dispatcher_run_gemm(
    int         handle,
    int         M,
    int         N,
    int         K,
    const void* a,
    const void* b,
    void*       c,
    int         stride_a,
    int         stride_b,
    int         stride_c,
    int         split_k,
    void*       stream,
    float*      elapsed_ms)
{
    const auto& r = registry();
    if(handle < 0 || handle >= static_cast<int>(r.size()))
        return DISPATCHER_ERR_NOT_FOUND;
    if(!a || !b || !c)
        return DISPATCHER_ERR_INVALID;
    if(M <= 0 || N <= 0 || K <= 0 || split_k < 1)
        return DISPATCHER_ERR_INVALID;

    const hipStream_t s = reinterpret_cast<hipStream_t>(stream);

    const int rc = r[static_cast<size_t>(handle)].run_fn(
        M, N, K, a, b, c,
        stride_a, stride_b, stride_c,
        split_k, s, elapsed_ms);

    if(rc == DISPATCHER_ERR_LAUNCH)
        return DISPATCHER_ERR_LAUNCH;
    if(rc != 0)
        return DISPATCHER_ERR_LAUNCH;
    return DISPATCHER_OK;
}

const char* dispatcher_version(void)
{
    // Bumped whenever the C API surface changes. Format: semver-branch.
    return "1.0.0-muozturk/dispatcher-te-parity";
}

} // extern "C"
