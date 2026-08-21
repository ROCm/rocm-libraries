// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Shared infrastructure for the block-scale quant GEMM ctypes bridges.
 *
 * The five per-op bridges (tensor_quant, rowcolquant, aquant, abquant, bquant)
 * each compile one kernel per .so, force-including the generated kernel header:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_<op>_ctypes_lib.cpp
 *
 * Because every bridge is its own translation unit, this common layer is
 * header-only: the templates / inline helpers below each get their own copy per
 * .so, so there is no shared library and no ODR concern. It centralizes the
 * infrastructure that used to be copy-pasted into all five sources -- device
 * memory management, GPU-arch validation, kernel-launch timing, init/cleanup,
 * and the exported C boilerplate -- so a fix to any of those happens exactly
 * once. Op-specific argument construction and shuffle behavior stay in the
 * per-op files.
 *
 * The generated kernel header (force-included before this one) must already
 * provide ck_tile (numeric_traits, stream_config, QuantGemmHostArgs, index_t)
 * and the KERNEL_NAME macro.
 */

#ifndef CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP
#define CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>

namespace quant_bridge {

// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// RAII owner for a device allocation. Frees on scope exit, which removes the
// hand-written `cleanup` lambda that every bridge used to duplicate: any early
// return (including from BRIDGE_HIP_CHECK) releases every buffer automatically.
template <typename T>
struct DeviceBuffer
{
    T* ptr = nullptr;

    DeviceBuffer()                               = default;
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if(ptr)
            (void)hipFree(ptr);
    }

    // Allocate `bytes` bytes; returns the hipMalloc status for BRIDGE_HIP_CHECK.
    hipError_t allocate(std::size_t bytes) { return hipMalloc(&ptr, bytes); }

    operator T*() const { return ptr; }
};

// Derive the GPU architecture from the running device (never assume one at
// compile time) and reject unsupported archs. gfx942 and gfx950 are always
// accepted; gfx90a is accepted only when the op supports it (aquant, bquant).
inline bool validate_supported_arch(const char* fn, bool allow_gfx90a = false)
{
    int dev = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
    {
        std::cerr << fn << ": could not query device architecture\n";
        return false;
    }
    const std::string arch(props.gcnArchName);
    const bool ok = arch.rfind("gfx950", 0) == 0 || arch.rfind("gfx942", 0) == 0 ||
                    (allow_gfx90a && arch.rfind("gfx90a", 0) == 0);
    if(!ok)
    {
        std::cerr << fn << ": unsupported GPU architecture '" << arch
                  << "' (supported: " << (allow_gfx90a ? "gfx90a, " : "") << "gfx942, gfx950)\n";
        return false;
    }
    return true;
}

// Build the stream_config used for a launch. When timing is requested use the
// GPU timer with warmup (cold_niters=3, nrepeat=10); otherwise run once.
inline ck_tile::stream_config make_stream_config(bool do_time)
{
    return ck_tile::stream_config{
        nullptr,          // stream_id_
        do_time,          // time_kernel_
        0,                // log_level_
        do_time ? 3 : 0,  // cold_niters_
        do_time ? 10 : 1, // nrepeat_
        do_time,          // is_gpu_timer_
        false,            // flush_cache_
        1,                // rotating_count_
    };
}

// Direct-launch the force-included kernel. Returns the kernel execution time in
// ms, or a negative value if the kernel reports unsupported args (callers treat
// <0 as an error and return -2, matching the previous behavior).
template <typename KernelT>
inline float launch(const ck_tile::QuantGemmHostArgs& args, bool do_time)
{
    return KernelT::launch(args, make_stream_config(do_time));
}

// The three guard helpers below replace the init / null-pointer / dimension
// checks that every bridge's run() used to inline verbatim. Each prints the same
// diagnostic as before and returns false so the caller can `return -1`. The
// per-op argument lists differ (4 vs 5 pointers; MNK plus op-specific QK/QN
// counts), so the pointer/dimension checks take an initializer_list.
inline bool check_initialized(const char* fn, bool initialized)
{
    if(!initialized)
    {
        std::cerr << fn << ": not initialized\n";
        return false;
    }
    return true;
}

inline bool check_non_null(const char* fn, std::initializer_list<const void*> ptrs)
{
    for(const void* p : ptrs)
    {
        if(!p)
        {
            std::cerr << fn << ": null pointer argument\n";
            return false;
        }
    }
    return true;
}

inline bool check_positive_dims(const char* fn, std::initializer_list<int64_t> dims)
{
    for(int64_t d : dims)
    {
        if(d <= 0)
        {
            std::cerr << fn << ": invalid dimensions\n";
            return false;
        }
    }
    return true;
}

// The identical tail every bridge's run() ended with: direct-launch the
// force-included kernel (return -2 if it rejects the args), copy C back to the
// host, publish the optional timing, return 0. C_dev is any type convertible to
// const CT* (e.g. DeviceBuffer<CT>).
template <typename KernelT, typename CT>
inline int launch_and_copyback(const char* fn,
                               const ck_tile::QuantGemmHostArgs& args,
                               void* C_host,
                               const CT* C_dev,
                               std::size_t mn_elems,
                               float* time_ms)
{
    const float exec_time = launch<KernelT>(args, time_ms != nullptr);
    if(exec_time < 0.0f)
    {
        std::cerr << fn << ": kernel reported unsupported args\n";
        return -2;
    }

    const hipError_t err =
        hipMemcpy(C_host, C_dev, elements_to_bytes<CT>(mn_elems), hipMemcpyDeviceToHost);
    if(err != hipSuccess)
    {
        std::cerr << fn << ": HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":"
                  << __LINE__ << "\n";
        return -1;
    }

    if(time_ms)
        *time_ms = exec_time;
    return 0;
}

} // namespace quant_bridge

// On a HIP error, print the failing op name + file/line and return -1. RAII
// DeviceBuffers free themselves on the return, so no cleanup call is needed.
#define BRIDGE_HIP_CHECK(fn, call)                                                                \
    do                                                                                            \
    {                                                                                             \
        hipError_t _err = (call);                                                                 \
        if(_err != hipSuccess)                                                                    \
        {                                                                                         \
            std::cerr << (fn) << ": HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ \
                      << ":" << __LINE__ << "\n";                                                 \
            return -1;                                                                            \
        }                                                                                         \
    } while(0)

// Emit the C API boilerplate shared by every bridge. Invoke once at the top of
// each op's `extern "C"` block (it declares the file-local g_initialized flag
// that the op's run() guard checks). KERNEL_NAME is force-included.
#define QUANT_BRIDGE_C_API()                                         \
    static bool g_initialized = false;                               \
    int dispatcher_initialize()                                      \
    {                                                                \
        g_initialized = true;                                        \
        return 0;                                                    \
    }                                                                \
    const char* dispatcher_get_kernel_name() { return KERNEL_NAME; } \
    int dispatcher_init() { return dispatcher_initialize(); }        \
    int dispatcher_get_kernel_count() { return 1; }                  \
    void dispatcher_cleanup() { g_initialized = false; }

#endif // CK_TILE_DISPATCHER_QUANT_BRIDGE_COMMON_HPP
