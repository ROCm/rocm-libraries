// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/// \file
/// \brief This file implements a host/device kernel abstraction.
///
/// rocRAND supports running generators on both the device and the host.
/// Since the generators should generate the same results on both of these,
/// generation kernels are "launched" with the same number of blocks, threads, etc,
/// on the host as they are on the GPU. This file contains abstraction to help with
/// making code generic for both of these "systems".

#ifndef ROCRAND_RNG_SYSTEM_H_
#define ROCRAND_RNG_SYSTEM_H_

#include "common.hpp"
#include "config_types.hpp"
#include "rocrand/rocrand.h"
#include "utils/threedim_iterator.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#if defined(ROCRAND_PARALLEL_STL) && __has_include(<execution>)
    #define ROCRAND_USE_PARALLEL_STL
    #include <execution>
#endif
#include <cstring>
#include <memory>
#include <new>

#include <stdint.h>

namespace rocrand_impl::system
{
namespace detail
{

inline rocrand_status is_stream_blocking(hipStream_t stream, bool& is_blocking)
{
    if(stream)
    {
        unsigned int     stream_flags;
        const hipError_t error = hipStreamGetFlags(stream, &stream_flags);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        is_blocking = (stream_flags & hipStreamNonBlocking) == 0;
    }
    else
    {
        is_blocking = true;
    }
    return ROCRAND_STATUS_SUCCESS;
}

/// Enqueues `Fn(*arg)` in `stream` and takes ownership of `arg`.
///
/// A host node captured into a graph passes the same argument to every launch of the graph, so
/// the callback cannot free it. When `stream` is capturing, the argument is handed to the graph
/// instead, which frees it when the graph and every executable instantiated from it are destroyed.
/// Otherwise, the argument is freed after `Fn` returns.
template<typename T, void (*Fn)(T&)>
hipError_t launch_host_func(hipStream_t stream, std::unique_ptr<T> arg)
{
    hipStreamCaptureStatus capture_status;
    hipGraph_t             capture_graph;
    hipError_t error = hipStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &capture_graph);
    if(error != hipSuccess)
    {
        return error;
    }

    if(capture_status == hipStreamCaptureStatusActive)
    {
        hipUserObject_t user_object;
        error = hipUserObjectCreate(
            &user_object,
            arg.get(),
            [](void* user_data) { delete static_cast<T*>(user_data); },
            1,
            hipUserObjectNoDestructorSync);
        if(error != hipSuccess)
        {
            return error;
        }
        T* const user_data = arg.release();
        error = hipGraphRetainUserObject(capture_graph, user_object, 1, hipGraphUserObjectMove);
        if(error != hipSuccess)
        {
            (void)hipUserObjectRelease(user_object, 1);
            return error;
        }
        return hipLaunchHostFunc(
            stream,
            [](void* user_data) { Fn(*static_cast<T*>(user_data)); },
            user_data);
    }

    // If this fails and the callback has not been invoked, the argument is leaked. It is unclear
    // whether hipLaunchHostFunc can return an error after the callback has already been invoked,
    // but in such case there would be a double free (crash) instead of a memory leak, so we will
    // just leak it.
    return hipLaunchHostFunc(
        stream,
        [](void* user_data)
        {
            std::unique_ptr<T> arg(static_cast<T*>(user_data));
            Fn(*arg);
        },
        arg.release());
}

} // namespace detail

/// \tparam UseHostFunc If true, launching will enqueue the kernel in the stream. Otherwise,
///   execute the kernel synchronously.
template<bool UseHostFunc>
struct host_system
{
    static constexpr bool is_device()
    {
        return false;
    }

    template<typename T>
    static rocrand_status alloc(T** ptr, size_t n)
    {
        hipError_t status = hipDeviceSynchronize();
        if(status != hipSuccess)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        *ptr = new(std::nothrow) T[n];
        if(!*ptr)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename T>
    static void free(T* ptr)
    {
        ROCRAND_HIP_FATAL_ASSERT(hipDeviceSynchronize());
        delete[] ptr;
    }

    static rocrand_status memcpy(void* dst, const void* src, size_t size, hipMemcpyKind /*kind*/)
    {
        hipError_t status = hipDeviceSynchronize();
        if(status != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(dst, src, size);
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename... UserArgs>
    struct KernelArgs
    {
        dim3                    num_blocks;
        dim3                    num_threads;
        std::tuple<UserArgs...> user_args;
    };

    // At the call site, the loops use a variable number of iterations,
    // so Clang isn't able to unroll them and will (wrongfully) complain about it.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
    template<host::target_arch Arch, typename Kernel, size_t... Is, typename... Args>
    static void invoke_kernel(dim3 block,
                              dim3 thread,
                              dim3 grid_dim,
                              dim3 block_dim,
                              std::index_sequence<Is...>,
                              const std::tuple<Args...>& args)
    {
        Kernel::template generate<Arch>(block, thread, grid_dim, block_dim, std::get<Is>(args)...);
    }
#pragma clang diagnostic pop

    template<typename Kernel, typename... Args>
    static void run_kernel(KernelArgs<Args...>& kernel_args)
    {
        const auto num_blocks    = kernel_args.num_blocks;
        const auto num_threads   = kernel_args.num_threads;
        const auto execute_block = [&](const dim3 block_idx)
        {
            for(uint32_t tz = 0; tz < num_threads.z; ++tz)
            {
                for(uint32_t ty = 0; ty < num_threads.y; ++ty)
                {
                    for(uint32_t tx = 0; tx < num_threads.x; ++tx)
                    {
                        invoke_kernel<host::target_arch{}, Kernel>(
                            block_idx,
                            dim3(tx, ty, tz),
                            num_blocks,
                            num_threads,
                            std::make_index_sequence<sizeof...(Args)>(),
                            kernel_args.user_args);
                    }
                }
            }
        };

        std::for_each(
#ifdef ROCRAND_USE_PARALLEL_STL
            std::execution::par_unseq,
#endif
            cpp_utils::threedim_iterator::begin(num_blocks),
            cpp_utils::threedim_iterator::end(num_blocks),
            execute_block);
    }

    template<typename Kernel,
             typename ConfigProvider
             = host::static_block_size_config_provider<ROCRAND_DEFAULT_MAX_BLOCK_SIZE>,
             typename T     = unsigned int,
             bool IsDynamic = false,
             typename... Args>
    static rocrand_status launch([[maybe_unused]] host::target_arch arch,
                                 dim3                               num_blocks,
                                 dim3                               num_threads,
                                 unsigned int                       shared_bytes,
                                 [[maybe_unused]] hipStream_t       stream,
                                 Args... args)
    {
        (void)IsDynamic; // Not relevant on host launches
        (void)shared_bytes; // shared memory not supported on host

        KernelArgs<Args...> kernel_args{num_blocks, num_threads, std::tuple<Args...>(args...)};

        if constexpr(UseHostFunc)
        {
            const hipError_t error
                = detail::launch_host_func<KernelArgs<Args...>, run_kernel<Kernel, Args...>>(
                    stream,
                    std::make_unique<KernelArgs<Args...>>(std::move(kernel_args)));
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_LAUNCH_FAILURE;
            }
        }
        else
        {
            run_kernel<Kernel, Args...>(kernel_args);
        }

        return ROCRAND_STATUS_SUCCESS;
    }

    /// Calls `Fn(*arg)`, enqueued in `stream` if `UseHostFunc` is true.
    template<typename T, void (*Fn)(T&)>
    static rocrand_status launch_host_func([[maybe_unused]] hipStream_t stream,
                                           std::unique_ptr<T>           arg)
    {
        if constexpr(UseHostFunc)
        {
            const hipError_t error = detail::launch_host_func<T, Fn>(stream, std::move(arg));
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            try
            {
                Fn(*arg);
            }
            catch(...)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    static rocrand_status is_host_func_blocking([[maybe_unused]] hipStream_t stream,
                                                bool&                        is_blocking)
    {
        if constexpr(UseHostFunc)
        {
            return detail::is_stream_blocking(stream, is_blocking);
        }
        else
        {
            is_blocking = true;
        }
        return ROCRAND_STATUS_SUCCESS;
    }
};

namespace detail
{

template<typename Kernel,
         typename ConfigProvider,
         typename T,
         bool              IsDynamic,
         host::target_arch Arch,
         typename... Args>
ROCRAND_INTERNAL_KERNEL __launch_bounds__((
    host::get_block_size<ConfigProvider, T, Arch>(IsDynamic)))
void trampoline_kernel(Args... args)
{
#if !defined(__SPIRV__)
    if constexpr(Arch == host::get_device_arch())
#endif
    {
        Kernel::template generate<Arch>(blockIdx, threadIdx, gridDim, blockDim, args...);
    }
}

} // namespace detail

struct device_system
{
    static constexpr bool is_device()
    {
        return true;
    }

    template<typename T>
    static rocrand_status alloc(T** ptr, size_t n)
    {
        hipError_t error = hipMalloc(ptr, sizeof(T) * n);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename T>
    static void free(T* ptr)
    {
        ROCRAND_HIP_FATAL_ASSERT(hipFree(ptr));
    }

    static rocrand_status memcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind)
    {
        hipError_t error = hipMemcpy(dst, src, size, kind);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename Kernel,
             typename ConfigProvider
             = host::static_block_size_config_provider<ROCRAND_DEFAULT_MAX_BLOCK_SIZE>,
             typename T     = unsigned int,
             bool IsDynamic = false,
             typename... Args>
    static rocrand_status launch(host::target_arch arch,
                                 dim3              num_blocks,
                                 dim3              num_threads,
                                 unsigned int      shared_bytes,
                                 hipStream_t       stream,
                                 Args... args)
    {
        bool launched = false;
        host::for_each_arch(
            [&](auto arch_tag)
            {
                if(arch_tag != arch)
                {
                    return;
                }

                detail::trampoline_kernel<Kernel, ConfigProvider, T, IsDynamic, arch_tag>
                    <<<num_blocks, num_threads, shared_bytes, stream>>>(args...);

                launched = true;
            });
        if(!launched)
        {
            detail::
                trampoline_kernel<Kernel, ConfigProvider, T, IsDynamic, host::target_arch::unknown>
                <<<num_blocks, num_threads, shared_bytes, stream>>>(args...);
        }
        if(hipGetLastError() != hipSuccess)
        {
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    /// Enqueues `Fn(*arg)` in `stream`.
    template<typename T, void (*Fn)(T&)>
    static rocrand_status launch_host_func(hipStream_t stream, std::unique_ptr<T> arg)
    {
        const hipError_t error = detail::launch_host_func<T, Fn>(stream, std::move(arg));
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    static rocrand_status is_host_func_blocking(hipStream_t stream, bool& is_blocking)
    {
        return detail::is_stream_blocking(stream, is_blocking);
    }
};

template<bool IsDevice>
struct syncthreads;

template<>
struct syncthreads<true>
{
    __device__
    void operator()()
    {
        __syncthreads();
    }
};

template<>
struct syncthreads<false>
{
    void operator()() {}
};

} // namespace rocrand_impl::system

#endif
