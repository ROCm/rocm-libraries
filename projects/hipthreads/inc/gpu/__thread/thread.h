// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __GPU___THREAD_THREAD_H__
#define __GPU___THREAD_THREAD_H__

#include <utility>
#include <type_traits>
#include <system_error>
#include <iostream>
#include <cstdint>
#include <memory>
#include <atomic>

// TODO: Define a custom assert macro for the GPU that cleans up the thread state before invoking __assert_fail(#expr,
// __FILE__, __LINE__, __ASSERT_FUNCTION). Maybe do the cleanup work in a function that wraps around __assert_fail?
#include <cassert>

#include "hip/hip_runtime.h"

#include <hip/std/__utility/swap.h>

#include "gpu/__support/hip_check.h"
#include "gpu/__clib/memcpy.h"
#include "gpu/__thread/id.h"
#include "gpu/__thread/worknode.h"

namespace gpu {

namespace internal {

//====================================================================================================================//
//      THREAD CLASS DEFINITION
//====================================================================================================================//

// TODO: split this file up into a WorkNode header and a thread header
class thread {
  public:
    // TODO: temporary measure. Should to be replaced with an actual class.
    // Right now a default constructed id is a valid thread id, and it shouldn't be.
    using id = __thread_id;

    // TODO: The default member initializer for worknode_d makes it impossible to have an instance of gpu::thread in
    // __shared__ or __device__ memory (pointers to gpu::thread are still allowed). This is not ideal.
    __host__ thread() noexcept;
    __device__ thread() noexcept {}
    __host__ __device__ thread(const thread &) = delete;
    __host__ __device__ thread(thread &&other) noexcept
#ifdef __HIP_DEVICE_COMPILE__
        : worknode_d(other.worknode_d), cached_tdata(std::move(other.cached_tdata)) {
        other.worknode_d = nullptr;
    }
#else
        : worknode_d(std::move(other.worknode_d)), cached_tdata(std::move(other.cached_tdata)) {}
#endif
    __host__ __device__ thread &operator=(const thread&) = delete;
    __host__ __device__ thread &operator=(thread &&other) noexcept;

    template <class Fn_t, class... Args_t>
    explicit __device__ thread(uint32_t width, Fn_t &&typed_fn, Args_t &&...args);
    template <class Fn_t, class... Args_t>
    explicit __host__ thread(uint32_t width, Fn_t &&typed_fn, Args_t &&...args);

    // TODO: replace the enable_if_t condition with one that checks if Fn_t is callable
    template <class Fn_t, class... Args_t,
              std::enable_if_t<!std::is_arithmetic_v<std::remove_reference_t<Fn_t>>,
                               bool> = true>
    explicit __device__ thread(Fn_t &&typed_fn, Args_t &&...args)
        : thread(1, std::forward<Fn_t>(typed_fn), std::forward<Args_t>(args)...) {}
    template <class Fn_t, class... Args_t,
              std::enable_if_t<!std::is_arithmetic_v<std::remove_reference_t<Fn_t>>,
                               bool> = true>
    explicit __host__ thread(Fn_t &&typed_fn, Args_t &&...args)
        : thread(1, std::forward<Fn_t>(typed_fn), std::forward<Args_t>(args)...) {}

    __host__ __device__ ~thread();

    __host__ __device__ void swap(thread& __t) noexcept { hip::std::swap(worknode_d, __t.worknode_d); hip::std::swap(cached_tdata, __t.cached_tdata); }

    __host__ __device__ thread::id get_id(uint32_t index = 0) const;
    __host__ __device__ bool joinable() const noexcept { return worknode_d != nullptr; }
    __host__ __device__ void join();
    __host__ __device__ void detach();

    __host__ __device__ static constexpr unsigned int max_width() noexcept { return 32; }
    __device__ static unsigned int hardware_concurrency() noexcept;
    __host__ static unsigned int hardware_concurrency() noexcept;

  private:
#ifdef __HIP_DEVICE_COMPILE__
    // If we don't initialize worknode_d to nullptr, operator= might fail when assigning to a default constructed gpu::thread.
    // TODO: Make WorkNodeDeleter work for both host and device and replace this with
    // hip::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> so we don't have to specialize between host and device
    WorkNode_Header *worknode_d = nullptr;
#else
    std::unique_ptr<WorkNode_Header, WorkNodeDeleter> worknode_d = nullptr;
#endif
    ThreadData cached_tdata;
};

} // namespace internal

//====================================================================================================================//
//      USER FACING API
//====================================================================================================================//

using internal::thread;

template <class Fn_t, class... Args_t>
inline __host__ thread::thread(uint32_t width, Fn_t &&typed_fn, Args_t &&...args) {
    if (width > max_width()) {
        throw std::length_error("thread::thread: width must not exceed " + std::to_string(max_width()));
    }

    auto worknode_h = WorkNode_Header::make_worknode(width, std::forward<Fn_t>(typed_fn), std::forward<Args_t>(args)...);
    cached_tdata = worknode_h->tdata;
    using WorkNode_t = typename decltype(worknode_h)::element_type;
    // First two are prerequisites for the third, and produce more user-friendly error messages
    // Note: is_trivially_copyable can behave strangely for extended lambdas. See
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 14.7.2.18 Extended Lambda Restrictions
    //
    // TODO: We really can't accept raw fn pointers right now because references to device functions from host code is
    // forbidden. However, if a __host__ __device__ function tries to construct a gpu::thread object using a function
    // object passed in from a __device__ function, the compiler seems to try to instantiate this __host__ template and
    // fail on this static_assert if we don't allow function types.
    static_assert(std::is_trivially_copyable_v<std::remove_reference_t<Fn_t>> || std::is_function_v<std::remove_reference_t<Fn_t>>);
    static_assert(((std::is_trivially_copyable_v<std::remove_reference_t<Args_t>> || std::is_function_v<std::remove_reference_t<Args_t>>) && ...));
    // We're about to memcpy the WorkNode from host to device memory. Make sure that's ok.
    static_assert(std::is_trivially_copyable_v<WorkNode_t>);
    // Check that it's safe-ish to do the memcpy using a WorkNode_Header* instead of a WorkNode_t*
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Winvalid-offsetof"     // To suppress warning.
    static_assert(offsetof(WorkNode_t, wrapper_fn) == offsetof(WorkNode_Header, wrapper_fn));
    #pragma GCC diagnostic pop

#ifndef __HIP_DEVICE_COMPILE__
    worknode_d = worknode_h->sendToGPU();
#endif
}

template <class Fn_t, class... Args_t>
inline __device__ thread::thread(uint32_t width [[maybe_unused]], Fn_t &&typed_fn [[maybe_unused]], Args_t &&...args [[maybe_unused]]) {
#ifdef __HIP_DEVICE_COMPILE__
    assert(width <= max_width());
    assert(threadIdx.x == 0);
    auto typed_worknode_ptr = WorkNode_Header::make_worknode(width, std::forward<Fn_t>(typed_fn), std::forward<Args_t>(args)...);
    cached_tdata = typed_worknode_ptr->tdata;

    // First two are prerequisites for the third, and produce more user-friendly error messages
    static_assert(std::is_trivially_destructible_v<Fn_t>);
    static_assert((std::is_trivially_destructible_v<Args_t> && ...));
    // gpu::thread loses the information about what type WorkNode<Callable_t> is, so can't call the destructor
    static_assert(std::is_trivially_destructible_v<decltype(*typed_worknode_ptr)>);

    worknode_d = typed_worknode_ptr;

    worknode_d->insertIntoMainQueue();
#endif // __HIP_DEVICE_COMPILE__
}

} // namespace gpu

namespace cuda::std {
    __host__ __device__ inline _LIBGPU_HIDE_FROM_ABI void swap(gpu::thread& __x, gpu::thread& __y) _NOEXCEPT { __x.swap(__y); }
}

#endif // __GPU___THREAD_THREAD_H__
