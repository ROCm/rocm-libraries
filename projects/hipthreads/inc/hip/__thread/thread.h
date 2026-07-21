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

#ifndef __LIBHIPTHREADS___THREAD_THREAD_H__
#define __LIBHIPTHREADS___THREAD_THREAD_H__

/**
 * @file
 * @brief GPU-adapted lightweight thread handle and launch wrapper.
 * @ingroup thread
 *
 * Provides a `cuda::wthread` API analogous (not identical) to `std::thread`,
 * adapted for HIP device execution queues. A wthread represents a unit of
 * GPU work submitted through an internal WorkNode. Construction captures a
 * callable and (optionally) a "width" (workgroup size) and schedules it for
 * execution. The handle can then be queried (`get_id`, `joinable`) and
 * synchronized (`join`, `detach`).
 *
 * Key differences vs `std::thread`:
 * - Callable and argument objects must satisfy device transfer constraints
 *   (trivially copyable / destructible as enforced by static_asserts).
 * - Width parameter (default 1) controls how many logical GPU lanes
 *   participate in the callable (upper bounded by `max_width()`).
 * - Host vs device constructors: host constructor copies a work node to
 *   device; device constructor inserts directly into a device queue.
 * - No native handle exposure yet (could be added later).
 * - Defining `cuda::wthread` variables with static storage duration is undefined behaviour.
 *
 * Thread life-cycle states:
 * - Default constructed: not joinable (no work node).
 * - Move constructed / assigned: ownership transferred, source becomes not joinable.
 * - After `detach()` or successful `join()`: not joinable.
 */

#include <utility>
#include <type_traits>
#include <system_error>
#include <iostream>
#include <cstdint>
#include <memory>
#include <atomic>

// TODO: Define a custom assert macro for the GPU that cleans up the wthread state before invoking __assert_fail(#expr,
// __FILE__, __LINE__, __ASSERT_FUNCTION). Maybe do the cleanup work in a function that wraps around __assert_fail?
#include <cassert>

#include "hip/hip_runtime.h"

#include <hip/std/__utility/swap.h>

#include "hip/__support/hip_check.h"
#include "hip/__clib/memcpy.h"
#include "hip/__thread/id.h"
#include "hip/__thread/worknode.h"

namespace cuda {

namespace internal {

//====================================================================================================================//
//      THREAD CLASS DEFINITION
//====================================================================================================================//

/**
 * @class wthread
 * @brief Handle owning (at most) one scheduled GPU work node (wave of execution).
 * @ingroup thread
 *
 * Constructing schedules a callable for execution (host or device path).
 * Non-copyable; movable to transfer ownership. Must be joined or detached
 * before destruction if joinable.
 *
 * Memory / transfer constraints:
 * - Captured callable and argument types must be trivially copyable (host path)
 *   or trivially destructible (device path) as enforced by static assertions.
 *
 * Concurrency model:
 * - `join()` waits for completion (synchronizing-with the end of the callable).
 * - `detach()` releases association; work may continue without the handle.
 *
 * Width:
 * - Optional first constructor parameter `width` ( <= `max_width()` ) describes
 *   how many lanes participate; implementation may map this to a warp subset.
 */
class wthread {
  public:
    /// Alias for thread identifier type (may be refined later).
    using id = __thread_id;

    // TODO: The default member initializer for worknode_d makes it impossible to have an instance of hip::wthread in
    // __shared__ or __device__ memory (pointers to hip::wthread are still allowed). This is not ideal.
    
    /// Default constructs a non-joinable wthread (no associated work node).
    __host__ wthread() noexcept;
    /// Device-side default constructor (no work node).
    __device__ wthread() noexcept {}

    /// \name Deleted copy / move operations
    /// These special members are intentionally disabled (handle is non-copyable / non-movable).
    ///@{
    __host__ __device__ wthread(const wthread &) = delete;
    ///@}

    /// Move construction transfers ownership; source becomes not joinable.
    __host__ __device__ wthread(wthread &&other) noexcept
#ifdef __HIP_DEVICE_COMPILE__
        : worknode_d(other.worknode_d), cached_tdata(::std::move(other.cached_tdata)) {
        other.worknode_d = nullptr;
    }
#else
        : worknode_d(::std::move(other.worknode_d)), cached_tdata(::std::move(other.cached_tdata)) {}
#endif
    ///@{
    __host__ __device__ wthread &operator=(const wthread&) = delete;
    ///@}

    /// Move assignment transfers ownership; source becomes not joinable.
    __host__ __device__ wthread &operator=(wthread &&other) noexcept;

    /**
     * @brief Construct with explicit width and callable (device path).
     * @param width Logical participation width (1..max_width()).
     * @param typed_fn Callable object.
     * @param args Argument pack forwarded to callable.
     */
    template <class Fn_t, class... Args_t>
    explicit __device__ wthread(uint32_t width, Fn_t &&typed_fn, Args_t &&...args);

    /**
     * @brief Construct with explicit width and callable (host path).
     * Schedules work node for device execution.
     */
    template <class Fn_t, class... Args_t>
    explicit __host__ wthread(uint32_t width, Fn_t &&typed_fn, Args_t &&...args);

    // TODO: replace the enable_if_t condition with one that checks if Fn_t is callable
    /**
     * @brief Device-side convenience constructor (width = 1) for initial drop-in replacement of `std::thread`.
     */
    template <class Fn_t, class... Args_t,
              ::std::enable_if_t<!::std::is_arithmetic_v<::std::remove_reference_t<Fn_t>>,
                               bool> = true>
    explicit __device__ wthread(Fn_t &&typed_fn, Args_t &&...args)
        : wthread(1, ::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...) {}

    /**
     * @brief Host-side convenience constructor (width = 1) for initial drop-in replacement of `std::thread`.
     */
    template <class Fn_t, class... Args_t,
              ::std::enable_if_t<!::std::is_arithmetic_v<::std::remove_reference_t<Fn_t>>,
                               bool> = true>
    explicit __host__ wthread(Fn_t &&typed_fn, Args_t &&...args)
        : wthread(1, ::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...) {}

    /// Destructor: wthread must be not joinable (joined or detached).
    __host__ __device__ ~wthread();

    /// Swaps underlying work node ownership with another wthread.
    __host__ __device__ void swap(wthread& __t) noexcept { hip::std::swap(worknode_d, __t.worknode_d); hip::std::swap(cached_tdata, __t.cached_tdata); }

    /**
     * @brief Returns the id of the (possibly width-partitioned) logical lane.
     * @param index Lane index (default 0).
     */
    __host__ __device__ wthread::id get_id(uint32_t index = 0) const;
    
    /// Returns true if an execution context is owned and not yet joined/detached.
    __host__ __device__ bool joinable() const noexcept { return worknode_d != nullptr; }

    /**
     * @brief Waits for completion of the associated work.
     * Undefined behavior if !joinable().
     */
    __host__ __device__ void join();

    /**
     * @brief Releases ownership allowing work to proceed independently.
     * After detach() the handle becomes not joinable.
     */
    __host__ __device__ void detach();

    /// Maximum supported width (implementation constant).
    __host__ __device__ static constexpr unsigned int max_width() noexcept { return 32; }

    /// Number of concurrent hardware slots usable (device variant).
    __device__ static unsigned int hardware_concurrency() noexcept;

    /// Number of concurrent hardware slots usable (host reflection / cached).
    __host__ static unsigned int hardware_concurrency() noexcept;

  private:
#ifdef __HIP_DEVICE_COMPILE__
    // If we don't initialize worknode_d to nullptr, operator= might fail when assigning to a default constructed hip::wthread.
    // TODO: Make WorkNodeDeleter work for both host and device and replace this with
    // hip::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> so we don't have to specialize between host and device
    WorkNode_Header *worknode_d = nullptr;
#else
    ::std::unique_ptr<WorkNode_Header, WorkNodeDeleter> worknode_d = nullptr;
#endif
    ThreadData cached_tdata;
};

/// Deprecated alias for hip::wthread
using thread = wthread;

} // namespace internal

//====================================================================================================================//
//      USER FACING API
//====================================================================================================================//

using internal::wthread;

template <class Fn_t, class... Args_t>
inline __host__ wthread::wthread(uint32_t width, Fn_t &&typed_fn, Args_t &&...args) {
    if (width > max_width()) {
        throw ::std::length_error("wthread::wthread: width must not exceed " + ::std::to_string(max_width()));
    }

    auto worknode_h = WorkNode_Header::make_worknode(width, ::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...);
    cached_tdata = worknode_h->tdata;
    using WorkNode_t = typename decltype(worknode_h)::element_type;
    // First two are prerequisites for the third, and produce more user-friendly error messages
    // Note: is_trivially_copyable can behave strangely for extended lambdas. See
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 14.7.2.18 Extended Lambda Restrictions
    //
    // TODO: We really can't accept raw fn pointers right now because references to device functions from host code is
    // forbidden. However, if a __host__ __device__ function tries to construct a hip::wthread object using a function
    // object passed in from a __device__ function, the compiler seems to try to instantiate this __host__ template and
    // fail on this static_assert if we don't allow function types.
    static_assert(::std::is_trivially_copyable_v<::std::remove_reference_t<Fn_t>> || ::std::is_function_v<::std::remove_reference_t<Fn_t>>);
    static_assert(((::std::is_trivially_copyable_v<::std::remove_reference_t<Args_t>> || ::std::is_function_v<::std::remove_reference_t<Args_t>>) && ...));
    // We're about to memcpy the WorkNode from host to device memory. Make sure that's ok.
    static_assert(::std::is_trivially_copyable_v<WorkNode_t>);
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
inline __device__ wthread::wthread(uint32_t width [[maybe_unused]], Fn_t &&typed_fn [[maybe_unused]], Args_t &&...args [[maybe_unused]]) {
#ifdef __HIP_DEVICE_COMPILE__
    assert(width <= max_width());
    assert(threadIdx.x == 0);
    auto typed_worknode_ptr = WorkNode_Header::make_worknode(width, ::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...);
    cached_tdata = typed_worknode_ptr->tdata;

    // First two are prerequisites for the third, and produce more user-friendly error messages
    static_assert(::std::is_trivially_destructible_v<Fn_t>);
    static_assert((::std::is_trivially_destructible_v<Args_t> && ...));
    // hip::wthread loses the information about what type WorkNode<Callable_t> is, so can't call the destructor
    static_assert(::std::is_trivially_destructible_v<decltype(*typed_worknode_ptr)>);

    worknode_d = typed_worknode_ptr;

    worknode_d->insertIntoMainQueue();
#endif // __HIP_DEVICE_COMPILE__
}

} // namespace cuda

namespace cuda::std {
    __host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI void swap(hip::wthread& __x, hip::wthread& __y) _NOEXCEPT { __x.swap(__y); }
}

#endif // __LIBHIPTHREADS___THREAD_THREAD_H__
