// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#ifndef __LIBHIPTHREADS___THREAD_JTHREAD_H__
#define __LIBHIPTHREADS___THREAD_JTHREAD_H__

/**
 * @file
 * @brief Auto-joining GPU thread handle (subset of std::jthread, no stop tokens).
 * @ingroup thread
 *
 * Provides a `cuda::jthread` API analogous (not identical) to `std::jthread`,
 * built on top of `cuda::thread`. Adds automatic `join()` on destruction and
 * on move-assignment to a joinable target.
 *
 * Key differences vs `std::jthread`:
 * - No stop tokens / stop_source in this initial version; the callable is
 *   expected to terminate on its own.
 * - Same device transfer constraints as `cuda::thread` apply to the callable
 *   and forwarded arguments.
 * - No `native_handle_type` exposure yet (could be added later).
 */

#include <type_traits>
#include <utility>

#include <hip/std/__utility/swap.h>

#include "hip/__thread/thread.h"

namespace cuda {

namespace internal {

//====================================================================================================================//
//      JTHREAD CLASS DEFINITION
//====================================================================================================================//

/**
 * @class jthread
 * @brief Auto-joining handle wrapping `cuda::thread`.
 * @ingroup thread
 *
 * Same scheduling and ownership model as `cuda::thread`; the only behavioural
 * differences are that the destructor and move-assignment auto-`join()` if
 * the target is still joinable. Stop tokens are not implemented.
 */
class jthread {
  public:
    /// Alias for jthread identifier type.
    using id = thread::id;

    /// Default constructs a non-joinable jthread (no associated work node).
    __host__ __device__ jthread() noexcept = default;

    /// Constructs a joinable jthread with explicit width (1..max_width()).
    /// Forwards to `cuda::thread`'s width-taking constructor.
    template <class Fn_t, class... Args_t>
    __host__ explicit jthread(uint32_t width, Fn_t&& typed_fn, Args_t&&... args)
        : __thread_(width, ::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...) {}
    template <class Fn_t, class... Args_t>
    __device__ explicit jthread(uint32_t width, Fn_t&& typed_fn, Args_t&&... args)
        : __thread_(width, ::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...) {}

    /// Convenience constructor (width = 1) — std::thread drop-in.
    ///
    /// Two `enable_if` constraints:
    ///   1. `!is_arithmetic` — disambiguates from the width-taking ctor when
    ///       the first argument is an integer-like type.
    ///   2. `!is_same_v<..., jthread>` — prevents this template from hijacking
    ///       the move/copy constructors when the first argument is itself a jthread.
    template <class Fn_t, class... Args_t,
              ::std::enable_if_t<!::std::is_arithmetic_v<::std::remove_reference_t<Fn_t>>
                              && !::std::is_same_v<::std::remove_cv_t<::std::remove_reference_t<Fn_t>>, jthread>, bool> = true>
    __host__ explicit jthread(Fn_t&& typed_fn, Args_t&&... args)
        : __thread_(::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...) {}
    template <class Fn_t, class... Args_t,
              ::std::enable_if_t<!::std::is_arithmetic_v<::std::remove_reference_t<Fn_t>>
                              && !::std::is_same_v<::std::remove_cv_t<::std::remove_reference_t<Fn_t>>, jthread>, bool> = true>
    __device__ explicit jthread(Fn_t&& typed_fn, Args_t&&... args)
        : __thread_(::std::forward<Fn_t>(typed_fn), ::std::forward<Args_t>(args)...) {}

    /// Destructor: auto-joins if joinable.
    __host__ __device__ ~jthread() {
        if (joinable()) join();
    }

    /// \name Deleted copy operations
    ///@{
    jthread(const jthread&)            = delete;
    jthread& operator=(const jthread&) = delete;
    ///@}

    /// Move construction transfers ownership; source becomes not joinable.
    __host__ __device__ jthread(jthread&&) noexcept = default;

    /// Move assignment: auto-joins LHS if it was joinable, then takes over RHS.
    __host__ __device__ jthread& operator=(jthread&& other) noexcept {
        if (this != &other) {
            if (joinable()) join();
            __thread_ = ::std::move(other.__thread_);
        }
        return *this;
    }

    /// Swaps underlying thread ownership with another jthread.
    __host__ __device__ void swap(jthread& other) noexcept { hip::std::swap(__thread_, other.__thread_); }

    /// Returns true if an execution context is owned and not yet joined/detached.
    [[nodiscard]] __host__ __device__ bool joinable() const noexcept { return __thread_.joinable(); }

    /**
     * @brief Returns the id of the (possibly width-partitioned) logical lane.
     * @param index Lane index (default 0).
     */
    [[nodiscard]] __host__ __device__ jthread::id get_id(uint32_t index = 0) const { return __thread_.get_id(index); }

    /**
     * @brief Waits for completion of the associated work.
     * Undefined behavior if !joinable().
     */
    __host__ __device__ void join() { __thread_.join(); }

    /**
     * @brief Releases ownership allowing work to proceed independently.
     * After detach() the handle becomes not joinable.
     */
    __host__ __device__ void detach() { __thread_.detach(); }

    /// Maximum supported width (forwards to cuda::thread::max_width()).
    __host__ __device__ static constexpr unsigned int max_width() noexcept { return thread::max_width(); }

    /// Number of concurrent hardware slots usable (forwards to cuda::thread).
    [[nodiscard]] __host__ __device__ static unsigned int hardware_concurrency() noexcept { return thread::hardware_concurrency(); }

  private:
    thread __thread_;
};

} // namespace internal

//====================================================================================================================//
//      USER FACING API
//====================================================================================================================//

using internal::jthread;

} // namespace cuda

namespace cuda::std {
    __host__ __device__ inline void swap(hip::jthread& __x, hip::jthread& __y) noexcept { __x.swap(__y); }
}

#endif // __LIBHIPTHREADS___THREAD_JTHREAD_H__
