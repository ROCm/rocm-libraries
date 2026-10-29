// -*- C++ -*-

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

#ifndef __LIBHIPTHREADS___MEMORY_MALLOC_H__
#define __LIBHIPTHREADS___MEMORY_MALLOC_H__

/**
 * @file
 * @brief C-style GPU allocation helpers (blocking malloc + async free).
 * @ingroup c_memory
 */

#include "hip/hip_runtime_api.h"
#include <cstddef>

#include "hip/__support/hip_check.h"

namespace cuda {

namespace internal {
// NOTE: NOT STATIC so that there is only one copy of the static local variable inside!
/**
 * @brief Internal stream used to enqueue hipMallocAsync / hipFreeAsync.
 * @warning Implementation detail; not part of the stable public API.
 */
inline hipStream_t &getEnqueingStream() {
    // TODO: investigate using hipExtStreamCreateWithCUMask for this
    static hipStream_t enqueingStream = []() -> hipStream_t {
        hipStream_t s;
        __LIBHIPTHREADS_HIP_CHECK__(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        return s;
    }();
    return enqueingStream;
}
}

/**
 * @brief Allocates size bytes of uninitialized GPU-accessible memory.
 *
 * Uses `hipMallocAsync` on an internal non-blocking stream, then synchronizes
 * that stream so the returned pointer is ready for immediate use.
 *
 * Differences vs C malloc:
 * - Throws on HIP failure instead of returning `nullptr`.
 * - Guarantees stream sync before return (blocking semantics).
 *
 * @note Unlike `hipMalloc`, this function is safe to call while `hip::wthread`
 *       objects are alive. It uses `hipMallocAsync` on a non-blocking stream,
 *       avoiding the deadlock that would occur with synchronous HIP APIs when
 *       the hipThreads persistent scheduler holds the GPU context.
 *
 * @param size Number of bytes to allocate.
 * @return Pointer to at least size bytes.
 * @throws HIP error via `__LIBHIPTHREADS_HIP_CHECK__` on failure.
 * @ingroup c_memory
 */
inline __host__ void *malloc(::std::size_t size) {
    void *ptr;
    __LIBHIPTHREADS_HIP_CHECK__(hipMallocAsync(&ptr, size, internal::getEnqueingStream()));
    __LIBHIPTHREADS_HIP_CHECK__(hipStreamSynchronize(internal::getEnqueingStream()));
    return ptr;
}

/**
 * @brief Asynchronously frees memory obtained from `hip::malloc`.
 *
 * Enqueues `hipFreeAsync` on the internal stream and returns immediately.
 * Passing nullptr is a no-op (matches free semantics).
 *
 * @param ptr Pointer previously returned by `hip::malloc` or `nullptr`.
 * @ingroup c_memory
 */
inline __host__ void free(void* ptr) {
    __LIBHIPTHREADS_HIP_CHECK__(hipFreeAsync(ptr, internal::getEnqueingStream()));
}

} // namespace cuda


#endif // __LIBHIPTHREADS___MEMORY_MALLOC_H__
