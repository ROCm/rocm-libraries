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

#ifndef __LIBHIPTHREADS___CLIB_MEMCPY_H__
#define __LIBHIPTHREADS___CLIB_MEMCPY_H__

/**
 * @file
 * @brief Host/device byte copying primitive (memcpy analogue).
 * @ingroup c_bytestring
 */

#include "hip/hip_runtime_api.h"
#include <cstddef>

namespace cuda {

/**
 * @brief Copies a block of memory from a source address to a destination address.
 *
 * Copies `count` bytes from the object pointed to by `src` to the object pointed
 * to by `dest`. Both objects are reinterpreted as arrays of `unsigned char`.
 *
 * This version is available for use in both `__host__` and `__device__` code.
 *
 * @warning The behavior is undefined if the memory areas of `src` and `dest` overlap.
 *
 * @param dest Pointer to the destination array where the content is to be copied.
 * @param src Pointer to the source of data to be copied.
 * @param count Number of bytes to copy.
 * @return A copy of `dest`.
 * @ingroup c_bytestring
 */
inline __host__ __device__ void *memcpy(void *dest, const void *src, ::std::size_t count) {
    unsigned char *d = reinterpret_cast<unsigned char *>(dest);
    const unsigned char *s = reinterpret_cast<const unsigned char *>(src);
    for (; count > 0; ++d, ++s, --count) {
        *d = *s;
    }
    return dest;
}

// TODO: Should we provide a version that enables host-to-device, device-to-host, etc.?
}

#endif // __LIBHIPTHREADS___CLIB_MEMCPY_H__
