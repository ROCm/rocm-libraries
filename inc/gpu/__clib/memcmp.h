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

#ifndef __GPU___CLIB_MEMCMP_H__
#define __GPU___CLIB_MEMCMP_H__

#include "hip/hip_runtime_api.h"
#include <cstddef>

namespace gpu {

inline __host__ __device__ int memcmp(const void *lhs, const void *rhs, std::size_t count) {
    for (const unsigned char *l = reinterpret_cast<const unsigned char *>(lhs),
                             *r = reinterpret_cast<const unsigned char *>(rhs);
         count > 0; ++l, ++r, --count) {
        if (*l != *r) {
            return (*l - *r);
        }
    }
    return 0;
}

}

#endif // __GPU___CLIB_MEMCMP_H__
