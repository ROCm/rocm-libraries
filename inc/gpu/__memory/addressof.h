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

#ifndef __GPU___ATOMIC___MEMORY_ADDRESSOF_H__
#define __GPU___ATOMIC___MEMORY_ADDRESSOF_H__

#include "hip/hip_runtime_api.h"
#include <type_traits>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::addressof
//====================================================================================================================//

template <class _Tp>
__host__ __device__ inline constexpr _Tp *addressof(_Tp &__x) noexcept {
    return __builtin_addressof(__x);
}

template <class _Tp>
__host__ __device__ _Tp *addressof(const _Tp &&) noexcept = delete;

} // namespace gpu

#endif // __GPU___ATOMIC___MEMORY_ADDRESSOF_H__
