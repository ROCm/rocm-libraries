// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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


#ifndef ROCPRIM_COMMON_HPP_
#define ROCPRIM_COMMON_HPP_

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

namespace detail
{
#ifndef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
    #define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
        do                                                                                           \
        {                                                                                            \
            auto _error = hipGetLastError();                                                         \
            if(_error != hipSuccess)                                                                 \
                return _error;                                                                       \
            if(debug_synchronous)                                                                    \
            {                                                                                        \
                std::cout << name << "(" << size << ")";                                             \
                auto __error = hipStreamSynchronize(stream);                                         \
                if(__error != hipSuccess)                                                            \
                    return __error;                                                                  \
                auto _end = std::chrono::steady_clock::now();                                        \
                auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
                std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
            }                                                                                        \
        }                                                                                            \
        while(0)
#endif // ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

#ifndef ROCPRIM_RETURN_ON_ERROR
    #define ROCPRIM_RETURN_ON_ERROR(...)      \
        do                                    \
        {                                     \
            hipError_t error = (__VA_ARGS__); \
            if(error != hipSuccess)           \
            {                                 \
                return error;                 \
            }                                 \
        }                                     \
        while(0)
#endif // ROCPRIM_RETURN_ON_ERROR

} // namespace detail

#endif // ROCPRIM_COMMON_HPP_
