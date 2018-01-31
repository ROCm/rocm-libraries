// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_CONFIG_HPP_
#define ROCPRIM_CONFIG_HPP_

#define BEGIN_ROCPRIM_NAMESPACE \
    namespace rocprim {

#define END_ROCPRIM_NAMESPACE \
    } /* rocprim */

#if defined(__HCC_HC__) && !defined(ROCPRIM_HIP_API)
    #ifndef ROCPRIM_HC_API
        #define ROCPRIM_HC_API
    #endif
#elif defined(__HIPCC__) && !defined(ROCPRIM_HC_API)
    #ifndef ROCPRIM_HIP_API
        #define ROCPRIM_HIP_API
    #endif
#endif

#if defined(ROCPRIM_HC_API)
    #include <hcc/hc.hpp>
    #include <hcc/hc_short_vector.hpp>

    #ifndef ROCPRIM_DEVICE
        #define ROCPRIM_DEVICE __attribute__((hc))
        #define ROCPRIM_HOST __attribute__((cpu))
        #define ROCPRIM_HOST_DEVICE __attribute__((hc, cpu))
        #define ROCPRIM_SHARED_MEMORY tile_static
    #endif
#elif defined(ROCPRIM_HIP_API)
    #include <hip/hip_runtime.h>

    #ifndef ROCPRIM_DEVICE
        #define ROCPRIM_DEVICE __device__
        #define ROCPRIM_HOST __host__
        #define ROCPRIM_HOST_DEVICE __host__ __device__
        #define ROCPRIM_SHARED_MEMORY __shared__
    #endif
#else
    #error "HIP and HC APIs are not available"
#endif

#endif // ROCPRIM_CONFIG_HPP_
