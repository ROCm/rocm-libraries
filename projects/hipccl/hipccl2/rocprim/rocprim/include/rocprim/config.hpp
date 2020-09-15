// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef ROCPRIM_DEVICE
    #define ROCPRIM_DEVICE __device__
    #define ROCPRIM_HOST __host__
    #define ROCPRIM_HOST_DEVICE __host__ __device__
    #define ROCPRIM_SHARED_MEMORY __shared__
    // TODO: These paremeters should be tuned for NAVI in the close future.
    #ifndef ROCPRIM_DEFAULT_MAX_BLOCK_SIZE
        #define ROCPRIM_DEFAULT_MAX_BLOCK_SIZE 256
    #endif
    #ifndef ROCPRIM_DEFAULT_MIN_WARPS_PER_EU
        #define ROCPRIM_DEFAULT_MIN_WARPS_PER_EU 1
    #endif
#endif

#if ( defined(__gfx801__) || \
      defined(__gfx802__) || \
      defined(__gfx803__) || \
      defined(__gfx810__) || \
      defined(__gfx900__) || \
      defined(__gfx902__) || \
      defined(__gfx904__) || \
      defined(__gfx906__) || \
      defined(__gfx908__) || \
      defined(__gfx909__) ) && \
      !defined(ROCPRIM_DISABLE_DPP)
    #define ROCPRIM_DETAIL_USE_DPP true
#else
    #define ROCPRIM_DETAIL_USE_DPP false
#endif

#ifdef ROCPRIM_DISABLE_LOOKBACK_SCAN
    #define ROCPRIM_DETAIL_USE_LOOKBACK_SCAN false
#else
    #define ROCPRIM_DETAIL_USE_LOOKBACK_SCAN true
#endif

// Defines targeted AMD architecture. Supported values:
// * 803 (gfx803)
// * 900 (gfx900)
#ifndef ROCPRIM_TARGET_ARCH
    #define ROCPRIM_TARGET_ARCH 0
#endif

#endif // ROCPRIM_CONFIG_HPP_
