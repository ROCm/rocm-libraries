/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
    #ifndef __HIP_DEVICE_COMPILE__
        #define __THRUST_HAS_HIPRT__ 1
        #define THRUST_HIP_RUNTIME_FUNCTION __host__ __device__ __forceinline__
        #define THRUST_RUNTIME_FUNCTION THRUST_HIP_RUNTIME_FUNCTION
    #else
        #define __THRUST_HAS_HIPRT__ 0
        #define THRUST_HIP_RUNTIME_FUNCTION __host__ __forceinline__
        #define THRUST_RUNTIME_FUNCTION THRUST_HIP_RUNTIME_FUNCTION
    #endif
#else
    #define __THRUST_HAS_HIPRT__ 0
    #define THRUST_HIP_RUNTIME_FUNCTION __host__ __forceinline__
    #define THRUST_RUNTIME_FUNCTION THRUST_HIP_RUNTIME_FUNCTION
#endif

// TODO: These paremeters should be tuned for NAVI.
#ifndef THRUST_HIP_DEFAULT_MAX_BLOCK_SIZE
  #define THRUST_HIP_DEFAULT_MAX_BLOCK_SIZE 256
#endif
#ifndef THRUST_HIP_DEFAULT_MIN_WARPS_PER_EU
  #define THRUST_HIP_DEFAULT_MIN_WARPS_PER_EU 1
#endif
#define THRUST_HIP_LAUNCH_BOUNDS(BlockSize) __launch_bounds__(BlockSize, THRUST_HIP_DEFAULT_MIN_WARPS_PER_EU)
#define THRUST_HIP_LAUNCH_BOUNDS_DEFAULT THRUST_HIP_LAUNCH_BOUNDS(THRUST_HIP_DEFAULT_MAX_BLOCK_SIZE)

#ifdef __HIP_DEVICE_COMPILE__
#define THRUST_HIP_DEVICE_CODE
#endif

#define THRUST_HIP_DEVICE_FUNCTION __device__ __forceinline__
#define THRUST_HIP_HOST_FUNCTION __host__ __forceinline__
#define THRUST_HIP_FUNCTION __host__ __device__ __forceinline__

#ifdef THRUST_HIP_DEBUG_SYNC
#define THRUST_HIP_DEBUG_SYNC_FLAG true
#define DEBUG
#else
#define THRUST_HIP_DEBUG_SYNC_FLAG false
#endif

// Workaround, so kernel(s) called by function is/are not lost,
// Implicit instantiation of function template
// that will be used in #if __THRUST_HAS_HIPRT__ block.
#define THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(function) do \
    { \
        auto ptr = function; \
        (void) ptr; \
    } while (0)

#define THRUST_ROCPRIM_NS_PREFIX namespace thrust {   namespace hip_rocprim {
#define THRUST_ROCPRIM_NS_POSTFIX }  }
