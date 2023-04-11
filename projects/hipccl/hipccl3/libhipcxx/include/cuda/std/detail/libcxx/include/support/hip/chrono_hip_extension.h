//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _LIBCUDACXX_SUPPORT_HIP_EXTENSION_H 
#define _LIBCUDACXX_SUPPORT_HIP_EXTENSION_H

/**
 * For C++20, the standard requires that chrono::system_clock yields a UNIX timestamp (see https://en.cppreference.com/w/cpp/chrono/system_clock).
 * There is currently no UNIX timestamp counter available on AMD hardware. This header file implements a workaround for C++20.
 * The idea is to send an initial host timestamp to the device and to store it in global memory.
 * IMPORTANT: This is an EXPERIMENTAL workaround. 
 * IMPORTANT: Any application requiring a C++20 conforming system clock (i.e. with UNIX timestamp epoch) needs to enable the workaround according to the following steps:
 * 1) The compile flag _LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP needs to be set (-D_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP).
 * 2) The linker flag --fgpu-rdc must be set.
 * 3) The macro LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS needs to be called at file scope level in a single translation unit, usually where the main function is located.
 *    The header "cuda/std/chrono" must be included to make this macro available.
 * 4) cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_current_device() or cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_device()
 *    need to be called on the host once to initialize the system clock for a given device.
 * 5) Subsequent calls to cuda::std::system_clock::now() will then return time_points starting at UNIX time.
 * 
 * Example source code file:
 * #include "cuda/std/chrono"
 * //... (other headers)
 * //The following macro call defines the state we need on the device (outside of any function at file scope) 
 * LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS   
 * //...
 * int main(int argc, char **argv) {
 * // initializes the system clock on the current device  
 *    cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_current_device();
 * // use the system clock on the device
 *    someKernelUsingSysclock<<<...>>>(...)
 * }
*/

#include "hip/hip_runtime.h"

#define LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS \
    __device__ uint64_t cuda::std::chrono::hip_gpu_ext::__unix_sysclock0 = 0; \
    __device__ uint64_t cuda::std::chrono::hip_gpu_ext::__offset_devclock0 = 0; \
    __device__ bool cuda::std::chrono::hip_gpu_ext::__is_sysclock_initialized = false;

namespace hip_gpu_ext {

extern __device__ uint64_t __unix_sysclock0;
extern __device__ uint64_t __offset_devclock0; 
extern __device__ bool     __is_sysclock_initialized;

inline __global__ void initialize_amdgpu_sysclock_kernel(uint64_t __host_unix_sysclock0) {
    __unix_sysclock0 = __host_unix_sysclock0;
    __offset_devclock0 = __builtin_amdgcn_s_memrealtime();
    __is_sysclock_initialized = true;
}

/* 
// This is a potential workaround to avoid a manual call to initialize_amdgpu_sysclock().
// An issue with this method is that the constructor is being invoked before main at which point
// any kernel calls are failing (UB).
  static struct amdgpu_sysclock_info {
    //no contents needed
    amdgpu_sysclock_info() {
        initialize_amdgpu_sysclock_kernel<<<1,1>>>(
            ::std::chrono::duration_cast<::std::chrono::nanoseconds>(
                ::std::chrono::system_clock::now().time_since_epoch()
            ).count());
            hipError_t hip_error = hipGetLastError();
            assert(hip_error==hipSuccess);
           
        hipDeviceSynchronize();
    }
} amdgpu_sysclock_info;*/

inline hipError_t initialize_amdgpu_sysclock_on_current_device() _NOEXCEPT {
    hipError_t __hip_error;
    initialize_amdgpu_sysclock_kernel<<<1,1>>>(
        ::std::chrono::duration_cast<::std::chrono::nanoseconds>(
                ::std::chrono::system_clock::now().time_since_epoch()
        ).count()
        );
    __hip_error = hipGetLastError(); assert(__hip_error==hipSuccess);
    return __hip_error;
}

inline hipError_t initialize_amdgpu_sysclock_on_device(int __device_id) _NOEXCEPT {
    hipError_t __hip_error;
    int __current_device_id;
    __hip_error = hipGetDevice(&__current_device_id); assert(__hip_error==hipSuccess);
    __hip_error = hipSetDevice(__device_id); assert(__hip_error==hipSuccess);
    initialize_amdgpu_sysclock_kernel<<<1,1>>>(
        ::std::chrono::duration_cast<::std::chrono::nanoseconds>(
                ::std::chrono::system_clock::now().time_since_epoch()
        ).count()
        );
    __hip_error = hipGetLastError(); assert(__hip_error==hipSuccess);
    __hip_error = hipDeviceSynchronize(); assert(__hip_error==hipSuccess);
    __hip_error = hipSetDevice(__current_device_id); assert(__hip_error==hipSuccess);
    return __hip_error;
}

}

#endif // _LIBCUDACXX_SUPPORT_HIP_EXTENSION_H