// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _LIBCUDACXX_SUPPORT_HIP_EXTENSION_H 
#define _LIBCUDACXX_SUPPORT_HIP_EXTENSION_H

/**
 * For C++20, the standard requires that chrono::system_clock yields a UNIX timestamp (see https://en.cppreference.com/w/cpp/chrono/system_clock).
 * There is currently no UNIX timestamp counter available on AMD hardware. This header file implements a workaround for C++20.
 * The idea is to send an initial host timestamp to the device and to store it in global memory.
 * IMPORTANT: This is an EXPERIMENTAL workaround. 
 * IMPORTANT: Any application requiring a C++20 conforming system clock (i.e. with UNIX timestamp epoch) needs to enable the workaround according to the following steps:
 * 1) The compile flag _LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP needs to be set (-D_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP).
 * 2) The linker flag -fgpu-rdc must be set.
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
     __constant__ long long cuda::std::chrono::hip_gpu_ext::__unix_sysclock0_host_ticks = -1; \
     __constant__ long long cuda::std::chrono::hip_gpu_ext::__offset_devclock0 = -1; \

#define LIBCUDACXX_HIP_CHECK(command)                                     \
{                                                                         \
    __hip_error = command;                                                \
    assert(__hip_error==hipSuccess);                                      \
}

namespace hip_gpu_ext {

extern  __constant__ long long __unix_sysclock0_host_ticks;
extern  __constant__ long long __offset_devclock0; 

__global__ void get_sysclock_offset_kernel(long long *__d_dev_sysclock_offset) {
    *__d_dev_sysclock_offset = wall_clock64();
}

inline hipError_t initialize_amdgpu_sysclock_on_current_device() _NOEXCEPT {
    hipError_t __hip_error;
    
    long long* __d_dev_sysclock_offset; 
    LIBCUDACXX_HIP_CHECK(hipMalloc(&__d_dev_sysclock_offset, sizeof(long long)));

    // get device sysclock offset and host unix timestamp at approximately the same time
    // FIXME(HIP): There will be some delays, e.g., due to kernel call overhead, so the clocks on the device and on the host will not be fully synchronized
    get_sysclock_offset_kernel<<<1,1>>>(__d_dev_sysclock_offset);

    LIBCUDACXX_HIP_CHECK(hipGetLastError(); assert(__hip_error==hipSuccess));

    long long __h_host_unix_sysclock_ticks_elapsed =  ::std::chrono::system_clock::now().time_since_epoch().count();

    long long __h_dev_sysclock_offset = -1;
    LIBCUDACXX_HIP_CHECK(hipMemcpy(&__h_dev_sysclock_offset, __d_dev_sysclock_offset, sizeof(long long), hipMemcpyDeviceToHost));

    LIBCUDACXX_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(__unix_sysclock0_host_ticks), &__h_host_unix_sysclock_ticks_elapsed, sizeof(long long))); 
    LIBCUDACXX_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(__offset_devclock0), &__h_dev_sysclock_offset, sizeof(long long))); 
    return __hip_error;
}

inline hipError_t initialize_amdgpu_sysclock_on_device(int __device_id) _NOEXCEPT {
    hipError_t __hip_error;
    int __current_device_id;
    LIBCUDACXX_HIP_CHECK(hipGetDevice(&__current_device_id)); 
    LIBCUDACXX_HIP_CHECK(hipSetDevice(__device_id));
    
    LIBCUDACXX_HIP_CHECK(initialize_amdgpu_sysclock_on_current_device());

    LIBCUDACXX_HIP_CHECK(hipDeviceSynchronize()); 
    LIBCUDACXX_HIP_CHECK(hipSetDevice(__current_device_id)); 
    return __hip_error;
}

}
#endif // _LIBCUDACXX_SUPPORT_HIP_EXTENSION_H