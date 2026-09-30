//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
#ifndef LIBCUDACXX_FORCE_INCLUDE_HIP
#define LIBCUDACXX_FORCE_INCLUDE_HIP

#include "hip/hip_runtime.h"
#include <hip/thread>

// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.
#include <stdio.h>
#include <stdlib.h>

#define HIP_CALL(err, ...) \
    do { \
        err = __VA_ARGS__; \
        if (err != hipSuccess) \
        { \
            printf("HIP ERROR, line %d: %s: %s\n", __LINE__,\
                   hipGetErrorName(err), hipGetErrorString(err)); \
            exit(1); \
        } \
    } while (false)

void list_devices()
{
    hipError_t err;
    int device_count;
    HIP_CALL(err, hipGetDeviceCount(&device_count));
    printf("HIP devices found: %d\n", device_count);

    int selected_device;
    HIP_CALL(err, hipGetDevice(&selected_device));

    for (int dev = 0; dev < device_count; ++dev)
    {
        hipDeviceProp_t device_prop;
        HIP_CALL(err, hipGetDeviceProperties(&device_prop, dev));

        printf("Device %d: \"%s\", ", dev, device_prop.name);
        if(dev == selected_device)
            printf("Selected, ");
        else
            printf("Unused, ");

        printf("CDNA %s\n", device_prop.gcnArchName);
        printf("CU%d%d, %zu [bytes]\n",
            device_prop.major, device_prop.minor,
            device_prop.totalGlobalMem);
    }
}


__host__ __device__
int fake_main(int, char**);

unsigned int cuda_thread_count = 1;

__global__
void fake_main_kernel(int * ret)
{
   *ret = fake_main(0, NULL);
}

int main(int argc, char** argv)
{
    // Check if the HIP driver/runtime are installed and working for sanity.
    hipError_t err;
    HIP_CALL(err, hipDeviceSynchronize());

    list_devices();

    int ret = fake_main(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    int * hip_ret = 0;
    HIP_CALL(err, hipMalloc(&hip_ret, sizeof(int)));

#ifdef TEST_NO_HIP_THREAD
    fake_main_kernel<<<1, cuda_thread_count>>>(hip_ret);
#else
    {
        hip::wthread thd(cuda_thread_count, [] __device__(int *ret) { *ret = fake_main(0, NULL); }, hip_ret);
        thd.join();
    }
#endif
     
    HIP_CALL(err, hipGetLastError());
    HIP_CALL(err, hipDeviceSynchronize());
    HIP_CALL(err, hipMemcpy(&ret, hip_ret, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CALL(err, hipFree(hip_ret));

    return ret;
}

#if defined(__HIP_PLATFORM_AMD__)
#define main __device__ __host__ fake_main
#else
#define main fake_main
#endif

#endif
