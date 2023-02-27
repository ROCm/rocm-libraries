#include <hip/hip_runtime.h>
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

// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.
#include <stdio.h>
#include <stdlib.h>


void list_devices()
{
    int device_count;
    hipGetDeviceCount(&device_count);
    printf("HIP devices found: %d\n", device_count);

    int selected_device;
    hipGetDevice(&selected_device);

    for (int dev = 0; dev < device_count; ++dev)
    {
        hipDeviceProp_t device_prop;
        hipGetDeviceProperties(&device_prop, dev);

        printf("Device %d: \"%s\", ", dev, device_prop.name);
        if(dev == selected_device)
            printf("Selected, ");
        else
            printf("Unused, ");

        printf("CDNA %s\n", device_prop.gcnArchName);
        printf("SM%d%d, %zu [bytes]\n",
            device_prop.major, device_prop.minor,
            device_prop.totalGlobalMem);
    }
}


__host__ __device__
int fake_main(int, char**);

int cuda_thread_count = 1;

__global__
void fake_main_kernel(int * ret)
{
   *ret = fake_main(0, NULL);
}

#define CUDA_CALL(err, ...) \
    do { \
        err = __VA_ARGS__; \
        if (err != hipSuccess) \
        { \
            printf("CUDA ERROR, line %d: %s: %s\n", __LINE__,\
                   hipGetErrorName(err), hipGetErrorString(err)); \
            exit(1); \
        } \
    } while (false)

int main(int argc, char** argv)
{
    // Check if the CUDA driver/runtime are installed and working for sanity.
    hipError_t err;
    CUDA_CALL(err, hipDeviceSynchronize());

    list_devices();

    int ret = fake_main(argc, argv);
    if (ret != 0)
    {
        return ret;
    }

    int * cuda_ret = 0;
    CUDA_CALL(err, hipMalloc(&cuda_ret, sizeof(int)));

    fake_main_kernel<<<1, cuda_thread_count>>>(cuda_ret);
    //hipLaunchKernel(static_cast<void*> (fake_main_kernel), 1, cuda_thread_count, 0, 0, cuda_ret);
    
    CUDA_CALL(err, hipGetLastError());
    CUDA_CALL(err, hipDeviceSynchronize());
    CUDA_CALL(err, hipMemcpy(&ret, cuda_ret, sizeof(int), hipMemcpyDeviceToHost));
    CUDA_CALL(err, hipFree(cuda_ret));

    return ret;
}

#if defined(__HIP_PLATFORM_AMD__)
#define main __device__ __host__ fake_main
#else
#define main fake_main
#endif
