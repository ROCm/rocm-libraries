// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "hip/hip_runtime.h"
#include <unistd.h>
#include <cuda/std/chrono>

LIBCUDACXX_HIP_DEFINE_SYSCLOCK_VARS

/*
* This example tests the workaround described in include/cuda/std/detail/libcxx/include/support/hip/chrono_hip_extension.h,
* which enables UNIX timestamps for HIP on AMD devices (for chrono::system_clock::now()).
* It needs to be compiled with -std=c++20 support and the compile time flag -D_LIBCUDACXX_EXPERIMENTAL_CHRONO_HIP.
*/

#define HIP_CHECK(command)                                                   \
{                                                                            \
  if (command != hipSuccess) {                                               \
    printf("HIP ERROR: %s", hipGetErrorString(command));                     \
    std::abort();                                                            \
  }                                                                          \
}

__global__ void print_sysclock() {
    using namespace cuda::std::chrono;
   
    auto time = system_clock::now();
    printf("current (UNIX) time stamp on device %lld\n", duration_cast<nanoseconds>(time.time_since_epoch()).count());
    printf("period of device sysclock: %ld/%ld\n", system_clock::period::num, system_clock::period::den);
    
    auto days_since_epoch = floor<days>(time);
    auto ymd = year_month_day(days_since_epoch);
    printf("current date (UTC): %d-%u-%u\n", static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()), static_cast<unsigned>(ymd.day()));
}

int main(){
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(cuda::std::chrono::hip_gpu_ext::initialize_amdgpu_sysclock_on_current_device());
    print_sysclock<<<1, 1>>>();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

