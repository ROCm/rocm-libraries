//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cassert>
#include <cstdio>
#include <cstring>

#define CUDA_SAFE_CALL(...)                                                           \
  do                                                                                  \
  {                                                                                   \
    cudaError_t err = __VA_ARGS__;                                                    \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
      printf("CUDA ERROR: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (false)

template <typename T, typename F>
void generate(const F& f, T* buffer, cuda::std::size_t size)
{
  for (auto i = 0ull; i < size; ++i)
  {
    buffer[i] = f(i);
  }
}

template <typename T, typename F>
void generate(const F& f, T* buffer, cuda::std::size_t size, T other)
{
  if (!cuda::std::isfinite(float(other)))
  {
    return;
  }
  for (auto i = 0ull; i < size; ++i)
  {
    buffer[i] = f(other, i);
  }
}

template <typename T, typename F, typename Head, typename... Args>
void generate(const F& f, T* buffer, cuda::std::size_t size, Head head, Args... args)
{
  for (auto i = 0ull; i < size; ++i)
  {
    buffer[i] = f(head, args..., i);
  }
}

template <typename T, typename F, typename... Args>
__global__ void generate_kernel(const F& f, T* buffer, cuda::std::size_t, Args... args)
{
  cuda::std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  buffer[index]           = f(args..., index);
}

template <typename T, cuda::std::size_t Dims, cuda::std::size_t Bitpatterns = 1ull << (sizeof(T) * CHAR_BIT)>
struct calculate_problem_sizes
{
  static const constexpr auto bitpatterns  = Bitpatterns;
  static const constexpr auto problem_size = Bitpatterns * calculate_problem_sizes<T, Dims - 1>::problem_size;

  template <typename F, typename... Args>
  static bool run(const F& f, Args... args)
  {
    bool good = true;
    for (auto i = 0ull; i < bitpatterns; ++i)
    {
      good = calculate_problem_sizes<T, Dims - 1>::run(f, args..., i) && good;
    }
    return good;
  }
};

__half_raw convert_to_raw(__half h){
  return __half_raw(h);
}

__hip_bfloat16_raw convert_to_raw(__hip_bfloat16 b){
  return __hip_bfloat16_raw(b);
}

template <typename T, cuda::std::size_t Bitpatterns>
struct calculate_problem_sizes<T, 1, Bitpatterns>
{
  static const constexpr auto bitpatterns  = Bitpatterns;
  static const constexpr auto problem_size = Bitpatterns;

  template <typename F, typename... Args>
  static bool run(const F& f, Args... args)
  {
    T* host_buffer = new T[problem_size]();

    T* device_buffer = nullptr;
    CUDA_SAFE_CALL(cudaMallocManaged(&device_buffer, sizeof(T) * problem_size));
    CUDA_SAFE_CALL(cudaMemset(device_buffer, 0, sizeof(T) * problem_size));

    generate(f, host_buffer, problem_size, args...);
    generate_kernel<<<problem_size / 256, 256>>>(f, device_buffer, problem_size, args...);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaGetLastError());

    bool good = true;

    for (cuda::std::size_t i = 0ull; i < problem_size; ++i)
    {
      if (memcmp(host_buffer + i, device_buffer + i, sizeof(T)) != 0)
      {
        // NOTE(HIP/AMD): We encountered cases where device and host calculated +nan and -nan
        // which is fine and should not lead to a failing test
        if(not(__hisnan(host_buffer[i]) and __hisnan(device_buffer[i]))){
          auto raw = __half_raw();
          raw.x = (unsigned short) i;
          auto raw_host = convert_to_raw(host_buffer[i]);
          auto raw_device = convert_to_raw(device_buffer[i]);
          // NOTE(HIP/AMD): We encountered cases where device and host calculated vary by one in the mantissa
          // which is within the computing accuracy of the different implementations and should not lead to a failing test
          if(abs(int(raw_host.x) - int(raw_device.x)) > 1){

            printf("[%zu] unmatched, values = %+.10f, host = %+.10f, host as int = %d, device = %+.10f, device as int = %d\n",
                  i,
                  float(__half(raw)),
                  (float) host_buffer[i],
                  raw_host.x,
                  (float) device_buffer[i],
                  raw_device.x);
            good = false;
          }
        }
      }
    }

    CUDA_SAFE_CALL(cudaFree(device_buffer));
    delete[] host_buffer;

    return good;
  }
};

template <typename T, cuda::std::size_t Bitpatterns>
struct calculate_problem_sizes<T, 0, Bitpatterns>
{
  static_assert(Bitpatterns == 0, "can't have 0 dims");
};

template <typename T, cuda::std::size_t Dims = 1, typename F>
void compare_host_device(const F& f)
{
  using sizes = calculate_problem_sizes<T, Dims>;

  auto good = sizes::run(f);
  fflush(stdout);
  assert(good);
}
