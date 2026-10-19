// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/config.h>

#include <thrust/device_vector.h>

// If you create a global `thrust::device_vector` with the default allocator,
// you'll get an error during program termination when the memory of the vector
// is freed, as the CUDA runtime cannot be used during program termination.
//
// To get around this, you can create your own allocator which ignores
// deallocation failures that occur because the CUDA runtime is shut down.

extern "C" cudaError_t cudaFreeIgnoreShutdown(void* ptr)
{
  cudaError_t const err = cudaFree(ptr);
  if (cudaSuccess == err || cudaErrorCudartUnloading == err)
  {
    return cudaSuccess;
  }
  return err;
}

using device_ignore_shutdown_memory_resource =
  thrust::system::cuda::detail::cuda_memory_resource<cudaMalloc, cudaFreeIgnoreShutdown, thrust::cuda::pointer<void>>;

template <typename T>
using device_ignore_shutdown_allocator =
  thrust::mr::stateless_resource_allocator<T, thrust::device_ptr_memory_resource<device_ignore_shutdown_memory_resource>>;

thrust::device_vector<double, device_ignore_shutdown_allocator<double>> d;

int main()
{
  d.resize(25);
}
