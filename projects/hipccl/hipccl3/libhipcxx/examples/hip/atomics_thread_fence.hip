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
#include <stdio.h>
#include <cuda/std/atomic>

__global__ void my_kernel(int* data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // perform some computation
    data[tid] = tid * 2;

    // wait for all threads in the block to finish computing
    cuda::std::atomic_thread_fence(cuda::std::memory_order_seq_cst);
    // atomic_thread_fence(__threadfence_block);

    // do some more computation that requires all threads to finish
    data[tid] += data[tid + 1];
}

int main()
{
    int num_threads = 128;
    int num_blocks = 4;
    int data_size = num_threads * num_blocks;

    // allocate device memory
    int* data_dev;
    hipMalloc((void**)&data_dev, sizeof(int) * data_size);

    // launch kernel
    my_kernel<<<num_blocks, num_threads>>>(data_dev);

    // copy data back to host
    int* data_host = (int*)malloc(sizeof(int) * data_size);
    hipMemcpy(data_host, data_dev, sizeof(int) * data_size, hipMemcpyDeviceToHost);

    // print results
    for (int i = 0; i < data_size - 1; i++)
    {
        if(data_host[i] != (i * 2) + (i + 1) *2) 
            std::abort();
    }

    // free memory
    free(data_host);
    hipFree(data_dev);

    return 0;
}
