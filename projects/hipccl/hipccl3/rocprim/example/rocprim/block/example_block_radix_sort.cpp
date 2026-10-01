// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../example_utils.hpp"

// Kernel 1: block-level radix sort for keys-only
__global__
void block_radix_sort_keys_kernel(const int* d_in, int* d_out)
{
    constexpr int BlockSize      = 4;
    constexpr int ItemsPerThread = 2;

    using block_sort_t = rocprim::block_radix_sort<int, BlockSize, ItemsPerThread>;

    __shared__ typename block_sort_t::storage_type storage;

    const int tid  = threadIdx.x;
    const int base = tid * ItemsPerThread;

    int keys[ItemsPerThread];
    for(int i = 0; i < ItemsPerThread; i++)
    {
        keys[i] = d_in[base + i];
    }

    // Sort in ascending order (full bit range by default)
    block_sort_t().sort(keys, storage);

    // Store back in blocked order -> the whole block is globally sorted now
    for(int i = 0; i < ItemsPerThread; i++)
    {
        d_out[base + i] = keys[i];
    }
}

// Kernel 2: block-level radix sort for key-value pairs
__global__
void block_radix_sort_pairs_kernel(const int* d_keys_in,
                                   const int* d_vals_in,
                                   int*       d_keys_out,
                                   int*       d_vals_out)
{
    constexpr int BlockSize      = 4;
    constexpr int ItemsPerThread = 2;

    using block_sort_kv_t = rocprim::block_radix_sort<int, BlockSize, ItemsPerThread, int>;

    __shared__ typename block_sort_kv_t::storage_type storage;

    const int tid  = threadIdx.x;
    const int base = tid * ItemsPerThread;

    int keys[ItemsPerThread];
    int vals[ItemsPerThread];

    for(int i = 0; i < ItemsPerThread; i++)
    {
        keys[i] = d_keys_in[base + i];
        vals[i] = d_vals_in[base + i];
    }

    // Sort keys ascending and permute values accordingly
    block_sort_kv_t().sort(keys, vals, storage);

    // Store back results in blocked order
    for(int i = 0; i < ItemsPerThread; i++)
    {
        d_keys_out[base + i] = keys[i];
        d_vals_out[base + i] = vals[i];
    }
}

int main()
{
    std::vector<int> keys_in = {7, 3, 5, 1, 6, 2, 4, 0};
    std::vector<int> vals_in = {70, 30, 50, 10, 60, 20, 40, 0};
    const int        size    = static_cast<int>(keys_in.size()); // 8

    // Device buffers
    common::device_ptr<int> d_keys_in(keys_in);
    common::device_ptr<int> d_vals_in(vals_in);
    common::device_ptr<int> d_keys_sorted(size);
    common::device_ptr<int> d_vals_sorted(size);

    // 1. keys-only sort
    hipLaunchKernelGGL(block_radix_sort_keys_kernel,
                       dim3(1),
                       dim3(4),
                       0,
                       0,
                       d_keys_in.get(),
                       d_keys_sorted.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto keys_only_out = d_keys_sorted.load();
    // Expected ascending order:
    std::vector<int> expected_keys = {0, 1, 2, 3, 4, 5, 6, 7};

    bool pass_keys = true;
    for(int i = 0; i < size; i++)
    {
        pass_keys = pass_keys && (keys_only_out[i] == expected_keys[i]);
    }
    ASSERT_TRUE(pass_keys);

    // 2. key-value pairs sort
    hipLaunchKernelGGL(block_radix_sort_pairs_kernel,
                       dim3(1),
                       dim3(4),
                       0,
                       0,
                       d_keys_in.get(),
                       d_vals_in.get(),
                       d_keys_sorted.get(),
                       d_vals_sorted.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto keys_out = d_keys_sorted.load();
    const auto vals_out = d_vals_sorted.load();

    // After sorting ascending by keys, values should follow the same permutation.
    // Since vals_in = key*10, the expected values are {0, 10, 20, 30, 40, 50, 60, 70}.
    std::vector<int> expected_vals = {0, 10, 20, 30, 40, 50, 60, 70};

    bool pass_pairs = true;
    for(int i = 0; i < size; i++)
    {
        pass_pairs
            = pass_pairs && (keys_out[i] == expected_keys[i]) && (vals_out[i] == expected_vals[i]);
    }
    ASSERT_TRUE(pass_pairs);

    return 0;
}
