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

// A tiny kernel to produce the initial value on device
__global__
void compute_intermediate(int* out, int val)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
        *out = val;
}

int main()
{
    // Host input
    std::vector<int> h_in = {1, 2, 3, 4, 5, 6, 7, 8};
    const size_t     N    = h_in.size();

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out(N);

    // Allocate device-side single int for the future_value
    int* d_intermediate = nullptr;
    HIP_CHECK(hipMalloc(&d_intermediate, sizeof(int)));

    // Create a stream to enforce ordering without host sync
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Launch producer: write initial value on device (e.g., 5)
    hipLaunchKernelGGL(compute_intermediate, dim3(1), dim3(1), 0, stream, d_intermediate, 5);

    // Wrap device pointer as future_value<int>
    const auto init = rocprim::future_value<int>{d_intermediate};

    // Query temp storage for rocprim::exclusive_scan
    void*  d_temp_storage = nullptr;
    size_t temp_bytes     = 0;

    HIP_CHECK(
        rocprim::exclusive_scan(d_temp_storage, // temporary storage pointer (nullptr to query)
                                temp_bytes, // will be filled with required size
                                d_in.get(), // input iterator
                                d_out.get(), // output iterator
                                init, // initial value (future_value<int>)
                                N, // number of items
                                rocprim::plus<int>{}, // scan op
                                stream));

    // Allocate temp storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_bytes));

    HIP_CHECK(rocprim::exclusive_scan(d_temp_storage,
                                      temp_bytes,
                                      d_in.get(),
                                      d_out.get(),
                                      init,
                                      N,
                                      rocprim::plus<int>{},
                                      stream));

    // Ensure completion
    HIP_CHECK(hipStreamSynchronize(stream));

    // Exclusive scan of [1..8] with init=5 => [5, 6, 8, 11, 15, 20, 26, 33]
    const auto       h_out    = d_out.load();
    std::vector<int> expected = {5, 6, 8, 11, 15, 20, 26, 33};

    bool passed = true;
    for(size_t i = 0; i < N; ++i)
    {
        passed = passed && (h_out[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    // Cleanup
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_intermediate));
    HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
