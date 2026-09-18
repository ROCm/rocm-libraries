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

int main()
{
    // Host-side input data
    std::vector<int> h_a = {9, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> h_b(9, 0);

    // Device buffers a, b
    int* d_a = nullptr;
    int* d_b = nullptr;
    HIP_CHECK(hipMalloc(&d_a, h_a.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, h_b.size() * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), h_b.size() * sizeof(int), hipMemcpyHostToDevice));

    const int num_copies = 3;

    // host arrays of device pointers
    int* h_sources[num_copies]      = {d_a + 0, d_a + 4, d_a + 7};
    int* h_destinations[num_copies] = {d_b + 5, d_b + 2, d_b + 0};

    size_t h_sizes[num_copies] = {
        3 * sizeof(int), // a[0..2] -> b[5..7]
        2 * sizeof(int), // a[4..5] -> b[2..3]
        2 * sizeof(int) // a[7..8] -> b[0..1]
    };

    int**   d_sources      = nullptr;
    int**   d_destinations = nullptr;
    size_t* d_sizes        = nullptr;

    HIP_CHECK(hipMalloc(&d_sources, num_copies * sizeof(int*)));
    HIP_CHECK(hipMalloc(&d_destinations, num_copies * sizeof(int*)));
    HIP_CHECK(hipMalloc(&d_sizes, num_copies * sizeof(size_t)));

    HIP_CHECK(hipMemcpy(d_sources, h_sources, num_copies * sizeof(int*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_destinations,
                        h_destinations,
                        num_copies * sizeof(int*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sizes, h_sizes, num_copies * sizeof(size_t), hipMemcpyHostToDevice));

    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temp storage size
    HIP_CHECK(rocprim::batch_memcpy(d_temp_storage,
                                    temp_storage_bytes,
                                    d_sources,
                                    d_destinations,
                                    d_sizes,
                                    num_copies));

    // Allocate temp storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Do actual device-side memcpy
    HIP_CHECK(rocprim::batch_memcpy(d_temp_storage,
                                    temp_storage_bytes,
                                    d_sources,
                                    d_destinations,
                                    d_sizes,
                                    num_copies));

    std::vector<int> out_b(h_b.size());
    HIP_CHECK(hipMemcpy(out_b.data(), d_b, out_b.size() * sizeof(int), hipMemcpyDeviceToHost));

    // Expected result according to the pattern above
    std::vector<int> expected = {7, 8, 4, 5, 0, 9, 1, 2, 0};

    int size = expected.size();

    bool passed = true;
    for(int i = 0; i < size; ++i)
    {
        passed = passed && (out_b[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    // Cleanup
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_sources));
    HIP_CHECK(hipFree(d_destinations));
    HIP_CHECK(hipFree(d_sizes));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));

    return 0;
}
