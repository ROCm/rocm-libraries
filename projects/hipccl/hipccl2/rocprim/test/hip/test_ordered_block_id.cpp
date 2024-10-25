// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <thread>

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/intrinsics/atomic.hpp>

__global__
void test_kernel(unsigned int* flags)
{
    const auto bid = blockIdx.x;
    const auto tid = threadIdx.x;
    if(bid != 0)
    {
        if(tid == 0)
        {
            while(rocprim::detail::atomic_load(&flags[bid - 1]) != 1)
            {
                continue;
            }
        }
    }
    if(tid == 0)
    {
        rocprim::detail::atomic_store(&flags[bid], 1);
    }
}

__host__
bool test_func(int block_count, int thread_count)
{
    unsigned int*             d_flags;
    std::vector<unsigned int> h_vec(block_count);
    HIP_CHECK(hipMalloc(&d_flags, block_count * sizeof(unsigned int)));
    test_kernel<<<block_count, thread_count>>>(d_flags);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_vec.data(),
                        d_flags,
                        block_count * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_flags));
    for(const auto i : h_vec)
    {
        if(i != 1)
        {
            return false;
        }
    }
    return true;
}

TEST(OrderedBlockId, Deadlock)
{
    // timer
    std::thread(
        [&]
        {
            std::this_thread::sleep_for(std::chrono::seconds(60));
            FAIL();
        })
        .detach();

    EXPECT_TRUE(test_func(1, 1));
    EXPECT_TRUE(test_func(10, 10));
    EXPECT_TRUE(test_func(100, 100));
    EXPECT_TRUE(test_func(1000, 1000));
    EXPECT_TRUE(test_func(3000, 1024));
    EXPECT_TRUE(test_func(5000, 1024));
    EXPECT_TRUE(test_func(10000, 1024));

    SUCCEED();
}
