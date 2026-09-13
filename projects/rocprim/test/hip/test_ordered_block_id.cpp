// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "../../common/utils_device_ptr.hpp"
// required rocprim headers
#include <rocprim/device/detail/ordered_block_id.hpp>
#include <rocprim/intrinsics/atomic.hpp>

__global__
void test_kernel_deadlock(unsigned int* flags)
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
bool test_func_deadlock(int block_count, int thread_count)
{
    common::device_ptr<unsigned int> d_flags(block_count);

    test_kernel_deadlock<<<block_count, thread_count>>>(d_flags.get());

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    auto h_vec = d_flags.load();
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

    EXPECT_TRUE(test_func_deadlock(1, 1));
    EXPECT_TRUE(test_func_deadlock(10, 10));
    EXPECT_TRUE(test_func_deadlock(100, 100));
    EXPECT_TRUE(test_func_deadlock(1000, 1000));
    EXPECT_TRUE(test_func_deadlock(3000, 1024));
    EXPECT_TRUE(test_func_deadlock(5000, 1024));
    EXPECT_TRUE(test_func_deadlock(10000, 1024));

    SUCCEED();
}

using namespace rocprim::detail;

__global__
void test_kernel(ordered_block_id<> ordered_bid, uint32_t* device_output_ids)
{
    __shared__ ordered_block_id<>::storage_type ordered_bid_storage;

    const auto gid      = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto tid      = threadIdx.x;
    const auto block_id = ordered_bid.get(tid, ordered_bid_storage);

    device_output_ids[gid] = block_id;
}

TEST(OrderedBlockID, Unique)
{
    using ordered_bid_type = ordered_block_id<>;

    size_t                     temp_storage_size   = 0;
    ordered_bid_type::id_type* ordered_bid_storage = nullptr;

    HIP_CHECK(
        temp_storage::partition(nullptr,
                                temp_storage_size,
                                temp_storage::make_linear_partition(temp_storage::make_partition(
                                    &ordered_bid_storage,
                                    ordered_bid_type::get_temp_storage_layout()))));

    common::device_ptr<char> temp_storage(temp_storage_size);

    HIP_CHECK(
        temp_storage::partition(temp_storage.get(),
                                temp_storage_size,
                                temp_storage::make_linear_partition(temp_storage::make_partition(
                                    &ordered_bid_storage,
                                    ordered_bid_type::get_temp_storage_layout()))));

    auto ordered_bid = ordered_bid_type::create(ordered_bid_storage);

    constexpr size_t grid_dim  = 8;
    constexpr size_t block_dim = 8;

    common::device_ptr<uint32_t> output_ids(grid_dim * block_dim);

    test_kernel<<<grid_dim, block_dim>>>(ordered_bid, output_ids.get());

    const auto h_output_ids = output_ids.load();

    for(uint32_t block = 0; block < grid_dim; block++)
    {
        const auto base = block * block_dim;
        uint32_t   id   = h_output_ids[base];

        // All threads within the block must have the same ID
        for(uint32_t thread = 0; thread < block_dim; thread++)
        {
            ASSERT_EQ(h_output_ids[base + thread], id);
        }
    }

    // Check that the assigned block IDs form the complete ordered sequence [0, grid_dim)
    std::vector<uint32_t> block_ids(grid_dim);
    for(uint32_t block = 0; block < grid_dim; block++)
    {
        block_ids[block] = h_output_ids[block * block_dim];
    }
    std::sort(block_ids.begin(), block_ids.end());

    for(uint32_t i = 0; i < grid_dim; i++)
    {
        ASSERT_EQ(block_ids[i], i);
    }
}
