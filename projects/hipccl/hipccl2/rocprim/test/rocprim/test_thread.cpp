// MIT License
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/intrinsics/thread.hpp>

#include "test_utils.hpp"

template<
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
struct params
{
    static constexpr unsigned int block_size_x = BlockSizeX;
    static constexpr unsigned int block_size_y = BlockSizeY;
    static constexpr unsigned int block_size_z = BlockSizeZ;
};

template<class Params>
class RocprimThreadTests : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<32, 1, 1>,
    params<64, 1, 1>,
    params<128, 1, 1>,
    params<256, 1, 1>,
    params<512, 1, 1>,
    params<1024, 1, 1>,

    params<16, 2, 1>,
    params<32, 2, 1>,
    params<64, 2, 1>,
    params<128, 2, 1>,
    params<256, 2, 1>,
    params<512, 2, 1>,

    params<8, 2, 2>,
    params<16, 2, 2>,
    params<32, 2, 2>,
    params<64, 2, 2>,
    params<128, 2, 2>,
    params<256, 2, 2>
> Params;

TYPED_TEST_CASE(RocprimThreadTests, Params);

template<
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
__global__
__launch_bounds__(1024, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void flat_id_kernel(unsigned int* device_output)
{
    unsigned int thread_id = rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
    device_output[thread_id] = thread_id;
}

TYPED_TEST(RocprimThreadTests, FlatBlockThreadID)
{
    using Type = unsigned int;
    constexpr size_t block_size_x = TestFixture::params::block_size_x;
    constexpr size_t block_size_y = TestFixture::params::block_size_y;
    constexpr size_t block_size_z = TestFixture::params::block_size_z;
    constexpr size_t block_size = block_size_x * block_size_y * block_size_z;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<Type> output(block_size, 0);

        // Calculate expected results on host
        std::vector<Type> expected(block_size, 0);
        for(size_t i = 0; i < block_size; i++)
        {
            expected[i] = i;
        }

        // Preparing device
        Type* device_output;
        HIP_CHECK(hipMalloc(&device_output, block_size * sizeof(typename decltype(output)::value_type)));

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                flat_id_kernel<
                    block_size_x, block_size_y, block_size_z
                >
            ),
            dim3(1), dim3(block_size_x, block_size_y, block_size_z), 0, 0,
            device_output
        );

        // Reading results from device
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
__global__
__launch_bounds__(1024, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void block_id_kernel(unsigned int* device_output)
{
    unsigned int block_id = rocprim::flat_block_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
    if(hipThreadIdx_x)
    {
        device_output[block_id] = block_id;
    }
}

TYPED_TEST(RocprimThreadTests, FlatBlockID)
{
    using Type = unsigned int;
    constexpr size_t block_size_x = TestFixture::params::block_size_x;
    constexpr size_t block_size_y = TestFixture::params::block_size_y;
    constexpr size_t block_size_z = TestFixture::params::block_size_z;
    constexpr size_t block_size = block_size_x * block_size_y * block_size_z;
    const size_t size = block_size * block_size;
    const auto grid_size = size / block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size() || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<Type> output(grid_size, 0);

        // Calculate expected results on host
        std::vector<Type> expected(grid_size, 0);
        for(size_t i = 0; i < grid_size; i++)
        {
            expected[i] = i;
        }

        // Preparing device
        Type* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                block_id_kernel<
                    block_size_x, block_size_y, block_size_z
                >
            ),
            dim3(block_size_x, block_size_y, block_size_z), dim3(block_size_x, block_size_y, block_size_z), 0, 0,
            device_output
        );

        // Reading results from device
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(typename decltype(output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        HIP_CHECK(hipFree(device_output));
    }
}
