// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/config.hpp>
#include <hipcub/block/block_adjacent_difference.hpp>
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/thread/thread_operators.hpp>

template<
    class T,
    class Output,
    class BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct params_subtract
{
    using type = T;
    using output = Output;
    using binary_function = BinaryFunction;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

template<class ParamsSubtract>
class HipcubBlockAdjacentDifferenceSubtract : public ::testing::Test {
public:
    using params_subtract = ParamsSubtract;
};

struct custom_op1
{
    template<class T>
    HIPCUB_HOST_DEVICE
    T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

struct custom_op2
{
    template<class T>
    HIPCUB_HOST_DEVICE
    T operator()(const T& a, const T& b) const
    {
        return (b + b) - a;
    }
};

using ParamsSubtract
    = ::testing::Types<params_subtract<unsigned int, int, hipcub::Sum, 64U, 1>,
                       params_subtract<int, bool, custom_op1, 128U, 1>,
                       params_subtract<float, int, custom_op2, 256U, 1>,
                       params_subtract<test_utils::half, int, custom_op1, 256U, 1>,
                       params_subtract<test_utils::bfloat16, int, custom_op2, 256U, 1>,
                       params_subtract<int, bool, custom_op1, 256U, 1>,

                       params_subtract<float, int, hipcub::Sum, 37U, 1>,
                       params_subtract<long long, char, custom_op1, 510U, 1>,
                       params_subtract<unsigned int, long long, custom_op2, 162U, 1>,
                       params_subtract<unsigned char, bool, hipcub::Sum, 255U, 1>,

                       params_subtract<int, char, custom_op1, 64U, 2>,
                       params_subtract<int, short, custom_op2, 128U, 4>,
                       params_subtract<unsigned short, unsigned char, hipcub::Sum, 256U, 7>,
                       params_subtract<short, short, custom_op1, 512U, 8>,

                       params_subtract<double, int, custom_op2, 33U, 5>,
                       params_subtract<double, unsigned int, hipcub::Sum, 464U, 2>,
                       params_subtract<unsigned short, int, custom_op1, 100U, 3>,
                       params_subtract<short, bool, custom_op2, 234U, 9>>;

TYPED_TEST_SUITE(HipcubBlockAdjacentDifferenceSubtract, ParamsSubtract);

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void subtract_left_kernel(const T* input, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    if (blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.SubtractLeft(thread_items, thread_output, BinaryFunction{}, tile_predecessor_item);
    }
    else
    {
        adjacent_difference.SubtractLeft(thread_items, thread_output, BinaryFunction{});
    }

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
}

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void subtract_left_partial_tile_kernel(const T* input, int* tile_sizes, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    int tile_size = tile_sizes[blockIdx.x];

    if(blockIdx.x % 2 == 1)
    {
        const T tile_predecessor_item = input[block_offset - 1];
        adjacent_difference.SubtractLeftPartialTile(thread_items,
                                                    thread_output,
                                                    BinaryFunction{},
                                                    tile_size,
                                                    tile_predecessor_item);
    }
    else
    {
        adjacent_difference.SubtractLeftPartialTile(thread_items,
                                                    thread_output,
                                                    BinaryFunction{},
                                                    tile_size);
    }

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
}

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void subtract_right_kernel(const T* input, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    if (blockIdx.x % 2 == 0)
    {
        const T tile_successor_item = input[block_offset + items_per_block];
        adjacent_difference.SubtractRight(thread_items, thread_output, BinaryFunction{}, tile_successor_item);
    }
    else
    {
        adjacent_difference.SubtractRight(thread_items, thread_output, BinaryFunction{});
    }

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
}

template<
    typename T,
    typename Output,
    typename StorageType,
    typename BinaryFunction,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void subtract_right_partial_tile_kernel(const T* input, int* tile_sizes, StorageType* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    T thread_items[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, input + block_offset, thread_items);

    hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

    Output thread_output[ItemsPerThread];

    int tile_size = tile_sizes[blockIdx.x];
    
    adjacent_difference.SubtractRightPartialTile(thread_items, thread_output, BinaryFunction{}, tile_size);

    hipcub::StoreDirectBlocked(lid, output + block_offset, thread_output);
}

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractLeft)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(unsigned int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item == 0) 
                {
                    expected[i]
                        = static_cast<output_type>(block_index % 2 == 1 ? op(input[i], input[i - 1]) : input[i]);
                } 
                else 
                {
                    expected[i] = static_cast<output_type>(op(input[i], input[i - 1]));
                }
            }
        }

        // Preparing Device
        type* d_input;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_left_kernel<type, output_type, stored_type, 
                                     binary_function, block_size,
                                     items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(output,
                                    expected,
                                    std::max(test_utils::precision<type>::value,
                                             test_utils::precision<stored_type>::value)));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
}

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractLeftPartialTile)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type> input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        const std::vector<int> tile_sizes 
            = test_utils::get_random_data<int>(grid_size, 0, items_per_block, seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if (item < tile_sizes[block_index]) 
                {
                    if(item == 0) 
                    {
                        expected[i] = static_cast<output_type>(
                            block_index % 2 == 1 ? op(input[i], input[i - 1]) : input[i]);
                    } 
                    else 
                    {
                        expected[i] = static_cast<output_type>(op(input[i], input[i - 1]));
                    }
                }
                else
                {
                    expected[i] = static_cast<output_type>(input[i]);
                }
            }
        }

        // Preparing Device
        type* d_input;
        int* d_tile_sizes;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_tile_sizes, tile_sizes.data(),
                tile_sizes.size() * sizeof(tile_sizes[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_left_partial_tile_kernel<type, output_type, stored_type, 
                                                  binary_function, block_size,
                                                  items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_tile_sizes, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(output,
                                    expected,
                                    std::max(test_utils::precision<type>::value,
                                             test_utils::precision<stored_type>::value)));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_tile_sizes));
        HIP_CHECK(hipFree(d_output));
    }
}

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractRight)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type>     input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if(item == items_per_block - 1) 
                {
                    expected[i]
                        = static_cast<output_type>(block_index % 2 == 0 ? op(input[i], input[i + 1]) : input[i]);
                } 
                else 
                {
                    expected[i] = static_cast<output_type>(op(input[i], input[i + 1]));
                }
            }
        }

        // Preparing Device
        type* d_input;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_right_kernel<type, output_type, stored_type, 
                                     binary_function, block_size,
                                     items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_near(output,
                                    expected,
                                    std::max(test_utils::precision<type>::value,
                                             test_utils::precision<stored_type>::value)));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
}

TYPED_TEST(HipcubBlockAdjacentDifferenceSubtract, SubtractRightPartialTile)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::params_subtract::type;
    using binary_function = typename TestFixture::params_subtract::binary_function;

    using output_type = typename TestFixture::params_subtract::output;

    using stored_type = std::conditional_t<std::is_same<output_type, bool>::value, int, output_type>;

    constexpr size_t block_size = TestFixture::params_subtract::block_size;
    constexpr size_t items_per_thread = TestFixture::params_subtract::items_per_thread;
    static constexpr int items_per_block = block_size * items_per_thread;
    static constexpr int size = items_per_block * 20;
    static constexpr int grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        const std::vector<type>     input = test_utils::get_random_data<type>(size, 0, 10, seed_value);
        std::vector<stored_type> output(size);

        const std::vector<int> tile_sizes 
            = test_utils::get_random_data<int>(grid_size, 0, items_per_block, seed_value);

        // Calculate expected results on host
        std::vector<stored_type> expected(size);
        binary_function op;
        
        for(size_t block_index = 0; block_index < grid_size; ++block_index)
        {
            for(int item = 0; item < items_per_block; ++item)
            {
                const size_t i = block_index * items_per_block + item;
                if (item < tile_sizes[block_index]) 
                {
                    if(item == tile_sizes[block_index] - 1 || item == items_per_block - 1) 
                    {
                        expected[i] = static_cast<output_type>(input[i]);
                    } 
                    else 
                    {
                        expected[i] = static_cast<output_type>(op(input[i], input[i + 1]));
                    }
                }
                else
                {
                    expected[i] = static_cast<output_type>(input[i]);
                }
            }
        }

        // Preparing Device
        type* d_input;
        int* d_tile_sizes;
        stored_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(output[0])));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input[0]),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_tile_sizes, tile_sizes.data(),
                tile_sizes.size() * sizeof(tile_sizes[0]),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                subtract_right_partial_tile_kernel<type, output_type, stored_type, 
                                                   binary_function, block_size,
                                                   items_per_thread
                >
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            d_input, d_tile_sizes, d_output
        );
        HIP_CHECK(hipGetLastError());

        // Reading results
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output[0]),
                hipMemcpyDeviceToHost
            )
        );

        using is_add_op = test_utils::is_add_operator<binary_function>;
        // clang-format off
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected,
            is_add_op::value
                ? std::max(test_utils::precision<type>::value, test_utils::precision<stored_type>::value)
                : std::is_same<type, stored_type>::value 
                    ? 0 
                    : test_utils::precision<stored_type>::value));
        // clang-format on

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_tile_sizes));
        HIP_CHECK(hipFree(d_output));
    }
}
