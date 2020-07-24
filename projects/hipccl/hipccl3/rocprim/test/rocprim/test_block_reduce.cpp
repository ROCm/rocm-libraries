// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/block/block_reduce.hpp>

// required test headers
#include "test_utils_types.hpp"

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------
template<class Params>
class RocprimBlockReduceSingleValueTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    static constexpr unsigned int block_size = Params::block_size;
};

TYPED_TEST_CASE(RocprimBlockReduceSingleValueTests, BlockParams);

template<
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void reduce_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    rocprim::block_reduce<T, BlockSize, Algorithm> breduce;
    breduce.reduce(value, value, BinaryOp());
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

template <
    class T,
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class BinaryOp
>
struct static_run_algo
{
    static void run(std::vector<T>& output,
                    std::vector<T>& output_reductions,
                    std::vector<T>& expected_reductions,
                    T* device_output,
                    T* device_output_reductions,
                    size_t grid_size,
                    bool check_equal)
    {
        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_kernel<BlockSize, Algorithm, T, BinaryOp>),
            dim3(grid_size), dim3(BlockSize), 0, 0,
            device_output, device_output_reductions
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        if(check_equal)
        {
            test_utils::assert_eq(output_reductions, expected_reductions);
        }
        else
        {
            test_utils::assert_near(output_reductions, expected_reductions, test_utils::precision_threshold<T>::percentage);
        }
    }
};

TYPED_TEST(RocprimBlockReduceSingleValueTests, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value = apply(binary_op, value, output[idx]);
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, false
        );
        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::raking_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, false
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

TYPED_TEST(RocprimBlockReduceSingleValueTests, ReduceMultiplies)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_multiplies, rocprim::multiplies<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 58;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output(size, 1);
        auto two_places = test_utils::get_random_data<unsigned int>(size/32, 0, size-1, seed_value);
        for(auto i : two_places)
        {
            output[i] = T(2);
        }
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 1;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value = apply(binary_op, value, output[idx]);
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, true
        );
        static_run_algo<T, block_size, rocprim::block_reduce_algorithm::raking_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, grid_size, true
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

template<
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void reduce_valid_kernel(T* device_output, T* device_output_reductions, const unsigned int valid_items)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    rocprim::block_reduce<T, BlockSize, Algorithm> breduce;
    breduce.reduce(value, value, valid_items, BinaryOp());
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

template <
    class T,
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class BinaryOp
>
struct static_run_valid
{
    static void run(std::vector<T>& output,
                    std::vector<T>& output_reductions,
                    std::vector<T>& expected_reductions,
                    T* device_output,
                    T* device_output_reductions,
                    const unsigned int valid_items,
                    size_t grid_size)
    {
        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_valid_kernel<BlockSize, Algorithm, T, BinaryOp>),
            dim3(grid_size), dim3(BlockSize), 0, 0,
            device_output, device_output_reductions, valid_items
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        test_utils::assert_near(output_reductions, expected_reductions, test_utils::precision_threshold<T>::percentage);
    }
};

TYPED_TEST(RocprimBlockReduceSingleValueTests, ReduceValid)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_plus, rocprim::plus<T>>::type;
    constexpr size_t block_size = TestFixture::block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const unsigned int valid_items = test_utils::get_random_value(block_size - 10, block_size, seed_value);

        // Given block size not supported
        if(block_size > test_utils::get_max_block_size())
        {
            return;
        }

        const size_t size = block_size * 58;
        const size_t grid_size = size / block_size;
        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < valid_items; j++)
            {
                auto idx = i * block_size + j;
                value = apply(binary_op, value, output[idx]);
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        static_run_valid<T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, valid_items, grid_size
        );
        static_run_valid<T, block_size, rocprim::block_reduce_algorithm::raking_reduce, binary_op_type>::run(
            output, output_reductions, expected_reductions,
            device_output, device_output_reductions, valid_items, grid_size
        );

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}


template<class Params>
class RocprimBlockReduceInputArrayTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    static constexpr unsigned int block_size = Params::block_size;
};

TYPED_TEST_CASE(RocprimBlockReduceInputArrayTests, BlockParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_reduce<T, BlockSize, Algorithm> breduce;
    T reduction;
    breduce.reduce(in_out, reduction, BinaryOp());

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

// Test for reduce
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_reduce_algorithm Algorithm = rocprim::block_reduce_algorithm::using_warp_reduce
>
void test_block_reduce_input_arrays()
{
    using binary_op_type = typename std::conditional<std::is_same<T, rocprim::half>::value, test_utils::half_maximum, rocprim::maximum<T>>::type;
    constexpr auto algorithm = Algorithm;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 0, 100, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            T value = 0;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                value = apply(binary_op, value, output[idx]);
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output_reductions, output_reductions.data(),
                output_reductions.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(reduce_array_kernel<block_size, items_per_thread, algorithm, T, binary_op_type>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        test_utils::assert_near(output_reductions, expected_reductions, 0.05);

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }

}

// Static for-loop
template <
    unsigned int First,
    unsigned int Last,
    class T,
    unsigned int BlockSize = 256U,
    rocprim::block_reduce_algorithm Algorithm = rocprim::block_reduce_algorithm::using_warp_reduce
>
struct static_for_input_array
{
    static void run()
    {
        test_block_reduce_input_arrays<T, BlockSize, items[First], Algorithm>();
        static_for_input_array<First + 1, Last, T, BlockSize, Algorithm>::run();
    }
};

template <
    unsigned int N,
    class T,
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm
>
struct static_for_input_array<N, N, T, BlockSize, Algorithm>
{
    static void run()
    {
    }
};

TYPED_TEST(RocprimBlockReduceInputArrayTests, Reduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    constexpr size_t block_size = TestFixture::block_size;

    static_for_input_array<0, 2, T, block_size, rocprim::block_reduce_algorithm::using_warp_reduce>::run();
    static_for_input_array<0, 2, T, block_size, rocprim::block_reduce_algorithm::raking_reduce>::run();
}
