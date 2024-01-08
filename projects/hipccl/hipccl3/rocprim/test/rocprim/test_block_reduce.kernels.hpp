// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_BLOCK_REDUCE_KERNELS_HPP_
#define TEST_BLOCK_REDUCE_KERNELS_HPP_

template<
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
__launch_bounds__(BlockSize)
void reduce_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    rocprim::block_reduce<T, BlockSize, Algorithm> breduce;
    breduce.reduce(value, value, BinaryOp());
    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = value;
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
        float precision = 0;
        if(!check_equal)
        {
            if(test_utils::is_plus_operator<BinaryOp>::value)
            {
                precision = test_utils::precision<T> / 2 * BlockSize;
            }
            if(test_utils::is_multiply_operator<BinaryOp>::value)
            {
                precision = std::pow(1.0 + test_utils::precision<T> / 2, BlockSize) - 1;
            }
            if(precision > 0.5)
            {
                std::cout << "Test skipped with size " << BlockSize
                          << " due to high relative error " << precision << std::endl;
                return;
            }
        }

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
        HIP_CHECK(hipGetLastError());

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
            test_utils::assert_near(output_reductions, expected_reductions, precision);
        }
    }
};

template<
    unsigned int BlockSize,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
__launch_bounds__(BlockSize)
void reduce_valid_kernel(T* device_output, T* device_output_reductions, const unsigned int valid_items)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    rocprim::block_reduce<T, BlockSize, Algorithm> breduce;
    breduce.reduce(value, value, valid_items, BinaryOp());
    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = value;
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
                    const std::vector<T>& expected_reductions,
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
        HIP_CHECK(hipGetLastError());

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        test_utils::assert_near(output_reductions,
                                expected_reductions,
                                test_utils::precision<T> * valid_items);
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    rocprim::block_reduce_algorithm Algorithm,
    class T,
    class BinaryOp
>
__global__
__launch_bounds__(BlockSize)
void reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    rocprim::block_reduce<T, BlockSize, Algorithm> breduce;
    T reduction;
    breduce.reduce(in_out, reduction, BinaryOp());

    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = reduction;
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
    using binary_op_type = rocprim::maximum<T>;

    static constexpr auto algorithm = Algorithm;
    static constexpr size_t block_size = BlockSize;
    static constexpr size_t items_per_thread = ItemsPerThread;

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
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 0, 100, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, T(0));

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), T(0));
        binary_op_type binary_op;
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            T value = T(0);
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                value = binary_op(value, output[idx]);
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output_reductions, output_reductions.size() * sizeof(T)));

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
        HIP_CHECK(hipGetLastError());

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        test_utils::assert_eq(output_reductions, expected_reductions);

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

#endif // TEST_BLOCK_REDUCE_KERNELS_HPP_
