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
#include <rocprim/config.hpp>
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>

// required test headers
#include "test_utils_types.hpp"

template<class Params>
class RocprimBlockExchangeTests : public ::testing::Test {
public:
    using params = Params;
};

TYPED_TEST_CASE(RocprimBlockExchangeTests, BlockParams);

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void blocked_to_striped_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_exchange<Type, block_size, ItemsPerThread> exchange;
    exchange.blocked_to_striped(input, output);

    rocprim::block_store_direct_blocked(lid, device_output + block_offset, output);
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void striped_to_blocked_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_exchange<Type, block_size, ItemsPerThread> exchange;
    exchange.striped_to_blocked(input, output);

    rocprim::block_store_direct_blocked(lid, device_output + block_offset, output);
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void blocked_to_warp_striped_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_exchange<Type, block_size, ItemsPerThread> exchange;
    exchange.blocked_to_warp_striped(input, output);

    rocprim::block_store_direct_blocked(lid, device_output + block_offset, output);
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void warp_striped_to_blocked_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);

    rocprim::block_exchange<Type, block_size, ItemsPerThread> exchange;
    exchange.warp_striped_to_blocked(input, output);

    rocprim::block_store_direct_blocked(lid, device_output + block_offset, output);
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void scatter_to_blocked_kernel(Type* device_input, OutputType* device_output, unsigned int* device_ranks)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    unsigned int ranks[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);
    rocprim::block_load_direct_blocked(lid, device_ranks + block_offset, ranks);

    rocprim::block_exchange<Type, block_size, ItemsPerThread> exchange;
    exchange.scatter_to_blocked(input, output, ranks);

    rocprim::block_store_direct_blocked(lid, device_output + block_offset, output);
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void scatter_to_striped_kernel(Type* device_input, OutputType* device_output, unsigned int* device_ranks)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    unsigned int ranks[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_input + block_offset, input);
    rocprim::block_load_direct_blocked(lid, device_ranks + block_offset, ranks);

    rocprim::block_exchange<Type, block_size, ItemsPerThread> exchange;
    exchange.scatter_to_striped(input, output, ranks);

    rocprim::block_store_direct_blocked(lid, device_output + block_offset, output);
}

// Test for exchange
template<
    class T,
    class U,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U
>
auto test_block_exchange()
-> typename std::enable_if<Method == 0>::type
{
    using type = T;
    using output_type = U;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, 0);

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset + ii * block_size + ti;
                input[i1] = values[i1];
                expected[i0] = values[i1];
            }
        }
    }

    // Preparing device
    type* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(blocked_to_striped_kernel<type, output_type, items_per_block, items_per_thread>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Reading results
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    class U,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U
>
auto test_block_exchange()
-> typename std::enable_if<Method == 1>::type
{
    using type = T;
    using output_type = U;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset + ii * block_size + ti;
                input[i0] = values[i1];
                expected[i1] = values[i1];
            }
        }
    }

    // Preparing device
    type* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(striped_to_blocked_kernel<type, output_type, items_per_block, items_per_thread>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Reading results
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    class U,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U
>
auto test_block_exchange()
-> typename std::enable_if<Method == 2>::type
{
    using type = T;
    using output_type = U;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));

    constexpr size_t warp_size =
        ::rocprim::detail::get_min_warp_size(block_size, size_t(::rocprim::warp_size()));
    constexpr size_t warps_no = (block_size + warp_size - 1) / warp_size;
    constexpr size_t items_per_warp = warp_size * items_per_thread;

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t wi = 0; wi < warps_no; wi++)
        {
            const size_t current_warp_size = wi == warps_no - 1
                ? (block_size % warp_size != 0 ? block_size % warp_size : warp_size)
                : warp_size;
            for(size_t li = 0; li < current_warp_size; li++)
            {
                for(size_t ii = 0; ii < items_per_thread; ii++)
                {
                    const size_t offset = bi * items_per_block + wi * items_per_warp;
                    const size_t i0 = offset + li * items_per_thread + ii;
                    const size_t i1 = offset + ii * current_warp_size + li;
                    input[i1] = values[i1];
                    expected[i0] = values[i1];
                }
            }
        }
    }

    // Preparing device
    type* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(blocked_to_warp_striped_kernel<
                type, output_type, items_per_block, items_per_thread
        >),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Reading results
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    class U,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U
>
auto test_block_exchange()
-> typename std::enable_if<Method == 3>::type
{
    using type = T;
    using output_type = U;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));

    constexpr size_t warp_size =
        ::rocprim::detail::get_min_warp_size(block_size, size_t(::rocprim::warp_size()));
    constexpr size_t warps_no = (block_size + warp_size - 1) / warp_size;
    constexpr size_t items_per_warp = warp_size * items_per_thread;

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t wi = 0; wi < warps_no; wi++)
        {
            const size_t current_warp_size = wi == warps_no - 1
                ? (block_size % warp_size != 0 ? block_size % warp_size : warp_size)
                : warp_size;
            for(size_t li = 0; li < current_warp_size; li++)
            {
                for(size_t ii = 0; ii < items_per_thread; ii++)
                {
                    const size_t offset = bi * items_per_block + wi * items_per_warp;
                    const size_t i0 = offset + li * items_per_thread + ii;
                    const size_t i1 = offset + ii * current_warp_size + li;
                    input[i0] = values[i1];
                    expected[i1] = values[i1];
                }
            }
        }
    }

    // Preparing device
    type* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_striped_to_blocked_kernel<type, output_type, items_per_block, items_per_thread>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Reading results
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<
    class T,
    class U,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U
>
auto test_block_exchange()
-> typename std::enable_if<Method == 4>::type
{
    using type = T;
    using output_type = U;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));
    std::vector<unsigned int> ranks(size);

    // Calculate input and expected results on host
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        auto block_ranks = ranks.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, std::mt19937{std::random_device{}()});
    }
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset + ranks[i0];
                input[i0] = values[i0];
                expected[i1] = values[i0];
            }
        }
    }

    // Preparing device
    type* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    unsigned int* device_ranks;
    HIP_CHECK(hipMalloc(&device_ranks, ranks.size() * sizeof(typename decltype(ranks)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_ranks, ranks.data(),
            ranks.size() * sizeof(unsigned int),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scatter_to_blocked_kernel<type, output_type, items_per_block, items_per_thread>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output, device_ranks
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Reading results
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_ranks));
}

template<
    class T,
    class U,
    int Method,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U
>
auto test_block_exchange()
-> typename std::enable_if<Method == 5>::type
{
    using type = T;
    using output_type = U;
    constexpr size_t block_size = BlockSize;
    constexpr size_t items_per_thread = ItemsPerThread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));
    std::vector<unsigned int> ranks(size);

    // Calculate input and expected results on host
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        auto block_ranks = ranks.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, std::mt19937{std::random_device{}()});
    }
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset
                    + ranks[i0] % block_size * items_per_thread
                    + ranks[i0] / block_size;
                input[i0] = values[i0];
                expected[i1] = values[i0];
            }
        }
    }

    // Preparing device
    type* device_input;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    unsigned int* device_ranks;
    HIP_CHECK(hipMalloc(&device_ranks, ranks.size() * sizeof(typename decltype(ranks)::value_type)));

    HIP_CHECK(
        hipMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            hipMemcpyHostToDevice
        )
    );

    HIP_CHECK(
        hipMemcpy(
            device_ranks, ranks.data(),
            ranks.size() * sizeof(unsigned int),
            hipMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scatter_to_striped_kernel<type, output_type, items_per_block, items_per_thread>),
        dim3(grid_size), dim3(block_size), 0, 0,
        device_input, device_output, device_ranks
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Reading results
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_ranks));
}

// Static for-loop
template <
    unsigned int First,
    unsigned int Last,
    class T,
    class U,
    int Method,
    unsigned int BlockSize = 256U
>
struct static_for
{
    static void run()
    {
        int device_id = test_common_utils::obtain_device_from_ctest();
        SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
        HIP_CHECK(hipSetDevice(device_id));

        test_block_exchange<T, U, Method, BlockSize, items[First]>();
        static_for<First + 1, Last, T, U, Method, BlockSize>::run();
    }
};

template <
    unsigned int N,
    class T,
    class U,
    int Method,
    unsigned int BlockSize
>
struct static_for<N, N, T, U, Method, BlockSize>
{
    static void run()
    {
    }
};

TYPED_TEST(RocprimBlockExchangeTests, BlockedToStriped)
{
    using type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 4, type, output_type, 0, block_size>::run();
}

TYPED_TEST(RocprimBlockExchangeTests, StripedToBlocked)
{
    using type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 4, type, output_type, 1, block_size>::run();
}

TYPED_TEST(RocprimBlockExchangeTests, BlockedToWarpStriped)
{
    using type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 4, type, output_type, 2, block_size>::run();
}

TYPED_TEST(RocprimBlockExchangeTests, WarpStripedToBlocked)
{
    using type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 4, type, output_type, 3, block_size>::run();
}

TYPED_TEST(RocprimBlockExchangeTests, ScatterToBlocked)
{
    using type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 4, type, output_type, 4, block_size>::run();
}

TYPED_TEST(RocprimBlockExchangeTests, ScatterToStriped)
{
    using type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;

    static_for<0, 4, type, output_type, 5, block_size>::run();
}
