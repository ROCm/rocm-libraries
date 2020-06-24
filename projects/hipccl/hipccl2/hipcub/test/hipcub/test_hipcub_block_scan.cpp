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

// hipcub API
#include "hipcub/block/block_scan.hpp"

// Params for tests
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    hipcub::BlockScanAlgorithm Algorithm = hipcub::BLOCK_SCAN_WARP_SCANS
>
struct params
{
    using type = T;
    static constexpr hipcub::BlockScanAlgorithm algorithm = Algorithm;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubBlockScanSingleValueTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr hipcub::BlockScanAlgorithm algorithm = Params::algorithm;
    static constexpr unsigned int block_size = Params::block_size;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // hipcub::BLOCK_SCAN_WARP_SCANS
    // -----------------------------------------------------------------------
    params<int, 64U>,
    params<int, 128U>,
    params<int, 256U>,
    params<int, 512U>,
    params<int, 65U>,
    params<int, 37U>,
    params<int, 162U>,
    params<int, 255U>,
    // uint tests
    params<unsigned int, 64U>,
    params<unsigned int, 256U>,
    params<unsigned int, 377U>,
    // long tests
    params<long, 64U>,
    params<long, 256U>,
    params<long, 377U>,
    // -----------------------------------------------------------------------
    // hipcub::BLOCK_SCAN_RAKING
    // -----------------------------------------------------------------------
    params<int, 64U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<int, 128U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<int, 256U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<int, 512U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<unsigned long, 65U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<long, 37U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<short, 162U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<unsigned int, 255U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<int, 377U, 1, hipcub::BLOCK_SCAN_RAKING>,
    params<unsigned char, 377U, 1, hipcub::BLOCK_SCAN_RAKING>
> SingleValueTestParams;

TYPED_TEST_CASE(HipcubBlockScanSingleValueTests, SingleValueTestParams);

template<
    unsigned int BlockSize,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void inclusive_scan_kernel(T* device_output)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(value, value, hipcub::Sum());

    device_output[index] = value;
}

TYPED_TEST(HipcubBlockScanSingleValueTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inclusive_scan_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void inclusive_scan_reduce_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    T reduction;
    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(value, value, hipcub::Sum(), reduction);
    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_reductions[i] = expected[(i+1) * block_size - 1];
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            hipMalloc(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inclusive_scan_reduce_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<
    unsigned int BlockSize,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void inclusive_scan_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(value, value, hipcub::Sum(), prefix_callback);

    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_bp[hipBlockIdx_x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, InclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / block_size);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            expected[i * block_size] = block_prefix;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_block_prefixes[i] = expected[(i+1) * block_size - 1];
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        HIP_CHECK(
            hipMalloc(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inclusive_scan_prefix_callback_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_bp, block_prefix
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_EQ(output_block_prefixes[i], expected_block_prefixes[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_bp));
    }
}

template<
    unsigned int BlockSize,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_kernel(T* device_output, T init)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(value, value, init, hipcub::Sum());
    device_output[index] = value;
}

TYPED_TEST(HipcubBlockScanSingleValueTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 241, seed_value);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
          expected[i * block_size] = init;
          for(size_t j = 1; j < block_size; j++)
          {
              auto idx = i * block_size + j;
              expected[idx] = output[idx-1] + expected[idx-1];
          }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
          hipMemcpy(
              device_output, output.data(),
              output.size() * sizeof(T),
              hipMemcpyHostToDevice
          )
        );

        // Launching kernel
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(exclusive_scan_kernel<block_size, algorithm, T>),
          dim3(grid_size), dim3(block_size), 0, 0,
          device_output, init
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
          hipMemcpy(
              output.data(), device_output,
              output.size() * sizeof(T),
              hipMemcpyDeviceToHost
          )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
          ASSERT_EQ(output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_reduce_kernel(T* device_output, T* device_output_reductions, T init)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T value = device_output[index];
    T reduction;
    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(value, value, init, hipcub::Sum(), reduction);
    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, ExclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            expected[i * block_size] = init;
            for(size_t j = 1; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }

            expected_reductions[i] = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected_reductions[i] += output[idx];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            hipMalloc(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(exclusive_scan_reduce_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions, init
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<
    unsigned int BlockSize,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(value, value, hipcub::Sum(), prefix_callback);

    device_output[index] = value;
    if(hipThreadIdx_x == 0)
    {
        device_output_bp[hipBlockIdx_x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, ExclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / block_size);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            expected[i * block_size] = block_prefix;
            for(size_t j = 1; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }

            expected_block_prefixes[i] = block_prefix;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected_block_prefixes[i] += output[idx];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        HIP_CHECK(
            hipMalloc(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(exclusive_scan_prefix_callback_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_bp, block_prefix
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_EQ(output_block_prefixes[i], expected_block_prefixes[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_bp));
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, CustomStruct)
{
    using base_type = typename TestFixture::type;
    using T = test_utils::custom_test_type<base_type>;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output(size);
        {
            std::vector<base_type> random_values =
                test_utils::get_random_data<base_type>(2 * output.size(), 2, 200, seed_value);
            for(size_t i = 0; i < output.size(); i++)
            {
                output[i].x = random_values[i],
                output[i].y = random_values[i + output.size()];
            }
        }

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inclusive_scan_kernel<block_size, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_output));
    }
}

// // ---------------------------------------------------------
// // Test for scan ops taking array of values as input
// // ---------------------------------------------------------

template<class Params>
class HipcubBlockScanInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr hipcub::BlockScanAlgorithm algorithm = Params::algorithm;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // hipcub::BlockScanAlgorithm::using_warp_scan
    // -----------------------------------------------------------------------
    params<float, 6U,   32>,
    params<float, 32,   2>,
    params<unsigned int, 256,  3>,
    params<int, 512,  4>,
    params<float, 37,   2>,
    params<float, 65,   5>,
    params<float, 162,  7>,
    params<float, 255,  15>,
    // -----------------------------------------------------------------------
    // hipcub::BLOCK_SCAN_RAKING
    // -----------------------------------------------------------------------
    params<float, 6U,   32, hipcub::BLOCK_SCAN_RAKING>,
    params<float, 32,   2,  hipcub::BLOCK_SCAN_RAKING>,
    params<int, 256,  3,  hipcub::BLOCK_SCAN_RAKING>,
    params<unsigned int, 512,  4,  hipcub::BLOCK_SCAN_RAKING>,
    params<float, 37,   2,  hipcub::BLOCK_SCAN_RAKING>,
    params<float, 65,   5,  hipcub::BLOCK_SCAN_RAKING>,
    params<float, 162,  7,  hipcub::BLOCK_SCAN_RAKING>,
    params<float, 255,  15, hipcub::BLOCK_SCAN_RAKING>
> InputArrayTestParams;

TYPED_TEST_CASE(HipcubBlockScanInputArrayTests, InputArrayTestParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void inclusive_scan_array_kernel(T* device_output)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize ) + hipThreadIdx_x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(in_out, in_out, hipcub::Sum());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

}

TYPED_TEST(HipcubBlockScanInputArrayTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inclusive_scan_array_kernel<block_size, items_per_thread, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        HIP_CHECK(hipFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void inclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize ) + hipThreadIdx_x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    T reduction;
    bscan_t(temp_storage).InclusiveScan(in_out, in_out, hipcub::Sum(), reduction);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_reductions[i] = expected[(i+1) * items_per_block - 1];
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            hipMalloc(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inclusive_scan_reduce_array_kernel<block_size, items_per_thread, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_NEAR(
                output_reductions[i], expected_reductions[i],
                static_cast<T>(0.05) * expected_reductions[i]
            );
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void inclusive_scan_array_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(in_out, in_out, hipcub::Sum(), prefix_callback);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_bp[hipBlockIdx_x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, InclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / items_per_block, 0);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = block_prefix;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_block_prefixes[i] = expected[(i+1) * items_per_block - 1];
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        HIP_CHECK(
            hipMalloc(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output_bp, output_block_prefixes.data(),
                output_block_prefixes.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                inclusive_scan_array_prefix_callback_kernel<block_size, items_per_thread, algorithm, T>
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_bp, block_prefix
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_NEAR(
                output_block_prefixes[i], expected_block_prefixes[i],
                static_cast<T>(0.05) * expected_block_prefixes[i]
            );
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_bp));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_array_kernel(T* device_output, T init)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(in_out, in_out, init, hipcub::Sum());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = init;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(exclusive_scan_array_kernel<block_size, items_per_thread, algorithm, T>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, init
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        HIP_CHECK(hipFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions, T init)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    T reduction;
    bscan_t(temp_storage).ExclusiveScan(in_out, in_out, init, hipcub::Sum(), reduction);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, ExclusiveScanReduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = init;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }
            for(size_t j = 0; j < items_per_block; j++)
            {
                expected_reductions[i] += output[i * items_per_block + j];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            hipMalloc(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        HIP_CHECK(
            hipMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                exclusive_scan_reduce_array_kernel<block_size, items_per_thread, algorithm, T>
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_reductions, init
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_NEAR(
                output_reductions[i], expected_reductions[i],
                static_cast<T>(0.05) * expected_reductions[i]
            );
        }
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    hipcub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_prefix_callback_array_kernel(
    T* device_output,
    T* device_output_bp,
    T block_prefix
)
{
    const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index+ j];
    }

    using bscan_t = hipcub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(in_out, in_out, hipcub::Sum(), prefix_callback);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(hipThreadIdx_x == 0)
    {
        device_output_bp[hipBlockIdx_x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, ExclusiveScanPrefixCallback)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / items_per_block);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = block_prefix;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }
            expected_block_prefixes[i] = block_prefix;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected_block_prefixes[i] += output[idx];
            }
        }

        // Writing to device memory
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        HIP_CHECK(
            hipMalloc(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                exclusive_scan_prefix_callback_array_kernel<block_size, items_per_thread, algorithm, T>
            ),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_output, device_output_bp, block_prefix
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_NEAR(
                output_block_prefixes[i], expected_block_prefixes[i],
                static_cast<T>(0.05) * expected_block_prefixes[i]
            );
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_bp));
    }
}
