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

#include "hipcub/warp/warp_scan.hpp"

// Params for tests
template<
    class T,
    unsigned int WarpSize
>
struct params
{
    using type = T;
    static constexpr unsigned int warp_size = WarpSize;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubWarpScanTests : public ::testing::Test {
public:
    using type = typename Params::type;
    static constexpr unsigned int warp_size = Params::warp_size;
};

typedef ::testing::Types<

    // shuffle based scan
    // Integer
    params<int, 2U>,
    params<int, 4U>,
    params<int, 8U>,
    params<int, 16U>,
    params<int, 32U>,
    #ifdef HIPCUB_ROCPRIM_API
        params<int, 64U>,
    #endif
    // Float
    params<float, 2U>,
    params<float, 4U>,
    params<float, 8U>,
    params<float, 16U>,
    params<float, 32U>,
    #ifdef HIPCUB_ROCPRIM_API
        params<float, 64U>,
    #endif

    // shared memory scan
    // Integer
    params<int, 3U>,
    params<int, 7U>,
    params<int, 15U>,
    #ifdef HIPCUB_ROCPRIM_API
        params<int, 37U>,
        params<int, 61U>,
    #endif
    // Float
    params<float, 3U>,
    params<float, 7U>,
    params<float, 15U>
    #ifdef HIPCUB_ROCPRIM_API
        ,params<float, 37U>,
        params<float, 61U>
    #endif

> HipcubWarpScanTestParams;

TYPED_TEST_CASE(HipcubWarpScanTests, HipcubWarpScanTestParams);

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void warp_inclusive_scan_kernel(T* device_input, T* device_output)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    auto scan_op = hipcub::Sum();
    wscan_t(storage[warp_id]).InclusiveScan(value, value, scan_op);

    device_output[index] = value;
}

TYPED_TEST(HipcubWarpScanTests, InclusiveScan)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size
    // is always test_utils::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        test_utils::is_power_of_two(logical_warp_size)
        ? test_utils::max<size_t>(test_utils::warp_size(), logical_warp_size * 4)
        : (test_utils::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > test_utils::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(output.size(), 0);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                expected[idx] = input[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size, logical_warp_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output
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
        if (std::is_integral<T>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]);
            }
        }
        else if (std::is_floating_point<T>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
                ASSERT_NEAR(output[i], expected[i], tolerance);
            }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void warp_inclusive_scan_reduce_kernel(
    T* device_input,
    T* device_output,
    T* device_output_reductions)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + ( hipBlockIdx_x * BlockSize );

    T value = device_input[index];
    T reduction = value;

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    if(hipBlockIdx_x%2 == 0)
    {
        auto scan_op = hipcub::Sum();
        wscan_t(storage[warp_id]).InclusiveScan(value, value, scan_op, reduction);
    }
    else
    {
        wscan_t(storage[warp_id]).InclusiveSum(value, value, reduction);
    }

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

TYPED_TEST(HipcubWarpScanTests, InclusiveScanReduce)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always test_utils::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        test_utils::is_power_of_two(logical_warp_size)
            ? test_utils::max<size_t>(test_utils::warp_size(), logical_warp_size * 4)
            : (test_utils::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > test_utils::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);

        // Calculate expected results on host
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                expected[idx] = input[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_reductions[i] = expected[(i+1) * logical_warp_size - 1];
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
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
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T, block_size, logical_warp_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output, device_output_reductions
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
        if (std::is_integral<T>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]);
            }

            for(size_t i = 0; i < output_reductions.size(); i++)
            {
                ASSERT_EQ(output_reductions[i], expected_reductions[i]);
            }
        }
        else if (std::is_floating_point<T>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
                ASSERT_NEAR(output[i], expected[i], tolerance);
            }

            for(size_t i = 0; i < output_reductions.size(); i++)
            {
                auto tolerance = std::max<T>(std::abs(0.1f * expected_reductions[i]), T(0.01f));
                ASSERT_NEAR(output_reductions[i], expected_reductions[i], tolerance);
            }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void warp_exclusive_scan_kernel(T* device_input, T* device_output, T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    auto scan_op = hipcub::Sum();
    wscan_t(storage[warp_id]).ExclusiveScan(value, value, init, scan_op);

    device_output[index] = value;
}

TYPED_TEST(HipcubWarpScanTests, ExclusiveScan)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always test_utils::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        test_utils::is_power_of_two(logical_warp_size)
        ? test_utils::max<size_t>(test_utils::warp_size(), logical_warp_size * 4)
        : (test_utils::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > test_utils::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(input.size(), 0);
        const T init = test_utils::get_random_value(0, 100, seed_value + seed_value_addition);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            expected[i * logical_warp_size] = init;
            for(size_t j = 1; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                expected[idx] = input[idx-1] + expected[idx-1];
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size, logical_warp_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output, init
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
        if (std::is_integral<T>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]);
            }
        }
        else if (std::is_floating_point<T>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
                ASSERT_NEAR(output[i], expected[i], tolerance);
            }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void warp_exclusive_scan_reduce_kernel(
    T* device_input,
    T* device_output,
    T* device_output_reductions,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];
    T reduction = value;

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    auto scan_op = hipcub::Sum();
    wscan_t(storage[warp_id]).ExclusiveScan(value, value, init, scan_op, reduction);

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

TYPED_TEST(HipcubWarpScanTests, ExclusiveReduceScan)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always test_utils::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        test_utils::is_power_of_two(logical_warp_size)
        ? test_utils::max<size_t>(test_utils::warp_size(), logical_warp_size * 4)
        : (test_utils::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > test_utils::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(input.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        const T init = test_utils::get_random_value(0, 100, seed_value + seed_value_addition);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
          expected[i * logical_warp_size] = init;
          for(size_t j = 1; j < logical_warp_size; j++)
          {
              auto idx = i * logical_warp_size + j;
              expected[idx] = input[idx-1] + expected[idx-1];
            }

            expected_reductions[i] = 0;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
              auto idx = i * logical_warp_size + j;
              expected_reductions[i] += input[idx];
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
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
            device_input, input.data(),
            input.size() * sizeof(T),
            hipMemcpyHostToDevice
          )
        );

        // Launching kernel
        hipLaunchKernelGGL(
          HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T, block_size, logical_warp_size>),
          dim3(grid_size), dim3(block_size), 0, 0,
          device_input, device_output, device_output_reductions, init
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
        if (std::is_integral<T>::value)
        {
          for(size_t i = 0; i < output.size(); i++)
          {
            ASSERT_EQ(output[i], expected[i]);
          }

          for(size_t i = 0; i < output_reductions.size(); i++)
          {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
          }
        }
        else if (std::is_floating_point<T>::value)
        {
          for(size_t i = 0; i < output.size(); i++)
          {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            ASSERT_NEAR(output[i], expected[i], tolerance);
          }

          for(size_t i = 0; i < output_reductions.size(); i++)
          {
            auto tolerance = std::max<T>(std::abs(0.1f * expected_reductions[i]), T(0.01f));
            ASSERT_NEAR(output_reductions[i], expected_reductions[i], tolerance);
          }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void warp_scan_kernel(
    T* device_input,
    T* device_inclusive_output,
    T* device_exclusive_output,
    T init)
{
    constexpr unsigned int warps_no = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T input = device_input[index];
    T inclusive_output, exclusive_output;

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    auto scan_op = hipcub::Sum();
    wscan_t(storage[warp_id]).Scan(input, inclusive_output, exclusive_output, init, scan_op);

    device_inclusive_output[index] = inclusive_output;
    device_exclusive_output[index] = exclusive_output;
}

TYPED_TEST(HipcubWarpScanTests, Scan)
{
    using T = typename TestFixture::type;
    // logical warp side for warp primitive, execution warp size is always test_utils::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        test_utils::is_power_of_two(logical_warp_size)
        ? test_utils::max<size_t>(test_utils::warp_size(), logical_warp_size * 4)
        : (test_utils::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > test_utils::warp_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, -100, 100, seed_value);
        std::vector<T> output_inclusive(size);
        std::vector<T> output_exclusive(size);
        std::vector<T> expected_inclusive(output_inclusive.size(), 0);
        std::vector<T> expected_exclusive(output_exclusive.size(), 0);
        const T init = test_utils::get_random_value(0, 100, seed_value + seed_value_addition);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            expected_exclusive[i * logical_warp_size] = init;
            expected_inclusive[i * logical_warp_size] = init;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                expected_inclusive[idx] = input[idx] + expected_inclusive[j > 0 ? idx-1 : idx];
                if(j > 0)
                {
                    expected_exclusive[idx] = input[idx-1] + expected_exclusive[idx-1];
                }
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_inclusive_output;
        HIP_CHECK(
            hipMalloc(
                &device_inclusive_output,
                output_inclusive.size() * sizeof(typename decltype(output_inclusive)::value_type)
            )
        );
        T* device_exclusive_output;
        HIP_CHECK(
            hipMalloc(
                &device_exclusive_output,
                output_exclusive.size() * sizeof(typename decltype(output_exclusive)::value_type)
            )
        );

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_scan_kernel<T, block_size, logical_warp_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_inclusive_output, device_exclusive_output, init
        );

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output_inclusive.data(), device_inclusive_output,
                output_inclusive.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(
            hipMemcpy(
                output_exclusive.data(), device_exclusive_output,
                output_exclusive.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Validating results
        if (std::is_integral<T>::value)
        {
            for(size_t i = 0; i < output_inclusive.size(); i++)
            {
                ASSERT_EQ(output_inclusive[i], expected_inclusive[i]);
                ASSERT_EQ(output_exclusive[i], expected_exclusive[i]);
            }
        }
        else if (std::is_floating_point<T>::value)
        {
            for(size_t i = 0; i < output_inclusive.size(); i++)
            {
                auto tolerance = std::max<T>(std::abs(0.1f * expected_inclusive[i]), T(0.01f));
                ASSERT_NEAR(output_inclusive[i], expected_inclusive[i], tolerance);

                tolerance = std::max<T>(std::abs(0.1f * expected_exclusive[i]), T(0.01f));
                ASSERT_NEAR(output_exclusive[i], expected_exclusive[i], tolerance);
            }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_inclusive_output));
        HIP_CHECK(hipFree(device_exclusive_output));
    }
}

TYPED_TEST(HipcubWarpScanTests, InclusiveScanCustomType)
{
    using base_type = typename TestFixture::type;
    using T = test_utils::custom_test_type<base_type>;
    // logical warp side for warp primitive, execution warp size is always test_utils::warp_size()
    constexpr size_t logical_warp_size = TestFixture::warp_size;
    constexpr size_t block_size =
        test_utils::is_power_of_two(logical_warp_size)
        ? test_utils::max<size_t>(test_utils::warp_size(), logical_warp_size * 4)
        : (test_utils::warp_size()/logical_warp_size) * logical_warp_size;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Given warp size not supported
    if(logical_warp_size > test_utils::warp_size())
    {
        return;
    }


    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);


        // Generate data
        std::vector<T> input(size);
        std::vector<T> output(size);
        std::vector<T> expected(output.size(), 0);

        // Initializing input data
        {
            auto random_values =
                test_utils::get_random_data<base_type>(2 * input.size(), 0, 100, seed_value);
            for(size_t i = 0; i < input.size(); i++)
            {
                input[i].x = random_values[i];
                input[i].y = random_values[i + input.size()];
            }
        }

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                expected[idx] = input[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size, logical_warp_size>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output
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
        if (std::is_integral<base_type>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]);
            }
        }
        else if (std::is_floating_point<base_type>::value)
        {
            for(size_t i = 0; i < output.size(); i++)
            {
                auto tolerance_x = std::max<base_type>(std::abs(0.1f * expected[i].x), base_type(0.01f));
                auto tolerance_y = std::max<base_type>(std::abs(0.1f * expected[i].y), base_type(0.01f));
                ASSERT_NEAR(output[i].x, expected[i].x, tolerance_x);
                ASSERT_NEAR(output[i].y, expected[i].y, tolerance_y);
            }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}
