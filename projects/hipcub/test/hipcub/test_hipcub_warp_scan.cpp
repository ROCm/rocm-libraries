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

#include <hipcub/warp/warp_scan.hpp>
#include <type_traits>

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
using HipcubWarpScanTestParams = ::testing::Types<

    // shuffle based scan
    // Integer
    params<int, 2U>,
    params<int, 4U>,
    params<int, 8U>,
    params<int, 16U>,
    params<int, 32U>,
#ifdef __HIP_PLATFORM_AMD__
    params<int, 64U>,
#endif
    // Float
    params<float, 2U>,
    params<float, 4U>,
    params<float, 8U>,
    params<float, 16U>,
    params<float, 32U>,
#ifdef __HIP_PLATFORM_AMD__
    params<float, 64U>,
#endif
    // Half
    params<test_utils::half, 2U>,
    params<test_utils::half, 4U>,
    params<test_utils::half, 8U>,
    params<test_utils::half, 16U>,
    params<test_utils::half, 32U>,
#ifdef __HIP_PLATFORM_AMD__
    params<test_utils::half, 64U>,
#endif
    // Bfloat16
    params<test_utils::bfloat16, 2U>,
    params<test_utils::bfloat16, 4U>,
    params<test_utils::bfloat16, 8U>,
    params<test_utils::bfloat16, 16U>,
    params<test_utils::bfloat16, 32U>,
#ifdef __HIP_PLATFORM_AMD__
    params<test_utils::bfloat16, 64U>,
#endif
    // shared memory scan
    // Integer
    params<int, 3U>,
    params<int, 7U>,
    params<int, 15U>,
#ifdef __HIP_PLATFORM_AMD__
    params<int, 37U>,
    params<int, 61U>,
#endif
    // Float
    params<float, 3U>,
    params<float, 7U>,
    params<float, 15U>
#ifdef __HIP_PLATFORM_AMD__
    ,
    params<float, 37U>,
    params<float, 61U>
#endif
    >;

TYPED_TEST_SUITE(HipcubWarpScanTests, HipcubWarpScanTestParams);

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_kernel(T* device_input, T* device_output)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    auto scan_op = hipcub::Sum();
    wscan_t(storage[warp_id]).InclusiveScan(value, value, scan_op);

    device_output[index] = value;
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_kernel(T* /*device_input*/, T* /*device_output*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

TYPED_TEST(HipcubWarpScanTests, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %u.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(output.size());

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output
            );
        }
        if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output
            );
        }

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
        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_initial_value_kernel(T* device_input, T* device_output, T initial_value)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    auto                                     scan_op = hipcub::Sum();
    wscan_t(storage[warp_id]).InclusiveScan(value, value, initial_value, scan_op);

    device_output[index] = value;
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_initial_value_kernel(T* /*device_input*/,
                                              T* /*device_output*/,
                                              T /*initial_value*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

TYPED_TEST(HipcubWarpScanTests, InclusiveScanInitialValue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive, execution warp size
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size  = 4;
    const size_t size       = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(output.size());
        T              initial_value = test_utils::get_random_data<T>(1, 2, 50, seed_value)[0];
        SCOPED_TRACE(testing::Message() << "with initial_value = " << initial_value);

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(initial_value);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_input,
            input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_output,
            output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(device_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_initial_value_kernel<T,
                                                                         block_size_ws32,
                                                                         logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input,
                device_output,
                initial_value);
        }
        if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_initial_value_kernel<T,
                                                                         block_size_ws64,
                                                                         logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input,
                device_output,
                initial_value);
        }

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(hipMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(T),
                            hipMemcpyDeviceToHost));

        // Validating results
        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_kernel(T* device_input,
                                       T* device_output,
                                       T* device_output_reductions)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
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

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_kernel(T* /*device_input*/,
                                       T* /*device_output*/,
                                       T* /*device_output_reductions*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

TYPED_TEST(HipcubWarpScanTests, InclusiveScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %u.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(output.size());
        std::vector<T> expected_reductions(output_reductions.size());

        // Calculate expected results on host
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            acc_type accumulator(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
            expected_reductions[i] = expected[(i+1) * logical_warp_size - 1];
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
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
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output, device_output_reductions
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output, device_output_reductions
            );
        }

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
        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_initial_value_kernel(T* device_input,
                                                     T* device_output,
                                                     T* device_output_reductions,
                                                     T  initial_value)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * BlockSize);

    T value     = device_input[index];
    T reduction = value;

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    wscan_t(storage[warp_id]).InclusiveScan(value, value, initial_value, hipcub::Sum(), reduction);

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_initial_value_kernel(T* /*device_input*/,
                                                     T* /*device_output*/,
                                                     T* /*device_output_reductions*/,
                                                     T /*initial_value*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

TYPED_TEST(HipcubWarpScanTests, InclusiveScanReduceInitialValue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size  = 4;
    const size_t size       = block_size * grid_size;

    // Check if warp size is supported
    if((logical_warp_size > current_device_warp_size)
       || (current_device_warp_size != ws32
           && current_device_warp_size != ws64)) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: "
               "%u.    Skipping test\n",
               logical_warp_size,
               block_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(output.size());
        std::vector<T> expected_reductions(output_reductions.size());
        T              initial_value = test_utils::get_random_data<T>(1, 2, 50, seed_value)[0];
        SCOPED_TRACE(testing::Message() << "with initial_value = " << initial_value);

        // Calculate expected results on host
        for(size_t i = 0; i < output.size() / logical_warp_size; i++)
        {
            acc_type accumulator(initial_value);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
            expected_reductions[i] = expected[(i + 1) * logical_warp_size - 1];
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_input,
            input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_output,
            output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_output_reductions,
            output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)));

        HIP_CHECK(
            hipMemcpy(device_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

        // Launching kernel
        if(current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_initial_value_kernel<T,
                                                                                block_size_ws32,
                                                                                logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws32),
                0,
                0,
                device_input,
                device_output,
                device_output_reductions,
                initial_value);
        }
        else if(current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_reduce_initial_value_kernel<T,
                                                                                block_size_ws64,
                                                                                logical_warp_size>),
                dim3(grid_size),
                dim3(block_size_ws64),
                0,
                0,
                device_input,
                device_output,
                device_output_reductions,
                initial_value);
        }

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(hipMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(T),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(hipMemcpy(output_reductions.data(),
                            device_output_reductions,
                            output_reductions.size() * sizeof(T),
                            hipMemcpyDeviceToHost));

        // Validating results
        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_kernel(T* device_input, T* device_output, T init)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int warp_id = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int index = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__ typename wscan_t::TempStorage storage[warps_no];
    auto scan_op = hipcub::Sum();
    wscan_t(storage[warp_id]).ExclusiveScan(value, value, init, scan_op);

    device_output[index] = value;
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_kernel(T* /*device_input*/, T* /*device_output*/, T /*init*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

TYPED_TEST(HipcubWarpScanTests, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %u.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(input.size());
        const T        init = static_cast<T>(
            test_utils::get_random_value(0, 100, seed_value + seed_value_addition));

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(init);
            expected[i * logical_warp_size] = init;
            for(size_t j = 1; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx - 1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output, init
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output, init
            );
        }

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
        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_reduce_kernel(T* device_input,
                                       T* device_output,
                                       T* device_output_reductions,
                                       T  init)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
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

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_reduce_kernel(T* /*device_input*/,
                                       T* /*device_output*/,
                                       T* /*device_output_reductions*/,
                                       T /*init*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

TYPED_TEST(HipcubWarpScanTests, ExclusiveReduceScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %u.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output(size);
        std::vector<T> output_reductions(size / logical_warp_size);
        std::vector<T> expected(input.size());
        std::vector<T> expected_reductions(output_reductions.size());
        const T        init = static_cast<T>(
            test_utils::get_random_value(0, 100, seed_value + seed_value_addition));

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator(init);
            expected[i * logical_warp_size] = init;
            for(size_t j = 1; j < logical_warp_size; j++)
            {
                auto idx      = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx - 1], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }

            acc_type accumulator_reductions(0);
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx               = i * logical_warp_size + j;
                accumulator_reductions = binary_op_host(input[idx], accumulator_reductions);
                expected_reductions[i] = static_cast<T>(accumulator_reductions);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        HIP_CHECK(
          test_common_utils::hipMallocHelper(
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
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
              HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T, block_size_ws32, logical_warp_size>),
              dim3(grid_size), dim3(block_size_ws32), 0, 0,
              device_input, device_output, device_output_reductions, init
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
              HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T, block_size_ws64, logical_warp_size>),
              dim3(grid_size), dim3(block_size_ws64), 0, 0,
              device_input, device_output, device_output_reductions, init
            );
        }

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
        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<T>::value * logical_warp_size);
        test_utils::assert_near(output_reductions,
                                expected_reductions,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_scan_kernel(T* device_input,
                      T* device_inclusive_output,
                      T* device_exclusive_output,
                      T  init)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
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

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize>
__global__ __launch_bounds__(BlockSize)
auto warp_scan_kernel(T* /*device_input*/,
                      T* /*device_inclusive_output*/,
                      T* /*device_exclusive_output*/,
                      T /*init*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

TYPED_TEST(HipcubWarpScanTests, Scan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %u.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 50, seed_value);
        std::vector<T> output_inclusive(size);
        std::vector<T> output_exclusive(size);
        std::vector<T> expected_inclusive(output_inclusive.size());
        std::vector<T> expected_exclusive(output_exclusive.size());
        const T        init = static_cast<T>(
            test_utils::get_random_value(0, 100, seed_value + seed_value_addition));

        // Calculate expected results on host
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            acc_type accumulator_inclusive(init);
            acc_type accumulator_exclusive(init);
            expected_exclusive[i * logical_warp_size] = init;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx                = i * logical_warp_size + j;
                accumulator_inclusive   = binary_op_host(input[idx], accumulator_inclusive);
                expected_inclusive[idx] = static_cast<T>(accumulator_inclusive);
                if(j > 0)
                {
                    accumulator_exclusive   = binary_op_host(input[idx - 1], accumulator_exclusive);
                    expected_exclusive[idx] = static_cast<T>(accumulator_exclusive);
                }
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_inclusive_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
                &device_inclusive_output,
                output_inclusive.size() * sizeof(typename decltype(output_inclusive)::value_type)
            )
        );
        T* device_exclusive_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(
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
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_inclusive_output, device_exclusive_output, init
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_inclusive_output, device_exclusive_output, init
            );
        }

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
        test_utils::assert_near(output_inclusive,
                                expected_inclusive,
                                test_utils::precision<T>::value * logical_warp_size);
        test_utils::assert_near(output_exclusive,
                                expected_exclusive,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_inclusive_output));
        HIP_CHECK(hipFree(device_exclusive_output));
    }
}

TYPED_TEST(HipcubWarpScanTests, InclusiveScanCustomType)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using base_type = typename TestFixture::type;
    using T = test_utils::custom_test_type<base_type>;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;
    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    // logical warp side for warp primitive
    constexpr size_t logical_warp_size = TestFixture::warp_size;

    // The different warp sizes
    constexpr size_t ws32 = size_t(HIPCUB_WARP_SIZE_32);
    constexpr size_t ws64 = size_t(HIPCUB_WARP_SIZE_64);

    // Block size of warp size 32
    constexpr size_t block_size_ws32
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws32, logical_warp_size * 4)
              : test_utils::max<size_t>((ws32 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    // Block size of warp size 64
    constexpr size_t block_size_ws64
        = test_utils::is_power_of_two(logical_warp_size)
              ? test_utils::max<size_t>(ws64, logical_warp_size * 4)
              : test_utils::max<size_t>((ws64 / logical_warp_size) * logical_warp_size,
                                        static_cast<size_t>(1));

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    const size_t block_size = current_device_warp_size == ws32 ? block_size_ws32 : block_size_ws64;
    unsigned int grid_size = 4;
    const size_t size = block_size * grid_size;

    // Check if warp size is supported
    if( (logical_warp_size > current_device_warp_size) ||
        (current_device_warp_size != ws32 && current_device_warp_size != ws64) ) // Only WarpSize 32 and 64 is supported
    {
        printf("Unsupported test warp size/computed block size: %zu/%zu. Current device warp size: %u.    Skipping test\n",
            logical_warp_size, block_size, current_device_warp_size);
        GTEST_SKIP();
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);


        // Generate data
        std::vector<T> input(size);
        std::vector<T> output(size);
        std::vector<T> expected(output.size());

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
            acc_type accumulator;
            accumulator.x = 0;
            accumulator.y = 0;
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                auto idx = i * logical_warp_size + j;
                accumulator   = binary_op_host(input[idx], accumulator);
                expected[idx] = static_cast<T>(accumulator);
            }
        }

        // Writing to device memory
        T* device_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        if (current_device_warp_size == ws32)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws32, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws32), 0, 0,
                device_input, device_output
            );
        }
        else if (current_device_warp_size == ws64)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_inclusive_scan_kernel<T, block_size_ws64, logical_warp_size>),
                dim3(grid_size), dim3(block_size_ws64), 0, 0,
                device_input, device_output
            );
        }

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

        test_utils::assert_near(output,
                                expected,
                                test_utils::precision<T>::value * logical_warp_size);

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}
