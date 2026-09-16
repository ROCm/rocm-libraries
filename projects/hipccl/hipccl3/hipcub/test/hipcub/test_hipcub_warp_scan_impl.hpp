// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_HIPCUB_WARP_SCAN_IMPL_HPP_
#define TEST_HIPCUB_WARP_SCAN_IMPL_HPP_

#include <hipcub/warp/warp_scan.hpp>
#include <type_traits>

template<bool IsInclusive, unsigned int LogicalWarpSize, class T, class ScanOpT>
std::vector<T>
    warp_scan_host(std::vector<T>& output, ScanOpT scan_op, int valid_items, T initial_value = T{})
{
    assert(output.size() % LogicalWarpSize == 0ul);

    using acc_type = typename test_utils::select_plus_operator_host<T>::acc_type;

    const unsigned int warps_no = static_cast<unsigned int>(output.size()) / LogicalWarpSize;
    std::vector<T>     reductions(warps_no);
    for(unsigned int w = 0; w < warps_no; ++w)
    {
        T*       warp_output = output.data() + w * LogicalWarpSize;
        acc_type accumulator = acc_type{initial_value};
        acc_type reduction   = warp_output[0];
        for(int i = 0; i < std::clamp(valid_items, 0, static_cast<int>(LogicalWarpSize)); i++)
        {
            const auto input_val = warp_output[i];
            if(i > 0)
            {
                reduction = scan_op(input_val, reduction);
            }
            if constexpr(IsInclusive)
            {
                accumulator    = scan_op(input_val, accumulator);
                warp_output[i] = static_cast<T>(accumulator);
            }
            else
            {
                warp_output[i] = static_cast<T>(accumulator);
                accumulator    = scan_op(input_val, accumulator);
            }
        }
        reductions[w] = static_cast<T>(reduction);
    }

    return reductions;
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_kernel(T* device_input, T* device_output, int valid_items)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__
    typename wscan_t::TempStorage storage[warps_no];

    auto scan_op = test_utils::plus{};
    if constexpr(Partial)
    {
        wscan_t(storage[warp_id]).InclusiveScanPartial(value, value, scan_op, valid_items);
    }
    else
    {
        (void)valid_items;
        wscan_t(storage[warp_id]).InclusiveScan(value, value, scan_op);
    }

    device_output[index] = value;
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_kernel(T* /*device_input*/, T* /*device_output*/, int /*valid_items*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<typename TestFixture, bool Partial>
void inclusive_scan_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;

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

        // Note: valid_items is only actually used when Partial is true
        const std::vector<int> valid_items_values
            = Partial ? test_utils::get_warp_valid_items<logical_warp_size>(seed_value)
                      : std::vector<int>{logical_warp_size};

        for(int valid_items : valid_items_values)
        {
            SCOPED_TRACE(testing::Message() << "with valid_items = " << valid_items);

            // Calculate expected results on host
            std::vector<T> expected(input);
            warp_scan_host<true /*IsInclusive*/, logical_warp_size>(expected,
                                                                    binary_op_host,
                                                                    valid_items);

            // Launching kernel
            if(current_device_warp_size == ws32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        warp_inclusive_scan_kernel<T, block_size_ws32, logical_warp_size, Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws32),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items);
                HIP_CHECK(hipGetLastError());
            }
            if(current_device_warp_size == ws64)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        warp_inclusive_scan_kernel<T, block_size_ws64, logical_warp_size, Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws64),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items);
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
        }
        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_initial_value_kernel(T*  device_input,
                                              T*  device_output,
                                              int valid_items,
                                              T   initial_value)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__
    typename wscan_t::TempStorage storage[warps_no];
    auto                          scan_op = test_utils::plus{};

    if constexpr(Partial)
    {
        wscan_t(storage[warp_id])
            .InclusiveScanPartial(value, value, initial_value, scan_op, valid_items);
    }
    else
    {
        (void)valid_items;
        wscan_t(storage[warp_id]).InclusiveScan(value, value, initial_value, scan_op);
    }

    device_output[index] = value;
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_initial_value_kernel(T* /*device_input*/,
                                              T* /*device_output*/,
                                              int /*valid_items*/,
                                              T /*initial_value*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<typename TestFixture, bool Partial>
void inclusive_scan_init_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;

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
        T              initial_value = test_utils::get_random_data<T>(1, 2, 50, seed_value)[0];
        SCOPED_TRACE(testing::Message() << "with initial_value = " << initial_value);

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

        // Note: valid_items is only actually used when Partial is true
        const std::vector<int> valid_items_values
            = Partial ? test_utils::get_warp_valid_items<logical_warp_size>(seed_value)
                      : std::vector<int>{logical_warp_size};

        for(int valid_items : valid_items_values)
        {
            SCOPED_TRACE(testing::Message() << "with valid_items = " << valid_items);

            // Calculate expected results on host
            std::vector<T> expected(input);
            warp_scan_host<true /*IsInclusive*/, logical_warp_size>(expected,
                                                                    binary_op_host,
                                                                    valid_items,
                                                                    initial_value);

            // Launching kernel
            if(current_device_warp_size == ws32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_inclusive_scan_initial_value_kernel<T,
                                                                             block_size_ws32,
                                                                             logical_warp_size,
                                                                             Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws32),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    initial_value);
                HIP_CHECK(hipGetLastError());
            }
            if(current_device_warp_size == ws64)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_inclusive_scan_initial_value_kernel<T,
                                                                             block_size_ws64,
                                                                             logical_warp_size,
                                                                             Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws64),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
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
        }
        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_kernel(T*  device_input,
                                       T*  device_output,
                                       int valid_items,
                                       T*  device_output_reductions)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * BlockSize);

    T value     = device_input[index];
    T reduction = value;

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__
    typename wscan_t::TempStorage storage[warps_no];
    auto                          scan_op = test_utils::plus{};

    if constexpr(Partial)
    {
        wscan_t(storage[warp_id])
            .InclusiveScanPartial(value, value, scan_op, valid_items, reduction);
    }
    else
    {
        (void)valid_items;
        if(hipBlockIdx_x % 2 == 0)
        {
            wscan_t(storage[warp_id]).InclusiveScan(value, value, scan_op, reduction);
        }
        else
        {
            wscan_t(storage[warp_id]).InclusiveSum(value, value, reduction);
        }
    }

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_kernel(T* /*device_input*/,
                                       T* /*device_output*/,
                                       int /*valid_items*/,
                                       T* /*device_output_reductions*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<typename TestFixture, bool Partial>
void inclusive_scan_reduce_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;

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

        // Note: valid_items is only actually used when Partial is true
        const std::vector<int> valid_items_values
            = Partial ? test_utils::get_warp_valid_items<logical_warp_size>(seed_value)
                      : std::vector<int>{logical_warp_size};

        for(int valid_items : valid_items_values)
        {
            SCOPED_TRACE(testing::Message() << "with valid_items = " << valid_items);

            // Calculate expected results on host
            std::vector<T> expected(input);
            std::vector<T> expected_reductions(output_reductions.size());
            expected_reductions
                = warp_scan_host<true /*IsInclusive*/, logical_warp_size>(expected,
                                                                          binary_op_host,
                                                                          valid_items);

            // Launching kernel
            if(current_device_warp_size == ws32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T,
                                                                      block_size_ws32,
                                                                      logical_warp_size,
                                                                      Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws32),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    device_output_reductions);
                HIP_CHECK(hipGetLastError());
            }
            else if(current_device_warp_size == ws64)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_inclusive_scan_reduce_kernel<T,
                                                                      block_size_ws64,
                                                                      logical_warp_size,
                                                                      Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws64),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    device_output_reductions);
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
        }
        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_initial_value_kernel(T*  device_input,
                                                     T*  device_output,
                                                     int valid_items,
                                                     T*  device_output_reductions,
                                                     T   initial_value)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * BlockSize);

    T value     = device_input[index];
    T reduction = value;

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__
    typename wscan_t::TempStorage storage[warps_no];
    auto                          scan_op = test_utils::plus{};

    if constexpr(Partial)
    {
        wscan_t(storage[warp_id])
            .InclusiveScanPartial(value, value, initial_value, scan_op, valid_items, reduction);
    }
    else
    {
        (void)valid_items;
        wscan_t(storage[warp_id]).InclusiveScan(value, value, initial_value, scan_op, reduction);
    }

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_inclusive_scan_reduce_initial_value_kernel(T* /*device_input*/,
                                                     T* /*device_output*/,
                                                     int /*valid_items*/,
                                                     T* /*device_output_reductions*/,
                                                     T /*initial_value*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<typename TestFixture, bool Partial>
void inclusive_scan_reduce_init_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;

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
        T              initial_value = test_utils::get_random_data<T>(1, 2, 50, seed_value)[0];
        SCOPED_TRACE(testing::Message() << "with initial_value = " << initial_value);

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

        // Note: valid_items is only actually used when Partial is true
        const std::vector<int> valid_items_values
            = Partial ? test_utils::get_warp_valid_items<logical_warp_size>(seed_value)
                      : std::vector<int>{logical_warp_size};

        for(int valid_items : valid_items_values)
        {
            SCOPED_TRACE(testing::Message() << "with valid_items = " << valid_items);

            // Calculate expected results on host
            std::vector<T> expected(input);
            std::vector<T> expected_reductions(output_reductions.size());
            expected_reductions
                = warp_scan_host<true /*IsInclusive*/, logical_warp_size>(expected,
                                                                          binary_op_host,
                                                                          valid_items,
                                                                          initial_value);

            // Launching kernel
            if(current_device_warp_size == ws32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        warp_inclusive_scan_reduce_initial_value_kernel<T,
                                                                        block_size_ws32,
                                                                        logical_warp_size,
                                                                        Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws32),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    device_output_reductions,
                    initial_value);
                HIP_CHECK(hipGetLastError());
            }
            else if(current_device_warp_size == ws64)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        warp_inclusive_scan_reduce_initial_value_kernel<T,
                                                                        block_size_ws64,
                                                                        logical_warp_size,
                                                                        Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws64),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
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
        }
        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_kernel(T* device_input, T* device_output, int valid_items, T init)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value = device_input[index];

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__
    typename wscan_t::TempStorage storage[warps_no];

    auto scan_op = test_utils::plus{};
    if constexpr(Partial)
    {
        wscan_t(storage[warp_id]).ExclusiveScanPartial(value, value, init, scan_op, valid_items);
    }
    else
    {
        (void)valid_items;
        wscan_t(storage[warp_id]).ExclusiveScan(value, value, init, scan_op);
    }

    device_output[index] = value;
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_kernel(T* /*device_input*/,
                                T* /*device_output*/,
                                int /*valid_items*/,
                                T /*init*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<typename TestFixture, bool Partial>
void exclusive_scan_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;

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
        const T        init = static_cast<T>(
            test_utils::get_random_value(0, 100, seed_value + seed_value_addition));
        SCOPED_TRACE(testing::Message() << "with init = " << init);

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

        // Note: valid_items is only actually used when Partial is true
        const std::vector<int> valid_items_values
            = Partial ? test_utils::get_warp_valid_items<logical_warp_size>(seed_value)
                      : std::vector<int>{logical_warp_size};

        for(int valid_items : valid_items_values)
        {
            SCOPED_TRACE(testing::Message() << "with valid_items = " << valid_items);

            // Calculate expected results on host
            std::vector<T> expected(input);
            warp_scan_host<false /*IsInclusive*/, logical_warp_size>(expected,
                                                                     binary_op_host,
                                                                     valid_items,
                                                                     init);

            // Launching kernel
            if(current_device_warp_size == ws32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        warp_exclusive_scan_kernel<T, block_size_ws32, logical_warp_size, Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws32),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    init);
                HIP_CHECK(hipGetLastError());
            }
            else if(current_device_warp_size == ws64)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        warp_exclusive_scan_kernel<T, block_size_ws64, logical_warp_size, Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws64),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    init);
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
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_reduce_kernel(
    T* device_input, T* device_output, int valid_items, T* device_output_reductions, T init)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    // Minimum size is 1
    constexpr unsigned int warps_no = test_utils::max(BlockSize / LogicalWarpSize, 1u);
    const unsigned int     warp_id  = test_utils::logical_warp_id<LogicalWarpSize>();
    unsigned int           index    = hipThreadIdx_x + (hipBlockIdx_x * hipBlockDim_x);

    T value     = device_input[index];
    T reduction = value;

    using wscan_t = hipcub::WarpScan<T, LogicalWarpSize>;
    __shared__
    typename wscan_t::TempStorage storage[warps_no];

    auto scan_op = test_utils::plus{};
    if constexpr(Partial)
    {
        wscan_t(storage[warp_id])
            .ExclusiveScanPartial(value, value, init, scan_op, valid_items, reduction);
    }
    else
    {
        (void)valid_items;
        wscan_t(storage[warp_id]).ExclusiveScan(value, value, init, scan_op, reduction);
    }

    device_output[index] = value;
    if((hipThreadIdx_x % LogicalWarpSize) == 0)
    {
        device_output_reductions[index / LogicalWarpSize] = reduction;
    }
}

template<class T, unsigned int BlockSize, unsigned int LogicalWarpSize, bool Partial>
__global__ __launch_bounds__(BlockSize)
auto warp_exclusive_scan_reduce_kernel(T* /*device_input*/,
                                       T* /*device_output*/,
                                       int /*valid_items*/,
                                       T* /*device_output_reductions*/,
                                       T /*init*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<typename TestFixture, bool Partial>
void exclusive_scan_reduce_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    // for bfloat16 and half we use double for host-side accumulation
    using binary_op_type_host = typename test_utils::select_plus_operator_host<T>::type;
    binary_op_type_host binary_op_host;

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
        const T        init = static_cast<T>(
            test_utils::get_random_value(0, 100, seed_value + seed_value_addition));
        SCOPED_TRACE(testing::Message() << "with init = " << init);

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

        // Note: valid_items is only actually used when Partial is true
        const std::vector<int> valid_items_values
            = Partial ? test_utils::get_warp_valid_items<logical_warp_size>(seed_value)
                      : std::vector<int>{logical_warp_size};

        for(int valid_items : valid_items_values)
        {
            SCOPED_TRACE(testing::Message() << "with valid_items = " << valid_items);

            // Calculate expected results on host
            std::vector<T> expected(input);
            std::vector<T> expected_reductions(output_reductions.size());
            expected_reductions
                = warp_scan_host<false /*IsInclusive*/, logical_warp_size>(expected,
                                                                           binary_op_host,
                                                                           valid_items,
                                                                           init);

            // Launching kernel
            if(current_device_warp_size == ws32)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T,
                                                                      block_size_ws32,
                                                                      logical_warp_size,
                                                                      Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws32),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    device_output_reductions,
                    init);
                HIP_CHECK(hipGetLastError());
            }
            else if(current_device_warp_size == ws64)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(warp_exclusive_scan_reduce_kernel<T,
                                                                      block_size_ws64,
                                                                      logical_warp_size,
                                                                      Partial>),
                    dim3(grid_size),
                    dim3(block_size_ws64),
                    0,
                    0,
                    device_input,
                    device_output,
                    valid_items,
                    device_output_reductions,
                    init);
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
            test_utils::assert_near(output_reductions,
                                    expected_reductions,
                                    test_utils::precision<T>::value * logical_warp_size);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

#endif // TEST_HIPCUB_WARP_SCAN_IMPL_HPP_
