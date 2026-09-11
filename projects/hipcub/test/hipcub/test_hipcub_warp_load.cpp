// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/warp/warp_load.hpp>

#include <type_traits>

template<
    class T,
    unsigned WarpSize,
    ::hipcub::WarpLoadAlgorithm Algorithm
>
struct Params
{
    using type = T;
    static constexpr unsigned warp_size = WarpSize;
    static constexpr ::hipcub::WarpLoadAlgorithm algorithm = Algorithm;
};

template<class Params>
class HipcubWarpLoadTest : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubWarpLoadTestParams = ::testing::Types<
    Params<int, 1U, ::hipcub::WARP_LOAD_DIRECT>,
    Params<int, 1U, ::hipcub::WARP_LOAD_STRIPED>,
    Params<int, 1U, ::hipcub::WARP_LOAD_VECTORIZE>,
    Params<int, 1U, ::hipcub::WARP_LOAD_TRANSPOSE>,

    Params<int, 16U, ::hipcub::WARP_LOAD_DIRECT>,
    Params<int, 16U, ::hipcub::WARP_LOAD_STRIPED>,
    Params<int, 16U, ::hipcub::WARP_LOAD_VECTORIZE>,
    Params<int, 16U, ::hipcub::WARP_LOAD_TRANSPOSE>,

    Params<int, 32U, ::hipcub::WARP_LOAD_DIRECT>,
    Params<int, 32U, ::hipcub::WARP_LOAD_STRIPED>,
    Params<int, 32U, ::hipcub::WARP_LOAD_VECTORIZE>,
    Params<int, 32U, ::hipcub::WARP_LOAD_TRANSPOSE>,

    Params<int, 64U, ::hipcub::WARP_LOAD_DIRECT>,
    Params<int, 64U, ::hipcub::WARP_LOAD_STRIPED>,
    Params<int, 64U, ::hipcub::WARP_LOAD_VECTORIZE>,
    Params<int, 64U, ::hipcub::WARP_LOAD_TRANSPOSE>
>;

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__ auto warp_load_test(T* d_input, T* d_output)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    using WarpLoadT = ::hipcub::WarpLoad<T, ItemsPerThread, Algorithm, LogicalWarpSize>;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    constexpr int tile_size = ItemsPerThread * LogicalWarpSize;

    const unsigned                             warp_id = threadIdx.x / LogicalWarpSize;
    __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
    T thread_data[ItemsPerThread];

    WarpLoadT(temp_storage[warp_id]).Load(d_input + warp_id * tile_size, thread_data);

    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__ auto warp_load_test(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_load_kernel(T* d_input, T* d_output)
{
    warp_load_test<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>(d_input, d_output);
}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__ auto warp_load_guarded_test(T* d_input, T* d_output, int valid_items, T oob_default)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    using WarpLoadT = ::hipcub::WarpLoad<T, ItemsPerThread, Algorithm, LogicalWarpSize>;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    constexpr int tile_size = ItemsPerThread * LogicalWarpSize;

    const unsigned                             warp_id = threadIdx.x / LogicalWarpSize;
    __shared__ typename WarpLoadT::TempStorage temp_storage[warps_in_block];
    T thread_data[ItemsPerThread];

    WarpLoadT(temp_storage[warp_id]).Load(
        d_input + warp_id * tile_size,
        thread_data,
        valid_items,
        oob_default
    );

    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__device__ auto
    warp_load_guarded_test(T* /*d_input*/, T* /*d_output*/, int /*valid_items*/, T /*oob_default*/)
        -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                    BlockSize,
         unsigned                    ItemsPerThread,
         unsigned                    LogicalWarpSize,
         ::hipcub::WarpLoadAlgorithm Algorithm,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_load_guarded_kernel(T*  d_input,
                                                                      T*  d_output,
                                                                      int valid_items,
                                                                      T   oob_default)
{
    warp_load_guarded_test<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>(d_input,
                                                                                  d_output,
                                                                                  valid_items,
                                                                                  oob_default);
}

template<class T>
std::vector<T> stripe_vector(const std::vector<T>& v, const size_t warp_size, const size_t items_per_thread)
{
    const size_t warp_items = warp_size * items_per_thread;
    std::vector<T> striped(v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        const size_t warp_idx = i % warp_items;
        const size_t other_warp_idx = (warp_idx % items_per_thread) * warp_size + (warp_idx / items_per_thread);
        const size_t other_idx = other_warp_idx + warp_items * (i / warp_items);
        striped[i] = v[other_idx];
    }
    return striped;
}

TYPED_TEST_SUITE(HipcubWarpLoadTest, HipcubWarpLoadTestParams);

TYPED_TEST(HipcubWarpLoadTest, WarpLoad)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    constexpr unsigned warp_size = TestFixture::params::warp_size;
    constexpr ::hipcub::WarpLoadAlgorithm algorithm = TestFixture::params::algorithm;
    constexpr unsigned items_per_thread = 4;
    constexpr unsigned block_size = 1024;
    constexpr unsigned items_count = items_per_thread * block_size;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    T* d_input{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, items_count * sizeof(T)));

    warp_load_kernel<block_size, items_per_thread, warp_size, algorithm>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input, d_output);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    auto expected = input;
    if (algorithm == ::hipcub::WarpLoadAlgorithm::WARP_LOAD_STRIPED)
    {
        expected = stripe_vector(input, warp_size, items_per_thread);
    }
    
    ASSERT_EQ(expected, output);
}

TYPED_TEST(HipcubWarpLoadTest, WarpLoadGuarded)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    constexpr unsigned warp_size = TestFixture::params::warp_size;
    constexpr ::hipcub::WarpLoadAlgorithm algorithm = TestFixture::params::algorithm;
    constexpr unsigned items_per_thread = 4;
    constexpr unsigned block_size = 1024;
    constexpr unsigned items_count = items_per_thread * block_size;
    constexpr int valid_items = warp_size / 4;
    constexpr T oob_default = std::numeric_limits<T>::max();

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    T* d_input{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, items_count * sizeof(T)));

    warp_load_guarded_kernel<block_size, items_per_thread, warp_size, algorithm>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input, d_output, valid_items, oob_default);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    auto expected = input;
    for (size_t warp_idx = 0; warp_idx < block_size / warp_size; ++warp_idx)
    {
        auto segment_begin = std::next(expected.begin(), warp_idx * warp_size * items_per_thread);
        auto segment_end = std::next(expected.begin(), (warp_idx + 1) * warp_size * items_per_thread);
        std::fill(std::next(segment_begin, valid_items), segment_end, oob_default);
    }
    
    if (algorithm == ::hipcub::WarpLoadAlgorithm::WARP_LOAD_STRIPED)
    {
        expected = stripe_vector(expected, warp_size, items_per_thread);
    }
    
    ASSERT_EQ(expected, output);
}
