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

#include <hipcub/warp/warp_store.hpp>

#include <type_traits>

template<
    class T,
    unsigned WarpSize,
    ::hipcub::WarpStoreAlgorithm Algorithm
>
struct Params
{
    using type = T;
    static constexpr unsigned warp_size = WarpSize;
    static constexpr ::hipcub::WarpStoreAlgorithm algorithm = Algorithm;
};

template<class Params>
class HipcubWarpStoreTest : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubWarpStoreTestParams = ::testing::Types<
    Params<int, 1U, ::hipcub::WARP_STORE_DIRECT>,
    Params<int, 1U, ::hipcub::WARP_STORE_STRIPED>,
    Params<int, 1U, ::hipcub::WARP_STORE_VECTORIZE>,
    Params<int, 1U, ::hipcub::WARP_STORE_TRANSPOSE>,

    Params<int, 16U, ::hipcub::WARP_STORE_DIRECT>,
    Params<int, 16U, ::hipcub::WARP_STORE_STRIPED>,
    Params<int, 16U, ::hipcub::WARP_STORE_VECTORIZE>,
    Params<int, 16U, ::hipcub::WARP_STORE_TRANSPOSE>,

    Params<int, 32U, ::hipcub::WARP_STORE_DIRECT>,
    Params<int, 32U, ::hipcub::WARP_STORE_STRIPED>,
    Params<int, 32U, ::hipcub::WARP_STORE_VECTORIZE>,
    Params<int, 32U, ::hipcub::WARP_STORE_TRANSPOSE>,

    Params<int, 64U, ::hipcub::WARP_STORE_DIRECT>,
    Params<int, 64U, ::hipcub::WARP_STORE_STRIPED>,
    Params<int, 64U, ::hipcub::WARP_STORE_VECTORIZE>,
    Params<int, 64U, ::hipcub::WARP_STORE_TRANSPOSE>
>;

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__device__ auto warp_store_test(T* d_input, T* d_output)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    T thread_data[ItemsPerThread];
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = d_input[threadIdx.x * ItemsPerThread + i];
    }

    using WarpStoreT = ::hipcub::WarpStore<T, ItemsPerThread, Algorithm, LogicalWarpSize>;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    constexpr int tile_size = ItemsPerThread * LogicalWarpSize;

    const unsigned                              warp_id = threadIdx.x / LogicalWarpSize;
    __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];

    WarpStoreT(temp_storage[warp_id]).Store(d_output + warp_id * tile_size, thread_data);
}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__device__ auto warp_store_test(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_store_kernel(T* d_input, T* d_output)
{
    warp_store_test<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>(d_input, d_output);
}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__device__ auto warp_store_guarded_test(T* d_input, T* d_output, int valid_items)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    T thread_data[ItemsPerThread];
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = d_input[threadIdx.x * ItemsPerThread + i];
    }

    using WarpStoreT = ::hipcub::WarpStore<T, ItemsPerThread, Algorithm, LogicalWarpSize>;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    constexpr int tile_size = ItemsPerThread * LogicalWarpSize;

    const unsigned                              warp_id = threadIdx.x / LogicalWarpSize;
    __shared__ typename WarpStoreT::TempStorage temp_storage[warps_in_block];

    WarpStoreT(temp_storage[warp_id]).Store(
        d_output + warp_id * tile_size,
        thread_data,
        valid_items
    );
}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__device__ auto warp_store_guarded_test(T* /*d_input*/, T* /*d_output*/, int /*valid_items*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                     BlockSize,
         unsigned                     ItemsPerThread,
         unsigned                     LogicalWarpSize,
         ::hipcub::WarpStoreAlgorithm Algorithm,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_store_guarded_kernel(T*  d_input,
                                                                       T*  d_output,
                                                                       int valid_items)
{
    warp_store_guarded_test<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm>(d_input,
                                                                                   d_output,
                                                                                   valid_items);
}

template<class T>
std::vector<T> stripe_vector(const std::vector<T>& v, const size_t warp_size, const size_t items_per_thread)
{
    const size_t period = warp_size * items_per_thread;
    std::vector<T> striped(v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        const size_t i_base = i % period;
        const size_t other_idx_base = ((items_per_thread * i_base) % period) + i_base / warp_size;
        const size_t other_idx = other_idx_base + period * (i / period);
        striped[i] = v[other_idx];
    }
    return striped;
}

TYPED_TEST_SUITE(HipcubWarpStoreTest, HipcubWarpStoreTestParams);

TYPED_TEST(HipcubWarpStoreTest, WarpStore)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    constexpr unsigned warp_size = TestFixture::params::warp_size;
    constexpr ::hipcub::WarpStoreAlgorithm algorithm = TestFixture::params::algorithm;
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

    warp_store_kernel<block_size, items_per_thread, warp_size, algorithm>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input, d_output);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    auto expected = input;
    if (algorithm == ::hipcub::WarpStoreAlgorithm::WARP_STORE_STRIPED)
    {
        expected = stripe_vector(input, warp_size, items_per_thread);
    }

    ASSERT_EQ(expected, output);
}

TYPED_TEST(HipcubWarpStoreTest, WarpStoreGuarded)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    constexpr unsigned warp_size = TestFixture::params::warp_size;
    constexpr ::hipcub::WarpStoreAlgorithm algorithm = TestFixture::params::algorithm;
    constexpr unsigned items_per_thread = 4;
    constexpr unsigned block_size = 1024;
    constexpr unsigned items_count = items_per_thread * block_size;
    constexpr int valid_items = warp_size / 4;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    T* d_input{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, items_count * sizeof(T)));
    HIP_CHECK(hipMemset(d_output, 0, items_count * sizeof(T)));

    warp_store_guarded_kernel<block_size, items_per_thread, warp_size, algorithm>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input, d_output, valid_items);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    auto expected = input;
    if (algorithm == ::hipcub::WarpStoreAlgorithm::WARP_STORE_STRIPED)
    {
        expected = stripe_vector(expected, warp_size, items_per_thread);
    }
    for (size_t warp_idx = 0; warp_idx < block_size / warp_size; ++warp_idx)
    {
        auto segment_begin = std::next(expected.begin(), warp_idx * warp_size * items_per_thread);
        auto segment_end = std::next(expected.begin(), (warp_idx + 1) * warp_size * items_per_thread);
        std::fill(std::next(segment_begin, valid_items), segment_end, static_cast<T>(0));
    }

    ASSERT_EQ(expected, output);
}
