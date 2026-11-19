// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include "../../common/utils.hpp"
#include "../../common/utils_device_ptr.hpp"

#include "test_utils.hpp"
#include "test_utils_data_generation.hpp"

#include <rocprim/config.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_load.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <stdint.h>
#include <type_traits>
#include <vector>

template<class T,
         unsigned int                ItemsPerThread,
         unsigned int                VirtualWaveSize,
         ::rocprim::warp_load_method Method>
struct Params
{
    using type = T;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int                warp_size        = VirtualWaveSize;
    static constexpr ::rocprim::warp_load_method method = Method;
};

template<class Params>
class WarpLoadTest : public ::testing::Test
{
public:
    using params = Params;
};

using WarpLoadTestParams = ::testing::Types<
    Params<int, 4U, 8U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<int, 4U, 8U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<int, 4U, 8U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<int, 4U, 8U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<int, 4U, 32U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<int, 4U, 32U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<int, 4U, 32U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<int, 4U, 32U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<int, 5U, 32U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<int, 5U, 32U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<int, 5U, 32U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<int, 5U, 32U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<int, 4U, 64U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<int, 4U, 64U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<int, 4U, 64U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<int, 4U, 64U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<float2, 4U, 32U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<float2, 4U, 32U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<float2, 4U, 32U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<float2, 4U, 32U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<int8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<int8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<int8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<int8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<uint8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<uint8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<uint8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<uint8_t, 4U, 1U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<float, 4U, 1U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<float, 4U, 1U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<float, 4U, 1U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<float, 4U, 1U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<double, 4U, 1U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<double, 4U, 1U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<double, 4U, 1U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<double, 4U, 1U, ::rocprim::warp_load_method::warp_load_transpose>,

    // half should be supported, but is missing some key operators.
    // we should uncomment these, as soon as these are implemented and the tests compile and work as intended.
    //Params<rocprim::half, 4U, 1U, ::rocprim::warp_load_method::warp_load_direct>,
    //Params<rocprim::half, 4U, 1U, ::rocprim::warp_load_method::warp_load_striped>,
    //Params<rocprim::half, 4U, 1U, ::rocprim::warp_load_method::warp_load_vectorize>,
    //Params<rocprim::half, 4U, 1U, ::rocprim::warp_load_method::warp_load_transpose>,

    Params<rocprim::bfloat16, 4U, 1U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<rocprim::bfloat16, 4U, 1U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<rocprim::bfloat16, 4U, 1U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<rocprim::bfloat16, 4U, 1U, ::rocprim::warp_load_method::warp_load_transpose>>;

template<unsigned int                BlockSize,
         unsigned int                ItemsPerThread,
         unsigned int                LogicalWarpSize,
         ::rocprim::warp_load_method Method,
         class T>
__device__
auto warp_load_test(T* d_input, T* d_output)
    -> std::enable_if_t<common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    static_assert(BlockSize % LogicalWarpSize == 0,
                  "LogicalWarpSize must be a divisor of BlockSize");
    using warp_load_type = ::rocprim::warp_load<T, ItemsPerThread, LogicalWarpSize, Method>;
    constexpr unsigned int tile_size = ItemsPerThread * LogicalWarpSize;
    constexpr unsigned int num_warps = BlockSize / LogicalWarpSize;
    const unsigned int     warp_id   = threadIdx.x / LogicalWarpSize;

    ROCPRIM_SHARED_MEMORY typename warp_load_type::storage_type storage[num_warps];
    T thread_data[ItemsPerThread];

    warp_load_type().load(d_input + warp_id * tile_size, thread_data, storage[warp_id]);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned int                BlockSize,
         unsigned int                ItemsPerThread,
         unsigned int                LogicalWarpSize,
         ::rocprim::warp_load_method Method,
         class T>
__device__
auto warp_load_test(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int                BlockSize,
         unsigned int                ItemsPerThread,
         unsigned int                LogicalWarpSize,
         ::rocprim::warp_load_method Method,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_load_kernel(T* d_input, T* d_output)
{
    warp_load_test<BlockSize, ItemsPerThread, LogicalWarpSize, Method>(d_input, d_output);
}

template<unsigned int                BlockSize,
         unsigned int                ItemsPerThread,
         unsigned int                LogicalWarpSize,
         ::rocprim::warp_load_method Method,
         class T>
__device__
auto warp_load_guarded_test(T* d_input, T* d_output, int valid_items, T oob_default)
    -> std::enable_if_t<common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    static_assert(BlockSize % LogicalWarpSize == 0,
                  "LogicalWarpSize must be a divisor of BlockSize");
    using warp_load_type = ::rocprim::warp_load<T, ItemsPerThread, LogicalWarpSize, Method>;
    constexpr unsigned int tile_size = ItemsPerThread * LogicalWarpSize;
    constexpr unsigned int num_warps = BlockSize / LogicalWarpSize;
    const unsigned         warp_id   = threadIdx.x / LogicalWarpSize;

    ROCPRIM_SHARED_MEMORY typename warp_load_type::storage_type storage[num_warps];
    T thread_data[ItemsPerThread];

    warp_load_type().load(
        d_input + warp_id * tile_size,
        thread_data,
        valid_items,
        oob_default,
        storage[warp_id]
    );

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned int                BlockSize,
         unsigned int                ItemsPerThread,
         unsigned int                LogicalWarpSize,
         ::rocprim::warp_load_method Method,
         class T>
__device__
auto warp_load_guarded_test(T* /*d_input*/, T* /*d_output*/, int /*valid_items*/, T /*oob_default*/)
    -> std::enable_if_t<!common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int                BlockSize,
         unsigned int                ItemsPerThread,
         unsigned int                LogicalWarpSize,
         ::rocprim::warp_load_method Method,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_load_guarded_kernel(T*  d_input,
                                                                      T*  d_output,
                                                                      int valid_items,
                                                                      T   oob_default)
{
    warp_load_guarded_test<BlockSize, ItemsPerThread, LogicalWarpSize, Method>(d_input,
                                                                               d_output,
                                                                               valid_items,
                                                                               oob_default);
}

template<class T>
std::vector<T> stripe_vector(const std::vector<T>& v,
                             const size_t warp_size,
                             const size_t items_per_thread)
{
    const size_t warp_items = warp_size * items_per_thread;
    std::vector<T> striped(v.size());
    for(size_t i = 0; i < v.size(); i++)
    {
        const size_t warp_idx = i % warp_items;
        const size_t other_warp_idx = (warp_idx % items_per_thread) * warp_size
            + (warp_idx / items_per_thread);
        const size_t other_idx = other_warp_idx + warp_items * (i / warp_items);
        striped[i] = v[other_idx];
    }
    return striped;
}

TYPED_TEST_SUITE(WarpLoadTest, WarpLoadTestParams);

TYPED_TEST(WarpLoadTest, WarpLoad)
{
    using T = typename TestFixture::params::type;
    constexpr unsigned int warp_size = TestFixture::params::warp_size;
    constexpr ::rocprim::warp_load_method method = TestFixture::params::method;
    constexpr unsigned int items_per_thread = 4;
    constexpr unsigned int block_size = 1024;
    constexpr unsigned int items_count = items_per_thread * block_size;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size, device_id);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(items_count);

    warp_load_kernel<block_size, items_per_thread, warp_size, method>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input.get(), d_output.get());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output = d_output.load();

    auto expected = input;
    if(method == ::rocprim::warp_load_method::warp_load_striped)
    {
        expected = stripe_vector(input, warp_size, items_per_thread);
    }

    ASSERT_EQ(expected, output);
}

TYPED_TEST(WarpLoadTest, WarpLoadGuarded)
{
    using T = typename TestFixture::params::type;
    constexpr unsigned int warp_size = TestFixture::params::warp_size;
    constexpr ::rocprim::warp_load_method method = TestFixture::params::method;
    constexpr unsigned int items_per_thread = TestFixture::params::items_per_thread;
    constexpr unsigned int block_size = 1024;
    constexpr unsigned int items_count = items_per_thread * block_size;
    constexpr unsigned int valid_items = warp_size / 4;

    const T oob_default = rocprim::numeric_limits<T>::max();

    int device_id = test_common_utils::obtain_device_from_ctest();
    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size, device_id);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(items_count);

    warp_load_guarded_kernel<block_size, items_per_thread, warp_size, method>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input.get(),
                                              d_output.get(),
                                              valid_items,
                                              oob_default);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output = d_output.load();

    auto expected = input;
    for(size_t warp_idx = 0; warp_idx < block_size / warp_size; ++warp_idx)
    {
        auto segment_begin = std::next(expected.begin(), warp_idx * warp_size * items_per_thread);
        auto segment_end = std::next(expected.begin(), (warp_idx + 1) * warp_size * items_per_thread);
        std::fill(std::next(segment_begin, valid_items), segment_end, oob_default);
    }

    if(method == ::rocprim::warp_load_method::warp_load_striped)
    {
        expected = stripe_vector(expected, warp_size, items_per_thread);
    }

    ASSERT_EQ(expected, output);
}
