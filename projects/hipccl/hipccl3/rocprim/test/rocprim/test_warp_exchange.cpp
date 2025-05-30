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
#include "../../common/warp_exchange.hpp"

#include "test_utils.hpp"

#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_exchange.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdint.h>
#include <type_traits>
#include <vector>

template<class T,
         unsigned int ItemsPerThread,
         unsigned int VirtualWaveSize,
         class ExchangeOp = void>
struct Params
{
    using type = T;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int warp_size        = VirtualWaveSize;
    using exchange_op = ExchangeOp;
};

template<class Params>
class WarpExchangeTest : public ::testing::Test
{
public:
    using params = Params;
};

using WarpExchangeTestParams = ::testing::Types<
    Params<int, 4U, 8U, common::BlockedToStripedOp>,
    Params<int8_t, 4U, 16U, common::BlockedToStripedOp>,
    Params<int16_t, 2U, 32U, common::BlockedToStripedOp>,
    Params<int64_t, 4U, 64U, common::BlockedToStripedOp>,
    Params<float, 4U, 32U, common::BlockedToStripedOp>,
    Params<double, 5U, 32U, common::BlockedToStripedOp>,
    // half should be supported, but is missing some key operators.
    // we should uncomment these, as soon as these are implemented and the tests compile and work as intended.
    //Params<rocprim::half, 4U, 64U, common::BlockedToStripedOp>,
    Params<rocprim::bfloat16, 4U, 8U, common::BlockedToStripedOp>,

    Params<int, 4U, 8U, common::BlockedToStripedShuffleOp>,
    Params<int8_t, 4U, 16U, common::BlockedToStripedShuffleOp>,
    Params<int16_t, 2U, 32U, common::BlockedToStripedShuffleOp>,
    Params<int64_t, 4U, 8U, common::BlockedToStripedShuffleOp>,
    Params<float, 4U, 32U, common::BlockedToStripedShuffleOp>,
    Params<double, 4U, 64U, common::BlockedToStripedShuffleOp>,
    //Params<rocprim::half, 4U,  8U, common::BlockedToStripedShuffleOp>,
    Params<rocprim::bfloat16, 4U, 8U, common::BlockedToStripedShuffleOp>,

    Params<int, 8U, 8U, common::BlockedToStripedShuffleOp>,
    Params<int8_t, 16U, 16U, common::BlockedToStripedShuffleOp>,
    Params<int16_t, 32U, 32U, common::BlockedToStripedShuffleOp>,
    Params<int64_t, 8U, 8U, common::BlockedToStripedShuffleOp>,
    Params<float, 32U, 32U, common::BlockedToStripedShuffleOp>,
    Params<double, 64U, 64U, common::BlockedToStripedShuffleOp>,
    //Params<rocprim::half, 8U,  8U, common::BlockedToStripedShuffleOp>,
    Params<rocprim::bfloat16, 8U, 8U, common::BlockedToStripedShuffleOp>,

    Params<float, 16U, 64U, common::BlockedToStripedShuffleOp>,
    Params<double, 8U, 32U, common::BlockedToStripedShuffleOp>,
    Params<int8_t, 32U, 64U, common::BlockedToStripedShuffleOp>,
    Params<int, 32U, 8U, common::BlockedToStripedShuffleOp>,
    Params<int64_t, 64U, 8U, common::BlockedToStripedShuffleOp>,
    Params<rocprim::bfloat16, 64U, 16U, common::BlockedToStripedShuffleOp>,
    Params<float, 2U, 16U, common::StripedToBlockedShuffleOp>,

    Params<int, 4U, 8U, common::StripedToBlockedOp>,
    Params<int8_t, 4U, 16U, common::StripedToBlockedOp>,
    Params<int16_t, 2U, 32U, common::StripedToBlockedOp>,
    Params<int64_t, 4U, 64U, common::StripedToBlockedOp>,
    Params<float, 4U, 32U, common::StripedToBlockedOp>,
    Params<double, 5U, 32U, common::StripedToBlockedOp>,
    //Params<rocprim::half, 4U, 64U, common::StripedToBlockedOp>,
    Params<rocprim::bfloat16, 4U, 8U, common::StripedToBlockedOp>,

    Params<int, 4U, 8U, common::StripedToBlockedShuffleOp>,
    Params<int8_t, 4U, 16U, common::StripedToBlockedShuffleOp>,
    Params<int16_t, 2U, 32U, common::StripedToBlockedShuffleOp>,
    Params<int64_t, 4U, 8U, common::StripedToBlockedShuffleOp>,
    Params<float, 4U, 32U, common::StripedToBlockedShuffleOp>,
    Params<double, 4U, 64U, common::StripedToBlockedShuffleOp>,
    //Params<rocprim::half, 4U,  8U, common::StripedToBlockedShuffleOp>,
    Params<rocprim::bfloat16, 4U, 8U, common::StripedToBlockedShuffleOp>,

    Params<int, 8U, 8U, common::StripedToBlockedShuffleOp>,
    Params<int8_t, 16U, 16U, common::StripedToBlockedShuffleOp>,
    Params<int16_t, 32U, 32U, common::StripedToBlockedShuffleOp>,
    Params<int64_t, 8U, 8U, common::StripedToBlockedShuffleOp>,
    Params<float, 32U, 32U, common::StripedToBlockedShuffleOp>,
    Params<double, 64U, 64U, common::StripedToBlockedShuffleOp>,
    //Params<rocprim::half, 8U,  8U, common::StripedToBlockedShuffleOp>,
    Params<rocprim::bfloat16, 8U, 8U, common::StripedToBlockedShuffleOp>,

    Params<float, 16U, 64U, common::StripedToBlockedShuffleOp>,
    Params<double, 8U, 32U, common::StripedToBlockedShuffleOp>,
    Params<int8_t, 32U, 64U, common::StripedToBlockedShuffleOp>,
    Params<int, 32U, 8U, common::StripedToBlockedShuffleOp>,
    Params<int64_t, 64U, 8U, common::StripedToBlockedShuffleOp>,
    Params<rocprim::bfloat16, 64U, 16U, common::StripedToBlockedShuffleOp>,
    Params<float, 2U, 16U, common::StripedToBlockedShuffleOp>>;

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class Op, class T>
__device__
auto warp_exchange_test(T* d_input, T* d_output)
    -> std::enable_if_t<common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    using warp_exchange_type         = ::rocprim::warp_exchange<T, ItemsPerThread, LogicalWarpSize>;
    constexpr unsigned int num_warps = ::rocprim::arch::wavefront::max_size() / LogicalWarpSize;
    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage[num_warps];

    T thread_data[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        thread_data[i] = d_input[threadIdx.x * ItemsPerThread + i];
    }

    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    Op{}(warp_exchange_type(), thread_data, thread_data, storage[warp_id]);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class Op, class T>
__device__
auto warp_exchange_test(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class Op, class T>
__device__
auto warp_exchange_test_not_inplace(T* d_input, T* d_output)
    -> std::enable_if_t<common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    using warp_exchange_type         = ::rocprim::warp_exchange<T, ItemsPerThread, LogicalWarpSize>;
    constexpr unsigned int num_warps = ::rocprim::arch::wavefront::max_size() / LogicalWarpSize;
    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage[num_warps];

    T thread_data[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        thread_data[i] = d_input[threadIdx.x * ItemsPerThread + i];
    }

    T output[ItemsPerThread];

    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    Op{}(warp_exchange_type(), thread_data, output, storage[warp_id]);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = output[i];
    }
}

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class Op, class T>
__device__
auto warp_exchange_test_not_inplace(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class Op, class T>
__global__ void warp_exchange_kernel(T* d_input, T* d_output, bool inplace = true)
{
    if(inplace)
    {
        warp_exchange_test<ItemsPerThread, LogicalWarpSize, Op>(d_input, d_output);
    }
    else
    {
        warp_exchange_test_not_inplace<ItemsPerThread, LogicalWarpSize, Op>(d_input, d_output);
    }
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

TYPED_TEST_SUITE(WarpExchangeTest, WarpExchangeTestParams);

TYPED_TEST(WarpExchangeTest, WarpExchange)
{
    using T = typename TestFixture::params::type;
    constexpr unsigned int warp_size = TestFixture::params::warp_size;
    constexpr unsigned int items_per_thread = TestFixture::params::items_per_thread;
    using exchange_op                       = typename TestFixture::params::exchange_op;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size, device_id);

    unsigned int hw_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, hw_warp_size));
    const unsigned int block_size  = hw_warp_size;
    const unsigned int items_count = items_per_thread * block_size;

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));
    auto expected = input;
    if(std::is_same<exchange_op, common::StripedToBlockedOp>::value
       || std::is_same<exchange_op, common::StripedToBlockedShuffleOp>::value)
    {
        input = stripe_vector(input, warp_size, items_per_thread);
    }

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(items_count);
    HIP_CHECK(hipMemset(d_output.get(), 0, items_count * sizeof(T)));

    warp_exchange_kernel<items_per_thread, warp_size, exchange_op>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input.get(), d_output.get());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    auto output = d_output.load();

    if(std::is_same<exchange_op, common::BlockedToStripedOp>::value
       || std::is_same<exchange_op, common::BlockedToStripedShuffleOp>::value)
    {
        expected = stripe_vector(expected, warp_size, items_per_thread);
    }

    ASSERT_EQ(expected, output);
}

TYPED_TEST_SUITE(WarpExchangeTest, WarpExchangeTestParams);

TYPED_TEST(WarpExchangeTest, WarpExchangeNotInplace)
{
    using T                                 = typename TestFixture::params::type;
    constexpr unsigned int warp_size        = TestFixture::params::warp_size;
    constexpr unsigned int items_per_thread = TestFixture::params::items_per_thread;
    using exchange_op                       = typename TestFixture::params::exchange_op;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size, device_id);

    unsigned int hw_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, hw_warp_size));
    const unsigned int block_size  = hw_warp_size;
    const unsigned int items_count = items_per_thread * block_size;

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));
    auto expected = input;
    if(std::is_same<exchange_op, common::StripedToBlockedOp>::value
       || std::is_same<exchange_op, common::StripedToBlockedShuffleOp>::value)
    {
        input = stripe_vector(input, warp_size, items_per_thread);
    }

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(items_count);
    HIP_CHECK(hipMemset(d_output.get(), 0, items_count * sizeof(T)));

    warp_exchange_kernel<items_per_thread, warp_size, exchange_op>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input.get(), d_output.get(), false);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    auto output = d_output.load();

    if(std::is_same<exchange_op, common::BlockedToStripedOp>::value
       || std::is_same<exchange_op, common::BlockedToStripedShuffleOp>::value)
    {
        expected = stripe_vector(expected, warp_size, items_per_thread);
    }

    ASSERT_EQ(expected, output);
}

using WarpExchangeScatterTestParams = ::testing::Types<Params<int, 4U, 8U>,
                                                       Params<int16_t, 4U, 16U>,
                                                       Params<int, 2U, 32U>,
                                                       Params<int64_t, 4U, 32U>,
                                                       Params<int, 5U, 32U>,
                                                       Params<int, 4U, 64U>>;

template<class Params>
class WarpExchangeScatterTest : public ::testing::Test
{
public:
    using params = Params;
};

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class T, class OffsetT>
__device__
auto warp_exchange_scatter_test(T* d_input, T* d_output, OffsetT* d_ranks)
    -> std::enable_if_t<common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    using warp_exchange_type = ::rocprim::warp_exchange<T, ItemsPerThread, LogicalWarpSize>;

    constexpr unsigned int num_warps = ::rocprim::arch::wavefront::max_size() / LogicalWarpSize;
    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage[num_warps];

    T thread_data[ItemsPerThread];
    OffsetT thread_ranks[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        thread_data[i]  = d_input[threadIdx.x * ItemsPerThread + i];
        thread_ranks[i] = d_ranks[threadIdx.x * ItemsPerThread + i];
    }

    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    warp_exchange_type{}.scatter_to_striped(thread_data,
                                            thread_data,
                                            thread_ranks,
                                            storage[warp_id]);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class T, class OffsetT>
__device__
auto warp_exchange_scatter_test(T* /*d_input*/, T* /*d_output*/, OffsetT* /*d_ranks*/)
    -> std::enable_if_t<!common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int ItemsPerThread, unsigned int LogicalWarpSize, class T, class OffsetT>
__global__ void warp_exchange_scatter_kernel(T* d_input, T* d_output, OffsetT* d_ranks)
{
    warp_exchange_scatter_test<ItemsPerThread, LogicalWarpSize>(d_input, d_output, d_ranks);
}

TYPED_TEST_SUITE(WarpExchangeScatterTest, WarpExchangeScatterTestParams);

TYPED_TEST(WarpExchangeScatterTest, WarpExchangeScatter)
{
    using T = typename TestFixture::params::type;
    constexpr unsigned int warp_size = TestFixture::params::warp_size;
    constexpr unsigned int items_per_thread = TestFixture::params::items_per_thread;
    using OffsetT = unsigned short;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size, device_id);
    unsigned int hw_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, hw_warp_size));
    const unsigned int block_size  = hw_warp_size;
    const unsigned int items_count = items_per_thread * block_size;

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));
    const auto                 expected = stripe_vector(input, warp_size, items_per_thread);
    std::default_random_engine prng(std::random_device{}());
    for(auto it = input.begin(), end = input.end(); it != end; it += warp_size)
    {
        std::shuffle(it, it + warp_size, prng);
    }
    std::vector<OffsetT> ranks(items_count);
    std::transform(input.begin(),
                   input.end(),
                   ranks.begin(),
                   [](const T input_val)
                   { return static_cast<OffsetT>(input_val) % (warp_size * items_per_thread); });

    common::device_ptr<OffsetT> d_ranks(ranks);
    common::device_ptr<T>       d_input(input);
    common::device_ptr<T>       d_output(items_count);
    HIP_CHECK(hipMemset(d_output.get(), 0, items_count * sizeof(T)));

    warp_exchange_scatter_kernel<items_per_thread, warp_size>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input.get(), d_output.get(), d_ranks.get());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    auto output = d_output.load();

    ASSERT_EQ(expected, output);
}
