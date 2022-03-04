// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_utils.hpp"

#include <rocprim/warp/warp_exchange.hpp>

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize,
    class ExchangeOp
>
struct Params
{
    using type = T;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int warp_size = WarpSize;
    using exchange_op = ExchangeOp;
};

template<class Params>
class WarpExchangeTest : public ::testing::Test
{
public:
    using params = Params;
};

struct BlockedToStripedOp
{
    template<
        class T,
        class warp_exchange_type,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage)
    {
        warp_exchange.blocked_to_striped(thread_data, thread_data, storage);
    }
};

struct BlockedToStripedShuffleOp
{
    template<
        class T,
        class warp_exchange_type,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/)
    {
        warp_exchange.blocked_to_striped_shuffle(thread_data, thread_data);
    }
};

struct StripedToBlockedOp
{
    template<
        class T,
        class warp_exchange_type,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage)
    {
        warp_exchange.striped_to_blocked(thread_data, thread_data, storage);
    }
};

struct StripedToBlockedShuffleOp
{
    template<
        class T,
        class warp_exchange_type,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/)
    {
        warp_exchange.striped_to_blocked_shuffle(thread_data, thread_data);
    }
};

struct ScatterToStripedOp
{
    template<
        class T,
        class OffsetT,
        class warp_exchange_type,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    OffsetT (&positions)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage)
    {
        warp_exchange.scatter_to_striped(thread_data, thread_data, positions, storage);
    }
};

using WarpExchangeTestParams = ::testing::Types<
    Params<int, 4U,  8U, BlockedToStripedOp>,
    Params<int, 4U, 16U, BlockedToStripedOp>,
    Params<int, 2U, 32U, BlockedToStripedOp>,
    Params<int, 4U, 32U, BlockedToStripedOp>,
    Params<int, 5U, 32U, BlockedToStripedOp>,
    Params<int, 4U, 64U, BlockedToStripedOp>,

    Params<int, 4U,  8U, BlockedToStripedShuffleOp>,
    Params<int, 4U, 16U, BlockedToStripedShuffleOp>,
    Params<int, 2U, 32U, BlockedToStripedShuffleOp>,
    Params<int, 4U, 32U, BlockedToStripedShuffleOp>,
    Params<int, 4U, 64U, BlockedToStripedShuffleOp>,

    Params<int, 4U,  8U, StripedToBlockedOp>,
    Params<int, 4U, 16U, StripedToBlockedOp>,
    Params<int, 2U, 32U, StripedToBlockedOp>,
    Params<int, 4U, 32U, StripedToBlockedOp>,
    Params<int, 5U, 32U, StripedToBlockedOp>,
    Params<int, 4U, 64U, StripedToBlockedOp>,

    Params<int, 4U,  8U, StripedToBlockedShuffleOp>,
    Params<int, 4U, 16U, StripedToBlockedShuffleOp>,
    Params<int, 2U, 32U, StripedToBlockedShuffleOp>,
    Params<int, 4U, 32U, StripedToBlockedShuffleOp>,
    Params<int, 4U, 64U, StripedToBlockedShuffleOp>
>;

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int LogicalWarpSize,
    class Op
>
__global__
__launch_bounds__(BlockSize)
void warp_exchange_kernel(T* d_input,
                          T* d_output)
{
    static_assert(BlockSize == LogicalWarpSize,
                  "BlockSize must be equal to LogicalWarpSize in this test");
    using warp_exchange_type = ::rocprim::warp_exchange<
        T,
        ItemsPerThread,
        test_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
    >;

    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage;

    T thread_data[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        thread_data[i] = d_input[hipThreadIdx_x * ItemsPerThread + i];
    }

    Op{}(warp_exchange_type(), thread_data, storage);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[hipThreadIdx_x * ItemsPerThread + i] = thread_data[i];
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
    using exchange_op = typename TestFixture::params::exchange_op;
    constexpr unsigned int block_size = warp_size;
    constexpr unsigned int items_count = items_per_thread * block_size;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));
    auto expected = input;
    if(std::is_same<exchange_op, StripedToBlockedOp>::value
        || std::is_same<exchange_op, StripedToBlockedShuffleOp>::value)
    {
        input = stripe_vector(input, warp_size, items_per_thread);
    }

    T* d_input{};
    HIP_CHECK(hipMalloc(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(hipMalloc(&d_output, items_count * sizeof(T)));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            warp_exchange_kernel<
                T,
                block_size,
                items_per_thread,
                warp_size,
                exchange_op
            >
        ),
        dim3(1), dim3(block_size), 0, 0,
        d_input, d_output
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    if(std::is_same<exchange_op, BlockedToStripedOp>::value
        || std::is_same<exchange_op, BlockedToStripedShuffleOp>::value)
    {
        expected = stripe_vector(expected, warp_size, items_per_thread);
    }

    ASSERT_EQ(expected, output);
}

using WarpExchangeScatterTestParams = ::testing::Types<
    Params<int, 4U,  8U, ScatterToStripedOp>,
    Params<int, 4U, 16U, ScatterToStripedOp>,
    Params<int, 2U, 32U, ScatterToStripedOp>,
    Params<int, 4U, 32U, ScatterToStripedOp>,
    Params<int, 4U, 64U, ScatterToStripedOp>
    >;

template<class Params>
class WarpExchangeScatterTest : public ::testing::Test
{
public:
    using params = Params;
};

template<
    class T,
    class OffsetT,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int LogicalWarpSize,
    class Op
>
__global__
__launch_bounds__(BlockSize)
void warp_exchange_scatter_kernel(T* d_input,
                                  T* d_output,
                                  OffsetT* d_ranks)
{
    static_assert(BlockSize == LogicalWarpSize,
                  "BlockSize must be equal to LogicalWarpSize in this test");
    using warp_exchange_type = ::rocprim::warp_exchange<
        T,
        ItemsPerThread,
        test_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
        >;

    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage;

    T thread_data[ItemsPerThread];
    OffsetT thread_ranks[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        thread_data[i] = d_input[hipThreadIdx_x * ItemsPerThread + i];
        thread_ranks[i] = d_ranks[hipThreadIdx_x * ItemsPerThread + i];
    }

    Op{}(warp_exchange_type(), thread_data, thread_ranks, storage);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[hipThreadIdx_x * ItemsPerThread + i] = thread_data[i];
    }
}

TYPED_TEST_SUITE(WarpExchangeScatterTest, WarpExchangeScatterTestParams);

TYPED_TEST(WarpExchangeScatterTest, WarpExchangeScatter)
{
    using T = typename TestFixture::params::type;
    constexpr unsigned int warp_size = TestFixture::params::warp_size;
    constexpr unsigned int items_per_thread = TestFixture::params::items_per_thread;
    using exchange_op = typename TestFixture::params::exchange_op;
    constexpr unsigned int block_size = warp_size;
    constexpr unsigned int items_count = items_per_thread * block_size;
    using OffsetT = unsigned short;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));
    auto expected = input;
    std::random_shuffle(input.begin(), input.end());
    std::vector<OffsetT> ranks(input.begin(), input.end());

    T* d_input{};
    HIP_CHECK(hipMalloc(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(hipMalloc(&d_output, items_count * sizeof(T)));
    OffsetT* d_ranks{};
    HIP_CHECK(hipMalloc(&d_ranks, items_count * sizeof(OffsetT)));
    HIP_CHECK(hipMemcpy(d_ranks, ranks.data(), items_count * sizeof(OffsetT), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            warp_exchange_scatter_kernel<
                T,
                OffsetT,
                block_size,
                items_per_thread,
                warp_size,
                exchange_op
                >
            ),
        dim3(1), dim3(block_size), 0, 0,
        d_input, d_output, d_ranks
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_ranks));

    if(std::is_same<exchange_op, ScatterToStripedOp>::value)
    {
        expected = stripe_vector(expected, warp_size, items_per_thread);
    }

    ASSERT_EQ(expected, output);
}