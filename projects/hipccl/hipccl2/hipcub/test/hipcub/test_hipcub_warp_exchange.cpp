// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipcub/warp/warp_exchange.hpp"

template<
    class T,
    unsigned ItemsPerThread,
    unsigned WarpSize
>
struct Params
{
    using type = T;
    static constexpr unsigned items_per_thread = ItemsPerThread;
    static constexpr unsigned warp_size = WarpSize;
};


template<class Params>
class HipcubWarpExchangeTest : public ::testing::Test
{
public:
    using params = Params;
};

using HipcubWarpExchangeTestParams = ::testing::Types<
    Params<char, 1U, 8U>,
    Params<char, 4U, 8U>,
    Params<char, 5U, 8U>,
    Params<char, 1U, 16U>,
    Params<char, 4U, 16U>,
    Params<char, 5U, 16U>,
    Params<char, 1U, 32U>,
    Params<char, 4U, 32U>,
    Params<char, 5U, 32U>,
    Params<char, 1U, 64U>,
    Params<char, 4U, 64U>,
    Params<char, 5U, 64U>,

    Params<int, 1U, 8U>,
    Params<int, 4U, 8U>,
    Params<int, 5U, 8U>,
    Params<int, 1U, 16U>,
    Params<int, 4U, 16U>,
    Params<int, 5U, 16U>,
    Params<int, 1U, 32U>,
    Params<int, 4U, 32U>,
    Params<int, 5U, 32U>,
    Params<int, 1U, 64U>,
    Params<int, 4U, 64U>,
    Params<int, 5U, 64U>,

    Params<double, 1U, 8U>,
    Params<double, 4U, 8U>,
    Params<double, 5U, 8U>,
    Params<double, 1U, 16U>,
    Params<double, 4U, 16U>,
    Params<double, 5U, 16U>,
    Params<double, 1U, 32U>,
    Params<double, 4U, 32U>,
    Params<double, 5U, 32U>,
    Params<double, 1U, 64U>,
    Params<double, 4U, 64U>,
    Params<double, 5U, 64U>
>;

TYPED_TEST_SUITE(HipcubWarpExchangeTest, HipcubWarpExchangeTestParams);

template<
    class T,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize
>
struct BlockedToStripedOp
{
    HIPCUB_DEVICE
    void operator()(
        ::hipcub::WarpExchange<T, ItemsPerThread, LogicalWarpSize> &warp_exchange,
        T (&thread_data)[ItemsPerThread]
    ) const
    {
        warp_exchange.BlockedToStriped(thread_data, thread_data);
    }
};

template<
    class T,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize
>
struct StripedToBlockedOp
{
    HIPCUB_DEVICE
    void operator()(
        ::hipcub::WarpExchange<T, ItemsPerThread, LogicalWarpSize> &warp_exchange,
        T (&thread_data)[ItemsPerThread]
    ) const
    {
        warp_exchange.StripedToBlocked(thread_data, thread_data);
    }
};

template<
    class T,
    unsigned BlockSize,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize,
    template<class, unsigned, unsigned> class Op
>
__global__
__launch_bounds__(BlockSize)
void warp_exchange_kernel(T* d_input, T* d_output)
{
    T thread_data[ItemsPerThread];
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = d_input[hipThreadIdx_x * ItemsPerThread + i];
    }

    using WarpExchangeT = ::hipcub::WarpExchange<
        T,
        ItemsPerThread,
        ::test_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
    >;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];
    const unsigned warp_id = hipThreadIdx_x / LogicalWarpSize;

    WarpExchangeT warp_exchange(temp_storage[warp_id]);
    Op<
        T,
        ItemsPerThread,
        ::test_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
    >{}(warp_exchange, thread_data);

    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        d_output[hipThreadIdx_x * ItemsPerThread + i] = thread_data[i];
    }
}

template<
    class T,
    class OffsetT,
    unsigned BlockSize,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_exchange_scatter_to_striped_kernel(T* d_input, T* d_output, OffsetT* d_ranks)
{
    T thread_data[ItemsPerThread];
    OffsetT thread_ranks[ItemsPerThread];
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = d_input[hipThreadIdx_x * ItemsPerThread + i];
        thread_ranks[i] = d_ranks[hipThreadIdx_x * ItemsPerThread + i];
    }

    using WarpExchangeT = ::hipcub::WarpExchange<
        T,
        ItemsPerThread,
        ::test_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
    >;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];
    const unsigned warp_id = hipThreadIdx_x / LogicalWarpSize;

    WarpExchangeT(temp_storage[warp_id]).ScatterToStriped(thread_data, thread_ranks);

    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        d_output[hipThreadIdx_x * ItemsPerThread + i] = thread_data[i];
    }
}

template<class T>
std::vector<T> stripe_vector(
    const std::vector<T>& v,
    const size_t warp_size,
    const size_t items_per_thread)
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

template<class T, class OffsetT>
std::vector<T> stripe_vector(
    const std::vector<T>& v,
    const std::vector<OffsetT>& ranks,
    const size_t threads_per_warp,
    const size_t items_per_thread)
{
    const size_t items_per_warp = threads_per_warp * items_per_thread;
    const size_t size = v.size();
    std::vector<T> striped(size);
    for (size_t warp_idx = 0, global_idx = 0; warp_idx < size / items_per_warp; ++warp_idx)
    {
        const size_t warp_offset = warp_idx * items_per_warp;
        for (size_t thread_idx = 0; thread_idx < threads_per_warp; ++thread_idx)
        {
            for (size_t item_idx = 0; item_idx < items_per_thread; ++item_idx, ++global_idx)
            {
                const size_t rank = ranks[global_idx];
                const size_t value_idx = warp_offset
                    + ((items_per_thread * rank) % items_per_warp)
                    + rank / threads_per_warp;
                const T value = v[global_idx];
                striped[value_idx] = value;
            }
        }
    }
    return striped;
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeStripedToBlocked)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    constexpr unsigned warp_size = TestFixture::params::warp_size;
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

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            warp_exchange_kernel<
                T,
                block_size,
                items_per_thread,
                warp_size,
                StripedToBlockedOp
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

    auto expected = stripe_vector(input, warp_size, items_per_thread);
    ASSERT_EQ(expected, output);
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeBlockedToStriped)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    constexpr unsigned warp_size = TestFixture::params::warp_size;
    constexpr unsigned items_per_thread = 4;
    constexpr unsigned block_size = 1024;
    constexpr unsigned items_count = items_per_thread * block_size;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));
    input = stripe_vector(input, warp_size, items_per_thread);

    T* d_input{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, items_count * sizeof(T)));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            warp_exchange_kernel<
                T,
                block_size,
                items_per_thread,
                warp_size,
                BlockedToStripedOp
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

    std::vector<T> expected(items_count);
    std::iota(expected.begin(), expected.end(), static_cast<T>(0));

    ASSERT_EQ(expected, output);
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeScatterToStriped)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;
    using OffsetT = int;
    constexpr unsigned warp_size = TestFixture::params::warp_size;
    constexpr unsigned items_per_thread = 4;
    constexpr unsigned block_size = 1024;
    constexpr unsigned items_count = items_per_thread * block_size;
    constexpr unsigned items_per_warp = warp_size * items_per_thread;
    constexpr int random_seed = 347268;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    std::vector<OffsetT> ranks(items_count);
    for (size_t i = 0; i < items_count / items_per_warp; ++i)
    {
        auto segment_begin = std::next(ranks.begin(), i * items_per_warp);
        auto segment_end = std::next(ranks.begin(), (i + 1) * items_per_warp);
        std::iota(segment_begin, segment_end, 0);
        std::shuffle(segment_begin, segment_end, std::default_random_engine(random_seed));
    }

    T* d_input{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    OffsetT* d_ranks{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_ranks, items_count * sizeof(OffsetT)));
    HIP_CHECK(hipMemcpy(d_ranks, ranks.data(), items_count * sizeof(OffsetT), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, items_count * sizeof(T)));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            warp_exchange_scatter_to_striped_kernel<
                T,
                OffsetT,
                block_size,
                items_per_thread,
                warp_size
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
    HIP_CHECK(hipFree(d_ranks));
    HIP_CHECK(hipFree(d_output));

    const std::vector<T> expected = stripe_vector(input, ranks, warp_size, items_per_thread);

    ASSERT_EQ(expected, output);
}
