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

#include "test_utils_data_generation.hpp"
#include "test_utils_half.hpp"
#include <hipcub/warp/warp_exchange.hpp>

#include <type_traits>

template<class T, unsigned ItemsPerThread, unsigned WarpSize>
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

using HipcubWarpExchangeTestParams = ::testing::Types<Params<char, 1U, 8U>,
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
                                                      Params<double, 5U, 64U>,

                                                      Params<test_utils::half, 1U, 8U>,
                                                      Params<test_utils::half, 4U, 8U>,
                                                      Params<test_utils::half, 5U, 8U>,
                                                      Params<test_utils::half, 1U, 16U>,
                                                      Params<test_utils::half, 4U, 16U>,
                                                      Params<test_utils::half, 5U, 16U>,
                                                      Params<test_utils::half, 1U, 32U>,
                                                      Params<test_utils::half, 4U, 32U>,
                                                      Params<test_utils::half, 5U, 32U>,
                                                      Params<test_utils::half, 1U, 64U>,
                                                      Params<test_utils::half, 4U, 64U>,
                                                      Params<test_utils::half, 5U, 64U>>;

TYPED_TEST_SUITE(HipcubWarpExchangeTest, HipcubWarpExchangeTestParams);

struct BlockedToStripedOp
{
    template<class WarpExchange, class T, unsigned ItemsPerThread>
    HIPCUB_DEVICE void operator()(WarpExchange& warp_exchange,
                                  T (&thread_data)[ItemsPerThread]) const
    {
        warp_exchange.BlockedToStriped(thread_data, thread_data);
    }
};

struct StripedToBlockedOp
{
    template<class WarpExchange, class T, unsigned ItemsPerThread>
    HIPCUB_DEVICE void operator()(WarpExchange& warp_exchange,
                                  T (&thread_data)[ItemsPerThread]) const
    {
        warp_exchange.StripedToBlocked(thread_data, thread_data);
    }
};

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__device__ auto warp_exchange_test(T* d_input, T* d_output)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    T thread_data[ItemsPerThread];
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = d_input[threadIdx.x * ItemsPerThread + i];
    }

    using WarpExchangeT                                           = ::hipcub::WarpExchange<T,
                                                 ItemsPerThread,
                                                 LogicalWarpSize,
                                                 1, // ARCH
                                                 Algorithm>;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];
    const unsigned                                 warp_id = threadIdx.x / LogicalWarpSize;

    WarpExchangeT warp_exchange(temp_storage[warp_id]);
    Op{}(warp_exchange, thread_data);

    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__device__ auto warp_exchange_test(T* /*d_input*/, T* /*d_output*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_exchange_kernel(T* d_input, T* d_output)
{
    warp_exchange_test<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm, Op>(d_input,
                                                                                  d_output);
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

template<class Params, ::hipcub::WarpExchangeAlgorithm Algorithm>
constexpr bool is_warp_exchange_test_enabled
#ifdef HIPCUB_CUB_API
    = (Algorithm == ::hipcub::WARP_EXCHANGE_SMEM)
      || (Params::warp_size == Params::items_per_thread);
#elif HIPCUB_ROCPRIM_API
    = (Algorithm == ::hipcub::WARP_EXCHANGE_SMEM)
      || (Params::warp_size % Params::items_per_thread == 0);
#endif

template<class Params, class Op, ::hipcub::WarpExchangeAlgorithm Algorithm>
std::enable_if_t<is_warp_exchange_test_enabled<Params, Algorithm>> run_warp_exchange_test()
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                             = typename Params::type;
    constexpr unsigned warp_size        = Params::warp_size;
    constexpr unsigned items_per_thread = Params::items_per_thread;
    constexpr unsigned block_size = 1024;
    constexpr unsigned items_count = items_per_thread * block_size;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    for(int i = 0; i < static_cast<int>(input.size()); i++)
    {
        input[i] = test_utils::convert_to_device<T>(i);
    }
    std::vector<T> expected;
    if(std::is_same<Op, BlockedToStripedOp>::value)
    {
        expected = input;
        input    = stripe_vector(input, warp_size, items_per_thread);
    } else
    {
        expected = stripe_vector(input, warp_size, items_per_thread);
    }

    T* d_input{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, items_count * sizeof(T)));
    HIP_CHECK(hipMemset(d_output, 0, items_count * sizeof(T)));

    warp_exchange_kernel<block_size, items_per_thread, warp_size, Algorithm, Op>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input, d_output);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    for(int i = 0; i < static_cast<int>(items_count); i++)
    {
        ASSERT_EQ(test_utils::convert_to_native(expected[i]),
                  test_utils::convert_to_native(output[i]))
            << "at index " << i;
    }
}

template<class Params, class Op, ::hipcub::WarpExchangeAlgorithm Algorithm>
std::enable_if_t<!is_warp_exchange_test_enabled<Params, Algorithm>> run_warp_exchange_test()
{
    GTEST_SKIP()
#ifdef HIPCUB_CUB_API
        << "WARP_EXCHANGE_SHUFFLE is only supported when ItemsPerThread is equal to WarpSize";
#else
        << "WARP_EXCHANGE_SHUFFLE is only supported when ItemsPerThread is a divisor of WarpSize";
#endif
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeStripedToBlockedSmem)
{
    run_warp_exchange_test<typename TestFixture::params,
                           StripedToBlockedOp,
                           ::hipcub::WARP_EXCHANGE_SMEM>();
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeStripedToBlockedShuffle)
{
    run_warp_exchange_test<typename TestFixture::params,
                           StripedToBlockedOp,
                           ::hipcub::WARP_EXCHANGE_SHUFFLE>();
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeBlockedToStripedSmem)
{
    run_warp_exchange_test<typename TestFixture::params,
                           BlockedToStripedOp,
                           ::hipcub::WARP_EXCHANGE_SMEM>();
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeBlockedToStripedShuffle)
{
    run_warp_exchange_test<typename TestFixture::params,
                           BlockedToStripedOp,
                           ::hipcub::WARP_EXCHANGE_SHUFFLE>();
}

template<unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T,
         class OffsetT>
__device__ auto warp_exchange_scatter_to_striped_test(T* d_input, T* d_output, OffsetT* d_ranks)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    T       thread_data[ItemsPerThread];
    OffsetT thread_ranks[ItemsPerThread];
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i]  = d_input[threadIdx.x * ItemsPerThread + i];
        thread_ranks[i] = d_ranks[threadIdx.x * ItemsPerThread + i];
    }

    using WarpExchangeT = ::hipcub::WarpExchange<T, ItemsPerThread, LogicalWarpSize>;
    constexpr unsigned                             warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];
    const unsigned                                 warp_id = threadIdx.x / LogicalWarpSize;

    WarpExchangeT(temp_storage[warp_id]).ScatterToStriped(thread_data, thread_ranks);

    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        d_output[threadIdx.x * ItemsPerThread + i] = thread_data[i];
    }
}

template<unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T,
         class OffsetT>
__device__ auto
    warp_exchange_scatter_to_striped_test(T* /*d_input*/, T* /*d_output*/, OffsetT* /*d_ranks*/)
        -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T,
         class OffsetT>
__global__ __launch_bounds__(BlockSize) void warp_exchange_scatter_to_striped_kernel(
    T* d_input, T* d_output, OffsetT* d_ranks)
{
    warp_exchange_scatter_to_striped_test<BlockSize, ItemsPerThread, LogicalWarpSize>(d_input,
                                                                                      d_output,
                                                                                      d_ranks);
}

template<class T, class OffsetT>
std::vector<T> stripe_vector(const std::vector<T>&       v,
                             const std::vector<OffsetT>& ranks,
                             const size_t                threads_per_warp,
                             const size_t                items_per_thread)
{
    const size_t   items_per_warp = threads_per_warp * items_per_thread;
    const size_t   size           = v.size();
    std::vector<T> striped(size);
    for(size_t warp_idx = 0, global_idx = 0; warp_idx < size / items_per_warp; ++warp_idx)
    {
        const size_t warp_offset = warp_idx * items_per_warp;
        for(size_t thread_idx = 0; thread_idx < threads_per_warp; ++thread_idx)
        {
            for(size_t item_idx = 0; item_idx < items_per_thread; ++item_idx, ++global_idx)
            {
                const size_t rank      = ranks[global_idx];
                const size_t value_idx = warp_offset + ((items_per_thread * rank) % items_per_warp)
                                         + rank / threads_per_warp;
                const T value      = v[global_idx];
                striped[value_idx] = value;
            }
        }
    }
    return striped;
}

TYPED_TEST(HipcubWarpExchangeTest, WarpExchangeScatterToStriped)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
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
    for(int i = 0; i < static_cast<int>(input.size()); i++)
    {
        input[i] = test_utils::convert_to_device<T>(i);
    }

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
    HIP_CHECK(hipMemset(d_output, 0, items_count * sizeof(T)));

    warp_exchange_scatter_to_striped_kernel<block_size, items_per_thread, warp_size>
        <<<dim3(1), dim3(block_size), 0, 0>>>(d_input, d_output, d_ranks);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_ranks));
    HIP_CHECK(hipFree(d_output));

    const std::vector<T> expected = stripe_vector(input, ranks, warp_size, items_per_thread);

    for(int i = 0; i < static_cast<int>(items_count); i++)
    {
        ASSERT_EQ(test_utils::convert_to_native(expected[i]),
                  test_utils::convert_to_native(output[i]))
            << "at index " << i;
    }
}
