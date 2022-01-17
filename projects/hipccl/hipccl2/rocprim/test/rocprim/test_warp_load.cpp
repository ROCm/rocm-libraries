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

#include <rocprim/warp/warp_load.hpp>

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize,
    ::rocprim::warp_load_method Method
>
struct Params
{
    using type = T;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int warp_size = WarpSize;
    static constexpr ::rocprim::warp_load_method method = Method;
};

template<class Params>
class WarpLoadTest : public ::testing::Test
{
public:
    using params = Params;
};

using WarpLoadTestParams = ::testing::Types<
    Params<int, 4U, 1U, ::rocprim::warp_load_method::warp_load_direct>,
    Params<int, 4U, 1U, ::rocprim::warp_load_method::warp_load_striped>,
    Params<int, 4U, 1U, ::rocprim::warp_load_method::warp_load_vectorize>,
    Params<int, 4U, 1U, ::rocprim::warp_load_method::warp_load_transpose>,

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
    Params<float2, 4U, 32U, ::rocprim::warp_load_method::warp_load_transpose>
>;

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int LogicalWarpSize,
    ::rocprim::warp_load_method Method
>
__global__
__launch_bounds__(BlockSize)
void warp_load_kernel(T* d_input,
                      T* d_output)
{
    static_assert(BlockSize % LogicalWarpSize == 0,
                  "LogicalWarpSize must be a divisor of BlockSize");
    using warp_load_type = ::rocprim::warp_load<
        T,
        ItemsPerThread,
        test_utils::DeviceSelectWarpSize<LogicalWarpSize>::value,
        Method
    >;
    constexpr unsigned int tile_size = ItemsPerThread * LogicalWarpSize;
    constexpr unsigned int num_warps = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = hipThreadIdx_x / LogicalWarpSize;

    ROCPRIM_SHARED_MEMORY typename warp_load_type::storage_type storage[num_warps];
    T thread_data[ItemsPerThread];

    warp_load_type().load(d_input + warp_id * tile_size, thread_data, storage[warp_id]);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        d_output[hipThreadIdx_x * ItemsPerThread + i] = thread_data[i];
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int LogicalWarpSize,
    ::rocprim::warp_load_method Method
>
__global__
__launch_bounds__(BlockSize)
void warp_load_guarded_kernel(T* d_input,
                              T* d_output,
                              int valid_items,
                              T oob_default)
{
    static_assert(BlockSize % LogicalWarpSize == 0,
                  "LogicalWarpSize must be a divisor of BlockSize");
    using warp_load_type = ::rocprim::warp_load<
        T,
        ItemsPerThread,
        test_utils::DeviceSelectWarpSize<LogicalWarpSize>::value,
        Method
    >;
    constexpr unsigned int tile_size = ItemsPerThread * LogicalWarpSize;
    constexpr unsigned int num_warps = BlockSize / LogicalWarpSize;
    const unsigned warp_id = hipThreadIdx_x / LogicalWarpSize;

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

TYPED_TEST_SUITE(WarpLoadTest, WarpLoadTestParams);

TYPED_TEST(WarpLoadTest, WarpLoad)
{
    using T = typename TestFixture::params::type;
    constexpr unsigned int warp_size = TestFixture::params::warp_size;
    constexpr ::rocprim::warp_load_method method = TestFixture::params::method;
    constexpr unsigned int items_per_thread = 4;
    constexpr unsigned int block_size = 1024;
    constexpr unsigned int items_count = items_per_thread * block_size;

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    T* d_input{};
    HIP_CHECK(hipMalloc(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(hipMalloc(&d_output, items_count * sizeof(T)));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            warp_load_kernel<
                T,
                block_size,
                items_per_thread,
                warp_size,
                method
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
    constexpr T oob_default = std::numeric_limits<T>::max();

    SKIP_IF_UNSUPPORTED_WARP_SIZE(warp_size);

    std::vector<T> input(items_count);
    std::iota(input.begin(), input.end(), static_cast<T>(0));

    T* d_input{};
    HIP_CHECK(hipMalloc(&d_input, items_count * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), items_count * sizeof(T), hipMemcpyHostToDevice));
    T* d_output{};
    HIP_CHECK(hipMalloc(&d_output, items_count * sizeof(T)));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            warp_load_guarded_kernel<
                T,
                block_size,
                items_per_thread,
                warp_size,
                method
            >
        ),
        dim3(1), dim3(block_size), 0, 0,
        d_input, d_output,
        valid_items, oob_default
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> output(items_count);
    HIP_CHECK(hipMemcpy(output.data(), d_output, items_count * sizeof(T), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

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
