/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <hipcub/thread/thread_load.hpp>
#include <hipcub/thread/thread_reduce.hpp>
#include <hipcub/thread/thread_scan.hpp>
#include <hipcub/thread/thread_search.hpp>
#include <hipcub/thread/thread_store.hpp>

#include "test_utils_bfloat16.hpp"
#include "test_utils_half.hpp"

#include "common_test_header.hpp"

#include <stdint.h>
#include <type_traits>
#include <vector>

template<class T>
struct params
{
    using type = T;
};

template<class Params>
class HipcubThreadOperationTests : public ::testing::Test
{
public:
    using type = typename Params::type;
};

using ThreadOperationTestParams = ::testing::Types<params<int8_t>,
                                                   params<int16_t>,
                                                   params<uint8_t>,
                                                   params<uint16_t>,
                                                   params<uint32_t>,
                                                   params<uint64_t>,
                                                   params<float>,
                                                   params<double>,
                                                   params<test_utils::bfloat16>,
                                                   params<test_utils::half>,
                                                   params<test_utils::custom_test_type<uint64_t>>,
                                                   params<test_utils::custom_test_type<double>>>;

TYPED_TEST_SUITE(HipcubThreadOperationTests, ThreadOperationTestParams);

template<class Type>
__global__
void thread_load_kernel(Type* volatile const device_input, Type* device_output)
{
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(index % 8 == hipcub::LOAD_DEFAULT)
    {
        device_output[index] = hipcub::ThreadLoad<hipcub::LOAD_DEFAULT>(device_input + index);
    }
    else if(index % 8 == hipcub::LOAD_CA)
    {
        device_output[index] = hipcub::ThreadLoad<hipcub::LOAD_CA>(device_input + index);
    }
    else if(index % 8 == hipcub::LOAD_CG)
    {
        device_output[index] = hipcub::ThreadLoad<hipcub::LOAD_CG>(device_input + index);
    }
    else if(index % 8 == hipcub::LOAD_CS)
    {
        device_output[index] = hipcub::ThreadLoad<hipcub::LOAD_CS>(device_input + index);
    }
    else if(index % 8 == hipcub::LOAD_CV)
    {
        device_output[index] = hipcub::ThreadLoad<hipcub::LOAD_CV>(device_input + index);
    }
    else if(index % 8 == hipcub::LOAD_LDG)
    {
        device_output[index] = hipcub::ThreadLoad<hipcub::LOAD_LDG>(device_input + index);
    }
    else if(index % 8 == hipcub::LOAD_VOLATILE)
    {
        device_output[index] = hipcub::ThreadLoad<hipcub::LOAD_VOLATILE>(device_input + index);
    }
    else // index % 8 == 7
    {
        // TODO: For compatibility, should also change this API in rocm backend
#ifdef __HIP_PLATFORM_NVIDIA__
        device_output[index] = hipcub::ThreadLoadVolatilePointer(
            device_input + index,
            ::cuda::std::integral_constant<bool, std::is_fundamental<Type>::value>{});
#else
        device_output[index] = hipcub::ThreadLoadVolatilePointer(
            device_input + index,
            std::bool_constant<std::is_fundamental<Type>::value>());
#endif
    }
}

TYPED_TEST(HipcubThreadOperationTests, Load)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::type;
    using native_T = test_utils::convert_to_native_t<T>;

    constexpr uint32_t block_size = 256;
    constexpr uint32_t grid_size = 128;
    constexpr uint32_t size = block_size * grid_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<native_T> input_native
            = test_utils::get_random_data<native_T>(size, 2, 100, seed_value);
        std::vector<T> input(size);

        for(size_t i = 0; i < size; i++)
        {
            input[i] = test_utils::convert_to_device<T>(input_native[i]);
        }

        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_load_kernel<T><<<grid_size, block_size>>>(device_input, device_output);

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(static_cast<native_T>(output[i]), static_cast<native_T>(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<uint32_t ItemsPerThread, class Type>
__global__
void thread_unroll_kernel(Type* volatile const device_input, Type* device_output)
{
    size_t id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t index = id * ItemsPerThread;

    if(id % 2 == 0)
    {
        hipcub::UnrolledThreadLoad<ItemsPerThread, hipcub::LOAD_VOLATILE>(device_input + index,
                                                                          device_output + index);
    }
    else
    {
        hipcub::UnrolledCopy<ItemsPerThread>(device_input + index, device_output + index);
    }
}

TYPED_TEST(HipcubThreadOperationTests, Unrolled)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::type;
    using native_T = test_utils::convert_to_native_t<T>;

    constexpr uint32_t block_size     = 256;
    constexpr uint32_t grid_size      = 128;
    constexpr uint32_t ItemsPerThread = 4;
    constexpr uint32_t size           = block_size * grid_size * ItemsPerThread;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<native_T> input_native
            = test_utils::get_random_data<native_T>(size, 2, 100, seed_value);
        std::vector<T> input(size);

        for(size_t i = 0; i < size; i++)
        {
            input[i] = test_utils::convert_to_device<T>(input_native[i]);
        }

        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(device_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

        thread_unroll_kernel<ItemsPerThread, T>
            <<<grid_size, block_size>>>(device_input, device_output);
        HIP_CHECK(hipGetLastError());

        // Reading results back
        HIP_CHECK(hipMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(T),
                            hipMemcpyDeviceToHost));

        // Verifying results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(static_cast<native_T>(output[i]), static_cast<native_T>(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class Type>
__global__
void thread_store_kernel(Type* const device_input, Type* device_output)
{
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(index % 7 == hipcub::STORE_DEFAULT)
    {
        hipcub::ThreadStore<hipcub::STORE_DEFAULT>(device_output + index, device_input[index]);
    }
    else if(index % 7 == hipcub::STORE_WB)
    {
        hipcub::ThreadStore<hipcub::STORE_WB>(device_output + index, device_input[index]);
    }
    else if(index % 7 == hipcub::STORE_CG)
    {
        hipcub::ThreadStore<hipcub::STORE_CG>(device_output + index, device_input[index]);
    }
    else if(index % 7 == hipcub::STORE_CS)
    {
        hipcub::ThreadStore<hipcub::STORE_CS>(device_output + index, device_input[index]);
    }
    else if(index % 7 == hipcub::STORE_WT)
    {
        hipcub::ThreadStore<hipcub::STORE_WT>(device_output + index, device_input[index]);
    }
    else if(index % 7 == hipcub::STORE_VOLATILE)
    {
        hipcub::ThreadStore<hipcub::STORE_VOLATILE>(device_output + index, device_input[index]);
    }
    else // index % 7 == 6
    {
        // TODO: For compatibility, should also change this API in rocm backend
#ifdef __HIP_PLATFORM_NVIDIA__
        hipcub::ThreadStoreVolatilePtr(
            device_output + index,
            device_input[index],
            ::cuda::std::integral_constant<bool, std::is_fundamental<Type>::value>{});
#else
        hipcub::ThreadStoreVolatilePtr(device_output + index,
                                       device_input[index],
                                       std::bool_constant<std::is_fundamental<Type>::value>());
#endif
    }
}

TYPED_TEST(HipcubThreadOperationTests, Store)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::type;
    using native_T = test_utils::convert_to_native_t<T>;

    constexpr uint32_t block_size = 256;
    constexpr uint32_t grid_size = 128;
    constexpr uint32_t size = block_size * grid_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<native_T> input_native
            = test_utils::get_random_data<native_T>(size, 2, 100, seed_value);
        std::vector<T> input(size);

        for(size_t i = 0; i < size; i++)
        {
            input[i] = test_utils::convert_to_device<T>(input_native[i]);
        }

        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_store_kernel<T><<<grid_size, block_size>>>(device_input, device_output);

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(static_cast<native_T>(output[i]), static_cast<native_T>(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class Type, int ItemsPerThread>
__global__
void iterate_thread_kernel(Type* const device_input, Type* device_output)
{
    size_t id    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t index = id * ItemsPerThread;

    if(id % 2 == 0)
    {
        hipcub::detail::iterate_thread_store<0, ItemsPerThread>::Dereference(device_output + index,
                                                                             device_input + index);
    }
    else
    {
        hipcub::detail::iterate_thread_store<0, ItemsPerThread>::template Store<
            hipcub::STORE_DEFAULT>(device_output + index, device_input + index);
    }
}

TYPED_TEST(HipcubThreadOperationTests, IterateStore)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::type;
    using native_T = test_utils::convert_to_native_t<T>;

    constexpr uint32_t block_size = 256;
    constexpr uint32_t grid_size  = 128;
    constexpr uint32_t ipt        = 4;
    constexpr uint32_t size       = block_size * grid_size * ipt;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<native_T> input_native
            = test_utils::get_random_data<native_T>(size, 2, 100, seed_value);
        std::vector<T> input(size);

        for(size_t i = 0; i < size; i++)
        {
            input[i] = test_utils::convert_to_device<T>(input_native[i]);
        }

        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(device_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

        iterate_thread_kernel<T, ipt><<<grid_size, block_size>>>(device_input, device_output);

        // Reading results back
        HIP_CHECK(hipMemcpy(output.data(),
                            device_output,
                            output.size() * sizeof(T),
                            hipMemcpyDeviceToHost));

        // Verifying results dereference
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

struct sum_op
{
    template<typename T> HIPCUB_HOST_DEVICE
    T
    operator()(const T& input_1,const T& input_2) const
    {
        return input_1 + input_2;
    }
};

template<class Type, int32_t Length>
__global__
void thread_reduce_kernel(Type* const device_input, Type* device_output)
{
    size_t input_index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * Length;
    size_t output_index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * Length;
    device_output[output_index]
        = hipcub::ThreadReduce<Length>(&device_input[input_index], sum_op());
}

TYPED_TEST(HipcubThreadOperationTests, Reduction)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::type;
    using native_T = test_utils::convert_to_native_t<T>;

    constexpr uint32_t length = 4;
    constexpr uint32_t block_size = 128 / length;
    constexpr uint32_t grid_size = 128;
    constexpr uint32_t size = block_size * grid_size * length;
    sum_op operation;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<native_T> input_native
            = test_utils::get_random_data<native_T>(size, 2, 100, seed_value);
        std::vector<T> input(size);

        for(size_t i = 0; i < size; i++)
        {
            input[i] = test_utils::convert_to_device<T>(input_native[i]);
        }

        std::vector<T> output(size);
        std::vector<T> expected(size);

        // Calculate expected results on host
        for(uint32_t grid_index = 0; grid_index < grid_size; grid_index++)
        {
            for(uint32_t i = 0; i < block_size; i++)
            {
                uint32_t offset = (grid_index * block_size + i) * length;
                T result = T(0);
                for(uint32_t j = 0; j < length; j++)
                {
                    result = operation(result, input[offset + j]);
                }
                expected[offset] = result;
            }
        }

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_reduce_kernel<T, length><<<grid_size, block_size>>>(device_input, device_output);

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output.size(); i+=length)
        {
            ASSERT_EQ(static_cast<native_T>(output[i]), static_cast<native_T>(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class Type, int32_t Length>
__global__
void thread_scan_kernel(Type* const device_input, Type* device_output)
{
    size_t input_index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * Length;
    size_t output_index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * Length;

    hipcub::internal::ThreadScanInclusive<Length>(&device_input[input_index],
                                                  &device_output[output_index],
                                                  sum_op());
}

TYPED_TEST(HipcubThreadOperationTests, Scan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::type;
    using native_T = test_utils::convert_to_native_t<T>;

    constexpr uint32_t length = 4;
    constexpr uint32_t block_size = 128 / length;
    constexpr uint32_t grid_size = 128;
    constexpr uint32_t size = block_size * grid_size * length;
    sum_op operation;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<native_T> input_native
            = test_utils::get_random_data<native_T>(size, 2, 100, seed_value);
        std::vector<T> input(size);

        for(size_t i = 0; i < size; i++)
        {
            input[i] = test_utils::convert_to_device<T>(input_native[i]);
        }

        std::vector<T> output(size);
        std::vector<T> expected(size);

        // Calculate expected results on host
        for(uint32_t grid_index = 0; grid_index < grid_size; grid_index++)
        {
            for(uint32_t i = 0; i < block_size; i++)
            {
                uint32_t offset = (grid_index * block_size + i) * length;
                T result = input[offset];
                expected[offset] = result;
                for(uint32_t j = 1; j < length; j++)
                {
                    result = operation(result, input[offset + j]);
                    expected[offset + j] = result;
                }
            }
        }

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_scan_kernel<T, length><<<grid_size, block_size>>>(device_input, device_output);

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(static_cast<native_T>(output[i]), static_cast<native_T>(expected[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class Type>
__global__
void thread_search_kernel(
    Type* const device_input,
    Type* device_lower_bound_output,
    Type* device_upper_bound_output,
    Type val,
    uint32_t num_items)
{
    size_t input_index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * num_items;
    size_t output_index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    device_lower_bound_output[output_index] =
        hipcub::LowerBound(device_input + input_index, num_items, val);

    device_upper_bound_output[output_index] =
        hipcub::UpperBound(device_input + input_index, num_items, val);
}

TYPED_TEST(HipcubThreadOperationTests, Bounds)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T        = typename TestFixture::type;
    using native_T = test_utils::convert_to_native_t<T>;
    using OffsetT  = uint32_t;

    constexpr uint32_t block_size = 256;
    constexpr uint32_t grid_size = 1;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        uint32_t num_items = test_utils::get_random_value(1, 12, seed_value);
        T val = test_utils::convert_to_device<T>(test_utils::get_random_value(2, 100, seed_value));

        uint32_t size = block_size * grid_size * num_items;

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 100, seed_value);

        std::vector<T> output_lower_bound(size / num_items);
        std::vector<T> output_upper_bound(size / num_items);

        std::vector<T> expected_lower_bound(size / num_items);
        std::vector<T> expected_upper_bound(size / num_items);

        // Calculate expected results on host
        for(uint32_t grid_index = 0; grid_index < grid_size; grid_index++)
        {
            for(uint32_t i = 0; i < block_size; i++)
            {
                uint32_t input_offset = (grid_index * block_size + i) * num_items;
                uint32_t output_offset = grid_index * block_size + i;
                uint32_t local_num_items = num_items;
                OffsetT retval = 0;

                // calculate expected lower bound
                while (local_num_items > 0)
                {
                    OffsetT half = local_num_items >> 1;
                    if(static_cast<native_T>(input[input_offset + retval + half])
                       < static_cast<native_T>(val))
                    {
                        retval = retval + (half + 1);
                        local_num_items = local_num_items - (half + 1);
                    }
                    else
                    {
                        local_num_items = half;
                    }
                }
                expected_lower_bound[output_offset] = retval;

                // calculate expected upper bound
                local_num_items = num_items;
                retval = 0;
                while (local_num_items > 0)
                {
                    OffsetT half = local_num_items >> 1;
                    if(static_cast<native_T>(val)
                       < static_cast<native_T>(input[input_offset + retval + half]))
                    {
                        local_num_items = half;
                    }
                    else
                    {
                        retval = retval + (half + 1);
                        local_num_items = local_num_items - (half + 1);
                    }
                }
                expected_upper_bound[output_offset] = retval;
            }
        }

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(T)));

        T* device_lower_bound_output;
        HIP_CHECK(hipMalloc(&device_lower_bound_output, output_lower_bound.size() * sizeof(T)));

        T* device_upper_bound_output;
        HIP_CHECK(hipMalloc(&device_upper_bound_output, output_upper_bound.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_search_kernel<T>
            <<<grid_size, block_size>>>
                (device_input, device_lower_bound_output, device_upper_bound_output, val, num_items);

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_lower_bound.data(), device_lower_bound_output,
                output_lower_bound.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_upper_bound.data(), device_upper_bound_output,
                output_upper_bound.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < output_lower_bound.size(); i++)
        {
            ASSERT_EQ(static_cast<native_T>(output_lower_bound[i]),
                      static_cast<native_T>(expected_lower_bound[i]));
            ASSERT_EQ(static_cast<native_T>(output_upper_bound[i]),
                      static_cast<native_T>(expected_upper_bound[i]));
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_lower_bound_output));
        HIP_CHECK(hipFree(device_upper_bound_output));
    }
}
