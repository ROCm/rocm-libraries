/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2021, Advanced Micro Devices, Inc.  All rights reserved.
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


#include "hipcub/thread/thread_load.hpp"
#include "hipcub/thread/thread_store.hpp"
#include "hipcub/thread/thread_reduce.hpp"
#include "hipcub/thread/thread_scan.hpp"
#include "hipcub/thread/thread_search.hpp"

#include "common_test_header.hpp"

template<
    class T,
    hipcub::CacheLoadModifier LoadModifier,
    hipcub::CacheStoreModifier StoreModifier
>
struct params
{
    using type = T;
    static constexpr hipcub::CacheLoadModifier load_modifier = LoadModifier;
    static constexpr hipcub::CacheStoreModifier store_modifier = StoreModifier;
};

template<class Params>
class HipcubThreadOperationTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr hipcub::CacheLoadModifier load_modifier = Params::load_modifier;
    static constexpr hipcub::CacheStoreModifier store_modifier = Params::store_modifier;
};

typedef ::testing::Types<
    params<int8_t, hipcub::LOAD_CA, hipcub::STORE_WB>,
    params<int16_t, hipcub::LOAD_CA, hipcub::STORE_WB>,
    params<uint8_t, hipcub::LOAD_CA, hipcub::STORE_WB>,
    params<uint16_t, hipcub::LOAD_CA, hipcub::STORE_WB>,
    params<uint32_t, hipcub::LOAD_CA, hipcub::STORE_WB>,
    params<uint64_t, hipcub::LOAD_CA, hipcub::STORE_WB>,

    params<int8_t, hipcub::LOAD_CG, hipcub::STORE_CG>,
    params<int16_t, hipcub::LOAD_CG, hipcub::STORE_CG>,
    params<uint8_t, hipcub::LOAD_CG, hipcub::STORE_CG>,
    params<uint16_t, hipcub::LOAD_CG, hipcub::STORE_CG>,
    params<uint32_t, hipcub::LOAD_CG, hipcub::STORE_CG>,
    params<uint64_t, hipcub::LOAD_CG, hipcub::STORE_CG>,

    params<int8_t, hipcub::LOAD_CV, hipcub::STORE_WT>,
    params<int16_t, hipcub::LOAD_CV, hipcub::STORE_WT>,
    params<uint8_t, hipcub::LOAD_CV, hipcub::STORE_WT>,
    params<uint16_t, hipcub::LOAD_CV, hipcub::STORE_WT>,
    params<uint32_t, hipcub::LOAD_CV, hipcub::STORE_WT>,
    params<uint64_t, hipcub::LOAD_CV, hipcub::STORE_WT>,
    params<test_utils::custom_test_type<uint64_t>, hipcub::LOAD_CV, hipcub::STORE_WB>,
    params<test_utils::custom_test_type<double>, hipcub::LOAD_CV, hipcub::STORE_WB>
> ThreadOperationTestParams;

TYPED_TEST_SUITE(HipcubThreadOperationTests, ThreadOperationTestParams);

template<class Type, hipcub::CacheLoadModifier Modifier>
__global__
void thread_load_kernel(Type* volatile const device_input, Type* device_output)
{
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    device_output[index] = hipcub::ThreadLoad<Modifier>(device_input + index);
}

TYPED_TEST(HipcubThreadOperationTests, Load)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr hipcub::CacheLoadModifier Modifier = TestFixture::load_modifier;

    constexpr uint32_t block_size = 256;
    constexpr uint32_t grid_size = 128;
    constexpr uint32_t size = block_size * grid_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 100, seed_value);
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

        thread_load_kernel<T, Modifier><<<grid_size, block_size>>>(device_input, device_output);

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
            ASSERT_EQ(output[i], expected[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class Type, hipcub::CacheStoreModifier Modifier>
__global__
void thread_store_kernel(Type* const device_input, Type* device_output)
{
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    hipcub::ThreadStore<Modifier>(device_output + index, device_input[index]);
}

TYPED_TEST(HipcubThreadOperationTests, Store)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr hipcub::CacheStoreModifier Modifier = TestFixture::store_modifier;
    constexpr uint32_t block_size = 256;
    constexpr uint32_t grid_size = 128;
    constexpr uint32_t size = block_size * grid_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 100, seed_value);
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

        thread_store_kernel<T, Modifier><<<grid_size, block_size>>>(device_input, device_output);

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
            ASSERT_EQ(output[i], expected[i]);
        }

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
    device_output[output_index] = hipcub::internal::ThreadReduce<Length>(&device_input[input_index], sum_op());
}

TYPED_TEST(HipcubThreadOperationTests, Reduction)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
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
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 100, seed_value);
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
        //std::vector<T> expected = input;

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
            //std::cout << "i: " << i << " " << expected[i] << " - " << output[i] << std::endl;
            ASSERT_EQ(output[i], expected[i]);
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

    using T = typename TestFixture::type;
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
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 100, seed_value);
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
            //std::cout << "i: " << i << " " << input[i] << " - " << expected[i] << " - " << output[i] << std::endl;
            ASSERT_EQ(output[i], expected[i]);
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

    using T = typename TestFixture::type;
    using OffsetT = uint32_t;
    constexpr uint32_t block_size = 256;
    constexpr uint32_t grid_size = 1;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        uint32_t num_items = test_utils::get_random_value(1, 12, seed_value);
        T val = test_utils::get_random_value(2, 100, seed_value);

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
                    if (input[input_offset + retval + half] < val)
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
                    if (val < input[input_offset + retval + half])
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
            ASSERT_EQ(output_lower_bound[i], expected_lower_bound[i]);
            ASSERT_EQ(output_upper_bound[i], expected_upper_bound[i]);
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_lower_bound_output));
        HIP_CHECK(hipFree(device_upper_bound_output));
    }
}
