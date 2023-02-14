/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2023, Advanced Micro Devices, Inc.  All rights reserved.
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

#include "rocprim/thread/thread_load.hpp"
#include "rocprim/thread/thread_store.hpp"
#include "rocprim/thread/thread_reduce.hpp"
#include "rocprim/thread/thread_scan.hpp"
#include "rocprim/thread/thread_search.hpp"

#include "../common_test_header.hpp"
#include "test_utils.hpp"

template<class T>
struct params
{
    using type = T;
};

template<class Params>
class RocprimThreadOperationTests : public ::testing::Test
{
public:
    using type = typename Params::type;
};

typedef ::testing::Types<
    params<uint8_t>,
    params<uint16_t>,
    params<uint32_t>,
    params<uint64_t>,
    params<test_utils::custom_test_type<uint64_t>>,
    params<test_utils::custom_test_type<double>>
> ThreadOperationTestParams;

TYPED_TEST_SUITE(RocprimThreadOperationTests, ThreadOperationTestParams);

template<class Type>
__global__
void thread_load_kernel(Type* volatile const device_input, Type* device_output)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    device_output[index] = rocprim::thread_load<rocprim::load_cg>(device_input + index);
}

TYPED_TEST(RocprimThreadOperationTests, Load)
{
    using T = typename TestFixture::type;
    static constexpr uint32_t block_size = 256;
    static constexpr uint32_t grid_size = 128;
    static constexpr uint32_t size = block_size * grid_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(thread_load_kernel<T>),
            grid_size, block_size, 0, 0,
            device_input, device_output
        );
        HIP_CHECK(hipGetLastError());

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

template<class Type>
__global__
void thread_store_kernel(Type* const device_input, Type* device_output)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    rocprim::thread_store<rocprim::store_wb>(device_output + index, device_input[index]);
}

TYPED_TEST(RocprimThreadOperationTests, Store)
{
    using T = typename TestFixture::type;
    static constexpr uint32_t block_size = 256;
    static constexpr uint32_t grid_size = 128;
    static constexpr uint32_t size = block_size * grid_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        T* device_input;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(thread_store_kernel<T>),
            grid_size, block_size, 0, 0,
            device_input, device_output
        );
        HIP_CHECK(hipGetLastError());

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
    template<typename T> ROCPRIM_HOST_DEVICE
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
    size_t input_index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;
    size_t output_index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;
    device_output[output_index] = rocprim::thread_reduce<Length>(&device_input[input_index], sum_op());
}

TYPED_TEST(RocprimThreadOperationTests, Reduction)
{
    using T = typename TestFixture::type;
    static constexpr uint32_t length = 4;
    static constexpr uint32_t block_size = 128 / length;
    static constexpr uint32_t grid_size = 128;
    static constexpr uint32_t size = block_size * grid_size * length;
    sum_op operation;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 200, seed_value);
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
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(thread_reduce_kernel<T, length>),
            grid_size, block_size, 0, 0,
            device_input, device_output
        );
        HIP_CHECK(hipGetLastError());

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
    size_t input_index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;
    size_t output_index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;

    rocprim::thread_scan_inclusive<Length>(&device_input[input_index],
                                                  &device_output[output_index],
                                                  sum_op());
}

TYPED_TEST(RocprimThreadOperationTests, Scan)
{
    using T = typename TestFixture::type;
    static constexpr uint32_t length = 4;
    static constexpr uint32_t block_size = 128 / length;
    static constexpr uint32_t grid_size = 128;
    static constexpr uint32_t size = block_size * grid_size * length;
    sum_op operation;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 200, seed_value);
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
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(thread_scan_kernel<T, length>),
            grid_size, block_size, 0, 0,
            device_input, device_output
        );
        HIP_CHECK(hipGetLastError());

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
