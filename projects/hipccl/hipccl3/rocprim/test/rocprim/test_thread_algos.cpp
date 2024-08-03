/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2024, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <algorithm>
#include <vector>

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

typedef ::testing::Types<params<uint8_t>,
                         params<uint16_t>,
                         params<uint32_t>,
                         params<uint64_t>,
                         params<int>,
                         params<rocprim::half>,
                         params<rocprim::bfloat16>,
                         params<float>,
                         params<double>,
                         params<test_utils::custom_test_type<uint64_t>>,
                         params<test_utils::custom_test_type<double>>>
    ThreadOperationTestParams;

TYPED_TEST_SUITE(RocprimThreadOperationTests, ThreadOperationTestParams);

template<class Type>
__global__
void thread_load_kernel(Type* volatile const device_input, Type* device_output)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated-declarations");
    device_output[index] = rocprim::thread_load<rocprim::load_cg>(device_input + index);
    ROCPRIM_CLANG_SUPPRESS_WARNING_POP
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
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_input),
                                                     input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_output),
                                                     output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_load_kernel<T><<<grid_size, block_size>>>(device_input, device_output);
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<class Type>
__global__
void thread_store_kernel(Type* const device_input, Type* device_output)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated-declarations")
    rocprim::thread_store<rocprim::store_wb>(device_output + index, device_input[index]);
    ROCPRIM_CLANG_SUPPRESS_WARNING_POP
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
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_input),
                                                     input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_output),
                                                     output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_store_kernel<T><<<grid_size, block_size>>>(device_input, device_output);
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

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
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_input),
                                                     input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_output),
                                                     output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_reduce_kernel<T, length><<<grid_size, block_size>>>(device_input, device_output);
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[i], expected[i]));
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
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_input),
                                                     input.size() * sizeof(T)));
        T* device_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_output),
                                                     output.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        thread_scan_kernel<T, length><<<grid_size, block_size>>>(device_input, device_output);
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));
    }
}

template<typename T = int>
struct CoordinateT
{
    T x;
    T y;
};

template<class Type, class OffsetT, class BinaryFunction, OffsetT Length>
__global__ void thread_search_kernel(Type* const    device_input1,
                                     Type* const    device_input2,
                                     OffsetT*       device_output_x,
                                     OffsetT*       device_output_y,
                                     const OffsetT  input1_size,
                                     const OffsetT  input2_size,
                                     BinaryFunction bin_op)
{
    const OffsetT        flat_id         = ::rocprim::detail::block_thread_id<0>();
    const OffsetT        flat_block_id   = ::rocprim::detail::block_id<0>();
    const OffsetT        flat_block_size = ::rocprim::detail::block_size<0>();
    const OffsetT        id              = flat_block_id * flat_block_size + flat_id;
    const OffsetT        partition_id    = id * Length;
    CoordinateT<OffsetT> coord;
    rocprim::merge_path_search(partition_id,
                               device_input1,
                               device_input2,
                               input1_size,
                               input2_size,
                               coord,
                               bin_op);

    device_output_x[id] = coord.x;
    device_output_y[id] = coord.y;
}

template<class T, class OffsetT, class BinaryFunction>
void merge_path_search_test()
{
    static constexpr OffsetT length     = 4;
    static constexpr OffsetT block_size = 128 / length;
    static constexpr OffsetT grid_size  = 128;
    static constexpr OffsetT index_size = block_size * grid_size;
    static constexpr OffsetT size       = index_size * length;

    BinaryFunction bin_op;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input1 = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> input2 = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        std::sort(input1.begin(), input1.end(), bin_op);
        std::sort(input2.begin(), input2.end(), bin_op);

        std::vector<OffsetT> output_x(index_size);
        std::vector<OffsetT> output_y(index_size);

        // Preparing device
        T* device_input1;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_input1),
                                                     input1.size() * sizeof(T)));
        T* device_input2;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_input2),
                                                     input2.size() * sizeof(T)));
        OffsetT* device_output_x;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_output_x),
                                                     output_x.size() * sizeof(OffsetT)));
        OffsetT* device_output_y;
        HIP_CHECK(test_common_utils::hipMallocHelper(reinterpret_cast<void**>(&device_output_y),
                                                     output_y.size() * sizeof(OffsetT)));

        HIP_CHECK(hipMemcpy(device_input1,
                            input1.data(),
                            input1.size() * sizeof(T),
                            hipMemcpyHostToDevice));

        HIP_CHECK(hipMemcpy(device_input2,
                            input2.data(),
                            input2.size() * sizeof(T),
                            hipMemcpyHostToDevice));

        thread_search_kernel<T, OffsetT, BinaryFunction, length>
            <<<grid_size, block_size>>>(device_input1,
                                        device_input2,
                                        device_output_x,
                                        device_output_y,
                                        input1.size(),
                                        input2.size(),
                                        bin_op);
        HIP_CHECK(hipGetLastError());

        // Reading results back
        HIP_CHECK(hipMemcpy(output_x.data(),
                            device_output_x,
                            output_x.size() * sizeof(OffsetT),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(hipMemcpy(output_y.data(),
                            device_output_y,
                            output_y.size() * sizeof(OffsetT),
                            hipMemcpyDeviceToHost));

        std::vector<T> combined_input(2 * size);
        std::merge(input1.begin(),
                   input1.end(),
                   input2.begin(),
                   input2.end(),
                   combined_input.begin(),
                   bin_op);

        OffsetT slice_index = 0;
        for(OffsetT i = 0; i < index_size - 1; i++)
        {
            // Create merged slice based on output of merge_path_search
            std::vector<T> slice_output(length);
            std::merge(input1.begin() + output_x[i],
                       input1.begin() + output_x[i + 1],
                       input2.begin() + output_y[i],
                       input2.begin() + output_y[i + 1],
                       slice_output.begin(),
                       bin_op);

            // Compare to slice of sorted list
            std::vector<T> slice_input(length);
            std::copy(combined_input.begin() + slice_index,
                      combined_input.begin() + slice_index + length,
                      slice_input.begin());

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(slice_input, slice_output));

            slice_index += length;
        }

        HIP_CHECK(hipFree(device_input1));
        HIP_CHECK(hipFree(device_input2));
        HIP_CHECK(hipFree(device_output_x));
        HIP_CHECK(hipFree(device_output_y));
    }
}

TYPED_TEST(RocprimThreadOperationTests, Search)
{
    using T       = typename TestFixture::type;
    using OffsetT = unsigned int;
    merge_path_search_test<T, OffsetT, rocprim::less<T>>();
    merge_path_search_test<T, OffsetT, rocprim::greater<T>>();
}
