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

#include "../common_test_header.hpp"

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_device_ptr.hpp"

#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"

#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/thread/thread_load.hpp>
#include <rocprim/thread/thread_reduce.hpp>
#include <rocprim/thread/thread_scan.hpp>
#include <rocprim/thread/thread_search.hpp>
#include <rocprim/thread/thread_store.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <stdint.h>
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

using ThreadOperationTestParams
    = ::testing::Types<params<uint8_t>,
                       params<uint16_t>,
                       params<uint32_t>,
                       params<uint64_t>,
                       params<int>,
                       params<rocprim::half>,
                       params<rocprim::bfloat16>,
                       params<float>,
                       params<double>,
                       params<common::custom_type<uint64_t, uint64_t, true>>,
                       params<common::custom_type<double, double, true>>
#if ROCPRIM_HAS_INT128_SUPPORT
                       ,
                       params<rocprim::uint128_t>
#endif
                       >;

TYPED_TEST_SUITE(RocprimThreadOperationTests, ThreadOperationTestParams);

template<class Type>
__global__
void thread_load_kernel(Type* volatile const device_input, Type* device_output)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index % rocprim::load_count == rocprim::load_default)
    {
        device_output[index] = rocprim::thread_load(device_input + index);
    }
    else if(index % rocprim::load_count == rocprim::load_ca)
    {
        device_output[index] = rocprim::thread_load<rocprim::load_ca>(device_input + index);
    }
    else if(index % rocprim::load_count == rocprim::load_cg)
    {
        device_output[index] = rocprim::thread_load<rocprim::load_cg>(device_input + index);
    }
    else if(index % rocprim::load_count == rocprim::load_nontemporal)
    {
        device_output[index]
            = rocprim::thread_load<rocprim::load_nontemporal>(device_input + index);
    }
    else if(index % rocprim::load_count == rocprim::load_cv)
    {
        device_output[index] = rocprim::thread_load<rocprim::load_cv>(device_input + index);
    }
    else if(index % rocprim::load_count == rocprim::load_ldg)
    {
        device_output[index] = rocprim::thread_load<rocprim::load_ldg>(device_input + index);
    }
    else // index % rocprim::load_count == rocprim::load_volatile
    {
        device_output[index] = rocprim::thread_load<rocprim::load_volatile>(device_input + index);
    }
}

TYPED_TEST(RocprimThreadOperationTests, Load)
{
    using T = typename TestFixture::type;
    static constexpr uint32_t block_size = 256;
    static constexpr uint32_t grid_size = 128;
    static constexpr uint32_t size = block_size * grid_size;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(input.size());

        thread_load_kernel<T><<<grid_size, block_size>>>(device_input.get(), device_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results back
        output = device_output.load();

        // Verifying results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }
}

template<uint32_t ItemsPerThread, class Type>
__global__
void thread_copy_unroll_kernel(Type* device_input, Type* device_output)
{
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t index     = thread_id * ItemsPerThread;

    if(thread_id % 2 == 0)
    {
        rocprim::unrolled_copy<ItemsPerThread>(device_input + index, device_output + index);
    }
    else
    {
        rocprim::unrolled_thread_load<ItemsPerThread, rocprim::load_default>(device_input + index,
                                                                             device_output + index);
    }
}

TYPED_TEST(RocprimThreadOperationTests, CopyUnroll)
{
    using T                                  = typename TestFixture::type;
    static constexpr uint32_t block_size     = 256;
    static constexpr uint32_t grid_size      = 128;
    static constexpr uint32_t ItemsPerThread = 4;
    static constexpr uint32_t size           = block_size * grid_size * ItemsPerThread;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(input.size());

        thread_copy_unroll_kernel<ItemsPerThread, T>
            <<<grid_size, block_size>>>(device_input.get(), device_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results back
        output = device_output.load();

        // Verifying results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }
}

template<class Type>
__global__
void thread_store_kernel(Type* const device_input, Type* device_output)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index % rocprim::store_count == rocprim::store_default)
    {
        rocprim::thread_store(device_output + index, device_input[index]);
    }
    else if(index % rocprim::store_count == rocprim::store_wb)
    {
        rocprim::thread_store<rocprim::store_wb>(device_output + index, device_input[index]);
    }
    else if(index % rocprim::store_count == rocprim::store_cg)
    {
        rocprim::thread_store<rocprim::store_cg>(device_output + index, device_input[index]);
    }
    else if(index % rocprim::store_count == rocprim::store_nontemporal)
    {
        rocprim::thread_store<rocprim::store_nontemporal>(device_output + index,
                                                          device_input[index]);
    }
    else if(index % rocprim::store_count == rocprim::store_wt)
    {
        rocprim::thread_store<rocprim::store_wt>(device_output + index, device_input[index]);
    }
    else // index % rocprim::store_count == rocprim::store_volatile
    {
        rocprim::thread_store<rocprim::store_volatile>(device_output + index, device_input[index]);
    }
}

TYPED_TEST(RocprimThreadOperationTests, StoreNontemporal)
{
    using T                              = typename TestFixture::type;
    static constexpr uint32_t block_size = 256;
    static constexpr uint32_t grid_size  = 128;
    static constexpr uint32_t size       = block_size * grid_size;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);

        // Calculate expected results on host
        std::vector<T> expected = input;

        // Preparing device
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(input.size());

        thread_store_kernel<T><<<grid_size, block_size>>>(device_input.get(), device_output.get());
        HIP_CHECK(hipGetLastError());

        // Reading results back
        output = device_output.load();

        // Verifying results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
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
    size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;

    device_output[index] = rocprim::thread_reduce<Length>(&device_input[index], sum_op());
}

template<class Type, int32_t Length>
__global__
void thread_reduce_kernel_array_prefix(Type* const device_input, Type* device_output, Type prefix)
{
    size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;

    Type device_input_array[Length];
    for(int32_t i = 0; i < Length; i++)
    {
        device_input_array[i] = device_input[index + i];
    }

    device_output[index] = rocprim::thread_reduce(device_input_array, sum_op(), prefix);
}

TYPED_TEST(RocprimThreadOperationTests, Reduction)
{
    using T                              = typename TestFixture::type;
    static constexpr uint32_t length     = 4;
    static constexpr uint32_t block_size = 128 / length;
    static constexpr uint32_t grid_size  = 128;
    static constexpr uint32_t size       = block_size * grid_size * length;
    sum_op                    operation;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(size);

        for(uint32_t prefix = 0; prefix < 2; prefix++)
        {
            // Calculate expected results on host
            for(uint32_t grid_index = 0; grid_index < grid_size; grid_index++)
            {
                for(uint32_t i = 0; i < block_size; i++)
                {
                    uint32_t offset = (grid_index * block_size + i) * length;
                    T        result = T(prefix);
                    for(uint32_t j = 0; j < length; j++)
                    {
                        result = operation(result, input[offset + j]);
                    }
                    expected[offset] = result;
                }
            }

            // Preparing device
            common::device_ptr<T> device_input(input);
            common::device_ptr<T> device_output(input.size());
            if(prefix == 0)
            {
                thread_reduce_kernel<T, length>
                    <<<grid_size, block_size>>>(device_input.get(), device_output.get());
            }
            else
            {
                thread_reduce_kernel_array_prefix<T, length>
                    <<<grid_size, block_size>>>(device_input.get(), device_output.get(), T(prefix));
            }

            HIP_CHECK(hipGetLastError());

            // Reading results back
            output = device_output.load();

            // Verifying results
            for(size_t i = 0; i < output.size(); i += length)
            {
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output[i], expected[i]));
            }
        }
    }
}

template<class Type, int32_t Length>
__global__
void thread_scan_inclusive_kernel(Type* const device_input, Type* device_output)
{
    size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;

    rocprim::thread_scan_inclusive<Length>(&device_input[index], &device_output[index], sum_op());
}

template<class Type, int32_t Length>
__global__
void thread_scan_inclusive_kernel_array_prefix(Type* const device_input,
                                               Type*       device_output,
                                               Type        prefix)
{
    size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;

    Type device_input_array[Length];
    for(int32_t i = 0; i < Length; i++)
    {
        device_input_array[i] = device_input[index + i];
    }

    rocprim::thread_scan_inclusive<Length>(&device_input[index],
                                           &device_output[index],
                                           sum_op(),
                                           prefix,
                                           threadIdx.x > 0);
}

TYPED_TEST(RocprimThreadOperationTests, ScanInclusive)
{
    using T                              = typename TestFixture::type;
    static constexpr uint32_t length     = 4;
    static constexpr uint32_t block_size = 128 / length;
    static constexpr uint32_t grid_size  = 128;
    static constexpr uint32_t size       = block_size * grid_size * length;
    sum_op                    operation;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(size);

        for(uint32_t prefix = 0; prefix < 2; prefix++)
        {
            // Calculate expected results on host
            for(uint32_t grid_index = 0; grid_index < grid_size; grid_index++)
            {
                for(uint32_t i = 0; i < block_size; i++)
                {
                    uint32_t offset = (grid_index * block_size + i) * length;
                    // Skip the first value of every block
                    T result         = i > 0 ? T(prefix) : T(0);
                    expected[offset] = result;
                    for(uint32_t j = 0; j < length; j++)
                    {
                        result               = operation(result, input[offset + j]);
                        expected[offset + j] = result;
                    }
                }
            }

            // Preparing device
            common::device_ptr<T> device_input(input);
            common::device_ptr<T> device_output(input.size());
            if(prefix == 0)
            {
                thread_scan_inclusive_kernel<T, length>
                    <<<grid_size, block_size>>>(device_input.get(), device_output.get());
            }
            else
            {
                thread_scan_inclusive_kernel_array_prefix<T, length>
                    <<<grid_size, block_size>>>(device_input.get(), device_output.get(), T(prefix));
            }
            HIP_CHECK(hipGetLastError());

            // Reading results back
            output = device_output.load();

            // Verifying results
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        }
    }
}

template<class Type, int32_t Length>
__global__
void thread_scan_exclusive_kernel(Type* const device_input, Type* device_output, Type prefix)
{
    size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * Length;

    Type device_input_array[Length];
    for(int32_t i = 0; i < Length; i++)
    {
        device_input_array[i] = device_input[index + i];
    }

    rocprim::thread_scan_exclusive<Length>(&device_input[index],
                                           &device_output[index],
                                           sum_op(),
                                           prefix,
                                           threadIdx.x > 0);
}

TYPED_TEST(RocprimThreadOperationTests, ScanExclusive)
{
    using T                              = typename TestFixture::type;
    static constexpr uint32_t length     = 4;
    static constexpr uint32_t block_size = 128 / length;
    static constexpr uint32_t grid_size  = 128;
    static constexpr uint32_t size       = block_size * grid_size * length;
    sum_op                    operation;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 2, 200, seed_value);
        std::vector<T> output(size);
        std::vector<T> expected(size);
        T              prefix = test_utils::get_random_value<T>(2, 200, seed_value);

        // Calculate expected results on host
        for(uint32_t grid_index = 0; grid_index < grid_size; grid_index++)
        {
            for(uint32_t i = 0; i < block_size; i++)
            {
                uint32_t offset = (grid_index * block_size + i) * length;

                // Initialize the scan with the first value and apply prefix if needed
                T inclusive = input[offset];
                inclusive   = i > 0 ? operation(prefix, inclusive) : inclusive;

                // First output value is the prefix
                expected[offset] = prefix;

                // Exclusive value to track for the next element
                T exclusive = inclusive;

                // Perform the rest of the scan
                for(uint32_t j = 1; j < length; j++)
                {
                    inclusive            = operation(exclusive, input[offset + j]);
                    expected[offset + j] = exclusive;
                    exclusive            = inclusive;
                }
            }
        }

        // Preparing device
        common::device_ptr<T> device_input(input);
        common::device_ptr<T> device_output(input.size());

        thread_scan_exclusive_kernel<T, length>
            <<<grid_size, block_size>>>(device_input.get(), device_output.get(), prefix);

        HIP_CHECK(hipGetLastError());

        // Reading results back
        output = device_output.load();

        // Verifying results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
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

template<class Type, class OffsetT, class BinaryFunction>
__global__
void thread_search_out_of_bounds_kernel(Type* const    device_input1,
                                        Type* const    device_input2,
                                        OffsetT*       device_output_x,
                                        OffsetT*       device_output_y,
                                        const OffsetT  input1_size,
                                        const OffsetT  input2_size,
                                        BinaryFunction bin_op)
{
    const OffsetT        partition_id = input1_size + input2_size + 1;
    CoordinateT<OffsetT> coord;
    rocprim::merge_path_search(partition_id,
                               device_input1,
                               device_input2,
                               input1_size,
                               input2_size,
                               coord,
                               bin_op);

    *device_output_x = coord.x;
    *device_output_y = coord.y;
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

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input1 = test_utils::get_random_data_wrapped<T>(size, 2, 200, seed_value);
        std::vector<T> input2 = test_utils::get_random_data_wrapped<T>(size, 2, 200, seed_value);

        std::sort(input1.begin(), input1.end(), bin_op);
        std::sort(input2.begin(), input2.end(), bin_op);

        std::vector<OffsetT> output_x;
        std::vector<OffsetT> output_y;
        OffsetT              output_oob_x, output_oob_y;

        // Preparing device
        common::device_ptr<T>       device_input1(input1);
        common::device_ptr<T>       device_input2(input2);
        common::device_ptr<OffsetT> device_output_x(index_size);
        common::device_ptr<OffsetT> device_output_y(index_size);
        common::device_ptr<OffsetT> device_output_oob_x(1);
        common::device_ptr<OffsetT> device_output_oob_y(1);

        thread_search_kernel<T, OffsetT, BinaryFunction, length>
            <<<grid_size, block_size>>>(device_input1.get(),
                                        device_input2.get(),
                                        device_output_x.get(),
                                        device_output_y.get(),
                                        input1.size(),
                                        input2.size(),
                                        bin_op);
        HIP_CHECK(hipGetLastError());

        thread_search_out_of_bounds_kernel<T, OffsetT, BinaryFunction>
            <<<grid_size, block_size>>>(device_input1.get(),
                                        device_input2.get(),
                                        device_output_oob_x.get(),
                                        device_output_oob_y.get(),
                                        input1.size(),
                                        input2.size(),
                                        bin_op);
        HIP_CHECK(hipGetLastError());

        // Reading results back
        output_x     = device_output_x.load();
        output_y     = device_output_y.load();
        output_oob_x = device_output_oob_x.load()[0];
        output_oob_y = device_output_oob_y.load()[0];

        std::vector<T> combined_input(2 * size);
        std::merge(input1.begin(),
                   input1.end(),
                   input2.begin(),
                   input2.end(),
                   combined_input.begin(),
                   bin_op);

        ASSERT_EQ(output_oob_x, input1.size());
        ASSERT_EQ(output_oob_y, input2.size());

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
    }
}

TYPED_TEST(RocprimThreadOperationTests, Search)
{
    using T       = typename TestFixture::type;
    using OffsetT = unsigned int;
    merge_path_search_test<T, OffsetT, rocprim::less<T>>();
    merge_path_search_test<T, OffsetT, rocprim::greater<T>>();
}

template<class Type, class OffsetT, OffsetT Length>
__global__
void upper_lower_bound_kernel(Type* const device_input, OffsetT* device_output, Type value)
{
    size_t index_input  = (blockIdx.x * blockDim.x + threadIdx.x) * Length;
    size_t index_output = (blockIdx.x * blockDim.x + threadIdx.x) * 3;

    device_output[index_output] = rocprim::lower_bound(&device_input[index_input], Length, value);
    device_output[index_output + 1]
        = rocprim::upper_bound(&device_input[index_input], Length, value);
    device_output[index_output + 2]
        = rocprim::static_upper_bound<Length>(&device_input[index_input], Length, value);
}

TYPED_TEST(RocprimThreadOperationTests, UpperLowerBound)
{
    using T                              = typename TestFixture::type;
    using OffsetT                        = uint32_t;
    static constexpr uint32_t length     = 16;
    static constexpr uint32_t block_size = 256 / length;
    static constexpr uint32_t grid_size  = 128;
    static constexpr uint32_t size       = block_size * grid_size * length;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 10, seed_value);
        std::vector<OffsetT> output(block_size * grid_size * 3);
        std::vector<OffsetT> expected(block_size * grid_size * 3);
        T                    value = test_utils::get_random_value<T>(1, 10, seed_value);

        // Calculate expected results on host
        for(uint32_t grid_index = 0; grid_index < grid_size; grid_index++)
        {
            for(uint32_t i = 0; i < block_size; i++)
            {
                uint32_t offset          = (grid_index * block_size + i) * length;
                uint32_t offset_expected = (grid_index * block_size + i) * 3;

                expected[offset_expected] = std::lower_bound(input.begin() + offset,
                                                             input.begin() + offset + length,
                                                             value)
                                            - (input.begin() + offset);
                expected[offset_expected + 1] = std::upper_bound(input.begin() + offset,
                                                                 input.begin() + offset + length,
                                                                 value)
                                                - (input.begin() + offset);

                expected[offset_expected + 2] = expected[offset_expected + 1];
            }
        }

        // Preparing device
        common::device_ptr<T>       device_input(input);
        common::device_ptr<OffsetT> device_output(output.size());

        upper_lower_bound_kernel<T, OffsetT, length>
            <<<grid_size, block_size>>>(device_input.get(), device_output.get(), value);

        HIP_CHECK(hipGetLastError());

        // Reading results back
        output = device_output.load();

        // Verifying results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }
}
