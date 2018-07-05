// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

// CUB's implementation of DeviceRunLengthEncode has unused parameters,
// disable the warning because all warnings are threated as errors:
#ifdef __HIP_PLATFORM_NVCC__
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// HIP API
#include <hip/hip_runtime.h>
// hipCUB API
#include <hipcub/hipcub.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

template<
    class Key,
    class Count,
    unsigned int MinSegmentLength,
    unsigned int MaxSegmentLength
>
struct params
{
    using key_type = Key;
    using count_type = Count;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceRunLengthEncode : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<int, int, 1, 1>,
    params<double, int, 3, 5>,
    params<float, int, 1, 10>,
    params<unsigned long long, size_t, 1, 30>,
    params<int, unsigned int, 20, 100>,
    params<float, unsigned long long, 100, 400>,
    params<unsigned int, unsigned int, 200, 600>,
    params<double, int, 100, 2000>,
    params<int, unsigned int, 1000, 5000>,
    params<unsigned int, size_t, 2048, 2048>,
    params<unsigned int, unsigned int, 1000, 50000>,
    params<unsigned long long, unsigned long long, 100000, 100000>
> Params;

TYPED_TEST_CASE(HipcubDeviceRunLengthEncode, Params);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        100000,
        (1 << 16) - 1220, (1 << 21) - 76543
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 100000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

TYPED_TEST(HipcubDeviceRunLengthEncode, Encode)
{
    using key_type = typename TestFixture::params::key_type;
    using count_type = typename TestFixture::params::count_type;
    using key_distribution_type = typename std::conditional<
        std::is_floating_point<key_type>::value,
        std::uniform_real_distribution<key_type>,
        std::uniform_int_distribution<key_type>
    >::type;

    const bool debug_synchronous = false;

    const std::vector<size_t> sizes = get_sizes();

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipStream_t stream = 0; // default

        // Generate data and calculate expected results
        std::vector<key_type> unique_expected;
        std::vector<count_type> counts_expected;
        size_t runs_count_expected = 0;

        std::vector<key_type> input(size);
        key_distribution_type key_delta_dis(1, 5);
        std::uniform_int_distribution<size_t> key_count_dis(
            TestFixture::params::min_segment_length,
            TestFixture::params::max_segment_length
        );
        std::vector<count_type> values_input = test_utils::get_random_data<count_type>(size, 0, 100);

        size_t offset = 0;
        key_type current_key = key_distribution_type(0, 100)(gen);
        while(offset < size)
        {
            size_t key_count = key_count_dis(gen);
            current_key += key_delta_dis(gen);

            const size_t end = std::min(size, offset + key_count);
            key_count = end - offset;
            for(size_t i = offset; i < end; i++)
            {
                input[i] = current_key;
            }

            unique_expected.push_back(current_key);
            runs_count_expected++;
            counts_expected.push_back(key_count);

            offset += key_count;
        }

        key_type * d_input;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                size * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );

        key_type * d_unique_output;
        count_type * d_counts_output;
        count_type * d_runs_count_output;
        HIP_CHECK(hipMalloc(&d_unique_output, runs_count_expected * sizeof(key_type)));
        HIP_CHECK(hipMalloc(&d_counts_output, runs_count_expected * sizeof(count_type)));
        HIP_CHECK(hipMalloc(&d_runs_count_output, sizeof(count_type)));

        size_t temporary_storage_bytes = 0;

        HIP_CHECK(
            hipcub::DeviceRunLengthEncode::Encode(
                nullptr, temporary_storage_bytes,
                d_input,
                d_unique_output, d_counts_output, d_runs_count_output,
                size,
                stream, debug_synchronous
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            hipcub::DeviceRunLengthEncode::Encode(
                d_temporary_storage, temporary_storage_bytes,
                d_input,
                d_unique_output, d_counts_output, d_runs_count_output,
                size,
                stream, debug_synchronous
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<key_type> unique_output(runs_count_expected);
        std::vector<count_type> counts_output(runs_count_expected);
        std::vector<count_type> runs_count_output(1);
        HIP_CHECK(
            hipMemcpy(
                unique_output.data(), d_unique_output,
                runs_count_expected * sizeof(key_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(
            hipMemcpy(
                counts_output.data(), d_counts_output,
                runs_count_expected * sizeof(count_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(
            hipMemcpy(
                runs_count_output.data(), d_runs_count_output,
                sizeof(count_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_unique_output));
        HIP_CHECK(hipFree(d_counts_output));
        HIP_CHECK(hipFree(d_runs_count_output));

        // Validating results

        ASSERT_EQ(runs_count_output[0], static_cast<count_type>(runs_count_expected));

        for(size_t i = 0; i < runs_count_expected; i++)
        {
            ASSERT_EQ(unique_output[i], unique_expected[i]);
            ASSERT_EQ(counts_output[i], counts_expected[i]);
        }
    }
}

TYPED_TEST(HipcubDeviceRunLengthEncode, NonTrivialRuns)
{
    using key_type = typename TestFixture::params::key_type;
    using count_type = typename TestFixture::params::count_type;
    using offset_type = typename TestFixture::params::count_type;
    using key_distribution_type = typename std::conditional<
        std::is_floating_point<key_type>::value,
        std::uniform_real_distribution<key_type>,
        std::uniform_int_distribution<key_type>
    >::type;

    const bool debug_synchronous = false;

    const std::vector<size_t> sizes = get_sizes();

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipStream_t stream = 0; // default

        // Generate data and calculate expected results
        std::vector<offset_type> offsets_expected;
        std::vector<count_type> counts_expected;
        size_t runs_count_expected = 0;

        std::vector<key_type> input(size);
        key_distribution_type key_delta_dis(1, 5);
        std::uniform_int_distribution<size_t> key_count_dis(
            TestFixture::params::min_segment_length,
            TestFixture::params::max_segment_length
        );
        std::bernoulli_distribution is_trivial_dis(0.1);
        std::vector<count_type> values_input = test_utils::get_random_data<count_type>(size, 0, 100);

        size_t offset = 0;
        key_type current_key = key_distribution_type(0, 100)(gen);
        while(offset < size)
        {
            size_t key_count;
            if(TestFixture::params::min_segment_length == 1 && is_trivial_dis(gen))
            {
                // Increased probability of trivial runs for long segments
                key_count = 1;
            }
            else
            {
                key_count = key_count_dis(gen);
            }
            current_key += key_delta_dis(gen);

            const size_t end = std::min(size, offset + key_count);
            key_count = end - offset;
            for(size_t i = offset; i < end; i++)
            {
                input[i] = current_key;
            }

            if(key_count > 1)
            {
                offsets_expected.push_back(offset);
                runs_count_expected++;
                counts_expected.push_back(key_count);
            }

            offset += key_count;
        }

        key_type * d_input;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                size * sizeof(key_type),
                hipMemcpyHostToDevice
            )
        );

        offset_type * d_offsets_output;
        count_type * d_counts_output;
        count_type * d_runs_count_output;
        HIP_CHECK(hipMalloc(&d_offsets_output, std::max<size_t>(1, runs_count_expected) * sizeof(offset_type)));
        HIP_CHECK(hipMalloc(&d_counts_output, std::max<size_t>(1, runs_count_expected) * sizeof(count_type)));
        HIP_CHECK(hipMalloc(&d_runs_count_output, sizeof(count_type)));

        size_t temporary_storage_bytes = 0;

        HIP_CHECK(
            hipcub::DeviceRunLengthEncode::NonTrivialRuns(
                nullptr, temporary_storage_bytes,
                d_input,
                d_offsets_output, d_counts_output, d_runs_count_output,
                size,
                stream, debug_synchronous
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0U);

        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            hipcub::DeviceRunLengthEncode::NonTrivialRuns(
                d_temporary_storage, temporary_storage_bytes,
                d_input,
                d_offsets_output, d_counts_output, d_runs_count_output,
                size,
                stream, debug_synchronous
            )
        );

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<offset_type> offsets_output(runs_count_expected);
        std::vector<count_type> counts_output(runs_count_expected);
        std::vector<count_type> runs_count_output(1);
        if(runs_count_expected > 0)
        {
            HIP_CHECK(
                hipMemcpy(
                    offsets_output.data(), d_offsets_output,
                    runs_count_expected * sizeof(offset_type),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    counts_output.data(), d_counts_output,
                    runs_count_expected * sizeof(count_type),
                    hipMemcpyDeviceToHost
                )
            );
        }
        HIP_CHECK(
            hipMemcpy(
                runs_count_output.data(), d_runs_count_output,
                sizeof(count_type),
                hipMemcpyDeviceToHost
            )
        );

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_offsets_output));
        HIP_CHECK(hipFree(d_counts_output));
        HIP_CHECK(hipFree(d_runs_count_output));

        // Validating results

        ASSERT_EQ(runs_count_output[0], static_cast<count_type>(runs_count_expected));

        for(size_t i = 0; i < runs_count_expected; i++)
        {
            ASSERT_EQ(offsets_output[i], offsets_expected[i]);
            ASSERT_EQ(counts_output[i], counts_expected[i]);
        }
    }
}
