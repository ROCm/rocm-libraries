// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifdef __HIP_PLATFORM_NVIDIA__
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "common_test_header.hpp"

// hipcub API
#include <hipcub/device/device_run_length_encode.hpp>

#include "test_utils_data_generation.hpp"

template<class Key,
         class Count,
         unsigned int MinSegmentLength,
         unsigned int MaxSegmentLength,
         bool         UseGraphs = false>
struct params
{
    using key_type                                   = Key;
    using count_type                                 = Count;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    static constexpr bool         use_graphs         = UseGraphs;
};

template<class Params>
class HipcubDeviceRunLengthEncode : public ::testing::Test
{
public:
    using params = Params;
};

using Params = ::testing::Types<params<int, int, 1, 1>,
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
                                params<unsigned long long, unsigned long long, 100000, 100000>,
                                // Test graph capture
                                params<int, int, 1, 1, true>,
                                params<float, int, 1, 10, true>>;

TYPED_TEST_SUITE(HipcubDeviceRunLengthEncode, Params);

TYPED_TEST(HipcubDeviceRunLengthEncode, Encode)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type   = typename TestFixture::params::key_type;
    using count_type = typename TestFixture::params::count_type;
    using key_distribution_type =
        typename std::conditional<std::is_floating_point<key_type>::value,
                                  std::uniform_real_distribution<key_type>,
                                  std::uniform_int_distribution<key_type>>::type;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> empty_sizes{0, 1};
        auto                      sizes = test_utils::get_sizes(seed_value);
        sizes.insert(std::end(sizes), std::begin(empty_sizes), std::end(empty_sizes));

        for(size_t size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data and calculate expected results
            std::vector<key_type>   unique_expected;
            std::vector<count_type> counts_expected;
            size_t                  runs_count_expected = 0;

            std::vector<key_type>                 input(size);
            key_distribution_type                 key_delta_dis(1, 5);
            std::uniform_int_distribution<size_t> key_count_dis(
                TestFixture::params::min_segment_length,
                TestFixture::params::max_segment_length);
            std::vector<count_type> values_input
                = test_utils::get_random_data<count_type>(size, 0, 100, seed_value);

            size_t                     offset = 0;
            std::default_random_engine gen(seed_value + seed_value_addition);
            key_type                   current_key = key_distribution_type(0, 100)(gen);
            while(offset < size)
            {
                size_t key_count = key_count_dis(gen);
                current_key += key_delta_dis(gen);

                const size_t end = _HIPCUB_STD::min(size, offset + key_count);
                key_count        = end - offset;
                for(size_t i = offset; i < end; i++)
                {
                    input[i] = current_key;
                }

                unique_expected.push_back(current_key);
                runs_count_expected++;
                counts_expected.push_back(key_count);

                offset += key_count;
            }

            key_type* d_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(key_type)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), size * sizeof(key_type), hipMemcpyHostToDevice));

            key_type*   d_unique_output;
            count_type* d_counts_output;
            count_type* d_runs_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_unique_output,
                                                         runs_count_expected * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_counts_output,
                                                         runs_count_expected * sizeof(count_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_runs_count_output, sizeof(count_type)));

            size_t temporary_storage_bytes = 0;

            HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(nullptr,
                                                            temporary_storage_bytes,
                                                            d_input,
                                                            d_unique_output,
                                                            d_counts_output,
                                                            d_runs_count_output,
                                                            size,
                                                            stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_input,
                                                            d_unique_output,
                                                            d_counts_output,
                                                            d_runs_count_output,
                                                            size,
                                                            stream));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<key_type>   unique_output(runs_count_expected);
            std::vector<count_type> counts_output(runs_count_expected);
            std::vector<count_type> runs_count_output(1);
            HIP_CHECK(hipMemcpy(unique_output.data(),
                                d_unique_output,
                                runs_count_expected * sizeof(key_type),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(counts_output.data(),
                                d_counts_output,
                                runs_count_expected * sizeof(count_type),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(runs_count_output.data(),
                                d_runs_count_output,
                                sizeof(count_type),
                                hipMemcpyDeviceToHost));

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

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(HipcubDeviceRunLengthEncode, NonTrivialRuns)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type    = typename TestFixture::params::key_type;
    using count_type  = typename TestFixture::params::count_type;
    using offset_type = typename TestFixture::params::count_type;
    using key_distribution_type =
        typename std::conditional<std::is_floating_point<key_type>::value,
                                  std::uniform_real_distribution<key_type>,
                                  std::uniform_int_distribution<key_type>>::type;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> empty_sizes{0, 1};
        auto                      sizes = test_utils::get_sizes(seed_value);
        sizes.insert(std::end(sizes), std::begin(empty_sizes), std::end(empty_sizes));

        for(size_t size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data and calculate expected results
            std::vector<offset_type> offsets_expected;
            std::vector<count_type>  counts_expected;
            size_t                   runs_count_expected = 0;

            std::vector<key_type>                 input(size);
            key_distribution_type                 key_delta_dis(1, 5);
            std::uniform_int_distribution<size_t> key_count_dis(
                TestFixture::params::min_segment_length,
                TestFixture::params::max_segment_length);
            std::bernoulli_distribution is_trivial_dis(0.1);
            std::vector<count_type>     values_input
                = test_utils::get_random_data<count_type>(size, 0, 100, seed_value);

            size_t                     offset = 0;
            std::default_random_engine gen(seed_value + seed_value_addition);
            key_type                   current_key = key_distribution_type(0, 100)(gen);
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

                const size_t end = _HIPCUB_STD::min(size, offset + key_count);
                key_count        = end - offset;
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

            key_type* d_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(key_type)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), size * sizeof(key_type), hipMemcpyHostToDevice));

            offset_type* d_offsets_output;
            count_type*  d_counts_output;
            count_type*  d_runs_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(
                &d_offsets_output,
                _HIPCUB_STD::max<size_t>(1, runs_count_expected) * sizeof(offset_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(
                &d_counts_output,
                _HIPCUB_STD::max<size_t>(1, runs_count_expected) * sizeof(count_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_runs_count_output, sizeof(count_type)));

            size_t temporary_storage_bytes = 0;

            HIP_CHECK(hipcub::DeviceRunLengthEncode::NonTrivialRuns(nullptr,
                                                                    temporary_storage_bytes,
                                                                    d_input,
                                                                    d_offsets_output,
                                                                    d_counts_output,
                                                                    d_runs_count_output,
                                                                    size,
                                                                    stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(hipcub::DeviceRunLengthEncode::NonTrivialRuns(d_temporary_storage,
                                                                    temporary_storage_bytes,
                                                                    d_input,
                                                                    d_offsets_output,
                                                                    d_counts_output,
                                                                    d_runs_count_output,
                                                                    size,
                                                                    stream));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<offset_type> offsets_output(runs_count_expected);
            std::vector<count_type>  counts_output(runs_count_expected);
            std::vector<count_type>  runs_count_output(1);
            if(runs_count_expected > 0)
            {
                HIP_CHECK(hipMemcpy(offsets_output.data(),
                                    d_offsets_output,
                                    runs_count_expected * sizeof(offset_type),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(counts_output.data(),
                                    d_counts_output,
                                    runs_count_expected * sizeof(count_type),
                                    hipMemcpyDeviceToHost));
            }
            HIP_CHECK(hipMemcpy(runs_count_output.data(),
                                d_runs_count_output,
                                sizeof(count_type),
                                hipMemcpyDeviceToHost));

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

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

struct counting_to_rl_transform_op_t
{
    unsigned int run_length;

    HIPCUB_HOST_DEVICE
    counting_to_rl_transform_op_t(unsigned int m_run_length)
        : run_length(m_run_length)
    {}

    HIPCUB_HOST_DEVICE
    size_t operator()(const size_t idx) const
    {
        return idx / static_cast<size_t>(run_length);
    }
};

template<bool non_trivial_runs = false,
         class InputIterator,
         class UniqueOrOffsetsOutputIterator,
         class CountsOutputIterator,
         class RunsCountOutputIterator>
void invoke_run_length_encode(void*                         temporary_storage,
                              size_t&                       storage_size,
                              InputIterator                 input,
                              UniqueOrOffsetsOutputIterator unique_or_offsets_output,
                              CountsOutputIterator          counts_output,
                              RunsCountOutputIterator       runs_count_output,
                              size_t                        size,
                              hipStream_t                   stream = 0)
{
    if constexpr(non_trivial_runs)
    {
        HIP_CHECK(hipcub::DeviceRunLengthEncode::NonTrivialRuns(
            temporary_storage,
            storage_size,
            input,
            unique_or_offsets_output, /*offsets_output*/
            counts_output,
            runs_count_output,
            size,
            stream));
    }
    else
    {
        HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(temporary_storage,
                                                        storage_size,
                                                        input,
                                                        unique_or_offsets_output, /*unique_output*/
                                                        counts_output,
                                                        runs_count_output,
                                                        size,
                                                        stream));
    }
}

template<bool non_trivial_runs = false, bool use_graphs = false>
void large_sizes_rle_test()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using count_type      = size_t;
    using offset_type     = size_t;
    using iota_iterator_t = test_utils::counting_iterator<offset_type>;

    hipStream_t stream = 0; // default
    if constexpr(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
        std::default_random_engine gen(seed_value);

        for(size_t size : test_utils::get_large_sizes(42))
        {
            std::uniform_int_distribution<unsigned int> run_length_distribution(
                2,
                std::numeric_limits<unsigned int>::max());

            const unsigned int                  run_length = run_length_distribution(gen);
            const counting_to_rl_transform_op_t counting_to_rl_transform_op{run_length};
            SCOPED_TRACE(testing::Message() << "with run_length = " << run_length);

            const count_type              runs_count = test_utils::ceiling_div(size, run_length);
            const std::vector<count_type> runs_count_expected{runs_count};
            SCOPED_TRACE(testing::Message() << "with runs_count = " << runs_count);

            size = runs_count * static_cast<count_type>(run_length);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate input like: 0, 0, ..., 0, 1, 1, ..., 1, ...
            // where each number is repeated run_length times
            auto d_input
                = test_utils::transform_iterator<iota_iterator_t, counting_to_rl_transform_op_t>(
                    iota_iterator_t{0},
                    counting_to_rl_transform_op);

            offset_type* d_unique_or_offsets_output;
            count_type*  d_counts_output;
            count_type*  d_runs_count_output;

            HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_unique_or_offsets_output,
                                                                runs_count * sizeof(offset_type)));
            HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_counts_output,
                                                                runs_count * sizeof(count_type)));
            HIP_CHECK_MEMORY(
                test_common_utils::hipMallocHelper(&d_runs_count_output, sizeof(count_type)));

            size_t temporary_storage_bytes;
            invoke_run_length_encode<non_trivial_runs>(nullptr,
                                                       temporary_storage_bytes,
                                                       d_input,
                                                       d_unique_or_offsets_output,
                                                       d_counts_output,
                                                       d_runs_count_output,
                                                       size,
                                                       stream);

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if constexpr(use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            invoke_run_length_encode<non_trivial_runs>(d_temporary_storage,
                                                       temporary_storage_bytes,
                                                       d_input,
                                                       d_unique_or_offsets_output,
                                                       d_counts_output,
                                                       d_runs_count_output,
                                                       size,
                                                       stream);

            if constexpr(use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<offset_type> unique_or_offsets_output(runs_count);
            std::vector<count_type>  counts_output(runs_count);
            count_type               runs_count_output;

            HIP_CHECK(hipMemcpy(&runs_count_output,
                                d_runs_count_output,
                                sizeof(count_type),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipFree(d_runs_count_output));

            if(runs_count > 0)
            {
                HIP_CHECK(hipMemcpy(unique_or_offsets_output.data(),
                                    d_unique_or_offsets_output,
                                    runs_count * sizeof(offset_type),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(counts_output.data(),
                                    d_counts_output,
                                    runs_count * sizeof(count_type),
                                    hipMemcpyDeviceToHost));

                HIP_CHECK(hipFree(d_unique_or_offsets_output));
                HIP_CHECK(hipFree(d_counts_output));
            }

            if constexpr(use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            // Validating results

            SCOPED_TRACE(testing::Message() << "runs_count_output");
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(runs_count_output, runs_count));

            for(offset_type i = 0; i < runs_count; i++)
            {
                const offset_type unique_or_offset_expected = non_trivial_runs ? i * run_length : i;

                SCOPED_TRACE(testing::Message() << "unique_or_offsets_output[" << i << "]");
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(unique_or_offsets_output[i], unique_or_offset_expected));
                SCOPED_TRACE(testing::Message() << "counts_output[" << i << "]");
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(counts_output[i], size_t{run_length}));
            }

            if constexpr(use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }
    if(use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceRunLengthEncode, LargeSizesEncode)
{
#if HAS_VALGRIND_H
    //Disable large tests to reduce valgrind run time
    if(RUNNING_ON_VALGRIND)
        GTEST_SKIP() << "Skipping LargeSizesEncode test under Valgrind";
#endif // HAS_VALGRIND_H

    large_sizes_rle_test<>();
}

TEST(RocprimDeviceRunLengthEncode, LargeSizesNonTrivialRuns)
{
#if HAS_VALGRIND_H
    //Disable large tests to reduce valgrind run time
    if(RUNNING_ON_VALGRIND)
        GTEST_SKIP() << "Skipping LargeSizesEncode test under Valgrind";
#endif // HAS_VALGRIND_H

    large_sizes_rle_test<true /*non_trivial_runs*/>();
}
