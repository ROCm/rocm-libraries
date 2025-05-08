// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "identity_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required rocprim headers
#include <rocprim/device/device_segmented_scan.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdint.h>
#include <vector>

template<class Input,
         class Output,
         class ScanOp = ::rocprim::plus<Input>,
         int Init
         = 0, // as only integral types supported, int is used here even for floating point inputs
         unsigned int MinSegmentLength = 0,
         unsigned int MaxSegmentLength = 1000,
         // Tests output iterator with void value_type (OutputIterator concept)
         // Segmented scan primitives which use head flags do not support this kind
         // of output iterators.
         bool UseIdentityIterator = false,
         bool UseGraphs           = false>
struct params
{
    using input_type                                    = Input;
    using output_type                                   = Output;
    using scan_op_type                                  = ScanOp;
    static constexpr int          init                  = Init;
    static constexpr unsigned int min_segment_length    = MinSegmentLength;
    static constexpr unsigned int max_segment_length    = MaxSegmentLength;
    static constexpr bool         use_identity_iterator = UseIdentityIterator;
    static constexpr bool         use_graphs            = UseGraphs;
};

template<class Params>
class RocprimDeviceSegmentedScan : public ::testing::Test
{
public:
    using params = Params;
};

using custom_short2  = common::custom_type<short, short, true>;
using custom_int2    = common::custom_type<int, int, true>;
using custom_double2 = common::custom_type<double, double, true>;
using half           = rocprim::half;
using bfloat16       = rocprim::bfloat16;

using Params = ::testing::Types<
    params<unsigned char, unsigned int, rocprim::plus<unsigned int>>,
    params<int, int, rocprim::plus<int>, -100, 0, 10000>,
    params<int8_t, int8_t, rocprim::plus<int8_t>, -100, 0, 10000>,
    params<custom_double2, custom_double2, rocprim::minimum<custom_double2>, 1000, 0, 10000>,
    params<custom_int2, custom_short2, rocprim::maximum<custom_int2>, 10, 1000, 10000>,
    params<double, double, rocprim::maximum<double>, 50, 2, 10>,
    params<float, float, rocprim::plus<float>, 123, 100, 200, true>,
    params<bfloat16, float, rocprim::plus<bfloat16>, 0, 3, 50, true>,
    params<bfloat16, bfloat16, rocprim::minimum<bfloat16>, 0, 1000, 30000>,
    params<half, float, rocprim::plus<float>, 0, 10, 200, true>,
    params<half, half, rocprim::minimum<half>, 0, 1000, 30000>,
    params<unsigned char, long long, rocprim::plus<int>, 10, 3000, 4000>,
    params<int, int, ::rocprim::plus<int>, 0, 0, 1000, false, true>>;

TYPED_TEST_SUITE(RocprimDeviceSegmentedScan, Params);

TYPED_TEST(RocprimDeviceSegmentedScan, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type   = typename TestFixture::params::input_type;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;
    using offset_type  = unsigned int;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;
    const bool     debug_synchronous     = false;

    scan_op_type scan_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    common::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default stream
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data and calculate expected results
            std::vector<output_type> values_expected(size);
            std::vector<input_type>  values_input
                = test_utils::get_random_data_wrapped<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            std::vector<size_t>      sizes;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                sizes.push_back(segment_length);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

                input_type aggregate    = values_input[offset];
                values_expected[offset] = aggregate;
                for(size_t i = offset + 1; i < end; i++)
                {
                    aggregate          = scan_op(aggregate, values_input[i]);
                    values_expected[i] = aggregate;
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            // intermediate results of inclusive scan are stored as input_type,
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<input_type>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            common::device_ptr<input_type>  d_values_input(values_input);
            common::device_ptr<offset_type> d_offsets(offsets);
            common::device_ptr<output_type> d_values_output(size);

            size_t temporary_storage_bytes;
            HIP_CHECK(rocprim::segmented_inclusive_scan(
                nullptr,
                temporary_storage_bytes,
                d_values_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output.get()),
                segments_count,
                d_offsets.get(),
                d_offsets.get() + 1,
                scan_op,
                stream,
                debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0);
            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::segmented_inclusive_scan(
                d_temporary_storage.get(),
                temporary_storage_bytes,
                d_values_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output.get()),
                segments_count,
                d_offsets.get(),
                d_offsets.get() + 1,
                scan_op,
                stream,
                debug_synchronous));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            auto values_output = d_values_output.load();

            if(size > 0)
            {
                const float single_op_precision = precision / max_segment_length;

                size_t current_offset     = 0;
                size_t current_size_index = 0;
                for(size_t i = 0; i < values_output.size(); ++i)
                {
                    if((i - current_offset) == sizes[current_size_index])
                    {
                        current_offset += sizes[current_size_index];
                        ++current_size_index;
                    }
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(values_output[i],
                                                values_expected[i],
                                                single_op_precision * (i - current_offset)));
                }
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

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type   = typename TestFixture::params::input_type;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;
    using offset_type  = unsigned int;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;
    const bool     debug_synchronous     = false;

    const input_type init = input_type{TestFixture::params::init};

    scan_op_type scan_op;

    std::random_device         rd;
    std::default_random_engine gen(rd());

    common::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length);

    hipStream_t stream = 0; // default stream
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data and calculate expected results
            std::vector<output_type> values_expected(size);
            std::vector<input_type>  values_input
                = test_utils::get_random_data_wrapped<input_type>(size, 0, 100, seed_value);

            std::vector<offset_type> offsets;
            std::vector<size_t>      sizes;
            unsigned int             segments_count     = 0;
            size_t                   offset             = 0;
            size_t                   max_segment_length = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                sizes.push_back(segment_length);
                offsets.push_back(offset);

                const size_t end   = std::min(size, offset + segment_length);
                max_segment_length = std::max(max_segment_length, end - offset);

                input_type aggregate    = init;
                values_expected[offset] = aggregate;
                for(size_t i = offset + 1; i < end; i++)
                {
                    aggregate          = scan_op(aggregate, values_input[i - 1]);
                    values_expected[i] = output_type(aggregate);
                }

                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            // intermediate results of exclusive scan are stored as decltype(init),
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<decltype(init)>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            common::device_ptr<input_type>  d_values_input(values_input);
            common::device_ptr<offset_type> d_offsets(offsets);
            common::device_ptr<output_type> d_values_output(size);

            size_t temporary_storage_bytes;
            HIP_CHECK(rocprim::segmented_exclusive_scan(
                nullptr,
                temporary_storage_bytes,
                d_values_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output.get()),
                segments_count,
                d_offsets.get(),
                d_offsets.get() + 1,
                init,
                scan_op,
                stream,
                debug_synchronous));

            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_GT(temporary_storage_bytes, 0);
            common::device_ptr<void>     d_temporary_storage(temporary_storage_bytes);
            test_utils::GraphHelper      gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::segmented_exclusive_scan(
                d_temporary_storage.get(),
                temporary_storage_bytes,
                d_values_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output.get()),
                segments_count,
                d_offsets.get(),
                d_offsets.get() + 1,
                init,
                scan_op,
                stream,
                debug_synchronous));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            auto values_output = d_values_output.load();

            if(size > 0)
            {
                const float single_op_precision = precision / max_segment_length;

                size_t current_offset     = 0;
                size_t current_size_index = 0;
                for(size_t i = 0; i < values_output.size(); ++i)
                {
                    if((i - current_offset) == sizes[current_size_index])
                    {
                        current_offset += sizes[current_size_index];
                        ++current_size_index;
                    }
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(values_output[i],
                                                values_expected[i],
                                                single_op_precision * (i - current_offset)));
                }
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

TYPED_TEST(RocprimDeviceSegmentedScan, InclusiveScanUsingHeadFlags)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Does not support output iterator with void value_type
    using input_type   = typename TestFixture::params::input_type;
    using flag_type    = unsigned int;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;

    const bool debug_synchronous = false;

    // scan function
    scan_op_type scan_op;

    hipStream_t stream = 0; // default stream
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<input_type> input
                = test_utils::get_random_data_wrapped<input_type>(size, 1, 10, seed_value);
            std::vector<flag_type> flags
                = test_utils::get_random_data_wrapped<flag_type>(size, 0, 10, seed_value);

            if(size != 0)
                flags[0] = 1U;

            // generate segments and find their maximum width
            size_t max_segment_length = 1;
            size_t curr_segment_start = 0;
            for(size_t i = 1; i < size; ++i)
            {
                if(flags[i] == 1U)
                {
                    size_t curr_segment = i - curr_segment_start;
                    if(curr_segment > max_segment_length)
                        max_segment_length = curr_segment;
                    curr_segment_start = i;
                }
                else
                    flags[i] = 0U;
            }
            {
                size_t curr_segment = size - curr_segment_start;
                if(curr_segment > max_segment_length)
                    max_segment_length = curr_segment;
            }

            // intermediate results of inclusive scan are stored as input_type,
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<input_type>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            common::device_ptr<input_type>  d_input(input);
            common::device_ptr<flag_type>   d_flags(flags);
            common::device_ptr<output_type> d_output(input.size());

            // Calculate expected results on host
            std::vector<output_type> expected(input.size());

            test_utils::host_inclusive_segmented_scan_headflags<input_type>(input.begin(),
                                                                            input.end(),
                                                                            flags.begin(),
                                                                            expected.begin(),
                                                                            scan_op);

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::segmented_inclusive_scan(nullptr,
                                                        temp_storage_size_bytes,
                                                        d_input.get(),
                                                        d_output.get(),
                                                        d_flags.get(),
                                                        input.size(),
                                                        scan_op,
                                                        stream,
                                                        debug_synchronous));

            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::segmented_inclusive_scan(d_temp_storage.get(),
                                                        temp_storage_size_bytes,
                                                        d_input.get(),
                                                        d_output.get(),
                                                        d_flags.get(),
                                                        input.size(),
                                                        scan_op,
                                                        stream,
                                                        debug_synchronous));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            auto output = d_output.load();

            if(size > 0)
            {
                const float single_op_precision = precision / max_segment_length;
                for(size_t i = 0; i < output.size(); ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(output[i], expected[i], single_op_precision * i));
                }
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

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScanUsingHeadFlags)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Does not support output iterator with void value_type
    using input_type   = typename TestFixture::params::input_type;
    using flag_type    = unsigned int;
    using output_type  = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using is_plus_op   = test_utils::is_plus_operator<scan_op_type>;

    const bool debug_synchronous = false;

    const input_type init = input_type{TestFixture::params::init};

    // scan function
    scan_op_type scan_op;

    hipStream_t stream = 0; // default stream
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<input_type> input
                = test_utils::get_random_data_wrapped<input_type>(size, 1, 10, seed_value);
            std::vector<flag_type> flags
                = test_utils::get_random_data_wrapped<flag_type>(size, 0, 10, seed_value);

            if(size != 0)
                flags[0] = 1U;

            // generate segments and find their maximum width
            size_t max_segment_length = 1;
            size_t curr_segment_start = 0;
            for(size_t i = 1; i < size; ++i)
            {
                if(flags[i] == 1U)
                {
                    size_t curr_segment = i - curr_segment_start;
                    if(curr_segment > max_segment_length)
                        max_segment_length = curr_segment;
                    curr_segment_start = i;
                }
                else
                    flags[i] = 0U;
            }
            {
                size_t curr_segment = size - curr_segment_start;
                if(curr_segment > max_segment_length)
                    max_segment_length = curr_segment;
            }

            // intermediate results of exclusive scan are stored as decltype(init),
            // not as is_plus_op::value_type
            const float precision
                = is_plus_op::value
                      ? std::max(test_utils::precision<typename is_plus_op::value_type>,
                                 test_utils::precision<decltype(init)>)
                            * max_segment_length
                      : 0;
            if(precision > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                continue;
            }

            common::device_ptr<input_type>  d_input(input);
            common::device_ptr<flag_type>   d_flags(flags);
            common::device_ptr<output_type> d_output(input.size());

            // Calculate expected results on host
            std::vector<output_type> expected(input.size());

            test_utils::host_exclusive_segmented_scan_headflags(input.begin(),
                                                                input.end(),
                                                                flags.begin(),
                                                                expected.begin(),
                                                                scan_op,
                                                                init);

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::segmented_exclusive_scan(nullptr,
                                                        temp_storage_size_bytes,
                                                        d_input.get(),
                                                        d_output.get(),
                                                        d_flags.get(),
                                                        init,
                                                        input.size(),
                                                        scan_op,
                                                        stream,
                                                        debug_synchronous));

            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::segmented_exclusive_scan(d_temp_storage.get(),
                                                        temp_storage_size_bytes,
                                                        d_input.get(),
                                                        d_output.get(),
                                                        d_flags.get(),
                                                        init,
                                                        input.size(),
                                                        scan_op,
                                                        stream,
                                                        debug_synchronous));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            auto output = d_output.load();

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipDeviceSynchronize());

            if(size > 0)
            {
                const float single_op_precision = precision / max_segment_length;
                for(size_t i = 0; i < output.size(); ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(output[i], expected[i], single_op_precision * i));
                }
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}
