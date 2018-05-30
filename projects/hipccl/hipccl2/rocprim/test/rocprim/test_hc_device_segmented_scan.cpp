// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

template<
    class Input,
    class Output,
    class ScanOp = ::rocprim::plus<Input>,
    int Init = 0, // as only integral types supported, int is used here even for floating point inputs
    unsigned int MinSegmentLength = 0,
    unsigned int MaxSegmentLength = 1000
>
struct params
{
    using input_type = Input;
    using output_type = Output;
    using scan_op_type = ScanOp;
    static constexpr input_type init = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class RocprimDeviceSegmentedScan : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<unsigned char, unsigned int, rocprim::plus<unsigned int>>,
    params<int, int, rocprim::plus<int>, -100, 0, 10000>,
    params<double, double, rocprim::minimum<double>, 1000, 0, 10000>,
    params<int, short, rocprim::maximum<int>, 10, 1000, 10000>,
    params<float, double, rocprim::maximum<double>, 50, 2, 10>,
    params<float, float, rocprim::plus<float>, 123, 100, 200>
> Params;

TYPED_TEST_CASE(RocprimDeviceSegmentedScan, Params);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        (1 << 17) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 1000000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

TYPED_TEST(RocprimDeviceSegmentedScan, InclusiveScan)
{
    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using result_type = output_type;

    using offset_type = unsigned int;
    const bool debug_synchronous = false;
    scan_op_type scan_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data and calculate expected results
        std::vector<output_type> values_expected(size);
        std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100);

        std::vector<offset_type> offsets;
        unsigned int segments_count = 0;
        size_t offset = 0;
        while(offset < size)
        {
            const size_t segment_length = segment_length_dis(gen);
            offsets.push_back(offset);

            const size_t end = std::min(size, offset + segment_length);
            result_type aggregate = values_input[offset];
            values_expected[offset] = aggregate;
            for(size_t i = offset + 1; i < end; i++)
            {
                aggregate = scan_op(aggregate, static_cast<result_type>(values_input[i]));
                values_expected[i] = aggregate;
            }

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        hc::array<input_type> d_values_input(hc::extent<1>(size), values_input.begin(), acc_view);
        hc::array<offset_type> d_offsets(hc::extent<1>(segments_count + 1), offsets.begin(), acc_view);
        hc::array<output_type> d_values_output(hc::extent<1>(size), acc_view);

        size_t temporary_storage_bytes;
        rocprim::segmented_inclusive_scan(
            nullptr, temporary_storage_bytes,
            d_values_input.accelerator_pointer(), d_values_output.accelerator_pointer(),
            segments_count,
            d_offsets.accelerator_pointer(), d_offsets.accelerator_pointer() + 1,
            scan_op,
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0);
        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rocprim::segmented_inclusive_scan(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_values_input.accelerator_pointer(), d_values_output.accelerator_pointer(),
            segments_count,
            d_offsets.accelerator_pointer(), d_offsets.accelerator_pointer() + 1,
            scan_op,
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<output_type> values_output = d_values_output;
        for(size_t i = 0; i < values_output.size(); i++)
        {
            if(std::is_integral<output_type>::value)
            {
                ASSERT_EQ(values_output[i], values_expected[i]) << "with index: " << i;
            }
            else
            {
                auto diff = std::max<output_type>(
                    std::abs(0.01 * values_expected[i]), output_type(0.01)
                );
                ASSERT_NEAR(values_output[i], values_expected[i], diff) << "with index: " << i;
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScan)
{
    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    using result_type = output_type;
    using offset_type = unsigned int;

    constexpr input_type init = TestFixture::params::init;
    const bool debug_synchronous = false;
    scan_op_type scan_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data and calculate expected results
        std::vector<output_type> values_expected(size);
        std::vector<input_type> values_input = test_utils::get_random_data<input_type>(size, 0, 100);

        std::vector<offset_type> offsets;
        unsigned int segments_count = 0;
        size_t offset = 0;
        while(offset < size)
        {
            const size_t segment_length = segment_length_dis(gen);
            offsets.push_back(offset);

            const size_t end = std::min(size, offset + segment_length);
            result_type aggregate = init;
            values_expected[offset] = aggregate;
            for(size_t i = offset + 1; i < end; i++)
            {
                aggregate = scan_op(aggregate, static_cast<result_type>(values_input[i-1]));
                values_expected[i] = aggregate;
            }

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        hc::array<input_type> d_values_input(hc::extent<1>(size), values_input.begin(), acc_view);
        hc::array<offset_type> d_offsets(hc::extent<1>(segments_count + 1), offsets.begin(), acc_view);
        hc::array<output_type> d_values_output(hc::extent<1>(size), acc_view);

        size_t temporary_storage_bytes;
        rocprim::segmented_exclusive_scan(
            nullptr, temporary_storage_bytes,
            d_values_input.accelerator_pointer(), d_values_output.accelerator_pointer(),
            segments_count,
            d_offsets.accelerator_pointer(), d_offsets.accelerator_pointer() + 1,
            init, scan_op,
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0);
        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rocprim::segmented_exclusive_scan(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_values_input.accelerator_pointer(), d_values_output.accelerator_pointer(),
            segments_count,
            d_offsets.accelerator_pointer(), d_offsets.accelerator_pointer() + 1,
            init, scan_op,
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<output_type> values_output = d_values_output;
        for(size_t i = 0; i < values_output.size(); i++)
        {
            if(std::is_integral<output_type>::value)
            {
                ASSERT_EQ(values_output[i], values_expected[i]) << "with index: " << i;
            }
            else
            {
                auto diff = std::max<output_type>(std::abs(0.01 * values_expected[i]), 0.01);
                ASSERT_NEAR(values_output[i], values_expected[i], diff) << "with index: " << i;
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, InclusiveScanUsingHeadFlags)
{
    using input_type = typename TestFixture::params::input_type;
    using flag_type = unsigned int;
    using output_type = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    const bool debug_synchronous = false;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<input_type> input = test_utils::get_random_data<input_type>(size, 1, 1);
        std::vector<flag_type> flags = test_utils::get_random_data<flag_type>(size, 0, 10);
        flags[0] = 1U;
        std::transform(
            flags.begin(), flags.end(), flags.begin(),
            [](flag_type a){ if(a==1U) return 1U; return 0U; }
        );

        hc::array<input_type> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<flag_type> d_flags(hc::extent<1>(size), flags.begin(), acc_view);
        hc::array<output_type> d_output(size, acc_view);
        acc_view.wait();

        // scan function
        scan_op_type scan_op;

        // Calculate expected results on host
        std::vector<output_type> expected(input.size());
        test_utils::host_inclusive_scan(
            rocprim::make_zip_iterator(
                rocprim::make_tuple(input.begin(), flags.begin())
            ),
            rocprim::make_zip_iterator(
                rocprim::make_tuple(input.end(), flags.end())
            ),
            rocprim::make_zip_iterator(
                rocprim::make_tuple(expected.begin(), rocprim::make_discard_iterator())
            ),
            [scan_op](const rocprim::tuple<output_type, flag_type>& t1,
                      const rocprim::tuple<output_type, flag_type>& t2)
                -> rocprim::tuple<output_type, flag_type>
            {
                if(!rocprim::get<1>(t2))
                {
                    return rocprim::make_tuple(
                        scan_op(rocprim::get<0>(t1), rocprim::get<0>(t2)),
                        rocprim::get<1>(t1) + rocprim::get<1>(t2)
                    );
                }
                return t2;
            }
        );

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::segmented_inclusive_scan(
            nullptr,
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_flags.accelerator_pointer(),
            input.size(),
            scan_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
        acc_view.wait();

        // Run
        rocprim::segmented_inclusive_scan(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_flags.accelerator_pointer(),
            input.size(),
            scan_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<output_type> output = d_output;
        for(size_t i = 0; i < output.size(); i++)
        {
            auto diff = std::max<output_type>(std::abs(0.1f * expected[i]), output_type(0.01f));
            if(std::is_integral<output_type>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff) << "with index: " << i;
        }
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScanUsingHeadFlags)
{
    using input_type = typename TestFixture::params::input_type;
    using flag_type = unsigned int;
    using output_type = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    constexpr input_type init = TestFixture::params::init;
    const bool debug_synchronous = false;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<input_type> input = test_utils::get_random_data<input_type>(size, 1, 1);
        std::vector<flag_type> flags = test_utils::get_random_data<flag_type>(size, 0, 10);
        flags[0] = 1U;
        std::transform(
            flags.begin(), flags.end(), flags.begin(),
            [](flag_type a){ if(a==1U) return 1U; return 0U; }
        );

        hc::array<input_type> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<flag_type> d_flags(hc::extent<1>(size), flags.begin(), acc_view);
        hc::array<output_type> d_output(size, acc_view);
        acc_view.wait();

        // scan function
        scan_op_type scan_op;

        // Calculate expected results on host
        std::vector<output_type> expected(input.size());
        // Modify input to perform exclusive operation on initial input.
        // This shifts input one to the right and initializes segments with init.
        expected[0] = init;
        std::transform(
            rocprim::make_zip_iterator(
                rocprim::make_tuple(input.begin(), flags.begin()+1)
            ),
            rocprim::make_zip_iterator(
                rocprim::make_tuple(input.end() - 1, flags.end())
            ),
            rocprim::make_zip_iterator(
                rocprim::make_tuple(expected.begin() + 1, rocprim::make_discard_iterator())
            ),
            [](const rocprim::tuple<input_type, flag_type>& t)
                -> rocprim::tuple<input_type, flag_type>
            {
                if(rocprim::get<1>(t))
                {
                    return rocprim::make_tuple(
                        static_cast<input_type>(init),
                        rocprim::get<1>(t)
                    );
                }
                return t;
            }
        );
        // Now we can run inclusive scan and get segmented exclusive results
        test_utils::host_inclusive_scan(
            rocprim::make_zip_iterator(
                rocprim::make_tuple(expected.begin(), flags.begin())
            ),
            rocprim::make_zip_iterator(
                rocprim::make_tuple(expected.end(), flags.end())
            ),
            rocprim::make_zip_iterator(
                rocprim::make_tuple(expected.begin(), rocprim::make_discard_iterator())
            ),
            [scan_op](const rocprim::tuple<output_type, flag_type>& t1,
                      const rocprim::tuple<output_type, flag_type>& t2)
                -> rocprim::tuple<output_type, flag_type>
            {
                if(!rocprim::get<1>(t2))
                {
                    return rocprim::make_tuple(
                        scan_op(rocprim::get<0>(t1), rocprim::get<0>(t2)),
                        rocprim::get<1>(t1) + rocprim::get<1>(t2)
                    );
                }
                return t2;
            }
        );

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        rocprim::segmented_exclusive_scan(
            nullptr,
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_flags.accelerator_pointer(),
            init,
            input.size(),
            scan_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
        acc_view.wait();

        // Run
        rocprim::segmented_exclusive_scan(
            d_temp_storage.accelerator_pointer(),
            temp_storage_size_bytes,
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            d_flags.accelerator_pointer(),
            init,
            input.size(),
            scan_op,
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<output_type> output = d_output;
        for(size_t i = 0; i < output.size(); i++)
        {
            auto diff = std::max<output_type>(std::abs(0.1f * expected[i]), output_type(0.01f));
            if(std::is_integral<output_type>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff) << "with index: " << i;
        }
    }
}
