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
// HIP API
#include <hip/hip_runtime.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

namespace rp = rocprim;

template<
    class Input,
    class Output,
    class ScanOp = ::rocprim::plus<Input>,
    int Init = 0, // as only integral types supported, int is used here even for floating point inputs
    unsigned int MinSegmentLength = 0,
    unsigned int MaxSegmentLength = 1000,
    // Tests output iterator with void value_type (OutputIterator concept)
    // Segmented scan primitives which use head flags do not support this kind
    // of output iterators.
    bool UseIdentityIterator = false
>
struct params
{
    using input_type = Input;
    using output_type = Output;
    using scan_op_type = ScanOp;
    static constexpr int init = Init;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDeviceSegmentedScan : public ::testing::Test {
public:
    using params = Params;
};

using custom_short2 = test_utils::custom_test_type<short>;
using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

typedef ::testing::Types<
    params<unsigned char, unsigned int, rocprim::plus<unsigned int>>,
    params<int, int, rocprim::plus<int>, -100, 0, 10000>,
    params<custom_double2, custom_double2, rocprim::minimum<custom_double2>, 1000, 0, 10000>,
    params<custom_int2, custom_short2, rocprim::maximum<custom_int2>, 10, 1000, 10000>,
    params<float, double, rocprim::maximum<double>, 50, 2, 10>,
    params<float, float, rocprim::plus<float>, 123, 100, 200, true>,
#ifndef __HIP__
    // hip-clang does not allow to convert half to float
    params<rp::half, float, rp::plus<float>, 0, 10, 300, true>,
    // hip-clang does provide host comparison operators
    params<rp::half, rp::half, test_utils::half_minimum, 0, 1000, 30000>,
#endif
    params<unsigned char, long long, rocprim::plus<int>, 10, 3000, 4000>
> Params;

TYPED_TEST_CASE(RocprimDeviceSegmentedScan, Params);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        (1 << 16) - 1220
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
    static constexpr bool use_identity_iterator =
        TestFixture::params::use_identity_iterator;
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

    hipStream_t stream = 0; // default stream

    const std::vector<size_t> sizes = get_sizes();
    for(size_t size : get_sizes())
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
                aggregate = scan_op(aggregate, values_input[i]);
                values_expected[i] = aggregate;
            }

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        input_type  * d_values_input;
        offset_type * d_offsets;
        output_type * d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(input_type)));
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(output_type)));
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(input_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_offsets, offsets.data(),
                (segments_count + 1) * sizeof(offset_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        size_t temporary_storage_bytes;
        HIP_CHECK(
            rocprim::segmented_inclusive_scan(
                nullptr, temporary_storage_bytes,
                d_values_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                segments_count,
                d_offsets, d_offsets + 1,
                scan_op,
                stream, debug_synchronous
            )
        );

        ASSERT_GT(temporary_storage_bytes, 0);
        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            rocprim::segmented_inclusive_scan(
                d_temporary_storage, temporary_storage_bytes,
                d_values_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                segments_count,
                d_offsets, d_offsets + 1,
                scan_op,
                stream, debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<output_type> values_output(size);
        HIP_CHECK(
            hipMemcpy(
                values_output.data(), d_values_output,
                values_output.size() * sizeof(output_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(values_output, values_expected, 0.01f));

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_values_output));
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScan)
{
    using input_type = typename TestFixture::params::input_type;
    using output_type = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    static constexpr bool use_identity_iterator =
        TestFixture::params::use_identity_iterator;
    using result_type = output_type;
    using offset_type = unsigned int;

    const input_type init = TestFixture::params::init;
    const bool debug_synchronous = false;
    scan_op_type scan_op;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    hipStream_t stream = 0; // default stream

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
                aggregate = scan_op(aggregate, values_input[i-1]);
                values_expected[i] = aggregate;
            }

            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        input_type  * d_values_input;
        offset_type * d_offsets;
        output_type * d_values_output;
        HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(input_type)));
        HIP_CHECK(hipMalloc(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(output_type)));
        HIP_CHECK(
            hipMemcpy(
                d_values_input, values_input.data(),
                size * sizeof(input_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_offsets, offsets.data(),
                (segments_count + 1) * sizeof(offset_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        size_t temporary_storage_bytes;
        HIP_CHECK(
            rocprim::segmented_exclusive_scan(
                nullptr, temporary_storage_bytes,
                d_values_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                segments_count,
                d_offsets, d_offsets + 1,
                init, scan_op,
                stream, debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        ASSERT_GT(temporary_storage_bytes, 0);
        void * d_temporary_storage;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(
            rocprim::segmented_exclusive_scan(
                d_temporary_storage, temporary_storage_bytes,
                d_values_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                segments_count,
                d_offsets, d_offsets + 1,
                init, scan_op,
                stream, debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<output_type> values_output(size);
        HIP_CHECK(
            hipMemcpy(
                values_output.data(), d_values_output,
                values_output.size() * sizeof(output_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(values_output, values_expected, 0.01f));

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_values_output));
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, InclusiveScanUsingHeadFlags)
{
    // Does not support output iterator with void value_type
    using input_type = typename TestFixture::params::input_type;
    using flag_type = unsigned int;
    using output_type = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default stream

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<input_type> input = test_utils::get_random_data<input_type>(size, 1, 10);
        std::vector<flag_type> flags = test_utils::get_random_data<flag_type>(size, 0, 10);
        flags[0] = 1U;
        std::transform(
            flags.begin(), flags.end(), flags.begin(),
            [](flag_type a){ if(a == 1U) return 1U; return 0U; }
        );

        input_type * d_input;
        flag_type * d_flags;
        output_type * d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input_type)));
        HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(flag_type)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(output_type)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_flags, flags.data(),
                flags.size() * sizeof(flag_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

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
        HIP_CHECK(
            rocprim::segmented_inclusive_scan(
                nullptr, temp_storage_size_bytes,
                d_input, d_output, d_flags,
                input.size(), scan_op, stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        void * d_temp_storage = nullptr;
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::segmented_inclusive_scan(
                d_temp_storage, temp_storage_size_bytes,
                d_input, d_output, d_flags,
                input.size(), scan_op, stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if output values are as expected
        std::vector<output_type> output(input.size());
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_flags));
        HIP_CHECK(hipFree(d_output));
    }
}

TYPED_TEST(RocprimDeviceSegmentedScan, ExclusiveScanUsingHeadFlags)
{
    // Does not support output iterator with void value_type
    using input_type = typename TestFixture::params::input_type;
    using flag_type = unsigned int;
    using output_type = typename TestFixture::params::output_type;
    using scan_op_type = typename TestFixture::params::scan_op_type;
    const input_type init = TestFixture::params::init;
    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default stream

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<input_type> input = test_utils::get_random_data<input_type>(size, 1, 10);
        std::vector<flag_type> flags = test_utils::get_random_data<flag_type>(size, 0, 10);
        flags[0] = 1U;
        std::transform(
            flags.begin(), flags.end(), flags.begin(),
            [](flag_type a){ if(a == 1U) return 1U; return 0U; }
        );

        input_type * d_input;
        flag_type * d_flags;
        output_type * d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input_type)));
        HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(flag_type)));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(output_type)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(input_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_flags, flags.data(),
                flags.size() * sizeof(flag_type),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

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
            [init](const rocprim::tuple<input_type, flag_type>& t)
                -> rocprim::tuple<input_type, flag_type>
            {
                if(rocprim::get<1>(t))
                {
                    return rocprim::make_tuple(
                        init,
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
        HIP_CHECK(
            rocprim::segmented_exclusive_scan(
                nullptr, temp_storage_size_bytes,
                d_input, d_output, d_flags, init,
                input.size(), scan_op, stream, debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        void * d_temp_storage = nullptr;
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::segmented_exclusive_scan(
                d_temp_storage, temp_storage_size_bytes,
                d_input, d_output, d_flags, init,
                input.size(), scan_op, stream, debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if output values are as expected
        std::vector<output_type> output(input.size());
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(output_type),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_flags));
        HIP_CHECK(hipFree(d_output));
    }
}
