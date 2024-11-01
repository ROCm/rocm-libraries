// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

// required test headers
#include "indirect_iterator.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "../common_test_header.hpp"

// required rocprim headers
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_nth_element.hpp>
#include <rocprim/functional.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include <cassert>
#include <cstddef>

// Params for tests
template<class KeyType,
         class CompareFunction    = rocprim::less<KeyType>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceNthelementParams
{
    using key_type                              = KeyType;
    using compare_function                      = CompareFunction;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_cpp_14(InputVector     input,
                           OutputVector    output,
                           size_t          nth_element,
                           CompareFunction compare_op)
{
    using key_type = typename InputVector::value_type;

    // Calculate sorted input results on host
    std::vector<key_type> sorted_input(input);
    std::sort(sorted_input.begin(), sorted_input.end(), compare_op);

    // Calculate sorted output results on host
    std::vector<key_type> sorted_output(output);

    // Sort numbers before nth element
    std::sort(sorted_output.begin(), sorted_output.begin() + nth_element, compare_op);

    // Sort numbers after nth element
    std::sort(sorted_output.begin() + nth_element + 1, sorted_output.end(), compare_op);

    // Check if the values are the same
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input));
}

#if CPP17
template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_cpp_17(InputVector     input,
                           OutputVector    output,
                           size_t          nth_element,
                           CompareFunction compare_op)
{
    using key_type = typename InputVector::value_type;

    std::vector<key_type> sorted_input(input);
    std::nth_element(sorted_input.begin(),
                     sorted_input.begin() + nth_element,
                     sorted_input.end(),
                     compare_op);

    // Sort numbers before nth element for input
    std::sort(sorted_input.begin(), sorted_input.begin() + nth_element, compare_op);

    // Sort numbers after nth element for input
    std::sort(sorted_input.begin() + nth_element + 1, sorted_input.end(), compare_op);

    // Calculate sorted output results on host
    std::vector<key_type> sorted_output(output);

    // Sort numbers before nth element for output
    std::sort(sorted_output.begin(), sorted_output.begin() + nth_element, compare_op);

    // Sort numbers after nth element for output
    std::sort(sorted_output.begin() + nth_element + 1, sorted_output.end(), compare_op);

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input));
}
#endif

template<class InputVector, class OutputVector, class CompareFunction>
void inline compare(InputVector     input,
                    OutputVector    output,
                    size_t          nth_element,
                    CompareFunction compare_op)
{
    compare_cpp_14(input, output, nth_element, compare_op);
#if CPP17
    // this comparison is only compiled and executed if c++17 is available
    compare_cpp_17(input, output, nth_element, compare_op);
#else
    ROCPRIM_PRAGMA_MESSAGE("c++17 not available skips direct comparison with std::nth_element");
#endif
}

// ---------------------------------------------------------
// Test for ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceNthelementTests : public ::testing::Test
{
public:
    using key_type                              = typename Params::key_type;
    using compare_function                      = typename Params::compare_function;
    using config                                = typename Params::config;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

using RocprimDeviceNthelementTestsParams = ::testing::Types<
    DeviceNthelementParams<unsigned short>,
    DeviceNthelementParams<signed char>,
    DeviceNthelementParams<int>,
    DeviceNthelementParams<test_utils::custom_test_type<int>>,
    DeviceNthelementParams<unsigned long>,
    DeviceNthelementParams<long long>,
    DeviceNthelementParams<float>,
    DeviceNthelementParams<int8_t>,
    DeviceNthelementParams<uint8_t>,
    DeviceNthelementParams<rocprim::half, rocprim::less<rocprim::half>>,
    DeviceNthelementParams<rocprim::bfloat16, rocprim::less<rocprim::bfloat16>>,
    DeviceNthelementParams<short>,
    DeviceNthelementParams<double>,
    DeviceNthelementParams<test_utils::custom_test_type<float>>,
    DeviceNthelementParams<test_utils::custom_float_type>,
    DeviceNthelementParams<test_utils::custom_test_array_type<int, 4>>,
    // DeviceNthelementParams<int, rocprim::less<int>, rocprim::default_config, true>, // Graphs currently do not work
    DeviceNthelementParams<int, rocprim::less<int>, rocprim::default_config, false, true>,
    DeviceNthelementParams<int, rocprim::greater<int>>,
    DeviceNthelementParams<
        int,
        rocprim::less<int>,
        rocprim::nth_element_config<128, 4, 32, 16, rocprim::block_radix_rank_algorithm::basic>>>;

TYPED_TEST_SUITE(RocprimDeviceNthelementTests, RocprimDeviceNthelementTestsParams);

TYPED_TEST(RocprimDeviceNthelementTests, NthelementKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                       = typename TestFixture::key_type;
    using compare_function               = typename TestFixture::compare_function;
    using config                         = typename TestFixture::config;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    // The size loop alternates between in place and not in place
    bool in_place = false;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            in_place           = !in_place;
            size_t nth_element = 0;
            if(size > 0)
            {
                nth_element = test_utils::get_random_value<size_t>(0, size - 1, seed_value);
            }

            SCOPED_TRACE(testing::Message() << "with nth_element = " << nth_element);

            // Generate data
            std::vector<key_type> input;
            if(rocprim::is_floating_point<key_type>::value)
            {
                input = test_utils::get_random_data<key_type>(size, -1000, 1000, seed_value);
            }
            else
            {
                input = test_utils::get_random_data<key_type>(
                    size,
                    test_utils::numeric_limits<key_type>::min(),
                    test_utils::numeric_limits<key_type>::max(),
                    seed_value);
            }
            std::vector<key_type> output(size);

            key_type* d_input;
            key_type* d_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(*d_input)));
            if(in_place)
            {
                d_output = d_input;
            }
            else
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output,
                                                             output.size() * sizeof(*d_output)));
            }

            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                input.size() * sizeof(*d_input),
                                hipMemcpyHostToDevice));

            const auto input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input);

            // compare function
            compare_function compare_op;

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            if(in_place)
            {
                HIP_CHECK(rocprim::nth_element<config>(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       input_it,
                                                       nth_element,
                                                       input.size(),
                                                       compare_op,
                                                       stream,
                                                       debug_synchronous));
            }
            else
            {
                HIP_CHECK(rocprim::nth_element<config>(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       input_it,
                                                       d_output,
                                                       nth_element,
                                                       input.size(),
                                                       compare_op,
                                                       stream,
                                                       debug_synchronous));
            }

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            if(in_place)
            {
                // Run
                HIP_CHECK(rocprim::nth_element<config>(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       input_it,
                                                       nth_element,
                                                       input.size(),
                                                       compare_op,
                                                       stream,
                                                       debug_synchronous));
            }
            else
            {
                // Run
                HIP_CHECK(rocprim::nth_element<config>(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       input_it,
                                                       d_output,
                                                       nth_element,
                                                       input.size(),
                                                       compare_op,
                                                       stream,
                                                       debug_synchronous));
            }

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(*d_output),
                                hipMemcpyDeviceToHost));

            if(size > 0)
            {
                compare(input, output, nth_element, compare_op);
            }

            HIP_CHECK(hipFree(d_input));
            if(!in_place)
            {
                HIP_CHECK(hipFree(d_output));
            }
            HIP_CHECK(hipFree(d_temp_storage));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

// This test is used to see if it does not end up in endless recursion when
// the equality bucket logic fails.
TEST(RocprimNthelementKeySameTests, NthelementKeySame)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type               = int;
    using compare_function       = rocprim::less<int>;
    const bool debug_synchronous = false;

    unsigned int seed_value = rand();
    for(size_t size : test_utils::get_sizes(seed_value))
    {
        hipStream_t stream = 0; // default
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        auto nth_element = 0;
        SCOPED_TRACE(testing::Message() << "with nth_element = " << nth_element);

        // Generate data
        std::vector<key_type> input(size, 8);
        std::vector<key_type> output(size);

        key_type* d_input;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(*d_input)));

        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(*d_input),
                            hipMemcpyHostToDevice));

        // compare function
        compare_function compare_op;

        // temp storage
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::nth_element(d_temp_storage,
                                       temp_storage_size_bytes,
                                       d_input,
                                       d_input,
                                       nth_element,
                                       input.size(),
                                       compare_op,
                                       stream,
                                       debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

        // Run
        HIP_CHECK(rocprim::nth_element(d_temp_storage,
                                       temp_storage_size_bytes,
                                       d_input,
                                       d_input,
                                       nth_element,
                                       input.size(),
                                       compare_op,
                                       stream,
                                       debug_synchronous));

        HIP_CHECK(hipGetLastError());

        // Copy output to host
        HIP_CHECK(hipMemcpy(output.data(),
                            d_input,
                            output.size() * sizeof(*d_input),
                            hipMemcpyDeviceToHost));

        // Check if the values are the same
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(input, output));

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}
