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
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "../common_test_header.hpp"

// required rocprim headers
#include <algorithm>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_partial_sort.hpp>
#include <rocprim/functional.hpp>

#include <iostream>
#include <iterator>
#include <vector>

#include <cassert>
#include <cstddef>

// Params for tests
template<class KeyType,
         class CompareFunction    = ::rocprim::less<KeyType>,
         class Config             = ::rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DevicePartialSortParams
{
    using key_type                              = KeyType;
    using compare_function                      = CompareFunction;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
class RocprimDevicePartialSortTests : public ::testing::Test
{
public:
    using key_type                              = typename Params::key_type;
    using compare_function                      = typename Params::compare_function;
    using config                                = typename Params::config;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

// TODO no graph support
using RocprimDevicePartialSortTestsParams = ::testing::Types<
    DevicePartialSortParams<unsigned short>,
    DevicePartialSortParams<char>,
    DevicePartialSortParams<const int>,
    DevicePartialSortParams<test_utils::custom_test_type<int>>,
    DevicePartialSortParams<unsigned long>,
    DevicePartialSortParams<long long, ::rocprim::greater<long long>>,
    DevicePartialSortParams<const float>,
    DevicePartialSortParams<int8_t>,
    DevicePartialSortParams<uint8_t>,
    DevicePartialSortParams<rocprim::half>,
    DevicePartialSortParams<rocprim::bfloat16>,
    DevicePartialSortParams<double>,
    DevicePartialSortParams<test_utils::custom_test_type<float>>,
    DevicePartialSortParams<test_utils::custom_float_type>,
    DevicePartialSortParams<test_utils::custom_test_array_type<int, 4>>,
    DevicePartialSortParams<int, ::rocprim::less<int>, rocprim::default_config, false, true>,
    DevicePartialSortParams<
        int,
        ::rocprim::less<int>,
        rocprim::partial_sort_config<
            rocprim::
                nth_element_config<128, 4, 32, 16, rocprim::block_radix_rank_algorithm::basic>>>>;

template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_partial_sort_cpp_14(InputVector     input,
                                        OutputVector    output,
                                        size_t          middle,
                                        CompareFunction compare_op)
{
    using key_type = typename InputVector::value_type;

    if(input.size() == 0)
    {
        return;
    }

    // Calculate sorted input results on host
    std::vector<key_type> sorted_input(input);
    std::sort(sorted_input.begin(), sorted_input.end(), compare_op);

    // Calculate sorted output results on host
    std::vector<key_type> sorted_output(output);
    std::sort(sorted_output.begin() + middle + 1, sorted_output.end(), compare_op);

    // Check if the values are the same
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input));
}

#if CPP17
template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_partial_sort_cpp_17(InputVector     input,
                                        OutputVector    output,
                                        size_t          middle,
                                        CompareFunction compare_op)
{
    using key_type = typename InputVector::value_type;

    if(input.size() == 0)
    {
        return;
    }

    // Calculate sorted input results on host
    std::vector<key_type> sorted_input(input);
    std::partial_sort(sorted_input.begin(),
                      sorted_input.begin() + middle + 1,
                      sorted_input.end(),
                      compare_op);
    std::sort(sorted_input.begin() + middle + 1, sorted_input.end(), compare_op);

    // Calculate sorted output results on host
    std::vector<key_type> sorted_output(output);
    std::sort(sorted_output.begin() + middle + 1, sorted_output.end(), compare_op);

    // Check if the values are the same
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input));
}
#endif

template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_partial_sort(InputVector     input,
                                 OutputVector    output,
                                 size_t          middle,
                                 CompareFunction compare_op)
{
    compare_partial_sort_cpp_14(input, output, middle, compare_op);
#if CPP17
    // this comparison is only compiled and executed if c++17 is available
    compare_partial_sort_cpp_17(input, output, middle, compare_op);
#else
    ROCPRIM_PRAGMA_MESSAGE("c++17 not available skips direct comparison with std::partial_sort");
#endif
}

TYPED_TEST_SUITE(RocprimDevicePartialSortTests, RocprimDevicePartialSortTestsParams);

TYPED_TEST(RocprimDevicePartialSortTests, PartialSort)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                              = std::remove_cv_t<typename TestFixture::key_type>;
    using compare_function                      = typename TestFixture::compare_function;
    using config                                = typename TestFixture::config;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool        use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            std::vector<size_t> middles = {0};

            if(size > 1)
            {
                middles.push_back(size - 1);
                middles.push_back(test_utils::get_random_value<size_t>(1, size - 1, seed_value));
            }

            for(size_t middle : middles)
            {
                SCOPED_TRACE(testing::Message() << "with middle = " << middle);

                hipStream_t stream = 0; // default
                if(TestFixture::use_graphs)
                {
                    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                }

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
                key_type* d_input;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(key_type)));
                HIP_CHECK(hipMemcpy(d_input,
                                    input.data(),
                                    size * sizeof(key_type),
                                    hipMemcpyHostToDevice));

                key_type* d_output;
                d_output = d_input;

                const auto input_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input);

                compare_function compare_op;

                // Allocate temporary storage
                size_t temp_storage_size_bytes{};
                HIP_CHECK(rocprim::partial_sort<config>(nullptr,
                                                        temp_storage_size_bytes,
                                                        input_it,
                                                        middle,
                                                        size,
                                                        compare_op,
                                                        stream,
                                                        debug_synchronous));

                ASSERT_GT(temp_storage_size_bytes, 0);
                void* d_temp_storage{};
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

                test_utils::GraphHelper gHelper;;
                if(TestFixture::use_graphs)
                {
                    gHelper.startStreamCapture(stream);
                }
                HIP_CHECK(rocprim::partial_sort<config>(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        input_it,
                                                        middle,
                                                        size,
                                                        compare_op,
                                                        stream,
                                                        debug_synchronous));

                HIP_CHECK(hipGetLastError());

                if(TestFixture::use_graphs)
                {
                    gHelper.createAndLaunchGraph(stream);
                }

                std::vector<key_type> output(size);
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    size * sizeof(key_type),
                                    hipMemcpyDeviceToHost));

                compare_partial_sort(input, output, middle, compare_op);

                HIP_CHECK(hipFree(d_input));
                HIP_CHECK(hipFree(d_temp_storage));

                if(TestFixture::use_graphs)
                {
                    gHelper.cleanupGraphHelper();
                    HIP_CHECK(hipStreamDestroy(stream));
                }
            }
        }
    }
}

template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_partial_sort_copy_cpp_14(InputVector     input,
                                             OutputVector    output,
                                             OutputVector    orignal_output,
                                             size_t          middle,
                                             CompareFunction compare_op)
{
    using key_type = typename InputVector::value_type;

    if(input.size() == 0)
    {
        return;
    }
    std::vector<key_type> expected_output;
    // Calculate sorted input results on host
    std::vector<key_type> sorted_input(input);
    std::sort(sorted_input.begin(), sorted_input.end(), compare_op);

    expected_output.insert(expected_output.end(),
                           sorted_input.begin(),
                           sorted_input.begin() + std::min(middle + 1, sorted_input.size()));

    if(middle + 1 < orignal_output.size())
    {
        expected_output.insert(expected_output.end(),
                               orignal_output.begin() + middle + 1,
                               orignal_output.end());
    }

    // Check if the values are the same
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected_output));
}

#if CPP17
template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_partial_sort_copy_cpp_17(InputVector     input,
                                             OutputVector    output,
                                             OutputVector    orignal_output,
                                             size_t          middle,
                                             CompareFunction compare_op)
{
    using key_type = typename InputVector::value_type;

    if(input.size() == 0)
    {
        return;
    }

    // Calculate sorted input results on host
    std::vector<key_type> sorted_output(orignal_output);
    std::partial_sort_copy(input.begin(),
                           input.end(),
                           sorted_output.begin(),
                           sorted_output.begin() + middle + 1,
                           compare_op);

    // Check if the values are the same
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, output));
}
#endif

template<class InputVector, class OutputVector, class CompareFunction>
void inline compare_partial_sort_copy(InputVector     input,
                                      OutputVector    output,
                                      OutputVector    orignal_output,
                                      size_t          middle,
                                      CompareFunction compare_op)
{
    compare_partial_sort_copy_cpp_14(input, output, orignal_output, middle, compare_op);
#if CPP17
    // this comparison is only compiled and executed if c++17 is available
    compare_partial_sort_copy_cpp_17(input, output, orignal_output, middle, compare_op);
#else
    ROCPRIM_PRAGMA_MESSAGE(
        "c++17 not available skips direct comparison with std::partial_sort_copy");
#endif
}

TYPED_TEST(RocprimDevicePartialSortTests, PartialSortCopy)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                              = std::remove_cv_t<typename TestFixture::key_type>;
    using compare_function                      = typename TestFixture::compare_function;
    using config                                = typename TestFixture::config;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool        input_is_const        = std::is_const_v<typename TestFixture::key_type>;
    constexpr bool        use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            std::vector<size_t> middles = {0};

            if(size > 1)
            {
                middles.push_back(size - 1);
                middles.push_back(test_utils::get_random_value<size_t>(1, size - 1, seed_value));
            }

            for(size_t middle : middles)
            {
                SCOPED_TRACE(testing::Message() << "with middle = " << middle);

                hipStream_t stream = 0; // default
                if(TestFixture::use_graphs)
                {
                    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                }

                std::vector<key_type> input;
                std::vector<key_type> output_original;
                if(rocprim::is_floating_point<key_type>::value)
                {
                    input = test_utils::get_random_data<key_type>(size, -1000, 1000, seed_value);
                    output_original
                        = test_utils::get_random_data<key_type>(size, -1000, 1000, seed_value + 1);
                }
                else
                {
                    input = test_utils::get_random_data<key_type>(
                        size,
                        test_utils::numeric_limits<key_type>::min(),
                        test_utils::numeric_limits<key_type>::max(),
                        seed_value);
                    output_original = test_utils::get_random_data<key_type>(
                        size,
                        test_utils::numeric_limits<key_type>::min(),
                        test_utils::numeric_limits<key_type>::max(),
                        seed_value + 1);
                }

                key_type* d_input;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(key_type)));
                HIP_CHECK(hipMemcpy(d_input,
                                    input.data(),
                                    size * sizeof(key_type),
                                    hipMemcpyHostToDevice));

                key_type* d_output;

                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(key_type)));
                HIP_CHECK(hipMemcpy(d_output,
                                    output_original.data(),
                                    size * sizeof(key_type),
                                    hipMemcpyHostToDevice));

                const auto input_it = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(
                    test_utils::wrap_in_const<input_is_const>(d_input));

                compare_function compare_op;

                // Allocate temporary storage
                size_t temp_storage_size_bytes{};

                HIP_CHECK(rocprim::partial_sort_copy<config>(nullptr,
                                                             temp_storage_size_bytes,
                                                             input_it,
                                                             d_output,
                                                             middle,
                                                             size,
                                                             compare_op,
                                                             stream,
                                                             debug_synchronous));

                ASSERT_GT(temp_storage_size_bytes, 0);
                void* d_temp_storage{};
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

                test_utils::GraphHelper gHelper;;
                if(TestFixture::use_graphs)
                {
                    gHelper.startStreamCapture(stream);
                }

                HIP_CHECK(rocprim::partial_sort_copy<config>(d_temp_storage,
                                                             temp_storage_size_bytes,
                                                             input_it,
                                                             d_output,
                                                             middle,
                                                             size,
                                                             compare_op,
                                                             stream,
                                                             debug_synchronous));

                HIP_CHECK(hipGetLastError());

                if(TestFixture::use_graphs)
                {
                    gHelper.createAndLaunchGraph(stream);
                }

                std::vector<key_type> output(size);
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    size * sizeof(key_type),
                                    hipMemcpyDeviceToHost));

                compare_partial_sort_copy(input, output, output_original, middle, compare_op);

                HIP_CHECK(hipFree(d_input));
                HIP_CHECK(hipFree(d_output));
                HIP_CHECK(hipFree(d_temp_storage));

                if(TestFixture::use_graphs)
                {
                    gHelper.cleanupGraphHelper();
                    HIP_CHECK(hipStreamDestroy(stream));
                }
            }
        }
    }
}
