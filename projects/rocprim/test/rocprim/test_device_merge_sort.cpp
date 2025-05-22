/// MIT License
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

#include "../../common/utils.hpp"
#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required rocprim headers
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_merge_sort.hpp>
#include <rocprim/device/device_merge_sort_config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>

// Params for tests
template<
    class KeyType,
    class ValueType = KeyType,
    class CompareFunction = ::rocprim::less<KeyType>,
    bool UseGraphs = false
>
struct DeviceSortParams
{
    using key_type = KeyType;
    using value_type = ValueType;
    using compare_function = CompareFunction;
    static constexpr bool use_graphs = UseGraphs;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceSortTests : public ::testing::Test
{
public:
    using key_type = typename Params::key_type;
    using value_type = typename Params::value_type;
    using compare_function = typename Params::compare_function;
    const bool debug_synchronous = false;
    bool use_graphs = Params::use_graphs;
};

using RocprimDeviceSortTestsParams = ::testing::Types<
    DeviceSortParams<unsigned short, int>,
    DeviceSortParams<signed char, common::custom_type<float, float, true>>,
    DeviceSortParams<int>,
    DeviceSortParams<common::custom_type<int, int, true>>,
    DeviceSortParams<unsigned long>,
    DeviceSortParams<long long>,
    DeviceSortParams<float, double>,
    DeviceSortParams<int8_t, int8_t>,
    DeviceSortParams<uint8_t, uint8_t>,
    DeviceSortParams<rocprim::half, rocprim::half, rocprim::less<rocprim::half>>,
    DeviceSortParams<rocprim::bfloat16, rocprim::bfloat16, rocprim::less<rocprim::bfloat16>>,
    DeviceSortParams<int, float, ::rocprim::greater<int>>,
    DeviceSortParams<short, common::custom_type<int, int, true>>,
    DeviceSortParams<double, common::custom_type<double, double, true>>,
    DeviceSortParams<common::custom_type<float, float, true>,
                     common::custom_type<double, double, true>>,
    DeviceSortParams<int, test_utils::custom_float_type>,
    DeviceSortParams<test_utils::custom_test_array_type<int, 4>>,
    DeviceSortParams<int, int, ::rocprim::less<int>, true>,
    DeviceSortParams<int, common::custom_huge_type<2048, float>>,
    DeviceSortParams<common::custom_huge_type<2048, float>>>;

static_assert(std::is_trivially_copyable<test_utils::custom_float_type>::value,
              "Type must be trivially copyable to cover merge sort specialized kernel");

TYPED_TEST_SUITE(RocprimDeviceSortTests, RocprimDeviceSortTestsParams);

TYPED_TEST(RocprimDeviceSortTests, SortKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using compare_function = typename TestFixture::compare_function;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    bool in_place = false;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if (TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            in_place = !in_place;

            // Generate data
            std::vector<key_type> input = test_utils::get_random_data_wrapped<key_type>(
                size,
                -100,
                100,
                seed_value); // float16 can't exceed 65504

            common::device_ptr<key_type> d_input(input);
            common::device_ptr<key_type> d_output_alloc;
            d_output_alloc.resize(in_place ? 0 : size);
            common::device_ptr<key_type>& d_output = in_place ? d_input : d_output_alloc;

            // compare function
            compare_function compare_op;

            // Calculate expected results on host
            std::vector<key_type> expected(input);
            std::stable_sort(expected.begin(), expected.end(), compare_op);

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(rocprim::merge_sort(nullptr,
                                          temp_storage_size_bytes,
                                          d_input.get(),
                                          d_output.get(),
                                          input.size(),
                                          compare_op,
                                          stream,
                                          debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::merge_sort(d_temp_storage.get(),
                                          temp_storage_size_bytes,
                                          d_input.get(),
                                          d_output.get(),
                                          input.size(),
                                          compare_op,
                                          stream,
                                          debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

            if (TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

// This test also ensures that merge_sort is stable
TYPED_TEST(RocprimDeviceSortTests, SortKeyValue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using compare_function = typename TestFixture::compare_function;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    bool in_place = false;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if (TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            in_place = !in_place;

            // Generate data
            std::vector<key_type> keys_input = test_utils::get_random_data_wrapped<key_type>(
                size,
                -100,
                100,
                seed_value); // float16 can't exceed 65504

            std::vector<value_type> values_input(size);
            test_utils::iota(values_input.begin(), values_input.end(), 0);

            common::device_ptr<key_type> d_keys_input(keys_input);
            common::device_ptr<key_type> d_keys_output_alloc;
            d_keys_output_alloc.resize(in_place ? 0 : size);
            common::device_ptr<key_type>& d_keys_output
                = in_place ? d_keys_input : d_keys_output_alloc;

            common::device_ptr<value_type> d_values_input(values_input);
            common::device_ptr<value_type> d_values_output_alloc;
            d_values_output_alloc.resize(in_place ? 0 : size);
            common::device_ptr<value_type>& d_values_output
                = in_place ? d_values_input : d_values_output_alloc;

            // compare function
            compare_function compare_op;

            // Calculate expected results on host
            using key_value = std::pair<key_type, value_type>;
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            std::stable_sort(expected.begin(),
                             expected.end(),
                             [compare_op](const key_value& a, const key_value& b)
                             { return compare_op(a.first, b.first); });

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(rocprim::merge_sort(nullptr,
                                          temp_storage_size_bytes,
                                          d_keys_input.get(),
                                          d_keys_output.get(),
                                          d_values_input.get(),
                                          d_values_output.get(),
                                          keys_input.size(),
                                          compare_op,
                                          stream,
                                          debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::merge_sort(d_temp_storage.get(),
                                          temp_storage_size_bytes,
                                          d_keys_input.get(),
                                          d_keys_output.get(),
                                          d_values_input.get(),
                                          d_values_output.get(),
                                          keys_input.size(),
                                          compare_op,
                                          stream,
                                          debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto keys_output   = d_keys_output.load();
            const auto values_output = d_values_output.load();

            // Check if output values are as expected
            std::vector<key_type> expected_key(expected.size());
            std::vector<value_type> expected_value(expected.size());
            for(size_t i = 0; i < expected.size(); i++)
            {
                expected_key[i] = expected[i].first;
                expected_value[i] = expected[i].second;
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected_key));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, expected_value));

            if (TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

void testLargeIndices()
{
    using key_type = uint8_t;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream            = 0; // default
    const bool  debug_synchronous = false;

    // Use this custom config with smaller items_per_thread to launch more than 2^32 threads with
    // at least some sizes that fit into device memory.
    using config = rocprim::merge_sort_config<256, 256, 1, 128, 128, 1, (1 << 17)>;

    // On Windows, sizes above 2^34 cause issues that we can't currently catch by examining
    // the hipMalloc return value or querying available memory. Workaround this for now
    // by setting a different maximum size for that platform.
#if defined(_WIN32)
    const size_t max_pow2 = 34;
#else
    const size_t max_pow2 = 37;
#endif
    for(size_t size : test_utils::get_large_sizes<max_pow2>(seeds[0]))
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const auto input
            = rocprim::make_transform_iterator(rocprim::make_counting_iterator<size_t>(0),
                                               rocprim::identity<key_type>());

        key_type*  d_output;
        hipError_t malloc_status = common::hipMallocHelper(&d_output, size * sizeof(*d_output));
        if(malloc_status == hipErrorOutOfMemory)
        {
            std::cout << "Out of memory. Skipping size = " << size << std::endl;
            break;
        }
        HIP_CHECK(malloc_status);

        // compare function
        rocprim::less<key_type> compare_op;

        // temp storage
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::merge_sort<config>(d_temp_storage,
                                              temp_storage_size_bytes,
                                              input,
                                              d_output,
                                              size,
                                              compare_op,
                                              stream,
                                              debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        malloc_status = common::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes);
        if(malloc_status == hipErrorOutOfMemory)
        {
            std::cout << "Out of memory. Skipping size = " << size << std::endl;
            HIP_CHECK(hipFree(d_output));
            break;
        }
        HIP_CHECK(malloc_status);

        // Run
        HIP_CHECK(rocprim::merge_sort<config>(d_temp_storage,
                                              temp_storage_size_bytes,
                                              input,
                                              d_output,
                                              size,
                                              compare_op,
                                              stream,
                                              debug_synchronous));
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        std::vector<key_type> output(size);
        HIP_CHECK(hipMemcpy(output.data(),
                            d_output,
                            output.size() * sizeof(*d_output),
                            hipMemcpyDeviceToHost));

        // Check if output values are as expected
        const size_t unique_keys    = size_t(rocprim::numeric_limits<key_type>::max()) + 1;
        const size_t segment_length = rocprim::detail::ceiling_div(size, unique_keys);
        const size_t full_segments  = size % unique_keys == 0 ? unique_keys : size % unique_keys;
        for(size_t i = 0; i < size; i += 4321)
        {
            key_type expected;
            if(i / segment_length < full_segments)
            {
                expected = key_type(i / segment_length);
            }
            else
            {
                expected = key_type((i - full_segments * segment_length) / (segment_length - 1)
                                    + full_segments);
            }
            ASSERT_EQ(output[i], expected) << "with index = " << i;
        }

        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}

TEST(RocprimDeviceSortTests, LargeIndices)
{
    testLargeIndices();
}
