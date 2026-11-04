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

#include "common_test_header.hpp"

// hipcub API
#include <hipcub/device/device_merge_sort.hpp>

#include "test_utils_data_generation.hpp"

// Only test sizes this big if CheckHugeSizes is set in params.
constexpr size_t huge_size = 1ull << 20;

template<class Key,
         class Value,
         class CompareFunction = test_utils::less,
         bool CheckHugeSizes   = false,
         bool UseGraphs        = false>
struct params
{
    using key_type                         = Key;
    using value_type                       = Value;
    using compare_function                 = CompareFunction;
    static constexpr bool check_huge_sizes = CheckHugeSizes;
    static constexpr bool use_graphs       = UseGraphs;
};

template<class Params>
class HipcubDeviceMergeSort : public ::testing::Test
{
public:
    using params = Params;
};

using Params = ::testing::Types<params<signed char, double, test_utils::greater>,
                                params<int, short>,
                                params<short, int, test_utils::greater>,
                                params<long long, char>,
                                params<double, unsigned int>,
                                params<double, int, test_utils::greater>,
                                params<float, int>,
                                params<test_utils::half, int>,
                                params<test_utils::half, int, test_utils::greater>,
                                params<test_utils::bfloat16, int>,
                                params<test_utils::bfloat16, int, test_utils::greater>,
                                params<int, test_utils::custom_test_type<float>>,
                                params<int, short, test_utils::less, false, true>,

                                // huge sizes to check correctness of more than 1 block per batch
                                params<float, char, test_utils::greater, true>>;

TYPED_TEST_SUITE(HipcubDeviceMergeSort, Params);

TYPED_TEST(HipcubDeviceMergeSort, SortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                  = typename TestFixture::params::key_type;
    using compare_function          = typename TestFixture::params::compare_function;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

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

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            if(size > huge_size && !check_huge_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<key_type> keys_input;
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            key_type* d_keys_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceMergeSort::SortKeys(d_temporary_storage,
                                                        temporary_storage_bytes,
                                                        d_keys_input,
                                                        size,
                                                        compare_function()));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceMergeSort::SortKeys(d_temporary_storage,
                                                        temporary_storage_bytes,
                                                        d_keys_input,
                                                        size,
                                                        compare_function(),
                                                        stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_input,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_input));

            bool is_sorted_result
                = std::is_sorted(keys_output.begin(), keys_output.end(), compare_function());

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(is_sorted_result, true));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceMergeSort, SortKeysCopy)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                  = typename TestFixture::params::key_type;
    using compare_function          = typename TestFixture::params::compare_function;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

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

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            if(size > huge_size && !check_huge_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<key_type> keys_input;
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceMergeSort::SortKeysCopy(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_keys_input,
                                                            d_keys_output,
                                                            size,
                                                            compare_function()));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceMergeSort::SortKeysCopy(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_keys_input,
                                                            d_keys_output,
                                                            size,
                                                            compare_function(),
                                                            stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_output));

            bool is_sorted_result
                = std::is_sorted(keys_output.begin(), keys_output.end(), compare_function());

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(is_sorted_result, true));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceMergeSort, StableSortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                  = typename TestFixture::params::key_type;
    using compare_function          = typename TestFixture::params::compare_function;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

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

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            if(size > huge_size && !check_huge_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            std::vector<key_type> keys_input;
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            key_type* d_keys_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceMergeSort::StableSortKeys(d_temporary_storage,
                                                              temporary_storage_bytes,
                                                              d_keys_input,
                                                              size,
                                                              compare_function()));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceMergeSort::SortKeys(d_temporary_storage,
                                                        temporary_storage_bytes,
                                                        d_keys_input,
                                                        size,
                                                        compare_function(),
                                                        stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            std::stable_sort(expected.begin(), expected.end(), compare_function());

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_input,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_input));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceMergeSort, StableSortKeysCopy)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                  = typename TestFixture::params::key_type;
    using compare_function          = typename TestFixture::params::compare_function;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

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

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            if(size > huge_size && !check_huge_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<key_type> keys_input;
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceMergeSort::StableSortKeysCopy(d_temporary_storage,
                                                                  temporary_storage_bytes,
                                                                  d_keys_input,
                                                                  d_keys_output,
                                                                  size,
                                                                  compare_function()));
            ASSERT_GT(temporary_storage_bytes, 0U);
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceMergeSort::StableSortKeysCopy(d_temporary_storage,
                                                                  temporary_storage_bytes,
                                                                  d_keys_input,
                                                                  d_keys_output,
                                                                  size,
                                                                  compare_function(),
                                                                  stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            std::stable_sort(expected.begin(), expected.end(), compare_function());

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_output));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceMergeSort, SortPairs)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                  = typename TestFixture::params::key_type;
    using value_type                = typename TestFixture::params::value_type;
    using compare_function          = typename TestFixture::params::compare_function;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

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

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            if(size > huge_size && !check_huge_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            compare_function compare_op;
            using key_value = std::pair<key_type, value_type>;

            // Generate data
            std::vector<key_type> keys_input;
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            std::vector<value_type> values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type* d_keys_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            // -- begin: Calculate expected results on host -- for performance reasons after a hipMalloc --
            using key_value = std::pair<key_type, value_type>;
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            // --
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            // -- continue: Calculate expected results on host -- for performance reasons after a hipMalloc --
            std::stable_sort(expected.begin(),
                             expected.end(),
                             [compare_op](const key_value& a, const key_value& b)
                             { return compare_op(a.first, b.first); });
            // --
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceMergeSort::SortPairs(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_keys_input,
                                                         d_values_input,
                                                         size,
                                                         compare_op,
                                                         stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceMergeSort::SortPairs(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_keys_input,
                                                         d_values_input,
                                                         size,
                                                         compare_op,
                                                         stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_input,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_input,
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_input));
            // -- continue: Calculate expected results on host -- for performance reasons after a hipFree --
            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }
            // -- end
            HIP_CHECK(hipFree(d_values_input));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, values_expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceMergeSort, SortPairsCopy)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                  = typename TestFixture::params::key_type;
    using value_type                = typename TestFixture::params::value_type;
    using compare_function          = typename TestFixture::params::compare_function;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

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

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            if(size > huge_size && !check_huge_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            compare_function compare_op;
            using key_value = std::pair<key_type, value_type>;

            // Generate data
            std::vector<key_type> keys_input;
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            std::vector<value_type> values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            // -- begin: Calculate expected results on host -- for performance reasons after a hipMalloc --
            using key_value = std::pair<key_type, value_type>;
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            // --
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            // -- continue: Calculate expected results on host -- for performance reasons after a hipMalloc --
            std::stable_sort(expected.begin(),
                             expected.end(),
                             [compare_op](const key_value& a, const key_value& b)
                             { return compare_op(a.first, b.first); });
            // --
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            value_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            // -- continue: Calculate expected results on host -- for performance reasons after a hipFree --
            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }
            // -- end
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceMergeSort::SortPairsCopy(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_keys_input,
                                                             d_values_input,
                                                             d_keys_output,
                                                             d_values_output,
                                                             size,
                                                             compare_op,
                                                             stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceMergeSort::SortPairsCopy(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_keys_input,
                                                             d_values_input,
                                                             d_keys_output,
                                                             d_values_output,
                                                             size,
                                                             compare_op,
                                                             stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_values_input));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, values_expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceMergeSort, StableSortPairs)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                  = typename TestFixture::params::key_type;
    using value_type                = typename TestFixture::params::value_type;
    using compare_function          = typename TestFixture::params::compare_function;
    constexpr bool check_huge_sizes = TestFixture::params::check_huge_sizes;

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

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            if(size > huge_size && !check_huge_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            compare_function compare_op;
            using key_value = std::pair<key_type, value_type>;

            // Generate data
            std::vector<key_type> keys_input;
            keys_input
                = test_utils::get_random_data<key_type>(size,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            std::vector<value_type> values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type* d_keys_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            // -- begin: Calculate expected results on host -- for performance reasons after a hipMalloc --
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            // --
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            // -- continue: Calculate expected results on host -- for performance reasons after a hipMalloc --
            std::stable_sort(expected.begin(),
                             expected.end(),
                             [compare_op](const key_value& a, const key_value& b)
                             { return compare_op(a.first, b.first); });
            // --
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceMergeSort::StableSortPairs(d_temporary_storage,
                                                               temporary_storage_bytes,
                                                               d_keys_input,
                                                               d_values_input,
                                                               size,
                                                               compare_op,
                                                               stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(hipcub::DeviceMergeSort::StableSortPairs(d_temporary_storage,
                                                               temporary_storage_bytes,
                                                               d_keys_input,
                                                               d_values_input,
                                                               size,
                                                               compare_op,
                                                               stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_input,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_input,
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_input));
            // -- continue: Calculate expected results on host -- for performance reasons after a hipFree --
            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }
            // -- end
            HIP_CHECK(hipFree(d_values_input));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, values_expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}
