// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/device/device_merge.hpp>
#include <hipcub/iterator/counting_input_iterator.hpp>

#include "identity_iterator.hpp"
#include "test_utils_data_generation.hpp"

template<class Key, class Value, class CompareFunction = test_utils::less, bool UseGraphs = false>
struct params
{
    using key_type                   = Key;
    using value_type                 = Value;
    using compare_function           = CompareFunction;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
class HipcubDeviceMerge : public ::testing::Test
{
public:
    using params = Params;
};

using Params
    = ::testing::Types<params<signed char, double, test_utils::greater>,
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
                       params<int, test_utils::custom_test_type<float>>
#ifdef __HIP_PLATFORM_AMD__
                       ,
                       params<int,
                              short,
                              test_utils::less,
                              true> // The graph support for DeviceMerge is broken on Titan V
#endif
                       >;

// size1, size2
std::vector<std::tuple<size_t, size_t>> get_sizes()
{
    std::vector<std::tuple<size_t, size_t>> sizes = {std::make_tuple(0, 0),
                                                     std::make_tuple(2, 1),
                                                     std::make_tuple(10, 10),
                                                     std::make_tuple(111, 111),
                                                     std::make_tuple(128, 1289),
                                                     std::make_tuple(12, 1000),
                                                     std::make_tuple(123, 3000),
                                                     std::make_tuple(1024, 512),
                                                     std::make_tuple(2345, 49),
                                                     std::make_tuple(17867, 41),
                                                     std::make_tuple(17867, 34567),
                                                     std::make_tuple(34567, (1 << 17) - 1220),
                                                     std::make_tuple(924353, 1723454)};
    return sizes;
}

TYPED_TEST_SUITE(HipcubDeviceMerge, Params);

TYPED_TEST(HipcubDeviceMerge, MergeKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type         = typename TestFixture::params::key_type;
    using compare_function = typename TestFixture::params::compare_function;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(auto sizes : get_sizes())
    {
        if((std::get<0>(sizes) == 0 || std::get<1>(sizes) == 0) && test_common_utils::use_hmm())
        {
            // hipMallocManaged() currently doesnt support zero byte allocation
            continue;
        }
        SCOPED_TRACE(testing::Message() << "with sizes = {" << std::get<0>(sizes) << ", "
                                        << std::get<1>(sizes) << "}");

        const size_t size1 = std::get<0>(sizes);
        const size_t size2 = std::get<1>(sizes);

        // compare function
        compare_function compare_op;

        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

            // Generate data
            std::vector<key_type> keys_input1
                = test_utils::get_random_data<key_type>(size1,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            std::vector<key_type> keys_input2
                = test_utils::get_random_data<key_type>(size2,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);

            std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
            std::sort(keys_input2.begin(), keys_input2.end(), compare_op);
            std::vector<key_type> keys_output(size1 + size2, (key_type)0);

            // Calculate expected results on host
            std::vector<key_type> expected(keys_output.size());
            std::merge(keys_input1.begin(),
                       keys_input1.end(),
                       keys_input2.begin(),
                       keys_input2.end(),
                       expected.begin(),
                       compare_op);

            key_type* d_keys_input1;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input1, size1 * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input1,
                                keys_input1.data(),
                                size1 * sizeof(key_type),
                                hipMemcpyHostToDevice));

            key_type* d_keys_input2;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input2, size2 * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input2,
                                keys_input2.data(),
                                size2 * sizeof(key_type),
                                hipMemcpyHostToDevice));

            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output,
                                                         keys_output.size() * sizeof(key_type)));

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(hipcub::DeviceMerge::MergeKeys(nullptr,
                                                     temp_storage_size_bytes,
                                                     d_keys_input1,
                                                     size1,
                                                     d_keys_input2,
                                                     size2,
                                                     d_keys_output,
                                                     compare_op,
                                                     stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            void* d_temporary_storage;
            // allocate temporary storage
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(hipcub::DeviceMerge::MergeKeys(d_temporary_storage,
                                                     temp_storage_size_bytes,
                                                     d_keys_input1,
                                                     size1,
                                                     d_keys_input2,
                                                     size2,
                                                     d_keys_output,
                                                     compare_op,
                                                     stream));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                keys_output.size() * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            // Check if keys_output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
            HIP_CHECK(hipFree(d_keys_input1));
            HIP_CHECK(hipFree(d_keys_input2));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_temporary_storage));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(HipcubDeviceMerge, MergePairs)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type         = typename TestFixture::params::key_type;
    using value_type       = typename TestFixture::params::value_type;
    using compare_function = typename TestFixture::params::compare_function;

    using key_value = std::pair<key_type, value_type>;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(auto sizes : get_sizes())
    {
        if((std::get<0>(sizes) == 0 || std::get<1>(sizes) == 0) && test_common_utils::use_hmm())
        {
            // hipMallocManaged() currently doesnt support zero byte allocation
            continue;
        }
        SCOPED_TRACE(testing::Message() << "with sizes = {" << std::get<0>(sizes) << ", "
                                        << std::get<1>(sizes) << "}");

        const size_t size1 = std::get<0>(sizes);
        const size_t size2 = std::get<1>(sizes);

        // compare function
        compare_function compare_op;

        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

            // Generate data
            std::vector<key_type> keys_input1
                = test_utils::get_random_data<key_type>(size1,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);
            std::vector<key_type> keys_input2
                = test_utils::get_random_data<key_type>(size2,
                                                        test_utils::numeric_limits<key_type>::min(),
                                                        test_utils::numeric_limits<key_type>::max(),
                                                        seed_value + seed_value_addition);

            std::sort(keys_input1.begin(), keys_input1.end(), compare_op);
            std::sort(keys_input2.begin(), keys_input2.end(), compare_op);

            std::vector<value_type> values_input1(size1);
            std::vector<value_type> values_input2(size2);
            std::iota(values_input1.begin(), values_input1.end(), 0);
            std::iota(values_input2.begin(), values_input2.end(), size1);

            std::vector<value_type> values_output(size1 + size2, (value_type)0);
            std::vector<key_type>   keys_output(size1 + size2, (key_type)0);

            // Calculate expected results on host
            std::vector<key_value> vector1(size1);
            std::vector<key_value> vector2(size2);

            for(size_t i = 0; i < size1; i++)
            {
                vector1[i] = key_value(keys_input1[i], values_input1[i]);
            }
            for(size_t i = 0; i < size2; i++)
            {
                vector2[i] = key_value(keys_input2[i], values_input2[i]);
            }

            std::vector<key_value> expected(size1 + size2);
            std::merge(vector1.begin(),
                       vector1.end(),
                       vector2.begin(),
                       vector2.end(),
                       expected.begin(),
                       [compare_op](const key_value& a, const key_value& b)
                       { return compare_op(a.first, b.first); });

            key_type* d_keys_input1;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input1, size1 * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input1,
                                keys_input1.data(),
                                size1 * sizeof(key_type),
                                hipMemcpyHostToDevice));

            key_type* d_keys_input2;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input2, size2 * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input2,
                                keys_input2.data(),
                                size2 * sizeof(key_type),
                                hipMemcpyHostToDevice));

            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output,
                                                         keys_output.size() * sizeof(key_type)));

            value_type* d_values_input1;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input1, size1 * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input1,
                                values_input1.data(),
                                size1 * sizeof(value_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input2;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input2, size2 * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input2,
                                values_input2.data(),
                                size2 * sizeof(value_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output,
                                                   values_output.size() * sizeof(value_type)));

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(hipcub::DeviceMerge::MergePairs(nullptr,
                                                      temp_storage_size_bytes,
                                                      d_keys_input1,
                                                      d_values_input1,
                                                      size1,
                                                      d_keys_input2,
                                                      d_values_input2,
                                                      size2,
                                                      d_keys_output,
                                                      d_values_output,
                                                      compare_op,
                                                      stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            void* d_temporary_storage;
            // allocate temporary storage
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(hipcub::DeviceMerge::MergePairs(d_temporary_storage,
                                                      temp_storage_size_bytes,
                                                      d_keys_input1,
                                                      d_values_input1,
                                                      size1,
                                                      d_keys_input2,
                                                      d_values_input2,
                                                      size2,
                                                      d_keys_output,
                                                      d_values_output,
                                                      compare_op,
                                                      stream));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                keys_output.size() * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                values_output.size() * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            // Check if keys_output values are as expected
            std::vector<key_type>   expected_key(expected.size());
            std::vector<value_type> expected_value(expected.size());
            for(size_t i = 0; i < expected.size(); i++)
            {
                expected_key[i]   = expected[i].first;
                expected_value[i] = expected[i].second;
            }

            // Check if keys_output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected_key));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, expected_value));

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipFree(d_keys_input1));
            HIP_CHECK(hipFree(d_keys_input2));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_input1));
            HIP_CHECK(hipFree(d_values_input2));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_temporary_storage));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

std::vector<std::tuple<size_t, size_t>> get_large_sizes()
{
    std::vector<std::tuple<size_t, size_t>> sizes = {std::make_tuple(0, 0),
                                                     std::make_tuple(1 << 28, 1 << 28),
                                                     std::make_tuple(1 << 30, 5000)};
    return sizes;
}

TEST(HipcubDeviceMerge, MergeLargeSizeIterators)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type         = int;
    using compare_function = test_utils::less;

    hipStream_t stream = 0; // default

    for(auto sizes : get_large_sizes())
    {
        if((std::get<0>(sizes) == 0 || std::get<1>(sizes) == 0) && test_common_utils::use_hmm())
        {
            // hipMallocManaged() currently doesnt support zero byte allocation
            continue;
        }
        SCOPED_TRACE(testing::Message() << "with sizes = {" << std::get<0>(sizes) << ", "
                                        << std::get<1>(sizes) << "}");

        const size_t size1       = std::get<0>(sizes);
        const size_t size2       = std::get<1>(sizes);
        const size_t output_size = size1 + size2;

        // compare function
        compare_function compare_op;

        // Generate data
        const auto input1 = rocprim::counting_iterator<key_type>(key_type{0});
        const auto input2
            = rocprim::counting_iterator<key_type>(key_type{static_cast<key_type>(size1)});
        std::vector<key_type> vec_input1(size1);
        std::vector<key_type> vec_input2(size2);
        std::iota(vec_input1.begin(), vec_input1.end(), 0);
        std::iota(vec_input2.begin(), vec_input2.end(), size1);

        // Calculate expected results on host
        std::vector<key_type> expected(output_size);
        std::merge(vec_input1.begin(),
                   vec_input1.end(),
                   vec_input2.begin(),
                   vec_input2.end(),
                   expected.begin(),
                   compare_op);

        key_type* d_keys_output;
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_keys_output, output_size * sizeof(key_type)));

        // Get size of d_temp_storage
        size_t temp_storage_size_bytes;
        HIP_CHECK(
            hipcub::DeviceMerge::MergeKeys(nullptr,
                                           temp_storage_size_bytes,
                                           input1,
                                           size1,
                                           input2,
                                           size2,
                                           test_utils::identity_iterator<key_type>(d_keys_output),
                                           compare_op,
                                           stream));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        void* d_temporary_storage;
        // allocate temporary storage
        HIP_CHECK(
            test_common_utils::hipMallocHelper(&d_temporary_storage, temp_storage_size_bytes));

        // Run
        HIP_CHECK(
            hipcub::DeviceMerge::MergeKeys(d_temporary_storage,
                                           temp_storage_size_bytes,
                                           input1,
                                           size1,
                                           input2,
                                           size2,
                                           test_utils::identity_iterator<key_type>(d_keys_output),
                                           compare_op,
                                           stream));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<key_type> keys_output(output_size, (key_type)0);
        HIP_CHECK(hipMemcpy(keys_output.data(),
                            d_keys_output,
                            keys_output.size() * sizeof(key_type),
                            hipMemcpyDeviceToHost));

        // Check if keys_output values are as expected
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));

        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_temporary_storage));
    }
}
