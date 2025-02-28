// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// required hipcub headers
#include <cstddef>
#include <hipcub/device/device_for.hpp>

// Params for tests
template<class InputType, bool UseGraphs = false>
struct DeviceForParams
{
    using input_type                 = InputType;
    static constexpr bool use_graphs = UseGraphs;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceForTests : public ::testing::Test
{
public:
    using input_type                        = typename Params::input_type;
    static constexpr bool use_graphs        = Params::use_graphs;
    static constexpr bool debug_synchronous = false;
};

using custom_short2  = test_utils::custom_test_type<short>;
using custom_int2    = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

using HipcubDeviceForTestsParams = ::testing::Types<DeviceForParams<int>,
                                                    DeviceForParams<int8_t>,
                                                    DeviceForParams<uint8_t>,
                                                    DeviceForParams<unsigned long>,
                                                    DeviceForParams<short>,
                                                    DeviceForParams<custom_short2>,
                                                    DeviceForParams<float>,
                                                    DeviceForParams<custom_double2>,
                                                    DeviceForParams<test_utils::half>,
                                                    DeviceForParams<test_utils::bfloat16>,
                                                    DeviceForParams<int, true>>;

TYPED_TEST_SUITE(HipcubDeviceForTests, HipcubDeviceForTestsParams);

template<class T>
struct plus
{
    HIPCUB_HOST_DEVICE
    inline void
        operator()(T& a) const
    {
        a = a + T(5);
    }
};

TYPED_TEST(HipcubDeviceForTests, ForEach)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<T> output(input.size(), T(0));

            T* d_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

            // Calculate expected results on host
            std::vector<T> expected(input);
            std::for_each(expected.begin(), expected.end(), plus<T>());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::ForEach(d_input, d_input + size, plus<T>(), stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_input,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<class T>
struct odd_count_device_t
{
    unsigned int* d_count;

    HIPCUB_DEVICE
    void          operator()(T i)
    {
        if(i % 2 == 1)
        {
            atomicAdd(d_count, 1);
        }
    }
};

template<class T>
struct odd_count_host_t
{
    unsigned int* d_count;

    void operator()(T i)
    {
        if(i % 2 == 1)
        {
            (*d_count)++;
        }
    }
};

TEST(HipcubDeviceForTestsTempStore, ForEachTempStore)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = int;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T>      input    = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            unsigned int        expected = 0;
            odd_count_host_t<T> host_op{&expected};

            T* d_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            unsigned int* d_count;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(unsigned int)));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(T)));
            odd_count_device_t<T> device_op{d_count};

            // Calculate expected results on host
            std::for_each(input.begin(), input.end(), host_op);

            // Run
            HIP_CHECK(hipcub::ForEach(d_input, d_input + size, device_op, stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if have same number of odd numbers
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            HIP_CHECK(hipFree(d_input));
        }
    }
}

TYPED_TEST(HipcubDeviceForTests, ForEachN)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            size_t n = size / 2;

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<T> output(input.size(), T(0));

            T* d_input;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

            // Calculate expected results on host
            std::vector<T> expected(input);
            std::for_each(expected.begin(), expected.begin() + n, plus<T>());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::ForEachN(d_input, n, plus<T>(), stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_input,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<class T>
struct count_device_t
{
    T*   d_count;

    HIPCUB_DEVICE
    void operator()(T i)
    {
        atomicAdd(d_count + i, 1);
    }
};

TEST(HipcubDeviceBulk, Bulk)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = int;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            T* d_count;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(T) * size));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(T) * size));
            count_device_t<T> device_op{d_count};

            // Run
            HIP_CHECK(hipcub::Bulk(size, device_op, stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            std::vector<T> output(size, T(0));
            HIP_CHECK(hipMemcpy(output.data(), d_count, sizeof(T) * size, hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<T> expected(size, T(1));

            // Check if whole array is filled with ones
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(expected, output));
        }
    }
}
