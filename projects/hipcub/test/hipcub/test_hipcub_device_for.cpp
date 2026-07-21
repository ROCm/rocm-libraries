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
#include <hipcub/device/device_for.hpp>
#include <hipcub/iterator/counting_input_iterator.hpp>

#include <cstddef>
#include <cstdint>
#include <new>

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
            if(TestFixture::use_graphs)
            {
                // Make sure previous ops on default stream (mem transfers) are done
                HIP_CHECK(hipStreamSynchronize(0));
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(hipcub::DeviceFor::ForEach(d_input, d_input + size, plus<T>(), stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_input,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));

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

struct count_device_t
{
    unsigned int* d_count;

    template<class T>
    HIPCUB_DEVICE
    void operator()(T)
    {
        atomicAdd(d_count, 1);
    }
};

struct count_host_t
{
    unsigned int* d_count;

    template<class T>
    void operator()(T)
    {
        (*d_count)++;
    }
};

TEST(HipcubDeviceForTests, ForEachTempStore)
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
            count_host_t        host_op{&expected};

            // Device pointers
            T*            d_input;
            unsigned int* d_count;

            // Allocate memory
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(unsigned int)));

            // Prepare ForEach
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));
            count_device_t device_op{d_count};

            // Calculate temporary storage
            size_t temp_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceFor::ForEach(nullptr,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_input + size,
                                                 device_op,
                                                 stream));
            void* d_temp_storage{};

            // Allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));

            // Calculate expected results on host
            std::for_each(input.begin(), input.end(), host_op);

            // Run ForEach
            HIP_CHECK(hipcub::DeviceFor::ForEach(d_temp_storage,
                                                 temp_storage_bytes,
                                                 d_input,
                                                 d_input + size,
                                                 device_op,
                                                 stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

            // Check if have same number of odd numbers
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_temp_storage));
            HIP_CHECK(hipFree(d_count));
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
            if(TestFixture::use_graphs)
            {
                // Make sure previous ops on default stream (mem transfers) are done
                HIP_CHECK(hipStreamSynchronize(0));
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(hipcub::DeviceFor::ForEachN(d_input, n, plus<T>(), stream));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_input,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipFree(d_input));
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(HipcubDeviceForTests, ForEachNTempStore)
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

            size_t n = size / 2;

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<T> output(input.size(), T(0));

            // Device pointers
            T* d_input;

            // Allocate memory
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));

            // Prepare ForEachN
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

            // Calculate temporary storage
            size_t temp_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceFor::ForEachN(nullptr,
                                                  temp_storage_bytes,
                                                  d_input,
                                                  n,
                                                  plus<T>(),
                                                  stream));
            void* d_temp_storage{};

            // Allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));

            // Calculate expected results on host
            std::vector<T> expected(input);
            std::for_each(expected.begin(), expected.begin() + n, plus<T>());

            // Run ForEachN
            HIP_CHECK(hipcub::DeviceFor::ForEachN(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_input,
                                                  n,
                                                  plus<T>(),
                                                  stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(output.data(),
                                d_input,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }
}

TYPED_TEST(HipcubDeviceForTests, ForEachCopy)
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
            std::vector<T> input    = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            unsigned int   expected = 0;
            count_host_t   host_op{&expected};

            T*            d_input;
            unsigned int* d_count;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));
            count_device_t device_op{d_count};

            // Calculate expected results on host
            std::for_each(input.begin(), input.end(), host_op);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                // Make sure previous ops on default stream (mem transfers) are done
                HIP_CHECK(hipStreamSynchronize(0));
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(hipcub::DeviceFor::ForEachCopy(d_input, d_input + size, device_op, stream));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_count));
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(HipcubDeviceForTests, ForEachCopyTempStore)
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
            std::vector<T> input    = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            unsigned int   expected = 0;
            count_host_t   host_op{&expected};

            // Device pointers
            T*            d_input;
            unsigned int* d_count;

            // Allocate memory
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(unsigned int)));

            // Prepare ForEachCopy
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));
            count_device_t device_op{d_count};

            // Calculate temporary storage
            size_t temp_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceFor::ForEachCopy(nullptr,
                                                     temp_storage_bytes,
                                                     d_input,
                                                     d_input + size,
                                                     device_op,
                                                     stream));
            void* d_temp_storage{};

            // Allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));

            // Calculate expected results on host
            std::for_each(input.begin(), input.end(), host_op);

            // Run
            HIP_CHECK(hipcub::DeviceFor::ForEachCopy(d_temp_storage,
                                                     temp_storage_bytes,
                                                     d_input,
                                                     d_input + size,
                                                     device_op,
                                                     stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

            // Check if have same number of odd numbers
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_temp_storage));
            HIP_CHECK(hipFree(d_count));
        }
    }
}

TYPED_TEST(HipcubDeviceForTests, ForEachCopyN)
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
            std::vector<T> input    = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            unsigned int   expected = 0;
            count_host_t   host_op{&expected};

            T*            d_input;
            unsigned int* d_count;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, input.size() * sizeof(int)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));
            count_device_t device_op{d_count};

            // Calculate expected results on host
            std::for_each(input.begin(), input.end(), host_op);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                // Make sure previous ops on default stream (mem transfers) are done
                HIP_CHECK(hipStreamSynchronize(0));
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(hipcub::DeviceFor::ForEachCopyN(d_input, size, device_op, stream));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_count));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST(HipcubDeviceForTests, ForCountingIterator)
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
            unsigned int expected = 0;
            count_host_t host_op{&expected};

            // Device pointers
            unsigned int* d_count;
            const auto    it = rocprim::counting_iterator<T>{0};
            // Allocate memory
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(unsigned int)));

            HIP_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));
            count_device_t device_op{d_count};

            // Calculate expected results on host
            std::for_each(it, it + size, host_op);

            HIP_CHECK(hipcub::DeviceFor::ForEach(it, it + size, device_op, stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

            // Check if have same number of odd numbers
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            HIP_CHECK(hipFree(d_count));
        }
    }
}

TEST(HipcubDeviceForTests, ForCopyCountingIterator)
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
            unsigned int expected = 0;
            count_host_t host_op{&expected};

            // Device pointers
            unsigned int* d_count;
            const auto    it = rocprim::counting_iterator<T>{0};

            // Allocate memory
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(unsigned int)));

            HIP_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));
            count_device_t device_op{d_count};

            // Calculate expected results on host
            std::for_each(it, it + size, host_op);

            HIP_CHECK(hipcub::DeviceFor::ForEachCopy(it, it + size, device_op, stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

            // Check if have same number of odd numbers
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            HIP_CHECK(hipFree(d_count));
        }
    }
}

TEST(HipcubDeviceForTests, ForEachCopyNTempStore)
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
            std::vector<T> input    = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            unsigned int   expected = 0;
            count_host_t   host_op{&expected};

            // Device pointers
            T*            d_input;
            unsigned int* d_count;

            // Allocate memory
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(unsigned int)));

            // Prepare ForEachCopyN
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));
            count_device_t device_op{d_count};

            // Calculate temporary storage
            size_t temp_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceFor::ForEachCopyN(nullptr,
                                                      temp_storage_bytes,
                                                      d_input,
                                                      size,
                                                      device_op,
                                                      stream));
            void* d_temp_storage{};

            // Allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));

            // Calculate expected results on host
            std::for_each(input.begin(), input.end(), host_op);

            // Run ForEachCopyN
            HIP_CHECK(hipcub::DeviceFor::ForEachCopyN(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_input,
                                                      size,
                                                      device_op,
                                                      stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            unsigned int h_count;
            HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

            // Check if have same number of odd numbers
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_count, expected));

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_temp_storage));
            HIP_CHECK(hipFree(d_count));
        }
    }
}

// ForEachInExtents only enables when the cccl mdspan extension is enabled
#if(defined(__HIP_PLATFORM_NVIDIA__) && defined(__cccl_lib_mdspan)) || defined(__HIP_PLATFORM_AMD__)

template<class TestParams1, class TestParams2>
struct HipcubTestParamsMerge
{};

template<class... Params1, class... Params2>
struct HipcubTestParamsMerge<::testing::Types<Params1...>, ::testing::Types<Params2...>>
{
    using type = ::testing::Types<Params1..., Params2...>;
};

template<class TestParamsFirst, class... TestParams>
struct HipcubTestParamsMergeAll
{
    using type = typename HipcubTestParamsMerge<
        TestParamsFirst,
        typename HipcubTestParamsMergeAll<TestParams...>::type>::type;
};

template<class TestParamsFirst>
struct HipcubTestParamsMergeAll<TestParamsFirst>
{
    using type = TestParamsFirst;
};

template<class ExtentsType, bool UseGraphs = false>
struct DeviceForEachInExtentsParams
{
    using extents_type               = ExtentsType;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
struct HipcubDeviceForEachInExtentsTests : public ::testing::Test
{
    using extents_type                      = typename Params::extents_type;
    static constexpr bool use_graphs        = Params::use_graphs;
    static constexpr bool debug_synchronous = false;
};

template<class IndexType>
using HipcubDeviceForEachInExtentsParamGenerator
    = ::testing::Types<DeviceForEachInExtentsParams<::hipcub::extents<IndexType>>,
                       DeviceForEachInExtentsParams<::hipcub::extents<IndexType, 5>>,
                       DeviceForEachInExtentsParams<::hipcub::extents<IndexType, 5, 3>>,
                       DeviceForEachInExtentsParams<::hipcub::extents<IndexType, 5, 3, 4>>,
                       DeviceForEachInExtentsParams<::hipcub::extents<IndexType, 2, 5, 3, 4>>>;

using HipcubDeviceForEachInExtentsTestsParams = typename HipcubTestParamsMergeAll<
    HipcubDeviceForEachInExtentsParamGenerator<std::int16_t>,
    HipcubDeviceForEachInExtentsParamGenerator<std::uint16_t>,
    HipcubDeviceForEachInExtentsParamGenerator<std::int32_t>,
    HipcubDeviceForEachInExtentsParamGenerator<std::uint32_t>,
    HipcubDeviceForEachInExtentsParamGenerator<std::int64_t>,
    HipcubDeviceForEachInExtentsParamGenerator<std::uint64_t>>::type;

template<int Rank = 0,
         typename T,
         typename ExtentsType,
         typename std::enable_if<Rank == ExtentsType::rank()>::type* = nullptr,
         typename... IndicesType>
inline void fill_linear_impl(std::vector<T>& vector,
                             const ExtentsType&,
                             size_t& pos,
                             IndicesType... indices)
{
    vector[pos++] = {indices...};
}

template<
    int Rank = 0,
    typename T,
    typename ExtentsType,
    typename std::enable_if<Rank<ExtentsType::rank()>::type* = nullptr,
                            typename... IndicesType> inline void
        fill_linear_impl(
            std::vector<T>& vector, const ExtentsType& ext, size_t& pos, IndicesType... indices)
{
    using extents_index_type = typename ExtentsType::index_type;
    for(extents_index_type i = 0; i < static_cast<extents_index_type>(ext.static_extent(Rank)); ++i)
    {
        fill_linear_impl<Rank + 1>(vector, ext, pos, indices..., i);
    }
}

template<typename T, typename IndexType, size_t... Extents>
inline void fill_linear(std::vector<T>& vector, const ::hipcub::extents<IndexType, Extents...>& ext)
{
    size_t pos = 0;
    fill_linear_impl(vector, ext, pos);
}

template<typename IndexType, int Size>
struct LinearStore
{
    using op_data_t = IndexType[Size];
    void* d_data;

    template<typename... Args>
    __device__ __forceinline__
    void operator()(IndexType idx, Args... args)
    {
        static_assert(sizeof...(Args) == Size, "wrong number of arguments");
        auto& i = static_cast<op_data_t*>(d_data)[idx];
        // We use the "placement new" operator to copy the data from an initializer list.
        new(&i) op_data_t{args...};
    }
};

TYPED_TEST_SUITE(HipcubDeviceForEachInExtentsTests, HipcubDeviceForEachInExtentsTestsParams);

TEST(HipcubDeviceForEachInExtentsTests, ForEachInExtentsAPI)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using item_t                = int;
    using data_t                = std::array<item_t, 3>;
    using extents_type          = hipcub::extents<item_t, 3, 2, 2>;
    constexpr auto extents_size = hipcub::extents_size<extents_type>::value;
    constexpr auto memory_size  = extents_size * sizeof(data_t);

    constexpr extents_type ext{};

    std::vector<data_t> expected = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1},
        {2, 0, 0},
        {2, 0, 1},
        {2, 1, 0},
        {2, 1, 1}
    };

    item_t* d_input = nullptr;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, memory_size));
    HIP_CHECK(hipMemset(d_input, 0, memory_size));

    struct Op
    {
        using op_data_t = item_t[3];
        void* d_data;

        __device__ __host__ __forceinline__
        void  operator()(int idx, int x, int y, int z)
        {
            auto& i = static_cast<op_data_t*>(d_data)[idx];
            // We use the "placement new" operator to copy the data from an initializer list.
            new(&i) op_data_t{x, y, z};
        }
    };

    HIP_CHECK(hipcub::DeviceFor::ForEachInExtents(ext, Op{d_input}));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<data_t> h_output(extents_size, {0, 0, 0});
    HIP_CHECK(hipMemcpy(h_output.data(), d_input, memory_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_output, expected));
    HIP_CHECK(hipFree(d_input));
}

TYPED_TEST(HipcubDeviceForEachInExtentsTests, ForEachInExtentsStatic)
{
    using extents_type       = typename TestFixture::extents_type;
    using extents_index_type = typename extents_type::index_type;
    using index_type         = extents_index_type;

    using item_t                = index_type;
    using data_t                = std::array<item_t, extents_type::rank()>;
    constexpr auto extents_size = hipcub::extents_size<extents_type>::value;
    constexpr auto memory_size  = extents_size * sizeof(data_t);
    constexpr auto rank         = extents_type::rank();
    using store_op_t            = LinearStore<index_type, rank>;

    extents_type ext{};

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    std::vector<data_t> expected;
    expected.reserve(extents_size);
    fill_linear(expected, ext);

    item_t* d_input = nullptr;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, memory_size));
    HIP_CHECK(hipMemset(d_input, 0, memory_size));

    HIP_CHECK(hipcub::DeviceFor::ForEachInExtents(ext, store_op_t{d_input}));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<data_t> h_output;
    h_output.reserve(extents_size);
    HIP_CHECK(hipMemcpy(h_output.data(), d_input, memory_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_output, expected));
    HIP_CHECK(hipFree(d_input));
}

#endif // (defined(__HIP_PLATFORM_NVIDIA__) && defined(__cccl_lib_mdspan)) || defined(__HIP_PLATFORM_AMD__)

template<class Params>
class HipcubDeviceForBulkTests : public HipcubDeviceForTests<Params>
{};

using HipcubDeviceForBulkTestsParams = ::testing::Types<DeviceForParams<std::int32_t>,
                                                        DeviceForParams<std::uint32_t>,
                                                        DeviceForParams<std::int64_t>,
                                                        DeviceForParams<std::uint64_t>>;

TYPED_TEST_SUITE(HipcubDeviceForBulkTests, HipcubDeviceForBulkTestsParams);

template<class T>
struct offset_count_device_t
{
    int* d_count;

    template<class OffsetT>
    HIPCUB_DEVICE
    void operator()(OffsetT i)
    {
        static_assert(std::is_same<T, OffsetT>::value, "T and OffsetT must be the same type");
        atomicAdd(d_count + i, 1);
    }
};

TYPED_TEST(HipcubDeviceForBulkTests, Bulk)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;

    hipStream_t stream = 0; // default
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            T n = static_cast<T>(size);

            int* d_count;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(int) * n));
            HIP_CHECK(hipMemset(d_count, 0, sizeof(int) * n));
            offset_count_device_t<T> device_op{d_count};

            // Run
            HIP_CHECK(hipcub::DeviceFor::Bulk(n, device_op, stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            std::vector<int> output(n, int(0));
            HIP_CHECK(hipMemcpy(output.data(), d_count, sizeof(int) * n, hipMemcpyDeviceToHost));

            std::vector<int> expected(n, int(1));

            // Check if whole array is filled with ones
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(expected, output));

            HIP_CHECK(hipFree(d_count));
        }
    }
}

TEST(HipcubDeviceForBulkTests, BulkTempStore)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = int32_t;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            T n = static_cast<T>(size);

            // Device pointers
            int* d_count;

            // Allocate memory
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_count, sizeof(int) * n));

            // Prepare Bulk
            HIP_CHECK(hipMemset(d_count, 0, sizeof(int) * n));
            offset_count_device_t<T> device_op{d_count};

            // Calculate temporary storage
            size_t temp_storage_bytes = 0;
            HIP_CHECK(hipcub::DeviceFor::Bulk(nullptr, temp_storage_bytes, n, device_op, stream));
            void* d_temp_storage{};

            // Allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));

            // Run Bulk
            HIP_CHECK(
                hipcub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, n, device_op, stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            std::vector<int> output(n, int(0));
            HIP_CHECK(hipMemcpy(output.data(), d_count, sizeof(int) * n, hipMemcpyDeviceToHost));

            std::vector<int> expected(n, int(1));

            // Check if whole array is filled with ones
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(expected, output));

            HIP_CHECK(hipFree(d_count));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }
}
