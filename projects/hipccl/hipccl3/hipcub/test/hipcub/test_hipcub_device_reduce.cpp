// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipcub/device/device_reduce.hpp"

// Params for tests
template<
    class InputType,
    class OutputType = InputType
>
struct DeviceReduceParams
{
    using input_type = InputType;
    using output_type = OutputType;
};

// ---------------------------------------------------------
// Test for reduction ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceReduceTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceReduceParams<int, long>,
    DeviceReduceParams<unsigned long>,
    DeviceReduceParams<short>,
    DeviceReduceParams<short, float>,
    DeviceReduceParams<int, double>
    #ifdef HIPCUB_ROCPRIM_API
    ,
    DeviceReduceParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<float>>,
    DeviceReduceParams<test_utils::custom_test_type<int>, test_utils::custom_test_type<float>>
    #endif

> HipcubDeviceReduceTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 17) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384, rand());
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(HipcubDeviceReduceTests, HipcubDeviceReduceTestsParams);

TYPED_TEST(HipcubDeviceReduceTests, Reduce)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(
                size,
                1,
                100,
                seed_value
            );
            std::vector<U> output(1, 0);

            T * d_input;
            U * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            U expected = U(0);
            for(unsigned int i = 0; i < input.size(); i++)
            {
                expected = expected + input[i];
            }

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                hipcub::DeviceReduce::Sum(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                hipcub::DeviceReduce::Sum(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, 0.01f));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceMinimum)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<U> output(1, 0);

            T * d_input;
            U * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            hipcub::Min min_op;
            // Calculate expected results on host
            U expected = U(std::numeric_limits<U>::max());
            for(unsigned int i = 0; i < input.size(); i++)
            {
                expected = min_op(expected, U(input[i]));
            }

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                hipcub::DeviceReduce::Min(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                hipcub::DeviceReduce::Min(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, 0.01f));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArgMinimum)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename hipcub::ArgIndexInputIterator<T*, int>;
    using key_value = typename Iterator::value_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 200, seed_value);
            std::vector<key_value> output(1);

            T * d_input;
            key_value * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(key_value)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            Iterator x(input.data());
            const key_value max(1, std::numeric_limits<T>::max());
            key_value expected = std::accumulate(x, x + size, max, hipcub::ArgMin());

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                hipcub::DeviceReduce::ArgMin(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                hipcub::DeviceReduce::ArgMin(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(key_value),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0].key, expected.key, 0.01f));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0].value, expected.value, 0.01f));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
}

TYPED_TEST(HipcubDeviceReduceTests, ReduceArgMaximum)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename hipcub::ArgIndexInputIterator<T*, int>;
    using key_value = typename Iterator::value_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100, seed_value);
            std::vector<key_value> output(1);

            T * d_input;
            key_value * d_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(key_value)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            Iterator x(input.data());
            const key_value min(1, std::numeric_limits<T>::lowest());
            key_value expected = std::accumulate(x, x + size, min, hipcub::ArgMax());

            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                hipcub::DeviceReduce::ArgMax(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                hipcub::DeviceReduce::ArgMax(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(key_value),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_EQ(output[0].key, expected.key);
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0].value, expected.value, 0.01f));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }
}
