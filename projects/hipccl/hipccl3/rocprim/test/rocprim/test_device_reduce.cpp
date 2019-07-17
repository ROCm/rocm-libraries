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

#include <iostream>
#include <vector>
#include <algorithm>

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

namespace rp = rocprim;

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    bool UseIdentityIterator = false
>
struct DeviceReduceParams
{
    using input_type = InputType;
    using output_type = OutputType;
    // Tests output iterator with void value_type (OutputIterator concept)
    static constexpr bool use_identity_iterator =  UseIdentityIterator;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceReduceTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
};

typedef ::testing::Types<
    DeviceReduceParams<unsigned int>,
    DeviceReduceParams<long, long, true>,
    DeviceReduceParams<short, int>,
    DeviceReduceParams<int, float>,
    DeviceReduceParams<int8_t, int8_t>,
    DeviceReduceParams<uint8_t, uint8_t>,
    DeviceReduceParams<rp::half, rp::half>,
    DeviceReduceParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<float>>,
    DeviceReduceParams<test_utils::custom_test_type<int>, test_utils::custom_test_type<float>>
> RocprimDeviceReduceTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 17) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceReduceTests, RocprimDeviceReduceTestsParams);

TYPED_TEST(RocprimDeviceReduceTests, ReduceEmptyInput)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    U * d_output;
    HIP_CHECK(hipMalloc(&d_output, sizeof(U)));

    const U initial_value = U(1234);

    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(
        rocprim::reduce(
            nullptr, temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(T(345)),
            d_output,
            initial_value,
            0, rocprim::minimum<U>(), stream, debug_synchronous
        )
    );

    void * d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(
        rocprim::reduce(
            d_temp_storage, temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(T(345)),
            d_output,
            initial_value,
            0, rocprim::minimum<U>(), stream, debug_synchronous
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    U output;
    HIP_CHECK(
        hipMemcpy(
            &output, d_output,
            sizeof(U),
            hipMemcpyDeviceToHost
        )
    );
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, initial_value));

    hipFree(d_output);
    hipFree(d_temp_storage);
}

TYPED_TEST(RocprimDeviceReduceTests, Reduce)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using binary_op_type = typename std::conditional<std::is_same<U, rp::half>::value, test_utils::half_plus, rp::plus<U>>::type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // precision of half differs between host and device with large plus reductions
        if(std::is_same<U, rp::half>::value && size >= 1024)
        {
            break;
        }

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);
        std::vector<U> output(1, 0);

        // reduce function
        binary_op_type plus_op;

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
            expected = plus_op(expected, input[i]);
        }

        // temp storage
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::reduce(
                d_temp_storage, temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                input.size(), rp::plus<U>(), stream, debug_synchronous
            )
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::reduce(
                d_temp_storage, temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                input.size(), rp::plus<U>(), stream, debug_synchronous
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

TYPED_TEST(RocprimDeviceReduceTests, ReduceMinimum)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using binary_op_type = typename std::conditional<std::is_same<U, rp::half>::value, test_utils::half_minimum, rp::minimum<U>>::type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);
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

        // reduce function
        binary_op_type min_op;

        // Calculate expected results on host
        U expected = U(test_utils::numeric_limits<U>::max());
        for(unsigned int i = 0; i < input.size(); i++)
        {
            expected = min_op(expected, input[i]);
        }

        // temp storage
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::reduce(
                d_temp_storage, temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                test_utils::numeric_limits<U>::max(), input.size(), rp::minimum<U>(), stream, debug_synchronous
            )
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::reduce(
                d_temp_storage, temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                test_utils::numeric_limits<U>::max(), input.size(), rp::minimum<U>(), stream, debug_synchronous
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

template<
    class Key,
    class Value
>
struct arg_min
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::key_value_pair<Key, Value>
    operator()(const rocprim::key_value_pair<Key, Value>& a,
               const rocprim::key_value_pair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

template<>
struct arg_min<int, rp::half>
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::key_value_pair<int, rp::half>
    operator()(const rocprim::key_value_pair<int, rp::half>& a,
               const rocprim::key_value_pair<int, rp::half>& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
        #else
        return (test_utils::half_less()(b.value, a.value) || (test_utils::half_equal_to()(a.value, b.value) && (b.key < a.key))) ? b : a;
        #endif
    }
};


TYPED_TEST(RocprimDeviceReduceTests, ReduceArgMinimum)
{
    using T = typename TestFixture::input_type;
    using key_value = rocprim::key_value_pair<int, T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_value> input(size);
        for (size_t i = 0; i < size; i++)
        {
            input[i].key = i;
            input[i].value = test_utils::get_random_value<T>(1, 100);
        }
        std::vector<key_value> output(1);

        key_value * d_input;
        key_value * d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(key_value)));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(key_value)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(key_value),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        arg_min<int, T> reduce_op;
        const key_value max(std::numeric_limits<int>::max(), test_utils::numeric_limits<T>::max());

        // Calculate expected results on host
        key_value expected = max;
        for(unsigned int i = 0; i < input.size(); i++)
        {
            expected = reduce_op(expected, input[i]);
        }

        // temp storage
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::reduce(
                d_temp_storage, temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                max, input.size(), reduce_op, stream, debug_synchronous
            )
        );

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::reduce(
                d_temp_storage, temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                max, input.size(), reduce_op, stream, debug_synchronous
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
