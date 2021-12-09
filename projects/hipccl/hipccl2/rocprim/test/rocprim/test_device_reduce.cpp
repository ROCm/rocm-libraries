// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

// required rocprim headers
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>

// required test headers
#include "test_utils_types.hpp"

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    bool UseIdentityIterator = false,
    size_t SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct DeviceReduceParams
{
    using input_type = InputType;
    using output_type = OutputType;
    // Tests output iterator with void value_type (OutputIterator concept)
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr size_t size_limit = SizeLimit;
};

template <unsigned int SizeLimit>
struct size_limit_config {
    using type = rocprim::reduce_config<256, 16, rocprim::block_reduce_algorithm::default_algorithm, SizeLimit>;
};

template <>
struct size_limit_config<ROCPRIM_GRID_SIZE_LIMIT> {
    using type = rocprim::default_config;
};

template <unsigned int SizeLimit>
using size_limit_config_t = typename size_limit_config<SizeLimit>::type;

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
    static constexpr size_t size_limit = Params::size_limit;
};

template<class Params>
class RocprimDeviceReducePrecisionTests : public RocprimDeviceReduceTests<Params>{};

typedef ::testing::Types<
    DeviceReduceParams<unsigned int>,
    DeviceReduceParams<long, long, true>,
    DeviceReduceParams<short, int>,
    DeviceReduceParams<int, float>,
    DeviceReduceParams<int, int, false, 512>,
    DeviceReduceParams<float, float, false, 2048>,
    DeviceReduceParams<int, int, false, 4096>,
    DeviceReduceParams<int, int, false, 2097152>,
    DeviceReduceParams<int, int, false, 1073741824>,
    DeviceReduceParams<int8_t, int8_t>,
    DeviceReduceParams<uint8_t, uint8_t>,
    DeviceReduceParams<rocprim::half, rocprim::half>,
    DeviceReduceParams<rocprim::bfloat16, rocprim::bfloat16>,
    DeviceReduceParams<test_utils::custom_test_type<float>, test_utils::custom_test_type<float>>,
    DeviceReduceParams<test_utils::custom_test_type<int>, test_utils::custom_test_type<float>>
> RocprimDeviceReduceTestsParams;

typedef ::testing::Types<
    DeviceReduceParams<float, float, false, 2048>,
    DeviceReduceParams<rocprim::half, rocprim::half>,
    DeviceReduceParams<rocprim::bfloat16, rocprim::bfloat16>
> RocprimDeviceReducePrecisionTestsParams;

std::vector<size_t> get_sizes(int seed_value)
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211, 512,
        1024, 2048, 5096,
        34567, (1 << 17) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_SUITE(RocprimDeviceReduceTests, RocprimDeviceReduceTestsParams);
TYPED_TEST_SUITE(RocprimDeviceReducePrecisionTests, RocprimDeviceReducePrecisionTestsParams);

TYPED_TEST(RocprimDeviceReduceTests, ReduceEmptyInput)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    // TODO: ReduceEmptyInput cause random faulire with bfloat16
    if( std::is_same<T, rocprim::bfloat16>::value || std::is_same<U, rocprim::bfloat16>::value )
        GTEST_SKIP();

    hipStream_t stream = 0; // default stream

    U * d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(U)));

    const U initial_value = U(1234);

    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(
        rocprim::reduce<Config>(
            nullptr, temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>(T(345)),
            d_output,
            initial_value,
            0, rocprim::minimum<U>(), stream, debug_synchronous
        )
    );

    void * d_temp_storage = nullptr;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(
        rocprim::reduce<Config>(
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

TYPED_TEST(RocprimDeviceReduceTests, ReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100, seed_value);
            std::vector<U> output(1, (U)0);

            T * d_input;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            U expected = test_utils::host_reduce(input.begin(), input.end(), rocprim::plus<U>());
            // temp storage
            size_t temp_storage_size_bytes;
            void * d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::reduce<Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input.size(), rocprim::plus<U>(), stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::reduce<Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input.size(), rocprim::plus<U>(), stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, test_utils::precision_threshold<T>::percentage));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }

}

TYPED_TEST(RocprimDeviceReduceTests, ReduceMinimum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using binary_op_type = typename test_utils::select_minimum_operator<U>::type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<U> output(1, (U)0);

            T * d_input;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
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
                rocprim::reduce<Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    test_utils::numeric_limits<U>::max(), input.size(), rocprim::minimum<U>(), stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::reduce<Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    test_utils::numeric_limits<U>::max(), input.size(), rocprim::minimum<U>(), stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, test_utils::precision_threshold<T>::percentage));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
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
struct arg_min<int, rocprim::half>
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::key_value_pair<int, rocprim::half>
    operator()(const rocprim::key_value_pair<int, rocprim::half>& a,
               const rocprim::key_value_pair<int, rocprim::half>& b) const
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
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using key_value = rocprim::key_value_pair<int, T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<key_value> input(size);
            for (size_t i = 0; i < size; i++)
            {
                input[i].key = (int)i;
                input[i].value = test_utils::get_random_data<T>(1, 1, 100, seed_value)[0];
            }
            std::vector<key_value> output(1);

            key_value * d_input;
            key_value * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(key_value)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(key_value)));
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
                rocprim::reduce<Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    max, input.size(), reduce_op, stream, debug_synchronous
                )
            );

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::reduce<Config>(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    max, input.size(), reduce_op, stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
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
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0].value, expected.value, test_utils::precision_threshold<T>::percentage));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_temp_storage);
        }
    }

}

std::vector<size_t> get_large_sizes(int seed_value)
{
    std::vector<size_t> sizes = {
        (size_t{1} << 30) - 1, size_t{1} << 30,
        (size_t{1} << 31) - 1, size_t{1} << 31,
        (size_t{1} << 32) - 1, size_t{1} << 32,
        (size_t{1} << 35) - 1, size_t{1} << 35,
        (size_t{1} << 37) - 1,
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(
        2, (size_t {1} << 30) + 1, (size_t {1} << 37) - 2, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TEST(RocprimDeviceReduceTests, LargeIndices)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                      = size_t;
    using Iterator               = rocprim::counting_iterator<T>;
    const bool debug_synchronous = false;

    const hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_large_sizes(seed_value);

        for(const auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const Iterator input {0};

            T* d_output = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(T)));

            // temp storage
            size_t temp_storage_size_bytes = 0;
            void*  d_temp_storage          = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::reduce(nullptr,
                                      temp_storage_size_bytes,
                                      input,
                                      d_output,
                                      size,
                                      rocprim::plus<T> {},
                                      stream,
                                      debug_synchronous));

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(rocprim::reduce(d_temp_storage,
                                      temp_storage_size_bytes,
                                      input,
                                      d_output,
                                      size,
                                      rocprim::plus<T> {},
                                      stream,
                                      debug_synchronous));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            T output = 0;
            HIP_CHECK(hipMemcpy(&output, d_output, sizeof(T), hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            // Sum of numbers 0 to n - 1 is n(n - 1) / 2, not that this is correct even in case of overflow
            // The division is not integer division but either n or n - 1 has to be even.
            T expected_output = (size % 2 == 0) ? size / 2 * (size - 1) : size * ((size - 1) / 2);

            ASSERT_EQ(output, expected_output);

            hipFree(d_temp_storage);
            hipFree(d_output);
        }
    }
}

TYPED_TEST(RocprimDeviceReducePrecisionTests, ReduceSumInputEqualExponentFunction)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config                                = size_limit_config_t<TestFixture::size_limit>;

    const std::vector<size_t> sizes = get_sizes(42);
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // numeric_limits<T>::denorm_min() does not work...
        T lowest = static_cast<T>(
            -1.0
            * static_cast<double>(
                test_utils::numeric_limits<
                    T>::min())); // smallest (closest to zero) normal (negative) non-zero number

        // Generate data
        std::vector<T> input(size, lowest);
        std::vector<U> output(1, (U)0);

        T* d_input;
        U* d_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
        HIP_CHECK(
            hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host mathematically (instead of using reduce on host)
        U expected  = static_cast<U>(static_cast<double>(size) * static_cast<double>(lowest));

        // temp storage
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::reduce<Config>(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
            input.size(),
            rocprim::plus<U>(),
            stream,
            debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(rocprim::reduce<Config>(
            d_temp_storage,
            temp_storage_size_bytes,
            d_input,
            test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
            input.size(),
            rocprim::plus<U>(),
            stream,
            debug_synchronous));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        HIP_CHECK(
            hipMemcpy(output.data(), d_output, output.size() * sizeof(U), hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // Check if output values are as expected
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
            output[0], expected, test_utils::precision_threshold<T>::percentage));

        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp_storage);
    }
}
