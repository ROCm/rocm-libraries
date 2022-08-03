// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/device/device_select.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>

// required test headers
#include "test_utils_types.hpp"
#include <numeric>

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class FlagType = unsigned int,
    bool UseIdentityIterator = false
>
struct DeviceSelectParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using flag_type = FlagType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDeviceSelectTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using flag_type = typename Params::flag_type;
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
};

typedef ::testing::Types<
    DeviceSelectParams<int, long>,
    DeviceSelectParams<int8_t, int8_t>,
    DeviceSelectParams<uint8_t, uint8_t>,
    DeviceSelectParams<rocprim::half, rocprim::half>,
    DeviceSelectParams<rocprim::bfloat16, rocprim::bfloat16>,
    DeviceSelectParams<unsigned char, float, int, true>,
    DeviceSelectParams<double, double, int, true>,
    DeviceSelectParams<test_utils::custom_test_type<double>, test_utils::custom_test_type<double>, int, true>
> RocprimDeviceSelectTestsParams;

std::vector<size_t> get_sizes(int seed_value)
{
    std::vector<size_t> sizes = {
        2, 32, 64, 256,
        1024, 2048,
        3072, 4096,
        27845, (1 << 18) + 1111
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_SUITE(RocprimDeviceSelectTests, RocprimDeviceSelectTestsParams);

TYPED_TEST(RocprimDeviceSelectTests, Flagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using F = typename TestFixture::flag_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<F> flags = test_utils::get_random_data<F>(size, 0, 1, seed_value);

            T * d_input;
            F * d_flags;
            U * d_output;
            unsigned int * d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, flags.size() * sizeof(F)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    d_flags, flags.data(),
                    flags.size() * sizeof(F),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            std::vector<U> expected;
            expected.reserve(input.size());
            for(size_t i = 0; i < input.size(); i++)
            {
                if(flags[i] != 0)
                {
                    expected.push_back(input[i]);
                }
            }

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::select(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input,
                    d_flags,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::select(
                    d_temp_storage,
                    temp_storage_size_bytes,
                    d_input,
                    d_flags,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            unsigned int selected_count_output = 0;
            HIP_CHECK(
                hipMemcpy(
                    &selected_count_output, d_selected_count_output,
                    sizeof(unsigned int),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            std::vector<U> output(input.size());
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            hipFree(d_input);
            hipFree(d_flags);
            hipFree(d_output);
            hipFree(d_selected_count_output);
            hipFree(d_temp_storage);
        }
    }

}

template<class T>
struct select_op
{
    __device__ __host__ inline
    bool operator()(const T& value) const
    {
        if(value < T(50)) return true;
        return false;
    }
};

template<>
struct select_op<rocprim::half>
{
    __device__ __host__ inline
    bool operator()(const rocprim::half& value) const
    {
        #if __HIP_DEVICE_COMPILE__
        if(value < rocprim::half(50)) return true;
        #else
        if(test_utils::half_less()(value, rocprim::half(50))) return true;
        #endif
        return false;
    }
};

template<>
struct select_op<rocprim::bfloat16>
{
    __device__ __host__ inline
    bool operator()(const rocprim::bfloat16& value) const
    {
        #if __HIP_DEVICE_COMPILE__
        if(value < rocprim::bfloat16(50)) return true;
        #else
        if(test_utils::bfloat16_less()(value, rocprim::bfloat16(50))) return true;
        #endif
        return false;
    }
};

TYPED_TEST(RocprimDeviceSelectTests, SelectOp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100, seed_value);

            T * d_input;
            U * d_output;
            unsigned int * d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(
                    d_input, input.data(),
                    input.size() * sizeof(T),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected results on host
            std::vector<U> expected;
            expected.reserve(input.size());
            for(size_t i = 0; i < input.size(); i++)
            {
                if(select_op<T>()(input[i]))
                {
                    expected.push_back(input[i]);
                }
            }

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                rocprim::select(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    select_op<T>(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                rocprim::select(
                    d_temp_storage,
                    temp_storage_size_bytes,
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    d_selected_count_output,
                    input.size(),
                    select_op<T>(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            unsigned int selected_count_output = 0;
            HIP_CHECK(
                hipMemcpy(
                    &selected_count_output, d_selected_count_output,
                    sizeof(unsigned int),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            std::vector<U> output(input.size());
            HIP_CHECK(
                hipMemcpy(
                    output.data(), d_output,
                    output.size() * sizeof(U),
                    hipMemcpyDeviceToHost
                )
            );
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            hipFree(d_input);
            hipFree(d_output);
            hipFree(d_selected_count_output);
            hipFree(d_temp_storage);
        }
    }

}

std::vector<float> get_discontinuity_probabilities()
{
    std::vector<float> probabilities = {
        0.05, 0.25, 0.5, 0.75, 0.95, 1
    };
    return probabilities;
}

TYPED_TEST(RocprimDeviceSelectTests, UniqueEmptyInput)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using op_type = typename test_utils::select_equal_to_operator<T>::type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    // Allocate and copy to device
    unsigned int * d_selected_count_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));

    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(
        rocprim::unique(
            nullptr,
            temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>((T)123),
            rocprim::make_discard_iterator(),
            d_selected_count_output,
            0,
            op_type(),
            stream,
            debug_synchronous
        )
    );

    void * d_temp_storage = nullptr;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(
        rocprim::unique(
            d_temp_storage,
            temp_storage_size_bytes,
            rocprim::make_constant_iterator<T>((T)123),
            rocprim::make_discard_iterator(),
            d_selected_count_output,
            0,
            op_type(),
            stream,
            debug_synchronous
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Check if number of selected value is 0
    unsigned int selected_count_output = 0;
    HIP_CHECK(
        hipMemcpy(
            &selected_count_output, d_selected_count_output,
            sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    ASSERT_EQ(selected_count_output, 0);

    hipFree(d_selected_count_output);
    hipFree(d_temp_storage);
}

TYPED_TEST(RocprimDeviceSelectTests, Unique)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using op_type = typename test_utils::select_equal_to_operator<T>::type;
    using scan_op_type = rocprim::plus<T>;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const auto sizes = get_sizes(seed_value);
        const auto probabilities = get_discontinuity_probabilities();
        for(auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<T> input(size);
                {
                    std::vector<T> input01 = test_utils::get_random_data01<T>(size, p, seed_value);
                    std::partial_sum(
                        input01.begin(), input01.end(), input.begin(), scan_op_type()
                    );
                }

                // Allocate and copy to device
                T * d_input;
                U * d_output;
                unsigned int * d_selected_count_output;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
                HIP_CHECK(
                    hipMemcpy(
                        d_input, input.data(),
                        input.size() * sizeof(T),
                        hipMemcpyHostToDevice
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // Calculate expected results on host
                std::vector<U> expected;
                expected.reserve(input.size());
                expected.push_back(input[0]);
                for(size_t i = 1; i < input.size(); i++)
                {
                    if(!op_type()(input[i-1], input[i]))
                    {
                        expected.push_back(input[i]);
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(
                    rocprim::unique(
                        nullptr,
                        temp_storage_size_bytes,
                        d_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                        d_selected_count_output,
                        input.size(),
                        op_type(),
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                void * d_temp_storage = nullptr;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
                HIP_CHECK(hipDeviceSynchronize());

                // Run
                HIP_CHECK(
                    rocprim::unique(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                        d_selected_count_output,
                        input.size(),
                        op_type(),
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                unsigned int selected_count_output = 0;
                HIP_CHECK(
                    hipMemcpy(
                        &selected_count_output, d_selected_count_output,
                        sizeof(unsigned int),
                        hipMemcpyDeviceToHost
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());
                ASSERT_EQ(selected_count_output, expected.size());

                // Check if output values are as expected
                std::vector<U> output(input.size());
                HIP_CHECK(
                    hipMemcpy(
                        output.data(), d_output,
                        output.size() * sizeof(U),
                        hipMemcpyDeviceToHost
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

                hipFree(d_input);
                hipFree(d_output);
                hipFree(d_selected_count_output);
                hipFree(d_temp_storage);
            }
        }
    }
}

// The operator must be only called, when we have valid element in a block
template<class T, class F>
struct element_equal_operator
{
    F *data;
    element_equal_operator(F* _data)
    {
      this->data = _data;
    }

    __host__ __device__
    bool operator()(const T& index_a, const T& index_b)  const
    {
        F lhs = data[index_a];
        F rhs = data[index_b];
        if (lhs != rhs) {
            return false;
        }
        return true;
    }
};

TEST(RocprimDeviceSelectTests, UniqueGuardedOperator)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = int64_t;
    using F = int64_t;
    using U = int64_t;
    using scan_op_type = rocprim::plus<T>;
    static constexpr bool use_identity_iterator = false;
    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default stream

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const auto sizes = get_sizes(seed_value);
        const auto probabilities = get_discontinuity_probabilities();
        for(auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<T> input = test_utils::get_random_data<T>(size, 0, size - 1, seed_value);

                std::vector<F> input_flag(size);
                {
                    std::vector<T> input01 = test_utils::get_random_data01<T>(size, p, seed_value + 1);
                    std::partial_sum(
                        input01.begin(), input01.end(), input_flag.begin(), scan_op_type()
                    );
                }


                // Allocate and copy to device
                T * d_input;
                F * d_flag;
                U * d_output;
                unsigned int * d_selected_count_output;
                HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(T)));
                HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_flag), input_flag.size() * sizeof(F)));
                HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), input.size() * sizeof(U)));
                HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_selected_count_output), sizeof(unsigned int)));
                HIP_CHECK(
                    hipMemcpy(
                        d_input, input.data(),
                        input.size() * sizeof(T),
                        hipMemcpyHostToDevice
                    )
                );
                HIP_CHECK(
                    hipMemcpy(
                        d_flag, input_flag.data(),
                        input_flag.size() * sizeof(F),
                        hipMemcpyHostToDevice
                    )
                );
                element_equal_operator<F, T> device_equal_op(d_flag);
                element_equal_operator<F, T> host_equal_op(input_flag.data());
                HIP_CHECK(hipDeviceSynchronize());

                // Calculate expected results on host
                std::vector<U> expected;
                expected.reserve(input.size());
                expected.push_back(input[0]);
                for(size_t i = 1; i < input.size(); i++)
                {
                    if(!host_equal_op(input[i-1], input[i]))
                    {
                        expected.push_back(input[i]);
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(
                    rocprim::unique(
                        nullptr,
                        temp_storage_size_bytes,
                        d_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                        d_selected_count_output,
                        input.size(),
                        device_equal_op,
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                void * d_temp_storage = nullptr;
                HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
                HIP_CHECK(hipDeviceSynchronize());

                // Run
                HIP_CHECK(
                    rocprim::unique(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                        d_selected_count_output,
                        input.size(),
                        device_equal_op,
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                unsigned int selected_count_output = 0;
                HIP_CHECK(
                    hipMemcpy(
                        &selected_count_output, d_selected_count_output,
                        sizeof(unsigned int),
                        hipMemcpyDeviceToHost
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());
                ASSERT_EQ(selected_count_output, expected.size());

                // Check if output values are as expected
                std::vector<U> output(input.size());
                HIP_CHECK(
                    hipMemcpy(
                        output.data(), d_output,
                        output.size() * sizeof(U),
                        hipMemcpyDeviceToHost
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

                hipFree(d_input);
                hipFree(d_flag);
                hipFree(d_output);
                hipFree(d_selected_count_output);
                hipFree(d_temp_storage);
            }
        }
    }
}

// Params for tests
template<
    typename KeyType,
    typename ValueType,
    typename OutputKeyType = KeyType, 
    typename OutputValueType = ValueType, 
    bool UseIdentityIterator = false
>
struct DeviceUniqueByKeyParams
{
    using key_type = KeyType;
    using value_type = ValueType;
    using output_key_type = OutputKeyType;
    using output_value_type = OutputValueType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

template<class Params>
class RocprimDeviceUniqueByKeyTests : public ::testing::Test
{
public:
    using key_type               = typename Params::key_type;
    using value_type             = typename Params::value_type;
    using output_key_type        = typename Params::output_key_type;
    using output_value_type      = typename Params::output_value_type;
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
};

typedef ::testing::Types<
    DeviceUniqueByKeyParams<int, int>,
    DeviceUniqueByKeyParams<double, double>,
    DeviceUniqueByKeyParams<rocprim::half, uint8_t>,
    DeviceUniqueByKeyParams<rocprim::bfloat16, uint8_t>,
    DeviceUniqueByKeyParams<uint8_t, long long>,
    DeviceUniqueByKeyParams<int, float, long, double>,
    DeviceUniqueByKeyParams<long long, uint8_t, long, int, true>,
    DeviceUniqueByKeyParams<test_utils::custom_test_type<double>, test_utils::custom_test_type<double>>
> RocprimDeviceUniqueByKeyTestParams;

TYPED_TEST_SUITE(RocprimDeviceUniqueByKeyTests, RocprimDeviceUniqueByKeyTestParams);

TYPED_TEST(RocprimDeviceUniqueByKeyTests, UniqueByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using output_key_type = typename TestFixture::output_key_type;
    using output_value_type = typename TestFixture::output_value_type;

    using op_type = typename test_utils::select_equal_to_operator<key_type>::type;

    using scan_op_type = rocprim::plus<key_type>;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const auto sizes = get_sizes(seed_value);
        const auto probabilities = get_discontinuity_probabilities();
        for(auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<key_type> input_keys(size);
                {
                    std::vector<key_type> input01 = test_utils::get_random_data01<key_type>(size, p, seed_value);
                    std::partial_sum(
                        input01.begin(), input01.end(), input_keys.begin(), scan_op_type()
                    );
                }
                const auto input_values
                    = test_utils::get_random_data<value_type>(size, -1000, 1000, seed_value);

                // Allocate and copy to device
                key_type*        d_keys_input;
                value_type*      d_values_input;
                output_key_type* d_keys_output;
                output_value_type* d_values_output;
                unsigned int * d_selected_count_output;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, input_keys.size() * sizeof(input_keys[0])));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, input_values.size() * sizeof(input_values[0])));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, input_keys.size() * sizeof(output_key_type)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output, input_values.size() * sizeof(output_value_type)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
                HIP_CHECK(
                    hipMemcpy(
                        d_keys_input, input_keys.data(),
                        input_keys.size() * sizeof(input_keys[0]),
                        hipMemcpyHostToDevice
                    )
                );
                HIP_CHECK(
                    hipMemcpy(
                        d_values_input, input_values.data(),
                        input_values.size() * sizeof(input_values[0]),
                        hipMemcpyHostToDevice
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // Calculate expected results on host
                std::vector<output_key_type> expected_keys;
                std::vector<output_value_type> expected_values;
                expected_keys.reserve(input_keys.size());
                expected_values.reserve(input_values.size());
                expected_keys.push_back(input_keys[0]);
                expected_values.push_back(input_values[0]);
                for(size_t i = 1; i < input_keys.size(); i++)
                {
                    if(!op_type()(input_keys[i-1], input_keys[i]))
                    {
                        expected_keys.push_back(input_keys[i]);
                        expected_values.push_back(input_values[i]);
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(
                    rocprim::unique_by_key(
                        nullptr,
                        temp_storage_size_bytes,
                        d_keys_input,
                        d_values_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_keys_output),
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                        d_selected_count_output,
                        input_keys.size(),
                        op_type(),
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                void * d_temp_storage = nullptr;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
                HIP_CHECK(hipDeviceSynchronize());

                // Run
                HIP_CHECK(
                    rocprim::unique_by_key(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_keys_input,
                        d_values_input,
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_keys_output),
                        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_values_output),
                        d_selected_count_output,
                        input_keys.size(),
                        op_type(),
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                unsigned int selected_count_output = 0;
                HIP_CHECK(
                    hipMemcpy(
                        &selected_count_output, d_selected_count_output,
                        sizeof(unsigned int),
                        hipMemcpyDeviceToHost
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());
                ASSERT_EQ(selected_count_output, expected_keys.size());

                // Check if outputs are as expected
                std::vector<output_key_type> output_keys(input_keys.size());
                HIP_CHECK(
                    hipMemcpy(
                        output_keys.data(), d_keys_output,
                        output_keys.size() * sizeof(output_keys[0]),
                        hipMemcpyDeviceToHost
                    )
                );
                std::vector<output_value_type> output_values(input_values.size());
                HIP_CHECK(
                    hipMemcpy(
                        output_values.data(), d_values_output,
                        output_values.size() * sizeof(output_values[0]),
                        hipMemcpyDeviceToHost
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_keys, expected_keys, expected_keys.size()));
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_values, expected_values, expected_values.size()));

                hipFree(d_keys_input);
                hipFree(d_values_input);
                hipFree(d_keys_output);
                hipFree(d_values_output);
                hipFree(d_selected_count_output);
                hipFree(d_temp_storage);
            }
        }
    }
}

class RocprimDeviceSelectLargeInputTests : public ::testing::TestWithParam<unsigned int> {
    public:
        const bool debug_synchronous = false;
};

INSTANTIATE_TEST_SUITE_P(RocprimDeviceSelectLargeInputFlaggedTest, RocprimDeviceSelectLargeInputTests, ::testing::Values(
    2048, 9643, 32768, 38713
));

std::vector<size_t> get_large_sizes()
{
    std::vector<size_t> sizes = {
        (size_t{1} << 30) - 1, size_t{1} << 30,
        (size_t{1} << 31) - 1, size_t{1} << 31,
        (size_t{1} << 32) - 1, size_t{1} << 32,
        (size_t{1} << 35) - 1
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(
        2, (size_t {1} << 30) + 1, (size_t {1} << 35) - 2, 0);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TEST_P(RocprimDeviceSelectLargeInputTests, LargeInputFlagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    unsigned int flag_selector = GetParam();

    using InputIterator = typename rocprim::counting_iterator<size_t>;

    const bool debug_synchronous = RocprimDeviceSelectLargeInputTests::debug_synchronous;

    hipStream_t stream = 0; // default stream

    const std::vector<size_t> sizes = get_large_sizes();

    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        size_t initial_value = 0;
        InputIterator input_begin(initial_value);

        auto flags_it = rocprim::make_transform_iterator(rocprim::make_counting_iterator(size_t(0)), [flag_selector](size_t i){
            if (i % flag_selector == 0) return 1;
            else return 0;
        });

        size_t selected_count_output = 0;
        size_t *d_selected_count_output;

        size_t expected_output_size = rocprim::detail::ceiling_div(size, flag_selector);
            
        size_t *d_output;
        std::vector<size_t> output(expected_output_size);

        // Calculate expected results on host
        std::vector<size_t> expected_output(expected_output_size);
        for (size_t i = 0; i < expected_output_size; i++) 
        {
            expected_output[i] = input_begin[i*flag_selector];
        }

        HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(d_output[0]) * expected_output_size));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(d_selected_count_output[0])));
        HIP_CHECK(hipDeviceSynchronize());

        // temp storage
        size_t temp_storage_size_bytes;
        void *d_temp_storage = nullptr;

        // Get size of d_temp_storage
        HIP_CHECK(
            rocprim::select(
                d_temp_storage,
                temp_storage_size_bytes,
                input_begin,
                flags_it,
                d_output,
                d_selected_count_output,
                size,
                stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        HIP_CHECK(
            rocprim::select(
                d_temp_storage,
                temp_storage_size_bytes,
                input_begin,
                flags_it,
                d_output,
                d_selected_count_output,
                size,
                stream,
                debug_synchronous
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if number of selected value is as expected
        HIP_CHECK(
            hipMemcpy(
                &selected_count_output, d_selected_count_output,
                sizeof(size_t),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());
        ASSERT_EQ(selected_count_output, expected_output_size);

        // Check if output values are as expected
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                sizeof(output[0]) * expected_output_size,
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected_output, expected_output.size()));

        hipFree(d_output);
        hipFree(d_selected_count_output);
        hipFree(d_temp_storage);
    }
}
