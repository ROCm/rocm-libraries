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
#include "hipcub/device/device_select.hpp"
#include "hipcub/iterator/counting_input_iterator.hpp"
#include "hipcub/iterator/discard_output_iterator.hpp"

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class FlagType = unsigned int
>
struct DeviceSelectParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using flag_type = FlagType;
};

template<class Params>
class HipcubDeviceSelectTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using flag_type = typename Params::flag_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceSelectParams<int, long>,
    DeviceSelectParams<unsigned char, float>
> HipcubDeviceSelectTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        2, 32, 64, 256,
        1024, 2048,
        3072, 4096,
        27845, (1 << 18) + 1111
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, static_cast<size_t>(1), static_cast<size_t>(16384), rand());
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_SUITE(HipcubDeviceSelectTests, HipcubDeviceSelectTestsParams);

TYPED_TEST(HipcubDeviceSelectTests, Flagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using F = typename TestFixture::flag_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, static_cast<T>(1), static_cast<T>(100), seed_value);
            std::vector<F> flags = test_utils::get_random_data<F>(
                size,
                static_cast<F>(0),
                static_cast<F>(1),
                seed_value + seed_value_addition
            );

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
                hipcub::DeviceSelect::Flagged(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input,
                    d_flags,
                    d_output,
                    d_selected_count_output,
                    input.size(),
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                hipcub::DeviceSelect::Flagged(
                    d_temp_storage,
                    temp_storage_size_bytes,
                    d_input,
                    d_flags,
                    d_output,
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
            for(size_t i = 0; i < expected.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
            }

            hipFree(d_input);
            hipFree(d_flags);
            hipFree(d_output);
            hipFree(d_selected_count_output);
            hipFree(d_temp_storage);
        }
    }
}

TEST(HipcubDeviceSelectTests, FlagNormalization)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                      = int;
    using U                      = long;
    using F                      = unsigned int;
    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default stream

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipcub::CountingInputIterator<T> d_input(0);
        hipcub::CountingInputIterator<F> d_flags(1);
        U*                               d_output;
        unsigned int*                    d_selected_count_output;

        HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(*d_output)));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output,
                                                     sizeof(*d_selected_count_output)));

        // Calculate expected results on host
        std::vector<U> expected;
        expected.reserve(size);
        for(size_t i = 0; i < size; i++)
        {
            if(d_flags[i] != 0)
            {
                expected.push_back(static_cast<U>(i));
            }
        }

        // temp storage
        size_t temp_storage_size_bytes = 0;
        // Get size of d_temp_storage
        HIP_CHECK(hipcub::DeviceSelect::Flagged(nullptr,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_flags,
                                                d_output,
                                                d_selected_count_output,
                                                size,
                                                stream,
                                                debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0U);

        // allocate temporary storage
        void* d_temp_storage = nullptr;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

        // Run
        HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_input,
                                                d_flags,
                                                d_output,
                                                d_selected_count_output,
                                                size,
                                                stream,
                                                debug_synchronous));

        // Check if number of selected value is as expected
        unsigned int selected_count_output = 0;
        HIP_CHECK(hipMemcpy(&selected_count_output,
                            d_selected_count_output,
                            sizeof(selected_count_output),
                            hipMemcpyDeviceToHost));

        ASSERT_EQ(selected_count_output, size);

        // Check if output values are as expected
        std::vector<U> output(size);
        HIP_CHECK(
            hipMemcpy(output.data(), d_output, size * sizeof(*d_output), hipMemcpyDeviceToHost));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }

        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}

struct TestSelectOp
{
    template<class T>
    __host__ __device__ inline
    bool operator()(const T& value) const
    {
        if(value > T(50)) return true;
        return false;
    }
};

TYPED_TEST(HipcubDeviceSelectTests, SelectOp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    TestSelectOp select_op;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, static_cast<T>(0), static_cast<T>(100), seed_value);

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
                if(select_op(input[i]))
                {
                    expected.push_back(input[i]);
                }
            }

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(
                hipcub::DeviceSelect::If(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input,
                    d_output,
                    d_selected_count_output,
                    input.size(),
                    select_op,
                    stream,
                    debug_synchronous
                )
            );
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            void * d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            // Run
            HIP_CHECK(
                hipcub::DeviceSelect::If(
                    d_temp_storage,
                    temp_storage_size_bytes,
                    d_input,
                    d_output,
                    d_selected_count_output,
                    input.size(),
                    select_op,
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
            for(size_t i = 0; i < expected.size(); i++)
            {
                ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
            }

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
        0.5, 0.25, 0.5, 0.75, 0.95
    };
    return probabilities;
}

TYPED_TEST(HipcubDeviceSelectTests, Unique)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    const auto sizes = get_sizes();
    const auto probabilities = get_discontinuity_probabilities();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(auto p : probabilities)
        {
            for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
            {
                unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
                SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<T> input(size);
                {
                    std::vector<T> input01 = test_utils::get_random_data01<T>(size, p, seed_value);
                    test_utils::host_inclusive_scan(
                        input01.begin(), input01.end(), input.begin(), hipcub::Sum()
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
                    if(!(input[i-1] == input[i]))
                    {
                        expected.push_back(input[i]);
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(
                    hipcub::DeviceSelect::Unique(
                        nullptr,
                        temp_storage_size_bytes,
                        d_input,
                        d_output,
                        d_selected_count_output,
                        input.size(),
                        stream,
                        debug_synchronous
                    )
                );
                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0U);

                // allocate temporary storage
                void * d_temp_storage = nullptr;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
                HIP_CHECK(hipDeviceSynchronize());

                // Run
                HIP_CHECK(
                    hipcub::DeviceSelect::Unique(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_input,
                        d_output,
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
                for(size_t i = 0; i < expected.size(); i++)
                {
                    ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
                }

                hipFree(d_input);
                hipFree(d_output);
                hipFree(d_selected_count_output);
                hipFree(d_temp_storage);
            }
        }
    }
}

TEST(HipcubDeviceSelectTests, UniqueDiscardOutputIterator)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const bool debug_synchronous = false;

    hipStream_t stream = 0; //default stream

    const auto sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        hipcub::CountingInputIterator<unsigned int> d_input(0);
        hipcub::DiscardOutputIterator<>             d_output;
        size_t*                                     d_selected_count_output;

        HIP_CHECK(test_common_utils::hipMallocHelper((&d_selected_count_output), sizeof(size_t)));

        // temp storage
        size_t temp_storage_size_bytes = 0;
        // Get size of d_temp_storage
        HIP_CHECK(hipcub::DeviceSelect::Unique(nullptr,
                                               temp_storage_size_bytes,
                                               d_input,
                                               d_output,
                                               d_selected_count_output,
                                               size,
                                               stream,
                                               debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0U);

        // allocate temporary storage
        void* d_temp_storage = nullptr;
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

        // Run
        HIP_CHECK(hipcub::DeviceSelect::Unique(d_temp_storage,
                                               temp_storage_size_bytes,
                                               d_input,
                                               d_output,
                                               d_selected_count_output,
                                               size,
                                               stream,
                                               debug_synchronous));

        // Check if number of selected value is as expected
        size_t selected_count_output = 0;
        HIP_CHECK(hipMemcpy(&selected_count_output,
                            d_selected_count_output,
                            sizeof(size_t),
                            hipMemcpyDeviceToHost));

        ASSERT_EQ(selected_count_output, size);

        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_selected_count_output));
    }
}

template<
    typename KeyType,
    typename ValueType,
    typename OutputKeyType = KeyType,
    typename OutputValueType = ValueType
>
struct DeviceUniqueByKeyParams
{
    using key_type = KeyType;
    using value_type = ValueType;
    using output_key_type = OutputKeyType;
    using output_value_type = OutputValueType;
};

template<class Params>
class HipcubDeviceUniqueByKeyTests : public ::testing::Test
{
public:
    using key_type = typename Params::key_type;
    using value_type = typename Params::value_type;
    using output_key_type = typename Params::output_key_type;
    using output_value_type = typename Params::output_value_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceUniqueByKeyParams<int, int>,
    DeviceUniqueByKeyParams<double, double>,
    DeviceUniqueByKeyParams<uint8_t, long long>,
    DeviceUniqueByKeyParams<test_utils::custom_test_type<double>, test_utils::custom_test_type<double>>
> HipcubDeviceUniqueByKeyTestsParams;

TYPED_TEST_SUITE(HipcubDeviceUniqueByKeyTests, HipcubDeviceUniqueByKeyTestsParams);

TYPED_TEST(HipcubDeviceUniqueByKeyTests, UniqueByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    using output_key_type = typename TestFixture::output_key_type;
    using output_value_type = typename TestFixture::output_value_type;

    const bool debug_synchronous = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream

    const auto sizes = get_sizes();
    const auto probabilities = get_discontinuity_probabilities();

    for (auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        for (auto p : probabilities)
        {
            for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
            {
                unsigned int seed_value = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
                SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<key_type> input_keys(size);
                {
                    std::vector<key_type> input01 = test_utils::get_random_data01<key_type>(size, p, seed_value);
                    test_utils::host_inclusive_scan(
                        input01.begin(), input01.end(), input_keys.begin(), hipcub::Sum()
                    );
                }

                const auto input_values = test_utils::get_random_data<value_type>(size, -1000, 1000, seed_value);                

                // Allocate and copy to device
                key_type* d_keys_input;
                value_type* d_values_input;
                output_key_type* d_keys_output;
                output_value_type* d_values_output;

                unsigned int* d_selected_count_output;

                HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, input_keys.size() * sizeof(input_keys[0])));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input, input_values.size() * sizeof(input_values[0])));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, input_keys.size() * sizeof(input_keys[0])));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output, input_values.size() * sizeof(input_values[0])));

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

                // Caclulate expected result on host
                std::vector<output_key_type> expected_keys;
                std::vector<output_value_type> expected_values;
                expected_keys.reserve(input_keys.size());
                expected_values.reserve(input_values.size());
                expected_keys.push_back(input_keys[0]);
                expected_values.push_back(input_values[0]);

                for (size_t i = 1; i < input_keys.size(); i++)
                {
                    if (!(input_keys[i-1] == input_keys[i]))
                    {
                        expected_keys.push_back(input_keys[i]);
                        expected_values.push_back(input_values[i]);
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get the size of d_temp_storage
                HIP_CHECK(
                    hipcub::DeviceSelect::UniqueByKey(
                        nullptr,
                        temp_storage_size_bytes,
                        d_keys_input,
                        d_values_input,
                        d_keys_output,
                        d_values_output,
                        d_selected_count_output,
                        input_keys.size(),
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

                // run
                HIP_CHECK(
                    hipcub::DeviceSelect::UniqueByKey(
                        d_temp_storage,
                        temp_storage_size_bytes,
                        d_keys_input,
                        d_values_input,
                        d_keys_output,
                        d_values_output,
                        d_selected_count_output,
                        input_keys.size(),
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
