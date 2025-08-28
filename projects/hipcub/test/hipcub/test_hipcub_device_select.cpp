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
#include <hipcub/device/device_select.hpp>
#include <hipcub/iterator/counting_input_iterator.hpp>
#include <hipcub/iterator/discard_output_iterator.hpp>

#include "single_index_iterator.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_half.hpp"

// Params for tests
template<class InputType,
         class OutputType = InputType,
         class FlagType   = unsigned int,
         bool UseGraphs   = false>
struct DeviceSelectParams
{
    using input_type                 = InputType;
    using output_type                = OutputType;
    using flag_type                  = FlagType;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
class HipcubDeviceSelectTests : public ::testing::Test
{
public:
    using input_type                 = typename Params::input_type;
    using output_type                = typename Params::output_type;
    using flag_type                  = typename Params::flag_type;
    static constexpr bool use_graphs = Params::use_graphs;
};

using HipcubDeviceSelectTestsParams
    = ::testing::Types<DeviceSelectParams<int, long>,
                       DeviceSelectParams<float, float>,
                       DeviceSelectParams<unsigned char, float>,
                       DeviceSelectParams<test_utils::half, test_utils::half>,
                       DeviceSelectParams<test_utils::bfloat16, test_utils::bfloat16>,
                       DeviceSelectParams<int, long, unsigned int, true>>;

TYPED_TEST_SUITE(HipcubDeviceSelectTests, HipcubDeviceSelectTestsParams);

TYPED_TEST(HipcubDeviceSelectTests, Flagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using F = typename TestFixture::flag_type;

    constexpr bool inplace = std::is_same<T, U>::value;

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
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(100),
                                                 seed_value);
            std::vector<F> flags = test_utils::get_random_data<F>(size,
                                                                  static_cast<F>(0),
                                                                  static_cast<F>(1),
                                                                  seed_value + seed_value_addition);

            T*            d_input;
            F*            d_flags;
            U*            d_output;
            unsigned int* d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, flags.size() * sizeof(F)));
            if HIPCUB_IF_CONSTEXPR(!inplace)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
            }
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(
                hipMemcpy(d_flags, flags.data(), flags.size() * sizeof(F), hipMemcpyHostToDevice));

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

            auto call = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
            {
                if HIPCUB_IF_CONSTEXPR(inplace)
                {
                    HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp_storage,
                                                            temp_storage_size_bytes,
                                                            d_input,
                                                            d_flags,
                                                            d_selected_count_output,
                                                            input.size(),
                                                            stream));
                }
                else
                {
                    HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp_storage,
                                                            temp_storage_size_bytes,
                                                            d_input,
                                                            d_flags,
                                                            d_output,
                                                            d_selected_count_output,
                                                            input.size(),
                                                            stream));
                }
            };

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            call(d_temp_storage, temp_storage_size_bytes);

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            call(d_temp_storage, temp_storage_size_bytes);

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            unsigned int selected_count_output = 0;
            HIP_CHECK(hipMemcpy(&selected_count_output,
                                d_selected_count_output,
                                sizeof(selected_count_output),
                                hipMemcpyDeviceToHost));
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            std::vector<U> output(input.size());
            if HIPCUB_IF_CONSTEXPR(inplace)
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_input,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            else
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_flags));
            if(!inplace)
            {
                HIP_CHECK(hipFree(d_output));
            }
            HIP_CHECK(hipFree(d_selected_count_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST(HipcubDeviceSelectTests, FlagNormalization)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = int;
    using U = long;
    using F = unsigned int;

    hipStream_t stream = 0; // default stream

    unsigned int seed_value = rand();

    for(size_t size : test_utils::get_sizes(seed_value))
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);
        rocprim::counting_iterator<T>    d_input(0);
        rocprim::counting_iterator<F>    d_flags(1);
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
                                                stream));

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
                                                stream));

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

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}

struct TestSelectOp
{
    template<class T>
    __host__ __device__
    inline bool
        operator()(const T& value) const
    {
        if(value > T(50))
            return true;
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

    constexpr bool inplace = std::is_same<T, U>::value;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    TestSelectOp select_op;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(0),
                                                 test_utils::convert_to_device<T>(100),
                                                 seed_value);

            T*            d_input;
            U*            d_output;
            unsigned int* d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            if HIPCUB_IF_CONSTEXPR(!inplace)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
            }
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

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

            auto call = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
            {
                if HIPCUB_IF_CONSTEXPR(inplace)
                {
                    HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_selected_count_output,
                                                       input.size(),
                                                       select_op,
                                                       stream));
                }
                else
                {
                    HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_output,
                                                       d_selected_count_output,
                                                       input.size(),
                                                       select_op,
                                                       stream));
                }
            };

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            call(d_temp_storage, temp_storage_size_bytes);

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            call(d_temp_storage, temp_storage_size_bytes);

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            unsigned int selected_count_output = 0;
            HIP_CHECK(hipMemcpy(&selected_count_output,
                                d_selected_count_output,
                                sizeof(selected_count_output),
                                hipMemcpyDeviceToHost));
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            std::vector<U> output(input.size());
            if HIPCUB_IF_CONSTEXPR(inplace)
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_input,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            else
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            if(!inplace)
            {
                HIP_CHECK(hipFree(d_output));
            }
            HIP_CHECK(hipFree(d_selected_count_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDeviceSelectTests, FlaggedIf)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using F = typename TestFixture::flag_type;

    constexpr bool inplace = std::is_same<T, U>::value;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    TestSelectOp select_flag_op;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            // Generate data
            std::vector<T> input
                = test_utils::get_random_data<T>(size,
                                                 test_utils::convert_to_device<T>(1),
                                                 test_utils::convert_to_device<T>(100),
                                                 seed_value);
            std::vector<F> flags = test_utils::get_random_data<F>(size,
                                                                  static_cast<F>(0),
                                                                  static_cast<F>(100),
                                                                  seed_value + seed_value_addition);

            T*            d_input;
            F*            d_flags;
            U*            d_output;
            unsigned int* d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, flags.size() * sizeof(F)));
            if HIPCUB_IF_CONSTEXPR(!inplace)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
            }
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(
                hipMemcpy(d_flags, flags.data(), flags.size() * sizeof(F), hipMemcpyHostToDevice));

            // Calculate expected results on host
            std::vector<U> expected;
            expected.reserve(input.size());
            for(size_t i = 0; i < input.size(); i++)
            {
                if(select_flag_op(flags[i]))
                {
                    expected.push_back(input[i]);
                }
            }

            auto call = [&](void* d_temp_storage, size_t& temp_storage_size_bytes)
            {
                if HIPCUB_IF_CONSTEXPR(inplace)
                {
                    HIP_CHECK(hipcub::DeviceSelect::FlaggedIf(d_temp_storage,
                                                              temp_storage_size_bytes,
                                                              d_input,
                                                              d_flags,
                                                              d_selected_count_output,
                                                              input.size(),
                                                              select_flag_op,
                                                              stream));
                }
                else
                {
                    HIP_CHECK(hipcub::DeviceSelect::FlaggedIf(d_temp_storage,
                                                              temp_storage_size_bytes,
                                                              d_input,
                                                              d_flags,
                                                              d_output,
                                                              d_selected_count_output,
                                                              input.size(),
                                                              select_flag_op,
                                                              stream));
                }
            };

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            call(d_temp_storage, temp_storage_size_bytes);

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0U);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            call(d_temp_storage, temp_storage_size_bytes);

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            unsigned int selected_count_output = 0;
            HIP_CHECK(hipMemcpy(&selected_count_output,
                                d_selected_count_output,
                                sizeof(selected_count_output),
                                hipMemcpyDeviceToHost));
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            std::vector<U> output(input.size());
            if HIPCUB_IF_CONSTEXPR(inplace)
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_input,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }
            else
            {
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_flags));
            if(!inplace)
            {
                HIP_CHECK(hipFree(d_output));
            }
            HIP_CHECK(hipFree(d_selected_count_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

std::vector<float> get_discontinuity_probabilities()
{
    std::vector<float> probabilities = {0.0001, 0.05, 0.25, 0.5, 0.75, 0.95};
    return probabilities;
}

TYPED_TEST(HipcubDeviceSelectTests, Unique)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const auto probabilities = get_discontinuity_probabilities();
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p= " << p);

                // Generate data
                std::vector<T> input(size);
                {
                    std::vector<T> input01 = test_utils::get_random_data01<T>(size, p, seed_value);
                    test_utils::host_inclusive_scan(input01.begin(),
                                                    input01.end(),
                                                    input.begin(),
                                                    hipcub::Sum());
                }

                // Allocate and copy to device
                T*            d_input;
                U*            d_output;
                unsigned int* d_selected_count_output;
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, input.size() * sizeof(U)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output,
                                                             sizeof(unsigned int)));
                HIP_CHECK(hipMemcpy(d_input,
                                    input.data(),
                                    input.size() * sizeof(T),
                                    hipMemcpyHostToDevice));
                HIP_CHECK(hipDeviceSynchronize());

                // Calculate expected results on host
                std::vector<U> expected;
                expected.reserve(input.size());
                expected.push_back(input[0]);
                for(size_t i = 1; i < input.size(); i++)
                {
                    if(!(test_utils::convert_to_native(input[i - 1])
                         == test_utils::convert_to_native(input[i])))
                    {
                        expected.push_back(input[i]);
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(hipcub::DeviceSelect::Unique(nullptr,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_output,
                                                       d_selected_count_output,
                                                       input.size(),
                                                       stream));
                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0U);

                // allocate temporary storage
                void* d_temp_storage = nullptr;
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));
                HIP_CHECK(hipDeviceSynchronize());

                test_utils::GraphHelper gHelper;
                if (TestFixture::use_graphs)
                    gHelper.startStreamCapture(stream);

                // Run
                HIP_CHECK(hipcub::DeviceSelect::Unique(d_temp_storage,
                                                       temp_storage_size_bytes,
                                                       d_input,
                                                       d_output,
                                                       d_selected_count_output,
                                                       input.size(),
                                                       stream));

                if (TestFixture::use_graphs)
                    gHelper.createAndLaunchGraph(stream);

                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                unsigned int selected_count_output = 0;
                HIP_CHECK(hipMemcpy(&selected_count_output,
                                    d_selected_count_output,
                                    sizeof(unsigned int),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipDeviceSynchronize());
                ASSERT_EQ(selected_count_output, expected.size());

                // Check if output values are as expected
                std::vector<U> output(input.size());
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(U),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipDeviceSynchronize());

                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

                if(TestFixture::use_graphs)
                    gHelper.cleanupGraphHelper();

                HIP_CHECK(hipFree(d_input));
                HIP_CHECK(hipFree(d_output));
                HIP_CHECK(hipFree(d_selected_count_output));
                HIP_CHECK(hipFree(d_temp_storage));
            }
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST(HipcubDeviceSelectTests, UniqueDiscardOutputIterator)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = 0; //default stream

    unsigned int seed_value = rand();

    for(size_t size : test_utils::get_sizes(seed_value))
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);
        rocprim::counting_iterator<unsigned int>    d_input(0);
        rocprim::discard_iterator                   d_output;
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
                                               stream));

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
                                               stream));

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

template<class T>
struct TestLargeIndicesSelectOp
{
    T max_value;
    __host__ __device__
    inline bool
        operator()(const T& value) const
    {
        return test_utils::less()(value, T(max_value));
    }
};

class HipcubDeviceSelectLargeIndicesTests : public ::testing::TestWithParam<unsigned int>
{
public:
    const bool debug_synchronous = false;
};

INSTANTIATE_TEST_SUITE_P(HipcubDeviceSelectLargeIndicesTest,
                         HipcubDeviceSelectLargeIndicesTests,
                         ::testing::Values(2048, 9643, 32768, 38713, 38713));

TEST_P(HipcubDeviceSelectLargeIndicesTests, LargeIndicesSelectOp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                   = size_t; // input_type
    using U                   = size_t; // output_type
    using selected_count_type = size_t;

    hipStream_t stream = 0; // default stream

    const auto selected_size = GetParam();

    for(size_t size : test_utils::get_large_sizes(0))
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

// Support for large indices in DeviceSelect is not implemented in CUB yet. Disable test meanwhile.
#ifdef __HIP_PLATFORM_NVIDIA__
        std::cout << "Test disabled for large sizes until support is present in CUB" << std::endl;
        GTEST_SKIP();
#endif

        // Generate data
        rocprim::counting_iterator<T>    d_input(0);
        U*                               d_output;
        selected_count_type*             d_selected_count_output;
        selected_count_type              expected_output_size = selected_size;
        TestLargeIndicesSelectOp<T>      select_op{expected_output_size};
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_output,
                                                     sizeof(d_output[0]) * expected_output_size));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output,
                                                     sizeof(d_selected_count_output[0])));

        // Calculate expected results on host
        std::vector<U> expected_output(expected_output_size);
        std::iota(expected_output.begin(), expected_output.end(), U(0));

        // Temp storage
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get the size of d_temp_storage
        HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                           temp_storage_size_bytes,
                                           d_input,
                                           d_output,
                                           d_selected_count_output,
                                           size,
                                           select_op,
                                           stream));

        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // Allocate temporary storage
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

        // Run
        HIP_CHECK(hipcub::DeviceSelect::If(d_temp_storage,
                                           temp_storage_size_bytes,
                                           d_input,
                                           d_output,
                                           d_selected_count_output,
                                           size,
                                           select_op,
                                           stream));
        HIP_CHECK(hipDeviceSynchronize());

        // Check if number of selected value is as expected
        selected_count_type selected_count_output = 0;
        HIP_CHECK(hipMemcpy(&selected_count_output,
                            d_selected_count_output,
                            sizeof(*d_selected_count_output),
                            hipMemcpyDeviceToHost));
        ASSERT_EQ(selected_count_output, selected_size);

        // Check if outputs are as expected
        std::vector<U> output(expected_output_size);
        HIP_CHECK(hipMemcpy(output.data(),
                            d_output,
                            sizeof(*d_output) * expected_output_size,
                            hipMemcpyDeviceToHost));

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_eq(output, expected_output, expected_output_size));

        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_selected_count_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}

template<typename KeyType,
         typename ValueType,
         typename EqualityOp        = hipcub::Equality,
         typename OutputKeyType     = KeyType,
         typename OutputValueType   = ValueType,
         typename SelectedCountType = unsigned int,
         bool UseGraphs             = false>
struct DeviceUniqueByKeyParams
{
    using key_type                   = KeyType;
    using value_type                 = ValueType;
    using equality_op                = EqualityOp;
    using output_key_type            = OutputKeyType;
    using output_value_type          = OutputValueType;
    using selected_count_type        = SelectedCountType;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
class HipcubDeviceUniqueByKeyTests : public ::testing::Test
{
public:
    using key_type                   = typename Params::key_type;
    using value_type                 = typename Params::value_type;
    using equality_op                = typename Params::equality_op;
    using output_key_type            = typename Params::output_key_type;
    using output_value_type          = typename Params::output_value_type;
    using selected_count_type        = typename Params::selected_count_type;
    static constexpr bool use_graphs = Params::use_graphs;
};

struct TestUniqueEqualityOp
{
    static constexpr int segment = 3;

    template<typename T>
    HIPCUB_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        static_assert(std::is_integral<T>::value, "T must be integral type");
        return a / segment == b / segment;
    }
};

using HipcubDeviceUniqueByKeyTestsParams = ::testing::Types<
    DeviceUniqueByKeyParams<int, int>,
    DeviceUniqueByKeyParams<size_t, int, TestUniqueEqualityOp, double, long long, size_t>,
    DeviceUniqueByKeyParams<double, float, hipcub::Equality, double, double, size_t>,
    DeviceUniqueByKeyParams<uint8_t, long long>,
    DeviceUniqueByKeyParams<test_utils::custom_test_type<double>,
                            test_utils::custom_test_type<int>>,
    DeviceUniqueByKeyParams<int, int, hipcub::Equality, int, int, unsigned int, true>>;

TYPED_TEST_SUITE(HipcubDeviceUniqueByKeyTests, HipcubDeviceUniqueByKeyTestsParams);

TYPED_TEST(HipcubDeviceUniqueByKeyTests, UniqueByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type            = typename TestFixture::key_type;
    using value_type          = typename TestFixture::value_type;
    using output_key_type     = typename TestFixture::output_key_type;
    using output_value_type   = typename TestFixture::output_value_type;
    using selected_count_type = typename TestFixture::selected_count_type;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const auto probabilities = get_discontinuity_probabilities();
    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);

            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p= " << p);

                // Generate data
                std::vector<key_type> input_keys(size);
                {
                    std::vector<key_type> input01
                        = test_utils::get_random_data01<key_type>(size, p, seed_value);
                    test_utils::host_inclusive_scan(input01.begin(),
                                                    input01.end(),
                                                    input_keys.begin(),
                                                    hipcub::Sum());
                }

                const auto input_values
                    = test_utils::get_random_data<value_type>(size, -1000, 1000, seed_value);

                // Allocate and copy to device
                key_type*          d_keys_input;
                value_type*        d_values_input;
                output_key_type*   d_keys_output;
                output_value_type* d_values_output;

                selected_count_type* d_selected_count_output;

                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_keys_input,
                                                       input_keys.size() * sizeof(*d_keys_input)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_input,
                                                             input_values.size()
                                                                 * sizeof(*d_values_input)));
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_keys_output,
                                                       input_keys.size() * sizeof(*d_keys_output)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_values_output,
                                                             input_values.size()
                                                                 * sizeof(*d_values_output)));

                HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output,
                                                             sizeof(*d_selected_count_output)));

                HIP_CHECK(hipMemcpy(d_keys_input,
                                    input_keys.data(),
                                    input_keys.size() * sizeof(input_keys[0]),
                                    hipMemcpyHostToDevice));

                HIP_CHECK(hipMemcpy(d_values_input,
                                    input_values.data(),
                                    input_values.size() * sizeof(input_values[0]),
                                    hipMemcpyHostToDevice));

                // Caclulate expected result on host
                std::vector<output_key_type>   expected_keys;
                std::vector<output_value_type> expected_values;
                expected_keys.reserve(input_keys.size());
                expected_values.reserve(input_values.size());
                expected_keys.push_back(input_keys[0]);
                expected_values.push_back(input_values[0]);

                typename TestFixture::equality_op equality_op;

                for(size_t i = 1; i < input_keys.size(); ++i)
                {
                    if(!equality_op(input_keys[i - 1], input_keys[i]))
                    {
                        expected_keys.push_back(input_keys[i]);
                        expected_values.push_back(input_values[i]);
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get the size of d_temp_storage
                HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(nullptr,
                                                            temp_storage_size_bytes,
                                                            d_keys_input,
                                                            d_values_input,
                                                            d_keys_output,
                                                            d_values_output,
                                                            d_selected_count_output,
                                                            input_keys.size(),
                                                            equality_op,
                                                            stream));

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                void* d_temp_storage = nullptr;
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

                test_utils::GraphHelper gHelper;
                if (TestFixture::use_graphs)
                    gHelper.startStreamCapture(stream);

                // run
                HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(d_temp_storage,
                                                            temp_storage_size_bytes,
                                                            d_keys_input,
                                                            d_values_input,
                                                            d_keys_output,
                                                            d_values_output,
                                                            d_selected_count_output,
                                                            input_keys.size(),
                                                            equality_op,
                                                            stream));

                if (TestFixture::use_graphs)
                    gHelper.createAndLaunchGraph(stream);

                // Check if number of selected value is as expected
                selected_count_type selected_count_output = 0;
                HIP_CHECK(hipMemcpy(&selected_count_output,
                                    d_selected_count_output,
                                    sizeof(selected_count_output),
                                    hipMemcpyDeviceToHost));

                ASSERT_EQ(selected_count_output, expected_keys.size());

                // Check if outputs are as expected
                std::vector<output_key_type> output_keys(input_keys.size());

                HIP_CHECK(hipMemcpy(output_keys.data(),
                                    d_keys_output,
                                    output_keys.size() * sizeof(output_keys[0]),
                                    hipMemcpyDeviceToHost));

                std::vector<output_value_type> output_values(input_values.size());

                HIP_CHECK(hipMemcpy(output_values.data(),
                                    d_values_output,
                                    output_values.size() * sizeof(output_values[0]),
                                    hipMemcpyDeviceToHost));

                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(output_keys, expected_keys, expected_keys.size()));
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(output_values, expected_values, expected_values.size()));

                if(TestFixture::use_graphs)
                    gHelper.cleanupGraphHelper();

                HIP_CHECK(hipFree(d_keys_input));
                HIP_CHECK(hipFree(d_values_input));
                HIP_CHECK(hipFree(d_keys_output));
                HIP_CHECK(hipFree(d_values_output));
                HIP_CHECK(hipFree(d_selected_count_output));
                HIP_CHECK(hipFree(d_temp_storage));
            }
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST(HipcubDeviceUniqueByKeyTests, LargeIndicesUniqueByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type            = size_t;
    using value_type          = size_t;
    using selected_count_type = size_t;

    hipStream_t stream = 0; // default stream

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        for(size_t size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            TestUniqueEqualityOp equality_op;

            const size_t selected_count
                = (size + TestUniqueEqualityOp::segment - 1) / TestUniqueEqualityOp::segment;
            const size_t output_index = selected_count - 1;
            const size_t input_index  = output_index * TestUniqueEqualityOp::segment;
            rocprim::counting_iterator<key_type>   d_keys_input(0);
            rocprim::counting_iterator<value_type> d_values_input(123);
            key_type*   d_keys_output;
            value_type* d_values_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, sizeof(*d_keys_output)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, sizeof(*d_values_output)));
            test_utils::single_index_iterator<key_type> keys_output_it(d_keys_output, output_index);
            test_utils::single_index_iterator<value_type> values_output_it(d_values_output,
                                                                           output_index);

            selected_count_type* d_selected_count_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_selected_count_output,
                                                         sizeof(*d_selected_count_output)));

            // temp storage
            size_t temp_storage_size_bytes;
            // Get the size of d_temp_storage
            HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(nullptr,
                                                        temp_storage_size_bytes,
                                                        d_keys_input,
                                                        d_values_input,
                                                        keys_output_it,
                                                        values_output_it,
                                                        d_selected_count_output,
                                                        size,
                                                        equality_op,
                                                        stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void* d_temp_storage = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            // run
            HIP_CHECK(hipcub::DeviceSelect::UniqueByKey(d_temp_storage,
                                                        temp_storage_size_bytes,
                                                        d_keys_input,
                                                        d_values_input,
                                                        keys_output_it,
                                                        values_output_it,
                                                        d_selected_count_output,
                                                        size,
                                                        equality_op,
                                                        stream));

            // Check if number of selected value is as expected
            selected_count_type selected_count_output = 0;
            HIP_CHECK(hipMemcpy(&selected_count_output,
                                d_selected_count_output,
                                sizeof(*d_selected_count_output),
                                hipMemcpyDeviceToHost));

            ASSERT_EQ(selected_count_output, selected_count);

            // Check if outputs are as expected
            key_type   output_key;
            value_type output_value;
            HIP_CHECK(hipMemcpy(&output_key,
                                d_keys_output,
                                sizeof(*d_keys_output),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&output_value,
                                d_values_output,
                                sizeof(*d_values_output),
                                hipMemcpyDeviceToHost));

            const key_type   expected_key   = d_keys_input[input_index];
            const value_type expected_value = d_values_input[input_index];
            ASSERT_EQ(output_key, expected_key);
            ASSERT_EQ(output_value, expected_value);

            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));
            HIP_CHECK(hipFree(d_selected_count_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }
}
