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
#include "identity_iterator.hpp"
#include <hipcub/device/device_partition.hpp>

#include "test_utils_data_generation.hpp"

#include <algorithm>
#include <array>

// Params for tests
template<class InputType,
         class OutputType         = InputType,
         class FlagType           = unsigned int,
         bool UseIdentityIterator = false,
         bool UseGraphs           = false>
struct DevicePartitionParams
{
    using input_type                            = InputType;
    using output_type                           = OutputType;
    using flag_type                             = FlagType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr bool use_graphs            = UseGraphs;
};

template<class Params>
class HipcubDevicePartitionTests : public ::testing::Test
{
public:
    using input_type                            = typename Params::input_type;
    using output_type                           = typename Params::output_type;
    using flag_type                             = typename Params::flag_type;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    static constexpr bool use_graphs            = Params::use_graphs;
};

using HipcubDevicePartitionTestsParams
    = ::testing::Types<DevicePartitionParams<int, int, unsigned char, true>,
                       DevicePartitionParams<unsigned int, unsigned long>,
                       DevicePartitionParams<unsigned char, float>,
                       DevicePartitionParams<int8_t, int8_t>,
                       DevicePartitionParams<uint8_t, uint8_t>,
                       DevicePartitionParams<test_utils::custom_test_type<long long>>,
                       DevicePartitionParams<int, int, unsigned char, false, true>>;

TYPED_TEST_SUITE(HipcubDevicePartitionTests, HipcubDevicePartitionTestsParams);

TYPED_TEST(HipcubDevicePartitionTests, Flagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    using F                                     = typename TestFixture::flag_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

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
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<F> flags = test_utils::get_random_data01<F>(size, 0.25, seed_value);

            T*            d_input;
            F*            d_flags;
            U*            d_output;
            unsigned int* d_selected_count_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(F)));
            HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(U)));
            HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(
                hipMemcpy(d_flags, flags.data(), flags.size() * sizeof(F), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected_selected and expected_rejected results on host
            std::vector<U> expected_selected;
            std::vector<U> expected_rejected;
            expected_selected.reserve(input.size() / 2);
            expected_rejected.reserve(input.size() / 2);
            for(size_t i = 0; i < input.size(); i++)
            {
                if(flags[i] != 0)
                {
                    expected_selected.push_back(input[i]);
                }
                else
                {
                    expected_rejected.push_back(input[i]);
                }
            }
            std::reverse(expected_rejected.begin(), expected_rejected.end());

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DevicePartition::Flagged(
                nullptr,
                temp_storage_size_bytes,
                d_input,
                d_flags,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                stream));
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void* d_temp_storage = nullptr;
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::DevicePartition::Flagged(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                d_flags,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected_selected
            unsigned int selected_count_output = 0;
            HIP_CHECK(hipMemcpy(&selected_count_output,
                                d_selected_count_output,
                                sizeof(unsigned int),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_EQ(selected_count_output, expected_selected.size());

            // Check if output values are as expected_selected
            std::vector<U> output(input.size());
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(U),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<U> output_rejected;
            for(size_t i = 0; i < expected_rejected.size(); i++)
            {
                auto j = i + expected_selected.size();
                output_rejected.push_back(output[j]);
            }
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_eq(output, expected_selected, expected_selected.size()));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_rejected,
                                                          expected_rejected,
                                                          expected_rejected.size()));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_flags));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_selected_count_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDevicePartitionTests, FlaggedLarge)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    using F                                     = typename TestFixture::flag_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }
    size_t seed_index = 0;
    size_t size       = 200'000;

    unsigned int seed_value
        = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
    std::vector<F> flags = test_utils::get_random_data01<F>(size, 0.25, seed_value);

    T*            d_input;
    F*            d_flags;
    U*            d_output;
    unsigned int* d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_flags, flags.size() * sizeof(F)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(U)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_flags, flags.data(), flags.size() * sizeof(F), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Calculate expected_selected and expected_rejected results on host
    std::vector<U> expected_selected;
    std::vector<U> expected_rejected;
    expected_selected.reserve(input.size() / 2);
    expected_rejected.reserve(input.size() / 2);
    for(size_t i = 0; i < input.size(); i++)
    {
        if(flags[i] != 0)
        {
            expected_selected.push_back(input[i]);
        }
        else
        {
            expected_rejected.push_back(input[i]);
        }
    }
    std::reverse(expected_rejected.begin(), expected_rejected.end());

    // temp storage
    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DevicePartition::Flagged(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        d_flags,
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
        d_selected_count_output,
        input.size(),
        stream));
    HIP_CHECK(hipDeviceSynchronize());

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    test_utils::GraphHelper gHelper;
    if(TestFixture::use_graphs)
        gHelper.startStreamCapture(stream);

    // Run
    HIP_CHECK(hipcub::DevicePartition::Flagged(
        d_temp_storage,
        temp_storage_size_bytes,
        d_input,
        d_flags,
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
        d_selected_count_output,
        static_cast<size_t>(input.size()),
        stream));

    if(TestFixture::use_graphs)
        gHelper.createAndLaunchGraph(stream);

    HIP_CHECK(hipDeviceSynchronize());

    // Check if number of selected value is as expected_selected
    unsigned int selected_count_output = 0;
    HIP_CHECK(hipMemcpy(&selected_count_output,
                        d_selected_count_output,
                        sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    ASSERT_EQ(selected_count_output, expected_selected.size());

    // Check if output values are as expected_selected
    std::vector<U> output(input.size());
    HIP_CHECK(hipMemcpy(output.data(), d_output, output.size() * sizeof(U), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<U> output_rejected;
    for(size_t i = 0; i < expected_rejected.size(); i++)
    {
        auto j = i + expected_selected.size();
        output_rejected.push_back(output[j]);
    }
    ASSERT_NO_FATAL_FAILURE(
        test_utils::assert_eq(output, expected_selected, expected_selected.size()));
    ASSERT_NO_FATAL_FAILURE(
        test_utils::assert_eq(output_rejected, expected_rejected, expected_rejected.size()));

    if(TestFixture::use_graphs)
        gHelper.cleanupGraphHelper();

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_selected_count_output));
    HIP_CHECK(hipFree(d_temp_storage));

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

// NOTE: The following lambdas cannot be inside the test because of nvcc
// The enclosing parent function ("TestBody") for an extended __host__ __device__ lambda cannot have private or protected access within its class
struct TestSelectOp
{
    template<class T>
    __host__ __device__
    bool operator()(const T& value) const
    {
        if(value == T(50))
            return true;
        return false;
    }
};

TYPED_TEST(HipcubDevicePartitionTests, If)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

#ifdef __HIP_PLATFORM_NVIDIA__
    TestSelectOp select_op;
#else
    auto select_op = [](const T& value) -> bool
    {
        if(value == T(50))
            return true;
        return false;
    };
#endif

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
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

            T*            d_input;
            U*            d_output;
            unsigned int* d_selected_count_output;
            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(U)));
            HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
            HIP_CHECK(hipDeviceSynchronize());

            // Calculate expected_selected and expected_rejected results on host
            std::vector<U> expected_selected;
            std::vector<U> expected_rejected;
            expected_selected.reserve(input.size() / 2);
            expected_rejected.reserve(input.size() / 2);
            for(size_t i = 0; i < input.size(); i++)
            {
                if(select_op(input[i]))
                {
                    expected_selected.push_back(input[i]);
                }
                else
                {
                    expected_rejected.push_back(input[i]);
                }
            }
            std::reverse(expected_rejected.begin(), expected_rejected.end());

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DevicePartition::If(
                nullptr,
                temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                select_op,
                stream));
            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void* d_temp_storage = nullptr;
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
            HIP_CHECK(hipDeviceSynchronize());

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::DevicePartition::If(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                d_selected_count_output,
                input.size(),
                select_op,
                stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected_selected
            unsigned int selected_count_output = 0;
            HIP_CHECK(hipMemcpy(&selected_count_output,
                                d_selected_count_output,
                                sizeof(unsigned int),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());
            ASSERT_EQ(selected_count_output, expected_selected.size());

            // Check if output values are as expected_selected
            std::vector<U> output(input.size());
            HIP_CHECK(hipMemcpy(output.data(),
                                d_output,
                                output.size() * sizeof(U),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<U> output_rejected;
            for(size_t i = 0; i < expected_rejected.size(); i++)
            {
                auto j = i + expected_selected.size();
                output_rejected.push_back(output[j]);
            }
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_eq(output, expected_selected, expected_selected.size()));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_rejected,
                                                          expected_rejected,
                                                          expected_rejected.size()));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_selected_count_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDevicePartitionTests, IfLarge)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

#ifdef __HIP_PLATFORM_NVIDIA__
    TestSelectOp select_op;
#else
    auto select_op = [](const T& value) -> bool
    {
        if(value == T(50))
            return true;
        return false;
    };
#endif

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    size_t       seed_index = 0;
    size_t       size       = 200'000;
    unsigned int seed_value
        = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    SCOPED_TRACE(testing::Message() << "with size= " << size);
    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

    T*            d_input;
    U*            d_output;
    unsigned int* d_selected_count_output;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(U)));
    HIP_CHECK(hipMalloc(&d_selected_count_output, sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Calculate expected_selected and expected_rejected results on host
    std::vector<U> expected_selected;
    std::vector<U> expected_rejected;
    expected_selected.reserve(input.size() / 2);
    expected_rejected.reserve(input.size() / 2);
    for(size_t i = 0; i < input.size(); i++)
    {
        if(select_op(input[i]))
        {
            expected_selected.push_back(input[i]);
        }
        else
        {
            expected_rejected.push_back(input[i]);
        }
    }
    std::reverse(expected_rejected.begin(), expected_rejected.end());

    // temp storage
    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DevicePartition::If(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
        d_selected_count_output,
        input.size(),
        select_op,
        stream));
    HIP_CHECK(hipDeviceSynchronize());

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    test_utils::GraphHelper gHelper;
    if(TestFixture::use_graphs)
        gHelper.startStreamCapture(stream);

    // Run
    HIP_CHECK(hipcub::DevicePartition::If(
        d_temp_storage,
        temp_storage_size_bytes,
        d_input,
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
        d_selected_count_output,
        static_cast<size_t>(input.size()),
        select_op,
        stream));

    if(TestFixture::use_graphs)
        gHelper.createAndLaunchGraph(stream);

    HIP_CHECK(hipDeviceSynchronize());

    // Check if number of selected value is as expected_selected
    unsigned int selected_count_output = 0;
    HIP_CHECK(hipMemcpy(&selected_count_output,
                        d_selected_count_output,
                        sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    ASSERT_EQ(selected_count_output, expected_selected.size());

    // Check if output values are as expected_selected
    std::vector<U> output(input.size());
    HIP_CHECK(hipMemcpy(output.data(), d_output, output.size() * sizeof(U), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<U> output_rejected;
    for(size_t i = 0; i < expected_rejected.size(); i++)
    {
        auto j = i + expected_selected.size();
        output_rejected.push_back(output[j]);
    }
    ASSERT_NO_FATAL_FAILURE(
        test_utils::assert_eq(output, expected_selected, expected_selected.size()));
    ASSERT_NO_FATAL_FAILURE(
        test_utils::assert_eq(output_rejected, expected_rejected, expected_rejected.size()));

    if(TestFixture::use_graphs)
        gHelper.cleanupGraphHelper();

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_selected_count_output));
    HIP_CHECK(hipFree(d_temp_storage));

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

namespace
{
template<typename T>
struct LessOp
{
    HIPCUB_HOST_DEVICE LessOp(const T& pivot) : pivot_{pivot} {}

    HIPCUB_HOST_DEVICE
    bool operator()(const T& val) const
    {
        return val < pivot_;
    }

private:
    T pivot_;
};
} // namespace

TYPED_TEST(HipcubDevicePartitionTests, IfThreeWay)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

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
            const auto input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

            // Outputs
            auto expected_first_output      = std::vector<U>(input.size());
            auto expected_second_output     = std::vector<U>(input.size());
            auto expected_unselected_output = std::vector<U>(input.size());
            auto selected_counts            = std::array<unsigned int, 2>{};

            T*            d_input             = nullptr;
            U*            d_first_output      = nullptr;
            U*            d_second_output     = nullptr;
            U*            d_unselected_output = nullptr;
            unsigned int* d_selected_counts   = nullptr;

            HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
            HIP_CHECK(hipMalloc(&d_first_output, input.size() * sizeof(U)));
            HIP_CHECK(hipMalloc(&d_second_output, input.size() * sizeof(U)));
            HIP_CHECK(hipMalloc(&d_unselected_output, input.size() * sizeof(U)));
            HIP_CHECK(hipMalloc(&d_selected_counts, sizeof(selected_counts)));
            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                input.size() * sizeof(input[0]),
                                hipMemcpyHostToDevice));

            const auto first_op  = LessOp<T>{static_cast<T>(30)};
            const auto second_op = LessOp<T>{static_cast<T>(60)};

            auto       copy          = input;
            const auto partion_point = std::stable_partition(copy.begin(), copy.end(), first_op);
            const auto second_partiton_point
                = std::stable_partition(partion_point, copy.end(), second_op);

            const auto expected_counts = std::array<unsigned int, 2>{
                static_cast<unsigned int>(partion_point - copy.begin()),
                static_cast<unsigned int>(second_partiton_point - partion_point)};

            const auto expected = [&]
            {
                auto result = std::vector<U>(copy.size());
                std::copy(copy.cbegin(), copy.cend(), result.begin());
                return result;
            }();

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(hipcub::DevicePartition::If(
                nullptr,
                temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_first_output),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_second_output),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unselected_output),
                d_selected_counts,
                static_cast<int>(input.size()),
                first_op,
                second_op,
                stream));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            void* d_temp_storage = nullptr;
            HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::use_graphs)
                gHelper.startStreamCapture(stream);

            // Run
            HIP_CHECK(hipcub::DevicePartition::If(
                d_temp_storage,
                temp_storage_size_bytes,
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_first_output),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_second_output),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unselected_output),
                d_selected_counts,
                static_cast<int>(input.size()),
                first_op,
                second_op,
                stream));

            if (TestFixture::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected_selected
            HIP_CHECK(hipMemcpy(selected_counts.data(),
                                d_selected_counts,
                                sizeof(selected_counts),
                                hipMemcpyDeviceToHost));
            ASSERT_EQ(selected_counts, expected_counts);

            // Check if output values are as expected_selected
            const auto output = [&]
            {
                auto result = std::vector<U>(input.size());
                HIP_CHECK(hipMemcpy(result.data(),
                                    d_first_output,
                                    expected_counts[0] * sizeof(result[0]),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(result.data() + expected_counts[0],
                                    d_second_output,
                                    expected_counts[1] * sizeof(result[0]),
                                    hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(result.data() + expected_counts[0] + expected_counts[1],
                                    d_unselected_output,
                                    (input.size() - expected_counts[0] - expected_counts[1])
                                        * sizeof(result[0]),
                                    hipMemcpyDeviceToHost));
                return result;
            }();

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

            if(TestFixture::use_graphs)
                gHelper.cleanupGraphHelper();

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_first_output));
            HIP_CHECK(hipFree(d_second_output));
            HIP_CHECK(hipFree(d_unselected_output));
            HIP_CHECK(hipFree(d_selected_counts));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(HipcubDevicePartitionTests, IfThreeWayLarge)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    hipStream_t stream = 0; // default
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    size_t       seed_index = 0;
    size_t       size       = 200'000;
    unsigned int seed_value
        = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    SCOPED_TRACE(testing::Message() << "with size= " << size);
    // Generate data
    const auto input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

    // Outputs
    auto expected_first_output      = std::vector<U>(input.size());
    auto expected_second_output     = std::vector<U>(input.size());
    auto expected_unselected_output = std::vector<U>(input.size());
    auto selected_counts            = std::array<unsigned int, 2>{};

    T*            d_input             = nullptr;
    U*            d_first_output      = nullptr;
    U*            d_second_output     = nullptr;
    U*            d_unselected_output = nullptr;
    unsigned int* d_selected_counts   = nullptr;

    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
    HIP_CHECK(hipMalloc(&d_first_output, input.size() * sizeof(U)));
    HIP_CHECK(hipMalloc(&d_second_output, input.size() * sizeof(U)));
    HIP_CHECK(hipMalloc(&d_unselected_output, input.size() * sizeof(U)));
    HIP_CHECK(hipMalloc(&d_selected_counts, sizeof(selected_counts)));
    HIP_CHECK(
        hipMemcpy(d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

    const auto first_op  = LessOp<T>{static_cast<T>(30)};
    const auto second_op = LessOp<T>{static_cast<T>(60)};

    auto       copy                  = input;
    const auto partion_point         = std::stable_partition(copy.begin(), copy.end(), first_op);
    const auto second_partiton_point = std::stable_partition(partion_point, copy.end(), second_op);

    const auto expected_counts = std::array<unsigned int, 2>{
        static_cast<unsigned int>(partion_point - copy.begin()),
        static_cast<unsigned int>(second_partiton_point - partion_point)};

    const auto expected = [&]
    {
        auto result = std::vector<U>(copy.size());
        std::copy(copy.cbegin(), copy.cend(), result.begin());
        return result;
    }();

    // temp storage
    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(hipcub::DevicePartition::If(
        nullptr,
        temp_storage_size_bytes,
        d_input,
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_first_output),
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_second_output),
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unselected_output),
        d_selected_counts,
        static_cast<int>(input.size()),
        first_op,
        second_op,
        stream));

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    void* d_temp_storage = nullptr;
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

    test_utils::GraphHelper gHelper;
    if(TestFixture::use_graphs)
        gHelper.startStreamCapture(stream);

    // Run
    HIP_CHECK(hipcub::DevicePartition::If(
        d_temp_storage,
        temp_storage_size_bytes,
        d_input,
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_first_output),
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_second_output),
        test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unselected_output),
        d_selected_counts,
        static_cast<size_t>(input.size()),
        first_op,
        second_op,
        stream));

    if(TestFixture::use_graphs)
        gHelper.createAndLaunchGraph(stream);

    HIP_CHECK(hipDeviceSynchronize());

    // Check if number of selected value is as expected_selected
    HIP_CHECK(hipMemcpy(selected_counts.data(),
                        d_selected_counts,
                        sizeof(selected_counts),
                        hipMemcpyDeviceToHost));
    ASSERT_EQ(selected_counts, expected_counts);

    // Check if output values are as expected_selected
    const auto output = [&]
    {
        auto result = std::vector<U>(input.size());
        HIP_CHECK(hipMemcpy(result.data(),
                            d_first_output,
                            expected_counts[0] * sizeof(result[0]),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(result.data() + expected_counts[0],
                            d_second_output,
                            expected_counts[1] * sizeof(result[0]),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(result.data() + expected_counts[0] + expected_counts[1],
                      d_unselected_output,
                      (input.size() - expected_counts[0] - expected_counts[1]) * sizeof(result[0]),
                      hipMemcpyDeviceToHost));
        return result;
    }();

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));

    if(TestFixture::use_graphs)
        gHelper.cleanupGraphHelper();

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_first_output));
    HIP_CHECK(hipFree(d_second_output));
    HIP_CHECK(hipFree(d_unselected_output));
    HIP_CHECK(hipFree(d_selected_counts));
    HIP_CHECK(hipFree(d_temp_storage));

    if(TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}
