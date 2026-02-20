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

#include "../common_test_header.hpp"

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "identity_iterator.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required rocprim headers
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_select.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <numeric>
#include <stdint.h>
#include <utility>
#include <vector>

// Params for tests
template<class InputType,
         class OutputType         = InputType,
         class FlagType           = unsigned int,
         bool UseIdentityIterator = false,
         bool UseGraphs           = false>
struct DeviceSelectParams
{
    using input_type                            = InputType;
    using output_type                           = OutputType;
    using flag_type                             = FlagType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr bool use_graphs            = UseGraphs;
};

template<class Params>
class RocprimDeviceSelectTests : public ::testing::Test
{
public:
    using input_type                            = typename Params::input_type;
    using output_type                           = typename Params::output_type;
    using flag_type                             = typename Params::flag_type;
    const bool            debug_synchronous     = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    static constexpr bool use_graphs            = Params::use_graphs;
};

using RocprimDeviceSelectTestsParams
    = ::testing::Types<DeviceSelectParams<int, long>,
                       DeviceSelectParams<int8_t, int8_t>,
                       DeviceSelectParams<uint8_t, uint8_t>,
                       DeviceSelectParams<rocprim::half, rocprim::half>,
                       DeviceSelectParams<rocprim::bfloat16, rocprim::bfloat16>,
                       DeviceSelectParams<float, float>,
                       DeviceSelectParams<short, char, rocprim::int128_t>,
                       DeviceSelectParams<unsigned char, float, int, true>,
                       DeviceSelectParams<double, double, int, true>,
                       DeviceSelectParams<common::custom_type<double, double, true>,
                                          common::custom_type<double, double, true>,
                                          int,
                                          true>,
                       DeviceSelectParams<int, int, unsigned int, false, true>,
                       DeviceSelectParams<common::custom_huge_type<1024, int>,
                                          common::custom_huge_type<1024, int>,
                                          int>>;

TYPED_TEST_SUITE(RocprimDeviceSelectTests, RocprimDeviceSelectTestsParams);

TYPED_TEST(RocprimDeviceSelectTests, Flagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    using F                                     = typename TestFixture::flag_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    hipStream_t stream = 0; // default stream
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 100, seed_value);
            std::vector<F> flags = test_utils::get_random_data_wrapped<F>(size, 0, 1, seed_value);

            common::device_ptr<T>            d_input;
            common::device_ptr<F>            d_flags;
            common::device_ptr<U>            d_output;
            common::device_ptr<unsigned int> d_selected_count_output;

            if(!d_input.resize_with_memory_check(size) || !d_flags.resize_with_memory_check(size)
               || !d_output.resize_with_memory_check(size)
               || !d_selected_count_output.resize_with_memory_check(1))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            d_input.store(input);
            d_flags.store(flags);

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
            HIP_CHECK(rocprim::select(
                nullptr,
                temp_storage_size_bytes,
                d_input.get(),
                d_flags.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                d_selected_count_output.get(),
                input.size(),
                stream,
                TestFixture::debug_synchronous));

            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage;

            if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::select(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                d_input.get(),
                d_flags.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                d_selected_count_output.get(),
                input.size(),
                stream,
                TestFixture::debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            const auto selected_count_output = d_selected_count_output.load()[0];
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            const auto output = d_output.load();
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<class T>
struct select_op
{
    __device__ __host__
    inline bool
        operator()(const T& value) const
    {
        return rocprim::less<T>()(value, T(50));
    }
};

TYPED_TEST(RocprimDeviceSelectTests, SelectOp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 0, 100, seed_value);

            common::device_ptr<T>            d_input;
            common::device_ptr<U>            d_output;
            common::device_ptr<unsigned int> d_selected_count_output;

            if(!d_input.resize_with_memory_check(size) || !d_output.resize_with_memory_check(size)
               || !d_selected_count_output.resize_with_memory_check(1))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            d_input.store(input);

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
            HIP_CHECK(rocprim::select(
                nullptr,
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                d_selected_count_output.get(),
                input.size(),
                select_op<T>(),
                stream,
                debug_synchronous));

            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage;

            if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::select(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                d_selected_count_output.get(),
                input.size(),
                select_op<T>(),
                stream,
                debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            const auto selected_count_output = d_selected_count_output.load()[0];
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            const auto output = d_output.load();
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceSelectTests, SelectFlagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    using F                                     = typename TestFixture::flag_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;

    hipStream_t stream = 0; // default stream
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 100, seed_value);
            std::vector<F> flags = test_utils::get_random_data_wrapped<F>(size, 0, 1, seed_value);

            common::device_ptr<T>            d_input;
            common::device_ptr<F>            d_flags;
            common::device_ptr<U>            d_output;
            common::device_ptr<unsigned int> d_selected_count_output;

            if(!d_input.resize_with_memory_check(size) || !d_flags.resize_with_memory_check(size)
               || !d_output.resize_with_memory_check(size)
               || !d_selected_count_output.resize_with_memory_check(1))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            d_input.store(input);
            d_flags.store(flags);

            // Calculate expected results on host
            std::vector<U> expected;
            expected.reserve(input.size());
            for(size_t i = 0; i < input.size(); i++)
            {
                if(select_op<F>()(flags[i]) != 0)
                {
                    expected.push_back(input[i]);
                }
            }

            // temp storage
            size_t temp_storage_size_bytes;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::select(
                nullptr,
                temp_storage_size_bytes,
                d_input.get(),
                d_flags.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                d_selected_count_output.get(),
                input.size(),
                select_op<F>(),
                stream,
                TestFixture::debug_synchronous));

            HIP_CHECK(hipDeviceSynchronize());

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage;

            if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::select(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                d_input.get(),
                d_flags.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                d_selected_count_output.get(),
                input.size(),
                select_op<F>(),
                stream,
                TestFixture::debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if number of selected value is as expected
            const auto selected_count_output = d_selected_count_output.load()[0];
            ASSERT_EQ(selected_count_output, expected.size());

            // Check if output values are as expected
            const auto output = d_output.load();
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

std::vector<float> get_discontinuity_probabilities()
{
    std::vector<float> probabilities = {0.05, 0.25, 0.5, 0.75, 0.95, 1};
    return probabilities;
}

TYPED_TEST(RocprimDeviceSelectTests, Unique)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using op_type      = rocprim::equal_to<T>;
    using scan_op_type = rocprim::plus<T>;

    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const auto probabilities = get_discontinuity_probabilities();
        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<T> input(size);
                {
                    std::vector<T> input01 = test_utils::get_random_data01<T>(size, p, seed_value);
                    std::partial_sum(input01.begin(), input01.end(), input.begin(), scan_op_type());
                }

                // Allocate and copy to device
                common::device_ptr<T>            d_input;
                common::device_ptr<U>            d_output;
                common::device_ptr<unsigned int> d_selected_count_output;

                if(!d_input.resize_with_memory_check(size)
                   || !d_output.resize_with_memory_check(size)
                   || !d_selected_count_output.resize_with_memory_check(1))
                {
                    std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                    break;
                }

                d_input.store(input);

                // Calculate expected results on host
                std::vector<U> expected;
                expected.reserve(input.size());
                if(size > 0)
                {
                    expected.push_back(input[0]);
                    for(size_t i = 1; i < input.size(); i++)
                    {
                        if(!op_type()(input[i - 1], input[i]))
                        {
                            expected.push_back(input[i]);
                        }
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(rocprim::unique(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                    d_selected_count_output.get(),
                    input.size(),
                    op_type(),
                    stream,
                    debug_synchronous));

                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                common::device_ptr<void> d_temp_storage;

                if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
                {
                    std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                    break;
                }

                test_utils::GraphHelper gHelper;
                if(TestFixture::use_graphs)
                {
                    gHelper.startStreamCapture(stream);
                }

                // Run
                HIP_CHECK(rocprim::unique(
                    d_temp_storage.get(),
                    temp_storage_size_bytes,
                    d_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                    d_selected_count_output.get(),
                    input.size(),
                    op_type(),
                    stream,
                    debug_synchronous));

                if(TestFixture::use_graphs)
                {
                    gHelper.createAndLaunchGraph(stream, true, false);
                }

                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                const auto selected_count_output = d_selected_count_output.load()[0];
                ASSERT_EQ(selected_count_output, expected.size());

                // Check if output values are as expected
                const auto output = d_output.load();
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

                if(TestFixture::use_graphs)
                {
                    gHelper.cleanupGraphHelper();
                }
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

// The operator must be only called, when we have valid element in a block
template<class T, class F>
struct element_equal_operator
{
    F* data;
    element_equal_operator(F* _data)
    {
        this->data = _data;
    }

    __host__ __device__
    bool operator()(const T& index_a, const T& index_b) const
    {
        F lhs = data[index_a];
        F rhs = data[index_b];
        if(lhs != rhs)
        {
            return false;
        }
        return true;
    }
};

template<bool UseGraphs = false>
void testUniqueGuardedOperator()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = int64_t;
    using F                                     = int64_t;
    using U                                     = int64_t;
    using scan_op_type                          = rocprim::plus<T>;
    static constexpr bool use_identity_iterator = false;
    const bool            debug_synchronous     = false;

    hipStream_t stream = 0; // default stream
    if(UseGraphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const auto probabilities = get_discontinuity_probabilities();
        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<T> input
                    = test_utils::get_random_data_wrapped<T>(size, 0, size - 1, seed_value);

                std::vector<F> input_flag(size);
                {
                    std::vector<T> input01
                        = test_utils::get_random_data01<T>(size, p, seed_value + 1);
                    std::partial_sum(input01.begin(),
                                     input01.end(),
                                     input_flag.begin(),
                                     scan_op_type());
                }

                // Allocate and copy to device
                common::device_ptr<T>                d_input(input);
                common::device_ptr<F>                d_flag(input_flag);
                common::device_ptr<U>                d_output(input.size());
                common::device_ptr<unsigned int>     d_selected_count_output(1);
                element_equal_operator<F, T>         device_equal_op(d_flag.get());
                element_equal_operator<F, T>         host_equal_op(input_flag.data());

                // Calculate expected results on host
                std::vector<U> expected;
                expected.reserve(input.size());
                if(size > 0)
                {
                    expected.push_back(input[0]);
                    for(size_t i = 1; i < input.size(); i++)
                    {
                        if(!host_equal_op(input[i - 1], input[i]))
                        {
                            expected.push_back(input[i]);
                        }
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(rocprim::unique(
                    nullptr,
                    temp_storage_size_bytes,
                    d_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                    d_selected_count_output.get(),
                    input.size(),
                    device_equal_op,
                    stream,
                    debug_synchronous));

                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

                test_utils::GraphHelper gHelper;
                if(UseGraphs)
                {
                    gHelper.startStreamCapture(stream);
                }

                // Run
                HIP_CHECK(rocprim::unique(
                    d_temp_storage.get(),
                    temp_storage_size_bytes,
                    d_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                    d_selected_count_output.get(),
                    input.size(),
                    device_equal_op,
                    stream,
                    debug_synchronous));

                if(UseGraphs)
                {
                    gHelper.createAndLaunchGraph(stream, true, false);
                }

                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                const auto selected_count_output = d_selected_count_output.load()[0];
                ASSERT_EQ(selected_count_output, expected.size());

                // Check if output values are as expected
                const auto output = d_output.load();
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected, expected.size()));

                if(UseGraphs)
                {
                    gHelper.cleanupGraphHelper();
                }
            }
        }
    }

    if(UseGraphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceSelectTests, UniqueGuardedOperator)
{
    testUniqueGuardedOperator();
}

TEST(RocprimDeviceSelectTests, UniqueGuardedOperatorWithGraphs)
{
    testUniqueGuardedOperator<true>();
}

// Params for tests
template<typename KeyType,
         typename ValueType,
         typename OutputKeyType   = KeyType,
         typename OutputValueType = ValueType,
         bool UseIdentityIterator = false,
         bool UseGraphs           = false>
struct DeviceUniqueByKeyParams
{
    using key_type                              = KeyType;
    using value_type                            = ValueType;
    using output_key_type                       = OutputKeyType;
    using output_value_type                     = OutputValueType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr bool use_graphs            = UseGraphs;
};

template<class Params>
class RocprimDeviceUniqueByKeyTests : public ::testing::Test
{
public:
    using key_type                              = typename Params::key_type;
    using value_type                            = typename Params::value_type;
    using output_key_type                       = typename Params::output_key_type;
    using output_value_type                     = typename Params::output_value_type;
    const bool            debug_synchronous     = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    const bool            use_graphs            = Params::use_graphs;
};

using RocprimDeviceUniqueByKeyTestParams
    = ::testing::Types<DeviceUniqueByKeyParams<int, int>,
                       DeviceUniqueByKeyParams<double, double>,
                       DeviceUniqueByKeyParams<rocprim::half, uint8_t>,
                       DeviceUniqueByKeyParams<rocprim::bfloat16, uint8_t>,
                       DeviceUniqueByKeyParams<uint8_t, long long>,
                       DeviceUniqueByKeyParams<int, float, long, double>,
                       DeviceUniqueByKeyParams<long long, uint8_t, long, int, true>,
                       DeviceUniqueByKeyParams<common::custom_type<double, double, true>,
                                               common::custom_type<double, double, true>>,
                       DeviceUniqueByKeyParams<int, int, int, int, false, true>,
                       DeviceUniqueByKeyParams<common::custom_huge_type<1024, int>, uint8_t>>;

TYPED_TEST_SUITE(RocprimDeviceUniqueByKeyTests, RocprimDeviceUniqueByKeyTestParams);

TYPED_TEST(RocprimDeviceUniqueByKeyTests, UniqueByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type          = typename TestFixture::key_type;
    using value_type        = typename TestFixture::value_type;
    using output_key_type   = typename TestFixture::output_key_type;
    using output_value_type = typename TestFixture::output_value_type;

    using op_type = rocprim::equal_to<key_type>;

    using scan_op_type                          = rocprim::plus<key_type>;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const auto probabilities = get_discontinuity_probabilities();
        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<key_type> input_keys(size);
                {
                    std::vector<key_type> input01
                        = test_utils::get_random_data01<key_type>(size, p, seed_value);
                    std::partial_sum(input01.begin(),
                                     input01.end(),
                                     input_keys.begin(),
                                     scan_op_type());
                }
                const auto input_values
                    = test_utils::get_random_data_wrapped<value_type>(size,
                                                                      -1000,
                                                                      1000,
                                                                      seed_value);

                // Allocate and copy to device
                common::device_ptr<key_type>          d_keys_input;
                common::device_ptr<value_type>        d_values_input;
                common::device_ptr<output_key_type>   d_keys_output;
                common::device_ptr<output_value_type> d_values_output;
                common::device_ptr<unsigned int>      d_selected_count_output;

                if(!d_keys_input.resize_with_memory_check(size)
                   || !d_values_input.resize_with_memory_check(size)
                   || !d_keys_output.resize_with_memory_check(size)
                   || !d_values_output.resize_with_memory_check(size)
                   || !d_selected_count_output.resize_with_memory_check(1))
                {
                    std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                    break;
                }

                d_keys_input.store(input_keys);
                d_values_input.store(input_values);

                // Calculate expected results on host
                std::vector<output_key_type>   expected_keys;
                std::vector<output_value_type> expected_values;
                expected_keys.reserve(input_keys.size());
                expected_values.reserve(input_values.size());
                if(size > 0)
                {
                    expected_keys.push_back(input_keys[0]);
                    expected_values.push_back(input_values[0]);
                    for(size_t i = 1; i < input_keys.size(); i++)
                    {
                        if(!op_type()(input_keys[i - 1], input_keys[i]))
                        {
                            expected_keys.push_back(input_keys[i]);
                            expected_values.push_back(input_values[i]);
                        }
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(rocprim::unique_by_key(
                    nullptr,
                    temp_storage_size_bytes,
                    d_keys_input.get(),
                    d_values_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_keys_output.get()),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_values_output.get()),
                    d_selected_count_output.get(),
                    input_keys.size(),
                    op_type(),
                    stream,
                    debug_synchronous));

                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                common::device_ptr<void> d_temp_storage;

                if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
                {
                    std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                    break;
                }

                test_utils::GraphHelper gHelper;
                if(TestFixture::use_graphs)
                {
                    gHelper.startStreamCapture(stream);
                }

                // Run
                HIP_CHECK(rocprim::unique_by_key(
                    d_temp_storage.get(),
                    temp_storage_size_bytes,
                    d_keys_input.get(),
                    d_values_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_keys_output.get()),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_values_output.get()),
                    d_selected_count_output.get(),
                    input_keys.size(),
                    op_type(),
                    stream,
                    debug_synchronous));

                if(TestFixture::use_graphs)
                {
                    gHelper.createAndLaunchGraph(stream, true, false);
                }

                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                const auto selected_count_output = d_selected_count_output.load()[0];
                ASSERT_EQ(selected_count_output, expected_keys.size());

                // Check if outputs are as expected
                const auto output_keys   = d_keys_output.load();
                const auto output_values = d_values_output.load();
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(output_keys, expected_keys, expected_keys.size()));
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(output_values, expected_values, expected_values.size()));

                if(TestFixture::use_graphs)
                {
                    gHelper.cleanupGraphHelper();
                }
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceUniqueByKeyTests, UniqueByKeyAlias)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // This test checks correctness of in-place unique_by_key (so input keys and values iterators
    // are passed as output iterators as well)
    using key_type          = typename TestFixture::key_type;
    using value_type        = typename TestFixture::value_type;
    using output_key_type   = key_type;
    using output_value_type = value_type;

    using op_type = rocprim::equal_to<key_type>;

    using scan_op_type                          = rocprim::plus<key_type>;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;

    hipStream_t stream = 0; // default stream
    if(TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const auto probabilities = get_discontinuity_probabilities();
        for(auto size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            for(auto p : probabilities)
            {
                SCOPED_TRACE(testing::Message() << "with p = " << p);

                // Generate data
                std::vector<key_type> input_keys(size);
                {
                    std::vector<key_type> input01
                        = test_utils::get_random_data01<key_type>(size, p, seed_value);
                    std::partial_sum(input01.begin(),
                                     input01.end(),
                                     input_keys.begin(),
                                     scan_op_type());
                }
                const auto input_values
                    = test_utils::get_random_data_wrapped<value_type>(size,
                                                                      -1000,
                                                                      1000,
                                                                      seed_value);

                // Allocate and copy to device
                common::device_ptr<key_type>     d_keys_input;
                common::device_ptr<value_type>   d_values_input;
                common::device_ptr<unsigned int> d_selected_count_output;

                if(!d_keys_input.resize_with_memory_check(size)
                   || !d_values_input.resize_with_memory_check(size)
                   || !d_selected_count_output.resize_with_memory_check(1))
                {
                    std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                    break;
                }

                d_keys_input.store(input_keys);
                d_values_input.store(input_values);

                // Calculate expected results on host
                std::vector<output_key_type>   expected_keys;
                std::vector<output_value_type> expected_values;
                expected_keys.reserve(input_keys.size());
                expected_values.reserve(input_values.size());
                if(size > 0)
                {
                    expected_keys.push_back(input_keys[0]);
                    expected_values.push_back(input_values[0]);
                    for(size_t i = 1; i < input_keys.size(); i++)
                    {
                        if(!op_type()(input_keys[i - 1], input_keys[i]))
                        {
                            expected_keys.push_back(input_keys[i]);
                            expected_values.push_back(input_values[i]);
                        }
                    }
                }

                // temp storage
                size_t temp_storage_size_bytes;
                // Get size of d_temp_storage
                HIP_CHECK(rocprim::unique_by_key(
                    nullptr,
                    temp_storage_size_bytes,
                    d_keys_input.get(),
                    d_values_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_keys_input.get()),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_values_input.get()),
                    d_selected_count_output.get(),
                    input_keys.size(),
                    op_type(),
                    stream,
                    debug_synchronous));

                HIP_CHECK(hipDeviceSynchronize());

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                common::device_ptr<void> d_temp_storage;

                if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
                {
                    std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                    break;
                }

                test_utils::GraphHelper gHelper;
                if(TestFixture::use_graphs)
                {
                    gHelper.startStreamCapture(stream);
                }

                // Run
                HIP_CHECK(rocprim::unique_by_key(
                    d_temp_storage.get(),
                    temp_storage_size_bytes,
                    d_keys_input.get(),
                    d_values_input.get(),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_keys_input.get()),
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                        d_values_input.get()),
                    d_selected_count_output.get(),
                    input_keys.size(),
                    op_type(),
                    stream,
                    debug_synchronous));

                if(TestFixture::use_graphs)
                {
                    gHelper.createAndLaunchGraph(stream);
                }

                HIP_CHECK(hipDeviceSynchronize());

                // Check if number of selected value is as expected
                const auto selected_count_output = d_selected_count_output.load()[0];
                ASSERT_EQ(selected_count_output, expected_keys.size());

                // Check if outputs are as expected
                const auto output_keys   = d_keys_input.load();
                const auto output_values = d_values_input.load();
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(output_keys, expected_keys, expected_keys.size()));
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_eq(output_values, expected_values, expected_values.size()));

                if(TestFixture::use_graphs)
                {
                    gHelper.cleanupGraphHelper();
                }
            }
        }
    }

    if(TestFixture::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

class RocprimDeviceSelectLargeInputTests
    : public ::testing::TestWithParam<std::pair<unsigned int, bool>>
{
public:
    const bool debug_synchronous = false;
};

INSTANTIATE_TEST_SUITE_P(
    RocprimDeviceSelectLargeInputFlaggedTest,
    RocprimDeviceSelectLargeInputTests,
    ::testing::Values(std::make_pair(2048,
                                     false), // params: flag_selector/segment_length, use_graphs
                      std::make_pair(9643, false),
                      std::make_pair(32768, false),
                      std::make_pair(38713, false),
                      std::make_pair(38713, true)));

TEST_P(RocprimDeviceSelectLargeInputTests, LargeInputFlagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    auto         param         = GetParam();
    unsigned int flag_selector = std::get<0>(param);
    const bool   use_graphs    = std::get<1>(param);

    using InputIterator = typename rocprim::counting_iterator<size_t>;

    const bool debug_synchronous = RocprimDeviceSelectLargeInputTests::debug_synchronous;

    hipStream_t stream = 0; // default stream
    if(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(auto size : test_utils::get_large_sizes(0))
    {
        // otherwise test is too long
        if(size > (size_t{1} << 35))
            break;
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        size_t        initial_value = 0;
        InputIterator input_begin(initial_value);

        auto flags_it = rocprim::make_transform_iterator(rocprim::make_counting_iterator(size_t(0)),
                                                         [flag_selector](size_t i)
                                                         {
                                                             if(i % flag_selector == 0)
                                                                 return 1;
                                                             else
                                                                 return 0;
                                                         });

        common::device_ptr<size_t> d_selected_count_output(1);

        size_t expected_output_size = rocprim::detail::ceiling_div(size, flag_selector);

        common::device_ptr<size_t> d_output(expected_output_size);

        // Calculate expected results on host
        std::vector<size_t> expected_output(expected_output_size);
        for(size_t i = 0; i < expected_output_size; i++)
        {
            expected_output[i] = input_begin[i * flag_selector];
        }

        // temp storage
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        HIP_CHECK(rocprim::select(nullptr,
                                  temp_storage_size_bytes,
                                  input_begin,
                                  flags_it,
                                  d_output.get(),
                                  d_selected_count_output.get(),
                                  size,
                                  stream,
                                  debug_synchronous));

        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);
        common::device_ptr<void>     d_temp_storage(temp_storage_size_bytes);
        test_utils::GraphHelper      gHelper;
        if(use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        // Run
        HIP_CHECK(rocprim::select(d_temp_storage.get(),
                                  temp_storage_size_bytes,
                                  input_begin,
                                  flags_it,
                                  d_output.get(),
                                  d_selected_count_output.get(),
                                  size,
                                  stream,
                                  debug_synchronous));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream, true, false);
        }

        HIP_CHECK(hipDeviceSynchronize());

        // Check if number of selected value is as expected
        const auto selected_count_output = d_selected_count_output.load()[0];
        ASSERT_EQ(selected_count_output, expected_output_size);

        // Check if output values are as expected
        const auto output = d_output.load();

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_eq(output, expected_output, expected_output.size()));

        if(use_graphs)
        {
            gHelper.cleanupGraphHelper();
        }
    }

    if(use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<class T>
struct large_select_op
{
    T max_value;
    __device__ __host__
    inline bool
        operator()(const T& value) const
    {
        return rocprim::less<T>()(value, T(max_value));
    }
};

TEST_P(RocprimDeviceSelectLargeInputTests, LargeInputSelectOp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    auto       param      = GetParam();
    const bool use_graphs = std::get<1>(param);

    const bool debug_synchronous = RocprimDeviceSelectLargeInputTests::debug_synchronous;

    hipStream_t stream = 0; // default stream
    if(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(auto size : test_utils::get_large_sizes(0))
    {
        const size_t selected_input = std::get<0>(param);
        auto         select_op      = large_select_op<size_t>{selected_input};

        // otherwise test is too long
        if(size > (size_t{1} << 35))
            break;
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        auto input_iota = rocprim::make_counting_iterator(std::size_t{0});

        common::device_ptr<size_t> d_selected_count_output(1);

        size_t expected_output_size = selected_input;

        common::device_ptr<size_t> d_output(expected_output_size);

        // Calculate expected results on host
        std::vector<size_t> expected_output(expected_output_size);
        std::iota(expected_output.begin(), expected_output.end(), 0);

        // temp storage
        size_t temp_storage_size_bytes;

        // Get size of d_temp_storage
        HIP_CHECK(rocprim::select(nullptr,
                                  temp_storage_size_bytes,
                                  input_iota,
                                  d_output.get(),
                                  d_selected_count_output.get(),
                                  size,
                                  select_op,
                                  stream,
                                  debug_synchronous));

        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        test_utils::GraphHelper gHelper;
        if(use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        // Run
        HIP_CHECK(rocprim::select(d_temp_storage.get(),
                                  temp_storage_size_bytes,
                                  input_iota,
                                  d_output.get(),
                                  d_selected_count_output.get(),
                                  size,
                                  select_op,
                                  stream,
                                  debug_synchronous));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        HIP_CHECK(hipDeviceSynchronize());

        // Check if number of selected value is as expected
        const auto selected_count_output = d_selected_count_output.load()[0];
        ASSERT_EQ(selected_count_output, expected_output_size);

        // Check if output values are as expected
        const auto output = d_output.load();

        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_eq(output, expected_output, expected_output.size()));

        if(use_graphs)
        {
            gHelper.cleanupGraphHelper();
        }
    }

    if(use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(RocprimDeviceSelectLargeInputTests, LargeInputSelectFlagged)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    auto       param      = GetParam();
    const bool use_graphs = std::get<1>(param);

    using InputIterator = typename rocprim::counting_iterator<size_t>;

    const bool debug_synchronous = RocprimDeviceSelectLargeInputTests::debug_synchronous;

    hipStream_t stream = 0; // default stream
    if(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(auto size : test_utils::get_large_sizes(0))
    {
        // otherwise test is too long
        if(size > (size_t{1} << 35))
            break;
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_t selected_flags = std::get<0>(param);
        auto         select_op      = large_select_op<size_t>{selected_flags};

        // Generate data
        size_t        initial_value = 0;
        InputIterator input_begin(initial_value);

        auto flags_it = rocprim::make_counting_iterator(size_t(0));

        common::device_ptr<size_t> d_selected_count_output(1);

        size_t expected_output_size = selected_flags;

        common::device_ptr<size_t> d_output(expected_output_size);

        // Calculate expected results on host
        std::vector<size_t> expected_output(expected_output_size);
        std::iota(expected_output.begin(), expected_output.end(), 0);

        // temp storage
        size_t temp_storage_size_bytes;

        // Get size of d_temp_storage
        HIP_CHECK(rocprim::select(nullptr,
                                  temp_storage_size_bytes,
                                  input_begin,
                                  flags_it,
                                  d_output.get(),
                                  d_selected_count_output.get(),
                                  size,
                                  select_op,
                                  stream,
                                  debug_synchronous));

        HIP_CHECK(hipDeviceSynchronize());

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        test_utils::GraphHelper gHelper;
        if(use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        // Run
        HIP_CHECK(rocprim::select(d_temp_storage.get(),
                                  temp_storage_size_bytes,
                                  input_begin,
                                  flags_it,
                                  d_output.get(),
                                  d_selected_count_output.get(),
                                  size,
                                  select_op,
                                  stream,
                                  debug_synchronous));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        HIP_CHECK(hipDeviceSynchronize());

        // Check if number of selected value is as expected
        const auto selected_count_output = d_selected_count_output.load()[0];
        ASSERT_EQ(selected_count_output, expected_output_size);

        // Check if output values are as expected
        const auto output = d_output.load();
        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_eq(output, expected_output, expected_output.size()));

        if(use_graphs)
        {
            gHelper.cleanupGraphHelper();
        }
    }

    if(use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(RocprimDeviceSelectLargeInputTests, LargeInputUnique)
{
    static constexpr bool debug_synchronous = false;

    auto               param          = GetParam();
    const unsigned int segment_length = std::get<0>(param);
    const bool         use_graphs     = std::get<1>(param);

    hipStream_t stream = 0;
    if(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    for(const auto size : test_utils::get_large_sizes(0))
    {
        // otherwise test is too long
        if(size > (size_t{1} << 35))
            break;
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        auto input_it = rocprim::make_transform_iterator(rocprim::make_counting_iterator(size_t(0)),
                                                         [segment_length](size_t i)
                                                         { return i / segment_length; });

        const size_t expected_output_size = rocprim::detail::ceiling_div(size, segment_length);
        std::vector<size_t> expected_output(expected_output_size);
        std::iota(expected_output.begin(), expected_output.end(), 0);

        common::device_ptr<size_t> d_output(expected_output_size);
        common::device_ptr<size_t> d_unique_count_output(1);

        size_t temp_storage_size_bytes{};

        HIP_CHECK(rocprim::unique(nullptr,
                                  temp_storage_size_bytes,
                                  input_it,
                                  d_output.get(),
                                  d_unique_count_output.get(),
                                  size,
                                  rocprim::equal_to<size_t>{},
                                  stream,
                                  debug_synchronous));

        ASSERT_GT(temp_storage_size_bytes, 0);
        common::device_ptr<void>     d_temp_storage(temp_storage_size_bytes);
        test_utils::GraphHelper      gHelper;
        if(use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        HIP_CHECK(rocprim::unique(d_temp_storage.get(),
                                  temp_storage_size_bytes,
                                  input_it,
                                  d_output.get(),
                                  d_unique_count_output.get(),
                                  size,
                                  rocprim::equal_to<size_t>{},
                                  stream,
                                  debug_synchronous));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        const auto unique_count_output = d_unique_count_output.load()[0];
        ASSERT_EQ(unique_count_output, expected_output_size);

        const auto output = d_output.load();
        ASSERT_NO_FATAL_FAILURE(
            test_utils::assert_eq(output, expected_output, expected_output.size()));

        if(use_graphs)
            gHelper.cleanupGraphHelper();
    }

    if(use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}
