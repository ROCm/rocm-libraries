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

#include "../../common/device_adjacent_difference.hpp"
#include "../../common/utils_custom_type.hpp"

#include "../../common/utils_device_ptr.hpp"
#include "indirect_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/intrinsics/atomic.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

std::string to_string(common::api_variant aliasing)
{
    switch(aliasing)
    {
        case common::api_variant::no_alias: return "no_alias";
        case common::api_variant::alias: return "alias";
        case common::api_variant::in_place: return "in_place";
    }
}

template <typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction  op,
                         std::true_type /*left*/)
{
    std::vector<Output> result(input.size());
    std::adjacent_difference(input.cbegin(), input.cend(), result.begin(), op);
    return result;
}

template <typename Output, typename T, typename BinaryFunction>
auto get_expected_result(const std::vector<T>& input,
                         const BinaryFunction  op,
                         std::false_type /*left*/)
{
    std::vector<Output> result(input.size());
    // "right" adjacent difference is just adjacent difference backwards
    std::adjacent_difference(input.crbegin(), input.crend(), result.rbegin(), op);
    return result;
}
} // namespace

// Params for tests
template<class InputType,
         class OutputType                        = InputType,
         bool                Left                = true,
         common::api_variant Aliasing            = common::api_variant::no_alias,
         bool                UseIdentityIterator = false,
         class Config                            = rocprim::default_config,
         bool UseGraphs                          = false,
         bool UseIndirectIterator                = false>
struct DeviceAdjacentDifferenceParams
{
    using input_type                                           = InputType;
    using output_type                                          = OutputType;
    static constexpr bool                left                  = Left;
    static constexpr common::api_variant aliasing              = Aliasing;
    static constexpr bool                use_identity_iterator = UseIdentityIterator;
    using config                                               = Config;
    static constexpr bool use_graphs                           = UseGraphs;
    static constexpr bool use_indirect_iterator                = UseIndirectIterator;
};

template <class Params>
class RocprimDeviceAdjacentDifferenceTests : public ::testing::Test
{
public:
    using input_type                                           = typename Params::input_type;
    using output_type                                          = typename Params::output_type;
    static constexpr bool                left                  = Params::left;
    static constexpr common::api_variant aliasing              = Params::aliasing;
    static constexpr bool                use_identity_iterator = Params::use_identity_iterator;
    static constexpr bool                use_indirect_iterator = Params::use_indirect_iterator;
    static constexpr bool                debug_synchronous     = false;
    using config                                               = typename Params::config;
    static constexpr bool use_graphs                           = Params::use_graphs;
};

using custom_double2     = common::custom_type<double, double, true>;
using custom_int64_array = test_utils::custom_test_array_type<std::int64_t, 8>;

using custom_config_0 = rocprim::adjacent_difference_config<128, 4>;

template<int SizeLimit>
using custom_size_limit_config
    = rocprim::adjacent_difference_config<1024,
                                          2,
                                          rocprim::block_load_method::block_load_transpose,
                                          rocprim::block_store_method::block_store_transpose,
                                          SizeLimit>;

using RocprimDeviceAdjacentDifferenceTestsParams = ::testing::Types<
    // Tests with default configuration
    DeviceAdjacentDifferenceParams<int>,
    DeviceAdjacentDifferenceParams<double>,
    DeviceAdjacentDifferenceParams<float>,
    DeviceAdjacentDifferenceParams<rocprim::bfloat16>,
    DeviceAdjacentDifferenceParams<float, double, false>,
    DeviceAdjacentDifferenceParams<int8_t, int8_t, true, common::api_variant::in_place>,
    DeviceAdjacentDifferenceParams<custom_double2,
                                   custom_double2,
                                   false,
                                   common::api_variant::in_place>,
    DeviceAdjacentDifferenceParams<rocprim::bfloat16, float, true, common::api_variant::no_alias>,
    DeviceAdjacentDifferenceParams<rocprim::half,
                                   rocprim::half,
                                   true,
                                   common::api_variant::alias,
                                   false>,
    DeviceAdjacentDifferenceParams<rocprim::half,
                                   rocprim::half,
                                   true,
                                   common::api_variant::in_place,
                                   false>,
    DeviceAdjacentDifferenceParams<custom_int64_array,
                                   custom_int64_array,
                                   false,
                                   common::api_variant::alias,
                                   true>,
    // Tests for void value_type
    DeviceAdjacentDifferenceParams<float, float, true, common::api_variant::in_place, true>,
    DeviceAdjacentDifferenceParams<float, float, true, common::api_variant::no_alias, true>,
    DeviceAdjacentDifferenceParams<float, float, true, common::api_variant::alias, true>,
    // Tests for supported config structs
    DeviceAdjacentDifferenceParams<rocprim::bfloat16,
                                   float,
                                   true,
                                   common::api_variant::no_alias,
                                   false,
                                   custom_config_0>,
    DeviceAdjacentDifferenceParams<rocprim::bfloat16, float, true, common::api_variant::alias>,
    // Tests for different size_limits
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   common::api_variant::no_alias,
                                   false,
                                   custom_size_limit_config<64>>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   common::api_variant::no_alias,
                                   false,
                                   custom_size_limit_config<8192>,
                                   false,
                                   true>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   common::api_variant::no_alias,
                                   false,
                                   custom_size_limit_config<10240>,
                                   false,
                                   true>,
    DeviceAdjacentDifferenceParams<int,
                                   int,
                                   true,
                                   common::api_variant::no_alias,
                                   false,
                                   rocprim::default_config,
                                   true>>;

TYPED_TEST_SUITE(RocprimDeviceAdjacentDifferenceTests, RocprimDeviceAdjacentDifferenceTestsParams);

TYPED_TEST(RocprimDeviceAdjacentDifferenceTests, AdjacentDifference)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                                    = typename TestFixture::input_type;
    using output_type                                          = typename TestFixture::output_type;
    static constexpr bool                left                  = TestFixture::left;
    static constexpr common::api_variant aliasing              = TestFixture::aliasing;
    const bool                           debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool                use_identity_iterator = TestFixture::use_identity_iterator;
    using Config                                               = typename TestFixture::config;

    SCOPED_TRACE(testing::Message()
                 << "left = " << left << ", api_variant = " << to_string(aliasing));

    for(std::size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if (TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

            common::device_ptr<T> d_input(input);

            static constexpr auto left_tag = rocprim::detail::bool_constant<left>{};
            static constexpr auto alias_tag
                = std::integral_constant<common::api_variant, aliasing>{};

            auto input_it
                = test_utils::wrap_in_indirect_iterator<TestFixture::use_indirect_iterator>(
                    d_input.get());

            // Allocate temporary storage
            std::size_t temp_storage_size;
            HIP_CHECK(common::dispatch_adjacent_difference<Config>(left_tag,
                                                                   alias_tag,
                                                                   nullptr,
                                                                   temp_storage_size,
                                                                   input_it,
                                                                   (output_type*){nullptr},
                                                                   size,
                                                                   rocprim::minus<>{},
                                                                   stream,
                                                                   debug_synchronous));

            ASSERT_GT(temp_storage_size, 0);

            common::device_ptr<void> d_temp_storage(temp_storage_size);

            test_utils::GraphHelper gHelper;

            // We might call the API multiple times, with almost the same parameter
            // (in-place and out-of-place)
            // we should be able to use the same amount of temp storage for and get the same
            // results (maybe with different types) for both.
            auto run_and_verify = [&](const auto output_it, auto* d_output)
            {
                if(TestFixture::use_graphs)
                {
                    gHelper.startStreamCapture(stream);
                }

                // Run
                HIP_CHECK(common::dispatch_adjacent_difference<Config>(left_tag,
                                                                       alias_tag,
                                                                       d_temp_storage.get(),
                                                                       temp_storage_size,
                                                                       input_it,
                                                                       output_it,
                                                                       size,
                                                                       rocprim::minus<>{},
                                                                       stream,
                                                                       debug_synchronous));
                HIP_CHECK(hipGetLastError());

                if(TestFixture::use_graphs)
                {
                    gHelper.createAndLaunchGraph(stream);
                }

                // input_type for in-place, output_type for out of place
                using current_output_type = std::remove_reference_t<decltype(*d_output)>;

                // allocate memory for output
                std::vector<current_output_type> output(size);

                // Copy output to host
                HIP_CHECK(hipMemcpy(output.data(),
                                    d_output,
                                    output.size() * sizeof(output[0]),
                                    hipMemcpyDeviceToHost));

                // Calculate expected results on host
                const auto expected
                    = get_expected_result<current_output_type>(input, rocprim::minus<>{}, left_tag);

                // Check if output values are as expected
                test_utils::assert_near(
                    output,
                    expected,
                    std::max(test_utils::precision<T>, test_utils::precision<output_type>));

                if(TestFixture::use_graphs)
                {
                    gHelper.cleanupGraphHelper();
                }
            };

            // if common::api_variant is not in_place we should check the non aliased function call
            if(aliasing != common::api_variant::in_place)
            {
                common::device_ptr<output_type> d_output(size);

                const auto output_it
                    = test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get());

                ASSERT_NO_FATAL_FAILURE(run_and_verify(output_it, d_output.get()));
            }

            // if common::api_variant is not no_alias we should check the inplace function call
            if(aliasing != common::api_variant::no_alias)
            {
                ASSERT_NO_FATAL_FAILURE(run_and_verify(input_it, d_input.get()));
            }

            if(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

// Params for tests
template<bool                Left      = true,
         common::api_variant Aliasing  = common::api_variant::no_alias,
         bool                UseGraphs = false>
struct DeviceAdjacentDifferenceLargeParams
{
    static constexpr bool                left       = Left;
    static constexpr common::api_variant aliasing   = Aliasing;
    static constexpr bool                use_graphs = UseGraphs;
};

template <class Params>
class RocprimDeviceAdjacentDifferenceLargeTests : public ::testing::Test
{
public:
    static constexpr bool                left              = Params::left;
    static constexpr common::api_variant aliasing          = Params::aliasing;
    static constexpr bool                debug_synchronous = false;
    static constexpr bool                use_graphs        = Params::use_graphs;
};

template<unsigned int SamplingRate>
class check_output_iterator
{
public:
    using flag_type = unsigned int;

private:
    class check_output
    {
    public:
        __device__ check_output(flag_type* incorrect_flag, size_t current_index, size_t* counter)
            : current_index_(current_index), incorrect_flag_(incorrect_flag), counter_(counter)
        {}

        __device__ check_output& operator=(size_t value)
        {
            if(value != current_index_)
            {
                rocprim::detail::atomic_store(incorrect_flag_, 1);
            }
            if(current_index_ % SamplingRate == 0)
            {
                atomicAdd(counter_, 1);
            }
            return *this;
        }

    private:
        size_t     current_index_;
        flag_type* incorrect_flag_;
        size_t*    counter_;
    };

public:
    using value_type        = size_t;
    using reference         = check_output;
    using pointer           = check_output*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;

    __host__ __device__ check_output_iterator(flag_type* const incorrect_flag,
                                              size_t* const    counter)
        : current_index_(0), incorrect_flag_(incorrect_flag), counter_(counter)
    {}

    __device__ bool operator==(const check_output_iterator& rhs) const
    {
        return current_index_ == rhs.current_index_;
    }
    __device__ bool operator!=(const check_output_iterator& rhs) const
    {
        return !(*this == rhs);
    }
    __device__ reference operator*()
    {
        return reference(incorrect_flag_, current_index_, counter_);
    }
    __device__ reference operator[](const difference_type distance) const
    {
        return *(*this + distance);
    }
    __host__ __device__ check_output_iterator& operator+=(const difference_type rhs)
    {
        current_index_ += rhs;
        return *this;
    }
    __host__ __device__ check_output_iterator& operator-=(const difference_type rhs)
    {
        current_index_ -= rhs;
        return *this;
    }
    __host__ __device__ difference_type operator-(const check_output_iterator& rhs) const
    {
        return current_index_ - rhs.current_index_;
    }
    __host__ __device__ check_output_iterator operator+(const difference_type rhs) const
    {
        return check_output_iterator(*this) += rhs;
    }
    __host__ __device__ check_output_iterator operator-(const difference_type rhs) const
    {
        return check_output_iterator(*this) -= rhs;
    }
    __host__ __device__ check_output_iterator& operator++()
    {
        ++current_index_;
        return *this;
    }
    __host__ __device__ check_output_iterator& operator--()
    {
        --current_index_;
        return *this;
    }
    __host__ __device__ check_output_iterator operator++(int)
    {
        return ++check_output_iterator{*this};
    }
    __host__ __device__ check_output_iterator operator--(int)
    {
        return --check_output_iterator{*this};
    }

private:
    size_t     current_index_;
    flag_type* incorrect_flag_;
    size_t*    counter_;
};

using RocprimDeviceAdjacentDifferenceLargeTestsParams = ::testing::Types<
    DeviceAdjacentDifferenceLargeParams<true, common::api_variant::no_alias>,
    DeviceAdjacentDifferenceLargeParams<false, common::api_variant::no_alias>,
    DeviceAdjacentDifferenceLargeParams<false, common::api_variant::alias>,
    DeviceAdjacentDifferenceLargeParams<true, common::api_variant::no_alias, true>>;

TYPED_TEST_SUITE(RocprimDeviceAdjacentDifferenceLargeTests,
                 RocprimDeviceAdjacentDifferenceLargeTestsParams);

TYPED_TEST(RocprimDeviceAdjacentDifferenceLargeTests, LargeIndices)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();

    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                                = size_t;
    static constexpr bool                is_left           = TestFixture::left;
    static constexpr common::api_variant aliasing          = TestFixture::aliasing;
    const bool                           debug_synchronous = TestFixture::debug_synchronous;
    static constexpr unsigned int        sampling_rate     = 10000;
    using OutputIterator                                   = check_output_iterator<sampling_rate>;
    using flag_type                                        = OutputIterator::flag_type;

    SCOPED_TRACE(testing::Message()
                 << "is_left = " << is_left << ", api_variant = " << to_string(aliasing));

    hipStream_t stream = 0; // default
    if (TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(std::size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            common::device_ptr<flag_type> d_incorrect_flag(1);
            HIP_CHECK(hipMemset(d_incorrect_flag.get(),
                                0,
                                sizeof(typename decltype(d_incorrect_flag)::value_type)));

            common::device_ptr<size_t> d_counter(1);
            HIP_CHECK(
                hipMemset(d_counter.get(), 0, sizeof(typename decltype(d_counter)::value_type)));

            // Note: hipMemset runs asynchronously unless the pointer it's passed refers to pinned memory.
            // Wait for it to complete here before we use the device variables we just set.
            HIP_CHECK(hipDeviceSynchronize());

            OutputIterator output(d_incorrect_flag.get(), d_counter.get());

            const auto input = rocprim::make_counting_iterator(T{0});

            // Return the position where the adjacent difference is expected to be written out.
            // When called with consecutive values the left value is returned at the left-handed difference, and the right value otherwise.
            // The return value is coherent with the boundary values.
            const auto op = [](const auto& larger_value, const auto& smaller_value)
            { return (smaller_value + larger_value) / 2 + (is_left ? 1 : 0); };

            static constexpr auto left_tag = rocprim::detail::bool_constant<is_left>{};
            static constexpr auto aliasing_tag
                = std::integral_constant<common::api_variant, aliasing>{};

            // Allocate temporary storage
            std::size_t temp_storage_size;
            HIP_CHECK(common::dispatch_adjacent_difference(left_tag,
                                                           aliasing_tag,
                                                           nullptr,
                                                           temp_storage_size,
                                                           input,
                                                           output,
                                                           size,
                                                           op,
                                                           stream,
                                                           debug_synchronous));

            ASSERT_GT(temp_storage_size, 0);

            common::device_ptr<void> d_temp_storage(temp_storage_size);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Capture the memset in the graph so that relaunching will have expected result
            HIP_CHECK(hipMemsetAsync(d_incorrect_flag.get(),
                                     0,
                                     sizeof(typename decltype(d_incorrect_flag)::value_type),
                                     stream));
            HIP_CHECK(hipMemsetAsync(d_counter.get(),
                                     0,
                                     sizeof(typename decltype(d_counter)::value_type),
                                     stream));

            // Run
            HIP_CHECK(common::dispatch_adjacent_difference(left_tag,
                                                           aliasing_tag,
                                                           d_temp_storage.get(),
                                                           temp_storage_size,
                                                           input,
                                                           output,
                                                           size,
                                                           op,
                                                           stream,
                                                           debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            // Copy output to host
            const auto incorrect_flag = d_incorrect_flag.load()[0];
            const auto counter        = d_counter.load()[0];

            ASSERT_EQ(incorrect_flag, 0);
            ASSERT_EQ(counter, rocprim::detail::ceiling_div(size, sampling_rate));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if (TestFixture::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}
