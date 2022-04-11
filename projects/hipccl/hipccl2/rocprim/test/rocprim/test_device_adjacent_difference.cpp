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

#include "test_utils_types.hpp"

#include <rocprim/device/device_adjacent_difference.hpp>

#include <rocprim/detail/various.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <numeric>
#include <type_traits>

namespace
{
template <typename T, unsigned int SizeLimit>
struct size_limit_config
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(T), sizeof(int));

    using type
        = rocprim::adjacent_difference_config<256,
                                              ::rocprim::max(1u, 16u / item_scale),
                                              rocprim::block_load_method::block_load_transpose,
                                              rocprim::block_store_method::block_store_transpose,
                                              SizeLimit>;
};

template <typename T>
struct size_limit_config<T, ROCPRIM_GRID_SIZE_LIMIT>
{
    using type = rocprim::default_config;
};

template <typename T, unsigned int SizeLimit>
using size_limit_config_t = typename size_limit_config<T, SizeLimit>::type;

std::vector<size_t> get_sizes(int seed_value)
{
    std::vector<size_t> sizes = {1, 10, 53, 211, 1024, 2048, 5096, 34567, (1 << 17) - 1220};
    if(!test_common_utils::use_hmm())
    {
        sizes.push_back(0);
    }
    const std::vector<size_t> random_sizes
        = test_utils::get_random_data<size_t>(2, 1, 16384, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

std::vector<size_t> get_large_sizes(int seed_value)
{
    // clang-format off
    std::vector<size_t> sizes = {
        (size_t{1} << 32) - 1, size_t{1} << 32,
        (size_t{1} << 35) - 1, size_t{1} << 35,
    };
    // clang-format on
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(
        2, (size_t {1} << 30) + 1, (size_t {1} << 35) - 2, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

template <typename Config = rocprim::default_config,
          typename InputIt,
          typename OutputIt,
          typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::false_type /*in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference<Config>(
        temporary_storage, storage_size, input, output, std::forward<Args>(args)...);
}

template <typename Config = rocprim::default_config,
          typename InputIt,
          typename OutputIt,
          typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::false_type /*in_place*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right<Config>(
        temporary_storage, storage_size, input, output, std::forward<Args>(args)...);
}

template <typename Config = rocprim::default_config,
          typename InputIt,
          typename OutputIt,
          typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::true_type /*in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_inplace<Config>(
        temporary_storage, storage_size, input, std::forward<Args>(args)...);
}

template <typename Config = rocprim::default_config,
          typename InputIt,
          typename OutputIt,
          typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::true_type /*in_place*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::rocprim::adjacent_difference_right_inplace<Config>(
        temporary_storage, storage_size, input, std::forward<Args>(args)...);
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
template <class InputType,
          class OutputType                 = InputType,
          bool         Left                = true,
          bool         InPlace             = false,
          bool         UseIdentityIterator = false,
          unsigned int SizeLimit           = ROCPRIM_GRID_SIZE_LIMIT>
struct DeviceAdjacentDifferenceParams
{
    using input_type                              = InputType;
    using output_type                             = OutputType;
    static constexpr bool   left                  = Left;
    static constexpr bool   in_place              = InPlace;
    static constexpr bool   use_identity_iterator = UseIdentityIterator;
    static constexpr size_t size_limit            = SizeLimit;
};

template <class Params>
class RocprimDeviceAdjacentDifferenceTests : public ::testing::Test
{
public:
    using input_type                              = typename Params::input_type;
    using output_type                             = typename Params::output_type;
    static constexpr bool   left                  = Params::left;
    static constexpr bool   in_place              = Params::in_place;
    static constexpr bool   use_identity_iterator = Params::use_identity_iterator;
    static constexpr bool   debug_synchronous     = false;
    static constexpr size_t size_limit            = Params::size_limit;
};

using custom_double2     = test_utils::custom_test_type<double>;
using custom_int64_array = test_utils::custom_test_array_type<std::int64_t, 8>;

using RocprimDeviceAdjacentDifferenceTestsParams = ::testing::Types<
    DeviceAdjacentDifferenceParams<int>,
    DeviceAdjacentDifferenceParams<float, double, false>,
    DeviceAdjacentDifferenceParams<int8_t, int8_t, true, true>,
    DeviceAdjacentDifferenceParams<custom_double2, custom_double2, false, true>,
    DeviceAdjacentDifferenceParams<rocprim::bfloat16, float, true, false, false, 512>,
    DeviceAdjacentDifferenceParams<rocprim::half, rocprim::half, true, true, false, 2048>,
    DeviceAdjacentDifferenceParams<custom_int64_array,
                                   custom_int64_array,
                                   false,
                                   true,
                                   true,
                                   4096>>;

TYPED_TEST_SUITE(RocprimDeviceAdjacentDifferenceTests, RocprimDeviceAdjacentDifferenceTestsParams);

TYPED_TEST(RocprimDeviceAdjacentDifferenceTests, AdjacentDifference)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using output_type                           = typename TestFixture::output_type;
    static constexpr bool left                  = TestFixture::left;
    static constexpr bool in_place              = TestFixture::in_place;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    static constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    using Config                                = size_limit_config_t<T, TestFixture::size_limit>;

    SCOPED_TRACE(testing::Message() << "left = " << left << ", in_place = " << in_place);

    for(std::size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<std::size_t> sizes = get_sizes(seed_value);
        for(auto size : sizes)
        {
            static constexpr hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            const std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<output_type> output(input.size());

            T*           d_input;
            output_type* d_output = nullptr;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(input[0])));
            HIP_CHECK(hipMemcpy(
                d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

            if(!in_place)
            {
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output,
                                                             output.size() * sizeof(output[0])));
            }

            static constexpr auto left_tag     = rocprim::detail::bool_constant<left> {};
            static constexpr auto in_place_tag = rocprim::detail::bool_constant<in_place> {};

            // Calculate expected results on host
            const auto expected
                = get_expected_result<output_type>(input, rocprim::minus<> {}, left_tag);

            const auto output_it
                = test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output);

            // Allocate temporary storage
            std::size_t temp_storage_size;
            void*       d_temp_storage = nullptr;
            HIP_CHECK(dispatch_adjacent_difference<Config>(left_tag,
                                                           in_place_tag,
                                                           d_temp_storage,
                                                           temp_storage_size,
                                                           d_input,
                                                           output_it,
                                                           size,
                                                           rocprim::minus<> {},
                                                           stream,
                                                           debug_synchronous));

            ASSERT_GT(temp_storage_size, 0);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size));

            // Run
            HIP_CHECK(dispatch_adjacent_difference<Config>(left_tag,
                                                           in_place_tag,
                                                           d_temp_storage,
                                                           temp_storage_size,
                                                           d_input,
                                                           output_it,
                                                           size,
                                                           rocprim::minus<> {},
                                                           stream,
                                                           debug_synchronous));
            HIP_CHECK(hipGetLastError());

            // Copy output to host
            HIP_CHECK(
                hipMemcpy(output.data(),
                          in_place ? static_cast<void*>(d_input) : static_cast<void*>(d_output),
                          output.size() * sizeof(output[0]),
                          hipMemcpyDeviceToHost));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
                output, expected, test_utils::precision_threshold<output_type>::percentage));

            hipFree(d_input);
            if(!in_place)
            {
                hipFree(d_output);
            }
            hipFree(d_temp_storage);
        }
    }
}

// Params for tests
template <bool Left = true, bool InPlace = false>
struct DeviceAdjacentDifferenceLargeParams
{
    static constexpr bool left     = Left;
    static constexpr bool in_place = InPlace;
};

template <class Params>
class RocprimDeviceAdjacentDifferenceLargeTests : public ::testing::Test
{
public:
    static constexpr bool left              = Params::left;
    static constexpr bool in_place          = Params::in_place;
    static constexpr bool debug_synchronous = false;
};

template<class T>
struct discard_write
{
    T value;

    __device__ operator T() const
    {
        return value;
    }
    __device__ discard_write& operator=(T)
    {
        return *this;
    }
};

using RocprimDeviceAdjacentDifferenceLargeTestsParams
    = ::testing::Types<DeviceAdjacentDifferenceLargeParams<true, false>,
                       DeviceAdjacentDifferenceLargeParams<true, true>,
                       DeviceAdjacentDifferenceLargeParams<false, false>,
                       DeviceAdjacentDifferenceLargeParams<true, true>>;

TYPED_TEST_SUITE(RocprimDeviceAdjacentDifferenceLargeTests,
                 RocprimDeviceAdjacentDifferenceLargeTestsParams);

TYPED_TEST(RocprimDeviceAdjacentDifferenceLargeTests, LargeIndices)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                 = size_t;
    using OutputIterator                    = rocprim::discard_iterator;
    static constexpr bool left              = TestFixture::left;
    static constexpr bool in_place          = TestFixture::in_place;
    const bool            debug_synchronous = TestFixture::debug_synchronous;

    SCOPED_TRACE(testing::Message() << "left = " << left << ", in_place = " << in_place);

    static constexpr hipStream_t stream = 0; // default

    for(std::size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_large_sizes(seed_value);

        for(const auto size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const OutputIterator output;

            // A transform iterator that can be written to (because in-place adjacent diff writes
            // to the input).
            // The conversion to T is used by flag_expected to actually perform the test
            const auto input = rocprim::make_transform_iterator(
                rocprim::make_counting_iterator(T {0}), [] __device__(const T i) {
                    return discard_write<T> {i};
                });

            int  flags[2] = {0, 0};
            int* d_flags  = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flags, sizeof(flags)));
            HIP_CHECK(hipMemcpy(d_flags, flags, sizeof(flags), hipMemcpyHostToDevice));

            const T expected            = test_utils::get_random_value<T>(1, size - 2, seed_value);
            static constexpr auto limit = ROCPRIM_GRID_SIZE_LIMIT;

            const T expected_above_limit
                = size - 2 > limit ? test_utils::get_random_value<T>(limit, size - 2, seed_value)
                                   : size - 2;

            SCOPED_TRACE(testing::Message() << "expected = " << expected);
            SCOPED_TRACE(testing::Message() << "expected_above_limit = " << expected_above_limit);

            const auto flag_expected = [=] __device__(const T& minuend, const T& subtrahend) {
                if(left)
                {
                    if(minuend == expected && subtrahend == expected - 1)
                    {
                        ++d_flags[0];
                    }
                    if(minuend == expected_above_limit && subtrahend == expected_above_limit - 1)
                    {
                        ++d_flags[1];
                    }
                }
                else
                {
                    if(minuend == expected && subtrahend == expected + 1)
                    {
                        ++d_flags[0];
                    }
                    if(minuend == expected_above_limit && subtrahend == expected_above_limit + 1)
                    {
                        ++d_flags[1];
                    }
                }
                return 0;
            };

            static constexpr auto left_tag     = rocprim::detail::bool_constant<left> {};
            static constexpr auto in_place_tag = rocprim::detail::bool_constant<in_place> {};

            // Allocate temporary storage
            std::size_t temp_storage_size;
            void*       d_temp_storage = nullptr;
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   in_place_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   flag_expected,
                                                   stream,
                                                   debug_synchronous));

            ASSERT_GT(temp_storage_size, 0);

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size));

            // Run
            HIP_CHECK(dispatch_adjacent_difference(left_tag,
                                                   in_place_tag,
                                                   d_temp_storage,
                                                   temp_storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   flag_expected,
                                                   stream,
                                                   debug_synchronous));

            // Copy output to host
            HIP_CHECK(hipMemcpy(flags, d_flags, sizeof(flags), hipMemcpyDeviceToHost));

            ASSERT_EQ(flags[0], 1);
            ASSERT_EQ(flags[1], 1);

            hipFree(d_flags);
        }
    }
}
