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

#ifndef HIPCUB_TEST_HIPCUB_DEVICE_RADIX_SORT_HPP_
#define HIPCUB_TEST_HIPCUB_DEVICE_RADIX_SORT_HPP_

#include "common_test_header.hpp"

// hipcub API
#include <hipcub/device/device_radix_sort.hpp>
#include <hipcub/util_type.hpp>

#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#define HIP_CHECK_MEMORY(condition)                                                         \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error == hipErrorOutOfMemory)                                                    \
        {                                                                                   \
            std::cout << "Out of memory. Skipping size = " << size << std::endl;            \
            (void)hipGetLastError(); /*reset error code to hipSuccess*/                     \
            break;                                                                          \
        }                                                                                   \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ \
                      << std::endl;                                                         \
            exit(error);                                                                    \
        }                                                                                   \
    }

template<class Key,
         class Value,
         bool         Descending      = false,
         unsigned int StartBit        = 0,
         unsigned int EndBit          = sizeof(Key) * 8,
         bool         CheckLargeSizes = false,
         bool         UseGraphs       = false>
struct params
{
    using key_type                                  = Key;
    using value_type                                = Value;
    static constexpr bool         descending        = Descending;
    static constexpr unsigned int start_bit         = StartBit;
    static constexpr unsigned int end_bit           = EndBit;
    static constexpr bool         check_large_sizes = CheckLargeSizes;
    static constexpr bool         use_graphs        = UseGraphs;
};

template<class Params>
class HipcubDeviceRadixSort : public ::testing::Test
{
public:
    using params = Params;
};

TYPED_TEST_SUITE_P(HipcubDeviceRadixSort);

template<class T>
auto generate_key_input(size_t size, unsigned int seed_value) HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    ->std::enable_if_t<hipcub::NumericTraits<T>::CATEGORY == hipcub::FLOATING_POINT,
                       std::vector<T>> HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
{
    auto result = test_utils::get_random_data<T>(size,
                                                 test_utils::numeric_limits<T>::min(),
                                                 test_utils::numeric_limits<T>::max(),
                                                 seed_value);
    test_utils::add_special_values(result, seed_value);
    return result;
}

template<class T>
auto generate_key_input(size_t size, unsigned int seed_value) HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    ->std::enable_if_t<hipcub::NumericTraits<T>::CATEGORY != hipcub::FLOATING_POINT,
                       std::vector<T>> HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
{
    using inner_t = typename test_utils::inner_type<T>::type;
    return test_utils::get_random_data<T>(size,
                                          test_utils::numeric_limits<inner_t>::min(),
                                          test_utils::numeric_limits<inner_t>::max(),
                                          seed_value);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temporary_storage,
                      size_t&     temporary_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      int         start_bit,
                      int         end_bit,
                      hipStream_t stream)
    -> std::enable_if_t<!Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeys(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             size,
                                             start_bit,
                                             end_bit,
                                             stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temporary_storage,
                      size_t&     temporary_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      int         start_bit,
                      int         end_bit,
                      hipStream_t stream)
    -> std::enable_if_t<Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeysDescending(d_temporary_storage,
                                                       temporary_storage_bytes,
                                                       d_keys_input,
                                                       d_keys_output,
                                                       size,
                                                       start_bit,
                                                       end_bit,
                                                       stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temporary_storage,
                      size_t&     temporary_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      int         start_bit,
                      int         end_bit,
                      hipStream_t stream)
    -> std::enable_if_t<!Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortKeys(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input,
                                                 d_keys_output,
                                                 size,
                                                 decomposer_t{},
                                                 stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortKeys(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input,
                                                 d_keys_output,
                                                 size,
                                                 decomposer_t{},
                                                 start_bit,
                                                 end_bit,
                                                 stream);
    }
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*       d_temporary_storage,
                      size_t&     temporary_storage_bytes,
                      Key*        d_keys_input,
                      Key*        d_keys_output,
                      size_t      size,
                      int         start_bit,
                      int         end_bit,
                      hipStream_t stream)
    -> std::enable_if_t<Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortKeysDescending(d_temporary_storage,
                                                           temporary_storage_bytes,
                                                           d_keys_input,
                                                           d_keys_output,
                                                           size,
                                                           decomposer_t{},
                                                           stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortKeysDescending(d_temporary_storage,
                                                           temporary_storage_bytes,
                                                           d_keys_input,
                                                           d_keys_output,
                                                           size,
                                                           decomposer_t{},
                                                           start_bit,
                                                           end_bit,
                                                           stream);
    }
}

template<typename TestFixture>
void sort_keys()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
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
            if(size > (1 << 20) && !check_large_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            const std::vector<key_type> keys_input = generate_key_input<key_type>(size, seed_value);

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            std::stable_sort(
                expected.begin(),
                expected.end(),
                test_utils::key_comparator<key_type, descending, start_bit, end_bit>());

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(invoke_sort_keys<descending>(nullptr,
                                                   temporary_storage_bytes,
                                                   d_keys_input,
                                                   d_keys_output,
                                                   size,
                                                   start_bit,
                                                   end_bit,
                                                   stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(invoke_sort_keys<descending>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_keys_input,
                                                   d_keys_output,
                                                   size,
                                                   start_bit,
                                                   end_bit,
                                                   stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_output));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output, expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temporary_storage,
                       size_t&     temporary_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      size,
                       int         start_bit,
                       int         end_bit,
                       hipStream_t stream)
    -> std::enable_if_t<!Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairs(d_temporary_storage,
                                              temporary_storage_bytes,
                                              d_keys_input,
                                              d_keys_output,
                                              d_values_input,
                                              d_values_output,
                                              size,
                                              start_bit,
                                              end_bit,
                                              stream);
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temporary_storage,
                       size_t&     temporary_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      size,
                       int         start_bit,
                       int         end_bit,
                       hipStream_t stream)
    -> std::enable_if_t<Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairsDescending(d_temporary_storage,
                                                        temporary_storage_bytes,
                                                        d_keys_input,
                                                        d_keys_output,
                                                        d_values_input,
                                                        d_values_output,
                                                        size,
                                                        start_bit,
                                                        end_bit,
                                                        stream);
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temporary_storage,
                       size_t&     temporary_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      size,
                       int         start_bit,
                       int         end_bit,
                       hipStream_t stream)
    -> std::enable_if_t<!Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortPairs(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys_input,
                                                  d_keys_output,
                                                  d_values_input,
                                                  d_values_output,
                                                  size,
                                                  decomposer_t{},
                                                  stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortPairs(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys_input,
                                                  d_keys_output,
                                                  d_values_input,
                                                  d_values_output,
                                                  size,
                                                  decomposer_t{},
                                                  start_bit,
                                                  end_bit,
                                                  stream);
    }
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*       d_temporary_storage,
                       size_t&     temporary_storage_bytes,
                       Key*        d_keys_input,
                       Key*        d_keys_output,
                       Value*      d_values_input,
                       Value*      d_values_output,
                       size_t      size,
                       int         start_bit,
                       int         end_bit,
                       hipStream_t stream)
    -> std::enable_if_t<Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortPairsDescending(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_keys_input,
                                                            d_keys_output,
                                                            d_values_input,
                                                            d_values_output,
                                                            size,
                                                            decomposer_t{},
                                                            stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortPairsDescending(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_keys_input,
                                                            d_keys_output,
                                                            d_values_input,
                                                            d_values_output,
                                                            size,
                                                            decomposer_t{},
                                                            start_bit,
                                                            end_bit,
                                                            stream);
    }
}

template<typename TestFixture>
void sort_pairs()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    using value_type                         = typename TestFixture::params::value_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
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
            if(size > (1 << 20) && !check_large_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            const std::vector<key_type> keys_input = generate_key_input<key_type>(size, seed_value);
            std::vector<value_type>     values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            value_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            using key_value = std::pair<key_type, value_type>;

            // Calculate expected results on host
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            std::stable_sort(
                expected.begin(),
                expected.end(),
                test_utils::
                    key_value_comparator<key_type, value_type, descending, start_bit, end_bit>());

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(invoke_sort_pairs<descending>(d_temporary_storage,
                                                    temporary_storage_bytes,
                                                    d_keys_input,
                                                    d_keys_output,
                                                    d_values_input,
                                                    d_values_output,
                                                    size,
                                                    start_bit,
                                                    end_bit,
                                                    stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(invoke_sort_pairs<descending>(d_temporary_storage,
                                                    temporary_storage_bytes,
                                                    d_keys_input,
                                                    d_keys_output,
                                                    d_values_input,
                                                    d_values_output,
                                                    size,
                                                    start_bit,
                                                    end_bit,
                                                    stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_values_input));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys_output,
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values_output,
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_output));

            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(values_output, values_expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*                      d_temporary_storage,
                      size_t&                    temporary_storage_bytes,
                      hipcub::DoubleBuffer<Key>& d_keys,
                      size_t                     size,
                      int                        start_bit,
                      int                        end_bit,
                      hipStream_t                stream)
    -> std::enable_if_t<!Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeys(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_keys,
                                             size,
                                             start_bit,
                                             end_bit,
                                             stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*                      d_temporary_storage,
                      size_t&                    temporary_storage_bytes,
                      hipcub::DoubleBuffer<Key>& d_keys,
                      size_t                     size,
                      int                        start_bit,
                      int                        end_bit,
                      hipStream_t                stream)
    -> std::enable_if_t<Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortKeysDescending(d_temporary_storage,
                                                       temporary_storage_bytes,
                                                       d_keys,
                                                       size,
                                                       start_bit,
                                                       end_bit,
                                                       stream);
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*                      d_temporary_storage,
                      size_t&                    temporary_storage_bytes,
                      hipcub::DoubleBuffer<Key>& d_keys,
                      size_t                     size,
                      int                        start_bit,
                      int                        end_bit,
                      hipStream_t                stream)
    -> std::enable_if_t<!Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortKeys(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys,
                                                 size,
                                                 decomposer_t{},
                                                 stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortKeys(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys,
                                                 size,
                                                 decomposer_t{},
                                                 start_bit,
                                                 end_bit,
                                                 stream);
    }
}

template<bool Descending, class Key>
auto invoke_sort_keys(void*                      d_temporary_storage,
                      size_t&                    temporary_storage_bytes,
                      hipcub::DoubleBuffer<Key>& d_keys,
                      size_t                     size,
                      int                        start_bit,
                      int                        end_bit,
                      hipStream_t                stream)
    -> std::enable_if_t<Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortKeysDescending(d_temporary_storage,
                                                           temporary_storage_bytes,
                                                           d_keys,
                                                           size,
                                                           decomposer_t{},
                                                           stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortKeysDescending(d_temporary_storage,
                                                           temporary_storage_bytes,
                                                           d_keys,
                                                           size,
                                                           decomposer_t{},
                                                           start_bit,
                                                           end_bit,
                                                           stream);
    }
}

template<typename TestFixture>
void sort_keys_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
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
            if(size > (1 << 20) && !check_large_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            const std::vector<key_type> keys_input = generate_key_input<key_type>(size, seed_value);
            key_type*                   d_keys_input;
            key_type*                   d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            std::stable_sort(
                expected.begin(),
                expected.end(),
                test_utils::key_comparator<key_type, descending, start_bit, end_bit>());

            hipcub::DoubleBuffer<key_type> d_keys(d_keys_input, d_keys_output);

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(invoke_sort_keys<descending>(nullptr,
                                                   temporary_storage_bytes,
                                                   d_keys,
                                                   size,
                                                   start_bit,
                                                   end_bit,
                                                   stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            void* d_temporary_storage;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(invoke_sort_keys<descending>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_keys,
                                                   size,
                                                   start_bit,
                                                   end_bit,
                                                   stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys.Current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output, expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                        d_temporary_storage,
                       size_t&                      temporary_storage_bytes,
                       hipcub::DoubleBuffer<Key>&   d_keys,
                       hipcub::DoubleBuffer<Value>& d_values,
                       size_t                       size,
                       int                          start_bit,
                       int                          end_bit,
                       hipStream_t                  stream)
    -> std::enable_if_t<!Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairs(d_temporary_storage,
                                              temporary_storage_bytes,
                                              d_keys,
                                              d_values,
                                              size,
                                              start_bit,
                                              end_bit,
                                              stream);
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                        d_temporary_storage,
                       size_t&                      temporary_storage_bytes,
                       hipcub::DoubleBuffer<Key>&   d_keys,
                       hipcub::DoubleBuffer<Value>& d_values,
                       size_t                       size,
                       int                          start_bit,
                       int                          end_bit,
                       hipStream_t                  stream)
    -> std::enable_if_t<Descending && !test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    return hipcub::DeviceRadixSort::SortPairsDescending(d_temporary_storage,
                                                        temporary_storage_bytes,
                                                        d_keys,
                                                        d_values,
                                                        size,
                                                        start_bit,
                                                        end_bit,
                                                        stream);
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                        d_temporary_storage,
                       size_t&                      temporary_storage_bytes,
                       hipcub::DoubleBuffer<Key>&   d_keys,
                       hipcub::DoubleBuffer<Value>& d_values,
                       size_t                       size,
                       int                          start_bit,
                       int                          end_bit,
                       hipStream_t                  stream)
    -> std::enable_if_t<!Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortPairs(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys,
                                                  d_values,
                                                  size,
                                                  decomposer_t{},
                                                  stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortPairs(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys,
                                                  d_values,
                                                  size,
                                                  decomposer_t{},
                                                  start_bit,
                                                  end_bit,
                                                  stream);
    }
}

template<bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                        d_temporary_storage,
                       size_t&                      temporary_storage_bytes,
                       hipcub::DoubleBuffer<Key>&   d_keys,
                       hipcub::DoubleBuffer<Value>& d_values,
                       size_t                       size,
                       int                          start_bit,
                       int                          end_bit,
                       hipStream_t                  stream)
    -> std::enable_if_t<Descending && test_utils::is_custom_test_type<Key>::value, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == sizeof(Key) * 8)
    {
        return hipcub::DeviceRadixSort::SortPairsDescending(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_keys,
                                                            d_values,
                                                            size,
                                                            decomposer_t{},
                                                            stream);
    }
    else
    {
        return hipcub::DeviceRadixSort::SortPairsDescending(d_temporary_storage,
                                                            temporary_storage_bytes,
                                                            d_keys,
                                                            d_values,
                                                            size,
                                                            decomposer_t{},
                                                            start_bit,
                                                            end_bit,
                                                            stream);
    }
}

template<typename TestFixture>
void sort_pairs_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    using value_type                         = typename TestFixture::params::value_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0; // default
    if(TestFixture::params::use_graphs)
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
            if(size > (1 << 20) && !check_large_sizes)
            {
                continue;
            }
            SCOPED_TRACE(testing::Message() << "with size= " << size);
            // Generate data
            const std::vector<key_type> keys_input = generate_key_input<key_type>(size, seed_value);
            std::vector<value_type>     values_input(size);
            std::iota(values_input.begin(), values_input.end(), 0);

            key_type* d_keys_input;
            key_type* d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(hipMemcpy(d_keys_input,
                                keys_input.data(),
                                size * sizeof(key_type),
                                hipMemcpyHostToDevice));

            value_type* d_values_input;
            value_type* d_values_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_input, size * sizeof(value_type)));
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_values_output, size * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                size * sizeof(value_type),
                                hipMemcpyHostToDevice));

            using key_value = std::pair<key_type, value_type>;

            // Calculate expected results on host
            std::vector<key_value> expected(size);
            for(size_t i = 0; i < size; i++)
            {
                expected[i] = key_value(keys_input[i], values_input[i]);
            }
            std::stable_sort(
                expected.begin(),
                expected.end(),
                test_utils::
                    key_value_comparator<key_type, value_type, descending, start_bit, end_bit>());

            hipcub::DoubleBuffer<key_type>   d_keys(d_keys_input, d_keys_output);
            hipcub::DoubleBuffer<value_type> d_values(d_values_input, d_values_output);

            void*  d_temporary_storage     = nullptr;
            size_t temporary_storage_bytes = 0;
            HIP_CHECK(invoke_sort_pairs<descending>(d_temporary_storage,
                                                    temporary_storage_bytes,
                                                    d_keys,
                                                    d_values,
                                                    size,
                                                    start_bit,
                                                    end_bit,
                                                    stream));

            ASSERT_GT(temporary_storage_bytes, 0U);

            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            test_utils::GraphHelper gHelper;
            if (TestFixture::params::use_graphs)
                gHelper.startStreamCapture(stream);

            HIP_CHECK(invoke_sort_pairs<descending>(d_temporary_storage,
                                                    temporary_storage_bytes,
                                                    d_keys,
                                                    d_values,
                                                    size,
                                                    start_bit,
                                                    end_bit,
                                                    stream));

            if (TestFixture::params::use_graphs)
                gHelper.createAndLaunchGraph(stream);

            HIP_CHECK(hipFree(d_temporary_storage));

            std::vector<key_type> keys_output(size);
            HIP_CHECK(hipMemcpy(keys_output.data(),
                                d_keys.Current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values.Current(),
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_values_output));

            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output, keys_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(values_output, values_expected));

            if(TestFixture::params::use_graphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(TestFixture::params::use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

inline void sort_keys_over_4g()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                                 = uint8_t;
    constexpr unsigned int start_bit               = 0;
    constexpr unsigned int end_bit                 = 8ull * sizeof(key_type);
    constexpr hipStream_t  stream                  = 0;
    constexpr size_t       size                    = (1ull << 32) + 32;
    constexpr size_t       number_of_possible_keys = 1ull << (8ull * sizeof(key_type));
    assert(std::is_unsigned<key_type>::value);

    hipDeviceProp_t dev_prop;
    HIP_CHECK(hipGetDeviceProperties(&dev_prop, device_id));

    // Radix sort requires 2 buffers of `size`, so a minimum of 8 GB of vram for this test.
    // This is more than some cards provide.
    if(static_cast<size_t>(dev_prop.totalGlobalMem * 0.9) < size * 2 * sizeof(key_type))
    {
        GTEST_SKIP() << "insufficient global memory";
    }

    std::vector<size_t> histogram(number_of_possible_keys, 0);
    const int           seed_value = rand();
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

    std::vector<key_type> keys_input
        = test_utils::get_random_data<key_type>(size,
                                                std::numeric_limits<key_type>::min(),
                                                std::numeric_limits<key_type>::max(),
                                                seed_value);

    //generate histogram of the randomly generated values
    std::for_each(keys_input.begin(), keys_input.end(), [&](const key_type& a) { histogram[a]++; });

    key_type* d_keys_input_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input_output, size * sizeof(key_type)));
    HIP_CHECK(hipMemcpy(d_keys_input_output,
                        keys_input.data(),
                        size * sizeof(key_type),
                        hipMemcpyHostToDevice));

    size_t temporary_storage_bytes;
    HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(nullptr,
                                                temporary_storage_bytes,
                                                d_keys_input_output,
                                                d_keys_input_output,
                                                size,
                                                start_bit,
                                                end_bit,
                                                stream));

    ASSERT_GT(temporary_storage_bytes, 0);
    void* d_temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

    HIP_CHECK(hipcub::DeviceRadixSort::SortKeys(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input_output,
                                                d_keys_input_output,
                                                size,
                                                start_bit,
                                                end_bit,
                                                stream));

    std::vector<key_type> output(keys_input.size());
    HIP_CHECK(hipMemcpy(output.data(),
                        d_keys_input_output,
                        size * sizeof(key_type),
                        hipMemcpyDeviceToHost));

    size_t counter = 0;
    for(size_t i = 0; i <= std::numeric_limits<key_type>::max(); ++i)
    {
        for(size_t j = 0; j < histogram[i]; ++j)
        {
            ASSERT_EQ(static_cast<size_t>(output[counter]), i);
            ++counter;
        }
    }
    ASSERT_EQ(counter, size);

    HIP_CHECK(hipFree(d_keys_input_output));
    HIP_CHECK(hipFree(d_temporary_storage));
}

inline void sort_keys_large_sizes()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = uint8_t;
    constexpr bool         descending = false;
    constexpr unsigned int start_bit  = 0;
    constexpr unsigned int end_bit    = 8;

    hipStream_t stream = 0;

    // Workaround: `hipMalloc` always returns `hipSuccess` even when allocation fails.
    // We limit the maximum size so this bug doesn't occur.
#ifdef _WIN32
    const std::vector<size_t> sizes = test_utils::get_large_sizes<34>(seeds[0]);
#else
    const std::vector<size_t> sizes = test_utils::get_large_sizes(seeds[0]);
#endif
    for(const size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_type> keys_input(size);
        std::iota(keys_input.begin(), keys_input.end(), 0);

        key_type* d_keys;
        HIP_CHECK_MEMORY(test_common_utils::hipMallocHelper(&d_keys, size * sizeof(key_type)));
        HIP_CHECK(
            hipMemcpy(d_keys, keys_input.data(), size * sizeof(key_type), hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_sort_keys<descending>(d_temporary_storage,
                                               temporary_storage_bytes,
                                               d_keys,
                                               d_keys,
                                               size,
                                               start_bit,
                                               end_bit,
                                               stream));

        ASSERT_GT(temporary_storage_bytes, 0U);

        HIP_CHECK_MEMORY(
            test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

        HIP_CHECK(invoke_sort_keys<descending>(d_temporary_storage,
                                               temporary_storage_bytes,
                                               d_keys,
                                               d_keys,
                                               size,
                                               start_bit,
                                               end_bit,
                                               stream));

        HIP_CHECK(hipFree(d_temporary_storage));

        std::vector<key_type> keys_output(size);
        HIP_CHECK(
            hipMemcpy(keys_output.data(), d_keys, size * sizeof(key_type), hipMemcpyDeviceToHost));

        HIP_CHECK(hipFree(d_keys));

        // Check if output values are as expected
        const size_t unique_keys    = size_t(std::numeric_limits<key_type>::max()) + 1;
        const size_t segment_length = test_utils::ceiling_div(size, unique_keys);
        const size_t full_segments  = size % unique_keys == 0 ? unique_keys : size % unique_keys;
        for(size_t i = 0; i < size; i += 4321)
        {
            key_type expected;
            if(i / segment_length < full_segments)
            {
                expected = key_type(i / segment_length);
            }
            else
            {
                expected = key_type((i - full_segments * segment_length) / (segment_length - 1)
                                    + full_segments);
            }
            ASSERT_EQ(keys_output[i], expected) << "with index = " << i;
        }
    }
}

#endif // HIPCUB_TEST_HIPCUB_DEVICE_RADIX_SORT_HPP_
