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

#ifndef TEST_DEVICE_RADIX_SORT_HPP_
#define TEST_DEVICE_RADIX_SORT_HPP_

#include "../common_test_header.hpp"

// required rocprim headers
#include <rocprim/detail/various.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_merge_sort_config.hpp>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/device/device_radix_sort_config.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types/double_buffer.hpp>

#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"
#include "test_utils_sort_checker.hpp"
#include "test_utils_sort_comparator.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>

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
class RocprimDeviceRadixSort : public ::testing::Test
{
public:
    using params = Params;
};

TYPED_TEST_SUITE_P(RocprimDeviceRadixSort);

template<class KeyIter>
auto generate_key_input(KeyIter keys_input, size_t size, engine_type& rng_engine)
    -> std::enable_if_t<
        rocprim::is_floating_point<typename std::iterator_traits<KeyIter>::value_type>::value>
{
    using key_type = typename std::iterator_traits<KeyIter>::value_type;
    test_utils::generate_random_data_n(keys_input,
                                       size,
                                       static_cast<key_type>(-1000),
                                       static_cast<key_type>(+1000),
                                       rng_engine);
    test_utils::add_special_values(keys_input, size, rng_engine);
}

template<class KeyIter>
auto generate_key_input(KeyIter keys_input, size_t size, engine_type& rng_engine)
    -> std::enable_if_t<
        !rocprim::is_floating_point<typename std::iterator_traits<KeyIter>::value_type>::value>
{
    using key_type = typename std::iterator_traits<KeyIter>::value_type;
    test_utils::generate_random_data_n(keys_input,
                                       size,
                                       rocprim::numeric_limits<key_type>::min(),
                                       rocprim::numeric_limits<key_type>::max(),
                                       rng_engine);
}

// Working around custom_float_test_type, which is both a float and a common::custom_type
template<class T>
constexpr bool is_custom_not_float_test_type
    = common::is_custom_type<T>::value && !std::is_same<test_utils::custom_float_type, T>::value;

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*        d_temporary_storage,
                      size_t&      temporary_storage_bytes,
                      Key*         d_keys_input,
                      Key*         d_keys_output,
                      size_t       size,
                      unsigned int start_bit,
                      unsigned int end_bit,
                      hipStream_t  stream,
                      bool         debug_synchronous)
    -> std::enable_if_t<!Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                            temporary_storage_bytes,
                                            d_keys_input,
                                            d_keys_output,
                                            size,
                                            start_bit,
                                            end_bit,
                                            stream,
                                            debug_synchronous);
}

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*        d_temporary_storage,
                      size_t&      temporary_storage_bytes,
                      Key*         d_keys_input,
                      Key*         d_keys_output,
                      size_t       size,
                      unsigned int start_bit,
                      unsigned int end_bit,
                      hipStream_t  stream,
                      bool         debug_synchronous)
    -> std::enable_if_t<Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_keys_desc<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input,
                                                 d_keys_output,
                                                 size,
                                                 start_bit,
                                                 end_bit,
                                                 stream,
                                                 debug_synchronous);
}

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*        d_temporary_storage,
                      size_t&      temporary_storage_bytes,
                      Key*         d_keys_input,
                      Key*         d_keys_output,
                      size_t       size,
                      unsigned int start_bit,
                      unsigned int end_bit,
                      hipStream_t  stream,
                      bool         debug_synchronous)
    -> std::enable_if_t<!Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input,
                                                d_keys_output,
                                                size,
                                                decomposer_t{},
                                                stream,
                                                debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input,
                                                d_keys_output,
                                                size,
                                                decomposer_t{},
                                                start_bit,
                                                end_bit,
                                                stream,
                                                debug_synchronous);
    }
}

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*        d_temporary_storage,
                      size_t&      temporary_storage_bytes,
                      Key*         d_keys_input,
                      Key*         d_keys_output,
                      size_t       size,
                      unsigned int start_bit,
                      unsigned int end_bit,
                      hipStream_t  stream,
                      bool         debug_synchronous)
    -> std::enable_if_t<Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_keys_desc<Config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_keys_input,
                                                     d_keys_output,
                                                     size,
                                                     decomposer_t{},
                                                     stream,
                                                     debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_keys_desc<Config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_keys_input,
                                                     d_keys_output,
                                                     size,
                                                     decomposer_t{},
                                                     start_bit,
                                                     end_bit,
                                                     stream,
                                                     debug_synchronous);
    }
}

template<typename TestFixture>
void sort_keys()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0;
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const bool debug_synchronous = false;

    bool in_place = false;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        auto sizes = test_utils::get_sizes(seed_value);
        sizes.push_back(1 << 23);

        for(size_t size : sizes)
        {
            if(size > (1 << 17) && !check_large_sizes)
            {
                break;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            engine_type rng_engine(seed_value);
            in_place = !in_place;

            // Generate data
            auto keys_input = std::make_unique<key_type[]>(size);
            generate_key_input(keys_input.get(), size, rng_engine);

            common::device_ptr<key_type>  d_keys_input(keys_input, size);
            common::device_ptr<key_type>  d_keys_output_alloc;
            common::device_ptr<key_type>& d_keys_output
                = in_place ? d_keys_input : d_keys_output_alloc;
            d_keys_output_alloc.resize(in_place ? 0 : size);

            // Use arbitrary custom config to increase test coverage without making more test cases
            using config = rocprim::radix_sort_config<rocprim::default_config,
                                                      rocprim::default_config,
                                                      rocprim::default_config,
                                                      1024 * 512>;

            size_t temporary_storage_bytes;
            HIP_CHECK((invoke_sort_keys<config, descending>(nullptr,
                                                            temporary_storage_bytes,
                                                            d_keys_input.get(),
                                                            d_keys_output.get(),
                                                            size,
                                                            start_bit,
                                                            end_bit,
                                                            stream,
                                                            debug_synchronous)));
            ASSERT_GT(temporary_storage_bytes, 0);
            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK((invoke_sort_keys<config, descending>(d_temporary_storage.get(),
                                                            temporary_storage_bytes,
                                                            d_keys_input.get(),
                                                            d_keys_output.get(),
                                                            size,
                                                            start_bit,
                                                            end_bit,
                                                            stream,
                                                            debug_synchronous)));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            if(size > 4096)
            {
                bool is_sorted = test_utils::device_sort_check(
                    d_keys_output.get(),
                    size,
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>{},
                    stream,
                    debug_synchronous);
                if(!is_sorted)
                {
                    // Load output to host
                    const auto keys_output = d_keys_output.load_to_unique_ptr();

                    // Calculate expected results on host
                    std::vector<key_type> expected(keys_input.get(), keys_input.get() + size);
                    std::stable_sort(
                        expected.begin(),
                        expected.end(),
                        test_utils::key_comparator<key_type, descending, start_bit, end_bit>{});

                    ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output.get(),
                                                                      keys_output.get() + size,
                                                                      expected.begin(),
                                                                      expected.end()));
                    FAIL();
                }
            }
            else
            {
                // Load output to host
                const auto keys_output = d_keys_output.load_to_unique_ptr();

                // Calculate expected results on host
                std::vector<key_type> expected(keys_input.get(), keys_input.get() + size);
                std::stable_sort(
                    expected.begin(),
                    expected.end(),
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>{});

                ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output.get(),
                                                                  keys_output.get() + size,
                                                                  expected.begin(),
                                                                  expected.end()));
            }
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*        d_temporary_storage,
                       size_t&      temporary_storage_bytes,
                       Key*         d_keys_input,
                       Key*         d_keys_output,
                       Value*       d_values_input,
                       Value*       d_values_output,
                       size_t       size,
                       unsigned int start_bit,
                       unsigned int end_bit,
                       hipStream_t  stream,
                       bool         debug_synchronous)
    -> std::enable_if_t<!Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             d_values_input,
                                             d_values_output,
                                             size,
                                             start_bit,
                                             end_bit,
                                             stream,
                                             debug_synchronous);
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*        d_temporary_storage,
                       size_t&      temporary_storage_bytes,
                       Key*         d_keys_input,
                       Key*         d_keys_output,
                       Value*       d_values_input,
                       Value*       d_values_output,
                       size_t       size,
                       unsigned int start_bit,
                       unsigned int end_bit,
                       hipStream_t  stream,
                       bool         debug_synchronous)
    -> std::enable_if_t<Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_pairs_desc<Config>(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys_input,
                                                  d_keys_output,
                                                  d_values_input,
                                                  d_values_output,
                                                  size,
                                                  start_bit,
                                                  end_bit,
                                                  stream,
                                                  debug_synchronous);
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*        d_temporary_storage,
                       size_t&      temporary_storage_bytes,
                       Key*         d_keys_input,
                       Key*         d_keys_output,
                       Value*       d_values_input,
                       Value*       d_values_output,
                       size_t       size,
                       unsigned int start_bit,
                       unsigned int end_bit,
                       hipStream_t  stream,
                       bool         debug_synchronous)
    -> std::enable_if_t<!Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input,
                                                 d_keys_output,
                                                 d_values_input,
                                                 d_values_output,
                                                 size,
                                                 decomposer_t{},
                                                 stream,
                                                 debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input,
                                                 d_keys_output,
                                                 d_values_input,
                                                 d_values_output,
                                                 size,
                                                 decomposer_t{},
                                                 start_bit,
                                                 end_bit,
                                                 stream,
                                                 debug_synchronous);
    }
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*        d_temporary_storage,
                       size_t&      temporary_storage_bytes,
                       Key*         d_keys_input,
                       Key*         d_keys_output,
                       Value*       d_values_input,
                       Value*       d_values_output,
                       size_t       size,
                       unsigned int start_bit,
                       unsigned int end_bit,
                       hipStream_t  stream,
                       bool         debug_synchronous)
    -> std::enable_if_t<Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_pairs_desc<Config>(d_temporary_storage,
                                                      temporary_storage_bytes,
                                                      d_keys_input,
                                                      d_keys_output,
                                                      d_values_input,
                                                      d_values_output,
                                                      size,
                                                      decomposer_t{},
                                                      stream,
                                                      debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_pairs_desc<Config>(d_temporary_storage,
                                                      temporary_storage_bytes,
                                                      d_keys_input,
                                                      d_keys_output,
                                                      d_values_input,
                                                      d_values_output,
                                                      size,
                                                      decomposer_t{},
                                                      start_bit,
                                                      end_bit,
                                                      stream,
                                                      debug_synchronous);
    }
}

// This test also ensures that (device) radix_sort is stable
template<typename TestFixture>
void sort_pairs()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    using value_type                         = typename TestFixture::params::value_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0;
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const bool debug_synchronous = false;

    bool in_place = false;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        auto sizes = test_utils::get_sizes(seed_value);
        sizes.push_back(1 << 23);

        for(size_t size : sizes)
        {
            if(size > (1 << 17) && !check_large_sizes)
            {
                break;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            engine_type rng_engine(seed_value);
            in_place = !in_place;

            // Generate data
            auto keys_input = std::make_unique<key_type[]>(size);
            generate_key_input(keys_input.get(), size, rng_engine);

            std::vector<value_type> values_input(size);
            test_utils::iota(values_input.begin(), values_input.end(), 0);

            common::device_ptr<key_type>  d_keys_input(keys_input, size);
            common::device_ptr<key_type>  d_keys_output_alloc;
            common::device_ptr<key_type>& d_keys_output
                = in_place ? d_keys_input : d_keys_output_alloc;
            d_keys_output_alloc.resize(in_place ? 0 : size);

            common::device_ptr<value_type>  d_values_input(values_input);
            common::device_ptr<value_type>  d_values_output_alloc;
            common::device_ptr<value_type>& d_values_output
                = in_place ? d_values_input : d_values_output_alloc;
            d_values_output_alloc.resize(in_place ? 0 : size);

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
            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }

            // Use arbitrary custom config to increase test coverage without making more test cases
            using config = rocprim::radix_sort_config<
                rocprim::kernel_config<256, 1>,
                rocprim::merge_sort_config<128, 64, 2, 128, 64, 2>,
                rocprim::radix_sort_onesweep_config<rocprim::kernel_config<128, 1>,
                                                    rocprim::kernel_config<128, 1>,
                                                    4>,
                1024 * 512>;

            test_utils::GraphHelper gHelper;

            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            size_t temporary_storage_bytes;
            HIP_CHECK((invoke_sort_pairs<config, descending>(nullptr,
                                                             temporary_storage_bytes,
                                                             d_keys_input.get(),
                                                             d_keys_output.get(),
                                                             d_values_input.get(),
                                                             d_values_output.get(),
                                                             size,
                                                             start_bit,
                                                             end_bit,
                                                             stream,
                                                             debug_synchronous)));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            ASSERT_GT(temporary_storage_bytes, 0);
            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            if(TestFixture::params::use_graphs)
            {
                gHelper.resetGraphHelper(stream);
            }

            HIP_CHECK((invoke_sort_pairs<config, descending>(d_temporary_storage.get(),
                                                             temporary_storage_bytes,
                                                             d_keys_input.get(),
                                                             d_keys_output.get(),
                                                             d_values_input.get(),
                                                             d_values_output.get(),
                                                             size,
                                                             start_bit,
                                                             end_bit,
                                                             stream,
                                                             debug_synchronous)));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            const auto keys_output   = d_keys_output.load_to_unique_ptr();
            const auto values_output = d_values_output.load();

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output.get(),
                                                              keys_output.get() + size,
                                                              keys_expected.begin(),
                                                              keys_expected.end()));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(values_output.begin(),
                                                              values_output.end(),
                                                              values_expected.begin(),
                                                              values_expected.end()));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*                        d_temporary_storage,
                      size_t&                      temporary_storage_bytes,
                      rocprim::double_buffer<Key>& d_keys,
                      size_t                       size,
                      unsigned int                 start_bit,
                      unsigned int                 end_bit,
                      hipStream_t                  stream,
                      bool                         debug_synchronous)
    -> std::enable_if_t<!Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                            temporary_storage_bytes,
                                            d_keys,
                                            size,
                                            start_bit,
                                            end_bit,
                                            stream,
                                            debug_synchronous);
}

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*                        d_temporary_storage,
                      size_t&                      temporary_storage_bytes,
                      rocprim::double_buffer<Key>& d_keys,
                      size_t                       size,
                      unsigned int                 start_bit,
                      unsigned int                 end_bit,
                      hipStream_t                  stream,
                      bool                         debug_synchronous)
    -> std::enable_if_t<Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_keys_desc<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys,
                                                 size,
                                                 start_bit,
                                                 end_bit,
                                                 stream,
                                                 debug_synchronous);
}

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*                        d_temporary_storage,
                      size_t&                      temporary_storage_bytes,
                      rocprim::double_buffer<Key>& d_keys,
                      size_t                       size,
                      unsigned int                 start_bit,
                      unsigned int                 end_bit,
                      hipStream_t                  stream,
                      bool                         debug_synchronous)
    -> std::enable_if_t<!Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys,
                                                size,
                                                decomposer_t{},
                                                stream,
                                                debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys,
                                                size,
                                                decomposer_t{},
                                                start_bit,
                                                end_bit,
                                                stream,
                                                debug_synchronous);
    }
}

template<class Config, bool Descending, class Key>
auto invoke_sort_keys(void*                        d_temporary_storage,
                      size_t&                      temporary_storage_bytes,
                      rocprim::double_buffer<Key>& d_keys,
                      size_t                       size,
                      unsigned int                 start_bit,
                      unsigned int                 end_bit,
                      hipStream_t                  stream,
                      bool                         debug_synchronous)
    -> std::enable_if_t<Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_keys_desc<Config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_keys,
                                                     size,
                                                     decomposer_t{},
                                                     stream,
                                                     debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_keys_desc<Config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_keys,
                                                     size,
                                                     decomposer_t{},
                                                     start_bit,
                                                     end_bit,
                                                     stream,
                                                     debug_synchronous);
    }
}

template<typename TestFixture>
void sort_keys_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0;
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const bool debug_synchronous = false;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        auto sizes = test_utils::get_sizes(seed_value);
        sizes.push_back(1 << 23);

        for(size_t size : sizes)
        {
            if(size > (1 << 17) && !check_large_sizes)
            {
                break;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            engine_type rng_engine(seed_value);

            // Generate data
            auto keys_input = std::make_unique<key_type[]>(size);
            generate_key_input(keys_input.get(), size, rng_engine);

            common::device_ptr<key_type> d_keys_input(keys_input, size);
            common::device_ptr<key_type> d_keys_output(size);

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input.get(), keys_input.get() + size);
            std::stable_sort(
                expected.begin(),
                expected.end(),
                test_utils::key_comparator<key_type, descending, start_bit, end_bit>());

            rocprim::double_buffer<key_type> d_keys(d_keys_input.get(), d_keys_output.get());

            size_t temporary_storage_bytes;
            HIP_CHECK(
                (invoke_sort_keys<rocprim::default_config, descending>(nullptr,
                                                                       temporary_storage_bytes,
                                                                       d_keys,
                                                                       size,
                                                                       start_bit,
                                                                       end_bit,
                                                                       stream,
                                                                       debug_synchronous)));

            ASSERT_GT(temporary_storage_bytes, 0);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(
                (invoke_sort_keys<rocprim::default_config, descending>(d_temporary_storage.get(),
                                                                       temporary_storage_bytes,
                                                                       d_keys,
                                                                       size,
                                                                       start_bit,
                                                                       end_bit,
                                                                       stream,
                                                                       debug_synchronous)));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            auto keys_output = std::make_unique<key_type[]>(size);
            HIP_CHECK(hipMemcpy(keys_output.get(),
                                d_keys.current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output.get(),
                                                              keys_output.get() + size,
                                                              expected.begin(),
                                                              expected.end()));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                          d_temporary_storage,
                       size_t&                        temporary_storage_bytes,
                       rocprim::double_buffer<Key>&   d_keys,
                       rocprim::double_buffer<Value>& d_values,
                       size_t                         size,
                       unsigned int                   start_bit,
                       unsigned int                   end_bit,
                       hipStream_t                    stream,
                       bool                           debug_synchronous)
    -> std::enable_if_t<!Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_keys,
                                             d_values,
                                             size,
                                             start_bit,
                                             end_bit,
                                             stream,
                                             debug_synchronous);
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                          d_temporary_storage,
                       size_t&                        temporary_storage_bytes,
                       rocprim::double_buffer<Key>&   d_keys,
                       rocprim::double_buffer<Value>& d_values,
                       size_t                         size,
                       unsigned int                   start_bit,
                       unsigned int                   end_bit,
                       hipStream_t                    stream,
                       bool                           debug_synchronous)
    -> std::enable_if_t<Descending && !is_custom_not_float_test_type<Key>, hipError_t>
{
    return rocprim::radix_sort_pairs_desc<Config>(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys,
                                                  d_values,
                                                  size,
                                                  start_bit,
                                                  end_bit,
                                                  stream,
                                                  debug_synchronous);
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                          d_temporary_storage,
                       size_t&                        temporary_storage_bytes,
                       rocprim::double_buffer<Key>&   d_keys,
                       rocprim::double_buffer<Value>& d_values,
                       size_t                         size,
                       unsigned int                   start_bit,
                       unsigned int                   end_bit,
                       hipStream_t                    stream,
                       bool                           debug_synchronous)
    -> std::enable_if_t<!Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys,
                                                 d_values,
                                                 size,
                                                 decomposer_t{},
                                                 stream,
                                                 debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys,
                                                 d_values,
                                                 size,
                                                 decomposer_t{},
                                                 start_bit,
                                                 end_bit,
                                                 stream,
                                                 debug_synchronous);
    }
}

template<class Config, bool Descending, class Key, class Value>
auto invoke_sort_pairs(void*                          d_temporary_storage,
                       size_t&                        temporary_storage_bytes,
                       rocprim::double_buffer<Key>&   d_keys,
                       rocprim::double_buffer<Value>& d_values,
                       size_t                         size,
                       unsigned int                   start_bit,
                       unsigned int                   end_bit,
                       hipStream_t                    stream,
                       bool                           debug_synchronous)
    -> std::enable_if_t<Descending && is_custom_not_float_test_type<Key>, hipError_t>
{
    using decomposer_t = test_utils::custom_test_type_decomposer<Key>;
    if(start_bit == 0 && end_bit == rocprim::detail::decomposer_max_bits<decomposer_t, Key>::value)
    {
        return rocprim::radix_sort_pairs_desc<Config>(d_temporary_storage,
                                                      temporary_storage_bytes,
                                                      d_keys,
                                                      d_values,
                                                      size,
                                                      decomposer_t{},
                                                      stream,
                                                      debug_synchronous);
    }
    else
    {
        return rocprim::radix_sort_pairs_desc<Config>(d_temporary_storage,
                                                      temporary_storage_bytes,
                                                      d_keys,
                                                      d_values,
                                                      size,
                                                      decomposer_t{},
                                                      start_bit,
                                                      end_bit,
                                                      stream,
                                                      debug_synchronous);
    }
}

// This test also ensures that (device) radix_sort with rocprim::double_buffer is stable
template<typename TestFixture>
void sort_pairs_double_buffer()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                           = typename TestFixture::params::key_type;
    using value_type                         = typename TestFixture::params::value_type;
    constexpr bool         descending        = TestFixture::params::descending;
    constexpr unsigned int start_bit         = TestFixture::params::start_bit;
    constexpr unsigned int end_bit           = TestFixture::params::end_bit;
    constexpr bool         check_large_sizes = TestFixture::params::check_large_sizes;

    hipStream_t stream = 0;
    if (TestFixture::params::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    const bool debug_synchronous = false;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        auto sizes = test_utils::get_sizes(seed_value);
        sizes.push_back(1 << 23);

        for(size_t size : sizes)
        {
            if(size > (1 << 17) && !check_large_sizes)
            {
                break;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            engine_type rng_engine(seed_value);

            // Generate data
            auto keys_input = std::make_unique<key_type[]>(size);
            generate_key_input(keys_input.get(), size, rng_engine);

            std::vector<value_type> values_input(size);
            test_utils::iota(values_input.begin(), values_input.end(), 0);

            common::device_ptr<key_type> d_keys_input(keys_input, size);
            common::device_ptr<key_type> d_keys_output(size);

            common::device_ptr<value_type> d_values_input(values_input);
            common::device_ptr<value_type> d_values_output(size);

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
            std::vector<key_type>   keys_expected(size);
            std::vector<value_type> values_expected(size);
            for(size_t i = 0; i < size; i++)
            {
                keys_expected[i]   = expected[i].first;
                values_expected[i] = expected[i].second;
            }

            rocprim::double_buffer<key_type>   d_keys(d_keys_input.get(), d_keys_output.get());
            rocprim::double_buffer<value_type> d_values(d_values_input.get(),
                                                        d_values_output.get());

            size_t temporary_storage_bytes;
            HIP_CHECK(
                (invoke_sort_pairs<rocprim::default_config, descending>(nullptr,
                                                                        temporary_storage_bytes,
                                                                        d_keys,
                                                                        d_values,
                                                                        size,
                                                                        start_bit,
                                                                        end_bit,
                                                                        stream,
                                                                        debug_synchronous)));

            ASSERT_GT(temporary_storage_bytes, 0);
            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(
                (invoke_sort_pairs<rocprim::default_config, descending>(d_temporary_storage.get(),
                                                                        temporary_storage_bytes,
                                                                        d_keys,
                                                                        d_values,
                                                                        size,
                                                                        start_bit,
                                                                        end_bit,
                                                                        stream,
                                                                        debug_synchronous)));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            auto keys_output = std::make_unique<key_type[]>(size);
            HIP_CHECK(hipMemcpy(keys_output.get(),
                                d_keys.current(),
                                size * sizeof(key_type),
                                hipMemcpyDeviceToHost));

            std::vector<value_type> values_output(size);
            HIP_CHECK(hipMemcpy(values_output.data(),
                                d_values.current(),
                                size * sizeof(value_type),
                                hipMemcpyDeviceToHost));

            if(TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(keys_output.get(),
                                                              keys_output.get() + size,
                                                              keys_expected.begin(),
                                                              keys_expected.end()));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_bit_eq(values_output.begin(),
                                                              values_output.end(),
                                                              values_expected.begin(),
                                                              values_expected.end()));
        }
    }

    if(TestFixture::params::use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

template<bool UseGraphs = false>
void sort_keys_over_4g()
{
    using key_type                                 = uint8_t;
    constexpr unsigned int start_bit               = 0;
    constexpr unsigned int end_bit                 = 8ull * sizeof(key_type);
    constexpr bool         debug_synchronous       = false;
    constexpr size_t       size                    = (1ull << 32) + 32;
    constexpr size_t       number_of_possible_keys = 1ull << (8ull * sizeof(key_type));
    hipStream_t stream = 0;
    if (UseGraphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    assert(std::is_unsigned<key_type>::value);
    std::vector<size_t> histogram(number_of_possible_keys, 0);
    const int           seed_value = rand();

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    std::vector<key_type> keys_input
        = test_utils::get_random_data_wrapped<key_type>(size,
                                                        rocprim::numeric_limits<key_type>::min(),
                                                        rocprim::numeric_limits<key_type>::max(),
                                                        seed_value);

    //generate histogram of the randomly generated values
    std::for_each(keys_input.begin(), keys_input.end(), [&](const key_type& a) { histogram[a]++; });

    common::device_ptr<key_type> d_keys_input_output(keys_input);
    size_t key_type_storage_bytes = size * sizeof(key_type);

    size_t temporary_storage_bytes;
    HIP_CHECK(rocprim::radix_sort_keys(nullptr,
                                       temporary_storage_bytes,
                                       d_keys_input_output.get(),
                                       d_keys_input_output.get(),
                                       size,
                                       start_bit,
                                       end_bit,
                                       stream,
                                       debug_synchronous));

    ASSERT_GT(temporary_storage_bytes, 0);

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

   size_t total_storage_bytes = key_type_storage_bytes +  temporary_storage_bytes;
   if(total_storage_bytes > (static_cast<size_t>(prop.totalGlobalMem * 0.90)))
   {
       GTEST_SKIP() << "Test case device memory requirement (" << total_storage_bytes
                    << " bytes) exceeds available memory on current device (" << prop.totalGlobalMem
                    << " bytes). Skipping test";
   }

   common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

   test_utils::GraphHelper gHelper;
   if(UseGraphs)
   {
       gHelper.startStreamCapture(stream);
   }

   HIP_CHECK(rocprim::radix_sort_keys(d_temporary_storage.get(),
                                      temporary_storage_bytes,
                                      d_keys_input_output.get(),
                                      d_keys_input_output.get(),
                                      size,
                                      start_bit,
                                      end_bit,
                                      stream,
                                      debug_synchronous));

   if(UseGraphs)
   {
       gHelper.createAndLaunchGraph(stream);
   }

   const auto output = d_keys_input_output.load();

   size_t counter = 0;
   for(size_t i = 0; i <= rocprim::numeric_limits<key_type>::max(); ++i)
   {
       for(size_t j = 0; j < histogram[i]; ++j)
       {
           ASSERT_EQ(static_cast<size_t>(output[counter]), i);
           ++counter;
       }
   }
    ASSERT_EQ(counter, size);

    if (UseGraphs)
    {
        gHelper.cleanupGraphHelper();
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

inline void sort_keys_large_sizes()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                   = uint8_t;
    constexpr unsigned int start_bit = 0;
    constexpr unsigned int end_bit   = 8;

    hipStream_t stream = 0;

    // Currently, CI enforces a hard limit of 96 GB on memory allocations.
    // Temporarily use sizes that will require less space than the limit.
    // On Windows, sizes above 2^34 (that are still under the 96 GB limit)
    // can hang due to issues that we can't currently catch by examining
    // the hipMalloc return value or querying available memory. Workaround
    // this for now by setting a different maximum size for that platform.
#if defined(_WIN32)
    const size_t max_pow2 = 34;
#else
    const size_t max_pow2 = 35;
#endif
    const std::vector<size_t> sizes = test_utils::get_large_sizes<max_pow2>(seeds[0]);
    for(const size_t size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        common::device_ptr<key_type> d_keys;
        if(!d_keys.resize_with_memory_check(size))
        {
            std::cout << "Out of memory. Skipping size = " << size << std::endl;
            break;
        }

        // Generate data
        std::vector<key_type> keys_input(size);
        std::iota(keys_input.begin(), keys_input.end(), 0);
        d_keys.store(keys_input);

        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::radix_sort_keys(nullptr,
                                           temporary_storage_bytes,
                                           d_keys.get(),
                                           d_keys.get(),
                                           size,
                                           start_bit,
                                           end_bit,
                                           stream));

        ASSERT_GT(temporary_storage_bytes, 0U);
        common::device_ptr<void> d_temporary_storage;
        if(!d_temporary_storage.resize_with_memory_check(temporary_storage_bytes))
        {
            std::cout << "Out of memory. Skipping size = " << size << std::endl;
            break;
        }

        HIP_CHECK(rocprim::radix_sort_keys(d_temporary_storage.get(),
                                           temporary_storage_bytes,
                                           d_keys.get(),
                                           d_keys.get(),
                                           size,
                                           start_bit,
                                           end_bit,
                                           stream));

        const auto keys_output = d_keys.load();

        // Check if output values are as expected
        const size_t unique_keys    = size_t(rocprim::numeric_limits<key_type>::max()) + 1;
        const size_t segment_length = rocprim::detail::ceiling_div(size, unique_keys);
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

#endif // TEST_DEVICE_RADIX_SORT_HPP_
