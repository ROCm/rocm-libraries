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

#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_checker.hpp"

#include <rocprim/config.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/functional.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

test_suite_type_def(suite_name, name_suffix)

    typed_test_suite_def(RocprimWarpSortStableTests, name_suffix, warp_params);

typed_test_def(RocprimWarpSortStableTests, name_suffix, Sort)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::params::type;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    static constexpr size_t items_per_thread  = TestFixture::params::items_per_thread;
    static constexpr size_t warps_in_block    = (items_per_thread >= 16)  ? 1
                                                : (items_per_thread >= 8) ? 2
                                                                          : 4;

    static constexpr size_t block_size = logical_warp_size * warps_in_block;

    static constexpr unsigned int grid_size = std::max<unsigned int>(4, 1024 / block_size);
    const size_t                  size      = items_per_thread * block_size * grid_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    if(logical_warp_size > current_device_warp_size
       || !rocprim::detail::is_power_of_two(logical_warp_size))
    {
        GTEST_SKIP() << "Unsupported warp size";
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data (with duplicates to test stability potential, though hard to verify without values)
        // Using a smaller range [0, 100] ensures duplicates appear.
        std::vector<T> input    = test_utils::get_random_data_wrapped<T>(size, 0, 100, seed_value);
        std::vector<T> expected = input;

        // Calculate expected results on host using std::stable_sort
        // Loop over each logical warp and sort independently
        for(size_t i = 0; i < expected.size() / logical_warp_size / items_per_thread; i++)
        {
            auto start_iter = expected.begin() + (i * logical_warp_size * items_per_thread);
            auto end_iter   = expected.begin() + ((i + 1) * logical_warp_size * items_per_thread);
            std::stable_sort(start_iter, end_iter);
        }

        // Run Test for Merge Path Algorithm
        run_stable_sort_key_only_case<rocprim::warp_sort_stable_algorithm::merge_path,
                                      items_per_thread,
                                      block_size,
                                      logical_warp_size>(input, expected, grid_size);

        // Run Test for Shuffle Algorithm
        run_stable_sort_key_only_case<rocprim::warp_sort_stable_algorithm::shuffle,
                                      items_per_thread,
                                      block_size,
                                      logical_warp_size>(input, expected, grid_size);
    }
}

typed_test_def(RocprimWarpSortStableTests, name_suffix, SortKeyInt)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using KeyT   = typename TestFixture::params::type;
    using ValueT = int; // Use int as value to easily verify stability (e.g. original index)
    using PairT  = std::pair<KeyT, ValueT>;

    static constexpr size_t logical_warp_size = TestFixture::params::warp_size;
    static constexpr size_t items_per_thread  = TestFixture::params::items_per_thread;
    static constexpr size_t warps_in_block    = (items_per_thread >= 16)  ? 1
                                                : (items_per_thread >= 8) ? 2
                                                                          : 4;

    static constexpr size_t       block_size = logical_warp_size * warps_in_block;
    static constexpr unsigned int grid_size  = std::max<unsigned int>(4, 1024 / block_size);

    const size_t size = items_per_thread * block_size * grid_size;

    unsigned int current_device_warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, current_device_warp_size));

    if(logical_warp_size > current_device_warp_size
       || !rocprim::detail::is_power_of_two(logical_warp_size))
    {
        GTEST_SKIP() << "Unsupported warp size";
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate Keys (with duplicates [0, 20] to ensure stability is tested)
        std::vector<KeyT> input_keys
            = test_utils::get_random_data_wrapped<KeyT>(size, 0, 20, seed_value);

        // Generate Values (just sequence 0, 1, 2... to strictly track position)
        std::vector<ValueT> input_values(size);
        for(size_t i = 0; i < size; ++i)
            input_values[i] = static_cast<ValueT>(i);

        // Prepare Host Reference Data
        std::vector<KeyT>   expected_keys   = input_keys;
        std::vector<ValueT> expected_values = input_values;

        // Calculate expected results on host
        size_t items_per_warp = logical_warp_size * items_per_thread;
        for(size_t i = 0; i < size / items_per_warp; i++)
        {
            size_t offset = i * items_per_warp;

            // Create a temp vector of pairs for this warp to sort
            std::vector<PairT> warp_data(items_per_warp);
            for(size_t j = 0; j < items_per_warp; ++j)
            {
                warp_data[j]
                    = std::make_pair(expected_keys[offset + j], expected_values[offset + j]);
            }

            // Perform stable sort on host
            std::stable_sort(warp_data.begin(),
                             warp_data.end(),
                             [](const PairT& a, const PairT& b) { return a.first < b.first; });

            // Write back to expected vectors
            for(size_t j = 0; j < items_per_warp; ++j)
            {
                expected_keys[offset + j]   = warp_data[j].first;
                expected_values[offset + j] = warp_data[j].second;
            }
        }

        // Run Test for Merge Path Algorithm
        run_stable_sort_key_value_case<rocprim::warp_sort_stable_algorithm::merge_path,
                                       items_per_thread,
                                       block_size,
                                       logical_warp_size>(input_keys,
                                                          input_values,
                                                          expected_keys,
                                                          expected_values,
                                                          grid_size);

        // Run Test for Shuffle Algorithm
        run_stable_sort_key_value_case<rocprim::warp_sort_stable_algorithm::shuffle,
                                       items_per_thread,
                                       block_size,
                                       logical_warp_size>(input_keys,
                                                          input_values,
                                                          expected_keys,
                                                          expected_values,
                                                          grid_size);
    }
}
