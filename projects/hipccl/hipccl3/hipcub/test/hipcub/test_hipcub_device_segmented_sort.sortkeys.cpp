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

#include "test_hipcub_device_segmented_sort.hpp"

TYPED_TEST(HipcubDeviceSegmentedSort, SortKeys)
{
    using key_type = typename TestFixture::params::key_type;
    using offset_type = unsigned int;
    constexpr SortMethod method = TestFixture::params::method;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int min_segment_length = TestFixture::params::min_segment_length;
    constexpr unsigned int max_segment_length = TestFixture::params::max_segment_length;
    constexpr hipStream_t stream = 0;
    constexpr bool debug_synchronous = false;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; ++seed_index)
    {
        const int seed_value = seed_index < random_seeds_count ? seeds[seed_index] : rand();
        for (const size_t size : get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            std::vector<key_type> keys_input;
            std::vector<offset_type> offsets;
            generate_input_data(
                keys_input, offsets,
                size, seed_value,
                min_segment_length, max_segment_length
            );
            const size_t segments_count = offsets.size() - 1;

            key_type * d_keys_input = hipMallocAndCopy(keys_input);
            offset_type * d_offsets = hipMallocAndCopy(offsets);
            key_type * d_keys_output{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));

            const std::vector<key_type> expected = generate_expected_data(keys_input, offsets, descending);

            size_t temporary_storage_bytes{};
            dispatch_sort_keys(
                method, static_cast<void *>(nullptr), temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                segments_count, d_offsets, d_offsets + 1
            );

            ASSERT_GT(temporary_storage_bytes, 0U);

            void * d_temporary_storage{};
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            dispatch_sort_keys(
                method, d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, size,
                segments_count, d_offsets, d_offsets + 1,
                stream, debug_synchronous
            );

            const std::vector<key_type> keys_output = download(d_keys_output, size);

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets));

            ASSERT_EQ(keys_output, expected);
        }
    }
}
