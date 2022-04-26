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

#include "test_device_segmented_radix_sort.hpp"

TYPED_TEST(RocprimDeviceSegmentedRadixSort, SortKeysDoubleBuffer)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    constexpr bool descending = TestFixture::params::descending;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(size_t size : sizes)
        {
            if (size == 0 && test_common_utils::use_hmm())
            {
                // hipMallocManaged() currently doesnt support zero byte allocation
                continue;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<key_type> keys_input;
            if(rocprim::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000, seed_value);
            }
            else
            {
                keys_input = test_utils::get_random_data<key_type>(
                    size,
                    std::numeric_limits<key_type>::min(),
                    std::numeric_limits<key_type>::max(),
                    seed_index
                );
            }

            std::vector<offset_type> offsets;
            unsigned int segments_count = 0;
            size_t offset = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);
                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            key_type * d_keys_input;
            key_type * d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_keys_input, keys_input.data(),
                    size * sizeof(key_type),
                    hipMemcpyHostToDevice
                )
            );

            offset_type * d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_offsets, offsets.data(),
                    (segments_count + 1) * sizeof(offset_type),
                    hipMemcpyHostToDevice
                )
            );

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(
                    expected.begin() + offsets[i],
                    expected.begin() + offsets[i + 1],
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>()
                );
            }

            rocprim::double_buffer<key_type> d_keys(d_keys_input, d_keys_output);

            // Use custom config
            using config = rocprim::segmented_radix_sort_config<7, 4, rocprim::kernel_config<192, 5>,
                rocprim::select_warp_sort_config_t<key_type>>;

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(
                rocprim::segmented_radix_sort_keys<config>(
                    nullptr, temporary_storage_bytes,
                    d_keys, size,
                    segments_count, d_offsets, d_offsets + 1,
                    start_bit, end_bit
                )
            );

            ASSERT_GT(temporary_storage_bytes, 0U);

            void * d_temporary_storage;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_keys_desc<config>(
                        d_temporary_storage, temporary_storage_bytes,
                        d_keys, size,
                        segments_count, d_offsets, d_offsets + 1,
                        start_bit, end_bit,
                        stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_keys<config>(
                        d_temporary_storage, temporary_storage_bytes,
                        d_keys, size,
                        segments_count, d_offsets, d_offsets + 1,
                        start_bit, end_bit,
                        stream, debug_synchronous
                    )
                );
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(
                hipMemcpy(
                    keys_output.data(), d_keys.current(),
                    size * sizeof(key_type),
                    hipMemcpyDeviceToHost
                )
            );

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));
        }
    }

}

TYPED_TEST(RocprimDeviceSegmentedRadixSort, SortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    static constexpr bool descending = TestFixture::params::descending;
    static constexpr unsigned int start_bit = TestFixture::params::start_bit;
    static constexpr unsigned int end_bit = TestFixture::params::end_bit;

    using offset_type = unsigned int;

    hipStream_t stream = 0;

    const bool debug_synchronous = false;

    std::random_device rd;
    std::default_random_engine gen(rd());

    std::uniform_int_distribution<size_t> segment_length_dis(
        TestFixture::params::min_segment_length,
        TestFixture::params::max_segment_length
    );

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        const std::vector<size_t> sizes = get_sizes(seed_value);
        for(size_t size : sizes)
        {
            if (size == 0 && test_common_utils::use_hmm())
            {
                // hipMallocManaged() currently doesnt support zero byte allocation
                continue;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<key_type> keys_input;
            if(rocprim::is_floating_point<key_type>::value)
            {
                keys_input = test_utils::get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000, seed_value);
            }
            else
            {
                keys_input = test_utils::get_random_data<key_type>(
                    size,
                    std::numeric_limits<key_type>::min(),
                    std::numeric_limits<key_type>::max(),
                    seed_index
                );
            }

            std::vector<offset_type> offsets;
            unsigned int segments_count = 0;
            size_t offset = 0;
            while(offset < size)
            {
                const size_t segment_length = segment_length_dis(gen);
                offsets.push_back(offset);
                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(size);

            key_type * d_keys_input;
            key_type * d_keys_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input, size * sizeof(key_type)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_output, size * sizeof(key_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_keys_input, keys_input.data(),
                    size * sizeof(key_type),
                    hipMemcpyHostToDevice
                )
            );

            offset_type * d_offsets;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_offsets, (segments_count + 1) * sizeof(offset_type)));
            HIP_CHECK(
                hipMemcpy(
                    d_offsets, offsets.data(),
                    (segments_count + 1) * sizeof(offset_type),
                    hipMemcpyHostToDevice
                )
            );

            // Calculate expected results on host
            std::vector<key_type> expected(keys_input);
            for(size_t i = 0; i < segments_count; i++)
            {
                std::stable_sort(
                    expected.begin() + offsets[i],
                    expected.begin() + offsets[i + 1],
                    test_utils::key_comparator<key_type, descending, start_bit, end_bit>()
                );
            }

            size_t temporary_storage_bytes = 0;
            HIP_CHECK(
                rocprim::segmented_radix_sort_keys(
                    nullptr, temporary_storage_bytes,
                    d_keys_input, d_keys_output, size,
                    segments_count, d_offsets, d_offsets + 1,
                    start_bit, end_bit
                )
            );

            ASSERT_GT(temporary_storage_bytes, 0U);

            void * d_temporary_storage;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

            if(descending)
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_keys_desc(
                        d_temporary_storage, temporary_storage_bytes,
                        d_keys_input, d_keys_output, size,
                        segments_count, d_offsets, d_offsets + 1,
                        start_bit, end_bit,
                        stream, debug_synchronous
                    )
                );
            }
            else
            {
                HIP_CHECK(
                    rocprim::segmented_radix_sort_keys(
                        d_temporary_storage, temporary_storage_bytes,
                        d_keys_input, d_keys_output, size,
                        segments_count, d_offsets, d_offsets + 1,
                        start_bit, end_bit,
                        stream, debug_synchronous
                    )
                );
            }

            std::vector<key_type> keys_output(size);
            HIP_CHECK(
                hipMemcpy(
                    keys_output.data(), d_keys_output,
                    size * sizeof(key_type),
                    hipMemcpyDeviceToHost
                )
            );

            HIP_CHECK(hipFree(d_temporary_storage));
            HIP_CHECK(hipFree(d_keys_input));
            HIP_CHECK(hipFree(d_keys_output));
            HIP_CHECK(hipFree(d_offsets));

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output, expected));
        }
    }

}
