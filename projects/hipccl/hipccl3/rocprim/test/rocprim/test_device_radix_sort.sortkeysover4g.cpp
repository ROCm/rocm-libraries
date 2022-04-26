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

#include "test_device_radix_sort.hpp"


TEST(RocprimDeviceRadixSort, SortKeysOver4G)
{
    using key_type = uint8_t;
    constexpr unsigned int start_bit = 0;
    constexpr unsigned int end_bit = 8ull * sizeof(key_type);
    constexpr hipStream_t stream = 0;
    constexpr bool debug_synchronous = false;
    constexpr size_t size = (1ull << 32) + 32;
    constexpr size_t number_of_possible_keys = 1ull << (8ull * sizeof(key_type));
    assert(std::is_unsigned<key_type>::value);
    std::vector<size_t> histogram(number_of_possible_keys, 0);
    const int seed_value = rand();

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    std::vector<key_type> keys_input = test_utils::get_random_data<key_type>(
        size,
        std::numeric_limits<key_type>::min(),
        std::numeric_limits<key_type>::max(),
        seed_value);

    //generate histogram of the randomly generated values
    std::for_each(keys_input.begin(), keys_input.end(), [&](const key_type &a){
        histogram[a]++;
    });

    key_type * d_keys_input_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys_input_output, size * sizeof(key_type)));
    HIP_CHECK(hipMemcpy(d_keys_input_output, keys_input.data(), size * sizeof(key_type), hipMemcpyHostToDevice));

    size_t temporary_storage_bytes;
    HIP_CHECK(
        rocprim::radix_sort_keys(
            nullptr, temporary_storage_bytes,
            d_keys_input_output, d_keys_input_output, size,
            start_bit, end_bit,
            stream, debug_synchronous
        )
    );

    ASSERT_GT(temporary_storage_bytes, 0);
    void * d_temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temporary_storage, temporary_storage_bytes));

    HIP_CHECK(
        rocprim::radix_sort_keys(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input_output, d_keys_input_output, size,
            start_bit, end_bit,
            stream, debug_synchronous
        )
    );
    
    std::vector<key_type> output(keys_input.size());
    HIP_CHECK(hipMemcpy(output.data(), d_keys_input_output, size * sizeof(key_type), hipMemcpyDeviceToHost));

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
