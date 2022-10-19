// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_utils.hpp"

#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/iterator/reverse_iterator.hpp>

TEST(RocprimReverseIteratorTests, HostVector)
{
    using T                      = int;
    static constexpr size_t size = 100;

    std::vector<T> v(size);
    std::iota(v.begin(), v.end(), T(0));
    auto v_reversed = v;
    std::reverse(v_reversed.begin(), v_reversed.end());

    auto rev_it = rocprim::make_reverse_iterator(v.end());
    for(size_t i = 0; i < size; i++, rev_it++)
    {
        ASSERT_EQ(v_reversed[i], *rev_it);
    }
    ASSERT_EQ(rev_it, rocprim::make_reverse_iterator(v.begin()));
}

TEST(RocprimReverseIteratorTests, DeviceVector)
{
    using T                            = int;
    static constexpr size_t size       = 100;
    static constexpr int    seed_value = 7;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    auto           input    = test_utils::get_random_data<T>(size, 0, 100, seed_value);
    std::vector<T> expected = input;
    std::sort(expected.rbegin(), expected.rend());
    std::vector<T> output(size);

    T* d_input{};
    T* d_output{};
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, size * sizeof(T)));

    auto d_output_it = rocprim::make_reverse_iterator(d_output + size);

    void*  d_temp_storage{};
    size_t temp_storage_bytes;

    HIP_CHECK(
        rocprim::radix_sort_keys(d_temp_storage, temp_storage_bytes, d_input, d_output_it, size));
    ASSERT_NE(temp_storage_bytes, static_cast<size_t>(0));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_bytes));
    HIP_CHECK(
        rocprim::radix_sort_keys(d_temp_storage, temp_storage_bytes, d_input, d_output_it, size));
    HIP_CHECK(hipMemcpy(output.data(), d_output, size * sizeof(T), hipMemcpyDeviceToHost));

    ASSERT_EQ(expected, output);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

TEST(RocprimReverseIteratorTests, TestIncrement)
{
    using T                      = int;
    static constexpr size_t size = 5;

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), T(0));

    auto it = rocprim::make_reverse_iterator(data.end());
    ASSERT_EQ(4, *it);

    auto original_it = it++;
    ASSERT_EQ(4, *original_it);
    ASSERT_EQ(3, *it);

    auto new_it = ++it;
    ASSERT_EQ(2, *new_it);
    ASSERT_EQ(2, *it);

    new_it = it + 1;
    ASSERT_EQ(1, *new_it);

    it = 1 + it;
    ASSERT_EQ(1, *it);

    it += 1;
    ASSERT_EQ(0, *it);

    ASSERT_EQ(rocprim::make_reverse_iterator(data.begin()), ++it);
}

TEST(RocprimReverseIteratorTests, TestDecrement)
{
    using T                      = int;
    static constexpr size_t size = 5;

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), T(0));

    auto it = rocprim::make_reverse_iterator(data.begin());
    ASSERT_EQ(0, *--it);
    ASSERT_EQ(0, *it--);
    ASSERT_EQ(1, *it);

    auto original_it = it -= 1;
    ASSERT_EQ(2, *it);
    ASSERT_EQ(2, *original_it);

    auto new_it = it - 1;
    ASSERT_EQ(2, *it);
    ASSERT_EQ(3, *new_it);

    ASSERT_EQ(rocprim::make_reverse_iterator(data.end()), it - 2);
}

TEST(RocprimReverseIteratorTests, TestDereference)
{
    using T                      = int;
    static constexpr size_t size = 5;

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), T(0));

    auto it = rocprim::make_reverse_iterator(data.end());

    ASSERT_EQ(4, *it);
    ASSERT_EQ(4, it[0]);
    ASSERT_EQ(2, it[2]);
    ASSERT_EQ(0, it[4]);
}

TEST(RocprimReverseIteratorTests, TestComparison)
{
    using T                      = int;
    static constexpr size_t size = 5;

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), T(0));

    auto begin_it = rocprim::make_reverse_iterator(data.end());
    auto end_it   = rocprim::make_reverse_iterator(data.begin());

    ASSERT_GT(end_it, begin_it);
    ASSERT_GE(end_it, begin_it);
    ASSERT_GE(end_it, end_it);

    ASSERT_LT(begin_it, end_it);
    ASSERT_LE(begin_it, end_it);
    ASSERT_LE(begin_it, begin_it);

    ASSERT_EQ(begin_it, begin_it);
    ASSERT_EQ(end_it, end_it);

    ASSERT_NE(begin_it, end_it);
}

TEST(RocprimReverseIteratorTests, TestPtrDifference)
{
    using T                      = int;
    static constexpr size_t size = 5;

    std::vector<T> data(size);

    auto begin_it = rocprim::make_reverse_iterator(data.end());
    auto end_it   = rocprim::make_reverse_iterator(data.begin());

    ASSERT_EQ(size, end_it - begin_it);
}
