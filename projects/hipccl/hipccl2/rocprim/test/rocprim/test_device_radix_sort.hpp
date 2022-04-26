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

#ifndef TEST_DEVICE_RADIX_SORT_HPP_
#define TEST_DEVICE_RADIX_SORT_HPP_

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/device/device_radix_sort.hpp>

// required test headers
#include "test_utils_types.hpp"
#include "test_utils_sort_comparator.hpp"

template<
    class Key,
    class Value,
    bool Descending = false,
    unsigned int StartBit = 0,
    unsigned int EndBit = sizeof(Key) * 8,
    bool CheckLargeSizes = false
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr bool descending = Descending;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int end_bit = EndBit;
    static constexpr bool check_large_sizes = CheckLargeSizes;
};

template<class Params>
class RocprimDeviceRadixSort : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<signed char, double, true>,
    params<int, short>,
    params<short, int, true>,
    params<long long, char>,
    params<double, unsigned int>,
    params<double, int, true>,
    params<float, int>,
    params<rocprim::half, long long>,
    params<rocprim::bfloat16, long long>,
    params<int8_t, int8_t>,
    params<uint8_t, uint8_t>,
    params<rocprim::half, rocprim::half>,
    params<rocprim::bfloat16, rocprim::bfloat16>,
    params<int, test_utils::custom_test_type<float>>,

    // start_bit and end_bit
    params<unsigned char, int, true, 0, 7>,
    params<unsigned short, int, true, 4, 10>,
    params<unsigned int, short, false, 3, 22>,
    params<uint8_t, int8_t, true, 0, 7>,
    params<uint8_t, uint8_t, true, 4, 10>,
    params<unsigned int, double, true, 4, 21>,
    params<unsigned int, rocprim::half, true, 0, 15>,
    params<unsigned short, rocprim::half, false, 3, 22>,
    params<unsigned int, rocprim::bfloat16, true, 0, 12>,
    params<unsigned short, rocprim::bfloat16, false, 3, 11>,
    params<unsigned long long, char, false, 8, 20>,
    params<unsigned short, test_utils::custom_test_type<double>, false, 8, 11>,
    // some params used by PyTorch's Randperm()
    params<int64_t, int64_t, false, 0, 34>,
    params<int64_t, float, true, 0, 34>,
    params<int64_t, rocprim::half, true, 0, 34>,
    params<int64_t, int64_t, false, 0, 34, true>,

    // large sizes to check correctness of more than 1 block per batch
    params<int, char, false, 0, 32, true>,
    params<int, char, true, 0, 32, true>,
    params<float, char, false, 0, 32, true>,
    params<float, char, true, 0, 32, true>
> Params;

TYPED_TEST_SUITE(RocprimDeviceRadixSort, Params);

inline std::vector<unsigned int> get_sizes(int seed_value)
{
    std::vector<unsigned int> sizes = { 0, 1, 10, 53, 211, 1024, 2049, 2345, 4096, 8196, 34567, (1 << 16) - 1220, (1 << 23) - 76543 };
    const std::vector<unsigned int> random_sizes = test_utils::get_random_data<unsigned int>(10, 1, 100000, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

#endif // TEST_DEVICE_RADIX_SORT_HPP_