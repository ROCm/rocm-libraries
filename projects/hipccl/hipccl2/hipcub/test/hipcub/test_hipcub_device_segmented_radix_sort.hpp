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

#ifndef HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_RADIX_SORT_HPP_
#define HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_RADIX_SORT_HPP_

#include "common_test_header.hpp"

// hipcub API
#include "hipcub/device/device_segmented_radix_sort.hpp"

template<
    class Key,
    class Value,
    bool Descending,
    unsigned int StartBit,
    unsigned int EndBit,
    unsigned int MinSegmentLength,
    unsigned int MaxSegmentLength
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr bool descending = Descending;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int end_bit = EndBit;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
};

template<class Params>
class HipcubDeviceSegmentedRadixSort : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    params<signed char, double, true, 0, 8, 0, 1000>,
    params<int, short, false, 0, 32, 0, 100>,
    params<short, int, true, 0, 16, 0, 10000>,
    params<long long, char, false, 0, 64, 4000, 8000>,
    params<double, unsigned int, false, 0, 64, 2, 10>,
    params<double, int, true, 0, 64, 2000, 10000>,
    params<float, int, false, 0, 32, 0, 1000>,

    // start_bit and end_bit
    params<unsigned char, int, true, 2, 5, 0, 100>,
    params<unsigned short, int, true, 4, 10, 0, 10000>,
    params<unsigned int, short, false, 3, 22, 1000, 10000>,
    params<unsigned int, double, true, 4, 21, 100, 100000>,
    params<unsigned int, short, true, 0, 15, 100000, 200000>,
    params<unsigned long long, char, false, 8, 20, 0, 1000>,
    params<unsigned short, double, false, 8, 11, 50, 200>
> Params;

TYPED_TEST_SUITE(HipcubDeviceSegmentedRadixSort, Params);

inline std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500,
        2345, 11001, 34567,
        1000000,
        (1 << 16) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 100000, rand());
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

#endif // HIPCUB_TEST_HIPCUB_DEVICE_SEGMENTED_RADIX_SORT_HPP_
