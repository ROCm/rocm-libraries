// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../shared/accuracy_test.h"
#include "../../shared/array_validator.h"
#include <gtest/gtest.h>

#include <random>
#include <unordered_set>

inline auto generate_valid_length_stride()
{
    // Array of tuples of length, stride.
    std::vector<std::tuple<std::vector<size_t>, std::vector<size_t>>> vals = {
        {{8}, {1}},
        {{8, 2}, {1, 0}},
        {{8, 8}, {8, 1}},
        {{8, 8, 8}, {64, 8, 1}},
        {{8, 8, 8}, {64, 7, 1}},
        {{8, 8, 8, 8}, {512, 64, 7, 1}},
        {{8, 8, 8, 8}, {512, 64, 8, 1}},
        {{8, 8, 8, 8, 8}, {4096, 512, 64, 8, 1}},
        {{8, 8, 8, 8, 8}, {4096, 512, 64, 7, 1}},
        {{8, 8, 8, 8, 8, 8}, {32768, 4096, 512, 64, 8, 1}},
        {{299, 307, 495}, {1006, 50, 674}},

    };

    return vals;
}

class valid_length_stride
    : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, std::vector<size_t>>>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<accuracy_test::ParamType>& info)
    {
        return info.param.token();
    }
};

auto direct_validity_test(const std::vector<size_t>& length,
                          const std::vector<size_t>& stride,
                          const int                  verbose)
{
    std::unordered_set<size_t> vals{};

    std::vector<size_t> index(length.size());
    std::fill(index.begin(), index.end(), 0);
    do
    {
        const int i = std::inner_product(index.begin(), index.end(), stride.begin(), (size_t)0);
        if(vals.find(i) == vals.end())
        {
            vals.insert(i);
        }
        else
        {
            return false;
        }
    } while(increment_rowmajor(index, length));

    return true;
}

TEST_P(valid_length_stride, direct_comparison)
{
    const std::vector<size_t> length = std::get<0>(GetParam());
    const std::vector<size_t> stride = std::get<1>(GetParam());

    if(verbose)
    {
        std::cout << "length:";
        for(const auto i : length)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "stride:";
        for(const auto i : stride)
            std::cout << " " << i;
        std::cout << "\n";
    }

    auto test_val = array_valid(length, stride, verbose);
    if(verbose)
    {
        std::cout << "test value is:      " << (test_val ? "valid" : "invalid") << "\n";
    }

    auto ref_val = direct_validity_test(length, stride, verbose);
    if(verbose)
    {
        std::cout << "reference value is: " << (ref_val ? "valid" : "invalid") << "\n";
    }

    EXPECT_EQ(test_val, ref_val);

    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(reference_test,
                         valid_length_stride,
                         ::testing::ValuesIn(generate_valid_length_stride()));
