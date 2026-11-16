// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../../shared/params_gen.h"
#include <chrono>
#include <gtest/gtest.h>

#include <random>
#include <unordered_set>

inline auto generate_length_stride()
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
        // invalid due to collision for (11, 0, 0, 1) and (0, 1, 1, 0):
        {{12, 2, 2, 2}, {7, 37, 43, 3}},
        // invalid due to collision for (1, 0, 0, 0, 3) and (0, 4, 1, 1, 0):
        {{2, 5, 4, 6, 4}, {750, 201, 16, 17, 29}},
        // 8-dimensional cases
        // invalid nondegenerate
        {{5, 5, 5, 5, 5, 5, 5, 5},
         {32, 5 * 32, 25 * 32, 125 * 32, 503, 5 * 503, 25 * 503, 125 * 503}},
        // invalid degenerate
        {{4, 4, 4, 5, 4, 4, 4, 4},
         {32, 8 * 32, 64 * 32, 512 * 32, 129, 8 * 129, 64 * 129, 512 * 129}},
        // valid
        {{4, 4, 4, 4, 4, 4, 4, 4},
         {32, 8 * 32, 64 * 32, 512 * 32, 129, 8 * 129, 64 * 129, 512 * 129}},
        // valid although not nested (flattened indices 0, 2, 3, 5)
        {{2, 2}, {2, 3}},
        // valid: the largest stride exceeds the span of the other (non-nested) dimensions,
        // which restricts its coefficient to 0
        {{2, 2, 8}, {2, 3, 6}},
        // invalid due to collision for (1, 0, 1) and (0, 1, 0), although neither of the
        // offsets 0 and 9 of the planes spanned by the first two dimensions lies in the other
        {{2, 2, 2}, {1, 10, 9}},
        // invalid: 16 elements but only 13 flattened indices within the span
        {{4, 4}, {3, 1}},
        // valid, not nested, and requiring an exhaustive search
        {{150, 27, 101, 3}, {182038, 88960, 193077, 88967}},
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

    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > test_prob)
    {
        GTEST_SKIP();
    }

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

    auto start    = std::chrono::high_resolution_clock::now();
    auto test_val = array_valid(length, stride, verbose);
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if(verbose)
    {
        std::cout << "test value is:      " << (test_val ? "valid" : "invalid") << "\n";
    }

    start             = std::chrono::high_resolution_clock::now();
    auto ref_val      = direct_validity_test(length, stride, verbose);
    end               = std::chrono::high_resolution_clock::now();
    auto ref_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if(verbose)
    {
        std::cout << "reference value is: " << (ref_val ? "valid" : "invalid") << "\n";
        if(ref_duration.count() < duration.count())
            std::cout << "============================== WARNING =============================="
                      << std::endl;
        std::cout << "Test execution time (" << duration.count() << " us) is ";
        if(ref_duration.count() < duration.count())
            std::cout << "larger than ";
        else if(ref_duration.count() == duration.count())
            std::cout << "equal to ";
        else
        {
            std::cout << 100.0 * static_cast<double>(duration.count())
                             / static_cast<double>(ref_duration.count())
                      << " % of ";
        }
        std::cout << "the reference execution time (" << ref_duration.count() << " us)."
                  << std::endl;
        if(ref_duration.count() < duration.count())
            std::cout << "========================== END OF WARNING ==========================="
                      << std::endl;
    }

    EXPECT_EQ(test_val, ref_val);

    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(reference_test,
                         valid_length_stride,
                         ::testing::ValuesIn(generate_length_stride()));
