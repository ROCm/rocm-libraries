/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <miopen/par_transform.hpp>

#include <algorithm>
#include <vector>

TEST(ParallelTransformTest, UnaryVersion)
{
    std::vector<int> st_seq1 {1, 2, 3, 4, 5, 9, 8, 7, 6, 5};
    std::transform(st_seq1.begin(), st_seq1.end(), st_seq1.begin(), [](int& a){ return a *= 2; });

    std::vector<int> mt_seq1 {1, 2, 3, 4, 5, 9, 8, 7, 6, 5};
    miopen::par_transform(mt_seq1.begin(), mt_seq1.end(), mt_seq1.begin(), [](int& a){ return a *= 2; });
    
    EXPECT_EQ(st_seq1, mt_seq1);
}

TEST(ParallelTransformTest, BinaryVersion)
{
    std::vector<int> st_seq1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> st_seq2 {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1};
    std::vector<int> st_result_seq{};
    st_result_seq.reserve(st_seq1.size());
    std::transform(st_seq1.begin(), st_seq1.end(), st_seq2.begin(), st_seq2.begin(), [](int const& a, int const& b){ return a + b; });
    
    std::vector<int> mt_seq1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> mt_seq2 {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1};
    std::vector<int> mt_result_seq{};
    mt_result_seq.reserve(mt_seq1.size());
    miopen::par_transform(mt_seq1.begin(), mt_seq1.end(), mt_seq2.begin(), mt_seq2.begin(), [](int const& a, int const& b){ return a + b; });
    EXPECT_EQ(st_result_seq, mt_result_seq);
}
