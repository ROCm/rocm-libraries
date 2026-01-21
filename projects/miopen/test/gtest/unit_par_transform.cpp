// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/par_transform.hpp>

#include <algorithm>
#include <vector>

TEST(CPU_ParallelTransformTest_NONE, UnaryVersion)
{
    std::vector<int> st_seq1{1, 2, 3, 4, 5, 9, 8, 7, 6, 5};
    std::transform(st_seq1.begin(), st_seq1.end(), st_seq1.begin(), [](int& a) { return a *= 2; });

    std::vector<int> mt_seq1{1, 2, 3, 4, 5, 9, 8, 7, 6, 5};
    miopen::par_transform(
        mt_seq1.begin(), mt_seq1.end(), mt_seq1.begin(), [](int& a) { return a *= 2; });

    EXPECT_EQ(st_seq1, mt_seq1);
}

TEST(CPU_ParallelTransformTest_NONE, BinaryVersion)
{
    std::vector<int> st_seq1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> st_seq2{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1};
    std::vector<int> st_result_seq{};
    st_result_seq.reserve(st_seq1.size());
    std::transform(st_seq1.begin(),
                   st_seq1.end(),
                   st_seq2.begin(),
                   st_seq2.begin(),
                   [](int const& a, int const& b) { return a + b; });

    std::vector<int> mt_seq1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> mt_seq2{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1};
    std::vector<int> mt_result_seq{};
    mt_result_seq.reserve(mt_seq1.size());
    miopen::par_transform(mt_seq1.begin(),
                          mt_seq1.end(),
                          mt_seq2.begin(),
                          mt_seq2.begin(),
                          [](int const& a, int const& b) { return a + b; });
    EXPECT_EQ(st_result_seq, mt_result_seq);
}
