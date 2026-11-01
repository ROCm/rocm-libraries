/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Host-only checks of the tuned hyper-parameter lookup used to pick a permutation instance.
// The lookup buckets every extent to an edge of a per-rank table; extents beyond the last edge
// must land in the last bucket instead of reading past the end of the table.

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "elementwise/elementwise_instance_selection.hpp"

namespace
{
    hiptensor::InstanceHyperParams selectFloatPermutation(std::vector<std::size_t> const& lengths)
    {
        // Identity output mode order, given as positions in the input as the registry does.
        std::vector<int32_t> outputMode(lengths.size());
        std::iota(outputMode.begin(), outputMode.end(), 0);
        return hiptensor::selectInstanceParams(lengths,
                                               outputMode,
                                               {HIPTENSOR_R_32F},
                                               {HIPTENSOR_R_32F},
                                               static_cast<ck::index_t>(lengths.size()));
    }

    void expectSameParams(hiptensor::InstanceHyperParams const& actual,
                          hiptensor::InstanceHyperParams const& expected)
    {
        EXPECT_EQ(actual.mBlockSize, expected.mBlockSize);
        EXPECT_EQ(actual.mM0PerBlock, expected.mM0PerBlock);
        EXPECT_EQ(actual.mM1PerBlock, expected.mM1PerBlock);
        EXPECT_EQ(actual.mM0PerThread, expected.mM0PerThread);
        EXPECT_EQ(actual.mM1PerThread, expected.mM1PerThread);
        EXPECT_EQ(actual.mThreadClusterArrangeOrder, expected.mThreadClusterArrangeOrder);
        EXPECT_EQ(actual.mInScalarPerVectorSeq, expected.mInScalarPerVectorSeq);
        EXPECT_EQ(actual.mOutScalarPerVectorSeq, expected.mOutScalarPerVectorSeq);
    }

    // What selectInstanceParams returns for a single input when the table has no matching row.
    hiptensor::InstanceHyperParams const defaultParams = {256, 64, 64, 4, 4, {0, 1}, {4}, {4}};
} // namespace

// The last rank-2 edge is 33,554,432 and the table has a tuned row for {33554432, 32} that
// differs from the default, so an extent just past the edge must select that same row.
TEST(ElementwiseInstanceSelectionTest, Rank2ExtentAboveLastEdgeUsesLastBucket)
{
    auto const atLastEdge = selectFloatPermutation({33554432, 32});
    EXPECT_NE(atLastEdge.mBlockSize, defaultParams.mBlockSize);

    expectSameParams(selectFloatPermutation({33554433, 32}), atLastEdge);
    expectSameParams(selectFloatPermutation({67108864, 32}), atLastEdge);
}

// The last rank-3 edge is 524,288, with a tuned row for {524288, 32, 32}.
TEST(ElementwiseInstanceSelectionTest, Rank3ExtentAboveLastEdgeUsesLastBucket)
{
    auto const atLastEdge = selectFloatPermutation({524288, 32, 32});
    EXPECT_NE(atLastEdge.mBlockSize, defaultParams.mBlockSize);

    expectSameParams(selectFloatPermutation({524289, 32, 32}), atLastEdge);
    expectSameParams(selectFloatPermutation({1048576, 32, 32}), atLastEdge);

    // The same holds when the large extent is not the leading one.
    expectSameParams(selectFloatPermutation({32, 32, 524289}),
                     selectFloatPermutation({32, 32, 524288}));
}

// The last rank-4 edge is 131,072. The table has no row in that bucket, so an extent beyond it
// falls through to the default hyper-parameters.
TEST(ElementwiseInstanceSelectionTest, Rank4ExtentAboveLastEdgeUsesLastBucket)
{
    expectSameParams(selectFloatPermutation({131073, 32, 32, 32}),
                     selectFloatPermutation({131072, 32, 32, 32}));
    expectSameParams(selectFloatPermutation({131073, 32, 32, 32}), defaultParams);
}
