// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <roc/hostnumerics/reshape_permutation.hpp>
#include <stdexcept>
#include <vector>

using namespace roc::hostnumerics;

// Tutorial: describe a padded tensor as factors, then make one factor run
// before another. The transform changes storage offsets; natural coordinates
// and the values associated with them do not change.
TEST(ReshapePermutationTutorial, PadsAndPermutesFactors) {
    const ReshapePermutation transform(Shape{2, 3}, Shape{2, 4}, Shape{2, 2, 2},
                                       Permutation{1, 0, 2});
    const std::array<int, 6> input{1, 2, 3, 4, 5, 6};
    std::vector<int> output(transform.physicalElementCount(), 0);
    for (size_t index = 0; index < input.size(); ++index)
        output[transform.physicalIndex(index)] = input[index];
    EXPECT_EQ(output, (std::vector<int>{1, 2, 4, 5, 3, 0, 6, 0}));
    EXPECT_TRUE(transform.naturalShape() == (Shape{2, 3}));
    EXPECT_TRUE(transform.paddedShape() == (Shape{2, 4}));
    EXPECT_TRUE(transform.reshapedShape() == (Shape{2, 2, 2}));
    EXPECT_EQ(transform.order(), (Permutation{1, 0, 2}));
    EXPECT_EQ(transform.physicalElementCount(), transform.paddedShape().elementCount());
    EXPECT_EQ(transform.physicalIndex(2), 4u);
    EXPECT_EQ(transform.physicalIndex(3), 2u);

    std::vector<int> loopOutput(transform.physicalElementCount(), 0);
    for (size_t pair = 0; pair < 2; ++pair)
        for (size_t row = 0; row < 2; ++row)
            for (size_t inPair = 0; inPair < 2; ++inPair) {
                const size_t column = 2 * pair + inPair;
                if (column < 3) loopOutput[(2 * pair + row) * 2 + inPair] = input[3 * row + column];
            }
    EXPECT_EQ(loopOutput, output);
}

TEST(ReshapePermutationTest, IdentityIncludesScalarsAndEmptyTensors) {
    const auto scalar = ReshapePermutation::identity(Shape{});
    EXPECT_EQ(scalar.naturalElementCount(), 1u);
    EXPECT_EQ(scalar.physicalIndex(0), 0u);
    const auto empty = ReshapePermutation::identity(Shape{0, 3});
    EXPECT_EQ(empty.physicalElementCount(), 0u);
    const auto matrix = ReshapePermutation::identity(Shape{2, 3});
    EXPECT_TRUE(matrix.paddedShape() == (Shape{2, 3}));
    EXPECT_EQ(matrix.physicalElementCount(), 6u);
    for (size_t index = 0; index < 6; ++index) EXPECT_EQ(matrix.physicalIndex(index), index);
}

TEST(ReshapePermutationTest, RejectsInconsistentShapesAndOrder) {
    EXPECT_THROW((ReshapePermutation({2, 3}, {6}, {6}, Permutation{0})), std::runtime_error);
    EXPECT_THROW((ReshapePermutation({2, 3}, {3, 2}, {6}, Permutation{0})), std::runtime_error);
    EXPECT_THROW((ReshapePermutation({2, 3}, {2, 4}, {2, 3}, Permutation{0, 1})),
                 std::runtime_error);
    EXPECT_THROW((ReshapePermutation({2, 3}, {2, 3}, {6}, Permutation{0, 1})), std::runtime_error);
}
