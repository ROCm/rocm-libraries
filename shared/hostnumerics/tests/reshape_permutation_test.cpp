// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <roc/hostnumerics/reshape_permutation.hpp>
#include <span>
#include <stdexcept>
#include <utility>
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
    copyWithReshapePermutation<int>(input, output, transform);
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

    // The same descriptor can be reused with a different caller-chosen padding.
    std::fill(output.begin(), output.end(), -7);
    copyWithReshapePermutation<int>(input, output, transform);
    EXPECT_EQ(output, (std::vector<int>{1, 2, 4, 5, 3, -7, 6, -7}));
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

TEST(ReshapePermutationTest, RejectsExtentAndStrideOverflow) {
    constexpr size_t maximum = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    // Shape accepts these extents; contiguous layout metadata must also fit ptrdiff_t.
    const Shape productOverflow{maximum, 2};
    EXPECT_THROW(
        (ReshapePermutation(productOverflow, productOverflow, productOverflow, Permutation{0, 1})),
        std::runtime_error);
    const Shape emptyWithOversizedExtent{0, maximum + 1};
    EXPECT_THROW((ReshapePermutation(emptyWithOversizedExtent, emptyWithOversizedExtent,
                                     emptyWithOversizedExtent, Permutation{0, 1})),
                 std::runtime_error);
}

TEST(ReshapePermutationTest, SelfMovePreservesPermutation) {
    ReshapePermutation transform({2, 3}, {2, 3}, {2, 3}, Permutation{1, 0});
    auto& alias = transform;
    transform = std::move(alias);
    const std::array<int, 6> input{0, 1, 2, 3, 4, 5};
    std::array<int, 6> output{};
    copyWithReshapePermutation<int>(input, output, transform);
    EXPECT_EQ(output, (std::array<int, 6>{0, 3, 1, 4, 2, 5}));
    EXPECT_EQ(transform.order(), (Permutation{1, 0}));
}

// Tutorial: store four 2 x 2 tiles from a 4 x 4 matrix. The last dimension
// runs fastest, both when visiting tiles and when visiting elements in a tile.
TEST(ReshapePermutationTutorial, TilesMatrix) {
    const std::vector<int> input{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
    const auto transform = ReshapePermutation::tiled({4, 4}, {2, 2});
    std::vector<int> output(transform.physicalElementCount());
    copyWithReshapePermutation<int>(input, output, transform);
    EXPECT_EQ(output,
              (std::vector<int>{0, 1, 10, 11, 2, 3, 12, 13, 20, 21, 30, 31, 22, 23, 32, 33}));
}

TEST(ReshapePermutationTest, TilesThreeDimensions) {
    const auto transform = ReshapePermutation::tiled({4, 6, 8}, {2, 3, 4});
    std::vector<int> input(size_t{4} * 6 * 8);
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> output(input.size());
    copyWithReshapePermutation<int>(input, output, transform);
    size_t destination = 0;
    // Independent tile traversal: outer coordinates first, then inner coordinates.
    for (size_t x = 0; x < 4; x += 2)
        for (size_t y = 0; y < 6; y += 3)
            for (size_t z = 0; z < 8; z += 4)
                for (size_t i = 0; i < 2; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 4; ++k)
                            EXPECT_EQ(output[destination++],
                                      input[((x + i) * 6 + y + j) * 8 + z + k]);
}

TEST(ReshapePermutationTest, CopiesScalarsEmptyTensorsAndWholeTiles) {
    const std::array<double, 1> input{1.25};
    std::array<double, 1> output{};
    copyWithReshapePermutation<double>(input, output, ReshapePermutation::tiled({}, {}));
    EXPECT_EQ(output, input);
    copyWithReshapePermutation<double>({}, {}, ReshapePermutation::tiled({0, 6}, {2, 3}));
    const std::vector<int> matrix{1, 2, 3, 4, 5, 6};
    std::vector<int> copied(matrix.size());
    copyWithReshapePermutation<int>(matrix, copied, ReshapePermutation::tiled({2, 3}, {2, 3}));
    EXPECT_EQ(copied, matrix);
    copyWithReshapePermutation<int>(matrix, copied, ReshapePermutation::identity({2, 3}));
    EXPECT_EQ(copied, matrix);
}

TEST(ReshapePermutationTest, CopyPreservesFloatingPointBits) {
    const std::array<uint32_t, 4> bits{0x7fc01234u, 0x80000000u, 0x7f800000u, 0xff800000u};
    const auto input = std::bit_cast<std::array<float, 4>>(bits);
    std::array<float, 4> output{};
    const ReshapePermutation transform({2, 2}, {2, 2}, {2, 2}, Permutation{1, 0});
    copyWithReshapePermutation<float>(input, output, transform);
    const auto copiedBits = std::bit_cast<std::array<uint32_t, 4>>(output);
    EXPECT_EQ(copiedBits, (std::array<uint32_t, 4>{bits[0], bits[2], bits[1], bits[3]}));
}

TEST(ReshapePermutationTest, RejectsInvalidTilesAndStorageCounts) {
    EXPECT_THROW(ReshapePermutation::tiled({4, 6}, {2}), std::runtime_error);
    EXPECT_THROW(ReshapePermutation::tiled({4, 6}, {2, 0}), std::runtime_error);
    EXPECT_THROW(ReshapePermutation::tiled({4, 6}, {0, 3}), std::runtime_error);
    EXPECT_THROW(ReshapePermutation::tiled({4, 6}, {3, 3}), std::runtime_error);
    EXPECT_THROW(ReshapePermutation::tiled({0, 6}, {2, 4}), std::runtime_error);
    const ReshapePermutation transform({2, 3}, {2, 4}, {2, 2, 2}, Permutation{1, 0, 2});
    const std::vector<int> input(6);
    std::vector<int> output(8);
    const std::vector<int> tooMuchInput(7);
    std::vector<int> tooMuchOutput(9);
    EXPECT_THROW(copyWithReshapePermutation<int>(tooMuchInput, output, transform),
                 std::runtime_error);
    EXPECT_THROW(copyWithReshapePermutation<int>(input, tooMuchOutput, transform),
                 std::runtime_error);
    EXPECT_THROW(copyWithReshapePermutation<int>(input, std::span(output).first(7), transform),
                 std::runtime_error);
    EXPECT_THROW(copyWithReshapePermutation<int>(std::span(input).first(5), output, transform),
                 std::runtime_error);
}
