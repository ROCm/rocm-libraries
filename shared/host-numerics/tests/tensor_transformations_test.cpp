// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testTensorTransformations() {
    using namespace roc::host_numerics;
    Tensor reshapeSource(ScalarType::Int32, Shape{2, 3});
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            reshapeSource.storeFrom({row, column}, static_cast<int32_t>(row * 3 + column));
    Tensor reshaped = reshapeSource.reshapeSharingStorage(Shape{3, 2});
    require(reshaped.shape() == Shape{3, 2} && reshaped.loadAs<int32_t>({1, 1}) == 3,
            "Tensor reshape changed logical linear order.");
    reshaped.storeFrom({2, 1}, 19);
    require(reshapeSource.loadAs<int32_t>({1, 2}) == 19,
            "Tensor reshape did not return a shallow alias.");
    requireInvalidArgument([&] { (void)reshapeSource.reshapeSharingStorage(Shape{7}); },
                           "Tensor reshape accepted a different element count.");

    Tensor packedPaddingSource(ScalarType::Int4, Shape{3});
    packedPaddingSource.storeFrom({0}, -8);
    packedPaddingSource.storeFrom({1}, -3);
    packedPaddingSource.storeFrom({2}, 7);
    const Tensor packedPadded = packedPaddingSource.copyWithZeroPadding(Shape{5});
    require(packedPadded.shape() == Shape{5} &&
                packedPadded.rawEncodedBackingStorage().size() == 3 &&
                packedPadded.loadAs<int32_t>({0}) == -8 &&
                packedPadded.loadAs<int32_t>({1}) == -3 && packedPadded.loadAs<int32_t>({2}) == 7 &&
                packedPadded.loadAs<int32_t>({3}) == 0 && packedPadded.loadAs<int32_t>({4}) == 0,
            "Tensor padding did not preserve packed values and zero-fill new elements.");
    requireInvalidArgument([&] { (void)packedPaddingSource.copyWithZeroPadding(Shape{2}); },
                           "Tensor padding accepted a shrinking shape.");
    requireInvalidArgument([&] { (void)packedPaddingSource.copyWithZeroPadding(Shape{3, 1}); },
                           "Tensor padding accepted a different rank.");

    Tensor permutationSource(ScalarType::Float6E3M2, Shape{2, 3});
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            permutationSource.storeFrom({row, column}, static_cast<int32_t>(3 * row + column));
    const std::array<size_t, 2> transpose{1, 0};
    const Tensor permutationResult = permutationSource.copyWithPermutedDimensions(transpose);
    require(permutationResult.shape() == Shape{3, 2} &&
                permutationResult.rawEncodedBackingStorage().size() == 5,
            "Tensor permutation produced the wrong packed output geometry.");
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            require(permutationResult.loadAs<float>({column, row}) ==
                        permutationSource.loadAs<float>({row, column}),
                    "Tensor permutation changed a packed value.");
    const std::array<size_t, 2> duplicatePermutation{0, 0};
    const std::array<size_t, 2> outOfRangePermutation{0, 2};
    const std::array<size_t, 1> wrongRankPermutation{0};
    requireInvalidArgument(
        [&] { (void)permutationSource.copyWithPermutedDimensions(duplicatePermutation); },
        "Tensor permutation accepted a duplicate dimension.");
    requireInvalidArgument(
        [&] { (void)permutationSource.copyWithPermutedDimensions(outOfRangePermutation); },
        "Tensor permutation accepted an out-of-range dimension.");
    requireInvalidArgument(
        [&] { (void)permutationSource.copyWithPermutedDimensions(wrongRankPermutation); },
        "Tensor permutation accepted the wrong rank.");
}
}  // namespace tensor_core_test
