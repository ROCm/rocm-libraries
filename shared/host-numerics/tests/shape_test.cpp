// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testShape() {
    using namespace roc::host_numerics;
    const Shape shape{2, 3};
    require(shape.rank() == 2, "Shape rank mismatch.");
    require(shape.extent(0) == 2 && shape.extent(1) == 3, "Shape extent mismatch.");
    require(shape.elementCount() == 6, "Shape element count mismatch.");
    const Shape dimensionalShape{2, 3, 4};
    require(
        dimensionalShape.elementCount(1, 3) == 12 && dimensionalShape.elementCountExcluding(1) == 8,
        "Shape dimension product helper mismatch.");
    require(Shape{2, 0, 4}.elementCountExcluding(1) == 8,
            "Shape excluded-dimension product incorrectly retained a zero extent.");
    const std::array<size_t, 3> logicalCoordinates{1, 0, 2};
    require(
        dimensionalShape.linearIndex(logicalCoordinates, IndexOrder::FirstDimensionFastest) == 13 &&
            dimensionalShape.linearIndex(logicalCoordinates, IndexOrder::LastDimensionFastest) ==
                14,
        "Shape linear-index conversion mismatch.");
    require(dimensionalShape.coordinates(17, IndexOrder::FirstDimensionFastest) ==
                    std::vector<size_t>({1, 2, 2}) &&
                dimensionalShape.coordinates(17, IndexOrder::LastDimensionFastest) ==
                    std::vector<size_t>({1, 1, 1}),
            "Shape coordinate conversion mismatch.");
    const Shape overflowingLinearShape{std::numeric_limits<size_t>::max(), 2};
    const std::array<size_t, 2> overflowingLinearCoordinates{std::numeric_limits<size_t>::max() - 1,
                                                             1};
    requireOverflow(
        [&] {
            (void)overflowingLinearShape.linearIndex(overflowingLinearCoordinates,
                                                     IndexOrder::LastDimensionFastest);
        },
        "Shape linear index accepted overflowing coordinate arithmetic.");
}
}  // namespace tensor_core_test
