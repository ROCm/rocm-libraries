// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include "hostnumerics_test_support.hpp"

namespace hostnumerics_test {
void testIndexSelection() {
    using namespace roc::hostnumerics;
    const Shape shape{2, 4};
    const std::vector<size_t> order{0, 1};
    const IndexSelection range(IndexRange(1, 2, 3), order);
    const IndexSelection list(std::vector<size_t>{5, 1, 3, 5}, order);
    const std::vector<size_t> expected{1, 3, 5};
    require(shape.selectedIndices(range) == expected && shape.selectedIndices(list) == expected,
            "Range and explicit selection disagree.");
    std::vector<std::vector<size_t>> coordinates;
    shape.forEachCoordinate(range, [&](size_t, std::span<const size_t> point) {
        coordinates.emplace_back(point.begin(), point.end());
    });
    require(coordinates == std::vector<std::vector<size_t>>{{0, 1}, {0, 3}, {1, 1}},
            "Coordinate visitation ignored the shape/order.");
    require(shape.selectsAll(IndexSelection(IndexRange(0, 1, 8), order)),
            "Full-coverage range was not recognized.");
    requireError([&] { (void)shape.selectedCount(IndexSelection(std::vector<size_t>{8}, order)); },
                 "Explicit bounds were not checked.");
    requireError(
        [&] {
            (void)shape.selectedCount(IndexSelection(IndexRange::all(), std::vector<size_t>{}));
        },
        "Explicit empty order was accepted for nonzero rank.");
    requireError([] { (void)IndexRange(0, 0, 3); }, "Zero stride was accepted.");
    requireError([] { (void)primeIndexRange(2, SIZE_MAX, 1); }, "Prime overflow was accepted.");
    const Shape huge{SIZE_MAX};
    const IndexSelection last(IndexRange(1, SIZE_MAX - 2, 2), LastDimensionFastest{});
    require(huge.selectedIndices(last) == std::vector<size_t>{1, SIZE_MAX - 1},
            "Range arithmetic wrapped at size_t boundary.");

    const Tensor a = Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{1, 2, 3, 4});
    const Tensor b = Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{5, 6, 7, 8});
    Tensor d =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});
    matmulInto(a, b, d, MatmulOptions(ScalarType::Float32),
               IndexSelection(IndexRange(1, 2, 1), std::vector<size_t>{1, 0}),
               GemmBackend::Blocked);
    require(d.loadAs<float>({1, 0}) == 43 && d.loadAs<float>({0, 1}) == -99 &&
                d.loadAs<float>({0, 0}) == -99 && d.loadAs<float>({1, 1}) == -99,
            "Selected GEMM modified the wrong coordinates.");
}
}  // namespace hostnumerics_test
