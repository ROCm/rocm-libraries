// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include "hostnumerics_test_support.hpp"

namespace hostnumerics_test {
void testIndexSelection() {
    using namespace roc::hostnumerics;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const Tensor operandA = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(a));
    const Tensor operandB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(b));
    Tensor d =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});

    matmulInto(operandA, operandB, d, MatmulOptions(ScalarType::Float32),
               IndexSelection::explicitIndicesLastDimensionFastest({0, 3}), GemmBackend::Automatic);
    require(d.loadAs<float>({0, 0}) == 19 && d.loadAs<float>({0, 1}) == -99 &&
                d.loadAs<float>({1, 0}) == -99 && d.loadAs<float>({1, 1}) == 50,
            "Selected-output GEMM modified the wrong elements.");

    Tensor firstDimensionFastestOutput =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});
    matmulInto(operandA, operandB, firstDimensionFastestOutput, MatmulOptions(ScalarType::Float32),
               IndexSelection::explicitIndices({1}, std::vector<size_t>{1, 0}),
               GemmBackend::Automatic);
    require(firstDimensionFastestOutput.loadAs<float>({0, 0}) == -99 &&
                firstDimensionFastestOutput.loadAs<float>({0, 1}) == -99 &&
                firstDimensionFastestOutput.loadAs<float>({1, 0}) == 43 &&
                firstDimensionFastestOutput.loadAs<float>({1, 1}) == -99,
            "Selected-output GEMM ignored the selection index order.");

    // SIZE_MAX is composite for the supported 32/64-bit size_t widths. The
    // candidate cannot advance to another odd value without wrapping.
    require(!detail::getNextPrime(std::numeric_limits<size_t>::max()).has_value(),
            "Prime-stride overflow did not return a failure.");
    requireError(
        [] {
            (void)IndexSelection::primeStrideLastDimensionFastest(
                2, std::numeric_limits<size_t>::max(), 1);
        },
        "Terminal prime-stride overflow did not throw.");
    const auto prime = IndexSelection::primeStrideLastDimensionFastest(10, 10, 3).indices(10);
    require(prime == std::vector<size_t>({0, 3, 6, 9}), "Prime-stride output selection mismatch.");
    require(IndexSelection::primeStrideLastDimensionFastest(20, 9, 1).indices(20) ==
                std::vector<size_t>({0, 11}),
            "Prime-stride output selection failed after a multiple of three.");
    require(IndexSelection::primeStrideLastDimensionFastest(60, 49, 1).indices(60) ==
                std::vector<size_t>({0, 53}),
            "Prime-stride output selection failed after a squared factor.");
    require(IndexSelection::primeStrideLastDimensionFastest(128, 121, 1).indices(128) ==
                std::vector<size_t>({0, 127}),
            "Prime-stride output selection failed after a larger squared factor.");
    require(IndexSelection::stridedLastDimensionFastest(2, 3, std::numeric_limits<size_t>::max())
                    .selectedCount(10) == 3,
            "Strided output selection reported the wrong count.");
    require(IndexSelection::stridedLastDimensionFastest(1, 2, 1).indices(10) ==
                std::vector<size_t>({1}),
            "Bounded strided output selection ignored its maximum count.");
    require(IndexSelection::stridedLastDimensionFastest(10, 3, std::numeric_limits<size_t>::max())
                    .selectedCount(10) == 0,
            "Out-of-range strided output selection reported a nonzero count.");
    require(IndexSelection::primeStrideLastDimensionFastest(10, 10, 0).selectsAll(),
            "Zero requested elements did not preserve all-output behavior.");
    const IndexSelection explicitSelection =
        IndexSelection::explicitIndicesLastDimensionFastest({3, 0, 3});
    require(explicitSelection.indices(4) == std::vector<size_t>({0, 3}),
            "Explicit output selection did not normalize to a unique ordered set.");
    require(explicitSelection.selectedCount(4) == 2,
            "Explicit output selection reported the wrong count.");
    bool rejectedOutOfRange = false;
    try {
        (void)explicitSelection.selectedCount(3);
    } catch (const std::runtime_error&) {
        rejectedOutOfRange = true;
    }
    require(rejectedOutOfRange, "Out-of-range explicit selection count did not fail.");
    rejectedOutOfRange = false;
    try {
        (void)explicitSelection.indices(3);
    } catch (const std::runtime_error&) {
        rejectedOutOfRange = true;
    }
    require(rejectedOutOfRange, "Out-of-range explicit selection materialization did not fail.");
}

}  // namespace hostnumerics_test
