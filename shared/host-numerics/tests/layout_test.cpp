// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testLayoutBasics() {
    using namespace roc::host_numerics;
    const Shape shape{2, 3};
    const Layout rowMajor = Layout::contiguousLastDimensionFastest(shape);
    const Layout columnMajor = Layout::contiguousFirstDimensionFastest(shape);
    require(rowMajor == Layout::contiguousLastDimensionFastest(shape),
            "Contiguous compatibility API changed dimension order.");
    require(rowMajor.shape().rank() == 2 && rowMajor.shape().dimensions().size() == 2 &&
                rowMajor.shape().extent(0) == 2 && rowMajor.shape().elementCount() == 6 &&
                rowMajor.shape().elementCount(0, 1) == 2 &&
                rowMajor.shape().elementCountExcluding(0) == 3 && rowMajor.stride(0) == 3 &&
                rowMajor.stride(1) == 1,
            "Last-dimension-fastest layout helper mismatch.");
    require(columnMajor.stride(0) == 1 && columnMajor.stride(1) == 2,
            "First-dimension-fastest layout helper mismatch.");

    const ptrdiff_t maximumOffset = std::numeric_limits<ptrdiff_t>::max();
    const ptrdiff_t minimumOffset = std::numeric_limits<ptrdiff_t>::min();
    const std::array<size_t, 1> edgeIndex{1};
    require(Layout(Shape{2}, {maximumOffset}, minimumOffset).elementOffset(edgeIndex) == -1,
            "Layout rejected an exactly representable positive offset contribution.");
    require(Layout(Shape{2}, {minimumOffset}, maximumOffset).elementOffset(edgeIndex) == -1,
            "Layout rejected an exactly representable negative offset contribution.");

    const std::array<size_t, 1> overflowIndex{2};
    const Layout positiveMultiplyOverflow(Shape{3}, {maximumOffset});
    requireOverflow([&] { (void)positiveMultiplyOverflow.elementOffset(overflowIndex); },
                    "Layout element offset accepted positive multiplication overflow.");
    const Layout negativeMultiplyOverflow(Shape{3}, {minimumOffset});
    requireOverflow([&] { (void)negativeMultiplyOverflow.elementOffset(overflowIndex); },
                    "Layout element offset accepted negative multiplication overflow.");
    const Layout positiveAddOverflow(Shape{2}, {1}, maximumOffset);
    requireOverflow([&] { (void)positiveAddOverflow.elementOffset(edgeIndex); },
                    "Layout element offset accepted positive addition overflow.");
    const Layout negativeAddOverflow(Shape{2}, {-1}, minimumOffset);
    requireOverflow([&] { (void)negativeAddOverflow.elementOffset(edgeIndex); },
                    "Layout element offset accepted negative addition overflow.");

    requireOverflow(
        [&] { (void)storageBytesForLayout(ScalarType::Float32, positiveMultiplyOverflow); },
        "Tensor layout bounds accepted multiplication overflow.");
    requireOverflow([&] { (void)storageBytesForLayout(ScalarType::Float32, positiveAddOverflow); },
                    "Tensor layout bounds accepted addition overflow.");

    if constexpr (std::numeric_limits<size_t>::digits >= 64 &&
                  std::numeric_limits<ptrdiff_t>::digits >= 63) {
        constexpr uint64_t storageBits = 4;
        constexpr uint64_t maximumElements = std::numeric_limits<uint64_t>::max() / storageBits;
        constexpr uint64_t totalBits = maximumElements * storageBits;
        constexpr uint64_t expectedBytes =
            totalBits / 8 + static_cast<uint64_t>(totalBits % 8 != 0);
        const Layout maximumPackedLayout(Shape{static_cast<size_t>(maximumElements)}, {1});
        require(storageBytesForLayout(ScalarType::Int4, maximumPackedLayout) ==
                    static_cast<size_t>(expectedBytes),
                "Packed tensor storage byte rounding overflowed at its valid boundary.");

        const Layout overflowingPackedLayout(Shape{static_cast<size_t>(maximumElements + 1)}, {1});
        requireOverflow(
            [&] { (void)storageBytesForLayout(ScalarType::Int4, overflowingPackedLayout); },
            "Packed tensor storage accepted a bit-count multiplication overflow.");
    }
}
}  // namespace tensor_core_test
