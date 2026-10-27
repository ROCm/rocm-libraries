// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_blocked_backend_test_support.hpp"

namespace blocked_backend_test {
void testSmallEdgeBlock() {
    using namespace roc::hostnumerics;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    Tensor d(ScalarType::Float32, Shape{2, 2});
    GemmTestCase problem(
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 2}, 0), std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {1, 3}, 0), std::span<const float>(b)),
        d.shareStorageWithLayout(Layout(Shape{2, 2}, {1, 2}, 0)), ScalarType::Float32);
    require(queryGemmSupport(problem, GemmBackend::Blocked).supported,
            "Blocked backend unexpectedly rejected the test GEMM.");
    const GemmTestRunInfo full = referenceGemm(problem, GemmBackend::Blocked);
    require(full.outputElementsWritten == 4 && full.outputElementsCovered == 4,
            "Full blocked GEMM reported the wrong output counts.");
    const Tensor expected =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{58, 139, 64, 154});
    require(compare(d, expected, {}).passed(), "Blocked backend result mismatch.");

    fillTensor(d, untouchedValue);
    roc::hostnumerics::detail::bindIndices(problem, problem.d.shape(), std::set<size_t>({0}),
                                           LastDimensionFastest{});
    const GemmTestRunInfo selected = referenceGemm(problem, GemmBackend::Blocked);
    require(selected.outputElementsWritten == 1 && selected.outputElementsCovered == 4,
            "Selected blocked GEMM reported the wrong write or coverage count.");
    require(d.loadAs<float>({0, 0}) == 58 && d.loadAs<float>({0, 1}) == untouchedValue &&
                d.loadAs<float>({1, 0}) == untouchedValue &&
                d.loadAs<float>({1, 1}) == untouchedValue,
            "Blocked backend partial output selection mismatch.");

    fillTensor(d, untouchedValue);
    roc::hostnumerics::detail::bindIndices(problem, problem.d.shape(), std::set<size_t>({}),
                                           LastDimensionFastest{});
    const GemmTestRunInfo empty = referenceGemm(problem, GemmBackend::Blocked);
    require(empty.outputElementsWritten == 0 && empty.outputElementsCovered == 0,
            "Empty blocked output selection reported output work.");
    requireOnlySelectedOutputsStored(d, std::set<size_t>({}));
}

void testExplicitSelectionBlockPlan() {
    constexpr size_t rows = 45;
    constexpr size_t reductionElements = 16;
    constexpr size_t columns = 70;

    const std::vector<float> a = makeValues(rows, reductionElements, 1);
    const std::vector<float> b = makeValues(reductionElements, columns, 2);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const auto selection = std::set<size_t>({
        44 * columns + 69,
        2 * columns + 3,
        10 * columns + 15,
        40 * columns + 5,
        2 * columns + 3,
    });
    const auto selectionOrder = LastDimensionFastest{};

    GemmTestCase blockedProblem =
        makeProblem(a, b, blockedOutput, rows, reductionElements, columns);
    roc::hostnumerics::detail::bindIndices(blockedProblem, blockedProblem.d.shape(), selection,
                                           selectionOrder);
    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Explicit blocked selection result mismatch.");
    require(run.outputElementsWritten == 4 && run.outputElementsCovered == 4,
            "Explicit blocked selection reported the wrong output counts.");
    requireOnlySelectedOutputsStored(blockedOutput, selection);
}

void testStridedSelectionBlockPlan() {
    constexpr size_t rows = 39;
    constexpr size_t reductionElements = 11;
    constexpr size_t columns = 67;

    const std::vector<float> a = makeValues(rows, reductionElements, 5);
    const std::vector<float> b = makeValues(reductionElements, columns, 6);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const auto selection = IndexRange(3, std::numeric_limits<size_t>::max(), 509);
    const auto selectionOrder = LastDimensionFastest{};

    GemmTestCase blockedProblem =
        makeProblem(a, b, blockedOutput, rows, reductionElements, columns);
    roc::hostnumerics::detail::bindIndices(blockedProblem, blockedProblem.d.shape(), selection,
                                           selectionOrder);
    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Strided blocked selection result mismatch.");
    require(run.outputElementsWritten == 6 && run.outputElementsCovered == 6,
            "Strided blocked selection reported the wrong output counts.");
    requireOnlySelectedOutputsStored(blockedOutput, selection);
}

void testBlockScaledSelectionBlockPlan() {
    using namespace roc::hostnumerics;

    constexpr size_t rows = 33;
    constexpr size_t reductionElements = 16;
    constexpr size_t columns = 35;
    constexpr size_t scaleBlocks = 2;

    const std::vector<float> a = makeValues(rows, reductionElements, 9);
    const std::vector<float> b = makeValues(reductionElements, columns, 10);
    std::vector<float> scaleA(rows * scaleBlocks);
    std::vector<float> scaleB(columns * scaleBlocks);
    for (size_t row = 0; row < rows; ++row) {
        scaleA[row * scaleBlocks] = row % 2 == 0 ? 1.0f : 2.0f;
        scaleA[row * scaleBlocks + 1] = row % 3 == 0 ? 0.5f : 1.0f;
    }
    for (size_t column = 0; column < columns; ++column) {
        scaleB[column * scaleBlocks] = column % 2 == 0 ? 2.0f : 0.5f;
        scaleB[column * scaleBlocks + 1] = column % 3 == 0 ? 1.0f : 4.0f;
    }

    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const auto selection = std::set<size_t>({0, 32 * columns + 32, 32 * columns + 34});
    const auto selectionOrder = LastDimensionFastest{};

    GemmTestCase blockedProblem =
        makeProblem(a, b, blockedOutput, rows, reductionElements, columns);
    const Tensor blockScaleA = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{rows, scaleBlocks}),
        std::span<const float>(scaleA));
    const Tensor blockScaleB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{columns, scaleBlocks}),
        std::span<const float>(scaleB));
    blockedProblem.blockScaleA = blockScaleA;
    blockedProblem.blockSizeA = 8;
    blockedProblem.blockScaleB = blockScaleB;
    blockedProblem.blockSizeB = 8;
    roc::hostnumerics::detail::bindIndices(blockedProblem, blockedProblem.d.shape(), selection,
                                           selectionOrder);

    const GemmTestRunInfo run = runAndCheck(blockedProblem, blockedOutput,
                                            "Block-scaled blocked selection result mismatch.");
    require(run.outputElementsWritten == 3 && run.outputElementsCovered == 3,
            "Block-scaled blocked selection reported the wrong output counts.");
    requireOnlySelectedOutputsStored(blockedOutput, selection);
}

}  // namespace blocked_backend_test
