// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using roc::host_numerics::GemmRequest;
using roc::host_numerics::GemmRunInfo;
using roc::host_numerics::OutputSelection;
using roc::host_numerics::Tensor;

constexpr float untouchedValue = -12345.0f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> makeValues(size_t rows, size_t columns, size_t seed) {
    std::vector<float> values(rows * columns);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            const int encoded = static_cast<int>((row * 11 + column * 7 + seed * 5) % 17) - 8;
            values[row * columns + column] = static_cast<float>(encoded) * 0.25f;
        }
    }
    return values;
}

GemmRequest makeProblem(const std::vector<float>& a, const std::vector<float>& b,
                        const std::vector<float>& c, roc::host_numerics::Tensor d, size_t rows,
                        size_t reductionElements, size_t columns) {
    using namespace roc::host_numerics;

    return GemmRequest(
        GemmOperand(Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{rows, reductionElements}),
            std::span<const float>(a))),
        GemmOperand(Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{reductionElements, columns}),
            std::span<const float>(b))),
        Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{rows, columns}),
            std::span<const float>(c)),
        std::move(d), ScalarType::Float32);
}

roc::host_numerics::Tensor makeOutput(size_t rows, size_t columns, float value) {
    using namespace roc::host_numerics;
    std::vector<float> values(rows * columns, value);
    return Tensor::copyNativeValues<float>(Shape{rows, columns}, values);
}

void fillTensor(const roc::host_numerics::Tensor& tensor, float value) {
    std::vector<size_t> indices(tensor.shape().rank(), 0);
    for (size_t linearIndex = 0; linearIndex < tensor.elementCount(); ++linearIndex) {
        tensor.storeFrom(indices, value);
        for (size_t dimension = tensor.shape().rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < tensor.shape()[index]) break;
            indices[index] = 0;
        }
    }
}

void configureFinalizer(GemmRequest& problem, const std::vector<float>& columnBias) {
    using namespace roc::host_numerics;

    problem.epilogue.alpha = 1.25;
    problem.epilogue.beta = -0.5;
    problem.epilogue.bias = VectorBinding{
        Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{columnBias.size()}),
            std::span<const float>(columnBias)),
        MatrixAxis::Column,
    };
    problem.epilogue.activation = Activation::Relu;
}

struct ParityRunInfo {
    GemmRunInfo pointwise;
    GemmRunInfo blocked;
};

ParityRunInfo runParity(GemmRequest& pointwiseProblem, GemmRequest& blockedProblem,
                        const roc::host_numerics::Tensor& pointwiseOutput,
                        const roc::host_numerics::Tensor& blockedOutput,
                        const char* mismatchMessage) {
    using namespace roc::host_numerics;

    const GemmRunInfo pointwise = referenceGemm(pointwiseProblem, GemmBackend::Pointwise);
    const GemmRunInfo blocked = referenceGemm(blockedProblem, GemmBackend::Blocked);
    require(blocked.backendUsed == GemmBackend::Blocked,
            "Blocked backend run information mismatch.");
    require(compare(blockedOutput, pointwiseOutput).passed(), mismatchMessage);
    return {
        .pointwise = pointwise,
        .blocked = blocked,
    };
}

void requireOnlySelectedOutputsStored(const roc::host_numerics::Tensor& output,
                                      const OutputSelection& selection) {
    std::vector<size_t> selected = selection.indices(output.elementCount());
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    for (size_t index = 0; index < output.elementCount(); ++index) {
        if (!std::binary_search(selected.begin(), selected.end(), index))
            require(output.loadAs<float>({index / output.shape()[1], index % output.shape()[1]}) ==
                        untouchedValue,
                    "Blocked backend modified an unselected output element.");
    }
}

void testFinalizerAndSmallEdgeBlock() {
    using namespace roc::host_numerics;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    const std::array<float, 2> bias{1, -1000};
    Tensor d(ScalarType::Float32, Shape{2, 2});
    GemmRequest problem(
        GemmOperand(Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 2}),
                                                     std::span<const float>(a))),
        GemmOperand(Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {1, 3}),
                                                     std::span<const float>(b))),
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)),
        d.shareStorageWithLayout(Layout(Shape{2, 2}, {1, 2})), ScalarType::Float32);
    problem.epilogue.alpha = 2;
    problem.epilogue.beta = 3;
    problem.epilogue.bias = VectorBinding{
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2}),
                                         std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.epilogue.activation = Activation::Relu;

    require(queryGemmSupport(problem, GemmBackend::Blocked).supported,
            "Blocked backend unexpectedly rejected the test GEMM.");
    const GemmRunInfo full = referenceGemm(problem, GemmBackend::Blocked);
    require(full.outputElementsWritten == 4 && full.outputElementsCovered == 4,
            "Full blocked GEMM reported the wrong output counts.");
    const Tensor expected =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{120, 0, 132, 0});
    require(compare(d, expected).passed(), "Blocked backend result mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    const GemmRunInfo selected = referenceGemm(problem, GemmBackend::Blocked);
    require(selected.outputElementsWritten == 1 && selected.outputElementsCovered == 4,
            "Selected blocked GEMM reported the wrong write or coverage count.");
    require(d.loadAs<float>({0, 0}) == 120 && d.loadAs<float>({0, 1}) == untouchedValue &&
                d.loadAs<float>({1, 0}) == untouchedValue &&
                d.loadAs<float>({1, 1}) == untouchedValue,
            "Blocked backend partial output selection mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({});
    const GemmRunInfo empty = referenceGemm(problem, GemmBackend::Blocked);
    require(empty.outputElementsWritten == 0 && empty.outputElementsCovered == 0,
            "Empty blocked output selection reported output work.");
    requireOnlySelectedOutputsStored(d, OutputSelection::explicitIndices({}));
}

void testExplicitSelectionBlockPlan() {
    constexpr size_t rows = 45;
    constexpr size_t reductionElements = 16;
    constexpr size_t columns = 70;

    const std::vector<float> a = makeValues(rows, reductionElements, 1);
    const std::vector<float> b = makeValues(reductionElements, columns, 2);
    const std::vector<float> c = makeValues(rows, columns, 3);
    const std::vector<float> bias = makeValues(1, columns, 4);
    Tensor pointwiseOutput = makeOutput(rows, columns, untouchedValue);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection = OutputSelection::explicitIndices({
        44 * columns + 69,
        2 * columns + 3,
        10 * columns + 15,
        40 * columns + 5,
        2 * columns + 3,
    });

    GemmRequest pointwiseProblem =
        makeProblem(a, b, c, pointwiseOutput, rows, reductionElements, columns);
    GemmRequest blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
    pointwiseProblem.outputSelection = selection;
    blockedProblem.outputSelection = selection;
    configureFinalizer(pointwiseProblem, bias);
    configureFinalizer(blockedProblem, bias);

    const ParityRunInfo run =
        runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
                  "Explicit blocked selection differs from the pointwise reference.");
    require(run.pointwise.outputElementsWritten == 4 && run.pointwise.outputElementsCovered == 4,
            "Pointwise explicit selection counts changed unexpectedly.");
    require(run.blocked.outputElementsWritten == 4 && run.blocked.outputElementsCovered == 1518,
            "Explicit blocked selection reported the wrong output counts.");
    requireOnlySelectedOutputsStored(blockedOutput, selection);
}

void testStridedSelectionBlockPlan() {
    constexpr size_t rows = 39;
    constexpr size_t reductionElements = 11;
    constexpr size_t columns = 67;

    const std::vector<float> a = makeValues(rows, reductionElements, 5);
    const std::vector<float> b = makeValues(reductionElements, columns, 6);
    const std::vector<float> c = makeValues(rows, columns, 7);
    const std::vector<float> bias = makeValues(1, columns, 8);
    Tensor pointwiseOutput = makeOutput(rows, columns, untouchedValue);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection = OutputSelection::strided(3, 509);

    GemmRequest pointwiseProblem =
        makeProblem(a, b, c, pointwiseOutput, rows, reductionElements, columns);
    GemmRequest blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
    pointwiseProblem.outputSelection = selection;
    blockedProblem.outputSelection = selection;
    configureFinalizer(pointwiseProblem, bias);
    configureFinalizer(blockedProblem, bias);

    const ParityRunInfo run =
        runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
                  "Strided blocked selection differs from the pointwise reference.");
    require(run.pointwise.outputElementsWritten == 6 && run.pointwise.outputElementsCovered == 6,
            "Pointwise strided selection counts changed unexpectedly.");
    require(run.blocked.outputElementsWritten == 6 && run.blocked.outputElementsCovered == 2272,
            "Strided blocked selection reported the wrong output counts.");
    requireOnlySelectedOutputsStored(blockedOutput, selection);
}

void testBlockScaledSelectionBlockPlan() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 33;
    constexpr size_t reductionElements = 16;
    constexpr size_t columns = 35;
    constexpr size_t scaleBlocks = 2;

    const std::vector<float> a = makeValues(rows, reductionElements, 9);
    const std::vector<float> b = makeValues(reductionElements, columns, 10);
    const std::vector<float> c(rows * columns, 0.0f);
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

    Tensor pointwiseOutput = makeOutput(rows, columns, untouchedValue);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection =
        OutputSelection::explicitIndices({0, 32 * columns + 32, 32 * columns + 34});

    GemmRequest pointwiseProblem =
        makeProblem(a, b, c, pointwiseOutput, rows, reductionElements, columns);
    GemmRequest blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
    const BlockScaleBinding blockScaleA{
        Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{rows, scaleBlocks}),
            std::span<const float>(scaleA)),
        8,
    };
    const BlockScaleBinding blockScaleB{
        Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{columns, scaleBlocks}),
            std::span<const float>(scaleB)),
        8,
    };
    pointwiseProblem.a.blockScale = blockScaleA;
    pointwiseProblem.b.blockScale = blockScaleB;
    blockedProblem.a.blockScale = blockScaleA;
    blockedProblem.b.blockScale = blockScaleB;
    pointwiseProblem.outputSelection = selection;
    blockedProblem.outputSelection = selection;

    const ParityRunInfo run =
        runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
                  "Block-scaled blocked selection differs from the pointwise reference.");
    require(run.pointwise.outputElementsWritten == 3 && run.pointwise.outputElementsCovered == 3,
            "Pointwise block-scaled selection counts changed unexpectedly.");
    require(run.blocked.outputElementsWritten == 3 && run.blocked.outputElementsCovered == 1027,
            "Block-scaled blocked selection reported the wrong output counts.");
    requireOnlySelectedOutputsStored(blockedOutput, selection);
}

void testBlockScaleAppliedAfterCompleteScaleSegment() {
    using namespace roc::host_numerics;

    constexpr size_t reductionElements = 16;
    const std::vector<float> a(reductionElements, 1.0f);
    std::vector<float> b(reductionElements, 1.0e37f);
    std::fill(b.begin() + reductionElements / 2, b.end(), -1.0e37f);
    const std::vector<float> c(1, 0.0f);
    const std::array<float, 1> scaleA{8.0f};
    Tensor pointwiseOutput = makeOutput(1, 1, untouchedValue);
    Tensor blockedOutput = makeOutput(1, 1, untouchedValue);

    GemmRequest pointwiseProblem = makeProblem(a, b, c, pointwiseOutput, 1, reductionElements, 1);
    GemmRequest blockedProblem = makeProblem(a, b, c, blockedOutput, 1, reductionElements, 1);
    const BlockScaleBinding blockScaleA{Tensor::copyNativeValues<float>(Shape{1, 1}, scaleA),
                                        reductionElements};
    pointwiseProblem.a.blockScale = blockScaleA;
    blockedProblem.a.blockScale = blockScaleA;

    runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
              "Blocked GEMM applied a one-sided block scale before its complete reduction "
              "segment.");
    require(std::isfinite(blockedOutput.loadAs<float>({0, 0})),
            "Blocked GEMM overflowed scale-segment partial sums.");
}

void testOneSidedBlockScaling() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 2;
    constexpr size_t reductionElements = 16;
    constexpr size_t columns = 2;
    const std::vector<float> a(rows * reductionElements, 1.0f);
    const std::vector<float> b(reductionElements * columns, 1.0f);
    const std::vector<float> c(rows * columns, 0.0f);
    const std::array<float, 4> scales{2.0f, 3.0f, 4.0f, 5.0f};
    const BlockScaleBinding blockScale{
        Tensor::copyNativeValues<float>(Shape{2, 2}, scales),
        8,
    };

    const auto checkOneSide = [&](bool scaleOperandA, const std::array<float, 4>& expected) {
        Tensor pointwiseOutput = makeOutput(rows, columns, untouchedValue);
        Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
        GemmRequest pointwiseProblem =
            makeProblem(a, b, c, pointwiseOutput, rows, reductionElements, columns);
        GemmRequest blockedProblem =
            makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
        if (scaleOperandA) {
            pointwiseProblem.a.blockScale = blockScale;
            blockedProblem.a.blockScale = blockScale;
        } else {
            pointwiseProblem.b.blockScale = blockScale;
            blockedProblem.b.blockScale = blockScale;
        }

        runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
                  "One-sided block scaling differs between Pointwise and Blocked GEMM.");
        require(
            compare(blockedOutput, Tensor::copyNativeValues<float>(Shape{rows, columns}, expected))
                .passed(),
            "One-sided block scaling produced an incorrect result.");
    };

    checkOneSide(true, {40.0f, 40.0f, 72.0f, 72.0f});
    checkOneSide(false, {40.0f, 72.0f, 40.0f, 72.0f});
}

void testOneSidedBlockScalingWithZeroReductionExtent() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 2;
    constexpr size_t columns = 2;
    const std::vector<float> empty;
    const std::vector<float> c{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor pointwiseOutput = makeOutput(rows, columns, untouchedValue);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    GemmRequest pointwiseProblem = makeProblem(empty, empty, c, pointwiseOutput, rows, 0, columns);
    GemmRequest blockedProblem = makeProblem(empty, empty, c, blockedOutput, rows, 0, columns);
    const BlockScaleBinding emptyScale{Tensor(ScalarType::Float32, Shape{rows, 0}), 8};
    pointwiseProblem.a.blockScale = emptyScale;
    blockedProblem.a.blockScale = emptyScale;
    pointwiseProblem.epilogue.beta = 2.0f;
    blockedProblem.epilogue.beta = 2.0f;

    runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
              "One-sided block scaling read scales for an empty reduction dimension.");
    require(compare(blockedOutput,
                    Tensor::copyNativeValues<float>(Shape{rows, columns},
                                                    std::array<float, 4>{2.0f, 4.0f, 6.0f, 8.0f}))
                .passed(),
            "One-sided zero-K block scaling did not apply beta to C.");
}

void testFullSelectionParity() {
    constexpr size_t rows = 35;
    constexpr size_t reductionElements = 17;
    constexpr size_t columns = 34;

    const std::vector<float> a = makeValues(rows, reductionElements, 11);
    const std::vector<float> b = makeValues(reductionElements, columns, 12);
    const std::vector<float> c = makeValues(rows, columns, 13);
    const std::vector<float> bias = makeValues(1, columns, 14);
    Tensor pointwiseOutput = makeOutput(rows, columns, untouchedValue);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);

    GemmRequest pointwiseProblem =
        makeProblem(a, b, c, pointwiseOutput, rows, reductionElements, columns);
    GemmRequest blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
    configureFinalizer(pointwiseProblem, bias);
    configureFinalizer(blockedProblem, bias);

    const ParityRunInfo run =
        runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
                  "Full blocked selection differs from the pointwise reference.");
    require(run.pointwise.outputElementsWritten == rows * columns &&
                run.pointwise.outputElementsCovered == rows * columns,
            "Pointwise full selection counts changed unexpectedly.");
    require(run.blocked.outputElementsWritten == rows * columns &&
                run.blocked.outputElementsCovered == rows * columns,
            "Full blocked selection did not preserve complete-output accounting.");
}

void testAutomaticSelectionUsesComponentCostPolicy() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 32;
    constexpr size_t reductionElements = 8;
    constexpr size_t columns = 32;
    const std::vector<float> a = makeValues(rows, reductionElements, 18);
    const std::vector<float> b = makeValues(reductionElements, columns, 19);
    const std::vector<float> c(rows * columns, 0.0f);
    Tensor output = makeOutput(rows, columns, untouchedValue);
    GemmRequest problem = makeProblem(a, b, c, output, rows, reductionElements, columns);

    const GemmSupportInfo fullSupport = queryGemmSupport(problem, GemmBackend::Blocked);
    require(fullSupport.supported && fullSupport.preferredForAutomaticExecution,
            "Blocked cost policy rejected dense reusable work.");
    const GemmRunInfo full = referenceGemm(problem);
    require(full.backendUsed == GemmBackend::Blocked && !full.fallbackReason,
            "Automatic GEMM did not select Blocked for dense reusable work.");
    const float firstOutput = output.loadAs<float>({0, 0});

    fillTensor(output, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    const GemmSupportInfo sparseSupport = queryGemmSupport(problem, GemmBackend::Blocked);
    require(sparseSupport.supported && !sparseSupport.preferredForAutomaticExecution,
            "Blocked cost policy preferred one sparse output block.");
    const GemmRunInfo sparse = referenceGemm(problem);
    require(sparse.backendUsed == GemmBackend::Pointwise && !sparse.fallbackReason,
            "Automatic GEMM did not keep sparse work Pointwise.");
    require(output.loadAs<float>({0, 0}) == firstOutput,
            "Automatic backend selection changed the selected numerical result.");
    requireOnlySelectedOutputsStored(output, problem.outputSelection);
}

void testParallelFullSelectionParity() {
    constexpr size_t rows = 128;
    constexpr size_t reductionElements = 128;
    constexpr size_t columns = 128;

    const std::vector<float> a = makeValues(rows, reductionElements, 15);
    const std::vector<float> b = makeValues(reductionElements, columns, 16);
    const std::vector<float> c = makeValues(rows, columns, 17);
    Tensor pointwiseOutput = makeOutput(rows, columns, untouchedValue);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);

    GemmRequest pointwiseProblem =
        makeProblem(a, b, c, pointwiseOutput, rows, reductionElements, columns);
    GemmRequest blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);

    const ParityRunInfo run =
        runParity(pointwiseProblem, blockedProblem, pointwiseOutput, blockedOutput,
                  "Parallel blocked GEMM differs from the pointwise reference.");
    require(run.blocked.outputElementsWritten == rows * columns &&
                run.blocked.outputElementsCovered == rows * columns,
            "Parallel blocked GEMM reported the wrong output counts.");
}

void testOverlappingOutputIsRejectedAcrossBackends() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 33;
    constexpr size_t reductionElements = 512;
    constexpr size_t columns = 64;

    std::vector<float> a(rows * reductionElements);
    for (size_t row = 0; row < rows; ++row)
        std::fill_n(a.begin() + row * reductionElements, reductionElements,
                    static_cast<float>(row + 1));
    const std::vector<float> b(reductionElements * columns, 1.0f);
    const std::vector<float> c(rows * columns, 0.0f);
    Tensor output(ScalarType::Float32, Layout(Shape{rows, columns}, {0, 0}));
    GemmRequest problem = makeProblem(a, b, c, output, rows, reductionElements, columns);

    require(!queryGemmSupport(problem, GemmBackend::Pointwise),
            "Pointwise GEMM accepted overlapping destination elements.");
    require(!queryGemmSupport(problem, GemmBackend::Blocked),
            "Blocked GEMM accepted overlapping destination elements.");
    require(!queryGemmSupport(problem, GemmBackend::Automatic),
            "Automatic GEMM accepted overlapping destination elements.");
}

void testOutputAliasingContract() {
    using namespace roc::host_numerics;

    constexpr size_t extent = 2;
    const std::vector<float> a{1, 2, 3, 4};
    const std::vector<float> b{5, 6, 7, 8};
    const std::vector<float> c{9, 10, 11, 12};

    GemmRequest exactCAndD =
        makeProblem(a, b, c, makeOutput(extent, extent, 0), extent, extent, extent);
    exactCAndD.d = exactCAndD.c;
    require(queryGemmSupport(exactCAndD, GemmBackend::Pointwise) &&
                queryGemmSupport(exactCAndD, GemmBackend::Blocked),
            "GEMM rejected an exact in-place C and D tensor.");
    exactCAndD.epilogue.beta = 1.0;
    referenceGemm(exactCAndD, GemmBackend::Blocked);
    const Tensor expectedCAndD =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{28, 32, 54, 62});
    require(compare(exactCAndD.d, expectedCAndD).passed(),
            "Blocked GEMM mishandled an exact in-place C and D tensor.");

    GemmRequest overlapsA =
        makeProblem(a, b, c, makeOutput(extent, extent, 0), extent, extent, extent);
    overlapsA.d = overlapsA.a.values;
    require(!queryGemmSupport(overlapsA, GemmBackend::Pointwise) &&
                !queryGemmSupport(overlapsA, GemmBackend::Blocked),
            "GEMM accepted destination storage that overlaps A.");

    GemmRequest differentlyMappedCAndD =
        makeProblem(a, b, c, makeOutput(extent, extent, 0), extent, extent, extent);
    differentlyMappedCAndD.d =
        differentlyMappedCAndD.c.shareStorageWithLayout(Layout(Shape{extent, extent}, {1, 2}));
    require(!queryGemmSupport(differentlyMappedCAndD, GemmBackend::Pointwise) &&
                !queryGemmSupport(differentlyMappedCAndD, GemmBackend::Blocked),
            "GEMM accepted differently mapped overlapping C and D tensors.");
}

void testParallelPointwiseSelection() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 64;
    constexpr size_t reductionElements = 4096;
    constexpr size_t columns = 64;
    constexpr float sentinel = -99.0f;

    std::vector<float> a(rows * reductionElements);
    for (size_t row = 0; row < rows; ++row)
        std::fill_n(a.begin() + row * reductionElements, reductionElements,
                    static_cast<float>(row + 1));
    const std::vector<float> b(reductionElements * columns, 1.0f);
    const std::vector<float> c(rows * columns, 0.0f);
    Tensor output = makeOutput(rows, columns, sentinel);
    GemmRequest problem = makeProblem(a, b, c, output, rows, reductionElements, columns);
    problem.outputSelection =
        OutputSelection::primeStride(output.elementCount(), output.elementCount(), 128);

    referenceGemm(problem, GemmBackend::Pointwise);

    const std::vector<size_t> selected = problem.outputSelection.indices(output.elementCount());
    for (const size_t linearIndex : selected) {
        const size_t row = linearIndex / columns;
        const size_t column = linearIndex % columns;
        require(output.loadAs<float>({row, column}) ==
                    static_cast<float>((row + 1) * reductionElements),
                "Parallel pointwise GEMM produced an incorrect selected output.");
    }
    require(output.loadAs<float>({0, 1}) == sentinel,
            "Parallel pointwise GEMM changed an unselected output.");
}
}  // namespace

int main() {
    testFinalizerAndSmallEdgeBlock();
    testExplicitSelectionBlockPlan();
    testStridedSelectionBlockPlan();
    testBlockScaledSelectionBlockPlan();
    testBlockScaleAppliedAfterCompleteScaleSegment();
    testOneSidedBlockScaling();
    testOneSidedBlockScalingWithZeroReductionExtent();
    testFullSelectionParity();
    testAutomaticSelectionUsesComponentCostPolicy();
    testParallelFullSelectionParity();
    testOverlappingOutputIsRejectedAcrossBackends();
    testOutputAliasingContract();
    testParallelPointwiseSelection();
    return 0;
}
