// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <span>
#include <stdexcept>
#include <vector>

#include "gemm_test_adapter.hpp"

namespace {
using roc::host_numerics::GemmTestCase;
using roc::host_numerics::GemmTestRunInfo;
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

GemmTestCase makeProblem(const std::vector<float>& a, const std::vector<float>& b,
                         const std::vector<float>& c, roc::host_numerics::Tensor d, size_t rows,
                         size_t reductionElements, size_t columns) {
    using namespace roc::host_numerics;

    return GemmTestCase(Tensor::copyNativeStorage<float>(
                            Layout::contiguousLastDimensionFastest(Shape{rows, reductionElements}),
                            std::span<const float>(a)),
                        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(
                                                             Shape{reductionElements, columns}),
                                                         std::span<const float>(b)),
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

void configureFinalizer(GemmTestCase& problem, const std::vector<float>& columnBias) {
    using namespace roc::host_numerics;

    problem.alpha = 1.25;
    problem.beta = -0.5;
    problem.bias = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{columnBias.size()}),
        std::span<const float>(columnBias));
    problem.activation = Activation::Relu;
}

Tensor expectedFloatResult(const GemmTestCase& problem) {
    using namespace roc::host_numerics;

    const size_t rows = problem.a.shape()[0];
    const size_t reductions = problem.a.shape()[1];
    const size_t columns = problem.b.shape()[1];
    Tensor expected = makeOutput(rows, columns, untouchedValue);
    const auto selected = problem.outputSelection.indices(problem.d.elementCount());
    const auto scaleValue = [&](const std::optional<Tensor>& scale, size_t row, size_t column) {
        return scale ? scale->broadcastTo(Shape{rows, columns}).loadAs<float>({row, column}) : 1.0f;
    };

    for (const size_t linearIndex : selected) {
        const auto coordinates =
            problem.d.shape().coordinates(linearIndex, problem.outputSelection.indexOrder());
        const size_t row = coordinates[0];
        const size_t column = coordinates[1];
        float accumulation = 0.0f;
        for (size_t blockBase = 0; blockBase < reductions;) {
            const size_t remainingA = problem.blockScaleA
                                          ? problem.blockSizeA - blockBase % problem.blockSizeA
                                          : reductions - blockBase;
            const size_t remainingB = problem.blockScaleB
                                          ? problem.blockSizeB - blockBase % problem.blockSizeB
                                          : reductions - blockBase;
            const size_t blockEnd =
                blockBase + std::min({reductions - blockBase, remainingA, remainingB});
            float partial = 0.0f;
            for (size_t reduction = blockBase; reduction < blockEnd; ++reduction) {
                float aValue = problem.a.loadAs<float>({row, reduction});
                float bValue = problem.b.loadAs<float>({reduction, column});
                for (const Tensor& scale : problem.preQuantizationScalesA)
                    aValue *= scale.broadcastTo(problem.a.shape()).loadAs<float>({row, reduction});
                for (const Tensor& scale : problem.preQuantizationScalesB)
                    bValue *=
                        scale.broadcastTo(problem.b.shape()).loadAs<float>({reduction, column});
                partial += aValue * bValue;
            }
            if (problem.blockScaleA)
                partial *=
                    problem.blockScaleA->loadAs<float>({row, blockBase / problem.blockSizeA});
            if (problem.blockScaleB)
                partial *=
                    problem.blockScaleB->loadAs<float>({column, blockBase / problem.blockSizeB});
            accumulation += partial;
            blockBase = blockEnd;
        }

        float result = problem.alpha.as<float>() * scaleValue(problem.scaleA, row, column) *
                           scaleValue(problem.scaleB, row, column) *
                           scaleValue(problem.scaleAlpha, row, column) * accumulation +
                       problem.beta.as<float>() * problem.scaleC.as<float>() *
                           problem.c.loadAs<float>({row, column});
        if (problem.bias)
            result += problem.bias->broadcastTo(Shape{rows, columns}).loadAs<float>({row, column});
        if (problem.activation == Activation::Relu) result = std::max(0.0f, result);
        require(problem.activation == Activation::None || problem.activation == Activation::Relu,
                "The test-only float oracle received an unsupported activation.");
        result *= problem.outputScale.as<float>();
        expected.storeFrom({row, column}, result);
    }
    return expected;
}

GemmTestRunInfo runAndCheck(GemmTestCase& problem, const Tensor& output,
                            const char* mismatchMessage) {
    using namespace roc::host_numerics;

    const Tensor expected = expectedFloatResult(problem);
    const GemmTestRunInfo run = referenceGemm(problem, GemmBackend::Blocked);
    require(run.backendUsed == GemmBackend::Blocked, "Blocked backend run information mismatch.");
    require(compare(output, expected).passed(), mismatchMessage);
    return run;
}

float roundThrough(roc::host_numerics::ScalarType type, float value) {
    using namespace roc::host_numerics;
    Tensor scalar(type, Shape{1});
    scalar.storeFrom({0}, value);
    return scalar.loadAs<float>({0});
}

void testReducedPrecisionAccumulators() {
    using namespace roc::host_numerics;

    constexpr size_t reductions = 64;
    const std::vector<float> values(reductions, 0.1f);
    for (const ScalarType type : {ScalarType::Float16, ScalarType::BFloat16}) {
        Tensor a = Tensor::copyValuesWithConversion(type, Shape{1, reductions},
                                                    std::span<const float>(values));
        Tensor b = Tensor::copyValuesWithConversion(type, Shape{reductions, 1},
                                                    std::span<const float>(values));
        Tensor c(type, Shape{1, 1});
        Tensor d(type, Shape{1, 1});
        GemmTestCase problem(a, b, c, d, type);

        float expected = 0.0f;
        float fullPrecision = 0.0f;
        for (size_t reduction = 0; reduction < reductions; ++reduction) {
            const float product = roundThrough(
                type, a.loadAs<float>({0, reduction}) * b.loadAs<float>({reduction, 0}));
            expected = roundThrough(type, expected + product);
            fullPrecision += a.loadAs<float>({0, reduction}) * b.loadAs<float>({reduction, 0});
        }

        const GemmTestRunInfo run = referenceGemm(problem, GemmBackend::Blocked);
        require(run.backendUsed == GemmBackend::Blocked && d.loadAs<float>({0, 0}) == expected,
                "Blocked GEMM did not preserve reduced-precision accumulation.");

        problem.accumulationRounding = AccumulationRounding::FullPrecision;
        referenceGemm(problem, GemmBackend::Blocked);
        require(d.loadAs<float>({0, 0}) == roundThrough(type, fullPrecision) &&
                    d.loadAs<float>({0, 0}) != expected,
                "Blocked GEMM did not honor full-precision accumulation.");
    }
}

void testSelectedBlockAccumulatorFamilies() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 32;
    constexpr size_t columns = 32;
    constexpr size_t reductions = 64;
    const OutputSelection selection = OutputSelection::explicitIndices({0});

    const std::vector<float> reducedA(rows * reductions, 0.1f);
    const std::vector<float> reducedB(reductions * columns, 0.1f);
    const std::vector<float> reducedC(rows * columns, 0.0f);
    for (const ScalarType accumulatorType : {ScalarType::Float16, ScalarType::BFloat16}) {
        Tensor output = makeOutput(rows, columns, untouchedValue);
        GemmTestCase problem =
            makeProblem(reducedA, reducedB, reducedC, output, rows, reductions, columns);
        problem.accumulatorType = accumulatorType;
        problem.outputSelection = selection;

        float expected = 0.0f;
        for (size_t reduction = 0; reduction < reductions; ++reduction) {
            const float product = roundThrough(accumulatorType, 0.1f * 0.1f);
            expected = roundThrough(accumulatorType, expected + product);
        }
        const GemmTestRunInfo run = referenceGemm(problem, GemmBackend::Blocked);
        require(run.outputElementsWritten == 1 && run.outputElementsCovered == rows * columns &&
                    output.loadAs<float>({0, 0}) == expected &&
                    output.loadAs<float>({0, 1}) == untouchedValue,
                "Selected blocked GEMM mishandled reduced-precision accumulation.");
    }

    const std::vector<int32_t> integerA(rows * 2, std::numeric_limits<int32_t>::max());
    const std::vector<int32_t> integerB(2 * columns, 2);
    const std::vector<int32_t> integerC(rows * columns, 0);
    const std::vector<int32_t> integerInitial(rows * columns, -99);
    Tensor integerOutput = Tensor::copyNativeValues<int32_t>(Shape{rows, columns}, integerInitial);
    GemmTestCase integerProblem(Tensor::copyNativeValues<int32_t>(Shape{rows, 2}, integerA),
                                Tensor::copyNativeValues<int32_t>(Shape{2, columns}, integerB),
                                Tensor::copyNativeValues<int32_t>(Shape{rows, columns}, integerC),
                                integerOutput, ScalarType::Int32);
    integerProblem.outputSelection = selection;
    const GemmTestRunInfo integerRun = referenceGemm(integerProblem, GemmBackend::Blocked);
    const int32_t wrappedProduct =
        std::bit_cast<int32_t>(static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) * 2U);
    const int32_t wrappedSum = std::bit_cast<int32_t>(static_cast<uint32_t>(wrappedProduct) * 2U);
    require(integerRun.outputElementsCovered == rows * columns &&
                integerOutput.loadAs<int32_t>({0, 0}) == wrappedSum &&
                integerOutput.loadAs<int32_t>({0, 1}) == -99,
            "Selected blocked GEMM mishandled Int32 wrapping accumulation.");

    using Complex = std::complex<float>;
    const std::vector<Complex> complexA(rows * 2, Complex(1.0f, 2.0f));
    const std::vector<Complex> complexB(2 * columns, Complex(3.0f, 4.0f));
    const std::vector<Complex> complexC(rows * columns, Complex(0.0f, 0.0f));
    const std::vector<Complex> complexInitial(rows * columns, Complex(-99.0f, -99.0f));
    Tensor complexOutput = Tensor::copyNativeValues<Complex>(Shape{rows, columns}, complexInitial);
    GemmTestCase complexProblem(Tensor::copyNativeValues<Complex>(Shape{rows, 2}, complexA),
                                Tensor::copyNativeValues<Complex>(Shape{2, columns}, complexB),
                                Tensor::copyNativeValues<Complex>(Shape{rows, columns}, complexC),
                                complexOutput, ScalarType::ComplexFloat32);
    complexProblem.outputSelection = selection;
    const GemmTestRunInfo complexRun = referenceGemm(complexProblem, GemmBackend::Blocked);
    require(complexRun.outputElementsCovered == rows * columns &&
                complexOutput.loadAs<Complex>({0, 0}) == Complex(-10.0f, 20.0f) &&
                complexOutput.loadAs<Complex>({0, 1}) == Complex(-99.0f, -99.0f),
            "Selected blocked GEMM mishandled complex accumulation.");
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
    GemmTestCase problem(
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b)),
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)),
        d.shareStorageWithLayout(Layout(Shape{2, 2}, {1, 2})), ScalarType::Float32);
    problem.alpha = 2;
    problem.beta = 3;
    problem.bias =
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2}),
                                         std::span<const float>(bias))
            .expandDims(1);
    problem.activation = Activation::Relu;

    require(queryGemmSupport(problem, GemmBackend::Blocked).supported,
            "Blocked backend unexpectedly rejected the test GEMM.");
    const GemmTestRunInfo full = referenceGemm(problem, GemmBackend::Blocked);
    require(full.outputElementsWritten == 4 && full.outputElementsCovered == 4,
            "Full blocked GEMM reported the wrong output counts.");
    const Tensor expected =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{120, 0, 132, 0});
    require(compare(d, expected).passed(), "Blocked backend result mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    const GemmTestRunInfo selected = referenceGemm(problem, GemmBackend::Blocked);
    require(selected.outputElementsWritten == 1 && selected.outputElementsCovered == 4,
            "Selected blocked GEMM reported the wrong write or coverage count.");
    require(d.loadAs<float>({0, 0}) == 120 && d.loadAs<float>({0, 1}) == untouchedValue &&
                d.loadAs<float>({1, 0}) == untouchedValue &&
                d.loadAs<float>({1, 1}) == untouchedValue,
            "Blocked backend partial output selection mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({});
    const GemmTestRunInfo empty = referenceGemm(problem, GemmBackend::Blocked);
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
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection = OutputSelection::explicitIndices({
        44 * columns + 69,
        2 * columns + 3,
        10 * columns + 15,
        40 * columns + 5,
        2 * columns + 3,
    });

    GemmTestCase blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
    blockedProblem.outputSelection = selection;
    configureFinalizer(blockedProblem, bias);

    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Explicit blocked selection result mismatch.");
    require(run.outputElementsWritten == 4 && run.outputElementsCovered == 1518,
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
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection = OutputSelection::strided(3, 509);

    GemmTestCase blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
    blockedProblem.outputSelection = selection;
    configureFinalizer(blockedProblem, bias);

    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Strided blocked selection result mismatch.");
    require(run.outputElementsWritten == 6 && run.outputElementsCovered == 2272,
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

    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection =
        OutputSelection::explicitIndices({0, 32 * columns + 32, 32 * columns + 34});

    GemmTestCase blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
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
    blockedProblem.outputSelection = selection;

    const GemmTestRunInfo run = runAndCheck(blockedProblem, blockedOutput,
                                            "Block-scaled blocked selection result mismatch.");
    require(run.outputElementsWritten == 3 && run.outputElementsCovered == 1027,
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
    Tensor blockedOutput = makeOutput(1, 1, untouchedValue);

    GemmTestCase blockedProblem = makeProblem(a, b, c, blockedOutput, 1, reductionElements, 1);
    const Tensor blockScaleA = Tensor::copyNativeValues<float>(Shape{1, 1}, scaleA);
    blockedProblem.blockScaleA = blockScaleA;
    blockedProblem.blockSizeA = reductionElements;

    runAndCheck(blockedProblem, blockedOutput,
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
    const Tensor blockScale = Tensor::copyNativeValues<float>(Shape{2, 2}, scales);

    const auto checkOneSide = [&](bool scaleOperandA, const std::array<float, 4>& expected) {
        Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
        GemmTestCase blockedProblem =
            makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
        if (scaleOperandA) {
            blockedProblem.blockScaleA = blockScale;
            blockedProblem.blockSizeA = 8;
        } else {
            blockedProblem.blockScaleB = blockScale;
            blockedProblem.blockSizeB = 8;
        }

        runAndCheck(blockedProblem, blockedOutput, "One-sided block scaling result mismatch.");
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
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    GemmTestCase blockedProblem = makeProblem(empty, empty, c, blockedOutput, rows, 0, columns);
    const Tensor emptyScale(ScalarType::Float32, Shape{rows, 0});
    blockedProblem.blockScaleA = emptyScale;
    blockedProblem.blockSizeA = 8;
    blockedProblem.beta = 2.0f;

    runAndCheck(blockedProblem, blockedOutput,
                "One-sided block scaling read scales for an empty reduction dimension.");
    require(compare(blockedOutput,
                    Tensor::copyNativeValues<float>(Shape{rows, columns},
                                                    std::array<float, 4>{2.0f, 4.0f, 6.0f, 8.0f}))
                .passed(),
            "One-sided zero-K block scaling did not apply beta to C.");
}

void testFullSelection() {
    constexpr size_t rows = 35;
    constexpr size_t reductionElements = 17;
    constexpr size_t columns = 34;

    const std::vector<float> a = makeValues(rows, reductionElements, 11);
    const std::vector<float> b = makeValues(reductionElements, columns, 12);
    const std::vector<float> c = makeValues(rows, columns, 13);
    const std::vector<float> bias = makeValues(1, columns, 14);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);

    GemmTestCase blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);
    configureFinalizer(blockedProblem, bias);

    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Full blocked selection result mismatch.");
    require(
        run.outputElementsWritten == rows * columns && run.outputElementsCovered == rows * columns,
        "Full blocked selection did not preserve complete-output accounting.");
}

void testAutomaticSelectionUsesBlockedBackend() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 32;
    constexpr size_t reductionElements = 8;
    constexpr size_t columns = 32;
    const std::vector<float> a = makeValues(rows, reductionElements, 18);
    const std::vector<float> b = makeValues(reductionElements, columns, 19);
    const std::vector<float> c(rows * columns, 0.0f);
    Tensor output = makeOutput(rows, columns, untouchedValue);
    GemmTestCase problem = makeProblem(a, b, c, output, rows, reductionElements, columns);

    const GemmSupportInfo fullSupport = queryGemmSupport(problem, GemmBackend::Blocked);
    require(fullSupport.supported, "Blocked backend rejected dense work.");
    const GemmTestRunInfo full = referenceGemm(problem);
    require(full.backendUsed == GemmBackend::Blocked && !full.fallbackReason,
            "Automatic GEMM did not select Blocked for dense reusable work.");
    const float firstOutput = output.loadAs<float>({0, 0});

    fillTensor(output, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    const GemmSupportInfo sparseSupport = queryGemmSupport(problem, GemmBackend::Blocked);
    require(sparseSupport.supported, "Blocked backend rejected sparse work.");
    const GemmTestRunInfo sparse = referenceGemm(problem);
    require(sparse.backendUsed == GemmBackend::Blocked && !sparse.fallbackReason,
            "Automatic GEMM did not use Blocked for sparse work.");
    require(sparse.outputElementsWritten == 1 && sparse.outputElementsCovered == rows * columns,
            "Sparse blocked GEMM reported the wrong write or coverage count.");
    require(output.loadAs<float>({0, 0}) == firstOutput,
            "Automatic backend selection changed the selected numerical result.");
    requireOnlySelectedOutputsStored(output, problem.outputSelection);
}

void testParallelFullSelection() {
    constexpr size_t rows = 128;
    constexpr size_t reductionElements = 128;
    constexpr size_t columns = 128;

    const std::vector<float> a = makeValues(rows, reductionElements, 15);
    const std::vector<float> b = makeValues(reductionElements, columns, 16);
    const std::vector<float> c = makeValues(rows, columns, 17);
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);

    GemmTestCase blockedProblem =
        makeProblem(a, b, c, blockedOutput, rows, reductionElements, columns);

    const GemmTestRunInfo run =
        runAndCheck(blockedProblem, blockedOutput, "Parallel blocked GEMM result mismatch.");
    require(
        run.outputElementsWritten == rows * columns && run.outputElementsCovered == rows * columns,
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
    GemmTestCase problem = makeProblem(a, b, c, output, rows, reductionElements, columns);

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

    GemmTestCase exactCAndD =
        makeProblem(a, b, c, makeOutput(extent, extent, 0), extent, extent, extent);
    exactCAndD.d = exactCAndD.c;
    require(static_cast<bool>(queryGemmSupport(exactCAndD, GemmBackend::Blocked)),
            "GEMM rejected an exact in-place C and D tensor.");
    exactCAndD.beta = 1.0;
    referenceGemm(exactCAndD, GemmBackend::Blocked);
    const Tensor expectedCAndD =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{28, 32, 54, 62});
    require(compare(exactCAndD.d, expectedCAndD).passed(),
            "Blocked GEMM mishandled an exact in-place C and D tensor.");

    GemmTestCase overlapsA =
        makeProblem(a, b, c, makeOutput(extent, extent, 0), extent, extent, extent);
    overlapsA.d = overlapsA.a;
    require(!queryGemmSupport(overlapsA, GemmBackend::Blocked),
            "GEMM accepted destination storage that overlaps A.");

    GemmTestCase differentlyMappedCAndD =
        makeProblem(a, b, c, makeOutput(extent, extent, 0), extent, extent, extent);
    differentlyMappedCAndD.d =
        differentlyMappedCAndD.c.shareStorageWithLayout(Layout(Shape{extent, extent}, {1, 2}));
    require(!queryGemmSupport(differentlyMappedCAndD, GemmBackend::Blocked),
            "GEMM accepted differently mapped overlapping C and D tensors.");
}

void testParallelSparseSelection() {
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
    GemmTestCase problem = makeProblem(a, b, c, output, rows, reductionElements, columns);
    problem.outputSelection =
        OutputSelection::primeStride(output.elementCount(), output.elementCount(), 128);

    const GemmTestRunInfo run = referenceGemm(problem, GemmBackend::Blocked);

    const std::vector<size_t> selected = problem.outputSelection.indices(output.elementCount());
    require(
        run.outputElementsWritten == selected.size() && run.outputElementsCovered == rows * columns,
        "Sparse blocked GEMM did not report its touched output blocks.");
    for (const size_t linearIndex : selected) {
        const size_t row = linearIndex / columns;
        const size_t column = linearIndex % columns;
        require(output.loadAs<float>({row, column}) ==
                    static_cast<float>((row + 1) * reductionElements),
                "Parallel blocked GEMM produced an incorrect selected output.");
    }
    require(output.loadAs<float>({0, 1}) == sentinel,
            "Parallel blocked GEMM changed an unselected output.");
}
}  // namespace

int main() {
    testReducedPrecisionAccumulators();
    testSelectedBlockAccumulatorFamilies();
    testFinalizerAndSmallEdgeBlock();
    testExplicitSelectionBlockPlan();
    testStridedSelectionBlockPlan();
    testBlockScaledSelectionBlockPlan();
    testBlockScaleAppliedAfterCompleteScaleSegment();
    testOneSidedBlockScaling();
    testOneSidedBlockScalingWithZeroReductionExtent();
    testFullSelection();
    testAutomaticSelectionUsesBlockedBackend();
    testParallelFullSelection();
    testOverlappingOutputIsRejectedAcrossBackends();
    testOutputAliasingContract();
    testParallelSparseSelection();
    return 0;
}
