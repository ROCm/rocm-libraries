// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <roc/host_validation/backends/blocked.hpp>
#include <roc/host_validation/comparison.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using roc::host_validation::BlockedGemmBackend;
using roc::host_validation::GemmRequest;
using roc::host_validation::GemmRunInfo;
using roc::host_validation::OutputSelection;
using roc::host_validation::Tensor;

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
                        const std::vector<float>& c, roc::host_validation::Tensor d, size_t rows,
                        size_t reductionElements, size_t columns) {
    using namespace roc::host_validation;

    return GemmRequest(
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{rows, reductionElements}),
                                              std::span<const float>(a))),
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{reductionElements, columns}),
                                              std::span<const float>(b))),
        Tensor::fromNative<float>(Layout::contiguous(Shape{rows, columns}),
                                  std::span<const float>(c)),
        std::move(d), ScalarType::Float32);
}

roc::host_validation::Tensor makeOutput(size_t rows, size_t columns, float value) {
    using namespace roc::host_validation;
    std::vector<float> values(rows * columns, value);
    return Tensor::fromNativeValues<float>(Shape{rows, columns}, values);
}

void fillTensor(const roc::host_validation::Tensor& tensor, float value) {
    std::vector<size_t> indices(tensor.shape().rank(), 0);
    for (size_t linearIndex = 0; linearIndex < tensor.size(); ++linearIndex) {
        tensor.storeFrom(indices, value);
        for (size_t dimension = tensor.shape().rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < tensor.shape()[index]) break;
            indices[index] = 0;
        }
    }
}

void configureFinalizer(GemmRequest& problem, const std::vector<float>& columnBias) {
    using namespace roc::host_validation;

    problem.epilogue.alpha = 1.25;
    problem.epilogue.beta = -0.5;
    problem.epilogue.bias = VectorBinding{
        Tensor::fromNative<float>(Layout::contiguous(Shape{columnBias.size()}),
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
                        const roc::host_validation::Tensor& pointwiseOutput,
                        const roc::host_validation::Tensor& blockedOutput,
                        const char* mismatchMessage) {
    using namespace roc::host_validation;

    BlockedGemmBackend backend;
    const GemmResult pointwise =
        referenceGemm(pointwiseProblem, {
                                            .backend = GemmBackend::Pointwise,
                                            .requireRequestedBackend = true,
                                        });
    const GemmResult blocked = referenceGemm(blockedProblem,
                                             {
                                                 .backend = GemmBackend::Blocked,
                                                 .requireRequestedBackend = true,
                                             },
                                             &backend);
    require(blocked.runInfo.backendUsed == GemmBackend::Blocked,
            "Blocked backend run information mismatch.");
    require(compare(blockedOutput, pointwiseOutput).passed(), mismatchMessage);
    return {
        .pointwise = pointwise.runInfo,
        .blocked = blocked.runInfo,
    };
}

void requireOnlySelectedOutputsStored(const roc::host_validation::Tensor& output,
                                      const OutputSelection& selection) {
    std::vector<size_t> selected = selection.indices(output.size());
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    for (size_t index = 0; index < output.size(); ++index) {
        if (!std::binary_search(selected.begin(), selected.end(), index))
            require(output.loadAs<float>({index / output.shape()[1], index % output.shape()[1]}) ==
                        untouchedValue,
                    "Blocked backend modified an unselected output element.");
    }
}

void testFinalizerAndSmallEdgeBlock() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    const std::array<float, 2> bias{1, -1000};
    Tensor d(ScalarType::Float32, Shape{2, 2});
    BlockedGemmBackend backend;

    GemmRequest problem(
        GemmOperand(
            Tensor::fromNative<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a))),
        GemmOperand(
            Tensor::fromNative<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b))),
        Tensor::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)),
        d.alias(Layout(Shape{2, 2}, {1, 2})), ScalarType::Float32);
    problem.epilogue.alpha = 2;
    problem.epilogue.beta = 3;
    problem.epilogue.bias = VectorBinding{
        Tensor::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.epilogue.activation = Activation::Relu;

    require(queryGemmSupport(problem,
                             {
                                 .backend = GemmBackend::Blocked,
                                 .requireRequestedBackend = true,
                             },
                             &backend)
                .supported,
            "Blocked backend unexpectedly rejected the test GEMM.");
    const GemmResult full = referenceGemm(problem,
                                          {
                                              .backend = GemmBackend::Blocked,
                                              .requireRequestedBackend = true,
                                          },
                                          &backend);
    require(full.runInfo.outputElementsWritten == 4 && full.runInfo.outputElementsCovered == 4,
            "Full blocked GEMM reported the wrong output counts.");
    const Tensor expected =
        Tensor::fromNativeValues<float>(Shape{2, 2}, std::array<float, 4>{120, 0, 132, 0});
    require(compare(d, expected).passed(), "Blocked backend result mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    const GemmResult selected = referenceGemm(problem,
                                              {
                                                  .backend = GemmBackend::Blocked,
                                                  .requireRequestedBackend = true,
                                              },
                                              &backend);
    require(
        selected.runInfo.outputElementsWritten == 1 && selected.runInfo.outputElementsCovered == 4,
        "Selected blocked GEMM reported the wrong write or coverage count.");
    require(d.loadAs<float>({0, 0}) == 120 && d.loadAs<float>({0, 1}) == untouchedValue &&
                d.loadAs<float>({1, 0}) == untouchedValue &&
                d.loadAs<float>({1, 1}) == untouchedValue,
            "Blocked backend partial output selection mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({});
    const GemmResult empty = referenceGemm(problem,
                                           {
                                               .backend = GemmBackend::Blocked,
                                               .requireRequestedBackend = true,
                                           },
                                           &backend);
    require(empty.runInfo.outputElementsWritten == 0 && empty.runInfo.outputElementsCovered == 0,
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
    using namespace roc::host_validation;

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
        Tensor::fromNative<float>(Layout::contiguous(Shape{rows, scaleBlocks}),
                                  std::span<const float>(scaleA)),
        8,
    };
    const BlockScaleBinding blockScaleB{
        Tensor::fromNative<float>(Layout::contiguous(Shape{columns, scaleBlocks}),
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
}  // namespace

int main() {
    testFinalizerAndSmallEdgeBlock();
    testExplicitSelectionBlockPlan();
    testStridedSelectionBlockPlan();
    testBlockScaledSelectionBlockPlan();
    testFullSelectionParity();
    return 0;
}
