// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <roc/host_validation/backends/tiled.hpp>
#include <roc/host_validation/comparison.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using roc::host_validation::GemmRequest;
using roc::host_validation::GemmRunInfo;
using roc::host_validation::OutputSelection;
using roc::host_validation::Tensor;
using roc::host_validation::TiledGemmBackend;

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
    GemmRunInfo canonical;
    GemmRunInfo tiled;
};

ParityRunInfo runParity(GemmRequest& canonicalProblem, GemmRequest& tiledProblem,
                        const roc::host_validation::Tensor& canonicalOutput,
                        const roc::host_validation::Tensor& tiledOutput,
                        const char* mismatchMessage) {
    using namespace roc::host_validation;

    TiledGemmBackend backend;
    const GemmResult canonical =
        referenceGemm(canonicalProblem, {
                                            .backend = GemmBackend::Canonical,
                                            .requireRequestedBackend = true,
                                        });
    const GemmResult tiled = referenceGemm(tiledProblem,
                                           {
                                               .backend = GemmBackend::Tiled,
                                               .requireRequestedBackend = true,
                                           },
                                           &backend);
    require(tiled.runInfo.backendUsed == GemmBackend::Tiled,
            "Tiled backend run information mismatch.");
    require(compare(tiledOutput, canonicalOutput).passed(), mismatchMessage);
    return {
        .canonical = canonical.runInfo,
        .tiled = tiled.runInfo,
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
                    "Tiled backend modified an unselected output element.");
    }
}

void testFinalizerAndSmallEdgeTile() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    const std::array<float, 2> bias{1, -1000};
    Tensor d(ScalarType::Float32, Shape{2, 2});
    TiledGemmBackend backend;

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
                                 .backend = GemmBackend::Tiled,
                                 .requireRequestedBackend = true,
                             },
                             &backend)
                .supported,
            "Tiled backend unexpectedly rejected the test GEMM.");
    const GemmResult full = referenceGemm(problem,
                                          {
                                              .backend = GemmBackend::Tiled,
                                              .requireRequestedBackend = true,
                                          },
                                          &backend);
    require(full.runInfo.outputElementsComputed == 4,
            "Full tiled GEMM reported the wrong computed output count.");
    const Tensor expected =
        Tensor::fromNativeValues<float>(Shape{2, 2}, std::array<float, 4>{120, 0, 132, 0});
    require(compare(d, expected).passed(), "Tiled backend result mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    const GemmResult selected = referenceGemm(problem,
                                              {
                                                  .backend = GemmBackend::Tiled,
                                                  .requireRequestedBackend = true,
                                              },
                                              &backend);
    require(selected.runInfo.outputElementsComputed == 4,
            "Selected tiled GEMM did not report the clipped tile area.");
    require(d.loadAs<float>({0, 0}) == 120 && d.loadAs<float>({0, 1}) == untouchedValue &&
                d.loadAs<float>({1, 0}) == untouchedValue &&
                d.loadAs<float>({1, 1}) == untouchedValue,
            "Tiled backend partial output selection mismatch.");

    fillTensor(d, untouchedValue);
    problem.outputSelection = OutputSelection::explicitIndices({});
    const GemmResult empty = referenceGemm(problem,
                                           {
                                               .backend = GemmBackend::Tiled,
                                               .requireRequestedBackend = true,
                                           },
                                           &backend);
    require(empty.runInfo.outputElementsComputed == 0,
            "Empty tiled output selection reported computed accumulators.");
    requireOnlySelectedOutputsStored(d, OutputSelection::explicitIndices({}));
}

void testExplicitSelectionTilePlan() {
    constexpr size_t rows = 45;
    constexpr size_t reductionElements = 16;
    constexpr size_t columns = 70;

    const std::vector<float> a = makeValues(rows, reductionElements, 1);
    const std::vector<float> b = makeValues(reductionElements, columns, 2);
    const std::vector<float> c = makeValues(rows, columns, 3);
    const std::vector<float> bias = makeValues(1, columns, 4);
    Tensor canonicalOutput = makeOutput(rows, columns, untouchedValue);
    Tensor tiledOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection = OutputSelection::explicitIndices({
        44 * columns + 69,
        2 * columns + 3,
        10 * columns + 15,
        40 * columns + 5,
        2 * columns + 3,
    });

    GemmRequest canonicalProblem =
        makeProblem(a, b, c, canonicalOutput, rows, reductionElements, columns);
    GemmRequest tiledProblem = makeProblem(a, b, c, tiledOutput, rows, reductionElements, columns);
    canonicalProblem.outputSelection = selection;
    tiledProblem.outputSelection = selection;
    configureFinalizer(canonicalProblem, bias);
    configureFinalizer(tiledProblem, bias);

    const ParityRunInfo run =
        runParity(canonicalProblem, tiledProblem, canonicalOutput, tiledOutput,
                  "Explicit tiled selection differs from the canonical reference.");
    require(run.canonical.outputElementsComputed == 4,
            "Canonical explicit selection count changed unexpectedly.");
    require(run.tiled.outputElementsComputed == 1518,
            "Explicit tiled selection did not count each unique edge-aware tile once.");
    requireOnlySelectedOutputsStored(tiledOutput, selection);
}

void testStridedSelectionTilePlan() {
    constexpr size_t rows = 39;
    constexpr size_t reductionElements = 11;
    constexpr size_t columns = 67;

    const std::vector<float> a = makeValues(rows, reductionElements, 5);
    const std::vector<float> b = makeValues(reductionElements, columns, 6);
    const std::vector<float> c = makeValues(rows, columns, 7);
    const std::vector<float> bias = makeValues(1, columns, 8);
    Tensor canonicalOutput = makeOutput(rows, columns, untouchedValue);
    Tensor tiledOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection = OutputSelection::strided(3, 509);

    GemmRequest canonicalProblem =
        makeProblem(a, b, c, canonicalOutput, rows, reductionElements, columns);
    GemmRequest tiledProblem = makeProblem(a, b, c, tiledOutput, rows, reductionElements, columns);
    canonicalProblem.outputSelection = selection;
    tiledProblem.outputSelection = selection;
    configureFinalizer(canonicalProblem, bias);
    configureFinalizer(tiledProblem, bias);

    const ParityRunInfo run =
        runParity(canonicalProblem, tiledProblem, canonicalOutput, tiledOutput,
                  "Strided tiled selection differs from the canonical reference.");
    require(run.canonical.outputElementsComputed == 6,
            "Canonical strided selection count changed unexpectedly.");
    require(run.tiled.outputElementsComputed == 2272,
            "Strided tiled selection reported the wrong executed tile area.");
    requireOnlySelectedOutputsStored(tiledOutput, selection);
}

void testBlockScaledSelectionTilePlan() {
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

    Tensor canonicalOutput = makeOutput(rows, columns, untouchedValue);
    Tensor tiledOutput = makeOutput(rows, columns, untouchedValue);
    const OutputSelection selection =
        OutputSelection::explicitIndices({0, 32 * columns + 32, 32 * columns + 34});

    GemmRequest canonicalProblem =
        makeProblem(a, b, c, canonicalOutput, rows, reductionElements, columns);
    GemmRequest tiledProblem = makeProblem(a, b, c, tiledOutput, rows, reductionElements, columns);
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
    canonicalProblem.a.blockScale = blockScaleA;
    canonicalProblem.b.blockScale = blockScaleB;
    tiledProblem.a.blockScale = blockScaleA;
    tiledProblem.b.blockScale = blockScaleB;
    canonicalProblem.outputSelection = selection;
    tiledProblem.outputSelection = selection;

    const ParityRunInfo run =
        runParity(canonicalProblem, tiledProblem, canonicalOutput, tiledOutput,
                  "Block-scaled tiled selection differs from the canonical reference.");
    require(run.canonical.outputElementsComputed == 3,
            "Canonical block-scaled selection count changed unexpectedly.");
    require(run.tiled.outputElementsComputed == 1027,
            "Block-scaled tiled selection reported the wrong executed tile area.");
    requireOnlySelectedOutputsStored(tiledOutput, selection);
}

void testFullSelectionParity() {
    constexpr size_t rows = 35;
    constexpr size_t reductionElements = 17;
    constexpr size_t columns = 34;

    const std::vector<float> a = makeValues(rows, reductionElements, 11);
    const std::vector<float> b = makeValues(reductionElements, columns, 12);
    const std::vector<float> c = makeValues(rows, columns, 13);
    const std::vector<float> bias = makeValues(1, columns, 14);
    Tensor canonicalOutput = makeOutput(rows, columns, untouchedValue);
    Tensor tiledOutput = makeOutput(rows, columns, untouchedValue);

    GemmRequest canonicalProblem =
        makeProblem(a, b, c, canonicalOutput, rows, reductionElements, columns);
    GemmRequest tiledProblem = makeProblem(a, b, c, tiledOutput, rows, reductionElements, columns);
    configureFinalizer(canonicalProblem, bias);
    configureFinalizer(tiledProblem, bias);

    const ParityRunInfo run =
        runParity(canonicalProblem, tiledProblem, canonicalOutput, tiledOutput,
                  "Full tiled selection differs from the canonical reference.");
    require(run.canonical.outputElementsComputed == rows * columns,
            "Canonical full selection count changed unexpectedly.");
    require(run.tiled.outputElementsComputed == rows * columns,
            "Full tiled selection did not preserve complete-output accounting.");
}
}  // namespace

int main() {
    testFinalizerAndSmallEdgeTile();
    testExplicitSelectionTilePlan();
    testStridedSelectionTilePlan();
    testBlockScaledSelectionTilePlan();
    testFullSelectionParity();
    return 0;
}
