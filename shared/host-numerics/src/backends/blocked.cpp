// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "detail/blocked_gemm.hpp"
#include "detail/reference_gemm.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics {
namespace {
using detail::GemmSupportInfo;

constexpr size_t outputBlockRows = 32;
constexpr size_t outputBlockColumns = 32;
constexpr size_t reductionBlockElements = 8;

struct SelectedOutputLocation {
    size_t blockRow;
    size_t blockColumn;
    size_t localIndex;
    size_t selectedIndex;
};

struct PlannedOutputBlock {
    size_t rowBase;
    size_t columnBase;
    size_t firstSelectedOutput;
    size_t selectedOutputCount;
};

struct SelectedOutputBlockPlan {
    std::vector<PlannedOutputBlock> blocks;
    std::vector<SelectedOutputLocation> locations;
};

SelectedOutputBlockPlan planSelectedOutputBlocks(const OutputSelection& selection,
                                                 const Shape& outputShape) {
    const size_t logicalElements = outputShape.elementCount();
    const std::vector<size_t> selectedIndices = selection.indices(logicalElements);
    std::vector<SelectedOutputLocation> locations;
    locations.reserve(selectedIndices.size());
    for (size_t selectedIndex = 0; selectedIndex < selectedIndices.size(); ++selectedIndex) {
        const size_t logicalIndex = selectedIndices[selectedIndex];
        const auto coordinates = outputShape.coordinates(logicalIndex, selection.indexOrder());
        const size_t row = coordinates[0];
        const size_t column = coordinates[1];
        locations.push_back({
            .blockRow = row / outputBlockRows,
            .blockColumn = column / outputBlockColumns,
            .localIndex =
                (row % outputBlockRows) * outputBlockColumns + column % outputBlockColumns,
            .selectedIndex = selectedIndex,
        });
    }

    // Group unique requests in the same block-major, row-major order as the full path.
    std::sort(locations.begin(), locations.end(),
              [](const SelectedOutputLocation& left, const SelectedOutputLocation& right) {
                  if (left.blockRow != right.blockRow) return left.blockRow < right.blockRow;
                  if (left.blockColumn != right.blockColumn)
                      return left.blockColumn < right.blockColumn;
                  return left.localIndex < right.localIndex;
              });
    locations.erase(
        std::unique(locations.begin(), locations.end(),
                    [](const SelectedOutputLocation& left, const SelectedOutputLocation& right) {
                        return left.blockRow == right.blockRow &&
                               left.blockColumn == right.blockColumn &&
                               left.localIndex == right.localIndex;
                    }),
        locations.end());

    SelectedOutputBlockPlan plan;
    plan.locations.reserve(locations.size());
    for (const SelectedOutputLocation& location : locations) {
        const size_t rowBase = location.blockRow * outputBlockRows;
        const size_t columnBase = location.blockColumn * outputBlockColumns;
        if (plan.blocks.empty() || plan.blocks.back().rowBase != rowBase ||
            plan.blocks.back().columnBase != columnBase) {
            plan.blocks.push_back({
                .rowBase = rowBase,
                .columnBase = columnBase,
                .firstSelectedOutput = plan.locations.size(),
                .selectedOutputCount = 0,
            });
        }
        plan.locations.push_back(location);
        ++plan.blocks.back().selectedOutputCount;
    }
    return plan;
}

void validateBlocked(const GemmInvocation& problem) {
    validateRuntimeGemm(problem);
}

template <typename Accumulator, bool QuantizeAccumulator>
detail::RuntimeGemmFinalizer<Accumulator> makeFinalizer(const GemmInvocation& problem) {
    if constexpr (QuantizeAccumulator)
        return detail::RuntimeGemmFinalizer<Accumulator>(
            problem, detail::gemmAccumulatorQuantizer<Accumulator>(problem));
    else
        return detail::RuntimeGemmFinalizer<Accumulator>(problem);
}

template <typename Accumulator, bool QuantizeAccumulator = false>
GemmExecutionInfo runBlocked(const GemmInvocation& problem, Tensor* selectedOutput = nullptr) {
    using namespace detail;
    constexpr bool needsExplicitArithmetic =
        QuantizeAccumulator || (std::is_integral_v<Accumulator> && std::is_signed_v<Accumulator>);

    const size_t m = problem.a.shape()[0];
    const size_t k = problem.a.shape()[1];
    const size_t n = problem.b.shape()[1];
    const size_t outputElementsWritten =
        problem.outputSelection.selectedCount(problem.d.shape().elementCount());
    size_t outputElementsCovered = 0;
    std::optional<SelectedOutputBlockPlan> selectedPlan;
    if (!problem.outputSelection.selectsAll()) {
        selectedPlan = planSelectedOutputBlocks(problem.outputSelection, problem.d.shape());
        for (const PlannedOutputBlock& block : selectedPlan->blocks)
            outputElementsCovered += std::min(outputBlockRows, m - block.rowBase) *
                                     std::min(outputBlockColumns, n - block.columnBase);
    }

    const RuntimeMatrixReader<Accumulator> a(problem.a);
    const RuntimeMatrixReader<Accumulator> b(problem.b);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.computeTypeA);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.computeTypeB);
    const RuntimeGemmFinalizer<Accumulator> finalizer =
        makeFinalizer<Accumulator, QuantizeAccumulator>(problem);
    const auto multiply = [&](Accumulator left, Accumulator right) {
        if constexpr (needsExplicitArithmetic)
            return finalizer.multiply(left, right);
        else
            return left * right;
    };
    const auto add = [&](Accumulator left, Accumulator right) {
        if constexpr (needsExplicitArithmetic)
            return finalizer.add(left, right);
        else
            return left + right;
    };
    const RuntimeMatrixOutputWriter<Accumulator> output(problem.d, problem.outputConversion);
    std::optional<RuntimeMatrixOutputWriter<Accumulator>> selectedOutputWriter;
    if (selectedOutput != nullptr)
        selectedOutputWriter.emplace(*selectedOutput, problem.outputConversion);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);
    std::vector<RuntimeMatrixReader<Accumulator>> preScalesA;
    std::vector<RuntimeMatrixReader<Accumulator>> preScalesB;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleB;
    preScalesA.reserve(problem.preQuantizationScalesA.size());
    for (const Tensor& scale : problem.preQuantizationScalesA)
        preScalesA.emplace_back(scale.broadcastTo(problem.a.shape()));
    preScalesB.reserve(problem.preQuantizationScalesB.size());
    for (const Tensor& scale : problem.preQuantizationScalesB)
        preScalesB.emplace_back(scale.broadcastTo(problem.b.shape()));
    if (problem.blockScaleA) blockScaleA.emplace(*problem.blockScaleA);
    if (problem.blockScaleB) blockScaleB.emplace(*problem.blockScaleB);

    const bool hasBlockScale = blockScaleA.has_value() || blockScaleB.has_value();

    const auto executeBlock = [&](size_t rowBase, size_t columnBase, bool storeAllOutputs,
                                  std::span<const SelectedOutputLocation> selectedOutputs) {
        const size_t rows = std::min(outputBlockRows, m - rowBase);
        const size_t columns = std::min(outputBlockColumns, n - columnBase);
        std::vector<Accumulator> accumulator(rows * columns, Accumulator(0));
        const size_t maximumReductions = std::min(reductionBlockElements, k);
        std::vector<Accumulator> aBlock(rows * maximumReductions);
        std::vector<Accumulator> bBlock(maximumReductions * columns);
        std::vector<Accumulator> partial(hasBlockScale ? rows * columns : 0);

        const auto accumulateTile = [&](std::vector<Accumulator>& destination, size_t reductionBase,
                                        size_t reductions) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    Accumulator value = conjugateIfNeeded(
                        a(rowBase + row, reductionBase + reduction), problem.conjugateA);
                    for (const auto& scale : preScalesA) {
                        if constexpr (needsExplicitArithmetic)
                            value =
                                multiply(value, scale(rowBase + row, reductionBase + reduction));
                        else
                            value *= scale(rowBase + row, reductionBase + reduction);
                    }
                    aBlock[row * reductions + reduction] = operandMath(quantizeA(value));
                }
            }
            for (size_t reduction = 0; reduction < reductions; ++reduction) {
                for (size_t column = 0; column < columns; ++column) {
                    Accumulator value = conjugateIfNeeded(
                        b(reductionBase + reduction, columnBase + column), problem.conjugateB);
                    for (const auto& scale : preScalesB) {
                        if constexpr (needsExplicitArithmetic)
                            value = multiply(value,
                                             scale(reductionBase + reduction, columnBase + column));
                        else
                            value *= scale(reductionBase + reduction, columnBase + column);
                    }
                    bBlock[reduction * columns + column] = operandMath(quantizeB(value));
                }
            }

            for (size_t row = 0; row < rows; ++row) {
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    const Accumulator aValue = aBlock[row * reductions + reduction];
                    for (size_t column = 0; column < columns; ++column) {
                        if constexpr (needsExplicitArithmetic)
                            destination[row * columns + column] =
                                add(destination[row * columns + column],
                                    multiply(aValue, bBlock[reduction * columns + column]));
                        else
                            destination[row * columns + column] +=
                                aValue * bBlock[reduction * columns + column];
                    }
                }
            }
        };

        if (!finalizer.skipsProduct() && !hasBlockScale) {
            for (size_t reductionBase = 0; reductionBase < k;
                 reductionBase += reductionBlockElements) {
                const size_t reductions = std::min(reductionBlockElements, k - reductionBase);
                accumulateTile(accumulator, reductionBase, reductions);
            }
        } else if (!finalizer.skipsProduct()) {
            for (size_t reductionBase = 0; reductionBase < k;) {
                size_t reductions = std::min(reductionBlockElements, k - reductionBase);
                if (blockScaleA)
                    reductions = std::min(reductions,
                                          problem.blockSizeA - reductionBase % problem.blockSizeA);
                if (blockScaleB)
                    reductions = std::min(reductions,
                                          problem.blockSizeB - reductionBase % problem.blockSizeB);
                const bool startsScaleSegment =
                    reductionBase == 0 ||
                    (blockScaleA && reductionBase % problem.blockSizeA == 0) ||
                    (blockScaleB && reductionBase % problem.blockSizeB == 0);
                if (startsScaleSegment) std::fill(partial.begin(), partial.end(), Accumulator(0));
                accumulateTile(partial, reductionBase, reductions);

                const size_t reductionEnd = reductionBase + reductions;
                const bool endsScaleSegment =
                    reductionEnd == k || (blockScaleA && reductionEnd % problem.blockSizeA == 0) ||
                    (blockScaleB && reductionEnd % problem.blockSizeB == 0);
                if (!endsScaleSegment) {
                    reductionBase = reductionEnd;
                    continue;
                }
                std::array<Accumulator, outputBlockColumns> bScales;
                for (size_t column = 0; column < columns; ++column)
                    bScales[column] = blockScaleB
                                          ? (*blockScaleB)(columnBase + column,
                                                           (reductionEnd - 1) / problem.blockSizeB)
                                          : Accumulator(1);
                for (size_t row = 0; row < rows; ++row) {
                    const Accumulator aScale =
                        blockScaleA
                            ? (*blockScaleA)(rowBase + row, (reductionEnd - 1) / problem.blockSizeA)
                            : Accumulator(1);
                    for (size_t column = 0; column < columns; ++column) {
                        if constexpr (needsExplicitArithmetic) {
                            const Accumulator scale = multiply(aScale, bScales[column]);
                            accumulator[row * columns + column] =
                                add(accumulator[row * columns + column],
                                    multiply(partial[row * columns + column], scale));
                        } else {
                            const Accumulator scale = aScale * bScales[column];
                            accumulator[row * columns + column] +=
                                partial[row * columns + column] * scale;
                        }
                    }
                }
                reductionBase = reductionEnd;
            }
        }

        if (storeAllOutputs) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < columns; ++column) {
                    output.store(rowBase + row, columnBase + column,
                                 finalizer.finalize(rowBase + row, columnBase + column,
                                                    accumulator[row * columns + column]));
                }
            }
        } else {
            for (const SelectedOutputLocation& selected : selectedOutputs) {
                const size_t row = selected.localIndex / outputBlockColumns;
                const size_t column = selected.localIndex % outputBlockColumns;
                const Accumulator value = finalizer.finalize(rowBase + row, columnBase + column,
                                                             accumulator[row * columns + column]);
                if (selectedOutputWriter)
                    selectedOutputWriter->store(0, selected.selectedIndex, value);
                else
                    output.store(rowBase + row, columnBase + column, value);
            }
        }
        return rows * columns;
    };

    const bool parallelOutput = detail::canParallelizeGemmOutput(problem);
    const size_t reductionWork = finalizer.skipsProduct() ? 0 : k;
    if (problem.outputSelection.selectsAll()) {
        const size_t rowBlockCount = (m + outputBlockRows - 1) / outputBlockRows;
        const size_t columnBlockCount = (n + outputBlockColumns - 1) / outputBlockColumns;
        const size_t blockCount = rowBlockCount * columnBlockCount;
        outputElementsCovered = m * n;
        detail::forEachParallelIndex(
            blockCount, detail::saturatedProduct(outputElementsCovered, reductionWork),
            parallelOutput, 1'000'000, [&](size_t block) {
                const size_t rowBase = (block / columnBlockCount) * outputBlockRows;
                const size_t columnBase = (block % columnBlockCount) * outputBlockColumns;
                (void)executeBlock(rowBase, columnBase, true, {});
            });
    } else {
        const std::span<const SelectedOutputLocation> selectedOutputs(selectedPlan->locations);
        detail::forEachParallelIndex(
            selectedPlan->blocks.size(),
            detail::saturatedProduct(outputElementsCovered, reductionWork), parallelOutput,
            1'000'000, [&](size_t index) {
                const PlannedOutputBlock& block = selectedPlan->blocks[index];
                (void)executeBlock(
                    block.rowBase, block.columnBase, false,
                    selectedOutputs.subspan(block.firstSelectedOutput, block.selectedOutputCount));
            });
    }

    return {
        .backendUsed = GemmBackend::Blocked,
        .fallbackReason = std::nullopt,
        .outputElementsWritten = outputElementsWritten,
        .outputElementsCovered = outputElementsCovered,
    };
}
}  // namespace

GemmSupportInfo detail::queryBlockedGemmSupport(const GemmInvocation& problem) {
    try {
        validateBlocked(problem);
        return {.supported = true, .reason = {}};
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

GemmExecutionInfo detail::runBlockedGemm(const GemmInvocation& problem) {
    const GemmSupportInfo support = queryBlockedGemmSupport(problem);
    if (!support) throw std::invalid_argument(support.reason);
    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            if (problem.accumulationRounding == AccumulationRounding::FullPrecision)
                return runBlocked<float, false>(problem);
            return runBlocked<float, true>(problem);
        case ScalarType::Float32:
            return runBlocked<float, false>(problem);
        case ScalarType::Float64:
            return runBlocked<double, false>(problem);
        case ScalarType::Int32:
            return runBlocked<int32_t, false>(problem);
        case ScalarType::ComplexFloat32:
            return runBlocked<std::complex<float>, false>(problem);
        case ScalarType::ComplexFloat64:
            return runBlocked<std::complex<double>, false>(problem);
        default:
            throw std::invalid_argument("Blocked backend accumulator type is unsupported.");
    }
}

GemmExecutionInfo detail::runBlockedGemmToSelectedOutput(const GemmInvocation& problem,
                                                         Tensor& selectedOutput) {
    validateBlocked(problem);
    if (problem.outputSelection.selectsAll())
        throw std::invalid_argument("Streaming blocked GEMM requires a partial selection.");
    const size_t selectedCount =
        problem.outputSelection.selectedCount(problem.d.shape().elementCount());
    if (selectedOutput.type() != problem.outputType ||
        selectedOutput.shape() != Shape{1, selectedCount})
        throw std::invalid_argument("Streaming blocked GEMM output shape or type mismatch.");

    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            if (problem.accumulationRounding == AccumulationRounding::FullPrecision)
                return runBlocked<float, false>(problem, &selectedOutput);
            return runBlocked<float, true>(problem, &selectedOutput);
        case ScalarType::Float32:
            return runBlocked<float, false>(problem, &selectedOutput);
        case ScalarType::Float64:
            return runBlocked<double, false>(problem, &selectedOutput);
        case ScalarType::Int32:
            return runBlocked<int32_t, false>(problem, &selectedOutput);
        case ScalarType::ComplexFloat32:
            return runBlocked<std::complex<float>, false>(problem, &selectedOutput);
        case ScalarType::ComplexFloat64:
            return runBlocked<std::complex<double>, false>(problem, &selectedOutput);
        default:
            throw std::invalid_argument("Blocked backend accumulator type is unsupported.");
    }
}
}  // namespace roc::host_numerics
