// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "detail/blocked_gemm.hpp"
#include "detail/reference_gemm.hpp"
#include "detail/threading.hpp"

namespace roc::host_numerics {
namespace {
using detail::GemmOperand;

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
    const size_t outputColumns = outputShape[1];
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
    const GemmSupportInfo pointwise = queryGemmSupport(problem, GemmBackend::Pointwise);
    if (!pointwise) throw std::invalid_argument(pointwise.reason);
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Blocked backend supports F32 and F64 accumulation.");
    const auto validateBlockScale = [&](const GemmOperand& operand) {
        if (!operand.blockScale) return;
        const size_t k = problem.a.values.shape()[1];
        if (operand.blockSize % reductionBlockElements != 0)
            throw std::invalid_argument(
                "Blocked backend requires block sizes divisible by its K block.");
        if (k % operand.blockSize != 0)
            throw std::invalid_argument(
                "Blocked backend requires K divisible by every block-scale size.");
    };
    validateBlockScale(problem.a);
    validateBlockScale(problem.b);
}

template <typename Accumulator>
GemmExecutionInfo runBlocked(const GemmInvocation& problem, Tensor* selectedOutput = nullptr) {
    using namespace detail;

    const RuntimeMatrixReader<Accumulator> a(problem.a.values);
    const RuntimeMatrixReader<Accumulator> b(problem.b.values);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const RuntimeGemmFinalizer<Accumulator> finalizer(problem);
    const RuntimeMatrixOutputWriter<Accumulator> output(problem.d,
                                                        problem.epilogue.outputConversion);
    std::optional<RuntimeMatrixOutputWriter<Accumulator>> selectedOutputWriter;
    if (selectedOutput != nullptr)
        selectedOutputWriter.emplace(*selectedOutput, problem.epilogue.outputConversion);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);
    std::vector<RuntimeMatrixReader<Accumulator>> preScalesA;
    std::vector<RuntimeMatrixReader<Accumulator>> preScalesB;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleB;
    preScalesA.reserve(problem.a.preQuantizationScales.size());
    for (const Tensor& scale : problem.a.preQuantizationScales)
        preScalesA.emplace_back(scale.broadcastTo(problem.a.values.shape()));
    preScalesB.reserve(problem.b.preQuantizationScales.size());
    for (const Tensor& scale : problem.b.preQuantizationScales)
        preScalesB.emplace_back(scale.broadcastTo(problem.b.values.shape()));
    if (problem.a.blockScale) blockScaleA.emplace(*problem.a.blockScale);
    if (problem.b.blockScale) blockScaleB.emplace(*problem.b.blockScale);

    const size_t m = problem.a.values.shape()[0];
    const size_t k = problem.a.values.shape()[1];
    const size_t n = problem.b.values.shape()[1];

    const auto executeBlock = [&](size_t rowBase, size_t columnBase, bool storeAllOutputs,
                                  std::span<const SelectedOutputLocation> selectedOutputs) {
        const size_t rows = std::min(outputBlockRows, m - rowBase);
        const size_t columns = std::min(outputBlockColumns, n - columnBase);
        std::vector<Accumulator> accumulator(rows * columns, Accumulator(0));
        const size_t maximumReductions = std::min(reductionBlockElements, k);
        std::vector<Accumulator> aBlock(rows * maximumReductions);
        std::vector<Accumulator> bBlock(maximumReductions * columns);
        const bool hasBlockScale = blockScaleA.has_value() || blockScaleB.has_value();
        std::vector<Accumulator> partial(hasBlockScale ? rows * columns : 0);

        for (size_t reductionBase = 0; !finalizer.alphaIsZero() && reductionBase < k;
             reductionBase += reductionBlockElements) {
            const size_t reductions = std::min(reductionBlockElements, k - reductionBase);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    Accumulator value = conjugateIfNeeded(
                        a(rowBase + row, reductionBase + reduction), problem.a.conjugate);
                    for (const auto& scale : preScalesA)
                        value *= scale(rowBase + row, reductionBase + reduction);
                    aBlock[row * reductions + reduction] = operandMath(quantizeA(value));
                }
            }
            for (size_t reduction = 0; reduction < reductions; ++reduction) {
                for (size_t column = 0; column < columns; ++column) {
                    Accumulator value = conjugateIfNeeded(
                        b(reductionBase + reduction, columnBase + column), problem.b.conjugate);
                    for (const auto& scale : preScalesB)
                        value *= scale(reductionBase + reduction, columnBase + column);
                    bBlock[reduction * columns + column] = operandMath(quantizeB(value));
                }
            }

            const bool startsScaleSegment =
                hasBlockScale &&
                (reductionBase == 0 || (blockScaleA && reductionBase % problem.a.blockSize == 0) ||
                 (blockScaleB && reductionBase % problem.b.blockSize == 0));
            if (startsScaleSegment) std::fill(partial.begin(), partial.end(), Accumulator(0));
            std::vector<Accumulator>& destination = hasBlockScale ? partial : accumulator;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    const Accumulator aValue = aBlock[row * reductions + reduction];
                    for (size_t column = 0; column < columns; ++column)
                        destination[row * columns + column] +=
                            aValue * bBlock[reduction * columns + column];
                }
            }
            const size_t reductionEnd = reductionBase + reductions;
            const bool endsScaleSegment =
                hasBlockScale &&
                (reductionEnd == k || (blockScaleA && reductionEnd % problem.a.blockSize == 0) ||
                 (blockScaleB && reductionEnd % problem.b.blockSize == 0));
            if (endsScaleSegment) {
                std::array<Accumulator, outputBlockColumns> bScales;
                for (size_t column = 0; column < columns; ++column)
                    bScales[column] = blockScaleB
                                          ? (*blockScaleB)(columnBase + column,
                                                           reductionBase / problem.b.blockSize)
                                          : Accumulator(1);
                for (size_t row = 0; row < rows; ++row) {
                    const Accumulator aScale =
                        blockScaleA
                            ? (*blockScaleA)(rowBase + row, reductionBase / problem.a.blockSize)
                            : Accumulator(1);
                    for (size_t column = 0; column < columns; ++column) {
                        const Accumulator scale = aScale * bScales[column];
                        accumulator[row * columns + column] +=
                            partial[row * columns + column] * scale;
                    }
                }
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

    size_t outputElementsCovered = 0;
    const size_t outputElementsWritten =
        problem.outputSelection.selectedCount(problem.d.shape().elementCount());
    const bool parallelOutput = detail::canParallelizeGemmOutput(problem);
    const size_t reductionWork = finalizer.alphaIsZero() ? 0 : k;
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
        const SelectedOutputBlockPlan plan =
            planSelectedOutputBlocks(problem.outputSelection, problem.d.shape());
        const std::span<const SelectedOutputLocation> selectedOutputs(plan.locations);
        for (const PlannedOutputBlock& block : plan.blocks)
            outputElementsCovered += std::min(outputBlockRows, m - block.rowBase) *
                                     std::min(outputBlockColumns, n - block.columnBase);
        detail::forEachParallelIndex(
            plan.blocks.size(), detail::saturatedProduct(outputElementsCovered, reductionWork),
            parallelOutput, 1'000'000, [&](size_t index) {
                const PlannedOutputBlock& block = plan.blocks[index];
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
        return {
            .supported = true,
            .reason = {},
            .preferredForAutomaticExecution = isBlockedGemmPreferredForAutomaticExecution(problem),
        };
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

bool detail::isBlockedGemmPreferredForAutomaticExecution(const GemmInvocation& problem) {
    const size_t selectedOutputCount =
        problem.outputSelection.selectedCount(problem.d.shape().elementCount());
    if (selectedOutputCount == 0) return false;

    const size_t reductionElements = problem.a.values.shape()[1];
    const size_t pointwiseWork = detail::saturatedProduct(selectedOutputCount, reductionElements);
    constexpr size_t minimumBlockedMultiplyAdds = 8'192;
    if (pointwiseWork < minimumBlockedMultiplyAdds) return false;
    if (problem.outputSelection.selectsAll()) return true;

    const SelectedOutputBlockPlan plan =
        planSelectedOutputBlocks(problem.outputSelection, problem.d.shape());
    size_t coveredOutputCount = 0;
    for (const PlannedOutputBlock& block : plan.blocks)
        coveredOutputCount += std::min(outputBlockRows, problem.d.shape()[0] - block.rowBase) *
                              std::min(outputBlockColumns, problem.d.shape()[1] - block.columnBase);

    const size_t blockedWork = detail::saturatedProduct(coveredOutputCount, reductionElements);
    const size_t pointwiseThreads =
        static_cast<size_t>(detail::operationThreadCount(pointwiseWork, 500'000));
    const size_t blockedThreads =
        std::min(plan.blocks.size(),
                 static_cast<size_t>(detail::operationThreadCount(blockedWork, 1'000'000)));

    constexpr long double blockedWorkAdvantage = 20.0L;
    const long double pointwiseCost =
        static_cast<long double>(pointwiseWork) / static_cast<long double>(pointwiseThreads);
    const long double blockedCost = static_cast<long double>(blockedWork) /
                                    static_cast<long double>(std::max<size_t>(1, blockedThreads)) /
                                    blockedWorkAdvantage;
    return blockedCost < pointwiseCost;
}

GemmExecutionInfo detail::runBlockedGemm(const GemmInvocation& problem) {
    const GemmSupportInfo support = queryBlockedGemmSupport(problem);
    if (!support) throw std::invalid_argument(support.reason);
    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            return runBlocked<float>(problem);
        case ScalarType::Float64:
            return runBlocked<double>(problem);
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
        case ScalarType::Float32:
            return runBlocked<float>(problem, &selectedOutput);
        case ScalarType::Float64:
            return runBlocked<double>(problem, &selectedOutput);
        default:
            throw std::invalid_argument("Blocked backend supports F32 and F64 accumulation.");
    }
}
}  // namespace roc::host_numerics
