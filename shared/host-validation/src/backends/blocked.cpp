// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <roc/host_validation/backends/blocked.hpp>
#include <span>
#include <stdexcept>
#include <vector>

#include "detail/reference_gemm.hpp"
#include "detail/threading.hpp"

namespace roc::host_validation {
namespace {
constexpr size_t outputBlockRows = 32;
constexpr size_t outputBlockColumns = 32;
constexpr size_t reductionBlockElements = 8;

struct SelectedOutputLocation {
    size_t blockRow;
    size_t blockColumn;
    size_t localIndex;
};

struct PlannedOutputBlock {
    size_t rowBase;
    size_t columnBase;
    size_t firstSelectedOutput;
    size_t selectedOutputCount;
};

struct SelectedOutputBlockPlan {
    std::vector<PlannedOutputBlock> blocks;
    std::vector<size_t> localIndices;
};

SelectedOutputBlockPlan planSelectedOutputBlocks(const OutputSelection& selection,
                                                 size_t logicalElements, size_t outputColumns) {
    const std::vector<size_t> selectedIndices = selection.indices(logicalElements);
    std::vector<SelectedOutputLocation> locations;
    locations.reserve(selectedIndices.size());
    for (const size_t logicalIndex : selectedIndices) {
        const size_t row = logicalIndex / outputColumns;
        const size_t column = logicalIndex % outputColumns;
        locations.push_back({
            .blockRow = row / outputBlockRows,
            .blockColumn = column / outputBlockColumns,
            .localIndex =
                (row % outputBlockRows) * outputBlockColumns + column % outputBlockColumns,
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
    plan.localIndices.reserve(locations.size());
    for (const SelectedOutputLocation& location : locations) {
        const size_t rowBase = location.blockRow * outputBlockRows;
        const size_t columnBase = location.blockColumn * outputBlockColumns;
        if (plan.blocks.empty() || plan.blocks.back().rowBase != rowBase ||
            plan.blocks.back().columnBase != columnBase) {
            plan.blocks.push_back({
                .rowBase = rowBase,
                .columnBase = columnBase,
                .firstSelectedOutput = plan.localIndices.size(),
                .selectedOutputCount = 0,
            });
        }
        plan.localIndices.push_back(location.localIndex);
        ++plan.blocks.back().selectedOutputCount;
    }
    return plan;
}

void validateBlocked(const GemmRequest& problem) {
    const GemmSupportInfo pointwise =
        queryGemmSupport(problem, {.backend = GemmBackend::Pointwise});
    if (!pointwise) throw std::invalid_argument(pointwise.reason);
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Blocked backend supports F32 and F64 accumulation.");
    if (problem.a.blockScale) {
        const size_t k = problem.a.values.shape()[1];
        if (problem.a.blockScale->blockSize % reductionBlockElements != 0 ||
            problem.b.blockScale->blockSize % reductionBlockElements != 0)
            throw std::invalid_argument(
                "Blocked backend requires block sizes divisible by its K block.");
        if (k % problem.a.blockScale->blockSize != 0 || k % problem.b.blockScale->blockSize != 0)
            throw std::invalid_argument(
                "Blocked backend requires K divisible by both block sizes.");
    }
}

template <typename Accumulator>
GemmRunInfo runBlocked(const GemmRequest& problem) {
    using namespace detail;

    const RuntimeMatrixReader<Accumulator> a(problem.a.values);
    const RuntimeMatrixReader<Accumulator> b(problem.b.values);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const RuntimeGemmFinalizer<Accumulator> finalizer(problem);
    const RuntimeMatrixOutputWriter<Accumulator> output(problem.d,
                                                        problem.epilogue.outputConversion);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);
    std::vector<RuntimeVectorReader<Accumulator>> preScalesA;
    std::vector<RuntimeVectorReader<Accumulator>> preScalesB;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleA;
    std::optional<RuntimeMatrixReader<Accumulator>> blockScaleB;
    preScalesA.reserve(problem.a.preQuantizationScales.size());
    for (const VectorBinding& binding : problem.a.preQuantizationScales)
        preScalesA.emplace_back(binding.values);
    preScalesB.reserve(problem.b.preQuantizationScales.size());
    for (const VectorBinding& binding : problem.b.preQuantizationScales)
        preScalesB.emplace_back(binding.values);
    if (problem.a.blockScale) {
        blockScaleA.emplace(problem.a.blockScale->values);
        blockScaleB.emplace(problem.b.blockScale->values);
    }

    const size_t m = problem.a.values.shape()[0];
    const size_t k = problem.a.values.shape()[1];
    const size_t n = problem.b.values.shape()[1];

    const auto executeBlock = [&](size_t rowBase, size_t columnBase, bool storeAllOutputs,
                                  std::span<const size_t> selectedOutputIndices) {
        const size_t rows = std::min(outputBlockRows, m - rowBase);
        const size_t columns = std::min(outputBlockColumns, n - columnBase);
        std::vector<Accumulator> accumulator(rows * columns, Accumulator(0));
        const size_t maximumReductions = std::min(reductionBlockElements, k);
        std::vector<Accumulator> aBlock(rows * maximumReductions);
        std::vector<Accumulator> bBlock(maximumReductions * columns);
        std::vector<Accumulator> partial(blockScaleA ? rows * columns : 0);

        for (size_t reductionBase = 0; !finalizer.alphaIsZero() && reductionBase < k;
             reductionBase += reductionBlockElements) {
            const size_t reductions = std::min(reductionBlockElements, k - reductionBase);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    Accumulator value = conjugateIfNeeded(
                        a(rowBase + row, reductionBase + reduction), problem.a.conjugate);
                    for (size_t scaleIndex = 0; scaleIndex < preScalesA.size(); ++scaleIndex) {
                        const auto& binding = problem.a.preQuantizationScales[scaleIndex];
                        const size_t index =
                            binding.values.shape()[0] == 1
                                ? 0
                                : (binding.axis == MatrixAxis::Row ? rowBase + row
                                                                   : reductionBase + reduction);
                        value *= preScalesA[scaleIndex][index];
                    }
                    aBlock[row * reductions + reduction] = operandMath(quantizeA(value));
                }
            }
            for (size_t reduction = 0; reduction < reductions; ++reduction) {
                for (size_t column = 0; column < columns; ++column) {
                    Accumulator value = conjugateIfNeeded(
                        b(reductionBase + reduction, columnBase + column), problem.b.conjugate);
                    for (size_t scaleIndex = 0; scaleIndex < preScalesB.size(); ++scaleIndex) {
                        const auto& binding = problem.b.preQuantizationScales[scaleIndex];
                        const size_t index =
                            binding.values.shape()[0] == 1
                                ? 0
                                : (binding.axis == MatrixAxis::Row ? reductionBase + reduction
                                                                   : columnBase + column);
                        value *= preScalesB[scaleIndex][index];
                    }
                    bBlock[reduction * columns + column] = operandMath(quantizeB(value));
                }
            }

            if (blockScaleA) std::fill(partial.begin(), partial.end(), Accumulator(0));
            std::vector<Accumulator>& destination = blockScaleA ? partial : accumulator;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    const Accumulator aValue = aBlock[row * reductions + reduction];
                    for (size_t column = 0; column < columns; ++column)
                        destination[row * columns + column] +=
                            aValue * bBlock[reduction * columns + column];
                }
            }
            if (blockScaleA) {
                const size_t blockA = reductionBase / problem.a.blockScale->blockSize;
                const size_t blockB = reductionBase / problem.b.blockScale->blockSize;
                std::array<Accumulator, outputBlockColumns> bScales;
                for (size_t column = 0; column < columns; ++column)
                    bScales[column] = (*blockScaleB)(columnBase + column, blockB);
                for (size_t row = 0; row < rows; ++row) {
                    const Accumulator aScale = (*blockScaleA)(rowBase + row, blockA);
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
            for (const size_t localIndex : selectedOutputIndices) {
                const size_t row = localIndex / outputBlockColumns;
                const size_t column = localIndex % outputBlockColumns;
                output.store(rowBase + row, columnBase + column,
                             finalizer.finalize(rowBase + row, columnBase + column,
                                                accumulator[row * columns + column]));
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
            planSelectedOutputBlocks(problem.outputSelection, problem.d.shape().elementCount(), n);
        const std::span<const size_t> localIndices(plan.localIndices);
        for (const PlannedOutputBlock& block : plan.blocks)
            outputElementsCovered += std::min(outputBlockRows, m - block.rowBase) *
                                     std::min(outputBlockColumns, n - block.columnBase);
        detail::forEachParallelIndex(
            plan.blocks.size(), detail::saturatedProduct(outputElementsCovered, reductionWork),
            parallelOutput, 1'000'000, [&](size_t index) {
                const PlannedOutputBlock& block = plan.blocks[index];
                (void)executeBlock(
                    block.rowBase, block.columnBase, false,
                    localIndices.subspan(block.firstSelectedOutput, block.selectedOutputCount));
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

GemmBackend BlockedGemmBackend::backend() const {
    return GemmBackend::Blocked;
}

GemmSupportInfo BlockedGemmBackend::querySupport(const GemmRequest& problem) const {
    try {
        validateBlocked(problem);
        return {.supported = true, .reason = {}};
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

GemmRunInfo BlockedGemmBackend::run(const GemmRequest& problem) const {
    const GemmSupportInfo support = querySupport(problem);
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
}  // namespace roc::host_validation
