// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <optional>
#include <roc/host_validation/backends/tiled.hpp>
#include <span>
#include <stdexcept>
#include <vector>

#include "detail/reference_gemm.hpp"

namespace roc::host_validation {
namespace {
constexpr size_t outputTileRows = 32;
constexpr size_t outputTileColumns = 32;
constexpr size_t reductionTileElements = 8;

struct SelectedOutputLocation {
    size_t tileRow;
    size_t tileColumn;
    size_t localIndex;
};

struct PlannedOutputTile {
    size_t rowBase;
    size_t columnBase;
    size_t firstSelectedOutput;
    size_t selectedOutputCount;
};

struct SelectedOutputTilePlan {
    std::vector<PlannedOutputTile> tiles;
    std::vector<size_t> localIndices;
};

SelectedOutputTilePlan planSelectedOutputTiles(const OutputSelection& selection,
                                               size_t logicalElements, size_t outputColumns) {
    const std::vector<size_t> selectedIndices = selection.indices(logicalElements);
    std::vector<SelectedOutputLocation> locations;
    locations.reserve(selectedIndices.size());
    for (const size_t logicalIndex : selectedIndices) {
        const size_t row = logicalIndex / outputColumns;
        const size_t column = logicalIndex % outputColumns;
        locations.push_back({
            .tileRow = row / outputTileRows,
            .tileColumn = column / outputTileColumns,
            .localIndex = (row % outputTileRows) * outputTileColumns + column % outputTileColumns,
        });
    }

    // Group unique requests in the same tile-major, row-major order as the full path.
    std::sort(locations.begin(), locations.end(),
              [](const SelectedOutputLocation& left, const SelectedOutputLocation& right) {
                  if (left.tileRow != right.tileRow) return left.tileRow < right.tileRow;
                  if (left.tileColumn != right.tileColumn)
                      return left.tileColumn < right.tileColumn;
                  return left.localIndex < right.localIndex;
              });
    locations.erase(
        std::unique(locations.begin(), locations.end(),
                    [](const SelectedOutputLocation& left, const SelectedOutputLocation& right) {
                        return left.tileRow == right.tileRow &&
                               left.tileColumn == right.tileColumn &&
                               left.localIndex == right.localIndex;
                    }),
        locations.end());

    SelectedOutputTilePlan plan;
    plan.localIndices.reserve(locations.size());
    for (const SelectedOutputLocation& location : locations) {
        const size_t rowBase = location.tileRow * outputTileRows;
        const size_t columnBase = location.tileColumn * outputTileColumns;
        if (plan.tiles.empty() || plan.tiles.back().rowBase != rowBase ||
            plan.tiles.back().columnBase != columnBase) {
            plan.tiles.push_back({
                .rowBase = rowBase,
                .columnBase = columnBase,
                .firstSelectedOutput = plan.localIndices.size(),
                .selectedOutputCount = 0,
            });
        }
        plan.localIndices.push_back(location.localIndex);
        ++plan.tiles.back().selectedOutputCount;
    }
    return plan;
}

void validateTiled(const GemmRequest& problem) {
    const GemmSupportInfo canonical =
        queryGemmSupport(problem, {.backend = GemmBackend::Canonical});
    if (!canonical) throw std::invalid_argument(canonical.reason);
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Tiled backend supports F32 and F64 accumulation.");
    if (problem.a.blockScale) {
        const size_t k = problem.a.values.shape()[1];
        if (problem.a.blockScale->blockSize % reductionTileElements != 0 ||
            problem.b.blockScale->blockSize % reductionTileElements != 0)
            throw std::invalid_argument(
                "Tiled backend requires block sizes divisible by its K tile.");
        if (k % problem.a.blockScale->blockSize != 0 || k % problem.b.blockScale->blockSize != 0)
            throw std::invalid_argument("Tiled backend requires K divisible by both block sizes.");
    }
}

template <typename Accumulator>
GemmRunInfo runTiled(const GemmRequest& problem) {
    using namespace detail;

    const RuntimeMatrixReader<Accumulator> a(problem.a.values);
    const RuntimeMatrixReader<Accumulator> b(problem.b.values);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const RuntimeGemmFinalizer<Accumulator> finalizer(problem);
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

    const auto executeTile = [&](size_t rowBase, size_t columnBase, bool storeAllOutputs,
                                 std::span<const size_t> selectedOutputIndices) {
        const size_t rows = std::min(outputTileRows, m - rowBase);
        const size_t columns = std::min(outputTileColumns, n - columnBase);
        std::vector<Accumulator> accumulator(rows * columns, Accumulator(0));

        for (size_t reductionBase = 0; !finalizer.alphaIsZero() && reductionBase < k;
             reductionBase += reductionTileElements) {
            const size_t reductions = std::min(reductionTileElements, k - reductionBase);
            std::vector<Accumulator> aTile(rows * reductions);
            std::vector<Accumulator> bTile(reductions * columns);
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
                    aTile[row * reductions + reduction] = operandMath(quantizeA(value));
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
                    bTile[reduction * columns + column] = operandMath(quantizeB(value));
                }
            }

            std::vector<Accumulator> partial(blockScaleA ? rows * columns : 0, Accumulator(0));
            std::vector<Accumulator>& destination = blockScaleA ? partial : accumulator;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    const Accumulator aValue = aTile[row * reductions + reduction];
                    for (size_t column = 0; column < columns; ++column)
                        destination[row * columns + column] +=
                            aValue * bTile[reduction * columns + column];
                }
            }
            if (blockScaleA) {
                const size_t blockA = reductionBase / problem.a.blockScale->blockSize;
                const size_t blockB = reductionBase / problem.b.blockScale->blockSize;
                for (size_t row = 0; row < rows; ++row) {
                    const Accumulator aScale = (*blockScaleA)(rowBase + row, blockA);
                    for (size_t column = 0; column < columns; ++column) {
                        const Accumulator scale =
                            aScale * (*blockScaleB)(columnBase + column, blockB);
                        accumulator[row * columns + column] +=
                            partial[row * columns + column] * scale;
                    }
                }
            }
        }

        if (storeAllOutputs) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < columns; ++column) {
                    finalizer.store(rowBase + row, columnBase + column,
                                    accumulator[row * columns + column]);
                }
            }
        } else {
            for (const size_t localIndex : selectedOutputIndices) {
                const size_t row = localIndex / outputTileColumns;
                const size_t column = localIndex % outputTileColumns;
                finalizer.store(rowBase + row, columnBase + column,
                                accumulator[row * columns + column]);
            }
        }
        return rows * columns;
    };

    size_t outputElementsComputed = 0;
    if (problem.outputSelection.selectsAll()) {
        for (size_t rowBase = 0; rowBase < m; rowBase += outputTileRows) {
            for (size_t columnBase = 0; columnBase < n; columnBase += outputTileColumns)
                outputElementsComputed += executeTile(rowBase, columnBase, true, {});
        }
    } else {
        const SelectedOutputTilePlan plan =
            planSelectedOutputTiles(problem.outputSelection, problem.d.shape().elementCount(), n);
        const std::span<const size_t> localIndices(plan.localIndices);
        for (const PlannedOutputTile& tile : plan.tiles) {
            outputElementsComputed += executeTile(
                tile.rowBase, tile.columnBase, false,
                localIndices.subspan(tile.firstSelectedOutput, tile.selectedOutputCount));
        }
    }

    return {
        .backendUsed = GemmBackend::Tiled,
        .fallbackReason = std::nullopt,
        .outputElementsComputed = outputElementsComputed,
    };
}
}  // namespace

GemmBackend TiledGemmBackend::backend() const {
    return GemmBackend::Tiled;
}

GemmSupportInfo TiledGemmBackend::querySupport(const GemmRequest& problem) const {
    try {
        validateTiled(problem);
        return {.supported = true, .reason = {}};
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

GemmRunInfo TiledGemmBackend::run(const GemmRequest& problem) const {
    const GemmSupportInfo support = querySupport(problem);
    if (!support) throw std::invalid_argument(support.reason);
    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            return runTiled<float>(problem);
        case ScalarType::Float64:
            return runTiled<double>(problem);
        default:
            throw std::invalid_argument("Tiled backend accumulator type is unsupported.");
    }
}
}  // namespace roc::host_validation
