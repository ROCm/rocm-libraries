// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <optional>
#include <roc/host_validation/backends/tiled.hpp>
#include <stdexcept>
#include <vector>

#include "detail/reference_gemm.hpp"

namespace roc::host_validation {
namespace {
void validateTiled(const GemmRequest& problem) {
    const GemmSupportInfo canonical =
        queryGemmSupport(problem, {.backend = GemmBackend::Canonical});
    if (!canonical) throw std::invalid_argument(canonical.reason);
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Tiled backend supports F32 and F64 accumulation.");
    if (problem.a.blockScale) {
        constexpr size_t tileReduction = 8;
        const size_t k = problem.a.values.shape()[1];
        if (problem.a.blockScale->blockSize % tileReduction != 0 ||
            problem.b.blockScale->blockSize % tileReduction != 0)
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
    std::vector<bool> selectedOutputs;
    if (!problem.outputSelection.selectsAll()) {
        selectedOutputs.assign(problem.d.shape().elementCount(), false);
        for (const size_t index : problem.outputSelection.indices(problem.d.shape().elementCount()))
            selectedOutputs[index] = true;
    }

    constexpr size_t tileRows = 32;
    constexpr size_t tileColumns = 32;
    constexpr size_t tileReduction = 8;
    for (size_t rowBase = 0; rowBase < m; rowBase += tileRows) {
        const size_t rows = std::min(tileRows, m - rowBase);
        for (size_t columnBase = 0; columnBase < n; columnBase += tileColumns) {
            const size_t columns = std::min(tileColumns, n - columnBase);
            std::vector<Accumulator> accumulator(rows * columns, Accumulator(0));

            for (size_t reductionBase = 0; !finalizer.alphaIsZero() && reductionBase < k;
                 reductionBase += tileReduction) {
                const size_t reductions = std::min(tileReduction, k - reductionBase);
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

            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < columns; ++column) {
                    const size_t globalRow = rowBase + row;
                    const size_t globalColumn = columnBase + column;
                    if (!selectedOutputs.empty() && !selectedOutputs[globalRow * n + globalColumn])
                        continue;
                    finalizer.store(globalRow, globalColumn, accumulator[row * columns + column]);
                }
            }
        }
    }

    return {
        .backendUsed = GemmBackend::Tiled,
        .fallbackReason = std::nullopt,
        .outputElementsComputed = problem.d.shape().elementCount(),
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
