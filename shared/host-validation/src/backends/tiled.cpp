// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <optional>
#include <roc/host_validation/backends/tiled.hpp>
#include <stdexcept>
#include <vector>

namespace roc::host_validation {
namespace {
void validateTiled(const GemmProblem& problem) {
    const GemmSupportInfo canonical = queryGemmSupport(problem, GemmBackend::Canonical);
    if (!canonical) throw std::invalid_argument(canonical.reason);
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Tiled backend supports F32 and F64 accumulation.");
    if (problem.a.blockScale || problem.b.blockScale)
        throw std::invalid_argument("Tiled backend does not yet support block scaling.");
    if (!problem.outputSelection.selectsAll())
        throw std::invalid_argument("Tiled backend currently requires all outputs.");
}

template <typename Accumulator>
GemmRunInfo runTiled(const GemmProblem& problem) {
    using namespace detail;

    const RuntimeMatrixReader<Accumulator> a(problem.a.values);
    const RuntimeMatrixReader<Accumulator> b(problem.b.values);
    const RuntimeMatrixReader<Accumulator> c(problem.c);
    const RuntimeMatrixWriter<Accumulator> d(problem.d);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);

    std::optional<RuntimeVectorReader<Accumulator>> bias;
    std::optional<RuntimeVectorReader<Accumulator>> scaleAlpha;
    std::optional<RuntimeVectorReader<Accumulator>> scaleA;
    std::optional<RuntimeVectorReader<Accumulator>> scaleB;
    if (problem.epilogue.bias) bias.emplace(problem.epilogue.bias->values);
    if (problem.epilogue.scaleAlpha) scaleAlpha.emplace(problem.epilogue.scaleAlpha->values);
    if (problem.epilogue.scaleA) scaleA.emplace(*problem.epilogue.scaleA);
    if (problem.epilogue.scaleB) scaleB.emplace(*problem.epilogue.scaleB);

    const size_t m = problem.a.values.shape()[0];
    const size_t k = problem.a.values.shape()[1];
    const size_t n = problem.b.values.shape()[1];
    const Accumulator alpha = runtimeScalar<Accumulator>(problem.epilogue.alpha, "alpha");
    const Accumulator beta = runtimeScalar<Accumulator>(problem.epilogue.beta, "beta");
    const Accumulator activationParameter0 =
        static_cast<Accumulator>(problem.epilogue.activationParameter0);
    const Accumulator activationParameter1 =
        static_cast<Accumulator>(problem.epilogue.activationParameter1);

    constexpr size_t tileRows = 32;
    constexpr size_t tileColumns = 32;
    constexpr size_t tileReduction = 8;
    for (size_t rowBase = 0; rowBase < m; rowBase += tileRows) {
        const size_t rows = std::min(tileRows, m - rowBase);
        for (size_t columnBase = 0; columnBase < n; columnBase += tileColumns) {
            const size_t columns = std::min(tileColumns, n - columnBase);
            std::vector<Accumulator> accumulator(rows * columns, Accumulator(0));

            for (size_t reductionBase = 0; reductionBase < k; reductionBase += tileReduction) {
                const size_t reductions = std::min(tileReduction, k - reductionBase);
                std::vector<Accumulator> aTile(rows * reductions);
                std::vector<Accumulator> bTile(reductions * columns);
                for (size_t row = 0; row < rows; ++row) {
                    for (size_t reduction = 0; reduction < reductions; ++reduction) {
                        Accumulator value = conjugateIfNeeded(
                            a(rowBase + row, reductionBase + reduction), problem.a.conjugate);
                        aTile[row * reductions + reduction] = operandMath(quantizeA(value));
                    }
                }
                for (size_t reduction = 0; reduction < reductions; ++reduction) {
                    for (size_t column = 0; column < columns; ++column) {
                        Accumulator value = conjugateIfNeeded(
                            b(reductionBase + reduction, columnBase + column), problem.b.conjugate);
                        bTile[reduction * columns + column] = operandMath(quantizeB(value));
                    }
                }

                for (size_t row = 0; row < rows; ++row) {
                    for (size_t reduction = 0; reduction < reductions; ++reduction) {
                        const Accumulator aValue = aTile[row * reductions + reduction];
                        for (size_t column = 0; column < columns; ++column)
                            accumulator[row * columns + column] +=
                                aValue * bTile[reduction * columns + column];
                    }
                }
            }

            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < columns; ++column) {
                    const size_t globalRow = rowBase + row;
                    const size_t globalColumn = columnBase + column;
                    Accumulator effectiveAlpha = alpha;
                    if (scaleA) effectiveAlpha *= (*scaleA)[globalRow];
                    if (scaleB) effectiveAlpha *= (*scaleB)[globalColumn];
                    if (scaleAlpha) {
                        const MatrixAxis axis = problem.epilogue.scaleAlpha->axis;
                        effectiveAlpha *=
                            (*scaleAlpha)[axis == MatrixAxis::Row ? globalRow : globalColumn];
                    }

                    Accumulator result = effectiveAlpha * accumulator[row * columns + column] +
                                         beta * c(globalRow, globalColumn);
                    if (bias) {
                        const MatrixAxis axis = problem.epilogue.bias->axis;
                        result += (*bias)[axis == MatrixAxis::Row ? globalRow : globalColumn];
                    }
                    result = applyActivation(problem.epilogue.activation, result,
                                             activationParameter0, activationParameter1);
                    d.store(globalRow, globalColumn, result);
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

GemmSupportInfo TiledGemmBackend::querySupport(const GemmProblem& problem) const {
    try {
        validateTiled(problem);
        return {.supported = true, .reason = {}};
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }
}

GemmRunInfo TiledGemmBackend::run(const GemmProblem& problem) const {
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
