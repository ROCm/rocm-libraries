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
#include "test_support.hpp"

#pragma once

namespace blocked_backend_test {
namespace {
using roc::host_numerics::GemmTestCase;
using roc::host_numerics::GemmTestRunInfo;
using roc::host_numerics::OutputSelection;
using roc::host_numerics::Tensor;

constexpr float untouchedValue = -12345.0f;
using roc::host_numerics::test::require;

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
                         roc::host_numerics::Tensor d, size_t rows, size_t reductionElements,
                         size_t columns) {
    using namespace roc::host_numerics;

    return GemmTestCase(Tensor::copyNativeStorage<float>(
                            Layout::contiguousLastDimensionFastest(Shape{rows, reductionElements}),
                            std::span<const float>(a)),
                        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(
                                                             Shape{reductionElements, columns}),
                                                         std::span<const float>(b)),
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

Tensor expectedFloatResult(const GemmTestCase& problem) {
    using namespace roc::host_numerics;

    const size_t rows = problem.a.shape()[0];
    const size_t reductions = problem.a.shape()[1];
    const size_t columns = problem.b.shape()[1];
    Tensor expected = makeOutput(rows, columns, untouchedValue);
    const auto selected = problem.outputSelection.indices(problem.d.elementCount());
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

        expected.storeFrom({row, column}, accumulation);
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

}  // namespace

void testReducedPrecisionAccumulators();
void testSelectedBlockAccumulatorFamilies();
void requireOnlySelectedOutputsStored(const roc::host_numerics::Tensor& output,
                                      const roc::host_numerics::OutputSelection& selection);
void testSmallEdgeBlock();
void testExplicitSelectionBlockPlan();
void testStridedSelectionBlockPlan();
void testBlockScaledSelectionBlockPlan();
void testBlockScaleAppliedAfterCompleteScaleSegment();
void testOneSidedBlockScaling();
void testOneSidedBlockScalingWithZeroReductionExtent();
void testFullSelection();
void testAutomaticSelectionUsesBlockedBackend();
void testParallelFullSelection();
void testOverlappingOutputIsRejectedAcrossBackends();
void testOutputCannotAliasInputs();
void testParallelSparseSelection();
}  // namespace blocked_backend_test
