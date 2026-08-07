// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <roc/host_validation/mx.hpp>

using namespace roc::host_validation;

namespace {
void checkReference(const MxGenerationProblem& problem) {
    const MxGenerationResult result = generateMx(problem);
    const TensorView data = result.data.view();
    const TensorView scales = result.scales.view();
    const TensorView scaleIndices = result.scaleIndices.view();
    const TensorView reference = result.reference.view();
    const size_t rows = problem.shape[0];
    const size_t columns = problem.shape[1];
    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            const size_t scaleIndex = scaleIndices.loadAs<uint32_t>({row, column});
            const double expected =
                data.loadAs<double>({row, column}) * scales.loadAs<double>({scaleIndex});
            const double observed = reference.loadAs<double>({row, column});
            assert(std::isnan(expected) ? std::isnan(observed) : expected == observed);
        }
    }
}
}  // namespace

int main() {
    const std::array typePairs{
        std::pair{ScalarType::Float8E5M2, ScalarType::E8M0},
        std::pair{ScalarType::Float8E4M3, ScalarType::E8M0},
        std::pair{ScalarType::Float6E2M3, ScalarType::E8M0},
        std::pair{ScalarType::Float6E3M2, ScalarType::E8M0},
        std::pair{ScalarType::Float4E2M1, ScalarType::E8M0},
        std::pair{ScalarType::Float4E2M1, ScalarType::Float8E4M3},
        std::pair{ScalarType::Float4E2M1, ScalarType::E5M3},
    };
    for (const auto [dataType, scaleType] : typePairs) {
        MxGenerationProblem problem;
        problem.dataType = dataType;
        problem.scaleType = scaleType;
        problem.shape = Shape{64, 3};
        problem.leadingDimension = 64;
        problem.blockAxis = 0;
        problem.blockSize = 32;
        problem.data.mode = MxGenerationMode::Bounded;
        problem.data.parameter0 = -1;
        problem.data.parameter1 = 1;
        checkReference(problem);
    }

    MxGenerationProblem nonContiguous;
    nonContiguous.dataType = ScalarType::Float4E2M1;
    nonContiguous.scaleType = ScalarType::E8M0;
    nonContiguous.shape = Shape{3, 64};
    nonContiguous.leadingDimension = 3;
    nonContiguous.blockAxis = 1;
    checkReference(nonContiguous);

    MxGenerationResult first = generateMx(nonContiguous);
    MxGenerationResult second = generateMx(nonContiguous);
    assert(first.data.storage().size() == second.data.storage().size());
    assert(std::equal(first.data.storage().begin(), first.data.storage().end(),
                      second.data.storage().begin()));
    assert(first.scales.storage().size() == second.scales.storage().size());
    assert(std::equal(first.scales.storage().begin(), first.scales.storage().end(),
                      second.scales.storage().begin()));
}
