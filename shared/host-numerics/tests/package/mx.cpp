// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_numerics/mx.hpp>

int main() {
    using namespace roc::host_numerics;
    MxGenerationProblem problem(
        Shape{32, 1},
        MxDataGeneration::preserveRange(GenerationRecipe::realOnly(GenerationRecipe::uniformReal(
                                            {.lower = -1.0, .upper = 1.0})),
                                        {.lower = -1.0, .upper = 1.0}));
    problem.dataType = ScalarType::Float4E2M1;
    problem.scaleType = ScalarType::E8M0;
    problem.leadingDimension = 32;
    problem.blockAxis = 0;
    problem.blockSize = 32;
    const MxGenerationResult result = generateMx(problem);
    return result.data.shape() == problem.shape && result.reference.shape() == problem.shape ? 0
                                                                                             : 1;
}
