// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/mx.hpp>

int main() {
    using namespace roc::host_validation;
    MxGenerationProblem problem;
    problem.dataType = ScalarType::Float4E2M1;
    problem.scaleType = ScalarType::E8M0;
    problem.shape = Shape{32, 1};
    problem.leadingDimension = 32;
    problem.blockAxis = 0;
    problem.blockSize = 32;
    const MxGenerationResult result = generateMx(problem);
    return result.data.shape() == problem.shape && result.reference.shape() == problem.shape ? 0
                                                                                             : 1;
}
