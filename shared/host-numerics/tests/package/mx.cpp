// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_numerics/mx.hpp>

int main() {
    using namespace roc::host_numerics;
    const Shape shape{32, 1};
    const MxDataGeneration generation = MxDataGeneration::preserveRange(
        GenerationRecipe::realOnly(GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0})),
        {.lower = -1.0, .upper = 1.0});
    MxGenerationOptions options;
    options.dataType = ScalarType::Float4E2M1;
    options.scaleType = ScalarType::E8M0;
    options.leadingDimension = 32;
    options.blockAxis = 0;
    options.blockSize = 32;
    const MxTensor result = generateMx(shape, generation, options);
    return result.data.shape() == shape && result.reference.shape() == shape ? 0 : 1;
}
