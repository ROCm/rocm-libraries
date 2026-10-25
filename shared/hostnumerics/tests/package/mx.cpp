// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/mx.hpp>

TEST(Package, Mx) {
    using namespace roc::hostnumerics;
    const Shape shape{32, 1};
    const MxDataGeneration generation = MxDataGeneration::preserveRange(
        GenerationRecipe::realOnlyLastDimensionFastest(
            GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}), 0),
        {.lower = -1.0, .upper = 1.0});
    MxGenerationOptions options;
    options.dataType = ScalarType::Float4E2M1;
    options.scaleType = ScalarType::E8M0;
    options.leadingDimension = 32;
    options.blockAxis = 0;
    options.blockSize = 32;
    const MxTensor result = generateMx(shape, generation, options);
    EXPECT_EQ(result.data.shape(), shape);
    EXPECT_EQ(result.reference.shape(), shape);
}
