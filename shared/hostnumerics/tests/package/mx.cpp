// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/mx.hpp>

TEST(Package, Mx) {
    using namespace roc::hostnumerics;
    const Shape shape{32, 1};
    const MxDataGeneration generation = MxDataGeneration::preserveRange(
        GenerationRecipe::realOnly(GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0}), 0,
                                   LastDimensionFastest{}),
        {.lower = -1.0, .upper = 1.0});
    MxGenerationOptions options{
        .dataType = ScalarType::Float4E2M1,
        .scaleType = ScalarType::E8M0,
        .leadingDimension = 32,
        .blockAxis = 0,
        .blockSize = 32,
        .scale = MxScaleGenerationMode::Derived,
    };
    const MxTensor result = generateMx(shape, generation, options);
    EXPECT_EQ(result.data.shape(), shape);
    EXPECT_EQ(result.reference.shape(), shape);
}
