// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <roc/host_validation/tensor.hpp>

namespace roc::host_validation {
enum class MxGenerationMode {
    Bounded,
    BoundedAlternatingSign,
    Unbounded,
    Identity,
    Ones,
    Zeros,
    Sequential,
    RowIndex,
    ColumnIndex,
    Checkerboard,
    ScaledDiagonal,
    Twos,
    NegativeOnes,
    Maximum,
    DenormalMinimum,
    DenormalMaximum,
    NaN,
    Infinity,
    Trigonometric,
    Normal,
    UniformInteger,
};

struct MxGenerationRecipe {
    MxGenerationMode mode = MxGenerationMode::Bounded;
    double parameter0 = -1.0;
    double parameter1 = 1.0;
};

struct MxGenerationProblem {
    ScalarType dataType = ScalarType::Float4E2M1;
    ScalarType scaleType = ScalarType::E8M0;
    Shape shape;
    ptrdiff_t leadingDimension = 0;
    size_t blockAxis = 0;
    size_t blockSize = 32;
    MxGenerationRecipe data;
    std::optional<MxGenerationRecipe> scale;
    uint32_t seed = 1713573849U;
};

struct MxGenerationResult {
    Tensor data;
    Tensor scales;
    Tensor scaleIndices;
    Tensor reference;
};

MxGenerationResult generateMx(const MxGenerationProblem& problem);
}  // namespace roc::host_validation
