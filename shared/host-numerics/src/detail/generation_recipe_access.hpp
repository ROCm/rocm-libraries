// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_numerics/generation.hpp>
#include <span>

namespace roc::host_numerics::detail {
double generatedNumericalValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                               const Shape& shape, size_t logicalIndex, ScalarType destinationType);
uint64_t generatedRawValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                           const Shape& shape, size_t logicalIndex, ScalarType destinationType);
void generateElement(Tensor destination, const GenerationRecipe& recipe,
                     std::span<const size_t> indices, size_t logicalIndex);
}  // namespace roc::host_numerics::detail
