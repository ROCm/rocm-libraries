// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_numerics/generation.hpp>
#include <span>

namespace roc::host_numerics::detail {
struct PreparedNumericalGenerator {
    const GenerationRecipe::Component* component;
    const void* pattern;
    uint64_t seed;
    uint64_t domain;
    ScalarType destinationType;
    double (*value)(const GenerationRecipe::Component&, const void*, uint64_t, uint64_t,
                    std::span<const size_t>, const Shape&, size_t, ScalarType);
    bool usesCoordinates;
    bool isConstant;

    double operator()(std::span<const size_t> indices, const Shape& shape,
                      size_t logicalIndex) const {
        return value(*component, pattern, seed, domain, indices, shape, logicalIndex,
                     destinationType);
    }
};

struct PreparedRawGenerator {
    const void* pattern;
    uint64_t seed;
    uint64_t domain;
    ScalarType destinationType;
    uint64_t (*value)(const void*, uint64_t, uint64_t, std::span<const size_t>, const Shape&,
                      size_t, ScalarType);
    bool usesCoordinates;

    uint64_t operator()(std::span<const size_t> indices, const Shape& shape,
                        size_t logicalIndex) const {
        return value(pattern, seed, domain, indices, shape, logicalIndex, destinationType);
    }
};

bool isRawGenerationComponent(const GenerationRecipe::Component& component);
PreparedNumericalGenerator prepareNumericalGenerator(const GenerationRecipe& recipe,
                                                     ScalarType destinationType);
PreparedRawGenerator prepareRawGenerator(const GenerationRecipe& recipe,
                                         ScalarType destinationType);
double generatedNumericalValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                               const Shape& shape, size_t logicalIndex, ScalarType destinationType);
uint64_t generatedRawValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                           const Shape& shape, size_t logicalIndex, ScalarType destinationType);
void generateElement(Tensor destination, const GenerationRecipe& recipe,
                     std::span<const size_t> indices, size_t logicalIndex);
void generateTensor(Tensor destination, const GenerationRecipe& recipe);
}  // namespace roc::host_numerics::detail
