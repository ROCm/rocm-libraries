// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/hostnumerics/generation.hpp>
#include <span>

#include "generation_values.hpp"

namespace roc::hostnumerics::detail {
struct PreparedGenerationValues {
    double constant = 0.0;
    double maximum = 0.0;
    std::span<const uint8_t> finiteEncodings;
    PreparedEncodedExponent exponent;
};
struct PreparedNumericalGenerator {
    const GenerationRecipe::Component* component;
    const void* pattern;
    uint64_t seed;
    uint64_t domain;
    ScalarType destinationType;
    double (*value)(const GenerationRecipe::Component&, const void*, uint64_t, uint64_t,
                    std::span<const size_t>, const Shape&, size_t, ScalarType,
                    const PreparedGenerationValues&);
    PreparedGenerationValues values;
    bool usesCoordinates;
    bool isConstant;
    size_t indexOffset = 0;

    double operator()(std::span<const size_t> indices, const Shape& shape,
                      size_t logicalIndex) const {
        return value(*component, pattern, seed, domain, indices, shape, logicalIndex + indexOffset,
                     destinationType, values);
    }
};

struct PreparedRawGenerator {
    const void* pattern;
    uint64_t seed;
    uint64_t domain;
    ScalarType destinationType;
    uint64_t (*value)(const void*, uint64_t, uint64_t, std::span<const size_t>, const Shape&,
                      size_t, ScalarType, const PreparedGenerationValues&);
    PreparedGenerationValues values;
    bool usesCoordinates;
    size_t indexOffset = 0;

    uint64_t operator()(std::span<const size_t> indices, const Shape& shape,
                        size_t logicalIndex) const {
        return value(pattern, seed, domain, indices, shape, logicalIndex + indexOffset,
                     destinationType, values);
    }
};

bool isRawGenerationComponent(const GenerationRecipe::Component& component);
PreparedNumericalGenerator prepareNumericalGenerator(const GenerationRecipe& recipe,
                                                     ScalarType destinationType,
                                                     const Shape& shape);
PreparedRawGenerator prepareRawGenerator(const GenerationRecipe& recipe, ScalarType destinationType,
                                         const Shape& shape);
double generatedNumericalValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                               const Shape& shape, size_t logicalIndex, ScalarType destinationType);
uint64_t generatedRawValue(const GenerationRecipe& recipe, std::span<const size_t> indices,
                           const Shape& shape, size_t logicalIndex, ScalarType destinationType);
void generateElement(Tensor destination, const GenerationRecipe& recipe,
                     std::span<const size_t> indices, size_t logicalIndex);
void generateTensor(Tensor destination, const GenerationRecipe& recipe);
void generateTensor(Tensor destination, const GenerationRecipe& recipe,
                    const ScalarConversionOptions& options);
void generateElement(Tensor destination, const GenerationRecipe& recipe,
                     std::span<const size_t> indices, size_t logicalIndex,
                     const ScalarConversionOptions& options);
}  // namespace roc::hostnumerics::detail
