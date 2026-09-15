// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/host_numerics/generation.hpp>

namespace roc::host_numerics::detail {
double indexedUniformUnit(uint64_t seed, uint64_t domain, uint64_t index);
ScalarType generationComponentType(ScalarType type);
double typeMaximum(ScalarType requestedType);
double typeLowest(ScalarType requestedType);
double typeDenormalMinimum(ScalarType requestedType);
double typeDenormalMaximum(ScalarType requestedType);
double typeNaN(ScalarType type);
double typeInfinity(ScalarType type, bool negative);
double randomEncodedExponentValue(const RandomEncodedExponentGenerationParameters& parameters,
                                  uint64_t seed, uint64_t domain, size_t logicalIndex,
                                  ScalarType destinationType);
uint64_t uniformFiniteEncodedValue(ScalarType type, uint64_t seed, uint64_t domain,
                                   size_t logicalIndex);
}  // namespace roc::host_numerics::detail
