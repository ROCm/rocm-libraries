// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipblaslt_datatype2string.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <vector>

namespace roc::host_validation::hipblaslt_adapter {
template <typename InputA, typename InputB, typename Output>
void initializeGroupedGemm(std::vector<InputA>& a, int64_t sizeA, std::vector<InputB>& b,
                           int64_t sizeB, std::vector<Output>& c, int64_t sizeC,
                           std::vector<float>& bias, int64_t sizeBias,
                           hipblaslt_initialization initialization) {
    RandomGenerator generator(69069);

    auto fillValues = [&](auto& values, int64_t size, DataPattern pattern, double lower = 0.0,
                          double upper = 0.0) {
        fill(std::span(values.data(), static_cast<size_t>(size)), pattern, generator, lower, upper);
    };

    if (initialization == hipblaslt_initialization::rand_int) {
        fillValues(a, sizeA, DataPattern::RandomInteger);
        fillValues(b, sizeB, DataPattern::AlternatingRandomInteger);
        fillValues(c, sizeC, DataPattern::RandomInteger);
        fillValues(bias, sizeBias, DataPattern::RandomInteger);
    } else if (initialization == hipblaslt_initialization::trig_float) {
        fillValues(a, sizeA, DataPattern::Sine);
        fillValues(b, sizeB, DataPattern::Cosine);
        fillValues(c, sizeC, DataPattern::Sine);
        fillValues(bias, sizeBias, DataPattern::Sine);
    } else if (initialization == hipblaslt_initialization::hpl) {
        fillValues(a, sizeA, DataPattern::UniformReal, -0.5, 0.5);
        fillValues(b, sizeB, DataPattern::UniformReal, -0.5, 0.5);
        fillValues(c, sizeC, DataPattern::UniformReal, -0.5, 0.5);
        fillValues(bias, sizeBias, DataPattern::UniformReal, -0.5, 0.5);
    } else if (initialization == hipblaslt_initialization::uniform_low_precision) {
        fillValues(a, sizeA, DataPattern::UniformReal, -6.0, 6.0);
        fillValues(b, sizeB, DataPattern::UniformReal, -6.0, 6.0);
        fillValues(c, sizeC, DataPattern::UniformReal, -6.0, 6.0);
        fillValues(bias, sizeBias, DataPattern::UniformReal, -6.0, 6.0);
    } else if (initialization == hipblaslt_initialization::special) {
        fillValues(a, sizeA, DataPattern::Constant, 65280.0);
        fillValues(b, sizeB, DataPattern::Constant, 0.0000607967376708984375);
        fillValues(c, sizeC, DataPattern::UniformReal, -0.5, 0.5);
        fillValues(bias, sizeBias, DataPattern::UniformReal, -0.5, 0.5);
    } else {
        fillValues(a, sizeA, DataPattern::Zero);
        fillValues(b, sizeB, DataPattern::Zero);
        fillValues(c, sizeC, DataPattern::Zero);
        fillValues(bias, sizeBias, DataPattern::Zero);
    }
}
}  // namespace roc::host_validation::hipblaslt_adapter
