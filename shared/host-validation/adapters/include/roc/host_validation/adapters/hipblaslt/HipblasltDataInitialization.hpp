// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <hipblaslt_datatype2string.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>

namespace roc::host_validation::hipblaslt_adapter {
template <typename T>
void initialize(std::span<T> values, hipblaslt_initialization initialization,
                DataPattern trigonometricPattern = DataPattern::Cosine) {
    RandomGenerator generator(69069);

    switch (initialization) {
        case hipblaslt_initialization::rand_int:
            fill(values, DataPattern::RandomInteger, generator);
            return;
        case hipblaslt_initialization::trig_float:
            fill(values, trigonometricPattern, generator);
            return;
        case hipblaslt_initialization::hpl:
            fill(values, DataPattern::UniformReal, generator, -0.5, 0.5);
            return;
        case hipblaslt_initialization::uniform_low_precision:
            fill(values, DataPattern::UniformReal, generator, -6.0, 6.0);
            return;
        case hipblaslt_initialization::special:
            fill(values, DataPattern::Constant, generator, 65280.0);
            return;
        case hipblaslt_initialization::zero:
        case hipblaslt_initialization::norm_dist:
        case hipblaslt_initialization::uniform_01:
        case hipblaslt_initialization::integer_exact:
        case hipblaslt_initialization::fp16_accumulator_probe:
        case hipblaslt_initialization::inf:
        case hipblaslt_initialization::neg_zero:
        case hipblaslt_initialization::neg_inf:
        case hipblaslt_initialization::nan:
        case hipblaslt_initialization::norm_dist_one_special:
            fill(values, DataPattern::Zero, generator);
            return;
    }

    fill(values, DataPattern::Zero, generator);
}

template <typename T>
void initialize(T* data, size_t size, hipblaslt_initialization initialization,
                DataPattern trigonometricPattern = DataPattern::Cosine) {
    initialize(std::span<T>(data, size), initialization, trigonometricPattern);
}

template <typename T>
void initializeCosineMatrix(T* data, size_t rows, size_t columns, ptrdiff_t leadingDimension,
                            ptrdiff_t batchStride, size_t batchCount) {
    for (size_t batch = 0; batch < batchCount; ++batch) {
        generate(MatrixView<T>(data + static_cast<ptrdiff_t>(batch) * batchStride, rows, columns, 1,
                               leadingDimension),
                 [=](size_t row, size_t column) {
                     const size_t logicalIndex = row + column * rows + batch * rows * columns;
                     return std::cos(static_cast<double>(logicalIndex));
                 });
    }
}
}  // namespace roc::host_validation::hipblaslt_adapter
