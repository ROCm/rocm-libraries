// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
// Describes population-variance LayerNorm over one tensor axis.
// referenceLayerNorm writes output in full and writes one mean and/or inverse
// variance value per slice when those destinations are present.
struct LayerNormProblem {
    LayerNormProblem(Tensor inputValues, Tensor outputValues, size_t normalizedAxis,
                     ScalarType accumulator)
        : input(std::move(inputValues)),
          output(std::move(outputValues)),
          axis(normalizedAxis),
          accumulatorType(accumulator) {}

    Tensor input;                           // Source tensor.
    Tensor output;                          // Same-shape caller-owned normalized destination.
    std::optional<Tensor> mean;             // Per-slice mean output with axis removed.
    std::optional<Tensor> inverseVariance;  // Per-slice 1 / sqrt(variance + epsilon).
    std::optional<Tensor> gamma;            // Optional scale vector indexed by axis coordinate.
    std::optional<Tensor> beta;             // Optional bias vector indexed by axis coordinate.
    size_t axis;                            // Nonempty dimension normalized per slice.
    ScalarType accumulatorType;             // Float32 or Float64 arithmetic.
    double epsilon = 1e-5;                  // Nonnegative value added to population variance.
};

// A slice fixes every coordinate except axis. Statistics counts are zero when
// the corresponding optional destination is absent.
struct LayerNormRunInfo {
    size_t slicesProcessed = 0;                 // Product of all input extents except axis.
    size_t outputElementsWritten = 0;           // output.shape().elementCount().
    size_t meanElementsWritten = 0;             // slicesProcessed when mean is present.
    size_t inverseVarianceElementsWritten = 0;  // slicesProcessed when present.
};

LayerNormRunInfo referenceLayerNorm(const LayerNormProblem& problem);
}  // namespace roc::host_validation
