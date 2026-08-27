// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
// Reusable population-variance LayerNorm descriptor. Optional statistic types
// request one mean and/or inverse-variance value per normalization slice.
struct LayerNormProblem {
    LayerNormProblem(Tensor inputValues, ScalarType resultType, size_t normalizedAxis,
                     ScalarType accumulator)
        : input(std::move(inputValues)),
          outputType(resultType),
          axis(normalizedAxis),
          accumulatorType(accumulator) {}

    Tensor input;                                   // Source tensor.
    ScalarType outputType;                          // Scalar type of the normalized result.
    std::optional<ScalarType> meanType;             // Requests per-slice means when present.
    std::optional<ScalarType> inverseVarianceType;  // Requests 1 / sqrt(variance + epsilon).
    std::optional<Tensor> gamma;  // Optional scale vector indexed by axis coordinate.
    std::optional<Tensor> beta;   // Optional bias vector indexed by axis coordinate.
    size_t axis;                  // Nonempty dimension normalized per slice.
    ScalarType accumulatorType;   // Float32 or Float64 arithmetic.
    double epsilon = 1e-5;        // Finite nonnegative population-variance offset.
};

// Binds a LayerNorm problem to caller-owned output and optional statistic
// destinations. Destination presence and scalar types must match the problem.
struct LayerNormRequest : LayerNormProblem {
    LayerNormRequest(Tensor inputValues, Tensor outputValues, std::optional<Tensor> meanValues,
                     std::optional<Tensor> inverseVarianceValues, size_t normalizedAxis,
                     ScalarType accumulator)
        : LayerNormProblem(std::move(inputValues), outputValues.type(), normalizedAxis,
                           accumulator),
          output(std::move(outputValues)),
          mean(std::move(meanValues)),
          inverseVariance(std::move(inverseVarianceValues)) {
        if (mean) meanType = mean->type();
        if (inverseVariance) inverseVarianceType = inverseVariance->type();
    }

    LayerNormRequest(Tensor inputValues, Tensor outputValues, size_t normalizedAxis,
                     ScalarType accumulator)
        : LayerNormRequest(std::move(inputValues), std::move(outputValues), std::nullopt,
                           std::nullopt, normalizedAxis, accumulator) {}

    LayerNormRequest(LayerNormProblem problem, Tensor outputValues,
                     std::optional<Tensor> meanValues = std::nullopt,
                     std::optional<Tensor> inverseVarianceValues = std::nullopt)
        : LayerNormProblem(std::move(problem)),
          output(std::move(outputValues)),
          mean(std::move(meanValues)),
          inverseVariance(std::move(inverseVarianceValues)) {}

    Tensor output;                          // Same-shape normalized destination written in full.
    std::optional<Tensor> mean;             // Per-slice destination with axis removed.
    std::optional<Tensor> inverseVariance;  // Per-slice destination with axis removed.
};

// A slice fixes every coordinate except axis. Statistics counts are zero when
// the corresponding optional destination is absent.
struct LayerNormRunInfo {
    size_t slicesProcessed = 0;                 // Product of all input extents except axis.
    size_t outputElementsWritten = 0;           // output.shape().elementCount().
    size_t meanElementsWritten = 0;             // slicesProcessed when mean is present.
    size_t inverseVarianceElementsWritten = 0;  // slicesProcessed when present.
};

// Owning LayerNorm result. Present tensors use contiguous layouts; callers
// requiring specific destination layouts use LayerNormRequest.
struct LayerNormResult {
    Tensor output;
    std::optional<Tensor> mean;
    std::optional<Tensor> inverseVariance;
    LayerNormRunInfo runInfo;
};

LayerNormRunInfo referenceLayerNorm(const LayerNormRequest& request);
LayerNormResult referenceLayerNorm(const LayerNormProblem& problem);
}  // namespace roc::host_validation
