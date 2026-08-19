// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/operation_types.hpp>
#include <utility>

namespace roc::host_validation {
enum class ActivationApplication {
    Forward,   // Writes activation(value).
    Gradient,  // Multiplies input by activation'(auxiliaryInput).
};

// Reusable standalone elementwise program applied to a rank-two input.
// Optional result types request owned per-element or AMax outputs.
struct EpilogueProblem {
    EpilogueProblem(Tensor inputTensor, ScalarType resultType, ScalarType compute)
        : input(std::move(inputTensor)), outputType(resultType), computeType(compute) {}

    Tensor input;            // Rank-two source read at selected coordinates.
    ScalarType outputType;   // Scalar type of the primary result.
    ScalarType computeType;  // Bias, activation, scaling, and gate arithmetic type.
    std::optional<ScalarType> rawOutputType;        // Scaled pre-gate result type.
    std::optional<ScalarType> auxiliaryOutputType;  // Pre-activation auxiliary result type.
    std::optional<ScalarType> amaxType;             // Max-absolute scalar result type.
    std::optional<Tensor> auxiliaryInput;           // Activation inputs required by Gradient mode.
    std::optional<Tensor> gateResidual;  // Applies gate * value + gate after outputScale.
    std::optional<VectorBinding> bias;   // Row- or column-indexed pre-activation addend.
    std::complex<double> outputScale = {1.0, 0.0};                  // Applied after activation.
    std::complex<double> auxiliaryScale = {1.0, 0.0};               // Applied to auxiliaryOutput.
    OutputConversion outputConversion = OutputConversion::Default;  // Final output encoding.
    Activation activation = Activation::None;  // Forward function or gradient factor.
    ActivationApplication activationApplication = ActivationApplication::Forward;
    double activationParameter0 = 0.0;                         // First activation-specific scalar.
    double activationParameter1 = 0.0;                         // Second activation-specific scalar.
    OutputSelection outputSelection = OutputSelection::all();  // Coordinates to mutate.
};

// Binds an epilogue problem to caller-owned destinations. Unselected output
// coordinates are preserved. accumulateAmax includes an existing AMax value.
struct EpilogueRequest : EpilogueProblem {
    EpilogueRequest(Tensor inputTensor, Tensor outputTensor, ScalarType compute)
        : EpilogueProblem(std::move(inputTensor), outputTensor.type(), compute),
          output(std::move(outputTensor)) {}

    EpilogueRequest(Tensor inputTensor, Tensor outputTensor, std::optional<Tensor> rawOutputTensor,
                    std::optional<Tensor> auxiliaryOutputTensor, std::optional<Tensor> amaxTensor,
                    ScalarType compute)
        : EpilogueProblem(std::move(inputTensor), outputTensor.type(), compute),
          output(std::move(outputTensor)),
          rawOutput(std::move(rawOutputTensor)),
          auxiliaryOutput(std::move(auxiliaryOutputTensor)),
          amax(std::move(amaxTensor)) {
        if (rawOutput) rawOutputType = rawOutput->type();
        if (auxiliaryOutput) auxiliaryOutputType = auxiliaryOutput->type();
        if (amax) amaxType = amax->type();
    }

    EpilogueRequest(EpilogueProblem problem, Tensor outputTensor,
                    std::optional<Tensor> rawOutputTensor = std::nullopt,
                    std::optional<Tensor> auxiliaryOutputTensor = std::nullopt,
                    std::optional<Tensor> amaxTensor = std::nullopt)
        : EpilogueProblem(std::move(problem)),
          output(std::move(outputTensor)),
          rawOutput(std::move(rawOutputTensor)),
          auxiliaryOutput(std::move(auxiliaryOutputTensor)),
          amax(std::move(amaxTensor)) {}

    Tensor output;                          // Primary selected-coordinate destination.
    std::optional<Tensor> rawOutput;        // Scaled pre-gate selected-coordinate destination.
    std::optional<Tensor> auxiliaryOutput;  // Pre-activation selected-coordinate destination.
    std::optional<Tensor> amax;             // One-element max-absolute destination.
    bool accumulateAmax = false;
};

// Counts writes to each caller-owned output.
struct EpilogueRunInfo {
    size_t outputElementsWritten = 0;           // Selected primary output coordinates.
    size_t rawOutputElementsWritten = 0;        // Zero when rawOutput is absent.
    size_t auxiliaryOutputElementsWritten = 0;  // Zero when auxiliaryOutput is absent.
    size_t amaxElementsWritten = 0;             // One when amax is present; zero otherwise.
};

// Owning result tensors are contiguous and initialized to semantic zero before
// selected coordinates are computed. Initialization is not included in runInfo.
struct EpilogueResult {
    Tensor output;
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
    EpilogueRunInfo runInfo;
};

EpilogueRunInfo referenceEpilogue(const EpilogueRequest& request);
EpilogueResult referenceEpilogue(const EpilogueProblem& problem);
EpilogueResult referenceEpilogue(const EpilogueProblem& problem,
                                 const TensorStorageAllocator& allocator);
}  // namespace roc::host_validation
