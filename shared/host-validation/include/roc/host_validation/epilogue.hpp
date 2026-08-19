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

// Describes the standalone elementwise program applied to a rank-two input.
// referenceEpilogue writes the selected coordinates of output and every
// present per-element output. If amax is present, it writes that scalar once
// after processing the selection.
struct EpilogueProblem {
    EpilogueProblem(Tensor inputTensor, Tensor outputTensor, ScalarType compute)
        : input(std::move(inputTensor)), output(std::move(outputTensor)), computeType(compute) {}

    Tensor input;            // Rank-two source read at selected output coordinates.
    Tensor output;           // Caller-owned destination written at selected coordinates.
    ScalarType computeType;  // Type used for bias, activation, scaling, and gate arithmetic.
    std::optional<Tensor> rawOutput;        // Scaled pre-gate values at selected coordinates.
    std::optional<Tensor> auxiliaryOutput;  // Pre-activation values times auxiliaryScale.
    std::optional<Tensor> auxiliaryInput;   // Activation inputs required by Gradient mode.
    std::optional<Tensor> gateResidual;     // Applies gate * value + gate after outputScale.
    std::optional<Tensor> amax;         // One-element output for max(abs(pre-scale activation)).
    bool accumulateAmax = false;        // Includes the existing amax value in the maximum.
    std::optional<VectorBinding> bias;  // Row- or column-indexed pre-activation addend.
    std::complex<double> outputScale = {1.0, 0.0};                  // Applied after activation.
    std::complex<double> auxiliaryScale = {1.0, 0.0};               // Applied to auxiliaryOutput.
    OutputConversion outputConversion = OutputConversion::Default;  // Final output encoding.
    Activation activation = Activation::None;  // Forward function or gradient factor.
    ActivationApplication activationApplication = ActivationApplication::Forward;
    double activationParameter0 = 0.0;                         // First activation-specific scalar.
    double activationParameter1 = 0.0;                         // Second activation-specific scalar.
    OutputSelection outputSelection = OutputSelection::all();  // Coordinates to mutate.
};

// Counts writes to each caller-owned output.
struct EpilogueRunInfo {
    size_t outputElementsWritten = 0;           // Selected primary output coordinates.
    size_t rawOutputElementsWritten = 0;        // Zero when rawOutput is absent.
    size_t auxiliaryOutputElementsWritten = 0;  // Zero when auxiliaryOutput is absent.
    size_t amaxElementsWritten = 0;             // One when amax is present; zero otherwise.
};

EpilogueRunInfo referenceEpilogue(const EpilogueProblem& problem);
}  // namespace roc::host_validation
