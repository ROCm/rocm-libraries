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
    Forward,
    Gradient,
};

struct EpilogueProblem {
    EpilogueProblem(Tensor inputTensor, Tensor outputTensor, ScalarType compute)
        : input(std::move(inputTensor)), output(std::move(outputTensor)), computeType(compute) {}

    Tensor input;
    Tensor output;
    ScalarType computeType;
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> auxiliaryInput;
    std::optional<Tensor> gateResidual;
    std::optional<Tensor> amax;
    bool accumulateAmax = false;
    std::optional<VectorBinding> bias;
    std::complex<double> outputScale = {1.0, 0.0};
    std::complex<double> auxiliaryScale = {1.0, 0.0};
    OutputConversion outputConversion = OutputConversion::Default;
    Activation activation = Activation::None;
    ActivationApplication activationApplication = ActivationApplication::Forward;
    double activationParameter0 = 0.0;
    double activationParameter1 = 0.0;
    OutputSelection outputSelection = OutputSelection::all();
};

struct EpilogueRunInfo {
    size_t outputElementsComputed = 0;
};

EpilogueRunInfo referenceEpilogue(const EpilogueProblem& problem);
}  // namespace roc::host_validation
