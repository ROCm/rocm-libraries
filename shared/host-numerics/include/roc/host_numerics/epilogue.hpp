// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <roc/host_numerics/tensor.hpp>

namespace roc::host_numerics {
enum class ActivationApplication {
    Forward,   // Writes activation(value).
    Gradient,  // Multiplies input by activation'(auxiliaryInput).
};

// Numerical policy and optional inputs for the standalone rank-two epilogue.
struct EpilogueOptions {
    explicit EpilogueOptions(ScalarType compute = ScalarType::Float32)
        : computeType(compute),
          outputScale(Tensor::scalar(compute, 1)),
          auxiliaryScale(Tensor::scalar(compute, 1)) {}

    ScalarType computeType;
    std::optional<Tensor> auxiliaryInput;
    std::optional<Tensor> gateResidual;
    std::optional<Tensor> bias;  // Any shape broadcast-compatible with input.
    Tensor outputScale;
    Tensor auxiliaryScale;
    OutputConversion outputConversion = OutputConversion::Default;
    ActivationFunction activation;
    ActivationApplication activationApplication = ActivationApplication::Forward;
    OutputSelection outputSelection = OutputSelection::all();
    bool accumulateAmax = false;
};

// Scalar types to allocate for an owning epilogue call. Optional types request
// the corresponding additional result tensor.
struct EpilogueOutputTypes {
    ScalarType output = ScalarType::Float32;
    std::optional<ScalarType> rawOutput;
    std::optional<ScalarType> auxiliaryOutput;
    std::optional<ScalarType> amax;
};

struct EpilogueOutputs {
    Tensor output;
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
};

EpilogueOutputs referenceEpilogue(Tensor input, const EpilogueOutputTypes& outputTypes = {},
                                  const EpilogueOptions& options = EpilogueOptions{});

// Writes selected coordinates into caller-owned destinations. The primary
// output may exactly alias input; other overlapping input/output or
// output/output storage is rejected.
void referenceEpilogueInto(Tensor input, EpilogueOutputs outputs,
                           const EpilogueOptions& options = EpilogueOptions{});
}  // namespace roc::host_numerics
