// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <utility>

#include "detail/reference_epilogue.hpp"

namespace roc::host_numerics {
void referenceEpilogueInto(Tensor input, EpilogueOutputs outputs, const EpilogueOptions& options) {
    const detail::EpilogueInvocation invocation(std::move(input), std::move(outputs), options);
    (void)detail::validateEpilogue(invocation);
    switch (options.computeType) {
        case ScalarType::Float32:
            return detail::referenceEpilogueTyped<float>(invocation);
        case ScalarType::Float64:
            return detail::referenceEpilogueTyped<double>(invocation);
        case ScalarType::Int32:
            return detail::referenceEpilogueTyped<int32_t>(invocation);
        default:
            throw std::invalid_argument("Unsupported reference epilogue compute type.");
    }
}

EpilogueOutputs referenceEpilogue(Tensor input, const EpilogueOutputTypes& outputTypes,
                                  const EpilogueOptions& options) {
    Tensor output(outputTypes.output, input.shape());
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
    if (outputTypes.rawOutput) rawOutput.emplace(*outputTypes.rawOutput, input.shape());
    if (outputTypes.auxiliaryOutput)
        auxiliaryOutput.emplace(*outputTypes.auxiliaryOutput, input.shape());
    if (outputTypes.amax) amax.emplace(*outputTypes.amax, Shape{1});
    detail::initializeOwnedEpilogueTensor(output);
    if (rawOutput) detail::initializeOwnedEpilogueTensor(*rawOutput);
    if (auxiliaryOutput) detail::initializeOwnedEpilogueTensor(*auxiliaryOutput);
    if (amax) detail::initializeOwnedEpilogueTensor(*amax);
    EpilogueOutputs outputs{
        .output = std::move(output),
        .rawOutput = std::move(rawOutput),
        .auxiliaryOutput = std::move(auxiliaryOutput),
        .amax = std::move(amax),
    };
    referenceEpilogueInto(std::move(input), outputs, options);
    return outputs;
}

}  // namespace roc::host_numerics
