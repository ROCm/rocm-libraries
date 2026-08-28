// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <utility>

#include "detail/reference_epilogue.hpp"

namespace roc::host_numerics {
EpilogueRunInfo referenceEpilogue(const EpilogueRequest& request) {
    (void)detail::validateEpilogueRequest(request);
    switch (request.computeType) {
        case ScalarType::Float32:
            return detail::referenceEpilogueTyped<float>(request);
        case ScalarType::Float64:
            return detail::referenceEpilogueTyped<double>(request);
        case ScalarType::Int32:
            return detail::referenceEpilogueTyped<int32_t>(request);
        default:
            throw std::invalid_argument("Unsupported reference epilogue compute type.");
    }
}

EpilogueResult referenceEpilogue(const EpilogueProblem& problem) {
    (void)detail::validateEpilogueProblem(problem);
    Tensor output(problem.outputType, problem.input.shape());
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
    if (problem.rawOutputType) rawOutput.emplace(*problem.rawOutputType, problem.input.shape());
    if (problem.auxiliaryOutputType)
        auxiliaryOutput.emplace(*problem.auxiliaryOutputType, problem.input.shape());
    if (problem.amaxType) amax.emplace(*problem.amaxType, Shape{1});
    EpilogueRequest request(problem, output, rawOutput, auxiliaryOutput, amax);
    (void)detail::validateEpilogueRequest(request);
    detail::initializeOwnedEpilogueTensor(output);
    if (rawOutput) detail::initializeOwnedEpilogueTensor(*rawOutput);
    if (auxiliaryOutput) detail::initializeOwnedEpilogueTensor(*auxiliaryOutput);
    if (amax) detail::initializeOwnedEpilogueTensor(*amax);
    const EpilogueRunInfo runInfo = referenceEpilogue(request);
    return {
        .output = std::move(output),
        .rawOutput = std::move(rawOutput),
        .auxiliaryOutput = std::move(auxiliaryOutput),
        .amax = std::move(amax),
        .runInfo = runInfo,
    };
}

}  // namespace roc::host_numerics
