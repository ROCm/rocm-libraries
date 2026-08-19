// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <utility>

#include "detail/reference_softmax.hpp"

namespace roc::host_validation {
SoftmaxRunInfo referenceSoftmax(const SoftmaxRequest& request) {
    detail::validateSoftmaxRequest(request);
    if (request.accumulatorType == ScalarType::Float32)
        return detail::referenceSoftmaxTyped<float>(request);
    return detail::referenceSoftmaxTyped<double>(request);
}

SoftmaxResult referenceSoftmax(const SoftmaxProblem& problem) {
    const Shape& outputShape = detail::validateSoftmaxProblem(problem);
    Tensor output(problem.outputType, Layout::contiguous(outputShape));
    SoftmaxRequest request(problem, output);
    const SoftmaxRunInfo runInfo = referenceSoftmax(request);
    return {.output = std::move(output), .runInfo = runInfo};
}

SoftmaxResult referenceSoftmax(const SoftmaxProblem& problem,
                               const TensorStorageAllocator& allocator) {
    const Shape& outputShape = detail::validateSoftmaxProblem(problem);
    Tensor output(problem.outputType, Layout::contiguous(outputShape), allocator);
    SoftmaxRequest request(problem, output);
    const SoftmaxRunInfo runInfo = referenceSoftmax(request);
    return {.output = std::move(output), .runInfo = runInfo};
}
}  // namespace roc::host_validation
