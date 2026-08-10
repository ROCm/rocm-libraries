// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/reference_softmax.hpp"

namespace roc::host_validation {
SoftmaxRunInfo referenceSoftmax(const SoftmaxProblem& problem) {
    detail::validateSoftmax(problem);
    if (problem.accumulatorType == ScalarType::Float32)
        return detail::referenceSoftmaxTyped<float>(problem);
    return detail::referenceSoftmaxTyped<double>(problem);
}
}  // namespace roc::host_validation
