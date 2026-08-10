// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "detail/reference_epilogue.hpp"

namespace roc::host_validation {
EpilogueRunInfo referenceEpilogue(const EpilogueProblem& problem) {
    detail::validateEpilogue(problem);
    switch (problem.computeType) {
        case ScalarType::Float32:
            return detail::referenceEpilogueTyped<float>(problem);
        case ScalarType::Float64:
            return detail::referenceEpilogueTyped<double>(problem);
        case ScalarType::Int32:
            return detail::referenceEpilogueTyped<int32_t>(problem);
        default:
            throw std::invalid_argument("Unsupported reference epilogue compute type.");
    }
}
}  // namespace roc::host_validation
