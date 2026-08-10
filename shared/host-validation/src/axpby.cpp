// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <complex>
#include <stdexcept>

#include "detail/reference_axpby.hpp"

namespace roc::host_validation {
AxpbyRunInfo referenceAxpby(const AxpbyProblem& problem) {
    detail::validateAxpby(problem);
    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            return detail::referenceAxpbyTyped<float>(problem);
        case ScalarType::Float64:
            return detail::referenceAxpbyTyped<double>(problem);
        case ScalarType::ComplexFloat32:
            return detail::referenceAxpbyTyped<std::complex<float>>(problem);
        case ScalarType::ComplexFloat64:
            return detail::referenceAxpbyTyped<std::complex<double>>(problem);
        default:
            throw std::invalid_argument("Unsupported reference AXPBY accumulator type.");
    }
}
}  // namespace roc::host_validation
