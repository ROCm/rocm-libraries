// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <complex>
#include <stdexcept>

#include "detail/reference_axpby.hpp"

namespace roc::host_numerics {
AxpbyRunInfo referenceAxpby(const AxpbyRequest& request) {
    detail::validateAxpbyRequest(request);
    switch (request.accumulatorType) {
        case ScalarType::Float32:
            return detail::referenceAxpbyTyped<float>(request);
        case ScalarType::Float64:
            return detail::referenceAxpbyTyped<double>(request);
        case ScalarType::ComplexFloat32:
            return detail::referenceAxpbyTyped<std::complex<float>>(request);
        case ScalarType::ComplexFloat64:
            return detail::referenceAxpbyTyped<std::complex<double>>(request);
        default:
            throw std::invalid_argument("Unsupported reference AXPBY accumulator type.");
    }
}

AxpbyResult referenceAxpby(const AxpbyProblem& problem) {
    const Shape& outputShape = detail::validateAxpbyProblem(problem);
    Tensor output(problem.outputType, outputShape);
    AxpbyRequest request(problem, output);
    const AxpbyRunInfo runInfo = referenceAxpby(request);
    return {.output = std::move(output), .runInfo = runInfo};
}

}  // namespace roc::host_numerics
