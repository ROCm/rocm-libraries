// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/linear_combination.hpp"

#include <complex>
#include <stdexcept>

namespace roc::host_numerics {
void linearCombinationInto(std::optional<Tensor> x, std::optional<Tensor> y, Tensor output,
                           const LinearCombinationOptions& options) {
    const detail::LinearCombinationInvocation invocation{
        .x = std::move(x), .y = std::move(y), .output = std::move(output), .options = options};
    detail::validateLinearCombinationInvocation(invocation);
    switch (options.accumulatorType) {
        case ScalarType::Float32:
            return detail::linearCombinationTyped<float>(invocation);
        case ScalarType::Float64:
            return detail::linearCombinationTyped<double>(invocation);
        case ScalarType::ComplexFloat32:
            return detail::linearCombinationTyped<std::complex<float>>(invocation);
        case ScalarType::ComplexFloat64:
            return detail::linearCombinationTyped<std::complex<double>>(invocation);
        default:
            throw std::invalid_argument("Unsupported linear combination accumulator type.");
    }
}

Tensor linearCombination(std::optional<Tensor> x, std::optional<Tensor> y, ScalarType outputType,
                         const LinearCombinationOptions& options) {
    const Shape outputShape = detail::validateLinearCombinationArguments(x, y, outputType, options);
    Tensor output(outputType, outputShape);
    linearCombinationInto(std::move(x), std::move(y), output, options);
    return output;
}

}  // namespace roc::host_numerics
