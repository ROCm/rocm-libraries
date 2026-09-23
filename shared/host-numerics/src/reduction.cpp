// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <complex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/reference_reduction.hpp"

namespace roc::host_numerics {
void referenceReduceInto(Tensor input, Tensor output, std::vector<size_t> axes,
                         ReductionOperation operation, ScalarType accumulatorType) {
    const detail::ReductionInvocation invocation{
        .input = std::move(input),
        .output = std::move(output),
        .axes = std::move(axes),
        .operation = operation,
        .accumulatorType = accumulatorType,
    };
    const detail::ReductionPlan plan = detail::validateReductionInvocation(invocation);
    switch (accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            return detail::referenceReductionTyped<float>(invocation, plan);
        case ScalarType::Float64:
            return detail::referenceReductionTyped<double>(invocation, plan);
        case ScalarType::Int32:
            return detail::referenceReductionTyped<int32_t>(invocation, plan);
        case ScalarType::ComplexFloat32:
            return detail::referenceReductionTyped<std::complex<float>>(invocation, plan);
        case ScalarType::ComplexFloat64:
            return detail::referenceReductionTyped<std::complex<double>>(invocation, plan);
        default:
            throw std::invalid_argument("Unsupported reference reduction accumulator type.");
    }
}

Tensor referenceReduce(Tensor input, std::vector<size_t> axes, ReductionOperation operation,
                       ScalarType outputType, ScalarType accumulatorType) {
    const detail::ReductionPlan plan =
        detail::validateReductionArguments(input, axes, operation, outputType, accumulatorType);
    Tensor output(outputType, plan.outputShape);
    referenceReduceInto(std::move(input), output, std::move(axes), operation, accumulatorType);
    return output;
}

void referenceSumInto(Tensor input, Tensor output, std::vector<size_t> axes,
                      ScalarType accumulatorType) {
    referenceReduceInto(std::move(input), std::move(output), std::move(axes),
                        ReductionOperation::Sum, accumulatorType);
}

Tensor referenceSum(Tensor input, std::vector<size_t> axes, ScalarType outputType,
                    ScalarType accumulatorType) {
    return referenceReduce(std::move(input), std::move(axes), ReductionOperation::Sum, outputType,
                           accumulatorType);
}

void referenceMaximumAbsoluteInto(Tensor input, Tensor output, ScalarType accumulatorType) {
    std::vector<size_t> axes(input.shape().rank());
    std::iota(axes.begin(), axes.end(), 0);
    referenceReduceInto(std::move(input), std::move(output), std::move(axes),
                        ReductionOperation::MaximumAbsolute, accumulatorType);
}

Tensor referenceMaximumAbsolute(Tensor input, ScalarType outputType, ScalarType accumulatorType) {
    std::vector<size_t> axes(input.shape().rank());
    std::iota(axes.begin(), axes.end(), 0);
    return referenceReduce(std::move(input), std::move(axes), ReductionOperation::MaximumAbsolute,
                           outputType, accumulatorType);
}

}  // namespace roc::host_numerics
