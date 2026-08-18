// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <complex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/reference_reduction.hpp"

namespace roc::host_validation {
ReductionRunInfo referenceReduce(const ReductionProblem& problem) {
    const detail::ReductionPlan plan = detail::validateReduction(problem);
    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            return detail::referenceReductionTyped<float>(problem, plan);
        case ScalarType::Float64:
            return detail::referenceReductionTyped<double>(problem, plan);
        case ScalarType::Int32:
            return detail::referenceReductionTyped<int32_t>(problem, plan);
        case ScalarType::ComplexFloat32:
            return detail::referenceReductionTyped<std::complex<float>>(problem, plan);
        case ScalarType::ComplexFloat64:
            return detail::referenceReductionTyped<std::complex<double>>(problem, plan);
        default:
            throw std::invalid_argument("Unsupported reference reduction accumulator type.");
    }
}

ReductionRunInfo referenceSum(const ReductionProblem& problem) {
    if (problem.operation != ReductionOperation::Sum)
        throw std::invalid_argument("referenceSum requires a sum reduction problem.");
    return referenceReduce(problem);
}

ReductionRunInfo referenceMaximumAbsolute(Tensor input, Tensor output, ScalarType accumulatorType) {
    std::vector<size_t> axes(input.shape().rank());
    std::iota(axes.begin(), axes.end(), 0);
    return referenceReduce(ReductionProblem(std::move(input), std::move(output), accumulatorType,
                                            std::move(axes), ReductionOperation::MaximumAbsolute));
}
}  // namespace roc::host_validation
