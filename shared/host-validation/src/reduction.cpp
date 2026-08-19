// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <complex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/reference_reduction.hpp"

namespace roc::host_validation {
ReductionRunInfo referenceReduce(const ReductionRequest& request) {
    const detail::ReductionPlan plan = detail::validateReductionRequest(request);
    switch (request.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            return detail::referenceReductionTyped<float>(request, plan);
        case ScalarType::Float64:
            return detail::referenceReductionTyped<double>(request, plan);
        case ScalarType::Int32:
            return detail::referenceReductionTyped<int32_t>(request, plan);
        case ScalarType::ComplexFloat32:
            return detail::referenceReductionTyped<std::complex<float>>(request, plan);
        case ScalarType::ComplexFloat64:
            return detail::referenceReductionTyped<std::complex<double>>(request, plan);
        default:
            throw std::invalid_argument("Unsupported reference reduction accumulator type.");
    }
}

ReductionResult referenceReduce(const ReductionProblem& problem) {
    const detail::ReductionPlan plan = detail::validateReductionProblem(problem);
    Tensor output(problem.outputType, plan.outputShape);
    ReductionRequest request(problem, output);
    const ReductionRunInfo runInfo = referenceReduce(request);
    return {.output = std::move(output), .runInfo = runInfo};
}

ReductionResult referenceReduce(const ReductionProblem& problem,
                                const TensorStorageAllocator& allocator) {
    const detail::ReductionPlan plan = detail::validateReductionProblem(problem);
    Tensor output(problem.outputType, plan.outputShape, allocator);
    ReductionRequest request(problem, output);
    const ReductionRunInfo runInfo = referenceReduce(request);
    return {.output = std::move(output), .runInfo = runInfo};
}

ReductionRunInfo referenceSum(const ReductionRequest& request) {
    if (request.operation != ReductionOperation::Sum)
        throw std::invalid_argument("referenceSum requires a sum reduction problem.");
    return referenceReduce(request);
}

ReductionResult referenceSum(const ReductionProblem& problem) {
    if (problem.operation != ReductionOperation::Sum)
        throw std::invalid_argument("referenceSum requires a sum reduction problem.");
    return referenceReduce(problem);
}

ReductionResult referenceSum(const ReductionProblem& problem,
                             const TensorStorageAllocator& allocator) {
    if (problem.operation != ReductionOperation::Sum)
        throw std::invalid_argument("referenceSum requires a sum reduction problem.");
    return referenceReduce(problem, allocator);
}

ReductionRunInfo referenceMaximumAbsolute(Tensor input, Tensor output, ScalarType accumulatorType) {
    std::vector<size_t> axes(input.shape().rank());
    std::iota(axes.begin(), axes.end(), 0);
    return referenceReduce(ReductionRequest(std::move(input), std::move(output), accumulatorType,
                                            std::move(axes), ReductionOperation::MaximumAbsolute));
}

ReductionResult referenceMaximumAbsolute(Tensor input, ScalarType outputType,
                                         ScalarType accumulatorType) {
    std::vector<size_t> axes(input.shape().rank());
    std::iota(axes.begin(), axes.end(), 0);
    return referenceReduce(ReductionProblem(std::move(input), outputType, accumulatorType,
                                            std::move(axes), ReductionOperation::MaximumAbsolute));
}

ReductionResult referenceMaximumAbsolute(Tensor input, ScalarType outputType,
                                         ScalarType accumulatorType,
                                         const TensorStorageAllocator& allocator) {
    std::vector<size_t> axes(input.shape().rank());
    std::iota(axes.begin(), axes.end(), 0);
    return referenceReduce(ReductionProblem(std::move(input), outputType, accumulatorType,
                                            std::move(axes), ReductionOperation::MaximumAbsolute),
                           allocator);
}
}  // namespace roc::host_validation
