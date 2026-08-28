// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <utility>

#include "detail/reference_layer_norm.hpp"

namespace roc::host_numerics {
LayerNormRunInfo referenceLayerNorm(const LayerNormRequest& request) {
    const detail::LayerNormPlan plan = detail::validateLayerNormRequest(request);
    if (request.accumulatorType == ScalarType::Float32)
        return detail::referenceLayerNormTyped<float>(request, plan);
    return detail::referenceLayerNormTyped<double>(request, plan);
}

LayerNormResult referenceLayerNorm(const LayerNormProblem& problem) {
    const detail::LayerNormPlan plan = detail::validateLayerNormProblem(problem);
    Tensor output(problem.outputType, problem.input.shape());
    std::optional<Tensor> mean;
    std::optional<Tensor> inverseVariance;
    if (problem.meanType) mean.emplace(*problem.meanType, plan.statisticsShape);
    if (problem.inverseVarianceType)
        inverseVariance.emplace(*problem.inverseVarianceType, plan.statisticsShape);
    LayerNormRequest request(problem, output, mean, inverseVariance);
    const LayerNormRunInfo runInfo = referenceLayerNorm(request);
    return {
        .output = std::move(output),
        .mean = std::move(mean),
        .inverseVariance = std::move(inverseVariance),
        .runInfo = runInfo,
    };
}

}  // namespace roc::host_numerics
