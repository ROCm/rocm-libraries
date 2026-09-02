// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <utility>

#include "detail/reference_layer_norm.hpp"

namespace roc::host_numerics {
void referenceLayerNormInto(Tensor input, LayerNormOutputs outputs,
                            const LayerNormOptions& options) {
    const detail::LayerNormInvocation invocation{
        .input = std::move(input), .outputs = std::move(outputs), .options = options};
    const detail::LayerNormPlan plan = detail::validateLayerNormInvocation(invocation);
    if (options.accumulatorType == ScalarType::Float32)
        return detail::referenceLayerNormTyped<float>(invocation, plan);
    return detail::referenceLayerNormTyped<double>(invocation, plan);
}

LayerNormOutputs referenceLayerNorm(Tensor input, const LayerNormOutputTypes& outputTypes,
                                    const LayerNormOptions& options) {
    const detail::LayerNormPlan plan =
        detail::validateLayerNormArguments(input, outputTypes, options);
    Tensor output(outputTypes.output, input.shape());
    std::optional<Tensor> mean;
    std::optional<Tensor> inverseVariance;
    if (outputTypes.mean) mean.emplace(*outputTypes.mean, plan.statisticsShape);
    if (outputTypes.inverseVariance)
        inverseVariance.emplace(*outputTypes.inverseVariance, plan.statisticsShape);
    LayerNormOutputs outputs{
        .output = std::move(output),
        .mean = std::move(mean),
        .inverseVariance = std::move(inverseVariance),
    };
    referenceLayerNormInto(std::move(input), outputs, options);
    return outputs;
}

}  // namespace roc::host_numerics
