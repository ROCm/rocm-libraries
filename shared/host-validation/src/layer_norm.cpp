// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/reference_layer_norm.hpp"

namespace roc::host_validation {
LayerNormRunInfo referenceLayerNorm(const LayerNormProblem& problem) {
    detail::validateLayerNorm(problem);
    if (problem.accumulatorType == ScalarType::Float32)
        return detail::referenceLayerNormTyped<float>(problem);
    return detail::referenceLayerNormTyped<double>(problem);
}
}  // namespace roc::host_validation
