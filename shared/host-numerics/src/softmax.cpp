// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <utility>

#include "detail/reference_softmax.hpp"

namespace roc::host_numerics {
void referenceSoftmaxInto(Tensor input, Tensor output, size_t axis, ScalarType accumulatorType) {
    const detail::SoftmaxInvocation invocation{
        .input = std::move(input),
        .output = std::move(output),
        .axis = axis,
        .accumulatorType = accumulatorType,
    };
    detail::validateSoftmaxInvocation(invocation);
    if (accumulatorType == ScalarType::Float32)
        return detail::referenceSoftmaxTyped<float>(invocation);
    return detail::referenceSoftmaxTyped<double>(invocation);
}

Tensor referenceSoftmax(Tensor input, size_t axis, ScalarType outputType,
                        ScalarType accumulatorType) {
    const Shape outputShape =
        detail::validateSoftmaxArguments(input, outputType, axis, accumulatorType);
    Tensor output(outputType, Layout::contiguousLastDimensionFastest(outputShape));
    referenceSoftmaxInto(std::move(input), output, axis, accumulatorType);
    return output;
}

}  // namespace roc::host_numerics
