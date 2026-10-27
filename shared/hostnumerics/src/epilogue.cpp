// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <utility>

#include "detail/reference_epilogue.hpp"

namespace roc::hostnumerics {
template <typename Indices>
static void epilogueInto(const Tensor& input, EpilogueOutputs outputs,
                         const EpilogueOptions& options, const Indices& indices,
                         std::span<const size_t> order) {
    detail::EpilogueInvocation invocation(input, std::move(outputs), options);
    detail::bindIndices(invocation, input.shape(), indices, order);
    (void)detail::validateEpilogue(invocation);
    switch (options.computeType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            return detail::referenceEpilogueTyped<float, true>(invocation);
        case ScalarType::Float32:
            return detail::referenceEpilogueTyped<float, false>(invocation);
        case ScalarType::Float64:
            return detail::referenceEpilogueTyped<double, false>(invocation);
        case ScalarType::Int32:
            return detail::referenceEpilogueTyped<int32_t, false>(invocation);
        case ScalarType::ComplexFloat32:
            return detail::referenceEpilogueTyped<std::complex<float>, false>(invocation);
        case ScalarType::ComplexFloat64:
            return detail::referenceEpilogueTyped<std::complex<double>, false>(invocation);
        default:
            throw std::runtime_error("Unsupported reference epilogue compute type.");
    }
}

template <typename Indices>
static EpilogueOutputs epilogueOwned(const Tensor& input, const EpilogueOutputTypes& outputTypes,
                                     const EpilogueOptions& options, const Indices& indices,
                                     std::span<const size_t> order) {
    Tensor output(outputTypes.output, input.shape());
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
    if (outputTypes.rawOutput) rawOutput.emplace(*outputTypes.rawOutput, input.shape());
    if (outputTypes.auxiliaryOutput)
        auxiliaryOutput.emplace(*outputTypes.auxiliaryOutput, input.shape());
    if (outputTypes.amax) amax.emplace(*outputTypes.amax, Shape{1});
    detail::initializeOwnedEpilogueTensor(output);
    if (rawOutput) detail::initializeOwnedEpilogueTensor(*rawOutput);
    if (auxiliaryOutput) detail::initializeOwnedEpilogueTensor(*auxiliaryOutput);
    if (amax) detail::initializeOwnedEpilogueTensor(*amax);
    EpilogueOutputs outputs{
        .output = std::move(output),
        .rawOutput = std::move(rawOutput),
        .auxiliaryOutput = std::move(auxiliaryOutput),
        .amax = std::move(amax),
    };
    epilogueInto(input, outputs, options, indices, order);
    return outputs;
}

void referenceEpilogueInto(const Tensor& input, EpilogueOutputs outputs,
                           const EpilogueOptions& options) {
    epilogueInto(input, std::move(outputs), options, IndexRange::all(input.elementCount()),
                 input.shape().indexOrderLastDimensionFastest());
}
EpilogueOutputs referenceEpilogue(const Tensor& input, const EpilogueOutputTypes& types,
                                  const EpilogueOptions& options) {
    return epilogueOwned(input, types, options, IndexRange::all(input.elementCount()),
                         input.shape().indexOrderLastDimensionFastest());
}
void referenceEpilogueInto(const Tensor& input, EpilogueOutputs outputs,
                           const EpilogueOptions& options, const IndexRange& range,
                           std::span<const size_t> order) {
    epilogueInto(input, std::move(outputs), options, range, order);
}
void referenceEpilogueInto(const Tensor& input, EpilogueOutputs outputs,
                           const EpilogueOptions& options, const std::set<size_t>& indices,
                           std::span<const size_t> order) {
    epilogueInto(input, std::move(outputs), options, indices, order);
}
EpilogueOutputs referenceEpilogue(const Tensor& input, const EpilogueOutputTypes& types,
                                  const EpilogueOptions& options, const IndexRange& range,
                                  std::span<const size_t> order) {
    return epilogueOwned(input, types, options, range, order);
}
EpilogueOutputs referenceEpilogue(const Tensor& input, const EpilogueOutputTypes& types,
                                  const EpilogueOptions& options, const std::set<size_t>& indices,
                                  std::span<const size_t> order) {
    return epilogueOwned(input, types, options, indices, order);
}

}  // namespace roc::hostnumerics
