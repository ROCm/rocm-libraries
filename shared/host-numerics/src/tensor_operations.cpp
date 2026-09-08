// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/tensor_operations.hpp"

#include <utility>

namespace roc::host_numerics {
namespace {
Tensor binary(detail::BinaryOperation operation, Tensor left, Tensor right, ScalarType outputType,
              ScalarType computeType, std::optional<Layout> outputLayout) {
    const Shape outputShape = detail::binaryOutputShape(left, right);
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::invalid_argument("Elementwise output layout shape mismatch.");
    Tensor output(outputType, layout);
    detail::runBinary(
        {std::move(left), std::move(right), output, computeType, OutputSelection::all()},
        operation);
    return output;
}

void binaryInto(detail::BinaryOperation operation, Tensor left, Tensor right, Tensor output,
                ScalarType computeType, OutputSelection selection) {
    detail::runBinary(
        {std::move(left), std::move(right), std::move(output), computeType, std::move(selection)},
        operation);
}

Tensor binarySameType(detail::BinaryOperation operation, Tensor left, Tensor right) {
    if (left.type() != right.type())
        throw std::invalid_argument(
            "Implicit elementwise operations require operands with the same type.");
    const ScalarType type = left.type();
    return binary(operation, std::move(left), std::move(right), type,
                  detail::implicitElementwiseComputeType(type), std::nullopt);
}
}  // namespace

Tensor add(Tensor left, Tensor right) {
    return binarySameType(detail::BinaryOperation::Add, std::move(left), std::move(right));
}

Tensor add(Tensor left, Tensor right, ScalarType outputType, ScalarType computeType,
           std::optional<Layout> outputLayout) {
    return binary(detail::BinaryOperation::Add, std::move(left), std::move(right), outputType,
                  computeType, std::move(outputLayout));
}

void addInto(Tensor left, Tensor right, Tensor output, ScalarType computeType,
             OutputSelection selection) {
    binaryInto(detail::BinaryOperation::Add, std::move(left), std::move(right), std::move(output),
               computeType, std::move(selection));
}

Tensor multiply(Tensor left, Tensor right) {
    return binarySameType(detail::BinaryOperation::Multiply, std::move(left), std::move(right));
}

Tensor multiply(Tensor left, Tensor right, ScalarType outputType, ScalarType computeType,
                std::optional<Layout> outputLayout) {
    return binary(detail::BinaryOperation::Multiply, std::move(left), std::move(right), outputType,
                  computeType, std::move(outputLayout));
}

void multiplyInto(Tensor left, Tensor right, Tensor output, ScalarType computeType,
                  OutputSelection selection) {
    binaryInto(detail::BinaryOperation::Multiply, std::move(left), std::move(right),
               std::move(output), computeType, std::move(selection));
}

Tensor activate(Tensor input, ActivationFunction function) {
    const ScalarType type = input.type();
    return activate(std::move(input), std::move(function), type,
                    detail::implicitElementwiseComputeType(type));
}

Tensor activate(Tensor input, ActivationFunction function, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout) {
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(input.shape()));
    if (layout.shape() != input.shape())
        throw std::invalid_argument("Activation output layout shape mismatch.");
    Tensor output(outputType, layout);
    detail::runActivation(
        {std::move(input), output, std::move(function), computeType, OutputSelection::all()});
    return output;
}

void activateInto(Tensor input, Tensor output, ActivationFunction function, ScalarType computeType,
                  OutputSelection selection) {
    detail::runActivation({std::move(input), std::move(output), std::move(function), computeType,
                           std::move(selection)});
}

}  // namespace roc::host_numerics
