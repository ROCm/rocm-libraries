// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/tensor_operations.hpp"

#include <utility>

namespace roc::host_numerics {
namespace {
Tensor binary(detail::BinaryOperation operation, const Tensor& left, const Tensor& right,
              ScalarType outputType, ScalarType computeType, std::optional<Layout> outputLayout) {
    const Shape outputShape = detail::binaryOutputShape(left, right);
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::invalid_argument("Elementwise output layout shape mismatch.");
    Tensor output(outputType, layout);
    detail::runBinary({left, right, output, computeType, OutputSelection::all()}, operation);
    return output;
}

void binaryInto(detail::BinaryOperation operation, const Tensor& left, const Tensor& right,
                Tensor output, ScalarType computeType, OutputSelection selection) {
    detail::runBinary({left, right, std::move(output), computeType, std::move(selection)},
                      operation);
}

Tensor binarySameType(detail::BinaryOperation operation, const Tensor& left, const Tensor& right) {
    if (left.type() != right.type())
        throw std::invalid_argument(
            "Implicit elementwise operations require operands with the same type.");
    const ScalarType type = left.type();
    return binary(operation, left, right, type, detail::implicitElementwiseComputeType(type),
                  std::nullopt);
}
}  // namespace

Tensor add(const Tensor& left, const Tensor& right) {
    return binarySameType(detail::BinaryOperation::Add, left, right);
}

Tensor add(const Tensor& left, const Tensor& right, ScalarType outputType, ScalarType computeType,
           std::optional<Layout> outputLayout) {
    return binary(detail::BinaryOperation::Add, left, right, outputType, computeType,
                  std::move(outputLayout));
}

void addInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
             OutputSelection selection) {
    binaryInto(detail::BinaryOperation::Add, left, right, std::move(output), computeType,
               std::move(selection));
}

Tensor subtract(const Tensor& left, const Tensor& right) {
    return binarySameType(detail::BinaryOperation::Subtract, left, right);
}

Tensor subtract(const Tensor& left, const Tensor& right, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout) {
    return binary(detail::BinaryOperation::Subtract, left, right, outputType, computeType,
                  std::move(outputLayout));
}

void subtractInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
                  OutputSelection selection) {
    binaryInto(detail::BinaryOperation::Subtract, left, right, std::move(output), computeType,
               std::move(selection));
}

Tensor multiply(const Tensor& left, const Tensor& right) {
    return binarySameType(detail::BinaryOperation::Multiply, left, right);
}

Tensor multiply(const Tensor& left, const Tensor& right, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout) {
    return binary(detail::BinaryOperation::Multiply, left, right, outputType, computeType,
                  std::move(outputLayout));
}

void multiplyInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
                  OutputSelection selection) {
    binaryInto(detail::BinaryOperation::Multiply, left, right, std::move(output), computeType,
               std::move(selection));
}

Tensor activate(const Tensor& input, ActivationFunction function) {
    const ScalarType type = input.type();
    return activate(input, std::move(function), type, detail::implicitElementwiseComputeType(type));
}

Tensor activate(const Tensor& input, ActivationFunction function, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout) {
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(input.shape()));
    if (layout.shape() != input.shape())
        throw std::invalid_argument("Activation output layout shape mismatch.");
    Tensor output(outputType, layout);
    detail::runActivation(
        {input, output, std::move(function), computeType, OutputSelection::all()});
    return output;
}

void activateInto(const Tensor& input, Tensor output, ActivationFunction function,
                  ScalarType computeType, OutputSelection selection) {
    detail::runActivation(
        {input, std::move(output), std::move(function), computeType, std::move(selection)});
}

}  // namespace roc::host_numerics
