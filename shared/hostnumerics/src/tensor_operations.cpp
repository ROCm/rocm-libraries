// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/tensor_operations.hpp"

#include <utility>

namespace roc::hostnumerics {
namespace {
Tensor binary(detail::BinaryOperation operation, const Tensor& left, const Tensor& right,
              ScalarType outputType, ScalarType computeType, std::optional<Layout> outputLayout) {
    const Shape outputShape = detail::binaryOutputShape(left, right);
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::runtime_error("Elementwise output layout shape mismatch.");
    Tensor output(outputType, layout);
    detail::BinaryInvocation invocation{left, right, output, computeType};
    detail::bindIndices(invocation, outputShape, IndexRange::all(outputShape.elementCount()),
                        outputShape.indexOrderLastDimensionFastest());
    detail::runBinary(invocation, operation);
    return output;
}

template <typename Indices>
void binaryInto(detail::BinaryOperation operation, const Tensor& left, const Tensor& right,
                Tensor output, ScalarType computeType, const Indices& indices,
                std::span<const size_t> order) {
    detail::BinaryInvocation invocation{left, right, std::move(output), computeType};
    detail::bindIndices(invocation, invocation.output.shape(), indices, order);
    detail::runBinary(invocation, operation);
}

Tensor binarySameType(detail::BinaryOperation operation, const Tensor& left, const Tensor& right) {
    if (left.type() != right.type())
        throw std::runtime_error(
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
             const IndexRange& indices, std::span<const size_t> order) {
    binaryInto(detail::BinaryOperation::Add, left, right, std::move(output), computeType, indices,
               order);
}
void addInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
             const std::set<size_t>& indices, std::span<const size_t> order) {
    binaryInto(detail::BinaryOperation::Add, left, right, std::move(output), computeType, indices,
               order);
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
                  const IndexRange& indices, std::span<const size_t> order) {
    binaryInto(detail::BinaryOperation::Subtract, left, right, std::move(output), computeType,
               indices, order);
}
void subtractInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
                  const std::set<size_t>& indices, std::span<const size_t> order) {
    binaryInto(detail::BinaryOperation::Subtract, left, right, std::move(output), computeType,
               indices, order);
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
                  const IndexRange& indices, std::span<const size_t> order) {
    binaryInto(detail::BinaryOperation::Multiply, left, right, std::move(output), computeType,
               indices, order);
}
void multiplyInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
                  const std::set<size_t>& indices, std::span<const size_t> order) {
    binaryInto(detail::BinaryOperation::Multiply, left, right, std::move(output), computeType,
               indices, order);
}

Tensor activate(const Tensor& input, ActivationFunction function) {
    const ScalarType type = input.type();
    return activate(input, std::move(function), type, detail::implicitElementwiseComputeType(type),
                    std::nullopt);
}

Tensor activate(const Tensor& input, ActivationFunction function, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout) {
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(input.shape()));
    if (layout.shape() != input.shape())
        throw std::runtime_error("Activation output layout shape mismatch.");
    Tensor output(outputType, layout);
    detail::ActivationInvocation invocation{input, output, std::move(function), computeType};
    detail::bindIndices(invocation, output.shape(), IndexRange::all(output.shape().elementCount()),
                        output.shape().indexOrderLastDimensionFastest());
    detail::runActivation(invocation);
    return output;
}

void activateInto(const Tensor& input, Tensor output, ActivationFunction function,
                  ScalarType computeType, const IndexRange& range, std::span<const size_t> order) {
    detail::ActivationInvocation invocation{input, std::move(output), std::move(function),
                                            computeType};
    detail::bindIndices(invocation, invocation.output.shape(), range, order);
    detail::runActivation(invocation);
}
void activateInto(const Tensor& input, Tensor output, ActivationFunction function,
                  ScalarType computeType, const std::set<size_t>& indices,
                  std::span<const size_t> order) {
    detail::ActivationInvocation invocation{input, std::move(output), std::move(function),
                                            computeType};
    detail::bindIndices(invocation, invocation.output.shape(), indices, order);
    detail::runActivation(invocation);
}

}  // namespace roc::hostnumerics
