// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <roc/hostnumerics/tensor_operations.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "reference_common.hpp"

namespace roc::hostnumerics::detail {
enum class BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Count,
};

struct BinaryInvocation {
    const Tensor& left;
    const Tensor& right;
    Tensor output;
    ScalarType computeType;
    IndexSelection selection = IndexSelection::allLastDimensionFastest();
};

struct ActivationInvocation {
    const Tensor& input;
    Tensor output;
    ActivationFunction function;
    ScalarType computeType;
    IndexSelection selection = IndexSelection::allLastDimensionFastest();
};

template <typename Function>
void forEachSelectedCoordinate(const Shape& shape, const IndexSelection& selection,
                               Function&& function) {
    const IndexIteration iteration(shape, selection.indexOrder(shape.rank()));
    std::vector<size_t> coordinates(shape.rank(), 0);
    if (selection.selectsAll()) {
        for (size_t index = 0; index < iteration.count(); ++index) {
            function(std::span<const size_t>(coordinates));
            iteration.advance(coordinates);
        }
        return;
    }

    for (const size_t logicalIndex : selection.indices(iteration.count())) {
        iteration.coordinates(logicalIndex, coordinates);
        function(std::span<const size_t>(coordinates));
    }
}

inline Shape binaryOutputShape(const Tensor& left, const Tensor& right) {
    return broadcastShapes(left.shape(), right.shape());
}

inline void validateBinaryInvocation(const BinaryInvocation& invocation) {
    const Shape outputShape = binaryOutputShape(invocation.left, invocation.right);
    if (invocation.output.shape() != outputShape)
        throw std::runtime_error("Elementwise input/output shapes differ.");
    if (!isComplexScalarType(invocation.computeType) &&
        (isComplexScalarType(invocation.left.type()) ||
         isComplexScalarType(invocation.right.type())))
        throw std::runtime_error("Real elementwise computation cannot consume a complex operand.");
    requireProvablyDistinctDestinationElementOffsets(invocation.output, "Elementwise", "output");
    rejectOverlappingTensorStorageUnlessIdenticallyMapped(
        invocation.output, invocation.left,
        "Elementwise output overlaps the left operand with a different storage mapping.");
    rejectOverlappingTensorStorageUnlessIdenticallyMapped(
        invocation.output, invocation.right,
        "Elementwise output overlaps the right operand with a different storage mapping.");
    (void)invocation.selection.selectedCount(outputShape.elementCount());
}

template <typename Accumulator, bool QuantizeResult>
void binaryTyped(const BinaryInvocation& invocation, BinaryOperation operation) {
    const Tensor left = invocation.left.broadcastTo(invocation.output.shape());
    const Tensor right = invocation.right.broadcastTo(invocation.output.shape());
    const RuntimeTensorReader<Accumulator> leftReader(left);
    const RuntimeTensorReader<Accumulator> rightReader(right);
    const RuntimeTensorWriter<Accumulator> output(invocation.output);
    const RuntimeQuantizer<Accumulator> quantize(
        QuantizeResult ? std::optional<ScalarType>(invocation.computeType) : std::nullopt);

    forEachSelectedCoordinate(
        invocation.output.shape(), invocation.selection, [&](std::span<const size_t> coordinates) {
            const Accumulator leftValue = leftReader(coordinates);
            const Accumulator rightValue = rightReader(coordinates);
            const Accumulator value = [&] {
                switch (operation) {
                    case BinaryOperation::Add:
                        return wrappingAdd(leftValue, rightValue);
                    case BinaryOperation::Subtract:
                        return wrappingSubtract(leftValue, rightValue);
                    case BinaryOperation::Multiply:
                        return wrappingMultiply(leftValue, rightValue);
                    case BinaryOperation::Count:
                        break;
                }
                throw std::runtime_error("Unsupported elementwise binary operation.");
            }();
            output.store(coordinates, quantize(value));
        });
}

inline void runBinary(const BinaryInvocation& invocation, BinaryOperation operation) {
    validateBinaryInvocation(invocation);
    switch (invocation.computeType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            return binaryTyped<float, true>(invocation, operation);
        case ScalarType::Float32:
            return binaryTyped<float, false>(invocation, operation);
        case ScalarType::Float64:
            return binaryTyped<double, false>(invocation, operation);
        case ScalarType::Int32:
            return binaryTyped<int32_t, false>(invocation, operation);
        case ScalarType::ComplexFloat32:
            return binaryTyped<std::complex<float>, false>(invocation, operation);
        case ScalarType::ComplexFloat64:
            return binaryTyped<std::complex<double>, false>(invocation, operation);
        default:
            throw std::runtime_error("Unsupported elementwise compute type.");
    }
}

inline ScalarType implicitElementwiseComputeType(ScalarType type) {
    switch (type) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::Int32:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            return type;
        default:
            throw std::runtime_error(
                "Elementwise type requires an explicit supported compute type.");
    }
}

inline void validateActivationInvocation(const ActivationInvocation& invocation) {
    if (invocation.output.shape() != invocation.input.shape())
        throw std::runtime_error("Activation input/output shapes differ.");
    requireProvablyDistinctDestinationElementOffsets(invocation.output, "Activation", "output");
    rejectOverlappingTensorStorageUnlessIdenticallyMapped(
        invocation.output, invocation.input,
        "Activation output overlaps the input with a different storage mapping.");
    if (isComplexScalarType(invocation.computeType) ||
        isComplexScalarType(invocation.input.type()) ||
        isComplexScalarType(invocation.output.type()))
        throw std::runtime_error("Activation requires real tensor types.");
    if (invocation.computeType == ScalarType::Int32) {
        const bool integerActivation =
            std::holds_alternative<AbsoluteActivation>(invocation.function) ||
            std::holds_alternative<ClippedReluActivation>(invocation.function) ||
            std::holds_alternative<ReluActivation>(invocation.function) ||
            std::holds_alternative<LeakyReluActivation>(invocation.function) ||
            std::holds_alternative<ReluDerivativeActivation>(invocation.function) ||
            std::holds_alternative<ClampActivation>(invocation.function);
        if (!integerActivation)
            throw std::runtime_error("Int32 activation does not support floating-point functions.");
    }
    (void)invocation.selection.selectedCount(invocation.output.shape().elementCount());
}

template <typename Accumulator, bool QuantizeResult>
void activationTyped(const ActivationInvocation& invocation) {
    const RuntimeTensorReader<Accumulator> input(invocation.input);
    const RuntimeTensorWriter<Accumulator> output(invocation.output);
    const RuntimeQuantizer<Accumulator> quantize(
        QuantizeResult ? std::optional<ScalarType>(invocation.computeType) : std::nullopt);
    const RuntimeActivation<Accumulator> function =
        runtimeActivation<Accumulator>(invocation.function);

    forEachSelectedCoordinate(
        invocation.output.shape(), invocation.selection, [&](std::span<const size_t> coordinates) {
            output.store(coordinates,
                         quantize(applyActivation(function.kind, input(coordinates),
                                                  function.parameter0, function.parameter1)));
        });
}

inline void runActivation(const ActivationInvocation& invocation) {
    validateActivationInvocation(invocation);
    switch (invocation.computeType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            return activationTyped<float, true>(invocation);
        case ScalarType::Float32:
            return activationTyped<float, false>(invocation);
        case ScalarType::Float64:
            return activationTyped<double, false>(invocation);
        case ScalarType::Int32:
            return activationTyped<int32_t, false>(invocation);
        default:
            throw std::runtime_error("Unsupported activation compute type.");
    }
}
}  // namespace roc::hostnumerics::detail
