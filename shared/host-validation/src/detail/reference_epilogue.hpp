// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <optional>
#include <roc/host_validation/epilogue.hpp>
#include <stdexcept>
#include <string>

#include "reference_common.hpp"

namespace roc::host_validation {
namespace detail {
template <typename Accumulator>
Accumulator activationGradientFactor(Activation activation, Accumulator value,
                                     Accumulator parameter0, Accumulator parameter1) {
    switch (activation) {
        case Activation::None:
            return Accumulator(1);
        case Activation::Absolute:
            if (value > Accumulator(0)) return Accumulator(1);
            if (value < Accumulator(0)) return Accumulator(-1);
            return Accumulator(0);
        case Activation::ClippedRelu:
            return value > parameter0 && value < parameter1 ? Accumulator(1) : Accumulator(0);
        case Activation::Relu:
            return value > Accumulator(0) ? Accumulator(1) : Accumulator(0);
        case Activation::Gelu:
            return applyActivation(Activation::GeluDerivative, value, parameter0, parameter1);
        case Activation::GeluScaling:
            return applyActivation(Activation::GeluDerivative, value, parameter0, parameter1) *
                   parameter0;
        case Activation::LeakyRelu:
            return value > Accumulator(0) ? Accumulator(1) : parameter0;
        case Activation::Sigmoid: {
            const Accumulator sigmoid =
                applyActivation(Activation::Sigmoid, value, parameter0, parameter1);
            return sigmoid * (Accumulator(1) - sigmoid);
        }
        case Activation::Tanh: {
            const Accumulator hyperbolicTangent =
                static_cast<Accumulator>(std::tanh(static_cast<float>(value * parameter0)));
            return parameter0 * parameter1 *
                   (Accumulator(1) - hyperbolicTangent * hyperbolicTangent);
        }
        case Activation::Silu: {
            const Accumulator sigmoid =
                applyActivation(Activation::Sigmoid, value, parameter0, parameter1);
            return sigmoid + value * sigmoid * (Accumulator(1) - sigmoid);
        }
        case Activation::Swish: {
            const Accumulator sigmoid = static_cast<Accumulator>(
                1.0f / (1.0f + std::exp(-static_cast<float>(parameter0 * value))));
            return sigmoid + parameter0 * value * sigmoid * (Accumulator(1) - sigmoid);
        }
        case Activation::Clamp:
            return value > parameter0 && value < parameter1 ? Accumulator(1) : Accumulator(0);
        case Activation::GeluDerivative:
        case Activation::ReluDerivative:
            throw std::invalid_argument(
                "Gradient application does not accept an explicit derivative activation.");
    }
    throw std::invalid_argument("Unsupported epilogue activation.");
}

inline void validateEpilogue(const EpilogueProblem& problem) {
    requireRank(problem.input.shape(), 2, "Reference epilogue", "input");
    requireRank(problem.output.shape(), 2, "Reference epilogue", "output");
    if (problem.input.shape() != problem.output.shape())
        throw std::invalid_argument("Reference epilogue input/output shape mismatch.");
    if (problem.computeType != ScalarType::Float32 && problem.computeType != ScalarType::Float64 &&
        problem.computeType != ScalarType::Int32)
        throw std::invalid_argument("Reference epilogue supports F32, F64, and I32 compute types.");
    if (scalarTypeInfo(problem.input.type()).category == ScalarCategory::Complex ||
        scalarTypeInfo(problem.output.type()).category == ScalarCategory::Complex)
        throw std::invalid_argument("Reference epilogue does not support complex tensors.");
    if (problem.computeType == ScalarType::Int32) {
        switch (problem.activation) {
            case Activation::None:
            case Activation::Absolute:
            case Activation::ClippedRelu:
            case Activation::Relu:
            case Activation::LeakyRelu:
            case Activation::ReluDerivative:
            case Activation::Clamp:
                break;
            default:
                throw std::invalid_argument(
                    "Int32 reference epilogue does not support floating-point activation.");
        }
        if (problem.amax)
            throw std::invalid_argument("Int32 reference epilogue does not support AMax.");
    }

    auto validateMatrix = [&](const auto& view, const char* name) {
        requireRank(view.shape(), 2, "Reference epilogue", name);
        if (view.shape() != problem.output.shape())
            throw std::invalid_argument(std::string("Reference epilogue ") + name +
                                        " shape mismatch.");
    };
    if (problem.rawOutput) validateMatrix(*problem.rawOutput, "raw output");
    if (problem.auxiliaryOutput) validateMatrix(*problem.auxiliaryOutput, "auxiliary output");
    if (problem.auxiliaryInput) validateMatrix(*problem.auxiliaryInput, "auxiliary input");
    if (problem.gateResidual) validateMatrix(*problem.gateResidual, "gate residual");
    if (problem.activationApplication == ActivationApplication::Gradient && !problem.auxiliaryInput)
        throw std::invalid_argument("Gradient epilogue requires an auxiliary input tensor.");
    if (problem.bias) {
        const size_t expected =
            axisExtent(problem.bias->axis, problem.output.shape()[0], problem.output.shape()[1]);
        validateRuntimeVector(problem.bias->values, expected, "Reference epilogue", "bias");
        if (scalarTypeInfo(problem.bias->values.type()).category == ScalarCategory::Complex)
            throw std::invalid_argument("Reference epilogue bias must be real.");
    }
    if (problem.amax && problem.amax->shape().elementCount() != 1)
        throw std::invalid_argument("Reference epilogue AMax output must contain one element.");
    (void)problem.outputSelection.selectedCount(problem.output.shape().elementCount());
}

template <typename Accumulator>
EpilogueRunInfo referenceEpilogueTyped(const EpilogueProblem& problem) {
    const RuntimeMatrixReader<Accumulator> input(problem.input);
    const RuntimeMatrixOutputWriter<Accumulator> output(problem.output, problem.outputConversion);
    std::optional<RuntimeMatrixWriter<Accumulator>> rawOutput;
    std::optional<RuntimeMatrixWriter<Accumulator>> auxiliaryOutput;
    std::optional<RuntimeMatrixReader<Accumulator>> auxiliaryInput;
    std::optional<RuntimeMatrixReader<Accumulator>> gateResidual;
    std::optional<RuntimeVectorReader<Accumulator>> bias;
    if (problem.rawOutput) rawOutput.emplace(*problem.rawOutput);
    if (problem.auxiliaryOutput) auxiliaryOutput.emplace(*problem.auxiliaryOutput);
    if (problem.auxiliaryInput) auxiliaryInput.emplace(*problem.auxiliaryInput);
    if (problem.gateResidual) gateResidual.emplace(*problem.gateResidual);
    if (problem.bias) bias.emplace(problem.bias->values);

    const Accumulator outputScale = runtimeScalar<Accumulator>(problem.outputScale, "output scale");
    const Accumulator auxiliaryScale =
        runtimeScalar<Accumulator>(problem.auxiliaryScale, "auxiliary scale");
    const Accumulator parameter0 =
        runtimeScalar<Accumulator>({problem.activationParameter0, 0.0}, "activation parameter 0");
    const Accumulator parameter1 =
        runtimeScalar<Accumulator>({problem.activationParameter1, 0.0}, "activation parameter 1");
    Accumulator maximum = Accumulator(0);
    if (problem.amax && problem.accumulateAmax) maximum = problem.amax->loadAs<Accumulator>({0});

    const size_t rows = problem.output.shape()[0];
    const size_t columns = problem.output.shape()[1];
    auto computeOutput = [&](size_t row, size_t column) {
        Accumulator value = input(row, column);
        if (bias) {
            const MatrixAxis axis = problem.bias->axis;
            value = wrappingAdd(value, (*bias)[axis == MatrixAxis::Row ? row : column]);
        }

        if (auxiliaryOutput)
            auxiliaryOutput->store(row, column, wrappingMultiply(value, auxiliaryScale));

        if (problem.activationApplication == ActivationApplication::Gradient) {
            const Accumulator factor = activationGradientFactor(
                problem.activation, (*auxiliaryInput)(row, column), parameter0, parameter1);
            value = wrappingMultiply(value, factor);
        } else {
            value = applyActivation(problem.activation, value, parameter0, parameter1);
        }

        if (problem.amax) maximum = std::max(maximum, static_cast<Accumulator>(std::abs(value)));

        value = wrappingMultiply(value, outputScale);
        if (rawOutput) rawOutput->store(row, column, value);
        if (gateResidual) {
            const Accumulator gate = (*gateResidual)(row, column);
            value = wrappingAdd(wrappingMultiply(gate, value), gate);
        }
        output.store(row, column, value);
    };

    const size_t logicalElements = problem.output.shape().elementCount();
    size_t computedElements = 0;
    if (problem.outputSelection.selectsAll()) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns; ++column) {
                computeOutput(row, column);
                ++computedElements;
            }
        }
    } else {
        const auto selected = problem.outputSelection.indices(logicalElements);
        for (const size_t logicalIndex : selected)
            computeOutput(logicalIndex / columns, logicalIndex % columns);
        computedElements = selected.size();
    }

    if (problem.amax) {
        const std::vector<size_t> indices(problem.amax->shape().rank(), 0);
        problem.amax->storeFrom(std::span<const size_t>(indices), maximum);
    }

    return {
        .outputElementsWritten = computedElements,
        .rawOutputElementsWritten = problem.rawOutput ? computedElements : 0,
        .auxiliaryOutputElementsWritten = problem.auxiliaryOutput ? computedElements : 0,
        .amaxElementsWritten = problem.amax ? size_t{1} : size_t{0},
    };
}
}  // namespace detail
}  // namespace roc::host_validation
