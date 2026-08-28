// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
#include <roc/host_numerics/epilogue.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "reference_common.hpp"

namespace roc::host_numerics {
namespace detail {
template <typename Accumulator>
Accumulator activationGradientFactor(Activation activation, Accumulator value,
                                     Accumulator parameter0, Accumulator parameter1) {
    using Transcendental = std::conditional_t<std::is_same_v<Accumulator, double>, double, float>;
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
            const Transcendental hyperbolicTangent = std::tanh(
                static_cast<Transcendental>(value) * static_cast<Transcendental>(parameter0));
            return static_cast<Accumulator>(
                static_cast<Transcendental>(parameter0) * static_cast<Transcendental>(parameter1) *
                (Transcendental(1) - hyperbolicTangent * hyperbolicTangent));
        }
        case Activation::Silu: {
            const Accumulator sigmoid =
                applyActivation(Activation::Sigmoid, value, parameter0, parameter1);
            return sigmoid + value * sigmoid * (Accumulator(1) - sigmoid);
        }
        case Activation::Swish: {
            const Transcendental beta = static_cast<Transcendental>(parameter0);
            const Transcendental x = static_cast<Transcendental>(value);
            const Transcendental sigmoid =
                Transcendental(1) / (Transcendental(1) + std::exp(-beta * x));
            return static_cast<Accumulator>(sigmoid +
                                            beta * x * sigmoid * (Transcendental(1) - sigmoid));
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

struct EpiloguePlan {
    size_t selectedElements = 0;
};

inline void validateEpilogueValueType(ScalarType type, const char* name) {
    if (!isConcreteScalarType(type))
        throw std::invalid_argument(std::string("Reference epilogue ") + name +
                                    " type is invalid.");
    const ScalarCategory category = scalarTypeInfo(type).category;
    if (category == ScalarCategory::Complex || category == ScalarCategory::Scale)
        throw std::invalid_argument(std::string("Reference epilogue ") + name +
                                    " must use a real arithmetic scalar type.");
}

inline void validateEpilogueActivation(const EpilogueProblem& problem) {
    switch (problem.activationApplication) {
        case ActivationApplication::Forward:
        case ActivationApplication::Gradient:
            break;
        default:
            throw std::invalid_argument("Reference epilogue activation application is invalid.");
    }
    switch (problem.activation) {
        case Activation::None:
        case Activation::Absolute:
        case Activation::ClippedRelu:
        case Activation::Relu:
        case Activation::Gelu:
        case Activation::GeluDerivative:
        case Activation::GeluScaling:
        case Activation::LeakyRelu:
        case Activation::ReluDerivative:
        case Activation::Sigmoid:
        case Activation::Tanh:
        case Activation::Silu:
        case Activation::Swish:
        case Activation::Clamp:
            break;
        default:
            throw std::invalid_argument("Reference epilogue activation is invalid.");
    }
    if (problem.activationApplication == ActivationApplication::Gradient &&
        (problem.activation == Activation::GeluDerivative ||
         problem.activation == Activation::ReluDerivative))
        throw std::invalid_argument(
            "Gradient application does not accept an explicit derivative activation.");
}

template <typename Accumulator>
inline void validateEpilogueScalars(const EpilogueProblem& problem) {
    (void)runtimeScalar<Accumulator>(problem.outputScale, "output scale");
    (void)runtimeScalar<Accumulator>(problem.auxiliaryScale, "auxiliary scale");
    (void)runtimeScalar<Accumulator>(problem.activationParameter0, "activation parameter 0");
    (void)runtimeScalar<Accumulator>(problem.activationParameter1, "activation parameter 1");
}

inline EpiloguePlan validateEpilogueProblem(const EpilogueProblem& problem) {
    requireRank(problem.input.shape(), 2, "Reference epilogue", "input");
    validateEpilogueValueType(problem.input.type(), "input");
    validateEpilogueValueType(problem.outputType, "output");
    if (problem.rawOutputType) validateEpilogueValueType(*problem.rawOutputType, "raw output");
    if (problem.auxiliaryOutputType)
        validateEpilogueValueType(*problem.auxiliaryOutputType, "auxiliary output");
    if (problem.amaxType) validateEpilogueValueType(*problem.amaxType, "AMax output");

    if (problem.computeType != ScalarType::Float32 && problem.computeType != ScalarType::Float64 &&
        problem.computeType != ScalarType::Int32)
        throw std::invalid_argument("Reference epilogue supports F32, F64, and I32 compute types.");
    validateEpilogueActivation(problem);
    switch (problem.outputConversion) {
        case OutputConversion::Default:
            break;
        case OutputConversion::SaturatingInt8:
            if (problem.outputType != ScalarType::Int8)
                throw std::invalid_argument(
                    "Saturating output conversion requires an Int8 output tensor.");
            break;
        default:
            throw std::invalid_argument("Reference epilogue output conversion is invalid.");
    }

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
        if (problem.amaxType)
            throw std::invalid_argument("Int32 reference epilogue does not support AMax.");
        validateEpilogueScalars<int32_t>(problem);
    } else if (problem.computeType == ScalarType::Float32) {
        validateEpilogueScalars<float>(problem);
    } else {
        validateEpilogueScalars<double>(problem);
    }

    auto validateMatrix = [&](const auto& view, const char* name) {
        requireRank(view.shape(), 2, "Reference epilogue", name);
        if (view.shape() != problem.input.shape())
            throw std::invalid_argument(std::string("Reference epilogue ") + name +
                                        " shape mismatch.");
        validateEpilogueValueType(view.type(), name);
    };
    if (problem.auxiliaryInput) validateMatrix(*problem.auxiliaryInput, "auxiliary input");
    if (problem.gateResidual) validateMatrix(*problem.gateResidual, "gate residual");
    if (problem.activationApplication == ActivationApplication::Gradient && !problem.auxiliaryInput)
        throw std::invalid_argument("Gradient epilogue requires an auxiliary input tensor.");
    if (problem.bias) {
        if (problem.bias->axis != MatrixAxis::Row && problem.bias->axis != MatrixAxis::Column)
            throw std::invalid_argument("Reference epilogue bias axis is invalid.");
        const size_t expected =
            axisExtent(problem.bias->axis, problem.input.shape()[0], problem.input.shape()[1]);
        validateRuntimeVector(problem.bias->values, expected, "Reference epilogue", "bias");
        validateEpilogueValueType(problem.bias->values.type(), "bias");
    }
    return {
        .selectedElements =
            problem.outputSelection.selectedCount(problem.input.shape().elementCount()),
    };
}

inline void validateEpilogueRequestStorage(const EpilogueRequest& request);

inline EpiloguePlan validateEpilogueRequest(const EpilogueRequest& request) {
    EpiloguePlan plan = validateEpilogueProblem(request);
    requireRank(request.output.shape(), 2, "Reference epilogue", "output");
    if (request.output.shape() != request.input.shape())
        throw std::invalid_argument("Reference epilogue input/output shape mismatch.");
    if (request.output.type() != request.outputType)
        throw std::invalid_argument("Reference epilogue output type differs from the problem.");
    if (request.rawOutput.has_value() != request.rawOutputType.has_value())
        throw std::invalid_argument(
            "Reference epilogue raw-output destination does not match the problem.");
    if (request.auxiliaryOutput.has_value() != request.auxiliaryOutputType.has_value())
        throw std::invalid_argument(
            "Reference epilogue auxiliary-output destination does not match the problem.");
    if (request.amax.has_value() != request.amaxType.has_value())
        throw std::invalid_argument(
            "Reference epilogue AMax destination does not match the problem.");

    auto validateOutputMatrix = [&](const std::optional<Tensor>& tensor,
                                    const std::optional<ScalarType>& type, const char* name) {
        if (!tensor) return;
        requireRank(tensor->shape(), 2, "Reference epilogue", name);
        if (tensor->shape() != request.input.shape())
            throw std::invalid_argument(std::string("Reference epilogue ") + name +
                                        " shape mismatch.");
        if (tensor->type() != *type)
            throw std::invalid_argument(std::string("Reference epilogue ") + name +
                                        " type differs from the problem.");
    };
    validateOutputMatrix(request.rawOutput, request.rawOutputType, "raw output");
    validateOutputMatrix(request.auxiliaryOutput, request.auxiliaryOutputType, "auxiliary output");
    if (request.amax) {
        if (request.amax->shape().elementCount() != 1)
            throw std::invalid_argument("Reference epilogue AMax output must contain one element.");
        if (request.amax->type() != *request.amaxType)
            throw std::invalid_argument("Reference epilogue AMax type differs from the problem.");
    } else if (request.accumulateAmax) {
        throw std::invalid_argument("Reference epilogue cannot accumulate an absent AMax output.");
    }
    validateEpilogueRequestStorage(request);
    return plan;
}

inline void validateEpilogueRequestStorage(const EpilogueRequest& request) {
    const std::array<const Tensor*, 4> outputs{
        &request.output,
        request.rawOutput ? &*request.rawOutput : nullptr,
        request.auxiliaryOutput ? &*request.auxiliaryOutput : nullptr,
        request.amax ? &*request.amax : nullptr,
    };
    const std::array<const Tensor*, 4> inputs{
        &request.input,
        request.auxiliaryInput ? &*request.auxiliaryInput : nullptr,
        request.gateResidual ? &*request.gateResidual : nullptr,
        request.bias ? &request.bias->values : nullptr,
    };
    for (const Tensor* output : outputs) {
        if (!output) continue;
        requireProvablyDistinctDestinationElementOffsets(*output, "Reference epilogue", "output");
        for (const Tensor* input : inputs) {
            if (!input) continue;
            const bool allowIdenticalInputOutput =
                output == &request.output && input == &request.input;
            if (allowIdenticalInputOutput)
                rejectOverlappingTensorStorageUnlessIdenticallyMapped(
                    *output, *input,
                    "Reference epilogue output overlaps an input with an unsafe storage mapping.");
            else
                rejectOverlappingTensorStorage(
                    *output, *input,
                    "Reference epilogue output overlaps an input with an unsafe storage mapping.");
        }
    }
    for (size_t left = 0; left < outputs.size(); ++left) {
        if (!outputs[left]) continue;
        for (size_t right = left + 1; right < outputs.size(); ++right) {
            if (outputs[right])
                rejectOverlappingTensorStorage(*outputs[left], *outputs[right],
                                               "Reference epilogue result tensors overlap.");
        }
    }
}

inline void initializeOwnedEpilogueTensor(Tensor tensor) {
    forEachIndex(tensor.shape(),
                 [&](std::span<const size_t> indices, size_t) { tensor.storeFrom(indices, 0.0); });
}

template <typename Accumulator>
EpilogueRunInfo referenceEpilogueTyped(const EpilogueRequest& problem) {
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
        runtimeScalar<Accumulator>(problem.activationParameter0, "activation parameter 0");
    const Accumulator parameter1 =
        runtimeScalar<Accumulator>(problem.activationParameter1, "activation parameter 1");
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
}  // namespace roc::host_numerics
