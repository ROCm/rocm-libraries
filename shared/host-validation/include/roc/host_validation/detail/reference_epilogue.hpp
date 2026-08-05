// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <complex>
#include <optional>
#include <roc/host_validation/detail/reference_common.hpp>
#include <stdexcept>
#include <string>

namespace roc::host_validation {
enum class ActivationApplication {
    Forward,
    Gradient,
};

struct EpilogueProblem {
    EpilogueProblem(TensorView inputTensor, MutableTensorView outputTensor, ScalarType compute)
        : input(std::move(inputTensor)), output(std::move(outputTensor)), computeType(compute) {}

    TensorView input;
    MutableTensorView output;
    ScalarType computeType;
    std::optional<MutableTensorView> rawOutput;
    std::optional<MutableTensorView> auxiliaryOutput;
    std::optional<TensorView> auxiliaryInput;
    std::optional<MutableTensorView> amax;
    bool accumulateAmax = false;
    std::optional<VectorBinding> bias;
    std::complex<double> outputScale = {1.0, 0.0};
    std::complex<double> auxiliaryScale = {1.0, 0.0};
    Activation activation = Activation::None;
    ActivationApplication activationApplication = ActivationApplication::Forward;
    double activationParameter0 = 0.0;
    double activationParameter1 = 0.0;
};

struct EpilogueRunInfo {
    size_t outputElementsComputed = 0;
};

namespace detail {
template <typename Accumulator>
Accumulator activationGradientFactor(Activation activation, Accumulator value,
                                     Accumulator parameter0, Accumulator parameter1) {
    switch (activation) {
        case Activation::None:
            return Accumulator(1);
        case Activation::Relu:
            return value > Accumulator(0) ? Accumulator(1) : Accumulator(0);
        case Activation::Gelu: {
            constexpr float coefficient0 = 0.0535161f;
            constexpr float coefficient1 = 0.398942f;
            constexpr float coefficient2 = 0.0356774f;
            constexpr float coefficient3 = 0.797885f;
            const float x = static_cast<float>(value);
            const float cube = x * x * x;
            const float first = coefficient0 * cube + coefficient1 * x;
            const float second = coefficient2 * cube + coefficient3 * x;
            const float derivative =
                0.5f * std::tanh(second) +
                first * (4.0f / std::pow(std::exp(-second) + std::exp(second), 2)) + 0.5f;
            return static_cast<Accumulator>(derivative);
        }
        case Activation::Silu:
        case Activation::Clamp:
            return applyActivation(activation, value, parameter0, parameter1);
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

    auto validateMatrix = [&](const auto& view, const char* name) {
        requireRank(view.shape(), 2, "Reference epilogue", name);
        if (view.shape() != problem.output.shape())
            throw std::invalid_argument(std::string("Reference epilogue ") + name +
                                        " shape mismatch.");
    };
    if (problem.rawOutput) validateMatrix(*problem.rawOutput, "raw output");
    if (problem.auxiliaryOutput) validateMatrix(*problem.auxiliaryOutput, "auxiliary output");
    if (problem.auxiliaryInput) validateMatrix(*problem.auxiliaryInput, "auxiliary input");
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
}

template <typename Accumulator>
EpilogueRunInfo referenceEpilogueTyped(const EpilogueProblem& problem) {
    const RuntimeMatrixReader<Accumulator> input(problem.input);
    const RuntimeMatrixWriter<Accumulator> output(problem.output);
    std::optional<RuntimeMatrixWriter<Accumulator>> rawOutput;
    std::optional<RuntimeMatrixWriter<Accumulator>> auxiliaryOutput;
    std::optional<RuntimeMatrixReader<Accumulator>> auxiliaryInput;
    std::optional<RuntimeVectorReader<Accumulator>> bias;
    if (problem.rawOutput) rawOutput.emplace(*problem.rawOutput);
    if (problem.auxiliaryOutput) auxiliaryOutput.emplace(*problem.auxiliaryOutput);
    if (problem.auxiliaryInput) auxiliaryInput.emplace(*problem.auxiliaryInput);
    if (problem.bias) bias.emplace(problem.bias->values);

    const Accumulator outputScale = runtimeScalar<Accumulator>(problem.outputScale, "output scale");
    const Accumulator auxiliaryScale =
        runtimeScalar<Accumulator>(problem.auxiliaryScale, "auxiliary scale");
    const Accumulator parameter0 = static_cast<Accumulator>(problem.activationParameter0);
    const Accumulator parameter1 = static_cast<Accumulator>(problem.activationParameter1);
    Accumulator maximum = Accumulator(0);
    if (problem.amax && problem.accumulateAmax) maximum = problem.amax->loadAs<Accumulator>({0});

    const size_t rows = problem.output.shape()[0];
    const size_t columns = problem.output.shape()[1];
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            Accumulator value = input(row, column);
            if (bias) {
                const MatrixAxis axis = problem.bias->axis;
                value += (*bias)[axis == MatrixAxis::Row ? row : column];
            }

            if (auxiliaryOutput) auxiliaryOutput->store(row, column, value * auxiliaryScale);

            if (problem.activationApplication == ActivationApplication::Gradient) {
                const Accumulator factor = activationGradientFactor(
                    problem.activation, (*auxiliaryInput)(row, column), parameter0, parameter1);
                value *= factor;
            } else {
                value = applyActivation(problem.activation, value, parameter0, parameter1);
            }

            if (problem.amax)
                maximum = std::max(maximum, static_cast<Accumulator>(std::abs(value)));

            value *= outputScale;
            output.store(row, column, value);
            if (rawOutput) rawOutput->store(row, column, value);
        }
    }

    if (problem.amax) {
        const std::vector<size_t> indices(problem.amax->shape().rank(), 0);
        problem.amax->storeFrom(std::span<const size_t>(indices), maximum);
    }

    return {.outputElementsComputed = problem.output.shape().elementCount()};
}
}  // namespace detail

inline EpilogueRunInfo referenceEpilogue(const EpilogueProblem& problem) {
    detail::validateEpilogue(problem);
    switch (problem.computeType) {
        case ScalarType::Float32:
            return detail::referenceEpilogueTyped<float>(problem);
        case ScalarType::Float64:
            return detail::referenceEpilogueTyped<double>(problem);
        case ScalarType::Int32:
            return detail::referenceEpilogueTyped<int32_t>(problem);
        default:
            throw std::invalid_argument("Unsupported reference epilogue compute type.");
    }
}
}  // namespace roc::host_validation
