// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <type_traits>
#include <utility>

namespace roc::host_numerics {
// NumPy-style elementwise addition and multiplication. Inputs use trailing-axis
// broadcasting, including rank-zero tensors. The short overloads require equal
// input types and preserve that type; explicit overloads make mixed/storage
// conversion semantics visible.
Tensor add(Tensor left, Tensor right);
Tensor add(Tensor left, Tensor right, ScalarType outputType, ScalarType computeType,
           std::optional<Layout> outputLayout = std::nullopt);
void addInto(Tensor left, Tensor right, Tensor output, ScalarType computeType,
             OutputSelection selection = OutputSelection::all());

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor add(Tensor tensor, Number value) {
    return add(tensor, Tensor::scalar(tensor.type(), std::move(value)));
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor add(Number value, Tensor tensor) {
    return add(Tensor::scalar(tensor.type(), std::move(value)), tensor);
}

Tensor multiply(Tensor left, Tensor right);
Tensor multiply(Tensor left, Tensor right, ScalarType outputType, ScalarType computeType,
                std::optional<Layout> outputLayout = std::nullopt);
void multiplyInto(Tensor left, Tensor right, Tensor output, ScalarType computeType,
                  OutputSelection selection = OutputSelection::all());

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor multiply(Tensor tensor, Number value) {
    return multiply(tensor, Tensor::scalar(tensor.type(), std::move(value)));
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor multiply(Number value, Tensor tensor) {
    return multiply(Tensor::scalar(tensor.type(), std::move(value)), tensor);
}

Tensor activate(Tensor input, ActivationFunction function);
Tensor activate(Tensor input, ActivationFunction function, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout = std::nullopt);
void activateInto(Tensor input, Tensor output, ActivationFunction function, ScalarType computeType,
                  OutputSelection selection = OutputSelection::all());

inline Tensor absolute(Tensor input) {
    return activate(std::move(input), AbsoluteActivation{});
}

inline Tensor relu(Tensor input) {
    return activate(std::move(input), ReluActivation{});
}

inline Tensor gelu(Tensor input) {
    return activate(std::move(input), GeluActivation{});
}

inline Tensor sigmoid(Tensor input) {
    return activate(std::move(input), SigmoidActivation{});
}

inline Tensor tanh(Tensor input, double inputScale = 1.0, double outputScale = 1.0) {
    return activate(std::move(input), TanhActivation{inputScale, outputScale});
}

inline Tensor silu(Tensor input) {
    return activate(std::move(input), SiluActivation{});
}

inline Tensor swish(Tensor input, double beta = 1.0) {
    return activate(std::move(input), SwishActivation{beta});
}

inline Tensor clip(Tensor input, double minimum, double maximum) {
    return activate(std::move(input), ClampActivation{minimum, maximum});
}
}  // namespace roc::host_numerics
