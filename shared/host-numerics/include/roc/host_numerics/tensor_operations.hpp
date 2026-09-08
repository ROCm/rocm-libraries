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
Tensor add(const Tensor& left, const Tensor& right);
Tensor add(const Tensor& left, const Tensor& right, ScalarType outputType, ScalarType computeType,
           std::optional<Layout> outputLayout = std::nullopt);
void addInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
             OutputSelection selection = OutputSelection::all());

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor add(const Tensor& tensor, Number value) {
    return add(tensor, Tensor::scalar(tensor.type(), std::move(value)));
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor add(Number value, const Tensor& tensor) {
    return add(Tensor::scalar(tensor.type(), std::move(value)), tensor);
}

Tensor multiply(const Tensor& left, const Tensor& right);
Tensor multiply(const Tensor& left, const Tensor& right, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout = std::nullopt);
void multiplyInto(const Tensor& left, const Tensor& right, Tensor output, ScalarType computeType,
                  OutputSelection selection = OutputSelection::all());

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor multiply(const Tensor& tensor, Number value) {
    return multiply(tensor, Tensor::scalar(tensor.type(), std::move(value)));
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor multiply(Number value, const Tensor& tensor) {
    return multiply(Tensor::scalar(tensor.type(), std::move(value)), tensor);
}

Tensor activate(const Tensor& input, ActivationFunction function);
Tensor activate(const Tensor& input, ActivationFunction function, ScalarType outputType,
                ScalarType computeType, std::optional<Layout> outputLayout = std::nullopt);
void activateInto(const Tensor& input, Tensor output, ActivationFunction function,
                  ScalarType computeType, OutputSelection selection = OutputSelection::all());

inline Tensor absolute(const Tensor& input) {
    return activate(input, AbsoluteActivation{});
}

inline Tensor relu(const Tensor& input) {
    return activate(input, ReluActivation{});
}

inline Tensor gelu(const Tensor& input) {
    return activate(input, GeluActivation{});
}

inline Tensor sigmoid(const Tensor& input) {
    return activate(input, SigmoidActivation{});
}

inline Tensor tanh(const Tensor& input, double inputScale = 1.0, double outputScale = 1.0) {
    return activate(input, TanhActivation{inputScale, outputScale});
}

inline Tensor silu(const Tensor& input) {
    return activate(input, SiluActivation{});
}

inline Tensor swish(const Tensor& input, double beta = 1.0) {
    return activate(input, SwishActivation{beta});
}

inline Tensor clip(const Tensor& input, double minimum, double maximum) {
    return activate(input, ClampActivation{minimum, maximum});
}

inline Tensor operator+(const Tensor& left, const Tensor& right) {
    return add(left, right);
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor operator+(const Tensor& tensor, Number value) {
    return add(tensor, std::move(value));
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor operator+(Number value, const Tensor& tensor) {
    return add(std::move(value), tensor);
}

inline Tensor operator*(const Tensor& left, const Tensor& right) {
    return multiply(left, right);
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor operator*(const Tensor& tensor, Number value) {
    return multiply(tensor, std::move(value));
}

template <typename Number>
    requires requires { nativeScalarType<std::remove_cvref_t<Number>>; }
Tensor operator*(Number value, const Tensor& tensor) {
    return multiply(std::move(value), tensor);
}
}  // namespace roc::host_numerics
