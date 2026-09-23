// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "reference_arithmetic.hpp"
#include "runtime_tensor_io.hpp"

namespace roc::host_numerics::detail {
inline float quantizeXFloat32(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits &= 0xffffe000U;
    return std::bit_cast<float>(bits);
}

template <typename Accumulator>
using RuntimeMathFunction = Accumulator (*)(Accumulator);

template <typename Accumulator>
Accumulator identityMath(Accumulator value) {
    return value;
}

inline float xfloat32Math(float value) {
    return quantizeXFloat32(value);
}

template <typename Accumulator>
RuntimeMathFunction<Accumulator> runtimeMathFunction(MathMode mode) {
    if (mode == MathMode::Default) return &identityMath<Accumulator>;
    if constexpr (std::is_same_v<Accumulator, float>) {
        if (mode == MathMode::XFloat32) return &xfloat32Math;
    }
    throw std::invalid_argument("XFloat32 math mode requires a Float32 accumulator.");
}

template <typename Accumulator>
Accumulator checkedRuntimeScalar(auto value, const char* name) {
    using Source = decltype(value);
    if constexpr (IsComplex<Accumulator>::value) {
        using Real = typename Accumulator::value_type;
        if constexpr (IsComplex<Source>::value)
            return Accumulator(static_cast<Real>(value.real()), static_cast<Real>(value.imag()));
        else
            return Accumulator(static_cast<Real>(value), Real(0));
    } else {
        if constexpr (IsComplex<Source>::value) {
            if (value.imag() != 0)
                throw std::invalid_argument(std::string("Real reference accumulator has complex ") +
                                            name + ".");
            return checkedRuntimeScalar<Accumulator>(value.real(), name);
        } else {
            if constexpr (std::is_integral_v<Accumulator>) {
                if constexpr (std::is_floating_point_v<Source>) {
                    const long double converted = static_cast<long double>(value);
                    if (!std::isfinite(converted) || std::trunc(converted) != converted ||
                        converted <
                            static_cast<long double>(std::numeric_limits<Accumulator>::lowest()) ||
                        converted >
                            static_cast<long double>(std::numeric_limits<Accumulator>::max()))
                        throw std::invalid_argument(
                            std::string("Integer reference accumulator has invalid ") + name + ".");
                } else if constexpr (std::is_same_v<Source, bool>) {
                    return static_cast<Accumulator>(value);
                } else if constexpr (std::is_integral_v<Source>) {
                    if (!std::in_range<Accumulator>(value))
                        throw std::invalid_argument(
                            std::string("Integer reference accumulator has invalid ") + name + ".");
                }
            }
            return static_cast<Accumulator>(value);
        }
    }
}

template <typename Accumulator>
Accumulator runtimeScalar(const Tensor& value, const char* name) {
    if (value.shape().rank() != 0)
        throw std::invalid_argument(std::string("Reference ") + name + " must be rank zero.");
    switch (scalarTypeInfo(value.type()).category) {
        case ScalarCategory::Boolean:
            return checkedRuntimeScalar<Accumulator>(value.item<bool>(), name);
        case ScalarCategory::SignedInteger:
            return checkedRuntimeScalar<Accumulator>(value.item<int64_t>(), name);
        case ScalarCategory::UnsignedInteger:
            return checkedRuntimeScalar<Accumulator>(value.item<uint64_t>(), name);
        case ScalarCategory::FloatingPoint:
        case ScalarCategory::Scale:
            return checkedRuntimeScalar<Accumulator>(value.item<double>(), name);
        case ScalarCategory::Complex:
            return checkedRuntimeScalar<Accumulator>(value.item<std::complex<double>>(), name);
    }
    throw std::invalid_argument("Invalid runtime scalar type.");
}

template <typename Accumulator>
struct RuntimeActivation {
    Activation kind = Activation::None;
    Accumulator parameter0 = Accumulator(0);
    Accumulator parameter1 = Accumulator(0);
};

template <typename Accumulator>
RuntimeActivation<Accumulator> runtimeActivation(const ActivationFunction& activation) {
    return std::visit(
        []<typename Function>(const Function& function) -> RuntimeActivation<Accumulator> {
            if constexpr (std::is_same_v<Function, IdentityActivation>)
                return {};
            else if constexpr (std::is_same_v<Function, AbsoluteActivation>)
                return {.kind = Activation::Absolute};
            else if constexpr (std::is_same_v<Function, ClippedReluActivation>)
                return {.kind = Activation::ClippedRelu,
                        .parameter0 = checkedRuntimeScalar<Accumulator>(function.lower, "lower"),
                        .parameter1 = checkedRuntimeScalar<Accumulator>(function.upper, "upper")};
            else if constexpr (std::is_same_v<Function, ReluActivation>)
                return {.kind = Activation::Relu};
            else if constexpr (std::is_same_v<Function, GeluActivation>)
                return {.kind = Activation::Gelu};
            else if constexpr (std::is_same_v<Function, GeluDerivativeActivation>)
                return {.kind = Activation::GeluDerivative};
            else if constexpr (std::is_same_v<Function, GeluScalingActivation>)
                return {.kind = Activation::GeluScaling,
                        .parameter0 = checkedRuntimeScalar<Accumulator>(function.scale, "scale")};
            else if constexpr (std::is_same_v<Function, LeakyReluActivation>)
                return {.kind = Activation::LeakyRelu,
                        .parameter0 = checkedRuntimeScalar<Accumulator>(function.negativeSlope,
                                                                        "negative slope")};
            else if constexpr (std::is_same_v<Function, ReluDerivativeActivation>)
                return {.kind = Activation::ReluDerivative};
            else if constexpr (std::is_same_v<Function, SigmoidActivation>)
                return {.kind = Activation::Sigmoid};
            else if constexpr (std::is_same_v<Function, TanhActivation>)
                return {.kind = Activation::Tanh,
                        .parameter0 =
                            checkedRuntimeScalar<Accumulator>(function.inputScale, "input scale"),
                        .parameter1 = checkedRuntimeScalar<Accumulator>(function.outputScale,
                                                                        "output scale")};
            else if constexpr (std::is_same_v<Function, SiluActivation>)
                return {.kind = Activation::Silu};
            else if constexpr (std::is_same_v<Function, SwishActivation>)
                return {.kind = Activation::Swish,
                        .parameter0 = checkedRuntimeScalar<Accumulator>(function.beta, "beta")};
            else
                return {
                    .kind = Activation::Clamp,
                    .parameter0 = checkedRuntimeScalar<Accumulator>(function.minimum, "minimum"),
                    .parameter1 = checkedRuntimeScalar<Accumulator>(function.maximum, "maximum")};
        },
        activation);
}
}  // namespace roc::host_numerics::detail
