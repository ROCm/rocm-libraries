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

namespace roc::host_numerics::detail {
template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
T wrappingAdd(T left, T right) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result = static_cast<Unsigned>(left) + static_cast<Unsigned>(right);
        return std::bit_cast<T>(result);
    } else {
        return left + right;
    }
}

template <typename T>
T wrappingMultiply(T left, T right) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result = static_cast<Unsigned>(left) * static_cast<Unsigned>(right);
        return std::bit_cast<T>(result);
    } else {
        return left * right;
    }
}

template <typename T>
T wrappingSubtract(T left, T right) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned result = static_cast<Unsigned>(left) - static_cast<Unsigned>(right);
        return std::bit_cast<T>(result);
    } else {
        return left - right;
    }
}

template <typename T>
T wrappingNegate(T value) {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        return std::bit_cast<T>(Unsigned{0} - static_cast<Unsigned>(value));
    } else {
        return -value;
    }
}

template <typename T>
T conjugateIfNeeded(const T& value, bool conjugate) {
    if constexpr (IsComplex<T>::value)
        return conjugate ? std::conj(value) : value;
    else
        return value;
}

template <typename Accumulator>
Accumulator applyActivation(Activation activation, Accumulator value, Accumulator parameter0,
                            Accumulator parameter1) {
    if constexpr (IsComplex<Accumulator>::value) {
        if (activation != Activation::None)
            throw std::invalid_argument(
                "Complex reference arithmetic does not support activation.");
        return value;
    } else {
        using Transcendental =
            std::conditional_t<std::is_same_v<Accumulator, double>, double, float>;
        switch (activation) {
            case Activation::None:
                return value;
            case Activation::Absolute:
                return value >= Accumulator(0) ? value : wrappingNegate(value);
            case Activation::ClippedRelu:
                return value > parameter0 ? std::min(value, parameter1)
                                          : std::min(Accumulator(0), parameter1);
            case Activation::Relu:
                return std::max(Accumulator(0), value);
            case Activation::Gelu: {
                constexpr Transcendental coefficient0 =
                    static_cast<Transcendental>(0.7978845608028654);
                constexpr Transcendental coefficient1 = static_cast<Transcendental>(0.044715);
                const Transcendental x = static_cast<Transcendental>(value);
                return static_cast<Accumulator>(
                    Transcendental(0.5) * x *
                    (Transcendental(1) +
                     std::tanh(coefficient0 * x * (Transcendental(1) + coefficient1 * x * x))));
            }
            case Activation::GeluDerivative: {
                constexpr Transcendental coefficient0 = static_cast<Transcendental>(0.0535161);
                constexpr Transcendental coefficient1 = static_cast<Transcendental>(0.398942);
                constexpr Transcendental coefficient2 = static_cast<Transcendental>(0.0356774);
                constexpr Transcendental coefficient3 = static_cast<Transcendental>(0.797885);
                const Transcendental x = static_cast<Transcendental>(value);
                const Transcendental cube = x * x * x;
                const Transcendental first = coefficient0 * cube + coefficient1 * x;
                const Transcendental second = coefficient2 * cube + coefficient3 * x;
                const Transcendental derivative =
                    Transcendental(0.5) * std::tanh(second) +
                    first *
                        (Transcendental(4) / std::pow(std::exp(-second) + std::exp(second), 2)) +
                    Transcendental(0.5);
                return static_cast<Accumulator>(derivative);
            }
            case Activation::GeluScaling:
                return applyActivation(Activation::Gelu, value, parameter0, parameter1) *
                       parameter0;
            case Activation::LeakyRelu:
                return value > Accumulator(0) ? value : wrappingMultiply(value, parameter0);
            case Activation::ReluDerivative:
                return value > Accumulator(0) ? Accumulator(1) : Accumulator(0);
            case Activation::Sigmoid: {
                const Transcendental x = static_cast<Transcendental>(value);
                return static_cast<Accumulator>(Transcendental(1) /
                                                (Transcendental(1) + std::exp(-x)));
            }
            case Activation::Tanh:
                return static_cast<Accumulator>(
                    std::tanh(static_cast<Transcendental>(value * parameter0)) *
                    static_cast<Transcendental>(parameter1));
            case Activation::Silu: {
                const Transcendental x = static_cast<Transcendental>(value);
                return static_cast<Accumulator>(x / (Transcendental(1) + std::exp(-x)));
            }
            case Activation::Swish: {
                const Transcendental x = static_cast<Transcendental>(value);
                const Transcendental beta = static_cast<Transcendental>(parameter0);
                return static_cast<Accumulator>(x / (Transcendental(1) + std::exp(-beta * x)));
            }
            case Activation::Clamp:
                return std::max(parameter0, std::min(value, parameter1));
        }
    }

    throw std::invalid_argument("Unsupported reference activation.");
}
}  // namespace roc::host_numerics::detail
