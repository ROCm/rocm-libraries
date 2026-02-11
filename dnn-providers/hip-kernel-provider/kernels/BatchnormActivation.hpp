// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipKernelMath.hpp"

namespace hip_kernel_plugin
{
namespace batchnorm
{

/// Note: Values match hip_kernel_utils::ActivationMode in HipKernelUtils.hpp
enum class ActivationMode : int
{
    PASTHRU = 0,
    LOGISTIC = 1,
    TANH = 2,
    RELU = 3,
    SOFTRELU = 4,
    ABS = 5,
    POWER = 6,
    CLIPPED_RELU = 7,
    LEAKY_RELU = 8,
    ELU = 9,
    CLAMP = 10
};

/// @brief Apply activation function to a single scalar value
/// @tparam T Floating point type (float, _Float16, etc.)
/// @tparam Mode Activation mode to apply
/// @param value Input value to apply activation to
/// @param alpha First activation parameter
/// @param beta Second activation parameter
/// @return Activated value
template <typename T, ActivationMode Mode>
__forceinline__ __device__ T applyActivation(T const& value, T const& alpha, T const& beta)
{
    static_assert(Mode == ActivationMode::PASTHRU || Mode == ActivationMode::RELU
                      || Mode == ActivationMode::CLIPPED_RELU || Mode == ActivationMode::CLAMP,
                  "Unsupported activation mode for batchnorm fusion");

    if constexpr(Mode == ActivationMode::PASTHRU)
    {
        return value;
    }
    else if constexpr(Mode == ActivationMode::RELU)
    {
        return max(value, T(0));
    }
    else if constexpr(Mode == ActivationMode::CLIPPED_RELU)
    {
        return min(alpha, max(value, T(0)));
    }
    else if constexpr(Mode == ActivationMode::CLAMP)
    {
        return max(alpha, min(beta, value));
    }
}

/// @brief Apply activation gradient for backward pass
/// @tparam T Floating point type (float, _Float16, etc.)
/// @tparam Mode Activation mode
/// @param dy Gradient from next layer
/// @param xnorm Normalized input value
/// @param scale Scale parameter from batchnorm
/// @param bias Bias parameter from batchnorm
/// @param alpha First activation parameter
/// @param beta Second activation parameter
/// @return Gradient with respect to input
template <typename T, ActivationMode Mode>
__forceinline__ __device__ T applyActivationGradient(
    T const& dy, T const& xnorm, T const& scale, T const& bias, T const& alpha, T const& beta)
{
    static_assert(Mode == ActivationMode::PASTHRU || Mode == ActivationMode::RELU
                      || Mode == ActivationMode::CLIPPED_RELU || Mode == ActivationMode::CLAMP,
                  "Unsupported activation mode for batchnorm fusion");

    if constexpr(Mode == ActivationMode::PASTHRU)
    {
        return dy;
    }
    else if constexpr(Mode == ActivationMode::RELU)
    {
        T activated = scale * xnorm + bias;
        return (activated > T(0)) ? dy : T(0);
    }
    else if constexpr(Mode == ActivationMode::CLIPPED_RELU)
    {
        T activated = scale * xnorm + bias;
        return (activated > T(0) && activated <= alpha) ? dy : T(0);
    }
    else if constexpr(Mode == ActivationMode::CLAMP)
    {
        T activated = scale * xnorm + bias;
        return (activated > alpha && activated <= beta) ? dy : T(0);
    }
}

} // namespace batchnorm
} // namespace hip_kernel_plugin
