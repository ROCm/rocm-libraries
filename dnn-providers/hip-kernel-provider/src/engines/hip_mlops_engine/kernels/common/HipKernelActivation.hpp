// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipKernelMath.hpp"

namespace hip_kernel_provider
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
                  "Unsupported activation mode");

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
/// @param input_value Input value that was passed to the activation function
/// @param alpha First activation parameter
/// @param beta Second activation parameter
/// @return Gradient with respect to input
template <typename T, ActivationMode Mode>
__forceinline__ __device__ T
    applyActivationGradient(T const& dy, T const& input_value, T const& alpha, T const& beta)
{
    static_assert(Mode == ActivationMode::PASTHRU || Mode == ActivationMode::RELU
                      || Mode == ActivationMode::CLIPPED_RELU || Mode == ActivationMode::CLAMP,
                  "Unsupported activation mode");

    if constexpr(Mode == ActivationMode::PASTHRU)
    {
        return dy;
    }
    else if constexpr(Mode == ActivationMode::RELU)
    {
        return (input_value > T(0)) ? dy : T(0);
    }
    else if constexpr(Mode == ActivationMode::CLIPPED_RELU)
    {
        return (input_value > T(0) && input_value <= alpha) ? dy : T(0);
    }
    else if constexpr(Mode == ActivationMode::CLAMP)
    {
        return (input_value > alpha && input_value <= beta) ? dy : T(0);
    }
}

} // namespace hip_kernel_provider
