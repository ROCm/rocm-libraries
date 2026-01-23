// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

namespace hipdnn_test_sdk::utilities
{

template <typename T>
constexpr int getMantissaBits()
{
    if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 7;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 10;
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 23;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return 52;
    }
    else
    {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for getMantissaBits");
        return 0;
    }
}

/**
 * @brief Calculates the minimum representable step (Unit in the Last Place - ULP) for a given type at a specific value.
 *
 * This function determines the precision of a floating-point type `T` at the magnitude of `value`.
 * It effectively calculates the gap between `value` and the next representable number in type `T`.
 *
 * @tparam T The floating-point type (e.g., float, half, hip_bfloat16).
 * @param value The value at which to calculate the precision.
 * @return The precision (ULP) of type `T` at `value`.
 */
template <typename T>
double getPrecision(double value)
{
    if(value == 0.0)
    {
        return 0.0;
    }

    double absVal = std::abs(value);
    double exponent = std::floor(std::log2(absVal));
    int mantissaBits = getMantissaBits<T>();

    return std::pow(2.0, exponent - mantissaBits);
}

/**
 * @brief Calculates the expected tolerance for Convolution Backward Weights (WrW) operations.
 *
 * This function estimates the maximum expected error due to floating-point accumulation during the
 * computation of weight gradients. It considers the accumulation of products of inputs and output gradients
 * over the batch and spatial dimensions.
 *
 * The tolerance is calculated by simulating the accumulation process using `ComputeType` precision
 * and adding the precision loss from casting the final result to `OutputType`.
 *
 * @tparam OutputType The data type of the output (weight gradients).
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param inputMin The minimum value in the input tensor.
 * @param inputMax The maximum value in the input tensor.
 * @param dyDims The dimensions of the output gradient tensor (dy).
 * @return The calculated tolerance value cast to `OutputType`.
 */
template <typename OutputType, typename ComputeType = float>
OutputType
    calculateConvWrwTolerance(double inputMin, double inputMax, const std::vector<int64_t>& dyDims)
{
    // dyDims: [N, K, Spatial...]
    // Accumulation for weights (dw) happens over N and Spatial dimensions.
    // dw[k, c, r, s] = sum_{n, h, w} dy[n, k, h, w] * x[n, c, h+r, w+s]

    if(dyDims.empty())
    {
        return static_cast<OutputType>(0.0);
    }

    int64_t nAccum = dyDims[0]; // Batch size N
    for(size_t i = 2; i < dyDims.size(); ++i)
    {
        nAccum *= dyDims[i]; // Spatial dimensions
    }

    double maxAbsInput = std::max(std::abs(inputMin), std::abs(inputMax));
    // Worst case: inputs are always max magnitude and signs align to always add up.
    // Each product is maxAbsInput * maxAbsInput.
    double maxProduct = maxAbsInput * maxAbsInput;

    double accumulatedTolerance = 0.0;

    // Simulate accumulation using ComputeType precision
    for(int64_t i = 1; i <= nAccum; ++i)
    {
        double currentVal = static_cast<double>(i) * maxProduct;
        accumulatedTolerance += getPrecision<ComputeType>(currentVal);
    }

    // Calculate final accumulated value
    double finalVal = static_cast<double>(nAccum) * maxProduct;

    // Calculate precision loss due to casting from ComputeType to OutputType
    // The error is bounded by the precision of the OutputType at the final value.
    double castTolerance = getPrecision<OutputType>(finalVal);

    // Total tolerance is the sum of accumulation error and cast error
    double totalTolerance = accumulatedTolerance + castTolerance;

    if constexpr(std::is_same_v<OutputType, hip_bfloat16>)
    {
        return static_cast<hip_bfloat16>(static_cast<float>(totalTolerance));
    }
    else
    {
        return static_cast<OutputType>(totalTolerance);
    }
}

} // namespace hipdnn_test_sdk::utilities
