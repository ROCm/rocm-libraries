// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>

namespace hipdnn_test_sdk::utilities::conv
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Calculates the expected tolerance for Convolution Backward Weights (WrW) operations.
 *
 * This function estimates the maximum expected error due to floating-point accumulation during the
 * computation of weight gradients. It considers the accumulation of products of inputs and output
 * gradients over the batch and spatial dimensions.
 *
 * The tolerance is calculated by simulating the accumulation process using `ComputeType` precision
 * and adding the precision loss from casting the final result to `OutputType`.
 *
 * Error Estimation Strategy:
 * - High Precision (FP32, FP64): Uses a linear worst-case bound (Classical).
 *   Error ≈ n * epsilon * maxProduct
 * - Lower Precision (FP16, BF16): Uses a statistical/probabilistic bound.
 *   Error ≈ k * sqrt(n) * epsilon * maxProduct
 *
 * It also accounts for precision loss if inputs are cast to a lower precision ComputeType.
 *
 * @tparam OutputType The data type of the output (weight gradients).
 * @tparam InputType The data type of the input tensor.
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param inputMin The minimum value in the input tensor.
 * @param inputMax The maximum value in the input tensor.
 * @param dyMin The minimum value in the output gradient tensor.
 * @param dyMax The maximum value in the output gradient tensor.
 * @param dyDims The dimensions of the output gradient tensor (dy).
 * @return The calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateConvWrwTolerance(double inputMin,
                                double inputMax,
                                double dyMin,
                                double dyMax,
                                const std::vector<int64_t>& dyDims)
{
    // Validate ComputeType
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    // dyDims: [N, K, Spatial...]
    // Accumulation for weights (dw) happens over N and Spatial dimensions.
    // dw[k, c, r, s] = sum_{n, h, w} dy[n, k, h, w] * x[n, c, h+r, w+s]

    if(dyDims.empty() || dyDims.size() < 2)
    {
        throw std::invalid_argument("dyDims must have at least 2 dimensions (N, K).");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(dyDims[0]); // Batch size N
    for(size_t i = 2; i < dyDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(dyDims[i]); // Spatial dimensions
    }

    double maxAbsInput = std::max(std::abs(inputMin), std::abs(inputMax));
    double maxAbsDy = std::max(std::abs(dyMin), std::abs(dyMax));

    // Worst case product magnitude
    double maxProduct = maxAbsInput * maxAbsDy;

    // Bound on sum(|x_i * y_i|)
    double sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    double accumulatedTolerance = 0.0;

    if constexpr(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>)
    {
        // High Precision: Linear bound (Classical)
        // Error <= gamma_2n * sum(|x_i * y_i|)
        // gamma_n = n * u / (1 - n * u)
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = 2 * n * u / (1 - 2 * n * u)
        double nU = 2.0 * static_cast<double>(numberOfAccumulations) * epsilon;

        if(nU >= 1.0)
        {
            throw std::overflow_error(
                "Number of accumulations is too large for the given precision. "
                "Error bound is undefined/infinite.");
        }

        double gamma = nU / (1.0 - nU);
        accumulatedTolerance = gamma * sumAbsProductBound;
    }
    else
    {
        // Lower Precision (FP16, BF16): Statistical bound (Probabilistic)
        // Error <= gamma_2n * sum(|x_i * y_i|)
        // gamma_n = sqrt(n) * u
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = sqrt(2n) * u
        // k_sigma = 6.0 for high confidence
        constexpr double K_SIGMA = 6.0;
        double gamma
            = K_SIGMA * std::sqrt(2.0 * static_cast<double>(numberOfAccumulations)) * epsilon;
        accumulatedTolerance = gamma * sumAbsProductBound;
    }

    // Calculate input casting error
    // If InputType has higher precision (smaller epsilon) than ComputeType, we lose precision on load (downcasting).
    // Example: double -> float.
    // If InputType has lower precision (larger epsilon) than ComputeType, we preserve precision (upcasting).
    // Example: half -> float.
    // We only need to add tolerance if we are downcasting.
    auto inputEpsilon = static_cast<double>(std::numeric_limits<InputType>::epsilon());
    if(inputEpsilon < epsilon)
    {
        // Input precision is higher than compute precision, so we have casting error.
        // We add this to the tolerance.
        // Note: This is a worst-case bound.
        //
        // Derivation:
        // Let x_approx = x_true * (1 + d_x) and dy_approx = dy_true * (1 + d_dy)
        // where |d_x|, |d_dy| <= epsilon_compute (relative error bound).
        // Product P_approx = x_approx * dy_approx ≈ x_true * dy_true * (1 + d_x + d_dy)
        // Error_P = |P_approx - P_true| ≈ |P_true| * |d_x + d_dy|
        // Error_P <= |P_true| * (|d_x| + |d_dy|) <= |P_true| * (epsilon + epsilon)
        // Error_P <= 2 * |P_true| * epsilon
        // Summing over N accumulations: Total_Error <= 2 * sum(|x_i * y_i|) * epsilon
        double castingError = 2.0 * sumAbsProductBound * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate final accumulated value magnitude for casting error
    double maxPossibleOutputValue = sumAbsProductBound;

    double castTolerance = 0.0;
    // Calculate precision loss due to casting from ComputeType to OutputType.
    // If OutputType has lower precision (larger epsilon) than ComputeType, we lose precision (downcasting).
    // Example: float -> half.
    // If OutputType has higher precision (smaller epsilon) than ComputeType, the value is exactly representable (upcasting).
    // Example: float -> double.
    // We only need to add tolerance if we are downcasting.
    auto outputEpsilon = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    if(outputEpsilon > epsilon)
    {
        // The error is bounded by the precision of the OutputType at the final value.
        castTolerance = std::abs(maxPossibleOutputValue) * outputEpsilon;
    }

    // Total tolerance is the sum of accumulation error and cast error
    double totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

} // namespace hipdnn_test_sdk::utilities::conv
