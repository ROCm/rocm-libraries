// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/utilities/StaticCast.hpp>
#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_test_sdk/utilities/NumericLimits.hpp>

namespace hipdnn_test_sdk::utilities::conv
{

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
 * @param dyMin The minimum value in the output gradient tensor.
 * @param dyMax The maximum value in the output gradient tensor.
 * @param dyDims The dimensions of the output gradient tensor (dy).
 * @return The calculated tolerance value cast to `OutputType`.
 */
template <typename OutputType, typename ComputeType = float>
OutputType calculateConvWrwTolerance(double inputMin,
                                     double inputMax,
                                     double dyMin,
                                     double dyMax,
                                     const std::vector<int64_t>& dyDims)
{
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

    // Worst case: inputs are always max magnitude and signs align to always add up.
    double maxProduct = maxAbsInput * maxAbsDy;

    // Calculate the worst-case accumulation error.
    //
    // We model the accumulation of 'numberOfAccumulations' products. In the worst-case scenario,
    // all inputs have the maximum magnitude, causing the accumulated value to grow linearly:
    // V_i = i * maxProduct
    //
    // At each accumulation step 'i', the floating-point addition introduces a rounding error.
    // This error is bounded by the machine epsilon relative to the current magnitude V_i.
    // Error_i <= V_i * epsilon_compute
    //
    // We approximate the error at step i as (V_i * epsilon_compute), where epsilon_compute
    // is the machine epsilon of the compute type. This assumes the relative error is constant,
    // which is a standard property of floating-point arithmetic.
    //
    // The total accumulation error is the sum of errors at each step:
    // TotalError ≈ sum_{i=1}^{N} (i * maxProduct * epsilon_compute)
    //            = maxProduct * epsilon_compute * sum_{i=1}^{N} (i)
    //            = maxProduct * epsilon_compute * (N * (N + 1) / 2)
    //
    // This analytical formula provides a conservative upper bound on the error without
    // requiring an O(N) loop, which is efficient for large accumulation counts.

    double epsilon = getEpsilon<ComputeType>();
    double accumulatedTolerance = maxProduct * epsilon * static_cast<double>(numberOfAccumulations)
                                  * static_cast<double>(numberOfAccumulations + 1) * 0.5;

    // Calculate final accumulated value
    double maxPossibleOutputValue = static_cast<double>(numberOfAccumulations) * maxProduct;

    // Calculate precision loss due to casting from ComputeType to OutputType
    // The error is bounded by the precision of the OutputType at the final value.
    // We approximate the resolution as value * epsilon.
    double outputEpsilon = getEpsilon<OutputType>();
    double castTolerance = std::abs(maxPossibleOutputValue) * outputEpsilon;

    // Total tolerance is the sum of accumulation error and cast error
    double totalTolerance = accumulatedTolerance + castTolerance;

    return hipdnn_data_sdk::utilities::staticCast<OutputType>(totalTolerance);
}

} // namespace hipdnn_test_sdk::utilities::conv
