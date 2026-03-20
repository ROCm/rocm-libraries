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
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>

namespace hipdnn_test_sdk::utilities
{

/**
 * @brief Computes Higham's error growth factor γ_k for floating-point accumulation.
 *
 * For k accumulations with machine epsilon u:
 *   - Linear (when nU < 0.01):     γ_k = (2*k*u) / (1 - 2*k*u)  [deterministic worst-case]
 *   - Statistical (when nU >= 0.01): γ_k = K_SIGMA * sqrt(2*k) * u [probabilistic, K_SIGMA=6]
 *
 * where nU = 2*k*u.
 *
 * The switch at nU = 0.01 is chosen because the linear bound inflates noticeably
 * above this point (at nU=0.01 the overshoot is ~1%, at nU=0.1 it's ~11%).
 *
 * This is a pure math function that never throws. Callers are responsible for
 * checking if the returned gamma is too large for their use case (e.g., gamma >= 0.5
 * means the error bound exceeds 50% of the signal).
 *
 * Reusable across conv, matmul, and any operation that accumulates k products.
 *
 * @param k Number of accumulations (reduction dimension).
 * @param epsilon Machine epsilon for the compute type.
 * @return The error growth factor γ_k as a double.
 */
inline double computeGamma(uint64_t k, double epsilon)
{
    constexpr double NU_THRESHOLD = 0.01;

    const double nU = 2.0 * static_cast<double>(k) * epsilon;

    if(nU < NU_THRESHOLD)
    {
        return nU / (1.0 - nU);
    }

    constexpr double K_SIGMA = 6.0;
    return K_SIGMA * std::sqrt(2.0 * static_cast<double>(k)) * epsilon;
}

} // namespace hipdnn_test_sdk::utilities

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

    const double maxAbsInput = std::max(std::abs(inputMin), std::abs(inputMax));
    const double maxAbsDy = std::max(std::abs(dyMin), std::abs(dyMax));

    // Worst case product magnitude
    const double maxProduct = maxAbsInput * maxAbsDy;

    // Bound on sum(|x_i * y_i|)
    const double sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    double accumulatedTolerance = 0.0;

    if constexpr(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>)
    {
        // High Precision: Linear bound (Classical)
        // Error <= gamma_2n * sum(|x_i * y_i|)
        // gamma_n = n * u / (1 - n * u)
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = 2 * n * u / (1 - 2 * n * u)
        const double nU = 2.0 * static_cast<double>(numberOfAccumulations) * epsilon;

        if(nU >= 1.0)
        {
            throw std::overflow_error(
                "Number of accumulations is too large for the given precision. "
                "Error bound is undefined/infinite.");
        }

        const double gamma = nU / (1.0 - nU);
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
        const double gamma
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
        const double castingError = 2.0 * sumAbsProductBound * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate final accumulated value magnitude for casting error
    const double maxPossibleOutputValue = sumAbsProductBound;

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
    const double totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

/**
 * @brief Calculates the expected tolerance for Convolution Backward Data (DGrad) operations.
 *
 * This function estimates the maximum expected error due to floating-point accumulation during the
 * computation of input gradients. It considers the accumulation of products of output gradients
 * and filter weights over the output channels and spatial filter dimensions.
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
 * @tparam OutputType The data type of the output (input gradients dx).
 * @tparam InputType The data type of the input tensors (dy and w).
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param dyMin The minimum value in the output gradient tensor (dy).
 * @param dyMax The maximum value in the output gradient tensor (dy).
 * @param wMin The minimum value in the filter weights tensor (w).
 * @param wMax The maximum value in the filter weights tensor (w).
 * @param wDims The dimensions of the filter weights tensor (w): [K, C, R, S] or [K, C, D, R, S].
 * @return The calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateConvDgradTolerance(
    double dyMin, double dyMax, double wMin, double wMax, const std::vector<int64_t>& wDims)
{
    // Validate ComputeType
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    // wDims: [K, C, Spatial...] for 2D: [K, C, R, S], for 3D: [K, C, D, R, S]
    // Accumulation for input gradients (dx) happens over K (output channels) and Spatial filter dimensions.
    // dx[n, c, h, w] = sum_{k, r, s} dy[n, k, p, q] * w[k, c, r, s]

    if(wDims.empty() || wDims.size() < 4)
    {
        throw std::invalid_argument(
            "wDims must have at least 4 dimensions for 2D convolution [K, C, R, S].");
    }

    // Number of accumulations = K * (product of spatial dimensions)
    // For 2D: K * R * S
    // For 3D: K * D * R * S
    auto numberOfAccumulations = static_cast<uint64_t>(wDims[0]); // K (output channels)
    for(size_t i = 2; i < wDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(wDims[i]); // Spatial dimensions
    }

    const double maxAbsDy = std::max(std::abs(dyMin), std::abs(dyMax));
    const double maxAbsW = std::max(std::abs(wMin), std::abs(wMax));

    // Worst case product magnitude
    const double maxProduct = maxAbsDy * maxAbsW;

    // Bound on sum(|dy_i * w_i|)
    const double sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    double accumulatedTolerance = 0.0;

    if constexpr(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>)
    {
        // High Precision: Linear bound (Classical)
        // Error <= gamma_2n * sum(|dy_i * w_i|)
        // gamma_n = n * u / (1 - n * u)
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = 2 * n * u / (1 - 2 * n * u)
        const double nU = 2.0 * static_cast<double>(numberOfAccumulations) * epsilon;

        if(nU >= 1.0)
        {
            throw std::overflow_error(
                "Number of accumulations is too large for the given precision. "
                "Error bound is undefined/infinite.");
        }

        const double gamma = nU / (1.0 - nU);
        accumulatedTolerance = gamma * sumAbsProductBound;
    }
    else
    {
        // Lower Precision (FP16, BF16): Statistical bound (Probabilistic)
        // Error <= gamma_2n * sum(|dy_i * w_i|)
        // gamma_n = sqrt(n) * u
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = sqrt(2n) * u
        // k_sigma = 6.0 for high confidence
        constexpr double K_SIGMA = 6.0;
        const double gamma
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
        // Let dy_approx = dy_true * (1 + d_dy) and w_approx = w_true * (1 + d_w)
        // where |d_dy|, |d_w| <= epsilon_compute (relative error bound).
        // Product P_approx = dy_approx * w_approx ≈ dy_true * w_true * (1 + d_dy + d_w)
        // Error_P = |P_approx - P_true| ≈ |P_true| * |d_dy + d_w|
        // Error_P <= |P_true| * (|d_dy| + |d_w|) <= |P_true| * (epsilon + epsilon)
        // Error_P <= 2 * |P_true| * epsilon
        // Summing over N accumulations: Total_Error <= 2 * sum(|dy_i * w_i|) * epsilon
        const double castingError = 2.0 * sumAbsProductBound * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate final accumulated value magnitude for casting error
    const double maxPossibleOutputValue = sumAbsProductBound;

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
    const double totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

/**
 * @brief Calculates the expected tolerance for Convolution Forward Propagation (fprop) operations.
 *
 * This function estimates the maximum expected error due to floating-point accumulation during the
 * computation of forward convolution. It considers the accumulation of products of inputs and weights
 * over the input channels and filter spatial dimensions.
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
 * @tparam OutputType The data type of the output (forward convolution output).
 * @tparam InputType The data type of the input tensor.
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param inputMin The minimum value in the input tensor.
 * @param inputMax The maximum value in the input tensor.
 * @param wMin The minimum value in the weight/filter tensor.
 * @param wMax The maximum value in the weight/filter tensor.
 * @param wDims The dimensions of the weight/filter tensor (w).
 * @return The calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateConvFpropTolerance(
    double inputMin, double inputMax, double wMin, double wMax, const std::vector<int64_t>& wDims)
{
    // Validate ComputeType
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    // wDims: [K, C, R, S]
    // Accumulation for output (y) happens over C (input channels) and R, S (filter spatial dimensions).
    // y[n, k, h, w] = sum_{c, r, s} x[n, c, h+r, w+s] * w[k, c, r, s]

    if(wDims.empty() || wDims.size() < 2)
    {
        throw std::invalid_argument("wDims must have at least 2 dimensions (K, C).");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(wDims[1]); // Input channels C
    for(size_t i = 2; i < wDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(wDims[i]); // Filter spatial dimensions R, S
    }

    const double maxAbsInput = std::max(std::abs(inputMin), std::abs(inputMax));
    const double maxAbsW = std::max(std::abs(wMin), std::abs(wMax));

    // Worst case product magnitude
    const double maxProduct = maxAbsInput * maxAbsW;

    // Bound on sum(|x_i * w_i|)
    const double sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    double accumulatedTolerance = 0.0;

    if constexpr(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>)
    {
        // High Precision: Linear bound (Classical)
        // Error <= gamma_2n * sum(|x_i * w_i|)
        // gamma_n = n * u / (1 - n * u)
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = 2 * n * u / (1 - 2 * n * u)
        const double nU = 2.0 * static_cast<double>(numberOfAccumulations) * epsilon;

        if(nU >= 1.0)
        {
            throw std::overflow_error(
                "Number of accumulations is too large for the given precision. "
                "Error bound is undefined/infinite.");
        }

        const double gamma = nU / (1.0 - nU);
        accumulatedTolerance = gamma * sumAbsProductBound;
    }
    else
    {
        // Lower Precision (FP16, BF16): Statistical bound (Probabilistic)
        // Error <= gamma_2n * sum(|x_i * w_i|)
        // gamma_n = sqrt(n) * u
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = sqrt(2n) * u
        // k_sigma = 6.0 for high confidence
        constexpr double K_SIGMA = 6.0;
        const double gamma
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
        // Let x_approx = x_true * (1 + d_x) and w_approx = w_true * (1 + d_w)
        // where |d_x|, |d_w| <= epsilon_compute (relative error bound).
        // Product P_approx = x_approx * w_approx ≈ x_true * w_true * (1 + d_x + d_w)
        // Error_P = |P_approx - P_true| ≈ |P_true| * |d_x + d_w|
        // Error_P <= |P_true| * (|d_x| + |d_w|) <= |P_true| * (epsilon + epsilon)
        // Error_P <= 2 * |P_true| * epsilon
        // Summing over N accumulations: Total_Error <= 2 * sum(|x_i * w_i|) * epsilon
        const double castingError = 2.0 * sumAbsProductBound * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate final accumulated value magnitude for casting error
    const double maxPossibleOutputValue = sumAbsProductBound;

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
    const double totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

} // namespace hipdnn_test_sdk::utilities::conv

namespace hipdnn_test_sdk::utilities::matmul
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::TensorView;

/**
 * @brief Computes the infinity-norm (max absolute row sum) of a matrix tensor.
 *
 * The infinity-norm is defined as: ||A||_inf = max_i (sum_j |A[i,j]|)
 *
 * This is the appropriate subordinate matrix norm for element-wise error bounds
 * via Higham's analysis: max_ij |error_ij| <= gamma_k * ||A||_inf * ||B||_inf.
 *
 * For batched tensors (>2D), the infinity-norm is computed across all batches,
 * returning the maximum row sum over all rows in all batches.
 *
 * Uses iterateAlongDimensions + ConstTensorView to correctly handle padded
 * and non-packed tensor layouts via stride-aware indexing.
 *
 * @tparam T The data type of the tensor elements.
 * @param tensor The input tensor (must have at least 2 dimensions).
 * @return The infinity-norm as a double.
 */
template <typename T>
double computeMatrixInfNorm(ITensor& tensor)
{
    using hipdnn_data_sdk::utilities::iterateAlongDimensions;

    const auto& dims = tensor.dims();
    const TensorView<T> view(tensor);

    auto cols = dims.back();

    // outerDims = [batch..., rows] — everything except the last dim (cols)
    auto outerDims = std::vector<int64_t>(dims.begin(), dims.end() - 1);

    double maxRowSum = 0.0;

    iterateAlongDimensions(outerDims, [&](const std::vector<int64_t>& outerIndices) {
        double rowSum = 0.0;

        auto fullIndices = outerIndices;
        fullIndices.push_back(0);

        for(int64_t j = 0; j < cols; ++j)
        {
            fullIndices.back() = j;
            rowSum += static_cast<double>(
                hipdnn_data_sdk::types::fabs(view.getHostValue(fullIndices)));
        }

        maxRowSum = std::max(maxRowSum, rowSum);
    });

    return maxRowSum;
}

/**
 * @brief Calculates the expected tolerance for Matrix Multiplication operations.
 *
 * This function estimates the maximum expected element-wise error due to floating-point
 * accumulation during the computation of C = A × B using Higham's error analysis.
 *
 * Norm Selection: Infinity-norm (max absolute row sum)
 * =====================================================
 * We use the infinity-norm ||A||_inf = max_i (sum_j |A[i,j]|) because Higham's element-wise
 * error bound for matrix multiplication gives:
 *   max_ij |fl(C)_ij - C_ij| <= gamma_k * ||A||_inf * ||B||_inf
 *
 * This is the correct subordinate matrix norm for bounding the maximum element-wise error,
 * which is what allClose-style validation checks.
 *
 * Error Bound (Higham's Analysis):
 * =================================
 * For C = A*B where A is m×k and B is k×n:
 *   max_ij |error_ij| <= γ_k * ||A||_inf * ||B||_inf
 *
 * where γ_k is the error growth factor:
 *   - High Precision (FP32, FP64): γ_k = (2*k*u) / (1 - 2*k*u)  (linear worst-case bound)
 *   - Low Precision (FP16, BF16):  γ_k = K_SIGMA * sqrt(2*k) * u  (statistical bound, K_SIGMA=6)
 *   - u = machine epsilon for ComputeType
 *
 * The function also accounts for precision loss from input/output casting if needed.
 *
 * @tparam OutputType The data type of the output matrix C.
 * @tparam InputType The data type of the input matrices A and B.
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param a The input matrix A.
 * @param b The input matrix B.
 * @return The calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateMatmulTolerance(ITensor& a, ITensor& b)
{
    // Validate ComputeType
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    // Validate tensor dimensions for matmul compatibility
    const auto& aDims = a.dims();
    const auto& bDims = b.dims();

    if(aDims.size() < 2 || bDims.size() < 2)
    {
        throw std::invalid_argument("Matrices must have at least 2 dimensions for matmul.");
    }

    // Extract reduction dimension K (last dim of A, second-to-last dim of B)
    const int64_t k = aDims[aDims.size() - 1];
    const int64_t bRows = bDims[bDims.size() - 2];

    if(k != bRows)
    {
        throw std::invalid_argument(
            "Matrix dimensions incompatible for multiplication: A columns != B rows.");
    }

    if(k <= 0)
    {
        throw std::invalid_argument("Reduction dimension k must be positive.");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(k);

    // Compute infinity-norms of input matrices (max absolute row sum)
    const double normA = computeMatrixInfNorm<InputType>(a);
    const double normB = computeMatrixInfNorm<InputType>(b);

    // Apply Higham's error bound: max_ij |error_ij| <= γ_k * ||A||_inf * ||B||_inf
    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    const double gamma = hipdnn_test_sdk::utilities::computeGamma(numberOfAccumulations, epsilon);

    constexpr double GAMMA_MAX = 0.5;
    if(gamma >= GAMMA_MAX)
    {
        throw std::overflow_error(
            "Error growth factor gamma >= 0.5: the accumulation error exceeds 50% of the signal. "
            "The computation may be numerically meaningless at this precision and reduction size.");
    }

    double accumulatedTolerance = gamma * normA * normB;

    // Calculate input casting error
    // If InputType has higher precision (smaller epsilon) than ComputeType, we lose precision
    // when loading inputs (downcasting). Example: double -> float.
    // If InputType has lower precision than ComputeType, no additional error (upcasting).
    auto inputEpsilon = static_cast<double>(std::numeric_limits<InputType>::epsilon());
    if(inputEpsilon < epsilon)
    {
        // Input precision is higher than compute precision, so we have casting error.
        // Each element loses precision proportional to epsilon when cast.
        // Worst-case error contribution from input casting:
        //   Error ≈ 2 * ||A||_inf * ||B||_inf * epsilon
        const double castingError = 2.0 * normA * normB * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate output casting error
    // If OutputType has lower precision (larger epsilon) than ComputeType, we lose precision
    // when storing the result (downcasting). Example: float -> half.
    auto outputEpsilon = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    double castTolerance = 0.0;

    if(outputEpsilon > epsilon)
    {
        // The error is bounded by the precision of the OutputType at the final accumulated value.
        // Maximum possible output magnitude: ||A||_inf * ||B||_inf
        const double maxPossibleOutputValue = normA * normB;
        castTolerance = maxPossibleOutputValue * outputEpsilon;
    }

    // Total tolerance is the sum of accumulation error and casting errors
    const double totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

} // namespace hipdnn_test_sdk::utilities::matmul
