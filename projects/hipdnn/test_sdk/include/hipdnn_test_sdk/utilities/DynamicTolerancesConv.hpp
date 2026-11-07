// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

namespace hipdnn_test_sdk::utilities::conv
{

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
    hipdnn_test_sdk::utilities::validateComputeType<ComputeType>();

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
    const double gamma = hipdnn_test_sdk::utilities::computeGamma(numberOfAccumulations, epsilon);
    hipdnn_test_sdk::utilities::validateGamma(gamma);

    double accumulatedTolerance = gamma * sumAbsProductBound;

    // Input casting error (2 distinct inputs: x and dy)
    accumulatedTolerance
        += hipdnn_test_sdk::utilities::computeInputCastingError<InputType, ComputeType>(
            sumAbsProductBound);

    // Output casting error
    const double outputCastError
        = hipdnn_test_sdk::utilities::computeOutputCastingError<OutputType, ComputeType>(
            sumAbsProductBound);

    const double totalTolerance = accumulatedTolerance + outputCastError;

    hipdnn_test_sdk::utilities::validateToleranceRange<OutputType>(totalTolerance);

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
    hipdnn_test_sdk::utilities::validateComputeType<ComputeType>();

    // wDims: [K, C, Spatial...] for 1D: [K, C, W], 2D: [K, C, R, S], 3D: [K, C, D, R, S]
    // Accumulation for input gradients (dx) happens over K (output channels) and Spatial filter dimensions.
    // dx[n, c, h, w] = sum_{k, r, s} dy[n, k, p, q] * w[k, c, r, s]

    if(wDims.empty() || wDims.size() < 3)
    {
        throw std::invalid_argument(
            "wDims must have at least 3 dimensions for convolution [K, C, Spatial...].");
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
    const double gamma = hipdnn_test_sdk::utilities::computeGamma(numberOfAccumulations, epsilon);
    hipdnn_test_sdk::utilities::validateGamma(gamma);

    double accumulatedTolerance = gamma * sumAbsProductBound;

    // Input casting error (2 distinct inputs: dy and w)
    accumulatedTolerance
        += hipdnn_test_sdk::utilities::computeInputCastingError<InputType, ComputeType>(
            sumAbsProductBound);

    // Output casting error
    const double outputCastError
        = hipdnn_test_sdk::utilities::computeOutputCastingError<OutputType, ComputeType>(
            sumAbsProductBound);

    const double totalTolerance = accumulatedTolerance + outputCastError;

    hipdnn_test_sdk::utilities::validateToleranceRange<OutputType>(totalTolerance);

    return static_cast<float>(totalTolerance);
}

/**
 * @brief Products summed per output element of a forward convolution: C/groups times the filter
 * volume, i.e. the product of every weight dimension but the first.
 *
 * @param wDims The dimensions of the weight/filter tensor (w), [K, C/groups, R, S, ...].
 * @return The number of products, 0 when an extent is 0 (ConvFpropErrorScale and
 * calculateConvFpropOutputTolerance reject that).
 * @throws std::invalid_argument if wDims has fewer than 2 dimensions or a negative extent.
 */
inline uint64_t convFpropAccumulationLength(const std::vector<int64_t>& wDims)
{
    // y[n, k, h, w] = sum_{c, r, s} x[n, c, h+r, w+s] * w[k, c, r, s]
    if(wDims.size() < 2)
    {
        throw std::invalid_argument("wDims must have at least 2 dimensions (K, C).");
    }
    if(std::any_of(wDims.begin(), wDims.end(), [](int64_t extent) { return extent < 0; }))
    {
        throw std::invalid_argument("wDims must not have a negative extent.");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(wDims[1]); // Input channels C
    for(size_t i = 2; i < wDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(wDims[i]); // Filter spatial dimensions
    }
    return numberOfAccumulations;
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
    hipdnn_test_sdk::utilities::validateComputeType<ComputeType>();

    // Accumulation for output (y) happens over C (input channels) and R, S (filter spatial dimensions).
    const uint64_t numberOfAccumulations = convFpropAccumulationLength(wDims);

    const double maxAbsInput = std::max(std::abs(inputMin), std::abs(inputMax));
    const double maxAbsW = std::max(std::abs(wMin), std::abs(wMax));

    // Worst case product magnitude
    const double maxProduct = maxAbsInput * maxAbsW;

    // Bound on sum(|x_i * w_i|)
    const double sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    const double gamma = hipdnn_test_sdk::utilities::computeGamma(numberOfAccumulations, epsilon);
    hipdnn_test_sdk::utilities::validateGamma(gamma);

    double accumulatedTolerance = gamma * sumAbsProductBound;

    // Input casting error (2 distinct inputs: x and w)
    accumulatedTolerance
        += hipdnn_test_sdk::utilities::computeInputCastingError<InputType, ComputeType>(
            sumAbsProductBound);

    // Output casting error
    const double outputCastError
        = hipdnn_test_sdk::utilities::computeOutputCastingError<OutputType, ComputeType>(
            sumAbsProductBound);

    const double totalTolerance = accumulatedTolerance + outputCastError;

    hipdnn_test_sdk::utilities::validateToleranceRange<OutputType>(totalTolerance);

    return static_cast<float>(totalTolerance);
}

/**
 * @brief Statistics of a convolution-forward problem that set the size of its rounding error.
 *
 * Everything here is cheap to compute from the graph and the input tensors, before or after the
 * engine runs; no reference output is needed.
 */
struct ConvFpropErrorScale
{
    /// K: terms summed per output element, C/groups times the filter volume (see
    /// convFpropAccumulationLength). At least 1.
    uint64_t accumulationLength = 0;
    /// Mean of x^2 over the input tensor. Finite and not negative.
    double inputMeanSquare = 0.0;
    /// Mean of w^2 over the weight tensor. Finite and not negative.
    double weightMeanSquare = 0.0;
    /// Product, over the fused multiplies, of the largest |scale| each applies to the convolution
    /// output (1 when there is none).
    double outputScale = 1.0;
};

/// An absolute and a relative tolerance, graded per element as |impl - ref| <= atol + rtol*|ref|.
struct AbsRelTolerance
{
    double atol = 0.0;
    double rtol = 0.0;
};

/**
 * @brief Per-element tolerance for a convolution-forward output stored as half or bfloat16, with
 * float accumulation, graded against a float-accumulating reference.
 *
 *   |impl_i - ref_i| <= rtol * |ref_i| + atol
 *   rtol = 2 * eps
 *   atol = (sqrt(K) * u32 + lambda * eps) * sigma + minNormal
 *   sigma = sqrt(K * mean(x^2) * mean(w^2)) * outputScale
 *
 * - eps is the machine epsilon of the output type, u32 the unit roundoff of float (2^-24),
 *   minNormal the smallest normal number of the output type.
 * - lambda is 12 for half and 0 for bfloat16.
 * - sigma is the typical magnitude of an output element, and holds only for inputs with a zero
 *   mean: for a nonzero mean the partial sums do not cancel, the output is larger than sigma, and
 *   only the rtol term grows with it.
 *
 * Each term is one way two correct implementations differ. rtol covers the two roundings of the
 * float result to the output type (2 * eps is a margin of 2 to 4 ulp over the one-ulp bound).
 * sqrt(K) * u32 * sigma is the random walk of float rounding errors when two sums of the same K
 * products add them in a different order; it decides an output near zero. lambda * eps * sigma
 * covers partial results some half kernels keep in half (12 is 1.5 times the largest 8 * eps *
 * sigma measured on MIOpen half kernels; bfloat16 showed none, so 0). minNormal covers flushed
 * denormals. The constants were fitted on 5,091 correct MIOpen outputs on gfx942 and checked on
 * further seeds there and on gfx90a without a failure. The derivation, the calibration data and
 * the comparison with calculateConvFpropTolerance() are written out in tolerances.md in the
 * documentation of the integration tests.
 *
 * @tparam OutputType half or bfloat16.
 * @param scale The problem statistics; see ConvFpropErrorScale.
 * @return atol and rtol for an allclose comparison.
 * @throws std::invalid_argument if scale.accumulationLength is 0, a mean square is negative or
 * not finite, or outputScale is not finite.
 * @throws std::overflow_error if atol exceeds the largest value of OutputType, or if sigma is not
 * a number (an overflowing product of the mean squares against a zero factor).
 */
template <typename OutputType>
AbsRelTolerance calculateConvFpropOutputTolerance(const ConvFpropErrorScale& scale)
{
    static_assert(std::is_same_v<OutputType, half> || std::is_same_v<OutputType, bfloat16>,
                  "calculateConvFpropOutputTolerance models half and bfloat16 outputs only");

    const auto validMoment = [](double v) { return std::isfinite(v) && v >= 0.0; };
    if(scale.accumulationLength == 0 || !validMoment(scale.inputMeanSquare)
       || !validMoment(scale.weightMeanSquare) || !std::isfinite(scale.outputScale))
    {
        throw std::invalid_argument("calculateConvFpropOutputTolerance: ConvFpropErrorScale needs "
                                    "a nonzero accumulationLength, finite non-negative mean "
                                    "squares and a finite outputScale.");
    }

    constexpr double OUTPUT_ROUNDING_MARGIN = 2.0;
    constexpr double IO_INTERMEDIATE_LAMBDA = std::is_same_v<OutputType, half> ? 12.0 : 0.0;
    const double floatUnitRoundoff = std::ldexp(1.0, -24);

    const auto eps = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    const auto minNormal = static_cast<double>(std::numeric_limits<OutputType>::min());
    const auto k = static_cast<double>(scale.accumulationLength);
    const double sigma = std::sqrt(k * scale.inputMeanSquare * scale.weightMeanSquare)
                         * std::abs(scale.outputScale);
    // Each field is valid on its own, but k * inputMeanSquare can overflow to infinity while
    // weightMeanSquare or outputScale is 0: infinity times 0 is NaN, which no range check
    // rejects, and a NaN atol fails every comparison.
    if(std::isnan(sigma))
    {
        throw std::overflow_error(
            "calculateConvFpropOutputTolerance: the typical output magnitude is not a number "
            "(the product of the statistics overflows against a zero factor).");
    }

    AbsRelTolerance tolerance;
    tolerance.rtol = OUTPUT_ROUNDING_MARGIN * eps;
    tolerance.atol
        = (std::sqrt(k) * floatUnitRoundoff + IO_INTERMEDIATE_LAMBDA * eps) * sigma + minNormal;
    hipdnn_test_sdk::utilities::validateToleranceRange<OutputType>(tolerance.atol);
    return tolerance;
}

} // namespace hipdnn_test_sdk::utilities::conv
