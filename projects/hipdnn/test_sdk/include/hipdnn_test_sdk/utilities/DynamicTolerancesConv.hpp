// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
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
 * @brief A place in a fused graph where a value is rounded to a type with fewer bits than float,
 * which a float-computing reference does not do at that place.
 *
 * A value v rounded to a type with epsilon eps_p moves by at most half a unit in the last place,
 * eps_p / 2 * |v|, and the elements of a tensor reach about twice their root mean square, so the
 * point contributes eps_p * magnitude at its own place (tolerances.md, "Rounding points"). The
 * error reaches the graph's output multiplied by the slope of the nodes between the two.
 */
struct RoundingPoint
{
    /// Machine epsilon of the type the value is rounded to. Finite and not negative.
    double epsilon = 0.0;
    /// Typical magnitude of the value rounded (root mean square). Finite and not negative.
    double magnitude = 0.0;
    /// Largest factor by which an error in the value reaches the graph's output: 1 through an
    /// add or subtract, the largest magnitude of the scale through a multiply, and the sum over
    /// the paths when the value reaches the output by several. Finite and not negative.
    double slope = 1.0;
};

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
    /// The places between the convolution and the output where a value is rounded to half or
    /// bfloat16 (none for a graph whose intermediates and pointwise nodes are float).
    std::vector<RoundingPoint> roundingPoints;
};

/// An absolute and a relative tolerance, graded per element as |impl - ref| <= atol + rtol*|ref|.
struct AbsRelTolerance
{
    double atol = 0.0;
    double rtol = 0.0;
};

/// The summation-order coefficient of a float output when nothing sets it: the smallest whole
/// number that leaves a headroom of at least 1.5 over the largest need of any solver family
/// measured, Winograd aside (2.46 on the calibration data, see tolerances.md). An engine whose
/// solvers err more sets its own.
inline constexpr double DEFAULT_FLOAT_SUMMATION_ORDER = 4.0;

/// The coefficients of the output tolerance an engine may set; an unset one keeps the default of
/// the output type.
struct ConvFpropCoefficients
{
    /// Multiplies sqrt(K) * u32 in the summation-order term. Default: 1 for half and bfloat16,
    /// DEFAULT_FLOAT_SUMMATION_ORDER for float. Finite and positive.
    std::optional<double> summationOrder;
};

/**
 * @brief Per-element tolerance for a convolution-forward output stored as half, bfloat16 or
 * float, with float accumulation, graded against a float-accumulating reference.
 *
 *   |impl_i - ref_i| <= rtol * |ref_i| + atol
 *   rtol = 2 * eps                       (half, bfloat16)
 *   rtol = 2 * eps + c * sqrt(K) * u32   (float)
 *   atol = (c * sqrt(K) * u32 + lambda * eps) * sigma
 *          + sum over the rounding points of (slope * eps_p * magnitude) + minNormal
 *   sigma = sqrt(K * mean(x^2) * mean(w^2)) * outputScale
 *
 * - eps is the machine epsilon of the output type, u32 the unit roundoff of float (2^-24),
 *   minNormal the smallest normal number of the output type.
 * - c is the summation-order coefficient: 1 for half and bfloat16, and for float
 *   DEFAULT_FLOAT_SUMMATION_ORDER, unless `coefficients` sets it.
 * - lambda is 12 for half and 0 for bfloat16 and float.
 * - sigma is the typical magnitude of an output element, and holds only for inputs with a zero
 *   mean: for a nonzero mean the partial sums do not cancel, the output is larger than sigma, and
 *   only the rtol term grows with it.
 *
 * The terms cover the two roundings of the float result to the output type (2 * eps), the order of
 * the float summation (c * sqrt(K) * u32, on sigma and for float on |ref|), partial results that
 * some half kernels keep in half (lambda * eps * sigma), the roundings of intermediates to half
 * or bfloat16 (the rounding points; see RoundingPoint) and flushed denormals (minNormal). The
 * derivation, the calibration of lambda and c and the comparison with calculateConvFpropTolerance()
 * are in tolerances.md in the documentation of the integration tests.
 *
 * @tparam OutputType half, bfloat16 or float.
 * @param scale The problem statistics; see ConvFpropErrorScale.
 * @param coefficients Engine-set coefficients; see ConvFpropCoefficients.
 * @return atol and rtol for an allclose comparison.
 * @throws std::invalid_argument if scale.accumulationLength is 0, a mean square is negative or
 * not finite, outputScale is not finite, a rounding point has a negative or not finite field, or
 * the summation-order coefficient is not finite and positive.
 * @throws std::overflow_error if atol exceeds the largest value of OutputType, or if sigma is not
 * a number (an overflowing product of the mean squares against a zero factor).
 */
template <typename OutputType>
AbsRelTolerance calculateConvFpropOutputTolerance(const ConvFpropErrorScale& scale,
                                                  const ConvFpropCoefficients& coefficients = {})
{
    static_assert(std::is_same_v<OutputType, half> || std::is_same_v<OutputType, bfloat16>
                      || std::is_same_v<OutputType, float>,
                  "calculateConvFpropOutputTolerance models half, bfloat16 and float outputs only");

    const auto validMoment = [](double v) { return std::isfinite(v) && v >= 0.0; };
    const bool validPoints = std::all_of(
        scale.roundingPoints.begin(), scale.roundingPoints.end(), [&](const RoundingPoint& p) {
            return validMoment(p.epsilon) && validMoment(p.magnitude) && validMoment(p.slope);
        });
    if(coefficients.summationOrder.has_value()
       && (!std::isfinite(*coefficients.summationOrder) || !(*coefficients.summationOrder > 0.0)))
    {
        throw std::invalid_argument("calculateConvFpropOutputTolerance: the summation-order "
                                    "coefficient must be finite and positive.");
    }
    if(scale.accumulationLength == 0 || !validMoment(scale.inputMeanSquare)
       || !validMoment(scale.weightMeanSquare) || !std::isfinite(scale.outputScale) || !validPoints)
    {
        throw std::invalid_argument("calculateConvFpropOutputTolerance: ConvFpropErrorScale needs "
                                    "a nonzero accumulationLength, finite non-negative mean "
                                    "squares, a finite outputScale and finite non-negative "
                                    "rounding points.");
    }

    constexpr double OUTPUT_ROUNDING_MARGIN = 2.0;
    constexpr double IO_INTERMEDIATE_LAMBDA = std::is_same_v<OutputType, half> ? 12.0 : 0.0;
    constexpr double ROUNDING_POINT_MARGIN = 1.0;
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

    double roundingPointError = 0.0;
    for(const auto& point : scale.roundingPoints)
    {
        roundingPointError += point.slope * point.epsilon * point.magnitude;
    }

    // The summation-order error of a float output is relative to the partial sums, which are
    // about the size of the output or larger: the same coefficient on |ref| as on sigma. For
    // half and bfloat16 that term is below 1 % of the rounding of the output and is left out.
    const double summationOrder = coefficients.summationOrder.value_or(
        std::is_same_v<OutputType, float> ? DEFAULT_FLOAT_SUMMATION_ORDER : 1.0);
    const double summationError = summationOrder * std::sqrt(k) * floatUnitRoundoff;

    AbsRelTolerance tolerance;
    tolerance.rtol = OUTPUT_ROUNDING_MARGIN * eps;
    if constexpr(std::is_same_v<OutputType, float>)
    {
        tolerance.rtol += summationError;
    }
    tolerance.atol = (summationError + IO_INTERMEDIATE_LAMBDA * eps) * sigma
                     + ROUNDING_POINT_MARGIN * roundingPointError + minNormal;
    hipdnn_test_sdk::utilities::validateToleranceRange<OutputType>(tolerance.atol);
    return tolerance;
}

} // namespace hipdnn_test_sdk::utilities::conv
