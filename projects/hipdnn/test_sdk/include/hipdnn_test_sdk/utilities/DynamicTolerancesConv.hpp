// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
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
 * @brief Products summed per element of dx in a convolution backward-data (dgrad): the output
 * channels of one group times the most filter taps that reach one dx element.
 *
 * Along a spatial dimension with filter extent R, stride s and dilation d, the taps r that reach
 * one dx position are those with r*d congruent to a fixed value modulo s. They repeat every
 * s / gcd(s, d) taps, so at most ceil(R / (s / gcd(s, d))) of them reach it. Padding and the
 * image border only remove taps, so this is the count of the elements that see the most.
 *
 * @param wDims The dimensions of the weight tensor, [K, C/groups, R, S, ...].
 * @param groups The number of groups (dx channels divided by wDims[1]); divides wDims[0].
 * @param stride The stride of each spatial dimension, one entry per filter dimension.
 * @param dilation The dilation of each spatial dimension, one entry per filter dimension.
 * @return The number of products, 0 when an extent is 0.
 * @throws std::invalid_argument if wDims has fewer than 2 dimensions or a negative extent,
 * groups is not positive or does not divide wDims[0], or stride or dilation has the wrong
 * length or an entry below 1.
 */
inline uint64_t convDgradAccumulationLength(const std::vector<int64_t>& wDims,
                                            int64_t groups,
                                            const std::vector<int64_t>& stride,
                                            const std::vector<int64_t>& dilation)
{
    // dx[n, c, h] = sum over k in the group of c, and over taps r with (h + pad - r*d) % s == 0,
    // of dy[n, k, (h + pad - r*d) / s] * w[k, c, r]
    if(wDims.size() < 2)
    {
        throw std::invalid_argument("wDims must have at least 2 dimensions (K, C).");
    }
    if(std::any_of(wDims.begin(), wDims.end(), [](int64_t extent) { return extent < 0; }))
    {
        throw std::invalid_argument("wDims must not have a negative extent.");
    }
    if(groups < 1 || wDims[0] % groups != 0)
    {
        throw std::invalid_argument("groups must be positive and divide the output channels.");
    }
    const size_t spatialRank = wDims.size() - 2;
    const auto atLeastOne = [](int64_t v) { return v >= 1; };
    if(stride.size() != spatialRank || dilation.size() != spatialRank
       || !std::all_of(stride.begin(), stride.end(), atLeastOne)
       || !std::all_of(dilation.begin(), dilation.end(), atLeastOne))
    {
        throw std::invalid_argument(
            "stride and dilation need one entry of at least 1 per filter dimension.");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(wDims[0] / groups);
    for(size_t i = 0; i < spatialRank; ++i)
    {
        const int64_t period = stride[i] / std::gcd(stride[i], dilation[i]);
        numberOfAccumulations *= static_cast<uint64_t>((wDims[i + 2] + period - 1) / period);
    }
    return numberOfAccumulations;
}

/**
 * @brief Products summed per element of dw in a convolution backward-weights (wrw): the batch
 * size times the spatial volume of dy.
 *
 * Products that read padding are zero, so for filter taps that reach past the image border this
 * is an upper bound.
 *
 * @param dyDims The dimensions of the output-gradient tensor, [N, K, spatial...].
 * @return The number of products, 0 when an extent is 0.
 * @throws std::invalid_argument if dyDims has fewer than 2 dimensions or a negative extent.
 */
inline uint64_t convWrwAccumulationLength(const std::vector<int64_t>& dyDims)
{
    // dw[k, c, r] = sum over n and h of x[n, c, h*s + r*d - pad] * dy[n, k, h]
    if(dyDims.size() < 2)
    {
        throw std::invalid_argument("dyDims must have at least 2 dimensions (N, K).");
    }
    if(std::any_of(dyDims.begin(), dyDims.end(), [](int64_t extent) { return extent < 0; }))
    {
        throw std::invalid_argument("dyDims must not have a negative extent.");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(dyDims[0]);
    for(size_t i = 2; i < dyDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(dyDims[i]);
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
    /// Mean of x over the input tensor. Finite.
    double inputMean = 0.0;
    /// Mean of w over the weight tensor. Finite.
    double weightMean = 0.0;
};

/// Statistics of a convolution backward-data (dgrad) problem; see ConvFpropErrorScale.
struct ConvDgradErrorScale
{
    /// Terms summed per dx element (see convDgradAccumulationLength). At least 1.
    uint64_t accumulationLength = 0;
    /// Mean of dy^2 over the output-gradient tensor. Finite and not negative.
    double dyMeanSquare = 0.0;
    /// Mean of w^2 over the weight tensor. Finite and not negative.
    double weightMeanSquare = 0.0;
    /// Product of the largest |scale| of each fused multiply of the dgrad result (1 for none).
    double outputScale = 1.0;
    /// The places between the convolution and the output where a value is rounded to half or
    /// bfloat16.
    std::vector<RoundingPoint> roundingPoints;
    /// Mean of dy over the output-gradient tensor. Finite.
    double dyMean = 0.0;
    /// Mean of w over the weight tensor. Finite.
    double weightMean = 0.0;
};

/// Statistics of a convolution backward-weights (wrw) problem; see ConvFpropErrorScale.
struct ConvWrwErrorScale
{
    /// Terms summed per dw element (see convWrwAccumulationLength). At least 1.
    uint64_t accumulationLength = 0;
    /// Mean of x^2 over the input tensor. Finite and not negative.
    double inputMeanSquare = 0.0;
    /// Mean of dy^2 over the output-gradient tensor. Finite and not negative.
    double dyMeanSquare = 0.0;
    /// Product of the largest |scale| of each fused multiply of the wrw result (1 for none).
    double outputScale = 1.0;
    /// The places between the convolution and the output where a value is rounded to half or
    /// bfloat16.
    std::vector<RoundingPoint> roundingPoints;
    /// Mean of x over the input tensor. Finite.
    double inputMean = 0.0;
    /// Mean of dy over the output-gradient tensor. Finite.
    double dyMean = 0.0;
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

namespace detail
{

enum class ConvDirection
{
    FPROP,
    DGRAD,
    WRW,
};

/// The calibrated allowances of the model, per direction and output type; tolerances.md has the
/// calibration. lambda: intermediates and products, in units of eps * sigma. summationOrder: c,
/// in units of sqrt(K) * u32.
///
///   (lambda, c)   float                half      bfloat16
///   forward       (0, 4)               (12, 1)   (0, 1)
///   dgrad         (32, 4)              (12, 1)   (6, 1)
///   wrw           (32, 4)              (36, 1)   (24, 1)
struct ConvToleranceAllowances
{
    double intermediateLambda = 0.0;
    double summationOrder = 1.0;
};

template <ConvDirection Direction, typename OutputType>
constexpr ConvToleranceAllowances convToleranceAllowances()
{
    static_assert(std::is_same_v<OutputType, half> || std::is_same_v<OutputType, bfloat16>
                      || std::is_same_v<OutputType, float>,
                  "the convolution output tolerance models half, bfloat16 and float outputs only");
    if constexpr(std::is_same_v<OutputType, float>)
    {
        return {Direction == ConvDirection::FPROP ? 0.0 : 32.0, DEFAULT_FLOAT_SUMMATION_ORDER};
    }
    else if constexpr(std::is_same_v<OutputType, half>)
    {
        return {Direction == ConvDirection::WRW ? 36.0 : 12.0, 1.0};
    }
    else if constexpr(Direction == ConvDirection::FPROP)
    {
        return {0.0, 1.0};
    }
    else
    {
        return {Direction == ConvDirection::WRW ? 24.0 : 6.0, 1.0};
    }
}

/// The statistics of the two tensors whose products a convolution sums, and what happens to the
/// sum after it.
struct ConvOutputStatistics
{
    uint64_t accumulationLength = 0;
    double firstMeanSquare = 0.0;
    double secondMeanSquare = 0.0;
    double firstMean = 0.0;
    double secondMean = 0.0;
    double outputScale = 1.0;
    const std::vector<RoundingPoint>* roundingPoints = nullptr;
};

/// The model every direction shares; see calculateConvFpropOutputTolerance().
template <ConvDirection Direction, typename OutputType>
AbsRelTolerance convOutputTolerance(const ConvOutputStatistics& s,
                                    std::optional<double> summationOrderOverride,
                                    const char* caller)
{
    constexpr auto ALLOWANCES = convToleranceAllowances<Direction, OutputType>();

    const auto validMoment = [](double v) { return std::isfinite(v) && v >= 0.0; };
    const bool validPoints = std::all_of(
        s.roundingPoints->begin(), s.roundingPoints->end(), [&](const RoundingPoint& p) {
            return validMoment(p.epsilon) && validMoment(p.magnitude) && validMoment(p.slope);
        });
    if(summationOrderOverride.has_value()
       && (!std::isfinite(*summationOrderOverride) || !(*summationOrderOverride > 0.0)))
    {
        throw std::invalid_argument(std::string(caller)
                                    + ": the summation-order coefficient must be finite and "
                                      "positive.");
    }
    if(s.accumulationLength == 0 || !validMoment(s.firstMeanSquare)
       || !validMoment(s.secondMeanSquare) || !std::isfinite(s.firstMean)
       || !std::isfinite(s.secondMean) || !std::isfinite(s.outputScale) || !validPoints)
    {
        throw std::invalid_argument(std::string(caller)
                                    + ": the error scale needs a nonzero accumulationLength, "
                                      "finite non-negative mean squares, finite means, a finite "
                                      "outputScale and finite non-negative rounding points.");
    }

    constexpr double OUTPUT_ROUNDING_MARGIN = 2.0;
    constexpr double ROUNDING_POINT_MARGIN = 1.0;
    const double floatUnitRoundoff = std::ldexp(1.0, -24);

    const auto eps = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    const auto minNormal = static_cast<double>(std::numeric_limits<OutputType>::min());
    const auto k = static_cast<double>(s.accumulationLength);
    // The root mean square of a sum of K independent products a*b: K * E[a^2] * E[b^2] for the
    // spread, plus K * (K - 1) * E[a]^2 * E[b]^2 for what the means add up to.
    const double meanProduct = s.firstMean * s.secondMean;
    const double sigma = std::sqrt(k * s.firstMeanSquare * s.secondMeanSquare
                                   + k * (k - 1.0) * meanProduct * meanProduct)
                         * std::abs(s.outputScale);
    // Each field is valid on its own, but k * firstMeanSquare can overflow to infinity while
    // secondMeanSquare or outputScale is 0: infinity times 0 is NaN, which no range check
    // rejects, and a NaN atol fails every comparison.
    if(std::isnan(sigma))
    {
        throw std::overflow_error(std::string(caller)
                                  + ": the typical output magnitude is not a number (the "
                                    "product of the statistics overflows against a zero "
                                    "factor).");
    }

    double roundingPointError = 0.0;
    for(const auto& point : *s.roundingPoints)
    {
        roundingPointError += point.slope * point.epsilon * point.magnitude;
    }

    // The summation-order error of a float output is relative to the partial sums, which are
    // about the size of the output or larger: the same coefficient on |ref| as on sigma. For
    // half and bfloat16 that term is below 1 % of the rounding of the output and is left out.
    const double summationOrder = summationOrderOverride.value_or(ALLOWANCES.summationOrder);
    const double summationError = summationOrder * std::sqrt(k) * floatUnitRoundoff;

    AbsRelTolerance tolerance;
    tolerance.rtol = OUTPUT_ROUNDING_MARGIN * eps;
    if constexpr(std::is_same_v<OutputType, float>)
    {
        tolerance.rtol += summationError;
    }
    tolerance.atol = (summationError + ALLOWANCES.intermediateLambda * eps) * sigma
                     + ROUNDING_POINT_MARGIN * roundingPointError + minNormal;
    hipdnn_test_sdk::utilities::validateToleranceRange<OutputType>(tolerance.atol);
    return tolerance;
}

} // namespace detail

/**
 * @brief Per-element tolerance for a convolution-forward output stored as half, bfloat16 or
 * float, with float accumulation, graded against a float-accumulating reference.
 *
 *   |impl_i - ref_i| <= rtol * |ref_i| + atol
 *   rtol = 2 * eps                       (half, bfloat16)
 *   rtol = 2 * eps + c * sqrt(K) * u32   (float)
 *   atol = (c * sqrt(K) * u32 + lambda * eps) * sigma
 *          + sum over the rounding points of (slope * eps_p * magnitude) + minNormal
 *   sigma = sqrt(K * mean(x^2) * mean(w^2) + K * (K - 1) * mean(x)^2 * mean(w)^2) * outputScale
 *
 * - eps is the machine epsilon of the output type, u32 the unit roundoff of float (2^-24),
 *   minNormal the smallest normal number of the output type.
 * - c is the summation-order coefficient: 1 for half and bfloat16, and for float
 *   DEFAULT_FLOAT_SUMMATION_ORDER, unless `coefficients` sets it.
 * - lambda is 12 for half and 0 for bfloat16 and float.
 * - sigma is the root mean square of an output element for independent inputs: with zero-mean
 *   inputs the partial sums cancel and it grows as sqrt(K); with a nonzero mean on both inputs
 *   the means add up and it grows as K.
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
 * not finite, a mean or outputScale is not finite, a rounding point has a negative or not finite
 * field, or the summation-order coefficient is not finite and positive.
 * @throws std::overflow_error if atol exceeds the largest value of OutputType, or if sigma is not
 * a number (an overflowing product of the statistics against a zero factor).
 */
template <typename OutputType>
AbsRelTolerance calculateConvFpropOutputTolerance(const ConvFpropErrorScale& scale,
                                                  const ConvFpropCoefficients& coefficients = {})
{
    return detail::convOutputTolerance<detail::ConvDirection::FPROP, OutputType>(
        {scale.accumulationLength,
         scale.inputMeanSquare,
         scale.weightMeanSquare,
         scale.inputMean,
         scale.weightMean,
         scale.outputScale,
         &scale.roundingPoints},
        coefficients.summationOrder,
        "calculateConvFpropOutputTolerance");
}

/**
 * @brief Per-element tolerance for a convolution backward-data (dgrad) output dx stored as half,
 * bfloat16 or float, with float accumulation, graded against a float-accumulating reference.
 *
 * The model of calculateConvFpropOutputTolerance() with dy and w in place of x and w, K from
 * convDgradAccumulationLength(), and its own calibrated constants: lambda 12 for half, 6 for
 * bfloat16 and 32 for float; c 1 for half and bfloat16 and DEFAULT_FLOAT_SUMMATION_ORDER for
 * float. The float lambda covers the shortest reductions, where a few products differ by several
 * float steps of a typical output.
 *
 * @tparam OutputType half, bfloat16 or float.
 * @param scale The problem statistics; see ConvDgradErrorScale.
 * @return atol and rtol for an allclose comparison.
 * @throws std::invalid_argument and std::overflow_error as calculateConvFpropOutputTolerance().
 */
template <typename OutputType>
AbsRelTolerance calculateConvDgradOutputTolerance(const ConvDgradErrorScale& scale)
{
    return detail::convOutputTolerance<detail::ConvDirection::DGRAD, OutputType>(
        {scale.accumulationLength,
         scale.dyMeanSquare,
         scale.weightMeanSquare,
         scale.dyMean,
         scale.weightMean,
         scale.outputScale,
         &scale.roundingPoints},
        std::nullopt,
        "calculateConvDgradOutputTolerance");
}

/**
 * @brief Per-element tolerance for a convolution backward-weights (wrw) output dw stored as half,
 * bfloat16 or float, with float accumulation, graded against a float-accumulating reference.
 *
 * The model of calculateConvFpropOutputTolerance() with x and dy in place of x and w, K from
 * convWrwAccumulationLength(), and its own calibrated constants: lambda 36 for half, 24 for
 * bfloat16 and 32 for float; c 1 for half and bfloat16 and DEFAULT_FLOAT_SUMMATION_ORDER for
 * float. The half and bfloat16 lambdas cover reductions whose partial results are kept in the
 * output type.
 *
 * @tparam OutputType half, bfloat16 or float.
 * @param scale The problem statistics; see ConvWrwErrorScale.
 * @return atol and rtol for an allclose comparison.
 * @throws std::invalid_argument and std::overflow_error as calculateConvFpropOutputTolerance().
 */
template <typename OutputType>
AbsRelTolerance calculateConvWrwOutputTolerance(const ConvWrwErrorScale& scale)
{
    return detail::convOutputTolerance<detail::ConvDirection::WRW, OutputType>(
        {scale.accumulationLength,
         scale.inputMeanSquare,
         scale.dyMeanSquare,
         scale.inputMean,
         scale.dyMean,
         scale.outputScale,
         &scale.roundingPoints},
        std::nullopt,
        "calculateConvWrwOutputTolerance");
}

} // namespace hipdnn_test_sdk::utilities::conv
