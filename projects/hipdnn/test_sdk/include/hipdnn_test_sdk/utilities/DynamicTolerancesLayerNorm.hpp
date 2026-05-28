// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

namespace hipdnn_test_sdk::utilities::layernorm
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Calculates the expected tolerance for LayerNorm forward operations.
 *
 * LayerNorm: For each batch position, compute mean and variance over the last
 * normalizedDimCount dimensions using Welford's online algorithm, then normalize:
 *   y[b,i] = scale[i] * (x[b,i] - mean_b) * invStd_b + bias[i]
 *
 * Two independent error paths propagate into the output y:
 *
 * Path A (variance): M2 accumulation error → invStd error → output error.
 *   Modeled using computeGamma(2M) for the M2 accumulation (2M effective
 *   accumulations: M multiply-adds for M2, M incremental updates for mean
 *   that feed the delta values).  Propagated through the nonlinear chain
 *   (div, sqrt, reciprocal) using the same derivative analysis as RMSNorm.
 *   Contribution: gammaVar(2M) / 2 * maxAbsScale.
 *
 * Path B (mean): mean error propagates directly into (x - mean) * invStd.
 *   |delta_mean| ~ gamma(M) * maxAbsX from M Welford incremental updates.
 *   Under the invStd ~ 1/maxAbsX assumption (same as Path A), the output
 *   contribution is gamma(M) * maxAbsScale.  This path does not exist in
 *   RMSNorm (which has no mean subtraction).
 *
 * Key insight: after normalization, |xHat| = |(x - mean) * invStd| is O(1)
 * (the RMS of xHat is exactly 1).  Therefore |y| ~ |scale| + |bias|, and
 * the tolerance scales with |scale|, not with |x|.
 *
 * @tparam OutputType  Data type of y output tensor
 * @tparam InputType   Data type of x input tensor
 * @tparam ComputeType Data type for intermediate computation (default: float)
 * @param xMin                    Minimum value in input tensor x
 * @param xMax                    Maximum value in input tensor x
 * @param scaleMin                Minimum value in scale tensor
 * @param scaleMax                Maximum value in scale tensor
 * @param normalizedElementCount  M = product of normalized dimensions (reduction dim)
 * @param biasMin                 Minimum value in bias tensor (0 if no bias)
 * @param biasMax                 Maximum value in bias tensor (0 if no bias)
 * @return Calculated tolerance value as float
 *
 * Known Limitations:
 * - Black-box: does not use kernel implementation details
 * - Mean path uses invStd ~ 1/maxAbsX assumption; for data with small variance
 *   relative to magnitude (e.g. x in [100, 100.1]), actual invStd >> 1/maxAbsX,
 *   and the mean error contribution is underestimated
 * - |xHat| ~ O(1) is the RMS value; individual elements can reach O(sqrt(M)),
 *   so tail elements may exceed the tolerance (rare for random data)
 * - NONLINEAR_OPS_UPPER_BOUND=5 models worst-case op count; hardware rsqrt fusion
 *   or multiply reordering may produce fewer rounding errors in practice
 * - GPU parallel Welford (block-wise reduction + tree merge) has different error
 *   characteristics than the sequential reference; this tolerance is designed for
 *   CPU-reference vs CPU-reference comparison only
 * - No backward pass support (only forward)
 * - No mean/invVariance-specific tolerance functions (training mode outputs)
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateLayernormFpropTolerance(double xMin,
                                       double xMax,
                                       double scaleMin,
                                       double scaleMax,
                                       int64_t normalizedElementCount,
                                       double biasMin = 0.0,
                                       double biasMax = 0.0)
{
    validateComputeType<ComputeType>();

    if(normalizedElementCount < 1)
    {
        throw std::invalid_argument("normalizedElementCount must be at least 1.");
    }

    // Compute bounds
    const double maxAbsScale = std::max(std::abs(scaleMin), std::abs(scaleMax));
    const double maxAbsBias = std::max(std::abs(biasMin), std::abs(biasMax));

    // Input range: both x and mean lie in [xMin, xMax], so Welford's
    // delta = x - mean has |delta| <= xMax - xMin.  For constant input
    // (range = 0), all deltas are zero and M2 = 0.
    const double range = std::max(xMax - xMin, 0.0);

    // =================================================================
    // Path A: Variance accumulation error (M2 → invStd → y)
    // =================================================================
    // Welford's M2 accumulates M products delta_i * delta2_i, each
    // bounded by range^2.  We model this as 2M effective accumulations
    // (M for M2 multiply-adds + M for mean incremental updates that
    // feed into the delta values).
    auto varianceAccumulations
        = static_cast<uint64_t>(2) * static_cast<uint64_t>(normalizedElementCount);
    const double maxProduct = range * range;
    const double sumAbsProductBound = static_cast<double>(varianceAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    // Variance path: computeGamma for 2M effective accumulations
    const double gammaVar = computeGamma(varianceAccumulations, epsilon);
    validateGamma(gammaVar);

    double varianceAccumTolerance = gammaVar * sumAbsProductBound;

    // Input casting error for the variance path (if InputType precision > ComputeType).
    // Welford squares a single tensor (delta * delta2, both from x), so only one
    // operand is cast per product term -- factor 1, same as RMSNorm.
    varianceAccumTolerance
        += computeInputCastingError<InputType, ComputeType>(sumAbsProductBound, 1);

    // =================================================================
    // Path B: Mean error propagation (delta_mean → y)
    // =================================================================
    // The mean error propagates into the output as:
    //   delta_y_from_mean = -scale * delta_mean * invStd
    //
    // |delta_mean| ~ gamma(M) * maxAbsX  (M incremental Welford updates,
    //   each contributing O(u * maxAbsX) from sub + div + add).
    //
    // Using the same invStd ~ 1/maxAbsX assumption as the variance path
    // (RMS-like bound: std ~ maxAbsX for well-spread data):
    //   |delta_y_from_mean| ~ gamma(M) * maxAbsX * (1/maxAbsX) * maxAbsScale
    //                       = gamma(M) * maxAbsScale
    //
    // Note: the ratio maxAbsX/std is data-dependent (~ 1.7 for uniform [-a,a],
    // ~ 3.5 for uniform [0,a]).  The invStd ~ 1/maxAbsX assumption gives an
    // O(1) ratio, which is the same modelling choice the variance path makes.
    auto meanAccumulations = static_cast<uint64_t>(normalizedElementCount);
    const double gammaMean = computeGamma(meanAccumulations, epsilon);
    // gammaMean < gammaVar always (M < 2M, computeGamma is monotonic),
    // so if gammaVar passed validateGamma, gammaMean is also valid.

    // =================================================================
    // Combined propagation
    // =================================================================
    // Variance path: accTol / (2 * sumAbsProductBound) * maxAbsScale = gammaVar/2 * maxAbsScale
    //   (the sumAbsProductBound cancels in numerator and denominator)
    // Mean path:     gammaMean * maxAbsScale
    // Nonlinear ops: NONLINEAR_OPS_UPPER_BOUND * u * maxAbsScale
    // Bias:          u * maxAbsBias
    constexpr double NONLINEAR_OPS_UPPER_BOUND = 5.0;

    double propagatedTolerance = 0.0;

    // When range == 0 (constant input), all deltas are zero: M2 = 0, mean = x_const,
    // xHat = 0, y = bias.  Both accumulation paths produce zero error, the nonlinear-ops
    // term on the xHat*scale chain vanishes, leaving only bias-related terms.
    if(range > 0.0)
    {
        // Variance path (sumAbsProductBound cancels)
        propagatedTolerance = (varianceAccumTolerance / (2.0 * sumAbsProductBound)) * maxAbsScale;
        // Mean path
        propagatedTolerance += gammaMean * maxAbsScale;
        // Nonlinear ops rounding
        propagatedTolerance += NONLINEAR_OPS_UPPER_BOUND * epsilon * maxAbsScale;
    }
    propagatedTolerance += maxAbsBias * epsilon;

    // Output casting error (if OutputType precision < ComputeType precision).
    // Output magnitude bound: |y| <= maxAbsScale + maxAbsBias (since |xHat| ~ O(1)).
    // When range == 0, the xHat*scale term is zero, so |y| <= maxAbsBias.
    const double maxOutputMagnitude = (range > 0.0 ? maxAbsScale : 0.0) + maxAbsBias;
    propagatedTolerance += computeOutputCastingError<OutputType, ComputeType>(maxOutputMagnitude);

    validateToleranceRange<OutputType>(propagatedTolerance);

    return static_cast<float>(propagatedTolerance);
}

} // namespace hipdnn_test_sdk::utilities::layernorm
