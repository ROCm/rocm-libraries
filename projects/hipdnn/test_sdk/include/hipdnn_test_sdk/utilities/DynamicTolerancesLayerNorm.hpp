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
 * Error sources:
 * 1. Mean accumulation via Welford: M incremental updates (dominant for mean output)
 * 2. M2 (variance numerator) accumulation: M multiply-adds (dominant for variance)
 * 3. Nonlinear operations: div(M2,M), sqrt, reciprocal (O(u) each)
 * 4. Output normalization: sub(x,mean), mul(invStd), mul(scale), add(bias)
 *
 * The combined error is modeled using computeGamma() for the accumulation stages
 * (2M effective accumulations covering both mean and M2), then propagated through
 * the nonlinear chain using derivative analysis identical to RMSNorm.
 *
 * Key insight: after normalization, |xHat| = |(x - mean) * invStd| is O(1) by
 * construction (zero mean, unit variance). Therefore |y| ~ |scale| + |bias|, and
 * the tolerance scales with |scale|, not with |x|. This cancellation is the same
 * mechanism as in RMSNorm and BatchNorm.
 *
 * Uses the shared computeGamma() helper for the accumulation error growth factor,
 * then propagates that error through the nonlinear chain (div, sqrt, reciprocal).
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
 * - Conservative bound: assumes Welford errors accumulate at worst-case rate
 * - Uses 2M effective accumulations (mean + M2) which may overestimate for
 *   Welford's algorithm whose incremental mean update is more stable than naive sum
 * - NONLINEAR_OPS_UPPER_BOUND=5 models worst-case op count; hardware rsqrt fusion
 *   or multiply reordering may produce fewer rounding errors in practice
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
    const double maxAbsX = std::max(std::abs(xMin), std::abs(xMax));
    const double maxAbsScale = std::max(std::abs(scaleMin), std::abs(scaleMax));
    const double maxAbsBias = std::max(std::abs(biasMin), std::abs(biasMax));

    // Welford accumulation error.
    // Welford's algorithm accumulates both mean and M2 over M elements.
    // Mean: M incremental updates (sub + div + add per step)
    // M2:   M multiply-adds (delta * delta2 + accumulate)
    // Combined effective accumulations: 2M (accounts for both passes).
    //
    // The M2 accumulation has products bounded by maxAbsX^2 (delta and delta2
    // are each bounded by the input range). This is structurally identical
    // to RMSNorm's sum-of-squares, with the same self-product bound.
    auto numberOfAccumulations
        = static_cast<uint64_t>(2) * static_cast<uint64_t>(normalizedElementCount);
    const double maxProduct = maxAbsX * maxAbsX; // Welford delta products
    const double sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    // Use shared computeGamma() for the error growth factor
    const double gamma = computeGamma(numberOfAccumulations, epsilon);
    validateGamma(gamma);

    double accumulatedTolerance = gamma * sumAbsProductBound;

    // Input casting error (if InputType precision > ComputeType precision).
    // Welford squares a single tensor (delta * delta2, both from x), so only one
    // operand is cast per product term -- factor 1, same as RMSNorm.
    accumulatedTolerance += computeInputCastingError<InputType, ComputeType>(sumAbsProductBound, 1);

    // Propagate accumulation error through the nonlinear chain to get output error.
    //
    // The derivation follows the same logic as RMSNorm:
    //   invStd = (M2/M + eps)^(-1/2), so d(invStd)/d(M2) = -1/(2M) * (M2/M+eps)^(-3/2)
    //   |delta_invStd / invStd| <= accTol / (2 * M2)    [since invStd = 1/std, eps << M2/M]
    //                            = accTol / (2 * sumAbsProductBound)  [worst-case M2 = sumAbsBound]
    //
    // After normalization:
    //   y = scale * (x - mean) * invStd + bias
    //   |xHat| = |(x - mean) * invStd| is O(1) by construction
    //   |delta_y| <= maxAbsScale * accTol / (2 * sumAbsProductBound)
    //             =  maxAbsScale * gamma / 2
    //
    // Additional per-op rounding (div(M2,M) + sqrt + recip + mul(x-mean,invStd) + mul(*,scale)):
    //   |delta_y| += NONLINEAR_OPS_UPPER_BOUND * u * maxAbsScale
    //             += maxAbsBias * epsilon
    constexpr double NONLINEAR_OPS_UPPER_BOUND = 5.0;

    double propagatedTolerance = 0.0;

    // When maxAbsX == 0, all inputs are zero. M2 = 0, mean = 0, invStd = 1/sqrt(eps),
    // y = 0*invStd*scale + bias = bias.
    // Accumulation error is zero, the nonlinear-ops term on the xHat*scale chain vanishes,
    // and maxOutputMagnitude reduces to maxAbsBias, leaving only the bias-related terms.
    if(maxAbsX > 0.0)
    {
        propagatedTolerance = (accumulatedTolerance / (2.0 * sumAbsProductBound)) * maxAbsScale;
        propagatedTolerance += NONLINEAR_OPS_UPPER_BOUND * epsilon * maxAbsScale;
    }
    propagatedTolerance += maxAbsBias * epsilon;

    // Output casting error (if OutputType precision < ComputeType precision).
    // Output magnitude bound: |y| <= maxAbsScale + maxAbsBias (since |xHat| ~ O(1)).
    // When maxAbsX == 0, the xHat*scale term is zero, so |y| <= maxAbsBias.
    const double maxOutputMagnitude = (maxAbsX > 0.0 ? maxAbsScale : 0.0) + maxAbsBias;
    propagatedTolerance += computeOutputCastingError<OutputType, ComputeType>(maxOutputMagnitude);

    validateToleranceRange<OutputType>(propagatedTolerance);

    return static_cast<float>(propagatedTolerance);
}

} // namespace hipdnn_test_sdk::utilities::layernorm
