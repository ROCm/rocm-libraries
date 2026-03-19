// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

namespace hipdnn_test_sdk::utilities::rmsnorm
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Calculates the expected tolerance for RMSNorm forward operations.
 *
 * RMSNorm: y[n,c,h,w] = x[n,c,h,w] / RMS(x) * scale[c] [+ bias[c]]
 * where RMS is computed across channels for each (batch, spatial) position.
 *
 * Error sources:
 * 1. Sum of squares accumulation: C multiply-adds (dominant)
 * 2. Nonlinear operations: div, sqrt, reciprocal (O(u) each)
 * 3. Output multiply by scale and optional bias add
 *
 * Uses the shared computeGamma() helper for the accumulation error growth factor,
 * then propagates that error through the nonlinear chain (div, sqrt, reciprocal).
 *
 * @tparam OutputType  Data type of y output tensor
 * @tparam InputType   Data type of x input tensor
 * @tparam ComputeType Data type for intermediate computation (default: float)
 * @param xMin       Minimum value in input tensor x
 * @param xMax       Maximum value in input tensor x
 * @param scaleMin   Minimum value in scale tensor
 * @param scaleMax   Maximum value in scale tensor
 * @param nChannels  Number of channels C (the reduction dimension)
 * @param biasMin    Minimum value in bias tensor (0 if no bias)
 * @param biasMax    Maximum value in bias tensor (0 if no bias)
 * @return Calculated tolerance value as float
 *
 * Known Limitations:
 * - Black-box: does not use kernel implementation details
 * - Conservative bound from nonlinear propagation (may overestimate for sparse
 *   activations where most channels are near zero but max(|x|) is large)
 * - Assumes RMS ~ max(|x|); sparse-spike workloads get looser tolerances
 * - NONLINEAR_OPS_UPPER_BOUND=5 models worst-case op count; hardware rsqrt fusion
 *   or multiply reordering may produce fewer rounding errors in practice
 * - No backward pass support (only forward)
 * - No invRms-specific tolerance (training mode)
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateRMSNormFwdTolerance(double xMin,
                                   double xMax,
                                   double scaleMin,
                                   double scaleMax,
                                   int64_t nChannels,
                                   double biasMin = 0.0,
                                   double biasMax = 0.0)
{
    validateComputeType<ComputeType>();

    if(nChannels < 1)
    {
        throw std::invalid_argument("nChannels must be at least 1.");
    }

    // Compute bounds
    const double maxAbsX = std::max(std::abs(xMin), std::abs(xMax));
    const double maxAbsScale = std::max(std::abs(scaleMin), std::abs(scaleMax));
    const double maxAbsBias = std::max(std::abs(biasMin), std::abs(biasMax));

    // Sum of squares accumulation error
    // S = sum_{c=0}^{C-1} x_c^2 (self-product)
    auto numberOfAccumulations = static_cast<uint64_t>(nChannels);
    const double maxProduct = maxAbsX * maxAbsX; // self-product
    const double sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    // Use shared computeGamma() for the error growth factor
    const double gamma = computeGamma(numberOfAccumulations, epsilon);
    validateGamma(gamma);

    double accumulatedTolerance = gamma * sumAbsProductBound;

    // Input casting error (if InputType precision > ComputeType precision).
    // Unlike conv (which casts two distinct tensors, factor 2), RMSNorm squares a single
    // tensor, so only one operand is cast per product term -- factor 1.
    // This error is added to accumulatedTolerance before nonlinear propagation, so it is
    // propagated through the same div/sqrt/recip chain as the accumulation error.
    accumulatedTolerance += computeInputCastingError<InputType, ComputeType>(sumAbsProductBound, 1);

    // Propagate accumulation error through the nonlinear chain to get output error.
    //
    // Derivation of the accTol / (2 * sumAbsProductBound) term:
    //   invRms = (S/C + eps)^(-1/2),  so  d(invRms)/dS = -1/(2C) * (S/C+eps)^(-3/2)
    //   |delta_invRms| = |d(invRms)/dS| * |delta_S| = accTol / (2C * RMS^3)
    //   |delta_invRms / invRms| = accTol / (2 * S)    [since invRms = 1/RMS, eps << S/C]
    //                           = accTol / (2 * C * mean(x^2))
    //   Using the worst-case bound S = sumAbsProductBound = C * maxAbsX^2:
    //     |delta_invRms / invRms| <= accTol / (2 * C * maxAbsX^2) = gamma / 2
    //   Absolute output error: y = x * invRms * scale, so
    //     |delta_y| = |x * invRms * scale| * |delta_invRms / invRms|
    //   Under the worst-case bound invRms = 1/RMS approx 1/maxAbsX:
    //     |x * invRms| <= maxAbsX * (1/maxAbsX) = 1
    //     |delta_y| <= maxAbsScale * accTol / (2 * sumAbsProductBound)
    //               =  maxAbsScale * gamma / 2
    //
    // Additional per-op rounding (div, sqrt, recip, 2 muls) and bias:
    //   Since |y| <= maxAbsScale (under the same invRms approx 1/maxAbsX assumption):
    //   |delta_y| += NONLINEAR_OPS_UPPER_BOUND * u * maxAbsScale
    //             += maxAbsBias * epsilon
    //
    // Upper bound on distinct rounding ops: div(S,C) + sqrt + recip + mul(x,invRms) + mul(*,scale).
    // Hardware may fuse some (e.g. rsqrt), so 5 is a modelling upper bound, not a literal count.
    constexpr double NONLINEAR_OPS_UPPER_BOUND = 5.0;

    double propagatedTolerance = 0.0;

    // When maxAbsX == 0, all inputs are zero. S = 0, invRms = 1/sqrt(eps),
    // y = 0*invRms*scale + bias = bias.
    // Accumulation error is zero, the nonlinear-ops term on the x*invRms*scale chain vanishes,
    // and maxOutputMagnitude reduces to maxAbsBias, leaving only the bias-related terms.
    if(maxAbsX > 0.0)
    {
        propagatedTolerance = (accumulatedTolerance / (2.0 * sumAbsProductBound)) * maxAbsScale;
        propagatedTolerance += NONLINEAR_OPS_UPPER_BOUND * epsilon * maxAbsScale;
    }
    propagatedTolerance += maxAbsBias * epsilon;

    // Output casting error (if OutputType precision < ComputeType precision).
    // Output magnitude bound: under invRms approx 1/maxAbsX, |y| <= maxAbsScale + maxAbsBias.
    // When maxAbsX == 0, the x*invRms*scale term is zero, so |y| <= maxAbsBias.
    const double maxOutputMagnitude = (maxAbsX > 0.0 ? maxAbsScale : 0.0) + maxAbsBias;
    propagatedTolerance += computeOutputCastingError<OutputType, ComputeType>(maxOutputMagnitude);

    validateToleranceRange<OutputType>(propagatedTolerance);

    return static_cast<float>(propagatedTolerance);
}

} // namespace hipdnn_test_sdk::utilities::rmsnorm
