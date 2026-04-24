// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

namespace hipdnn_test_sdk::utilities::batchnorm
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Calculates the expected tolerance for Batch Norm Forward Training y output.
 *
 * BN Training: mean = E[x], var = E[x^2] - E[x]^2, invVar = 1/sqrt(var+eps),
 *              y = scale * (x - mean) * invVar + bias
 *
 * Error sources:
 * 1. Mean accumulation (NHW additions): gamma_NHW * max|x|
 * 2. Variance accumulation (NHW multiply-adds): gamma_NHW * max|x|^2
 * 3. invVariance nonlinear chain (div, sqrt, reciprocal): propagated from variance
 * 4. Normalization per-element ops (sub, 2x mul, bias add)
 *
 * Key insight: after normalization, |xHat| = |(x - mean)/sqrt(var)| is O(1),
 * so the output magnitude is bounded by |scale| + |bias|. The tolerance scales
 * with |scale| (not |x|) because normalization cancels the input magnitude.
 *
 * Uses the shared computeGamma() helper for the accumulation error growth factor.
 *
 * @tparam OutputType  Data type of y output tensor
 * @tparam InputType   Data type of x input tensor
 * @tparam ComputeType Data type for intermediate computation (default: float)
 * @param xMin                Minimum value in input tensor x
 * @param xMax                Maximum value in input tensor x
 * @param scaleMin            Minimum value in scale tensor
 * @param scaleMax            Maximum value in scale tensor
 * @param biasMin             Minimum value in bias tensor
 * @param biasMax             Maximum value in bias tensor
 * @param nElementsPerChannel Number of elements per channel: N * H * W (reduction dim)
 * @param epsilonBn           Batch norm epsilon (added to variance before sqrt)
 * @return Calculated tolerance value as float
 *
 * Known Limitations:
 * - Black-box: does not use kernel implementation details
 * - Assumes |xHat| ~ O(1) after normalization. This holds statistically (E[xHat]=0,
 *   Var(xHat)=1), but individual elements can be large when variance is small relative
 *   to individual deviations (x_i - mean). For near-constant data with tiny variance,
 *   the tolerance may underestimate error on the y output.
 * - The variance bound (3*gamma*maxAbsX^2) is an absolute bound. The two-pass formula
 *   E[x^2] - E[x]^2 can suffer catastrophic cancellation when variance << E[x^2],
 *   making the relative error of variance arbitrarily large. The absolute bound remains
 *   valid and sufficient for test tolerance validation, but users should be aware that
 *   low-variance inputs may exhibit larger relative errors in intermediate quantities.
 * - Nonlinear op count (6 = 3 chain + 3 element-wise) for the normalize path.
 *   The variance division and subtraction rounding (2u terms) are tracked explicitly
 *   in the accumulated tolerance.
 * - Running statistics tolerance not included (Bessel's M/(M-1) amplification for
 *   running variance is deferred to Phase 2).
 * - No backward pass support (only forward training)
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateBatchnormTrainingTolerance(double xMin,
                                          double xMax,
                                          double scaleMin,
                                          double scaleMax,
                                          double biasMin,
                                          double biasMax,
                                          int64_t nElementsPerChannel,
                                          double epsilonBn = 1e-5)
{
    validateComputeType<ComputeType>();

    if(nElementsPerChannel < 1)
    {
        throw std::invalid_argument("nElementsPerChannel must be at least 1.");
    }

    if(epsilonBn <= 0.0)
    {
        throw std::invalid_argument("epsilonBn must be positive.");
    }

    const double maxAbsX = std::max(std::abs(xMin), std::abs(xMax));
    const double maxAbsScale = std::max(std::abs(scaleMin), std::abs(scaleMax));
    const double maxAbsBias = std::max(std::abs(biasMin), std::abs(biasMax));

    const auto nhw = static_cast<uint64_t>(nElementsPerChannel);
    const auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    const double gammaNHW = computeGamma(nhw, epsilon);
    validateGamma(gammaNHW);

    // Sum-of-squares accumulation error for variance path (same structure as RMSNorm).
    // S = sum_{i=0}^{NHW-1} x_i^2 (self-product)
    const double sumAbsProductBound = static_cast<double>(nhw) * maxAbsX * maxAbsX;

    // Variance accumulation error: gamma * S (Higham bound)
    // Plus 2u * S for division-by-NHW and subtraction (E[x^2] - mu^2) rounding ops.
    double accumulatedTolerance
        = gammaNHW * sumAbsProductBound + 2.0 * epsilon * sumAbsProductBound;

    // Input casting error for variance path (single tensor self-product, factor 1).
    accumulatedTolerance += computeInputCastingError<InputType, ComputeType>(sumAbsProductBound, 1);

    // Upper bound on distinct rounding ops in the normalize chain:
    //   div(varAccum,NHW) + sqrt + recip = 3 nonlinear ops
    //   sub(x,mean) + mul(*,invVar) + mul(*,scale) = 3 element-wise ops
    constexpr double NONLINEAR_OPS = 3.0;
    constexpr double ELEM_OPS = 3.0;

    double propagatedTolerance = 0.0;

    // When maxAbsX == 0, all inputs are zero. mean=0, var=0, invVar=1/sqrt(eps_bn),
    // y = scale*0*invVar + bias = bias. All accumulation errors vanish.
    if(maxAbsX > 0.0)
    {
        // Variance path: propagate sum-of-squares error through invVar to output.
        // Same derivation as RMSNorm: accTol / (2*S) * maxAbsScale
        propagatedTolerance = (accumulatedTolerance / (2.0 * sumAbsProductBound)) * maxAbsScale;

        // Mean path: delta_mean ~ gamma * maxAbsX, propagated through scale*invVar.
        // |scale * invVar * delta_mean| ~ maxAbsScale * gamma (since invVar ~ 1/maxAbsX)
        propagatedTolerance += gammaNHW * maxAbsScale;

        // Per-op rounding for nonlinear chain and element-wise normalize ops.
        propagatedTolerance += (NONLINEAR_OPS + ELEM_OPS) * epsilon * maxAbsScale;
    }
    propagatedTolerance += maxAbsBias * epsilon;

    // Output casting error.
    // Conservative bound: instead of assuming |xHat| ~ O(1) cancels input magnitude,
    // use invVarEstimate from max(maxAbsX^2, epsilonBn) to bound output magnitude.
    // This handles the case where variance is small and |xHat| >> 1.
    const double varEst = std::max(maxAbsX * maxAbsX, epsilonBn);
    const double invVarEst = 1.0 / std::sqrt(varEst + epsilonBn);
    const double maxOutputMagnitude
        = (maxAbsX > 0.0 ? maxAbsScale * maxAbsX * invVarEst : 0.0) + maxAbsBias;
    propagatedTolerance += computeOutputCastingError<OutputType, ComputeType>(maxOutputMagnitude);

    validateToleranceRange<OutputType>(propagatedTolerance);

    return static_cast<float>(propagatedTolerance);
}

/**
 * @brief Calculates expected tolerance for saved mean output.
 *
 * mean[c] = (1/NHW) * sum x[n,c,h,w]
 * Error dominated by summation of NHW terms plus division.
 *
 * @tparam OutputType  Data type of mean output tensor
 * @tparam InputType   Data type of x input tensor
 * @tparam ComputeType Data type for intermediate computation (default: float)
 * @param xMin                Minimum value in input tensor x
 * @param xMax                Maximum value in input tensor x
 * @param nElementsPerChannel Number of elements per channel: N * H * W
 * @return Calculated tolerance value as float
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateBatchnormMeanTolerance(double xMin, double xMax, int64_t nElementsPerChannel)
{
    validateComputeType<ComputeType>();

    if(nElementsPerChannel < 1)
    {
        throw std::invalid_argument("nElementsPerChannel must be at least 1.");
    }

    const double maxAbsX = std::max(std::abs(xMin), std::abs(xMax));
    const auto nhw = static_cast<uint64_t>(nElementsPerChannel);
    const auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    const double gammaNHW = computeGamma(nhw, epsilon);
    validateGamma(gammaNHW);

    // mean = (sum x_i) / NHW
    // Summation error: gamma_NHW * max|x| (Higham bound for sum of NHW terms,
    // divided by NHW yields gamma * max|x| since gamma already absorbs the scaling).
    // Division error: u * |mean| <= u * maxAbsX
    double tolerance = gammaNHW * maxAbsX + epsilon * maxAbsX;

    // Input casting (single tensor, factor 1): signalBound = NHW * maxAbsX.
    // After division by NHW, casting contribution is ~ computeEpsilon * maxAbsX.
    const double inputCastError
        = computeInputCastingError<InputType, ComputeType>(static_cast<double>(nhw) * maxAbsX, 1);
    if(maxAbsX > 0.0)
    {
        tolerance += inputCastError / static_cast<double>(nhw);
    }

    // Output casting.
    tolerance += computeOutputCastingError<OutputType, ComputeType>(maxAbsX);

    validateToleranceRange<OutputType>(tolerance);

    return static_cast<float>(tolerance);
}

/**
 * @brief Calculates expected tolerance for saved invVariance output.
 *
 * invVar[c] = 1 / sqrt(var[c] + epsilon_bn)
 * Error from variance accumulation propagated through the nonlinear chain.
 *
 * Uses varEstimate = max(maxAbsX^2, epsilonBn) as the variance floor, which:
 * - Avoids the O(eps_bn^{-3/2}) blowup from the raw (var+eps)^{-3/2} amplification
 * - Handles the small-x regime where eps_bn dominates variance naturally
 * - Provides correct scaling for large maxAbsX (tolerance decreases as 1/maxAbsX)
 *
 * @tparam OutputType  Data type of invVariance output tensor
 * @tparam InputType   Data type of x input tensor (unused — invVar has no per-element
 *                     input casting; retained for API consistency with other BN tolerances)
 * @tparam ComputeType Data type for intermediate computation (default: float)
 * @param xMin                Minimum value in input tensor x
 * @param xMax                Maximum value in input tensor x
 * @param nElementsPerChannel Number of elements per channel: N * H * W
 * @param epsilonBn           Batch norm epsilon (added to variance before sqrt)
 * @return Calculated tolerance value as float
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateBatchnormInvVarianceTolerance(double xMin,
                                             double xMax,
                                             int64_t nElementsPerChannel,
                                             double epsilonBn = 1e-5)
{
    validateComputeType<ComputeType>();

    if(nElementsPerChannel < 1)
    {
        throw std::invalid_argument("nElementsPerChannel must be at least 1.");
    }

    if(epsilonBn <= 0.0)
    {
        throw std::invalid_argument("epsilonBn must be positive.");
    }

    const double maxAbsX = std::max(std::abs(xMin), std::abs(xMax));
    const auto nhw = static_cast<uint64_t>(nElementsPerChannel);
    const auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    const double gammaNHW = computeGamma(nhw, epsilon);
    validateGamma(gammaNHW);

    // Variance error (see plan Section 2, Stage 2, and Section 15 Corrections):
    //   |delta_variance| <= gamma * max|x|^2       (varAccum/NHW error)
    //                     + 2 * maxAbsX * deltaMean (mean^2 error)
    //                     + 2u * max|x|^2           (division + subtraction rounding)
    //   = 3 * gamma * max|x|^2 + 2u * max|x|^2    (since deltaMean ~ gamma * maxAbsX)
    const double deltaVariance
        = 3.0 * gammaNHW * maxAbsX * maxAbsX + 2.0 * epsilon * maxAbsX * maxAbsX;

    // Nonlinear ops: add(epsilon_bn) + sqrt + reciprocal
    constexpr double NONLINEAR_OPS = 3.0;

    // Variance floor: use max(maxAbsX^2, epsilonBn) to avoid blowup when var is small.
    // When maxAbsX > 0 but maxAbsX^2 < epsilonBn, the eps_bn term dominates (var+eps).
    const double varEstimate = std::max(maxAbsX * maxAbsX, epsilonBn);

    // Propagation through invVar = 1/sqrt(var + eps_bn):
    //   d(invVar)/d(var) = -1/2 * (var + eps)^{-3/2}
    //   |delta_invVar| <= deltaVariance / (2 * (varEstimate + epsilonBn)^{3/2})
    //                   + NONLINEAR_OPS * u * invVarEstimate
    const double varPlusEps = varEstimate + epsilonBn;
    const double invVarEstimate = 1.0 / std::sqrt(varPlusEps);
    double tolerance = deltaVariance / (2.0 * varPlusEps * std::sqrt(varPlusEps))
                       + NONLINEAR_OPS * epsilon * invVarEstimate;

    // Output casting: invVar ~ 1/sqrt(varEstimate + eps_bn)
    tolerance += computeOutputCastingError<OutputType, ComputeType>(invVarEstimate);

    validateToleranceRange<OutputType>(tolerance);

    return static_cast<float>(tolerance);
}

} // namespace hipdnn_test_sdk::utilities::batchnorm
