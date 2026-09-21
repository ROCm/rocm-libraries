// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

namespace hipdnn_test_sdk::utilities::sdpa
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Calculates the expected tolerance for SDPA forward operations.
 *
 * SDPA: O = softmax(Q @ K^T * scale) @ V
 *
 * Three error accumulation stages:
 *
 * Stage 1 (Q @ K^T): D (head_dim) multiply-accumulate operations per score.
 *   Standard dot-product error bounded by computeGamma(D) * sumAbsProductBound.
 *   The score error is propagated through softmax (L1-Lipschitz-1, upper bound)
 *   and then weighted by V in the P@V stage.
 *
 * Stage 2 (Softmax): Skv accumulations of exp() values plus division.
 *   Bounded using Blanchard et al.: sum |delta_P| <= gamma(Skv) + (Skv+1)*u
 *   since sum(P) = 1.  Softmax is self-bounding (probabilities in [0, 1]).
 *
 * Stage 3 (P @ V): Skv multiply-accumulate operations per output element.
 *   Tight bound from convex combination: sum |P[skv] * V[skv,dv]| <= max(|V|)
 *   since P is a probability distribution.
 *
 * Input casting errors are computed per-stage:
 *   - Q/K cast affects Stage 1 (2 distinct inputs)
 *   - V cast affects Stage 3 (1 distinct input)
 *
 * @tparam OutputType  Data type of O output tensor
 * @tparam InputType   Data type of Q, K, V input tensors
 * @tparam ComputeType Data type for intermediate computation (default: float)
 * @param qMin     Minimum value in Q tensor
 * @param qMax     Maximum value in Q tensor
 * @param kMin     Minimum value in K tensor
 * @param kMax     Maximum value in K tensor
 * @param vMin     Minimum value in V tensor
 * @param vMax     Maximum value in V tensor
 * @param headDim  D - the head dimension (Q@K^T dot product length)
 * @param seqKv    Skv - key/value sequence length (softmax + P@V accumulation length)
 * @param scale    Attention scale factor (nullopt = use default 1/sqrt(D))
 * @return Calculated tolerance value as float
 *
 * Known Limitations:
 * - Black-box: does not use kernel implementation details
 * - Uses max-norm bounds; actual data distribution may allow tighter tolerances
 * - Softmax L1-Lipschitz-1 bound is an upper bound (contraction, not equality)
 * - Does not account for causal mask reducing effective sequence length
 * - Does not account for dropout (when supported)
 * - No backward pass tolerance (SdpaBwd)
 *
 * Future Improvements:
 * - ITensor& approach for tighter bounds using actual Q/K/V norms
 * - Per-position tolerance (accounts for causal mask effective Skv)
 * - SDPA backward pass tolerance (when SdpaBwd is added)
 * - Dropout error contribution (when dropout is supported in CPU ref)
 * - FP8 support (descale/scale tensor integration)
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateSdpaFwdTolerance(double qMin,
                                double qMax,
                                double kMin,
                                double kMax,
                                double vMin,
                                double vMax,
                                int64_t headDim,
                                int64_t seqKv,
                                std::optional<double> scale = std::nullopt)
{
    validateComputeType<ComputeType>();

    if(headDim < 1)
    {
        throw std::invalid_argument("headDim must be at least 1.");
    }
    if(seqKv < 1)
    {
        throw std::invalid_argument("seqKv must be at least 1.");
    }

    // Compute bounds
    const double maxAbsQ = std::max(std::abs(qMin), std::abs(qMax));
    const double maxAbsK = std::max(std::abs(kMin), std::abs(kMax));
    const double maxAbsV = std::max(std::abs(vMin), std::abs(vMax));

    // Resolve scale: use provided value or default 1/sqrt(headDim)
    const double absScale = std::abs(scale.value_or(1.0 / std::sqrt(static_cast<double>(headDim))));

    const auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    // =========================================================================
    // Stage 1: Q @ K^T dot product error
    // =========================================================================
    // dot = sum_{d=0}^{D-1} Q[d] * K[d]
    // S = dot * scale
    const auto d = static_cast<uint64_t>(headDim);
    const double maxProductQk = maxAbsQ * maxAbsK;
    const double sumAbsProductBoundQk = static_cast<double>(d) * maxProductQk;

    const double gammaD = computeGamma(d, epsilon);
    validateGamma(gammaD);

    // Absolute error in score S:
    //   dot product error * |scale| + scale-multiply rounding
    const double scoreError
        = gammaD * sumAbsProductBoundQk * absScale + sumAbsProductBoundQk * absScale * epsilon;

    // =========================================================================
    // Stage 2: Softmax error (Blanchard et al.)
    // =========================================================================
    // Softmax is self-bounding: probabilities in [0, 1], sum = 1.
    // Score error propagates through softmax with L1 Lipschitz constant <= 1
    // (upper bound via Jacobian J_ij = P_i(delta_ij - P_j); not tight).
    //
    // Internal softmax absolute error bound (Blanchard et al.):
    //   |delta_P[skv]| <= P[skv] * (gamma_Skv + (Skv + 1) * u)
    // Summing over all positions (using sum P = 1):
    //   sum |delta_P| <= gamma_Skv + (Skv + 1) * u
    const auto skv = static_cast<uint64_t>(seqKv);

    const double gammaSkv = computeGamma(skv, epsilon);
    validateGamma(gammaSkv);

    const double softmaxError = gammaSkv + (static_cast<double>(skv) + 1.0) * epsilon;

    // =========================================================================
    // Stage 3: P @ V weighted sum error
    // =========================================================================
    // O[dv] = sum_{skv} P[skv] * V[skv, dv]
    // P is a probability distribution (sums to 1), so this is a convex combination:
    //   sum |P[skv] * V[skv,dv]| <= max(|V|)  (tight bound)
    //
    // Reuses gammaSkv since P@V has Skv accumulations (same dimension).
    const double pvAccumError = gammaSkv * maxAbsV;

    // =========================================================================
    // Combined tolerance
    // =========================================================================
    // 1. Q@K^T error propagated through softmax (L1-Lipschitz-1) then P@V:
    //    scoreError * maxAbsV
    // 2. Softmax internal error propagated through P@V:
    //    softmaxError * maxAbsV
    // 3. P@V accumulation error (convex combination tight bound):
    //    pvAccumError
    double totalTolerance = scoreError * maxAbsV // Stage 1 propagated
                            + softmaxError * maxAbsV // Stage 2 propagated
                            + pvAccumError; // Stage 3

    // =========================================================================
    // Input casting error — per stage
    // =========================================================================
    // Stage 1 (Q@K^T): 2 distinct input tensors (Q, K).
    //   Signal bound = sumAbsProductBoundQk * absScale (score magnitude).
    //   Propagated through softmax + P@V → multiply by maxAbsV.
    const double castErrorQkt
        = computeInputCastingError<InputType, ComputeType>(sumAbsProductBoundQk * absScale, 2);
    totalTolerance += castErrorQkt * maxAbsV;

    // Stage 3 (P@V): 1 distinct input tensor (V).
    //   Signal bound = maxAbsV (tight, since P is a probability distribution).
    //   Already in output space.
    const double castErrorPv = computeInputCastingError<InputType, ComputeType>(maxAbsV, 1);
    totalTolerance += castErrorPv;

    // =========================================================================
    // Output casting error
    // =========================================================================
    // Output magnitude bound: O is a convex combination of V, so |O| <= max(|V|).
    const double maxOutputMagnitude = maxAbsV;
    totalTolerance += computeOutputCastingError<OutputType, ComputeType>(maxOutputMagnitude);

    // =========================================================================
    // Validate and return
    // =========================================================================
    validateToleranceRange<OutputType>(totalTolerance);

    return static_cast<float>(totalTolerance);
}

} // namespace hipdnn_test_sdk::utilities::sdpa
