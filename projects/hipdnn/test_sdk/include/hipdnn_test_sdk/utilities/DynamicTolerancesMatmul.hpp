// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <stdexcept>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

namespace hipdnn_test_sdk::utilities::matmul
{
using hipdnn_data_sdk::utilities::ITensor;

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
    hipdnn_test_sdk::utilities::validateComputeType<ComputeType>();

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
    const double normA = hipdnn_test_sdk::utilities::computeMatrixInfNorm<InputType>(a);
    const double normB = hipdnn_test_sdk::utilities::computeMatrixInfNorm<InputType>(b);

    // Apply Higham's error bound: max_ij |error_ij| <= γ_k * ||A||_inf * ||B||_inf
    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    const double gamma = hipdnn_test_sdk::utilities::computeGamma(numberOfAccumulations, epsilon);
    hipdnn_test_sdk::utilities::validateGamma(gamma);

    const double signalBound = normA * normB;
    double accumulatedTolerance = gamma * signalBound;

    // Input casting error (2 distinct inputs: A and B)
    accumulatedTolerance
        += hipdnn_test_sdk::utilities::computeInputCastingError<InputType, ComputeType>(
            signalBound);

    // Output casting error
    const double outputCastError
        = hipdnn_test_sdk::utilities::computeOutputCastingError<OutputType, ComputeType>(
            signalBound);

    const double totalTolerance = accumulatedTolerance + outputCastError;

    hipdnn_test_sdk::utilities::validateToleranceRange<OutputType>(totalTolerance);

    return static_cast<float>(totalTolerance);
}

} // namespace hipdnn_test_sdk::utilities::matmul
