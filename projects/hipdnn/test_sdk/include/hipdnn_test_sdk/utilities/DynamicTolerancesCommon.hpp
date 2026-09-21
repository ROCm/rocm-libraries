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
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>

namespace hipdnn_test_sdk::utilities
{

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Validates that ComputeType is a supported floating-point type.
 *
 * Place `validateComputeType<ComputeType>()` at the top of any tolerance function
 * to enforce the supported type set at compile time.
 */
template <typename ComputeType>
constexpr void validateComputeType()
{
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");
}

/**
 * @brief Computes the input casting error when InputType has higher precision than ComputeType.
 *
 * When inputs are downcast from a higher-precision type (e.g., double -> float), each loaded
 * value incurs a rounding error bounded by epsilon of the compute type. For operations with
 * two distinct input tensors (conv, matmul), each product term has two cast operands (factor 2).
 * For self-product operations (e.g., x^2 in RMSNorm), only one operand is cast (factor 1).
 *
 * @tparam InputType   Data type of the input tensor(s).
 * @tparam ComputeType Data type used for intermediate computation.
 * @param signalBound  Upper bound on sum(|products|) or similar signal magnitude.
 * @param numDistinctInputs Number of distinct input tensors cast per product (1 for self-product,
 *                          2 for two-tensor products like conv/matmul). Default: 2.
 * @return Additional tolerance due to input casting, or 0.0 if no downcasting occurs.
 */
template <typename InputType, typename ComputeType>
double computeInputCastingError(double signalBound, int numDistinctInputs = 2)
{
    auto inputEpsilon = static_cast<double>(std::numeric_limits<InputType>::epsilon());
    auto computeEpsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    if(inputEpsilon < computeEpsilon)
    {
        return static_cast<double>(numDistinctInputs) * signalBound * computeEpsilon;
    }

    return 0.0;
}

/**
 * @brief Computes the output casting error when OutputType has lower precision than ComputeType.
 *
 * When the accumulated result is downcast to a lower-precision output type (e.g., float -> half),
 * the rounding error is bounded by the output epsilon times the maximum output magnitude.
 *
 * @tparam OutputType  Data type of the output tensor.
 * @tparam ComputeType Data type used for intermediate computation.
 * @param maxOutputMagnitude Upper bound on the absolute value of the output.
 * @return Additional tolerance due to output casting, or 0.0 if no downcasting occurs.
 */
template <typename OutputType, typename ComputeType>
double computeOutputCastingError(double maxOutputMagnitude)
{
    auto outputEpsilon = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    auto computeEpsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    if(outputEpsilon > computeEpsilon)
    {
        return maxOutputMagnitude * outputEpsilon;
    }

    return 0.0;
}

/**
 * @brief Throws if the computed tolerance exceeds the maximum representable value of OutputType.
 *
 * @tparam OutputType Data type of the output tensor.
 * @param tolerance   The computed tolerance value.
 * @throws std::overflow_error if tolerance > OutputType::max().
 */
template <typename OutputType>
void validateToleranceRange(double tolerance)
{
    if(tolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }
}

/**
 * @brief Throws if the error growth factor gamma is too large for meaningful computation.
 *
 * When gamma >= maxGamma (default 0.5), the accumulation error exceeds a significant fraction
 * of the signal, indicating the computation is numerically meaningless at this precision.
 *
 * @param gamma    The error growth factor from computeGamma().
 * @param maxGamma Threshold above which to throw (default: 0.5 = 50% signal error).
 * @throws std::overflow_error if gamma >= maxGamma.
 */
inline void validateGamma(double gamma, double maxGamma = 0.5)
{
    if(gamma >= maxGamma)
    {
        throw std::overflow_error(
            "Error growth factor gamma >= " + std::to_string(maxGamma)
            + ": the accumulation error exceeds a significant fraction of the signal. "
              "The computation may be numerically meaningless at this precision and reduction "
              "size.");
    }
}

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
double computeMatrixInfNorm(hipdnn_data_sdk::utilities::ITensor& tensor)
{
    using hipdnn_data_sdk::utilities::iterateAlongDimensions;
    using hipdnn_data_sdk::utilities::TensorView;

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

} // namespace hipdnn_test_sdk::utilities
