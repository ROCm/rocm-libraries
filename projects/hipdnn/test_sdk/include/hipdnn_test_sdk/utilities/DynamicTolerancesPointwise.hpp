// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <stdexcept>
#include <type_traits>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>

namespace hipdnn_test_sdk::utilities::pointwise
{

/**
 * @brief Classification of pointwise operations by error characteristics.
 *
 * Each class has a distinct error multiplier C that reflects how many ULPs of
 * divergence are expected between reference and GPU implementations.
 * Backward ops get higher C values than forward due to chain rule amplification.
 *
 * Five categories (A-E), with D and E split into forward/backward:
 * - Bitwise (A): mathematically exact, errors from casts/canonicalization only
 * - Linear (B): single IEEE-rounded ops, error scales with operand magnitude
 * - Rational (C): Newton-Raphson-based ops, input conditioning sensitive
 * - Transcendental (D): polynomial/table approximations, implementation-dependent
 * - Composite (E): compositions of transcendentals, error from worst component
 */
enum class PointwiseErrorClass : uint8_t
{
    BITWISE, ///< Comparisons, select, relu, abs, neg, identity, floor, ceil
    LINEAR, ///< add, sub, mul, add_square
    RATIONAL, ///< div, reciprocal, sqrt, rsqrt
    TRANSCENDENTAL_FWD, ///< exp, log, sin, tan, tanh_fwd, sigmoid_fwd, erf, elu_fwd
    TRANSCENDENTAL_BWD, ///< tanh_bwd, sigmoid_bwd, elu_bwd
    COMPOSITE_FWD, ///< gelu_fwd, gelu_approx_tanh_fwd, softplus_fwd, swish_fwd
    COMPOSITE_BWD ///< gelu_bwd, gelu_approx_tanh_bwd, softplus_bwd, swish_bwd
};

namespace detail
{

/// Returns the error multiplier C for a given error class and precision.
/// C_high is for float/double compute; C_low is for half/bf16 compute.
/// C_low ≈ 2 × C_high because low-precision polynomial evaluations
/// accumulate proportionally more relative error per coefficient.
template <typename ComputeType>
constexpr double getErrorMultiplier(PointwiseErrorClass errorClass)
{
    constexpr bool IS_HIGH_PRECISION
        = std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>;

    switch(errorClass)
    {
    case PointwiseErrorClass::BITWISE:
        return IS_HIGH_PRECISION ? 1.0 : 2.0;
    case PointwiseErrorClass::LINEAR:
        return IS_HIGH_PRECISION ? 2.0 : 4.0;
    case PointwiseErrorClass::RATIONAL:
        return IS_HIGH_PRECISION ? 4.0 : 8.0;
    case PointwiseErrorClass::TRANSCENDENTAL_FWD:
        return IS_HIGH_PRECISION ? 8.0 : 16.0;
    case PointwiseErrorClass::TRANSCENDENTAL_BWD:
        return IS_HIGH_PRECISION ? 12.0 : 24.0;
    case PointwiseErrorClass::COMPOSITE_FWD:
        return IS_HIGH_PRECISION ? 16.0 : 32.0;
    case PointwiseErrorClass::COMPOSITE_BWD:
        return IS_HIGH_PRECISION ? 24.0 : 48.0;
    default:
        throw std::logic_error("Unhandled PointwiseErrorClass — add it to getErrorMultiplier");
    }
}

} // namespace detail

/**
 * @brief Calculates the expected tolerance for element-wise pointwise operations.
 *
 * Error model:
 * @code
 *   tolerance = C[class][precision] * epsilon_compute * scale
 *             + (inputDowncast  ? epsilon_compute * scale : 0)
 *             + (outputDowncast ? epsilon_output  * scale : 0)
 * @endcode
 *
 * The @p scale parameter is typically @c max(|input|) for unbounded-output ops
 * (exp, log, add, gelu), but should be @c 1.0 for bounded-output ops
 * (sigmoid -> [0,1], tanh -> [-1,1], erf -> [-1,1]). The caller decides
 * based on the operation being tested.
 *
 * @tparam OutputType  Data type of the output tensor.
 * @tparam InputType   Data type of the input tensor(s).
 * @tparam ComputeType Data type used for intermediate computation (default: float).
 * @param scale        Error scaling factor. Use max(|input|) for unbounded ops,
 *                     1.0 for bounded-output transcendentals (sigmoid, tanh, erf).
 * @param errorClass   Classification of the pointwise operation.
 * @return Calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculatePointwiseTolerance(double scale, PointwiseErrorClass errorClass)
{
    hipdnn_test_sdk::utilities::validateComputeType<ComputeType>();

    auto epsilonCompute = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    // Core error: C * epsilon_compute * scale
    const double c = detail::getErrorMultiplier<ComputeType>(errorClass);
    double tolerance = c * epsilonCompute * scale;

    // Input casting error: 1 distinct input for pointwise ops
    tolerance
        += hipdnn_test_sdk::utilities::computeInputCastingError<InputType, ComputeType>(scale, 1);

    // Output casting error
    tolerance
        += hipdnn_test_sdk::utilities::computeOutputCastingError<OutputType, ComputeType>(scale);

    return static_cast<float>(tolerance);
}

} // namespace hipdnn_test_sdk::utilities::pointwise
