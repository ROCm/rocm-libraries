// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/UnaryOperationFunctors.hpp>
#include <type_traits>

namespace hipdnn_test_sdk::utilities::pointwise
{

// Binary arithmetic operations with explicit ComputeType and OutputType
// ComputeType: The type used for intermediate calculations
// OutputType: The type returned from the operation

template <typename ComputeType, typename OutputType = ComputeType>
struct Add
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto result = static_cast<ComputeType>(x0) + static_cast<ComputeType>(x1);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct Subtract
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto result = static_cast<ComputeType>(x0) - static_cast<ComputeType>(x1);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct Multiply
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto result = static_cast<ComputeType>(x0) * static_cast<ComputeType>(x1);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct Max
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<OutputType>(
            std::max(static_cast<ComputeType>(x0), static_cast<ComputeType>(x1)));
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct Min
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<OutputType>(
            std::min(static_cast<ComputeType>(x0), static_cast<ComputeType>(x1)));
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct Divide
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto result = static_cast<ComputeType>(x0) / static_cast<ComputeType>(x1);
        return static_cast<OutputType>(result);
    }
};

// ADD_SQUARE(x, y) = x + y^2
template <typename ComputeType, typename OutputType = ComputeType>
struct AddSquare
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto y = static_cast<ComputeType>(x1);
        auto result = static_cast<ComputeType>(x0) + y * y;
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType>
struct CompareGreater
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<ComputeType>(x0) > static_cast<ComputeType>(x1);
    }
};

template <typename ComputeType>
struct CompareEqual
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<ComputeType>(x0) == static_cast<ComputeType>(x1);
    }
};

template <typename ComputeType>
struct CompareNotEqual
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<ComputeType>(x0) != static_cast<ComputeType>(x1);
    }
};

template <typename ComputeType>
struct CompareGreaterEqual
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<ComputeType>(x0) >= static_cast<ComputeType>(x1);
    }
};

template <typename ComputeType>
struct CompareLess
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<ComputeType>(x0) < static_cast<ComputeType>(x1);
    }
};

template <typename ComputeType>
struct CompareLessEqual
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return static_cast<ComputeType>(x0) <= static_cast<ComputeType>(x1);
    }
};

struct LogicalAnd
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return isTrue(x0) && isTrue(x1);
    }
};

struct LogicalOr
{
    template <typename X0, typename X1>
    bool operator()(const X0& x0, const X1& x1) const
    {
        return isTrue(x0) || isTrue(x1);
    }
};

// Backward activation operations: dx = dy * local_gradient
// Takes upstream gradient dy and forward input x.

template <typename ComputeType, typename OutputType = ComputeType>
struct ReluBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);
        auto localGradient = (xCompute > ComputeType{0}) ? ComputeType{1} : ComputeType{0};
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

// CLAMP is f(x) = min(max(x, lowerClip), upperClip)
// Thus, we have:
// f'(x) = 1, if x > lowerClip && x < upperClip
// f'(x) = 0, if x < lowerClip || x > upperClip
// The derivatives at the bounds are technically undefined, but we follow convention
// of treating the upper bound as inclusive and the lower bound as exclusive.
// e.g. https://github.com/ROCm/rocm-libraries/blob/develop/projects/miopen/src/kernels/bnorm_spatial_activation_functions.h#L75

// Leaky ReLU is f(x) = x, if x > 0; f(x) = lowerSlope * x, otherwise
// Thus, we have:
// f'(x) = 1, if x > 0
// f'(x) = lowerSlope, if x < 0
// Again, the derivative at 0 is technically undefined, but we follow convention of treating f'(0) = lowerSlope.
template <typename ComputeType, typename OutputType = ComputeType>
struct ParameterizedReluBackward
{
    ComputeType lowerClip;
    ComputeType upperClip;
    ComputeType lowerSlope;

    ParameterizedReluBackward(ComputeType lowerClipVal,
                              ComputeType upperClipVal,
                              ComputeType lowerSlopeVal)
        : lowerClip(lowerClipVal)
        , upperClip(upperClipVal)
        , lowerSlope(lowerSlopeVal)
    {
    }

    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        ComputeType localGradient;
        if(xCompute <= lowerClip)
        {
            localGradient = lowerSlope;
        }
        else if(xCompute > upperClip)
        {
            localGradient = ComputeType{0};
        }
        else
        {
            localGradient = ComputeType{1};
        }

        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct SigmoidBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        ComputeType sigmoidVal = ComputeType{1} / (ComputeType{1} + exp(-xCompute));
        auto localGradient = sigmoidVal * (ComputeType{1} - sigmoidVal);
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct TanhBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::tanh;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        ComputeType tanhVal = tanh(xCompute);
        auto localGradient = ComputeType{1} - (tanhVal * tanhVal);
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct GeluBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::erf;
        using hipdnn_data_sdk::types::exp;
        using hipdnn_data_sdk::types::sqrt;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        // d/dx [x * Phi(x)] = Phi(x) + x * phi(x), with Phi the standard normal CDF and phi its PDF
        static const ComputeType s_kSqrt2 = sqrt(ComputeType{2});
        static const ComputeType s_kInvSqrt2Pi
            = ComputeType{1} / sqrt(ComputeType{2} * ComputeType{M_PI});
        auto cdf = ComputeType{0.5} * (ComputeType{1} + erf(xCompute / s_kSqrt2));
        auto pdf = exp(ComputeType{-0.5} * xCompute * xCompute) * s_kInvSqrt2Pi;
        return static_cast<OutputType>(dyCompute * (cdf + xCompute * pdf));
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct GeluApproxTanhBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::sqrt;
        using hipdnn_data_sdk::types::tanh;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        // Matches GeluApproxTanhForward: 0.5 * x * (1 + tanh(u)), u = sqrt(2/pi) * (x + c * x^3)
        static const ComputeType s_kSqrt2OverPi = sqrt(ComputeType{2} / ComputeType{M_PI});
        static const auto s_kCoeff = ComputeType{0.044715};
        auto inner = s_kSqrt2OverPi * (xCompute + s_kCoeff * xCompute * xCompute * xCompute);
        auto tanhVal = tanh(inner);
        auto innerGradient
            = s_kSqrt2OverPi * (ComputeType{1} + ComputeType{3} * s_kCoeff * xCompute * xCompute);
        auto localGradient
            = ComputeType{0.5} * (ComputeType{1} + tanhVal)
              + ComputeType{0.5} * xCompute * (ComputeType{1} - tanhVal * tanhVal) * innerGradient;
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct SwishBackward
{
    ComputeType beta;

    explicit SwishBackward(ComputeType betaVal = ComputeType{1})
        : beta(betaVal)
    {
    }

    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        // d/dx [x * sigmoid(beta * x)] = s + beta * x * s * (1 - s), with s = sigmoid(beta * x)
        ComputeType sigmoidVal = ComputeType{1} / (ComputeType{1} + exp(-beta * xCompute));
        auto localGradient
            = sigmoidVal + beta * xCompute * sigmoidVal * (ComputeType{1} - sigmoidVal);
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct EluBackward
{
    ComputeType alpha;

    explicit EluBackward(ComputeType alphaVal = ComputeType{1})
        : alpha(alphaVal)
    {
    }

    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        // d/dx ELU(x) = 1 for x > 0, alpha * exp(x) otherwise
        auto localGradient = xCompute > ComputeType{0} ? ComputeType{1} : alpha * exp(xCompute);
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType, typename OutputType = ComputeType>
struct SoftplusBackward
{
    ComputeType beta;

    explicit SoftplusBackward(ComputeType betaVal = ComputeType{1})
        : beta(betaVal)
    {
    }

    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        // d/dx [log(1 + exp(beta * x)) / beta] = sigmoid(beta * x)
        auto localGradient = ComputeType{1} / (ComputeType{1} + exp(-beta * xCompute));
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

} // namespace hipdnn_test_sdk::utilities::pointwise
