// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <limits>
#include <type_traits>

namespace hipdnn_test_sdk::utilities::pointwise
{

// Unary operations with explicit ComputeType and OutputType
// ComputeType: The type used for intermediate calculations
// OutputType: The type returned from the operation

template <typename ComputeType = float, typename OutputType = ComputeType>
struct ReluForward
{
    ComputeType lowerClip;
    ComputeType upperClip;
    ComputeType lowerSlope;

    ReluForward(ComputeType lowerClipVal = ComputeType{0},
                ComputeType upperClipVal = std::numeric_limits<ComputeType>::max(),
                ComputeType lowerSlopeVal = ComputeType{0})
        : lowerClip(lowerClipVal)
        , upperClip(upperClipVal)
        , lowerSlope(lowerSlopeVal)
    {
    }

    template <typename X>
    OutputType operator()(const X& x) const
    {
        auto xCompute = static_cast<ComputeType>(x);

        ComputeType result;
        if(xCompute <= lowerClip)
        {
            result = (lowerSlope * (xCompute - lowerClip)) + lowerClip;
        }
        else if(xCompute >= upperClip)
        {
            result = upperClip;
        }
        else
        {
            result = xCompute;
        }
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct SigmoidForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto xCompute = static_cast<ComputeType>(x);
        auto result = ComputeType{1} / (ComputeType{1} + exp(-xCompute));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct TanhForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::tanh;
        auto xCompute = static_cast<ComputeType>(x);
        auto result = tanh(xCompute);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct Identity
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        return static_cast<OutputType>(static_cast<ComputeType>(x));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct AbsoluteValue
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::abs;
        auto result = abs(static_cast<ComputeType>(x));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct Negation
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        auto result = -static_cast<ComputeType>(x);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct GeluForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::erf;
        using hipdnn_data_sdk::types::sqrt;
        auto xCompute = static_cast<ComputeType>(x);
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        static const ComputeType s_kSqrt2 = sqrt(ComputeType{2});
        auto result = ComputeType{0.5} * xCompute * (ComputeType{1} + erf(xCompute / s_kSqrt2));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct GeluApproxTanhForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::sqrt;
        using hipdnn_data_sdk::types::tanh;
        auto xCompute = static_cast<ComputeType>(x);
        // GELU_tanh(x) ~= 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        static const ComputeType s_kSqrt2OverPi = sqrt(ComputeType{2} / ComputeType{3.14159265});
        static const auto s_kCoeff = ComputeType{0.044715};
        auto inner = s_kSqrt2OverPi * (xCompute + s_kCoeff * xCompute * xCompute * xCompute);
        auto result = ComputeType{0.5} * xCompute * (ComputeType{1} + tanh(inner));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct SwishForward
{
    ComputeType beta;

    explicit SwishForward(ComputeType betaVal = ComputeType{1})
        : beta(betaVal)
    {
    }

    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto xCompute = static_cast<ComputeType>(x);
        // Swish(x) = x * sigmoid(beta * x) = x / (1 + exp(-beta * x))
        auto result = xCompute / (ComputeType{1} + exp(-beta * xCompute));
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct ExpForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        return static_cast<OutputType>(exp(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct LogForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::log;
        return static_cast<OutputType>(log(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct SqrtForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::sqrt;
        return static_cast<OutputType>(sqrt(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct RsqrtForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::sqrt;
        return static_cast<OutputType>(ComputeType{1} / sqrt(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct ReciprocalForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        return static_cast<OutputType>(ComputeType{1} / static_cast<ComputeType>(x));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct SinForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::sin;
        return static_cast<OutputType>(sin(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct TanForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::tan;
        return static_cast<OutputType>(tan(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct ErfForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::erf;
        return static_cast<OutputType>(erf(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct CeilForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::ceil;
        return static_cast<OutputType>(ceil(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct FloorForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::floor;
        return static_cast<OutputType>(floor(static_cast<ComputeType>(x)));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct EluForward
{
    ComputeType alpha;

    explicit EluForward(ComputeType alphaVal = ComputeType{1})
        : alpha(alphaVal)
    {
    }

    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto xCompute = static_cast<ComputeType>(x);
        // ELU(x) = x for x > 0, alpha * (exp(x) - 1) otherwise
        auto result
            = xCompute > ComputeType{0} ? xCompute : alpha * (exp(xCompute) - ComputeType{1});
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct SoftplusForward
{
    ComputeType beta;

    explicit SoftplusForward(ComputeType betaVal = ComputeType{1})
        : beta(betaVal)
    {
    }

    template <typename X>
    OutputType operator()(const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        using hipdnn_data_sdk::types::log;
        auto xCompute = static_cast<ComputeType>(x);
        auto scaled = beta * xCompute;
        // Softplus(x) = log(1 + exp(beta * x)) / beta; revert to x when exp would overflow
        // and the result is indistinguishable from the linear regime.
        // The threshold of 20 is chosen as a safe value to avoid overflow in the exponential function
        // (The same value is used in PyTorch's implementation of Softplus)
        ComputeType result
            = scaled > ComputeType{20} ? xCompute : log(ComputeType{1} + exp(scaled)) / beta;
        return static_cast<OutputType>(result);
    }
};

// Truthiness for logical operations; half/bfloat16 have no conversion to bool.
template <typename X>
bool isTrue(const X& x)
{
    if constexpr(std::is_same_v<X, bool>)
    {
        return x;
    }
    else
    {
        return static_cast<double>(x) != 0.0;
    }
}

struct LogicalNot
{
    template <typename X>
    bool operator()(const X& x) const
    {
        return !isTrue(x);
    }
};

} // namespace hipdnn_test_sdk::utilities::pointwise
