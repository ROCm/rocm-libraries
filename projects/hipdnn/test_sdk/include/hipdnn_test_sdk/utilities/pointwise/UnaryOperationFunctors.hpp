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

} // namespace hipdnn_test_sdk::utilities::pointwise
