// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <limits>
#include <type_traits>

namespace hipdnn_sdk
{
namespace test_utilities
{
namespace pointwise
{

template <typename ComputeType = float>
struct ReluForward
{
    ComputeType lowerClip;
    ComputeType upperClip;
    ComputeType lowerSlope;

    ReluForward(ComputeType lowerClip = ComputeType{0},
                ComputeType upperClip = std::numeric_limits<ComputeType>::max(),
                ComputeType lowerSlope = ComputeType{0})
        : lowerClip(lowerClip)
        , upperClip(upperClip)
        , lowerSlope(lowerSlope)
    {
    }

    template <typename X>
    auto operator()(const X& x) const -> X
    {
        if constexpr(std::is_same_v<ComputeType, X>)
        {
            // Same precision: compute directly in target type
            auto xVal = x;
            if(xVal <= lowerClip)
            {
                ComputeType result = (lowerSlope * (xVal - lowerClip)) + lowerClip;
                return result;
            }
            if(xVal >= upperClip)
            {
                return upperClip;
            }
            return xVal;
        }
        else
        {
            // Mixed precision: explicit casting with clear intent
            auto xCompute = static_cast<ComputeType>(x);

            if(xCompute <= lowerClip)
            {
                ComputeType result = (lowerSlope * (xCompute - lowerClip)) + lowerClip;
                X output = safeConvert<X>(result);
                return output;
            }
            if(xCompute >= upperClip)
            {
                X output = safeConvert<X>(upperClip);
                return output;
            }
            X output = safeConvert<X>(xCompute);
            return output;
        }
    }
};

template <typename ComputeType = float>
struct SigmoidForward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        if constexpr(std::is_same_v<ComputeType, X>)
        {
            // Same precision: compute directly in target type
            auto xVal = x;
            return ComputeType{1} / (ComputeType{1} + std::exp(-xVal));
        }
        else
        {
            // Mixed precision: explicit casting with clear intent
            auto xCompute = static_cast<ComputeType>(x);
            ComputeType result = ComputeType{1} / (ComputeType{1} + std::exp(-xCompute));
            X output = safeConvert<X>(result);
            return output;
        }
    }
};

template <typename ComputeType = float>
struct TanhForward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        if constexpr(std::is_same_v<ComputeType, X>)
        {
            // Same precision: compute directly in target type
            auto xVal = x;
            return std::tanh(xVal);
        }
        else
        {
            // Mixed precision: explicit casting with clear intent
            auto xCompute = static_cast<ComputeType>(x);
            ComputeType result = std::tanh(xCompute);
            X output = safeConvert<X>(result);
            return output;
        }
    }
};

struct AbsoluteValue
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        return static_cast<X>(std::abs(x));
    }
};

struct Negation
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        return -x;
    }
};

} // namespace pointwise
} // namespace test_utilities
} // namespace hipdnn_sdk
