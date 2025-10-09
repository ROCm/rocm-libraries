// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <type_traits>

namespace hipdnn_sdk
{
namespace test_utilities
{
namespace pointwise
{

struct Add
{
    template <typename X0, typename X1>
    auto operator()(const X0& x0, const X1& x1) const -> decltype(x0 + x1)
    {
        return x0 + x1;
    }
};

struct Subtract
{
    template <typename X0, typename X1>
    auto operator()(const X0& x0, const X1& x1) const -> decltype(x0 - x1)
    {
        return x0 - x1;
    }
};

// Backward activation operations: dx = dy * local_gradient
// Takes input x and upstream gradient dy, returns downstream gradient dx

template <typename ComputeType = float>
struct ReluBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X
    {
        if constexpr(std::is_same_v<ComputeType, X>)
        {
            // Same precision: compute directly in target type
            auto xVal = x;
            auto dyVal = dy;
            auto localGradient = (xVal > ComputeType{0}) ? ComputeType{1} : ComputeType{0};
            return dyVal * localGradient;
        }
        else
        {
            // Mixed precision: explicit casting with clear intent
            auto xCompute = static_cast<ComputeType>(x);
            auto dyCompute = static_cast<ComputeType>(dy);
            auto localGradient = (xCompute > ComputeType{0}) ? ComputeType{1} : ComputeType{0};
            ComputeType result = dyCompute * localGradient;
            X output = safeConvert<X>(result);
            return output;
        }
    }
};

template <typename ComputeType = float>
struct ParameterizedReluBackward
{
    ComputeType lowerClip;
    ComputeType upperClip;
    ComputeType lowerSlope;

    ParameterizedReluBackward(ComputeType lowerClip, ComputeType upperClip, ComputeType lowerSlope)
        : lowerClip(lowerClip)
        , upperClip(upperClip)
        , lowerSlope(lowerSlope)
    {
    }

    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X
    {
        if constexpr(std::is_same_v<ComputeType, X>)
        {
            // Same precision: compute directly in target type
            auto xVal = x;
            auto dyVal = dy;

            ComputeType localGradient;
            if(xVal < lowerClip)
            {
                localGradient = lowerSlope;
            }
            else if(xVal > upperClip)
            {
                localGradient = ComputeType{0};
            }
            else
            {
                localGradient = ComputeType{1};
            }

            return dyVal * localGradient;
        }
        else
        {
            // Mixed precision: explicit casting with clear intent
            auto xCompute = static_cast<ComputeType>(x);
            auto dyCompute = static_cast<ComputeType>(dy);

            ComputeType localGradient;
            if(xCompute < lowerClip)
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

            ComputeType result = dyCompute * localGradient;
            X output = safeConvert<X>(result);
            return output;
        }
    }
};

template <typename ComputeType = float>
struct SigmoidBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X
    {
        if constexpr(std::is_same_v<ComputeType, X>)
        {
            // Same precision: compute directly in target type
            auto xVal = x;
            auto dyVal = dy;

            ComputeType sigmoidVal = ComputeType{1} / (ComputeType{1} + std::exp(-xVal));
            auto localGradient = sigmoidVal * (ComputeType{1} - sigmoidVal);
            return dyVal * localGradient;
        }
        else
        {
            // Mixed precision: explicit casting with clear intent
            auto xCompute = static_cast<ComputeType>(x);
            auto dyCompute = static_cast<ComputeType>(dy);

            ComputeType sigmoidVal = ComputeType{1} / (ComputeType{1} + std::exp(-xCompute));
            auto localGradient = sigmoidVal * (ComputeType{1} - sigmoidVal);
            ComputeType result = dyCompute * localGradient;
            X output = safeConvert<X>(result);
            return output;
        }
    }
};

template <typename ComputeType = float>
struct TanhBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X
    {
        if constexpr(std::is_same_v<ComputeType, X>)
        {
            // Same precision: compute directly in target type
            auto xVal = x;
            auto dyVal = dy;

            ComputeType tanhVal = std::tanh(xVal);
            auto localGradient = ComputeType{1} - (tanhVal * tanhVal);
            return dyVal * localGradient;
        }
        else
        {
            // Mixed precision: explicit casting with clear intent
            auto xCompute = static_cast<ComputeType>(x);
            auto dyCompute = static_cast<ComputeType>(dy);

            ComputeType tanhVal = std::tanh(xCompute);
            auto localGradient = ComputeType{1} - (tanhVal * tanhVal);
            ComputeType result = dyCompute * localGradient;
            X output = safeConvert<X>(result);
            return output;
        }
    }
};

} // namespace pointwise
} // namespace test_utilities
} // namespace hipdnn_sdk
