// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <limits>

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

struct ReluForward
{
    float lowerClip;
    float upperClip;
    float lowerSlope;

    ReluForward(float lowerClip = 0.0f,
                float upperClip = std::numeric_limits<float>::max(),
                float lowerSlope = 0.0f)
        : lowerClip(lowerClip)
        , upperClip(upperClip)
        , lowerSlope(lowerSlope)
    {
    }

    template <typename X>
    auto operator()(const X& x) const -> X
    {
        auto lowerClipX = static_cast<X>(lowerClip);
        auto upperClipX = static_cast<X>(upperClip);
        auto lowerSlopeX = static_cast<X>(lowerSlope);

        if(x <= lowerClipX)
        {
            return lowerSlopeX * (x - lowerClipX) + lowerClipX;
        }
        if(x >= upperClipX)
        {
            return upperClipX;
        }
        return x;
    }
};

struct ReluBackward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        return (x > static_cast<X>(0.0)) ? static_cast<X>(1.0) : static_cast<X>(0.0);
    }
};

struct ParameterizedReluBackward
{
    float lowerClip;
    float upperClip;
    float lowerSlope;

    ParameterizedReluBackward(float lowerClip, float upperClip, float lowerSlope)
        : lowerClip(lowerClip)
        , upperClip(upperClip)
        , lowerSlope(lowerSlope)
    {
    }

    template <typename X>
    auto operator()(const X& x) const -> X
    {
        auto lowerClipX = static_cast<X>(lowerClip);
        auto upperClipX = static_cast<X>(upperClip);
        auto lowerSlopeX = static_cast<X>(lowerSlope);

        if(x < lowerClipX)
        {
            return lowerSlopeX;
        }
        if(x > upperClipX)
        {
            return static_cast<X>(0.0);
        }
        return static_cast<X>(1.0);
    }
};

struct SigmoidForward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        auto xVal = static_cast<X>(x);
        auto oneVal = static_cast<X>(1.0);
        return oneVal / (oneVal + static_cast<X>(std::exp(-xVal)));
    }
};

struct SigmoidBackward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        auto xVal = static_cast<X>(x);
        auto oneVal = static_cast<X>(1.0);
        auto sigmoid = oneVal / (oneVal + static_cast<X>(std::exp(-xVal)));
        return sigmoid * (oneVal - sigmoid);
    }
};

struct TanhForward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        return static_cast<X>(std::tanh(static_cast<X>(x)));
    }
};

struct TanhBackward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        auto tanhX = static_cast<X>(std::tanh(static_cast<X>(x)));
        auto oneVal = static_cast<X>(1.0);
        return oneVal - tanhX * tanhX;
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
