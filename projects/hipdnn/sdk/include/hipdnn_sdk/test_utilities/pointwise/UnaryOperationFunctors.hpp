// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <limits>

namespace hipdnn_sdk
{
namespace test_utilities
{
namespace pointwise
{

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
    auto operator()(const X& x) const -> X;

    template <>
    auto operator()<float>(const float& x) const -> float
    {
        if(x <= lowerClip)
        {
            return (lowerSlope * (x - lowerClip)) + lowerClip;
        }
        if(x >= upperClip)
        {
            return upperClip;
        }
        return x;
    }

    template <>
    auto operator()<double>(const double& x) const -> double
    {
        auto lowerClipD = static_cast<double>(lowerClip);
        auto upperClipD = static_cast<double>(upperClip);
        auto lowerSlopeD = static_cast<double>(lowerSlope);

        if(x <= lowerClipD)
        {
            return (lowerSlopeD * (x - lowerClipD)) + lowerClipD;
        }
        if(x >= upperClipD)
        {
            return upperClipD;
        }
        return x;
    }

    template <>
    auto operator()<half>(const half& x) const -> half
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);

        if(xf <= lowerClip)
        {
            float result = (lowerSlope * (xf - lowerClip)) + lowerClip;
            return static_cast<half>(result);
        }
        if(xf >= upperClip)
        {
            return static_cast<half>(upperClip);
        }
        return static_cast<half>(xf);
    }

    template <>
    auto operator()<hip_bfloat16>(const hip_bfloat16& x) const -> hip_bfloat16
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);

        if(xf <= lowerClip)
        {
            float result = (lowerSlope * (xf - lowerClip)) + lowerClip;
            return static_cast<hip_bfloat16>(result);
        }
        if(xf >= upperClip)
        {
            return static_cast<hip_bfloat16>(upperClip);
        }
        return static_cast<hip_bfloat16>(xf);
    }

    template <>
    auto operator()<int32_t>(const int32_t& x) const -> int32_t
    {
        // Use double precision for int32 calculations to avoid precision loss
        auto xd = static_cast<double>(x);
        auto lowerClipD = static_cast<double>(lowerClip);
        auto upperClipD = static_cast<double>(upperClip);
        auto lowerSlopeD = static_cast<double>(lowerSlope);

        if(xd <= lowerClipD)
        {
            double result = (lowerSlopeD * (xd - lowerClipD)) + lowerClipD;
            return static_cast<int32_t>(result);
        }
        if(xd >= upperClipD)
        {
            return static_cast<int32_t>(upperClipD);
        }
        return static_cast<int32_t>(xd);
    }
};

struct SigmoidForward
{
    template <typename X>
    auto operator()(const X& x) const -> X;

    template <>
    auto operator()<float>(const float& x) const -> float
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    template <>
    auto operator()<double>(const double& x) const -> double
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    template <>
    auto operator()<half>(const half& x) const -> half
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        float result = 1.0f / (1.0f + std::exp(-xf));
        return static_cast<half>(result);
    }

    template <>
    auto operator()<hip_bfloat16>(const hip_bfloat16& x) const -> hip_bfloat16
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        float result = 1.0f / (1.0f + std::exp(-xf));
        return static_cast<hip_bfloat16>(result);
    }

    template <>
    auto operator()<int32_t>(const int32_t& x) const -> int32_t
    {
        // Use double precision for int32 calculations to avoid precision loss
        auto xd = static_cast<double>(x);
        double result = 1.0 / (1.0 + std::exp(-xd));
        return static_cast<int32_t>(result);
    }
};

struct TanhForward
{
    template <typename X>
    auto operator()(const X& x) const -> X;

    template <>
    auto operator()<float>(const float& x) const -> float
    {
        return std::tanh(x);
    }

    template <>
    auto operator()<double>(const double& x) const -> double
    {
        return std::tanh(x);
    }

    template <>
    auto operator()<half>(const half& x) const -> half
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        float result = std::tanh(xf);
        return static_cast<half>(result);
    }

    template <>
    auto operator()<hip_bfloat16>(const hip_bfloat16& x) const -> hip_bfloat16
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        float result = std::tanh(xf);
        return static_cast<hip_bfloat16>(result);
    }

    template <>
    auto operator()<int32_t>(const int32_t& x) const -> int32_t
    {
        // Use double precision for int32 calculations to avoid precision loss
        auto xd = static_cast<double>(x);
        double result = std::tanh(xd);
        return static_cast<int32_t>(result);
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
