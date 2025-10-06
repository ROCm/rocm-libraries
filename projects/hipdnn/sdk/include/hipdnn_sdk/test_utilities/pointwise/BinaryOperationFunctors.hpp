// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

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

struct ReluBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X;

    template <>
    auto operator()<float, float>(const float& x, const float& dy) const -> float
    {
        // dx = dy * (x > 0 ? 1 : 0)
        auto localGradient = (x > 0.0f) ? 1.0f : 0.0f;
        return dy * localGradient;
    }

    template <>
    auto operator()<double, double>(const double& x, const double& dy) const -> double
    {
        // dx = dy * (x > 0 ? 1 : 0)
        auto localGradient = (x > 0.0) ? 1.0 : 0.0;
        return dy * localGradient;
    }

    template <>
    auto operator()<half, half>(const half& x, const half& dy) const -> half
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        auto localGradient = (xf > 0.0f) ? 1.0f : 0.0f;
        return static_cast<half>(dyf * localGradient);
    }

    template <>
    auto operator()<hip_bfloat16, hip_bfloat16>(const hip_bfloat16& x, const hip_bfloat16& dy) const
        -> hip_bfloat16
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        auto localGradient = (xf > 0.0f) ? 1.0f : 0.0f;
        return static_cast<hip_bfloat16>(dyf * localGradient);
    }

    template <>
    auto operator()<int32_t, int32_t>(const int32_t& x, const int32_t& dy) const -> int32_t
    {
        // Use double precision for int32 calculations to avoid precision loss
        auto xd = static_cast<double>(x);
        auto dyd = static_cast<double>(dy);

        auto localGradient = (xd > 0.0) ? 1.0 : 0.0;
        return static_cast<int32_t>(dyd * localGradient);
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

    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X;

    template <>
    auto operator()<float, float>(const float& x, const float& dy) const -> float
    {
        float localGradient;
        if(x < lowerClip)
        {
            localGradient = lowerSlope;
        }
        else if(x > upperClip)
        {
            localGradient = 0.0f;
        }
        else
        {
            localGradient = 1.0f;
        }

        return dy * localGradient;
    }

    template <>
    auto operator()<double, double>(const double& x, const double& dy) const -> double
    {
        auto lowerClipD = static_cast<double>(lowerClip);
        auto upperClipD = static_cast<double>(upperClip);
        auto lowerSlopeD = static_cast<double>(lowerSlope);

        double localGradient;
        if(x < lowerClipD)
        {
            localGradient = lowerSlopeD;
        }
        else if(x > upperClipD)
        {
            localGradient = 0.0;
        }
        else
        {
            localGradient = 1.0;
        }

        return dy * localGradient;
    }

    template <>
    auto operator()<half, half>(const half& x, const half& dy) const -> half
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        float localGradient;
        if(xf < lowerClip)
        {
            localGradient = lowerSlope;
        }
        else if(xf > upperClip)
        {
            localGradient = 0.0f;
        }
        else
        {
            localGradient = 1.0f;
        }

        return static_cast<half>(dyf * localGradient);
    }

    template <>
    auto operator()<hip_bfloat16, hip_bfloat16>(const hip_bfloat16& x, const hip_bfloat16& dy) const
        -> hip_bfloat16
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        float localGradient;
        if(xf < lowerClip)
        {
            localGradient = lowerSlope;
        }
        else if(xf > upperClip)
        {
            localGradient = 0.0f;
        }
        else
        {
            localGradient = 1.0f;
        }

        return static_cast<hip_bfloat16>(dyf * localGradient);
    }

    template <>
    auto operator()<int32_t, int32_t>(const int32_t& x, const int32_t& dy) const -> int32_t
    {
        // Use double precision for int32 calculations to avoid precision loss
        auto xd = static_cast<double>(x);
        auto dyd = static_cast<double>(dy);
        auto lowerClipD = static_cast<double>(lowerClip);
        auto upperClipD = static_cast<double>(upperClip);
        auto lowerSlopeD = static_cast<double>(lowerSlope);

        double localGradient;
        if(xd < lowerClipD)
        {
            localGradient = lowerSlopeD;
        }
        else if(xd > upperClipD)
        {
            localGradient = 0.0;
        }
        else
        {
            localGradient = 1.0;
        }

        return static_cast<int32_t>(dyd * localGradient);
    }
};

struct SigmoidBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X;

    template <>
    auto operator()<float, float>(const float& x, const float& dy) const -> float
    {
        // dx = dy * sigmoid(x) * (1 - sigmoid(x))
        float sigmoidVal = 1.0f / (1.0f + std::exp(-x));
        auto localGradient = sigmoidVal * (1.0f - sigmoidVal);
        return dy * localGradient;
    }

    template <>
    auto operator()<double, double>(const double& x, const double& dy) const -> double
    {
        // dx = dy * sigmoid(x) * (1 - sigmoid(x))
        double sigmoidVal = 1.0 / (1.0 + std::exp(-x));
        auto localGradient = sigmoidVal * (1.0 - sigmoidVal);
        return dy * localGradient;
    }

    template <>
    auto operator()<half, half>(const half& x, const half& dy) const -> half
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        float sigmoidVal = 1.0f / (1.0f + std::exp(-xf));
        auto localGradient = sigmoidVal * (1.0f - sigmoidVal);
        return static_cast<half>(dyf * localGradient);
    }

    template <>
    auto operator()<hip_bfloat16, hip_bfloat16>(const hip_bfloat16& x, const hip_bfloat16& dy) const
        -> hip_bfloat16
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        float sigmoidVal = 1.0f / (1.0f + std::exp(-xf));
        auto localGradient = sigmoidVal * (1.0f - sigmoidVal);
        return static_cast<hip_bfloat16>(dyf * localGradient);
    }

    template <>
    auto operator()<int32_t, int32_t>(const int32_t& x, const int32_t& dy) const -> int32_t
    {
        // Use double precision for int32 calculations to avoid precision loss
        auto xd = static_cast<double>(x);
        auto dyd = static_cast<double>(dy);

        double sigmoidVal = 1.0 / (1.0 + std::exp(-xd));
        auto localGradient = sigmoidVal * (1.0 - sigmoidVal);
        return static_cast<int32_t>(dyd * localGradient);
    }
};

struct TanhBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> X;

    template <>
    auto operator()<float, float>(const float& x, const float& dy) const -> float
    {
        // dx = dy * (1 - tanh(x)^2)
        float tanhVal = std::tanh(x);
        auto localGradient = 1.0f - (tanhVal * tanhVal);
        return dy * localGradient;
    }

    template <>
    auto operator()<double, double>(const double& x, const double& dy) const -> double
    {
        // dx = dy * (1 - tanh(x)^2)
        double tanhVal = std::tanh(x);
        auto localGradient = 1.0 - (tanhVal * tanhVal);
        return dy * localGradient;
    }

    template <>
    auto operator()<half, half>(const half& x, const half& dy) const -> half
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        float tanhVal = std::tanh(xf);
        auto localGradient = 1.0f - (tanhVal * tanhVal);
        return static_cast<half>(dyf * localGradient);
    }

    template <>
    auto operator()<hip_bfloat16, hip_bfloat16>(const hip_bfloat16& x, const hip_bfloat16& dy) const
        -> hip_bfloat16
    {
        // Use float precision for computation to avoid precision loss
        auto xf = static_cast<float>(x);
        auto dyf = static_cast<float>(dy);

        float tanhVal = std::tanh(xf);
        auto localGradient = 1.0f - (tanhVal * tanhVal);
        return static_cast<hip_bfloat16>(dyf * localGradient);
    }

    template <>
    auto operator()<int32_t, int32_t>(const int32_t& x, const int32_t& dy) const -> int32_t
    {
        // Use double precision for int32 calculations to avoid precision loss
        auto xd = static_cast<double>(x);
        auto dyd = static_cast<double>(dy);

        double tanhVal = std::tanh(xd);
        auto localGradient = 1.0 - (tanhVal * tanhVal);
        return static_cast<int32_t>(dyd * localGradient);
    }
};

} // namespace pointwise
} // namespace test_utilities
} // namespace hipdnn_sdk
