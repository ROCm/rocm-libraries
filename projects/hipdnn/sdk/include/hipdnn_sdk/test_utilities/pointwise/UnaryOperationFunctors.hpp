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
        auto xCompute = static_cast<ComputeType>(x);

        if(xCompute <= lowerClip)
        {
            ComputeType result = (lowerSlope * (xCompute - lowerClip)) + lowerClip;
            return safeConvert<X>(result);
        }
        if(xCompute >= upperClip)
        {
            return safeConvert<X>(upperClip);
        }
        return safeConvert<X>(xCompute);
    }
};

template <typename ComputeType = float>
struct SigmoidForward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        auto xCompute = static_cast<ComputeType>(x);
        ComputeType result = ComputeType{1} / (ComputeType{1} + std::exp(-xCompute));
        return safeConvert<X>(result);
    }
};

template <typename ComputeType = float>
struct TanhForward
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        auto xCompute = static_cast<ComputeType>(x);
        ComputeType result = std::tanh(xCompute);
        return safeConvert<X>(result);
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
