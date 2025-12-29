// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceUtilities.hpp>
#include <limits>
#include <type_traits>

namespace hipdnn_test_sdk::utilities::pointwise
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
    auto operator()(const X& x) const -> ComputeType
    {
        auto xCompute = static_cast<ComputeType>(x);

        if(xCompute <= lowerClip)
        {
            return (lowerSlope * (xCompute - lowerClip)) + lowerClip;
        }
        if(xCompute >= upperClip)
        {
            return upperClip;
        }
        return xCompute;
    }
};

template <typename ComputeType = float>
struct SigmoidForward
{
    template <typename X>
    auto operator()(const X& x) const -> ComputeType
    {
        auto xCompute = static_cast<ComputeType>(x);
        return ComputeType{1} / (ComputeType{1} + std::exp(-xCompute));
    }
};

template <typename ComputeType = float>
struct TanhForward
{
    template <typename X>
    auto operator()(const X& x) const -> ComputeType
    {
        auto xCompute = static_cast<ComputeType>(x);
        return std::tanh(xCompute);
    }
};

struct Identity
{
    template <typename X>
    auto operator()(const X& x) const -> X
    {
        return x;
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

} // namespace hipdnn_test_sdk::utilities::pointwise
