// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <type_traits>

namespace hipdnn_sdk::test_utilities::pointwise
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

struct Multiply
{
    template <typename X0, typename X1>
    auto operator()(const X0& x0, const X1& x1) const -> decltype(x0 * x1)
    {
        return x0 * x1;
    }
};

// Backward activation operations: dx = dy * local_gradient
// Takes input x and upstream gradient dy, returns downstream gradient dx

template <typename ComputeType = float>
struct ReluBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> ComputeType
    {
        auto xCompute = static_cast<ComputeType>(x);
        auto dyCompute = static_cast<ComputeType>(dy);
        auto localGradient = (xCompute > ComputeType{0}) ? ComputeType{1} : ComputeType{0};
        return dyCompute * localGradient;
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
    auto operator()(const X& x, const Dy& dy) const -> ComputeType
    {
        auto xCompute = static_cast<ComputeType>(x);
        auto dyCompute = static_cast<ComputeType>(dy);

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

        return dyCompute * localGradient;
    }
};

template <typename ComputeType = float>
struct SigmoidBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> ComputeType
    {
        auto xCompute = static_cast<ComputeType>(x);
        auto dyCompute = static_cast<ComputeType>(dy);

        ComputeType sigmoidVal = ComputeType{1} / (ComputeType{1} + std::exp(-xCompute));
        auto localGradient = sigmoidVal * (ComputeType{1} - sigmoidVal);
        return dyCompute * localGradient;
    }
};

template <typename ComputeType = float>
struct TanhBackward
{
    template <typename X, typename Dy>
    auto operator()(const X& x, const Dy& dy) const -> ComputeType
    {
        auto xCompute = static_cast<ComputeType>(x);
        auto dyCompute = static_cast<ComputeType>(dy);

        ComputeType tanhVal = std::tanh(xCompute);
        auto localGradient = ComputeType{1} - (tanhVal * tanhVal);
        return dyCompute * localGradient;
    }
};

} // namespace hipdnn_sdk::test_utilities::pointwise
