// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/UnaryOperationFunctors.hpp>
#include <type_traits>

namespace hipdnn_test_sdk::utilities::pointwise
{

// Binary arithmetic operations with explicit ComputeType and OutputType
// ComputeType: The type used for intermediate calculations
// OutputType: The type returned from the operation

template <typename ComputeType = float, typename OutputType = ComputeType>
struct Add
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto result = static_cast<ComputeType>(x0) + static_cast<ComputeType>(x1);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct Subtract
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto result = static_cast<ComputeType>(x0) - static_cast<ComputeType>(x1);
        return static_cast<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct Multiply
{
    template <typename X0, typename X1>
    OutputType operator()(const X0& x0, const X1& x1) const
    {
        auto result = static_cast<ComputeType>(x0) * static_cast<ComputeType>(x1);
        return static_cast<OutputType>(result);
    }
};

// Backward activation operations: dx = dy * local_gradient
// Takes upstream gradient dy and forward input x.

template <typename ComputeType = float, typename OutputType = ComputeType>
struct ReluBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);
        auto localGradient = (xCompute > ComputeType{0}) ? ComputeType{1} : ComputeType{0};
        return static_cast<OutputType>(dyCompute * localGradient);
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
template <typename ComputeType = float, typename OutputType = ComputeType>
struct ParameterizedReluBackward
{
    ComputeType lowerClip;
    ComputeType upperClip;
    ComputeType lowerSlope;

    ParameterizedReluBackward(ComputeType lowerClipVal,
                              ComputeType upperClipVal,
                              ComputeType lowerSlopeVal)
        : lowerClip(lowerClipVal)
        , upperClip(upperClipVal)
        , lowerSlope(lowerSlopeVal)
    {
    }

    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

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

        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct SigmoidBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::exp;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        ComputeType sigmoidVal = ComputeType{1} / (ComputeType{1} + exp(-xCompute));
        auto localGradient = sigmoidVal * (ComputeType{1} - sigmoidVal);
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct TanhBackward
{
    template <typename Dy, typename X>
    OutputType operator()(const Dy& dy, const X& x) const
    {
        using hipdnn_data_sdk::types::tanh;
        auto dyCompute = static_cast<ComputeType>(dy);
        auto xCompute = static_cast<ComputeType>(x);

        ComputeType tanhVal = tanh(xCompute);
        auto localGradient = ComputeType{1} - (tanhVal * tanhVal);
        return static_cast<OutputType>(dyCompute * localGradient);
    }
};

} // namespace hipdnn_test_sdk::utilities::pointwise
