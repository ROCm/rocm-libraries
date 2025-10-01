// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

namespace hipdnn_sdk
{
namespace test_utilities
{

namespace batchnorm
{

template <typename T>
constexpr float getToleranceInference();

template <typename T>
constexpr float getToleranceTraining();

template <typename T>
constexpr float getToleranceBackward();

template <>
constexpr float getToleranceInference<double>()
{
    return 1e-7f; // this needs to be changed when double is supported
}

template <>
constexpr float getToleranceInference<float>()
{
    return 2e-4f;
}

template <>
constexpr float getToleranceInference<half>()
{
    return 5e-4f;
}

template <>
constexpr float getToleranceInference<hip_bfloat16>()
{
    return 5e-3f;
}

template <>
constexpr float getToleranceBackward<double>()
{
    return 1e-7f; // this needs to be changed when double is supported
}

template <>
constexpr float getToleranceBackward<float>()
{
    return 1e-3f;
}

template <>
constexpr float getToleranceBackward<half>()
{
    return 4e-4f;
}

template <>
constexpr float getToleranceBackward<hip_bfloat16>()
{
    return 3e-3f;
}

} // namespace bn

namespace conv
{

template <typename T>
constexpr float getToleranceFwd();

template <>
constexpr float getToleranceFwd<float>()
{
    return 3e-6f;
}

template <>
constexpr float getToleranceFwd<half>()
{
    return 1e-2f;
}

template <>
constexpr float getToleranceFwd<hip_bfloat16>()
{
    return 1e-2f;
}

} // namespace conv

namespace pointwise
{

template <typename T>
constexpr float getTolerance();

template <>
constexpr float getTolerance<double>()
{
    return 1e-7f;
}

template <>
constexpr float getTolerance<float>()
{
    return 1e-5f;
}

template <>
constexpr float getTolerance<half>()
{
    return 1e-3f;
}

template <>
constexpr float getTolerance<hip_bfloat16>()
{
    return 1e-2f;
}

} // namespace pointwise

} // namespace test_utilities
} // namespace hipdnn_sdk
