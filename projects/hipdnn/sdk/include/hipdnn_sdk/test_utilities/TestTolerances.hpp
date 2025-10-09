// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

namespace batchnorm
{

template <typename T>
constexpr T getToleranceInference()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7; // this needs to be changed when double is supported
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 2e-4f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 5e-4_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 5e-3_bf;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr T getToleranceTraining();

template <typename T>
constexpr T getToleranceBackward()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7; // this needs to be changed when double is supported
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 2e-3f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 4e-4_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 3e-3_bf;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace bn

namespace conv
{

template <typename T>
constexpr T getToleranceFwd()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 8.5e-6f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 1e-2_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 1e-2_bf;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace conv

namespace pointwise
{

template <typename T>
constexpr T getTolerance()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7;
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 1e-3_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 1e-2_bf;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace pointwise

} // namespace test_utilities
} // namespace hipdnn_sdk
