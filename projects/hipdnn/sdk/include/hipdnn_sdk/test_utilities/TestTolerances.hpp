// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

namespace hipdnn_sdk
{
namespace test_utilities
{

namespace conv
{

template <typename T>
constexpr float getToleranceFwd();

template <>
constexpr float getToleranceFwd<float>()
{
    return 15e-7f;
}

template <>
constexpr float getToleranceFwd<half>()
{
    return 15e-4f;
}

template <>
constexpr float getToleranceFwd<hip_bfloat16>()
{
    return 1e-2f;
}

} // namespace conv

} // namespace test_utilities
} // namespace hipdnn_sdk
