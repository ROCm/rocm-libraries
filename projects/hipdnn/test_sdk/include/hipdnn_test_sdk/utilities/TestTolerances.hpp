// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_test_sdk::utilities
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
constexpr T getToleranceInferenceWithVariance()
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
        // ~32% more lenient for BN with variance vs BN with inv variance (5e-4_h)
        return 6.6e-4_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        // ~4% more lenient for BN with variance vs BN with inv variance (5e-3_bf)
        return 5.2e-3_bf;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr T getToleranceTraining()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7; // this needs to be changed when double is supported
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 4e-3f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 4e-3_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 8e-3_bf;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

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

template <typename T>
constexpr T getRmsToleranceTraining()
{
    // RMS tolerance values for use with CpuFpReferenceMiopenRmsValidation
    // These match MIOpen's relative RMS error tolerance (typically 0.4% = 4e-3)
    if constexpr(std::is_same_v<T, double>)
    {
        return 4e-3; // 0.4% relative RMS error
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 4e-3f; // 0.4% relative RMS error
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 4e-3_h; // 0.4% relative RMS error
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 8e-3_bf; // 0.8% relative RMS error (more lenient for bfloat16)
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace batchnorm

namespace conv
{

template <typename T>
constexpr T getToleranceFwd()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
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

template <typename T>
constexpr T getToleranceBwd()
{
    // Note: MIOpen seems to have some accuracy issues with 16-bit fp in some cases (like iGemm),
    //       so we relax the tolerance a bit here.
    //       See: ConvDriver<Tgpu, Tref>::VerifyBackward()

    // Since we can't predict which engine will run, we have to go with the lowest common denominator
    // of tolerances.

    if constexpr(std::is_same_v<T, float>)
    {
        return 8.5e-6f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 2e-2_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 2e-2_bf;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr T getToleranceWrw()
{
    // For more information as to why these tolerances are what they are, please
    // refer to driver\conv_driver.hpp -> int ConvDriver<Tgpu, Tref>::VerifyBackward()

    // Since we can't predict which engine will run, we have to go with the lowest common denominator
    // of tolerances.

    if constexpr(std::is_same_v<T, float>)
    {
        return 2e-4f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 2e-1_h;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return 2e-1_bf;
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
    else if constexpr(std::is_same_v<T, int8_t>)
    {
        return 0;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace pointwise

} // namespace hipdnn_test_sdk::utilities
