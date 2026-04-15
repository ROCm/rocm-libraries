// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <hipdnn_data_sdk/types.hpp>

namespace hipdnn_test_sdk::utilities
{

// Import types for convenience
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

namespace batchnorm
{

// Note: All tolerance functions return float since tolerances are comparison thresholds
// and don't need to match the data type being tested.

template <typename T>
constexpr float getToleranceInference()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7f; // this needs to be changed when double is supported
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 2e-4f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 5e-4f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 5e-3f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr float getToleranceInferenceWithVariance()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7f; // this needs to be changed when double is supported
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 2e-4f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 8e-4f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 7e-3f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr float getToleranceTraining()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7f; // this needs to be changed when double is supported
    }
    else if constexpr(std::is_same_v<T, float> || std::is_same_v<T, half>)
    {
        return 4e-3f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 8e-3f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr float getToleranceBackward()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7f; // this needs to be changed when double is supported
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 2e-3f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 4e-4f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 3e-3f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace batchnorm

namespace rmsnorm
{

template <typename T>
constexpr float getTolerance()
{
    // RMS tolerance values for use with CpuFpReferenceMiopenRmsValidation
    // These match MIOpen's relative RMS error tolerance (typically 0.4% = 4e-3)
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float> || std::is_same_v<T, half>)
    {
        return 4e-3f; // 0.4% relative RMS error
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 8e-3f; // 0.8% relative RMS error (more lenient for bfloat16)
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}
} // namespace rmsnorm

namespace conv
{

template <typename T>
constexpr float getToleranceFwd()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>)
    {
        // Relaxed from 1e-2f to account for Winograd solvers (e.g. ConvWinoRageRxS),
        // which introduce higher absolute error (~1-3 ULP) for bfloat16.
        // See: https://github.com/ROCm/rocm-libraries/issues/5286
        return 1e-1f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr float getRelativeToleranceFwd()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 1e-2f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        // See: https://github.com/ROCm/rocm-libraries/issues/5286
        return 2e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr float getToleranceBwd()
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
    else if constexpr(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>)
    {
        return 2e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

template <typename T>
constexpr float getToleranceWrw()
{
    // For more information as to why these tolerances are what they are, please
    // refer to driver\conv_driver.hpp -> int ConvDriver<Tgpu, Tref>::VerifyBackward()

    // Since we can't predict which engine will run, we have to go with the lowest common denominator
    // of tolerances.

    if constexpr(std::is_same_v<T, float>)
    {
        return 2e-4f;
    }
    else if constexpr(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>)
    {
        return 2e-1f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace conv

namespace matmul
{

template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half> || std::is_same_v<T, bfloat16>)
    {
        return 1e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace matmul

namespace reduction
{

template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 1e-2f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 5e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace reduction

namespace pointwise
{

template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-7f;
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 1e-3f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 1e-2f;
    }
    else if constexpr(std::is_same_v<T, int8_t>)
    {
        return 0.0f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace pointwise

namespace layernorm
{

template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, double>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        return 1e-4f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 1e-3f;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return 1e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace layernorm

} // namespace hipdnn_test_sdk::utilities
