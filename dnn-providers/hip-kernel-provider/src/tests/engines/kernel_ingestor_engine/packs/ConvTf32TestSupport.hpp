// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>

/**
 * @file ConvTf32TestSupport.hpp
 * @brief Shared pieces of the conv engines' TF32 device tests: the archs with three-bf16 forms,
 *        planted infinities and a class-aware comparison (an infinity must come out as that
 *        infinity, NaN where both meet; a finite result within tolerance).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_tf32_test
{

/// The TF32 permission variable every hip-kernel-provider TF32 form reads.
inline constexpr const char* TF32_ENV = "HIP_KERNEL_PROVIDER_CONV_TF32";

/// Three-bf16's error bound per reduction element for values in [-1, 1]: lo x lo dropped and the
/// lo parts rounded, about 2^-16 of each product.
inline constexpr float THREE_BF16_BOUND = 3e-5f;

/// Whether @p gcnArchName has bf16 matrix cores the three-bf16 forms run on (gfx90a, gfx942,
/// gfx950, gfx11, gfx12; gfx908's half-rate bf16 is slower than its fp32 MFMA).
inline bool hasThreeBf16(const std::string& gcnArchName)
{
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(gcnArchName));
    return arch == "gfx90a" || arch == "gfx942" || arch == "gfx950" || arch.rfind("gfx11", 0) == 0
           || arch.rfind("gfx12", 0) == 0;
}

/// +inf and -inf planted into @p input (the tensor the reduction reads, besides the filter).
template <typename T>
void plantInfinities(hipdnn_data_sdk::utilities::Tensor<T>& input)
{
    int64_t count = 1;
    for(const auto extent : input.dims())
    {
        count *= extent;
    }
    auto* data = static_cast<T*>(input.rawHostData());
    data[0] = static_cast<T>(INFINITY);
    data[count / 2] = static_cast<T>(-INFINITY);
    input.markHostModified();
}

/// A result matches its reference: within @p tolerance where the reference is finite, the same
/// infinity, or NaN where the reference is NaN.
inline bool matches(float got, float want, float tolerance)
{
    if(std::isnan(want))
    {
        return std::isnan(got);
    }
    if(std::isinf(want))
    {
        return got == want;
    }
    return std::fabs(got - want) <= tolerance;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_tf32_test
