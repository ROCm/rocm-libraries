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
        // bf16 machine epsilon is 2^-7 ≈ 7.8e-3. Output quantization alone
        // can produce ~1 ULP difference between GPU and CPU reference; 5e-3
        // (< 1 ULP) was sub-format-precision and caused spurious failures
        // with semantically correct input ranges.
        return 8e-3f;
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

template <typename T>
constexpr float getMxTolerance()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1e-3f;
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

namespace moe
{

// MoE grouped matmul is a per-expert GEMM over K; reuse the matmul error model.
template <typename T>
constexpr float getToleranceFwd()
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

// MoE grouped matmul backward is a per-expert GEMM reducing over that expert's token
// rows; reuse the matmul error model.
template <typename T>
constexpr float getToleranceBwd()
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

} // namespace moe

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

namespace sdpa
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
        return 1e-2f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace sdpa

namespace resample
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
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace resample

namespace rope
{

// Tolerance for y, computed in f32 on both sides from the same f32 angles: each rotated
// element is two products and one sum, plus the scale, so engine and reference differ only
// in sin/cos rounding (a few f32 ulp, with full range reduction at angles of thousands of
// radians) and in contraction, a few 1e-7 relative to |x| at most. A half or bfloat16 y is
// that f32 value rounded once on each side, so the two may land one output ulp apart:
// 2^-10 relative for half, 2^-7 for bfloat16, under the bounds below.
template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 2e-3f;
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

} // namespace rope

namespace softmax
{

// Tolerance for s (in [0, 1]) and the f32 log-sum-exp stats, both computed in f32.
// Engine and reference differ only in the f32 summation order over the row and in exp()
// rounding: about n * eps relative in the worst case, far below 2e-5 for a few thousand
// columns, and the comparison's rtol term scales with the LSE's magnitude (|stats| < 20 at
// logits in [-8, 8] and a 4096-wide row). A half or bfloat16 s is that f32 value rounded
// once on each side, so the two may land one output ulp apart: 2^-10 relative for half,
// 2^-7 for bfloat16, under the bounds below.
template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        return 2e-5f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 2e-3f;
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

} // namespace softmax

namespace topk
{

// Exact for every dtype: the values output is a copy of the selected input elements (no
// arithmetic), and the indices are integers, which the integer validators compare exactly
// whatever the tolerance. Any difference is a wrong selection, not rounding.
template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float> || std::is_same_v<T, half>
                 || std::is_same_v<T, bfloat16>)
    {
        return 0.0f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace topk

namespace gather
{

// Gather copies x's elements into y without arithmetic or conversion (x and y share one
// dtype), so engine and reference must agree bit for bit: any nonzero difference is a wrong
// row, a wrong column or a corrupted value, never rounding.
template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float> || std::is_same_v<T, half>
                 || std::is_same_v<T, bfloat16>)
    {
        return 0.0f;
    }
    else
    {
        static_assert(false, "Type not supported");
    }
}

} // namespace gather

namespace paged_cache_store
{

// Exact for every container dtype, FP8 included. A stored element is cast(x * scale): one
// f32 product, correctly rounded on both sides, then one round-to-nearest-even cast to the
// container type (none when it is x's own), so engine and reference produce the same bits.
// Every element no slot names keeps its starting contents on both sides. Any difference is
// a wrong slot, a missed or extra write, or a corrupted value, never rounding.
constexpr float getTolerance()
{
    return 0.0f;
}

} // namespace paged_cache_store

} // namespace hipdnn_test_sdk::utilities
