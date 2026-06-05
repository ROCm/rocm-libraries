// rocwmma_ue8m0.hpp — UE8M0 block-wise shared exponent for OCP MX formats
//
// UE8M0: unsigned 8-bit exponent, bias=127, value = 2^(val - 127)
// Used as block-wise shared scale for MXFP4, MXFP8, MXBF16 per OCP MX spec.
// Not a float — does not participate in matrix arithmetic directly.
//
// Usage:
//   #include <rocwmma/rocwmma_ue8m0.hpp>
//   ue8m0_t block_exp = 131;  // = 2^(131-127) = 2^4 = 16.0
//   float scale = ue8m0_to_float(block_exp);
//   // → conv: C *= scale

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace rocwmma {

// UE8M0 is a uint8_t encoding a floating scale factor 2^(val-127).
using ue8m0_t = uint8_t;

// Convert single UE8M0 value to IEEE754 float using bit manipulation.
// float bits: sign(1) | exponent(8) | mantissa(23)
// For UE8M0: val = exp_field, so float = 0 | (val) | 0 = 2^(val-127)
__host__ __device__ inline float ue8m0_to_float(ue8m0_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 23;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

// Convert float scale back to UE8M0 (clamped to [0,255]).
__host__ __device__ inline ue8m0_t float_to_ue8m0(float s) {
    // Extract exponent from IEEE754 float
    uint32_t bits;
    __builtin_memcpy(&bits, &s, sizeof(bits));
    int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
    if(exp < 0) return 0;
    if(exp > 128) return 255;
    return static_cast<ue8m0_t>(exp + 127);
}

// Vector: UE8M0 array to float array (for kernel epilogue)
template <int N>
__host__ __device__ inline void ue8m0_array_to_float(
    ue8m0_t const* src, float* dst) {
    #pragma unroll
    for(int i = 0; i < N; ++i)
        dst[i] = ue8m0_to_float(src[i]);
}

// Vector: float array to UE8M0 array (for offline prep)
template <int N>
__host__ __device__ inline void float_array_to_ue8m0(
    float const* src, ue8m0_t* dst) {
    #pragma unroll
    for(int i = 0; i < N; ++i)
        dst[i] = float_to_ue8m0(src[i]);
}

} // namespace rocwmma
