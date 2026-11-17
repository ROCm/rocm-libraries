// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The epilogue pass's activations: the functions the hip-kernel-provider convolution engines apply
// in their kernels (ConvEpilogue.h there), under this provider's names, so a fused graph gets the
// same activation on every engine. The form follows the output type: few-ulp for an fp32 output,
// the faster one for a 16-bit output (RepackEpiForm).

#pragma once

template <typename OutputType>
struct RepackEpiForm
{
    static constexpr int value = 0;
};
template <>
struct RepackEpiForm<float>
{
    static constexpr int value = 1;
};

/// x times a gate in [0, 1]: a gate that is exactly 0 gives 0, so a gate that saturates at -inf
/// gives 0 rather than -inf * 0. NaN stays NaN (a NaN gate is not 0).
__device__ inline float repackEpiGated(float v, float gate)
{
    return gate == 0.0f ? 0.0f : v * gate;
}

// The activations below are built from the hardware exp2, log2 and reciprocal (v_exp_f32,
// v_log_f32, v_rcp_f32: 1 ulp each) rather than the library's expf/erfcf/expm1f/log1pf and IEEE
// division, which cost them 25-130 instructions an element. Not fast math: NaN stays NaN, the
// infinities keep their limits and signed zeros their sign.
//
// Each comes in two forms, chosen by the output type (template argument P):
//   P = 1, an fp32 output: e^v corrects the rounding of v log2(e), and GELU keeps its argument
//     and exponent to twice the float precision. A few ulp.
//   P = 0, a 16-bit output: exp2(v log2(e)) as it rounds. Its error, up to a few hundred fp32 ulp
//     in the negative tails, stays under 0.1 ulp of the 16-bit output.
// Each states its worst-case error in fp32 ulp of the result before the output rounding: over
// every fp32 input, against a double reference, for results of magnitude 2^-122 and up (both
// forms bit-identical on gfx942 and gfx950). Every error of either form stays under 1.3% of the
// fp32 pointwise tolerance (1e-5 + 1e-5 |reference|). Results below 2^-122 in magnitude may flush
// to zero where an intermediate underflows (an absolute error under 2e-36).
//
// They stay free of branches: a branch in the per-element epilogue splits the unrolled store code
// of the tile kernels into blocks, and their register allocation then grows by up to the tile's
// outputs (measured on the depthwise tiles: 49 to 215 VGPRs). Where a select chooses between v and
// an expensive value, the value passes an empty asm so the compiler does not turn the select into
// a branch around it.
constexpr float REPACK_EPI_LOG2E = 1.44269504088896341f; // log2(e) rounded to float
constexpr float REPACK_EPI_LOG2E_LO = 1.925963033500011e-8f; // log2(e) minus that
constexpr float REPACK_EPI_LN2 = 0.69314718055994531f;

/// e^(hi + lo) for |lo| far below 1: exp2(hi log2 e), corrected to first order by the rounding of
/// that product and by lo, which keeps it within 2 ulp at any magnitude. 0 at -inf, inf at +inf,
/// NaN for NaN.
__device__ inline float repackEpiExpParts(float hi, float lo)
{
    const float y = hi * REPACK_EPI_LOG2E;
    const float r
        = fmaf(hi, REPACK_EPI_LOG2E, -y) + fmaf(hi, REPACK_EPI_LOG2E_LO, lo * REPACK_EPI_LOG2E);
    const float e = __builtin_amdgcn_exp2f(y);
    // Past +-128 the result is inf, 0 or subnormal (flushed), and r may be NaN (inf - inf).
    return fabsf(y) < 128.0f ? fmaf(e, r * REPACK_EPI_LN2, e) : e;
}

/// e^v: P 1 corrected (2 ulp), P 0 as exp2(v log2 e) rounds (relative |v| 2^-24).
template <int P>
__device__ inline float repackEpiExp(float v)
{
    if constexpr(P != 0)
    {
        return repackEpiExpParts(v, 0.0f);
    }
    else
    {
        return __builtin_amdgcn_exp2f(REPACK_EPI_LOG2E * v);
    }
}

/// 1 / (1 + e^-v); sigmoid(-inf) = rcp(inf) = 0. Error: P 1 3.6 ulp; P 0 64 ulp (near -84), 9.0
/// for |x| <= 8.
template <int P>
__device__ inline float repackEpiSigmoid(float v)
{
    return __builtin_amdgcn_rcpf(1.0f + repackEpiExp<P>(-v));
}

/// x * sigmoid(beta x): swish, silu at beta 1. Error at beta 1: P 1 3.2 ulp for |x| <= 8 and 3.7 above -80; P 0 8.9
/// for |x| <= 8 and 63 above -80. Another beta adds the rounding of beta x, relative
/// |beta x| 2^-24 in the tail (at 1.702, |x| <= 8: P 1 13 ulp, P 0 20). Below -88 / beta the gate
/// underflows and the result (magnitude under 1e-36) flushes to zero.
template <int P>
__device__ inline float repackEpiSwish(float v, float beta)
{
    return repackEpiGated(v, repackEpiSigmoid<P>(beta * v));
}

/// P(t) of the Chebyshev fit of Numerical Recipes' erfcc: erfc(z) = t e^(-z^2 + P(t)) for z >= 0,
/// t = 1 / (1 + z / 2), fractional error 1.2e-7 in exact arithmetic.
__device__ inline float repackEpiErfcPoly(float t)
{
    float p = 0.17087277f;
    p = fmaf(p, t, -0.82215223f);
    p = fmaf(p, t, 1.48851587f);
    p = fmaf(p, t, -1.13520398f);
    p = fmaf(p, t, 0.27886807f);
    p = fmaf(p, t, -0.18628806f);
    p = fmaf(p, t, 0.09678418f);
    p = fmaf(p, t, 0.37409196f);
    p = fmaf(p, t, 1.00002368f);
    return fmaf(p, t, -1.26551223f);
}

/// erfc(a / sqrt 2) for a >= 0. P 1 keeps z = a / sqrt 2 and z^2 to twice the float precision, so
/// the tail's exponent loses nothing to their rounding; P 0 rounds both. 0 at +inf, NaN for NaN.
template <int P>
__device__ inline float repackEpiErfcScaled(float a)
{
    if constexpr(P != 0)
    {
        constexpr float RSQRT2 = 0.707106769084930419921875f; // 1 / sqrt 2 rounded to float
        constexpr float RSQRT2_LO = 1.2101617485882343e-8f; // 1 / sqrt 2 minus that
        const float zh = a * RSQRT2;
        const float zl = fmaf(a, RSQRT2, -zh) + a * RSQRT2_LO;
        const float t = __builtin_amdgcn_rcpf(fmaf(0.5f, zh, 1.0f));
        const float p = repackEpiErfcPoly(t);
        const float h = zh * zh;
        const float l = fmaf(zh, zh, -h) + 2.0f * zh * zl;
        return t * repackEpiExpParts(-h, -l) * __builtin_amdgcn_exp2f(p * REPACK_EPI_LOG2E);
    }
    else
    {
        const float z = a * 0.70710678118654752f;
        const float t = __builtin_amdgcn_rcpf(fmaf(0.5f, z, 1.0f));
        return t * repackEpiExp<0>(fmaf(-z, z, repackEpiErfcPoly(t)));
    }
}

/// GELU with the exact normal CDF, x 0.5 erfc(-x / sqrt 2), relative in the negative tail: the
/// gate is 0.5 erfc(|x| / sqrt 2) at or below 0 and 1 minus it above. Error: P 1 8.8 ulp; P 0 310
/// ulp (near -12.5, where GELU is -1e-34), 119 for |x| <= 8.
template <int P>
__device__ inline float repackEpiGelu(float v)
{
    const float half = 0.5f * repackEpiErfcScaled<P>(fabsf(v));
    return repackEpiGated(v, v <= 0.0f ? half : 1.0f - half);
}

/// tanh: below |v| 0.625 the odd minimax polynomial v + v^3 P(v^2) (under 1 ulp; e^(2|v|) - 1
/// would cancel there), above it 1 - 2 / (e^(2|v|) + 1); signed like v, 1 at infinity, NaN for
/// NaN. Error: 1.6 ulp (either form).
template <int P>
__device__ inline float repackEpiTanh(float v)
{
    const float a = fabsf(v);
    const float s = a * a;
    float p = -0.005704977549612522f;
    p = fmaf(p, s, 0.020639078691601753f);
    p = fmaf(p, s, -0.05373971164226532f);
    p = fmaf(p, s, 0.13331441581249237f);
    p = fmaf(p, s, -0.3333328068256378f);
    const float small = fmaf(a * s, p, a);
    const float large = 1.0f - 2.0f * __builtin_amdgcn_rcpf(1.0f + repackEpiExp<P>(2.0f * a));
    return copysignf(a < 0.625f ? small : large, v);
}

/// GELU with the tanh approximation (PyTorch's approximate="tanh"): its gate 0.5 (1 + tanh u) is
/// sigmoid(2u), which keeps the negative tail's relative accuracy. The rounding of u carries a
/// relative error of about |4u| 2^-24 into the gate in the negative tail, as in any fp32
/// evaluation of the formula. Error: P 1 94 ulp for |x| <= 8 (at -7.7, where the result is
/// -5e-19), 184 at -9.95; P 0 134 for |x| <= 8, 230 near -9.8.
template <int P>
__device__ inline float repackEpiGeluTanh(float v)
{
    const float inner = 0.79788456080286536f * fmaf(0.044715f * v, v * v, v);
    return repackEpiGated(v, repackEpiSigmoid<P>(2.0f * inner));
}

/// ELU: v above 0, alpha (e^v - 1) at or below: above -0.5 the minimax polynomial v + v^2 P(v)
/// (under 1 ulp; e^v - 1 would cancel there), else e^v - 1. Error: 1.8 ulp (either form), and the
/// rounding of the product by alpha.
template <int P>
__device__ inline float repackEpiElu(float v, float alpha)
{
    float p = 0.0011259219609200954f;
    p = fmaf(p, v, 0.008193294517695904f);
    p = fmaf(p, v, 0.04163204878568649f);
    p = fmaf(p, v, 0.16666288673877716f);
    p = fmaf(p, v, 0.49999985098838806f);
    const float small = fmaf(v * v, p, v);
    float below = alpha * (v > -0.5f ? small : repackEpiExp<P>(v) - 1.0f);
    asm("" : "+v"(below));
    return v > 0.0f ? v : below;
}

/// log(1 + e) for e in [0, 1], relative for small e: log(u) e / (u - 1) with u = 1 + e corrects
/// the rounding of u (Goldberg), and e itself where u rounds to 1.
__device__ inline float repackEpiLog1pUnit(float e)
{
    const float u = 1.0f + e;
    const float l = __builtin_amdgcn_logf(u) * REPACK_EPI_LN2;
    return u == 1.0f ? e : l * (e * __builtin_amdgcn_rcpf(u - 1.0f));
}

/// Softplus as PyTorch computes it, log(1 + e^(beta v)) / beta, and v itself where beta v > 20;
/// as max(beta v, 0) + log(1 + e^-|beta v|), so the exponential never overflows. Error at beta 1
/// and 2: P 1 4.9 ulp; P 0 63 ulp (near beta v = -84), 11 and 16 for |x| <= 8. Another beta adds
/// the rounding of beta v.
template <int P>
__device__ inline float repackEpiSoftplus(float v, float beta)
{
    const float bv = beta * v;
    float smooth = (fmaxf(bv, 0.0f) + repackEpiLog1pUnit(repackEpiExp<P>(-fabsf(bv))))
                   * __builtin_amdgcn_rcpf(beta);
    asm("" : "+v"(smooth));
    return bv > 20.0f ? v : smooth;
}
