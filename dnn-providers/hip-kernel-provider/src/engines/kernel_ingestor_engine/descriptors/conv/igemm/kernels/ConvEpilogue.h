// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fusion hook of the conv engines. Every kernel that writes the conv output calls an epilogue
// functor on each fp32 result before its single rounding to the output type:
//
//   chan   = epi.chan(ch, ok);                       once per output column a thread owns
//   out[o] = Elem<T>::store(epi(acc, chan, o, live)); per element
//
//   ch    the column's channel (the output channel, groups included); ok false for a column past
//         the tensor's edge (no loads there)
//   chan  Epi::Chan, the per-channel operands (bias, scale, ...) held in registers
//   acc   the convolution's fp32 result for one output element
//   o     its element offset in the output tensor (computed only when Epi::NEEDS_OFFSET)
//   live  false for lanes past the tensor's edge, whose result is discarded
//
// Unset CONV_EPI: ConvEpiNone, the identity. The kernel signature, the argument block and the
// code are those of an unfused build. CONV_EPI 1: the kernel gains a second parameter,
// ConvEpiArgs (ConvEpilogueArgs.h), and the functor ConvEpiGen whose body the provider generates
// from the graph's pointwise nodes:
//
//   CONV_EPI_CHAN_FIELDS  the per-channel registers (float members)
//   CONV_EPI_CHAN_CODE    fills c from channel ch, loading only when ok
//   CONV_EPI_CODE         the per-element body: transforms float v and returns it
//
// The generated code reads operand tensors through e.operand[] and values through e.scalar[];
// it depends on the pattern only, never on a value. The whole chain runs in fp32 and the kernel
// rounds once. No fast-math: relu, clamp and the activations keep their NaN and infinity rules.

#pragma once

struct ConvEpiNone
{
    static constexpr bool NEEDS_OFFSET = false;
    struct Chan
    {
    };
    __device__ Chan chan(int, bool) const
    {
        return {};
    }
    __device__ float operator()(float v, const Chan&, long long, bool) const
    {
        return v;
    }
};

#if defined(CONV_EPI) && CONV_EPI

#include "ConvEpilogueArgs.h"

__device__ inline float convEpiLoadF32(const void* p, long long i)
{
    return static_cast<const float*>(p)[i];
}
__device__ inline float convEpiLoadF16(const void* p, long long i)
{
    return static_cast<float>(static_cast<const _Float16*>(p)[i]);
}
__device__ inline float convEpiLoadBf16(const void* p, long long i)
{
    return __builtin_bit_cast(
        float, static_cast<unsigned>(static_cast<const unsigned short*>(p)[i]) << 16);
}

/// The graph's RELU_FWD: at or below the lower clip the slope line through it, at or above the
/// upper clip the upper clip, else v. NaN fails every compare and stays NaN. A zero slope gives
/// the lower clip itself, so relu(-inf) is the clip, not 0 * -inf.
__device__ inline float convEpiRelu(float v, float lower, float upper, float slope)
{
    if(v <= lower)
    {
        return slope == 0.0f ? lower : lower + slope * (v - lower);
    }
    return v >= upper ? upper : v;
}

/// x times a gate in [0, 1]: a gate that is exactly 0 gives 0, so a gate that saturates at -inf
/// gives 0 rather than -inf * 0. NaN stays NaN (a NaN gate is not 0).
__device__ inline float convEpiGated(float v, float gate)
{
    return gate == 0.0f ? 0.0f : v * gate;
}

__device__ inline float convEpiSigmoid(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

/// x * sigmoid(beta x): swish, silu at beta 1.
__device__ inline float convEpiSwish(float v, float beta)
{
    return convEpiGated(v, convEpiSigmoid(beta * v));
}

/// GELU with the exact normal CDF, 0.5 erfc(-x / sqrt 2), accurate in the negative tail.
__device__ inline float convEpiGelu(float v)
{
    return convEpiGated(v, 0.5f * erfcf(-v * 0.70710678118654752f));
}

/// GELU with the tanh approximation (PyTorch's approximate="tanh").
__device__ inline float convEpiGeluTanh(float v)
{
    const float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
    return convEpiGated(v, 0.5f * (1.0f + tanhf(inner)));
}

/// ELU: v above 0, alpha (e^v - 1) at or below.
__device__ inline float convEpiElu(float v, float alpha)
{
    return v > 0.0f ? v : alpha * expm1f(v);
}

/// Softplus as PyTorch computes it: log(1 + e^(beta v)) / beta, and v itself where beta v > 20.
__device__ inline float convEpiSoftplus(float v, float beta)
{
    const float bv = beta * v;
    return bv > 20.0f ? v : log1pf(expf(bv)) / beta;
}

#ifndef CONV_EPI_NEEDS_OFFSET
#define CONV_EPI_NEEDS_OFFSET 0
#endif
#ifndef CONV_EPI_CHAN_FIELDS
#define CONV_EPI_CHAN_FIELDS
#endif
#ifndef CONV_EPI_CHAN_CODE
#define CONV_EPI_CHAN_CODE
#endif
#ifndef CONV_EPI_CODE
#error "CONV_EPI 1 needs the generated CONV_EPI_CODE"
#endif

struct ConvEpiGen
{
    ConvEpiArgs e;
    static constexpr bool NEEDS_OFFSET = CONV_EPI_NEEDS_OFFSET != 0;
    struct Chan
    {
        CONV_EPI_CHAN_FIELDS
    };
    __device__ Chan chan(int ch, bool ok) const
    {
        Chan c{};
        (void)ch;
        (void)ok;
        CONV_EPI_CHAN_CODE
        return c;
    }
    __device__ float operator()(float v, const Chan& c, long long o, bool live) const
    {
        (void)c;
        (void)o;
        (void)live;
        CONV_EPI_CODE
        return v;
    }
};

/// The channel of a flat output offset walked with a fixed step (grid-stride loops of the split
/// reductions and other flat passes): one division per thread at the start, then additions only.
/// The channel of offset o is (o / e.chanStride) % e.chans.
struct ConvEpiFlatChannel
{
    long long inner;
    long long rem;
    long long remStep;
    int chans;
    int ch;
    int chStep;

    __device__ ConvEpiFlatChannel(const ConvEpiArgs& e, long long first, long long step)
        : inner(e.chanStride)
        , chans(e.chans)
    {
        const long long q = first / inner;
        rem = first - q * inner;
        ch = static_cast<int>(q % chans);
        const long long qs = step / inner;
        remStep = step - qs * inner;
        chStep = static_cast<int>(qs % chans);
    }

    /// Moves to the offset one step further.
    __device__ void advance()
    {
        rem += remStep;
        int carry = 0;
        if(rem >= inner)
        {
            rem -= inner;
            carry = 1;
        }
        ch += chStep + carry;
        if(ch >= chans)
        {
            ch -= chans;
        }
    }
};

#define CONV_EPI_PARAM , ConvEpiArgs epiArgs
#define CONV_EPI_ARG \
    , ConvEpiGen     \
    {                \
        epiArgs      \
    }
#else
#define CONV_EPI_PARAM
#define CONV_EPI_ARG
#endif
