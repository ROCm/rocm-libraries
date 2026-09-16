// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Fused forward convolution + elementwise add, one thread per element of the *added*
// output tensor. The graph this serves is ConvolutionFwd -> Pointwise(ADD) where the
// convolution's y is virtual: it has no caller buffer, so the sum lives in a register
// and is consumed by the add in the same thread. That is the elementwise-same-iteration
// -space fusion case, so there is one launch and no workspace.
//
// Everything about the problem except the element type and the block size is a runtime
// argument: extents, group counts, padding, stride, dilation, and every tensor's strides.
// Strides in particular are arguments and not assumptions -- layout is not a field in
// the hipDNN schema, it exists only as the stride pattern, so an NCHW and an NHWC graph
// of the same dims differ here and nowhere else.
//
// Two conventions are taken from the hipDNN CPU reference rather than from the operation's
// popular name, because both are places where libraries disagree:
//
//   * The input coordinate is `y*stride + r*dilation - prePadding`, and a coordinate
//     outside [0, extent) contributes nothing at all rather than contributing a zero
//     product. post_padding never enters the mapping; it only ever widened the output
//     extent, and the output extent arrives here as outP/outQ read off the graph's own
//     y tensor. (CpuFpReferenceConvolution.hpp, fprop's spatial-index loop.)
//   * Groups are not a schema field. They are implied by the channel relationship
//     groups = x.dims[1] / w.dims[1], with w's first dimension indexed by the *global*
//     output channel and x's channel offset by the group. (Same file, `nGroups` /
//     `baseInputChannel` / `wIdx`.)
//
// The accumulator is float regardless of storage type -- the precedent is ConvFwd.cpp in
// this same directory, where a _Float16 accumulator was found to lose too much precision
// against the CPU float reference. The accumulated sum is then rounded back through the
// element type before the add, because the reference stores the convolution's y to a real
// tensor of that type and the add reads it back; keeping full float precision across the
// fusion boundary would be a *different* computation from the one being validated. For
// HKP_CONV_BIAS_TYPE=FLOAT the round trip is the identity.

#include <cstdint>

// The element type. Two levels of paste, deliberately: `##` suppresses expansion of its
// operands, so a one-level macro yields HKP_CB_T_HKP_CONV_BIAS_TYPE, which does not exist.
#ifndef HKP_CONV_BIAS_TYPE
#error "HKP_CONV_BIAS_TYPE must be defined (FLOAT, HALF or BFLOAT16)"
#endif

// The block size the launch will use. Bound into __launch_bounds__ so it is a real
// specialization axis and not a token the compiler never sees: a wrong value here is a
// register-allocation decision, not a comment.
#ifndef HKP_CONV_BIAS_BLOCK_SIZE
#error "HKP_CONV_BIAS_BLOCK_SIZE must be defined (the block's x extent, e.g. 256)"
#endif

#define HKP_CB_T_FLOAT float
#define HKP_CB_T_HALF _Float16
#define HKP_CB_T_BFLOAT16 __bf16

#define HKP_CB_PASTE(tag) HKP_CB_T_##tag
#define HKP_CB_CAT(tag) HKP_CB_PASTE(tag)

using HkpConvBiasElement = HKP_CB_CAT(HKP_CONV_BIAS_TYPE);

// Spelled behind a name so the entry point's declaration contains exactly one parenthesised
// group, its parameter list. The launch ABI is transcribed downstream from that declaration,
// and a second paren group in it is an invitation to transcribe the wrong thing.
#define HKP_CONV_BIAS_LAUNCH_BOUNDS __launch_bounds__(HKP_CONV_BIAS_BLOCK_SIZE)

/// Fused ConvolutionFwd(CROSS_CORRELATION) + Pointwise(ADD).
///
/// out[n, c, p, q] = element(  sum over (cl, r, s) of
///                               x[n, g*wC + cl, p*strideH + r*dilationH - padH,
///                                              q*strideW + s*dilationW - padW]
///                             * w[k, cl, r, s]  )
///                   + bias[n, c, p, q]
///
/// with k = (convK == 1 ? 0 : c), g = k / (convK / (xC / wC)), terms whose input
/// coordinate falls outside the x extents omitted, and every tensor addressed through its
/// own strides. `bias` is addressed with strides the caller has already zeroed on any axis
/// the bias broadcasts along, so this kernel needs no bias extents and no broadcast test.
///
/// The grid is sized from the shape by the caller, so the last block is partially
/// populated and this kernel owns its own bounds check.
extern "C" __global__ HKP_CONV_BIAS_LAUNCH_BOUNDS void ConvBiasFusedFwd(
    const HkpConvBiasElement* __restrict__ x,
    const HkpConvBiasElement* __restrict__ w,
    const HkpConvBiasElement* __restrict__ bias,
    HkpConvBiasElement* __restrict__ out,
    // Extents of the pointwise output, which is the iteration space.
    int32_t outN,
    int32_t outC,
    int32_t outP,
    int32_t outQ,
    // Channel counts: w.dims[0], x.dims[1], w.dims[1]. groups and channels-per-group are
    // derived from these rather than passed, so they cannot disagree with the tensors.
    int32_t convK,
    int32_t xC,
    int32_t wC,
    // x spatial extents, and the filter window.
    int32_t xH,
    int32_t xW,
    int32_t filtR,
    int32_t filtS,
    // pre_padding, stride, dilation -- one value per spatial axis.
    int32_t padH,
    int32_t padW,
    int32_t strideH,
    int32_t strideW,
    int32_t dilationH,
    int32_t dilationW,
    // Strides in elements, in the graph's own [N, C, H, W] / [K, C, R, S] axis order.
    int64_t xStrideN,
    int64_t xStrideC,
    int64_t xStrideH,
    int64_t xStrideW,
    int64_t wStrideK,
    int64_t wStrideC,
    int64_t wStrideR,
    int64_t wStrideS,
    int64_t biasStrideN,
    int64_t biasStrideC,
    int64_t biasStrideH,
    int64_t biasStrideW,
    int64_t outStrideN,
    int64_t outStrideC,
    int64_t outStrideH,
    int64_t outStrideW)
{
    // int64_t throughout: outN*outC*outP*outQ exceeds 2^31 well inside the shapes this
    // admits, and a 32-bit total would wrap and silently defeat the guard below rather
    // than merely mis-sizing it.
    const int64_t total
        = static_cast<int64_t>(outN) * outC * static_cast<int64_t>(outP) * outQ;

    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= total)
    {
        return;
    }

    // Unravel over (n, c, p, q) with q fastest. This is an iteration-space order, not a
    // memory order: the store below goes through outStride*, so the two need not agree.
    int64_t remaining = index;
    const int64_t qOut = remaining % outQ;
    remaining /= outQ;
    const int64_t pOut = remaining % outP;
    remaining /= outP;
    const int64_t cOut = remaining % outC;
    remaining /= outC;
    const int64_t nOut = remaining;

    // Which convolution output channel this element reads. convK == 1 is the broadcast
    // case -- one filter, added to every channel of the bias tensor; convK == outC is the
    // aligned case. The caller admits no other relationship.
    const int64_t kOut = (convK == 1) ? int64_t{0} : cOut;

    const int64_t groups = static_cast<int64_t>(xC) / wC;
    const int64_t kPerGroup = static_cast<int64_t>(convK) / groups;
    const int64_t group = kOut / kPerGroup;
    const int64_t baseInputChannel = group * wC;

    const int64_t wBase = kOut * wStrideK;

    float accumulator = 0.0f;
    for(int64_t cl = 0; cl < wC; ++cl)
    {
        const int64_t xChannelOffset = (baseInputChannel + cl) * xStrideC;
        const int64_t wChannelOffset = wBase + cl * wStrideC;

        for(int64_t r = 0; r < filtR; ++r)
        {
            const int64_t hIn = pOut * strideH + r * dilationH - padH;
            if(hIn < 0 || hIn >= xH)
            {
                continue;
            }
            const int64_t xRowOffset = xChannelOffset + hIn * xStrideH;
            const int64_t wRowOffset = wChannelOffset + r * wStrideR;

            for(int64_t s = 0; s < filtS; ++s)
            {
                const int64_t wIn = qOut * strideW + s * dilationW - padW;
                if(wIn < 0 || wIn >= xW)
                {
                    continue;
                }

                const int64_t xIndex = nOut * xStrideN + xRowOffset + wIn * xStrideW;
                const int64_t wIndex = wRowOffset + s * wStrideS;
                accumulator += static_cast<float>(x[xIndex]) * static_cast<float>(w[wIndex]);
            }
        }
    }

    // Round through the element type before the add: the reference materializes y as a
    // tensor of this type between the two nodes, so this is the value the add sees.
    const float convolved = static_cast<float>(static_cast<HkpConvBiasElement>(accumulator));

    const int64_t biasIndex = nOut * biasStrideN + cOut * biasStrideC + pOut * biasStrideH
                              + qOut * biasStrideW;
    const int64_t outIndex
        = nOut * outStrideN + cOut * outStrideC + pOut * outStrideH + qOut * outStrideW;

    out[outIndex]
        = static_cast<HkpConvBiasElement>(convolved + static_cast<float>(bias[biasIndex]));
}
