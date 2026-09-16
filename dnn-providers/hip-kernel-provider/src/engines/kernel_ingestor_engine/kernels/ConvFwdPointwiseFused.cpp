// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Fused ConvolutionFwd + unary Pointwise for a rank-4 hipDNN graph, compiled through
// hipRTC.  The convolution's output tensor is `virtual` in the graph that motivated this
// kernel, so it has no caller buffer at all: the accumulator stays in a register, the
// activation is applied to it, and only the Pointwise output is ever stored.  That is why
// this is one launch and needs no workspace.
//
// Indexing is done through the graph's own strides rather than an assumed packed layout.
// Layout is not a field in `tensor_attributes.fbs`; it exists only as the relationship
// between dims and strides, so NCHW, NHWC and any other permutation are the same code
// path here and differ only in the stride arguments the handler passes.
//
// Algorithm provenance: the (group, batch, out-channel) x (output-plane) decomposition and
// the hoisting of the batch/group/channel base offsets out of the accumulation loop follow
// MIOpen's runtime-compiled reference kernel
// `projects/miopen/src/kernels/gpu_reference_kernel/naive_conv.cpp`
// (`naive_conv_fwd_nchw`, the ASSUME_PACKED == false branch).  What is NOT taken from it:
// its grid-stride loop over the output plane -- a graph with N = K = G = 1 would then
// occupy a single workgroup -- is replaced by a second grid dimension over the plane, so
// the plane is spread across the device and the kernel guards its own tail.
//
// The accumulation order (channel, then filter row, then filter column, skipping positions
// outside the padded input) is deliberately the same order as the CPU reference in
// `hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp::fprop`, so the two sides drift
// only through fused-multiply-add contraction and not through summation order.
//
// Two compile-time axes, both proved by their absence being a compile error:
//
//   HKP_IO_DTYPE   storage element type for x, w and y.
//   HKP_ACTIVATION which unary Pointwise mode is fused into the epilogue.
//
// Everything else -- extents, strides, convolution geometry and the ReLU clip parameters --
// is a runtime argument.  Floats cannot be rendered into a `-D` at all on the integration
// path, and extents cost registers rather than correctness.

#ifndef HKP_IO_DTYPE
#error "HKP_IO_DTYPE must be defined: one of HKP_DTYPE_FLOAT, HKP_DTYPE_HALF, HKP_DTYPE_BFLOAT16"
#endif

#ifndef HKP_ACTIVATION
#error "HKP_ACTIVATION must be defined: one of HKP_ACT_IDENTITY, HKP_ACT_RELU_FWD, HKP_ACT_ABS, HKP_ACT_NEG"
#endif

#include <cstdint>

// Two-level paste.  A one-level `##` would concatenate the macro's *name* instead of its
// expansion, so HKP_IO_DTYPE would resolve to the type named HKP_TYPE_HKP_IO_DTYPE -- which
// does not exist, but only after the bug has already chosen the wrong path once.
#define HKP_CAT_IMPL(first, second) first##second
#define HKP_CAT(first, second) HKP_CAT_IMPL(first, second)

// Legal HKP_IO_DTYPE tags.  A tag with no row here names an undeclared type and fails to
// compile, which is the point: hipRTC is the only thing standing between a mis-specialized
// descriptor and a silently wrong number.
#define HKP_TYPE_HKP_DTYPE_FLOAT float
#define HKP_TYPE_HKP_DTYPE_HALF _Float16
#define HKP_TYPE_HKP_DTYPE_BFLOAT16 __bf16

using HkpElement = HKP_CAT(HKP_TYPE_, HKP_IO_DTYPE);

// Legal HKP_ACTIVATION tags, resolved to a function name by the same paste.  An `#if
// HKP_ACTIVATION == ...` chain would be the wrong tool: the preprocessor scores an unknown
// identifier in an `#if` as 0, so a mistyped tag would silently select whichever branch
// tests against 0 rather than failing.  An undeclared function does not have that failure
// mode.
#define HKP_ACTIVATION_FN(tag) HKP_CAT(hkpActivation_, tag)

/// RELU_FWD, reproducing `hipdnn_test_sdk::pointwise::ReluForward` branch for branch.  The
/// three parameters carry the graph's relu_lower_clip / relu_upper_clip /
/// relu_lower_clip_slope; when the attributes are absent the reference substitutes 0,
/// FLT_MAX and 0, and the handler must pass exactly those.  Written as the same three-way
/// comparison rather than as fmaxf(v, 0) so that the infinity and NaN cases agree with the
/// reference too: +inf clamps to upperClip, and a NaN falls through both comparisons and is
/// propagated.
__device__ __forceinline__ float
    hkpActivation_HKP_ACT_RELU_FWD(float value, float lowerClip, float upperClip, float lowerSlope)
{
    if(value <= lowerClip)
    {
        return (lowerSlope * (value - lowerClip)) + lowerClip;
    }
    if(value >= upperClip)
    {
        return upperClip;
    }
    return value;
}

__device__ __forceinline__ float hkpActivation_HKP_ACT_IDENTITY(float value, float, float, float)
{
    return value;
}

__device__ __forceinline__ float hkpActivation_HKP_ACT_ABS(float value, float, float, float)
{
    return value < 0.0f ? -value : value;
}

__device__ __forceinline__ float hkpActivation_HKP_ACT_NEG(float value, float, float, float)
{
    return -value;
}

/// Fused rank-4 convolution forward + unary pointwise.
///
/// Launch geometry, which the dispatch handler owns and this kernel assumes:
///   block = (blockSize, 1, 1)               blockSize is a runtime choice, 256 is the
///                                           geometry this round was proved at
///   grid  = (ceil(outHeight * outWidth / blockSize),
///            groupCount * batchCount * outChannelsPerGroup,
///            1)
/// The grid's x extent is rounded up, so the final block in x is partially populated and
/// this kernel -- not the handler -- owns the bounds check.  The y extent is exact; it is
/// checked anyway, because a launch that disagrees with this kernel about the argument list
/// or the geometry is diagnosed nowhere else in the stack.
extern "C" __global__ void hkpConvFwdPointwiseFused(const HkpElement* __restrict__ x,
                                                    const HkpElement* __restrict__ w,
                                                    HkpElement* __restrict__ y,
                                                    int64_t batchCount,
                                                    int64_t groupCount,
                                                    int64_t inChannelsPerGroup,
                                                    int64_t outChannelsPerGroup,
                                                    int64_t inHeight,
                                                    int64_t inWidth,
                                                    int64_t filterHeight,
                                                    int64_t filterWidth,
                                                    int64_t outHeight,
                                                    int64_t outWidth,
                                                    int64_t strideHeight,
                                                    int64_t strideWidth,
                                                    int64_t dilationHeight,
                                                    int64_t dilationWidth,
                                                    int64_t prePadHeight,
                                                    int64_t prePadWidth,
                                                    int64_t xStrideBatch,
                                                    int64_t xStrideChannel,
                                                    int64_t xStrideHeight,
                                                    int64_t xStrideWidth,
                                                    int64_t wStrideOutChannel,
                                                    int64_t wStrideInChannel,
                                                    int64_t wStrideHeight,
                                                    int64_t wStrideWidth,
                                                    int64_t yStrideBatch,
                                                    int64_t yStrideChannel,
                                                    int64_t yStrideHeight,
                                                    int64_t yStrideWidth,
                                                    float reluLowerClip,
                                                    float reluUpperClip,
                                                    float reluLowerClipSlope)
{
    // int64_t throughout: a tensor offset is dims x strides and overflows 32 bits long
    // before the shapes stop being reasonable, and a wrapped offset would defeat the
    // `>= total` guards below rather than trip them.
    const int64_t planeCount = groupCount * batchCount * outChannelsPerGroup;
    const int64_t planeIndex = static_cast<int64_t>(blockIdx.y);
    if(planeIndex >= planeCount)
    {
        return;
    }

    const int64_t spatialCount = outHeight * outWidth;
    const int64_t spatialIndex = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(spatialIndex >= spatialCount)
    {
        return;
    }

    // Unravel (group, batch, out-channel-within-group) from the y grid dimension, and
    // (row, column) from the x one.  Consecutive threads take consecutive output columns,
    // so a layout whose fastest axis is the output width -- NCHW's -- is coalesced.
    const int64_t channelInGroup = planeIndex % outChannelsPerGroup;
    const int64_t planeRest = planeIndex / outChannelsPerGroup;
    const int64_t batch = planeRest % batchCount;
    const int64_t group = planeRest / batchCount;

    const int64_t outColumn = spatialIndex % outWidth;
    const int64_t outRow = spatialIndex / outWidth;

    // The weight tensor is [groupCount * outChannelsPerGroup][inChannelsPerGroup][R][S], so
    // its leading index is the *global* output channel -- the same flattening the CPU
    // reference uses when it computes wIdx = gIdx * yChannelsPerGroup + kIdx.
    const int64_t outChannel = (group * outChannelsPerGroup) + channelInGroup;
    const int64_t baseInChannel = group * inChannelsPerGroup;

    const HkpElement* xPlane = x + (batch * xStrideBatch) + (baseInChannel * xStrideChannel);
    const HkpElement* wPlane = w + (outChannel * wStrideOutChannel);

    // Accumulate in float whatever the storage type is: a _Float16 accumulator loses far
    // more against the float CPU reference than the extra registers cost.
    float accumulator = 0.0f;

    for(int64_t channel = 0; channel < inChannelsPerGroup; ++channel)
    {
        const HkpElement* xChannel = xPlane + (channel * xStrideChannel);
        const HkpElement* wChannel = wPlane + (channel * wStrideInChannel);

        for(int64_t filterRow = 0; filterRow < filterHeight; ++filterRow)
        {
            // Cross-correlation: the filter is read in its natural order and the input
            // window advances with the output index.  CONVOLUTION -- the flipped-kernel
            // enumerator of ConvMode -- is a different operation and is rejected upstream,
            // not approximated here.
            const int64_t inRow
                = (outRow * strideHeight) + (filterRow * dilationHeight) - prePadHeight;
            if(inRow < 0 || inRow >= inHeight)
            {
                continue;
            }

            for(int64_t filterColumn = 0; filterColumn < filterWidth; ++filterColumn)
            {
                const int64_t inColumn
                    = (outColumn * strideWidth) + (filterColumn * dilationWidth) - prePadWidth;
                if(inColumn < 0 || inColumn >= inWidth)
                {
                    continue;
                }

                const float xValue = static_cast<float>(
                    xChannel[(inRow * xStrideHeight) + (inColumn * xStrideWidth)]);
                const float wValue = static_cast<float>(
                    wChannel[(filterRow * wStrideHeight) + (filterColumn * wStrideWidth)]);
                accumulator += xValue * wValue;
            }
        }
    }

    // The convolution's own output tensor is virtual, so it is never stored; the activation
    // consumes the accumulator directly.  That elides one round-trip through the
    // intermediate's declared element type, which is why the engine may only claim graphs
    // whose convolution output is declared FLOAT -- the type the accumulator already has.
    const float activated = HKP_ACTIVATION_FN(HKP_ACTIVATION)(
        accumulator, reluLowerClip, reluUpperClip, reluLowerClipSlope);

    y[(batch * yStrideBatch) + (outChannel * yStrideChannel) + (outRow * yStrideHeight)
      + (outColumn * yStrideWidth)]
        = static_cast<HkpElement>(activated);
}
