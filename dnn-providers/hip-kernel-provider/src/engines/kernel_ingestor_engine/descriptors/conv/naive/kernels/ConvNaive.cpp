// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Naive direct convolution: forward, backward data and backward weights.
//
// The coverage floor of the conv engine group. One thread computes one output element and
// loops over everything that contributes to it, in fp32. It is correct for every layout,
// group count, padding, stride, dilation and 1-D/2-D/3-D shape, and slow; the engines that
// launch it only accept problems small enough for that not to matter.
//
// Compiled by HIPRTC with -DCONV_NAIVE_TYPE=<float|_Float16|__bf16>, the element type of all
// three tensors. Every shape is a runtime argument (ConvNaiveArgs.h), so one compile per
// element type serves every problem.

// Embedded headers are keyed by bare file name (src/cmake/KernelEmbedding.cmake).
#include "ConvNaiveArgs.h"

#ifndef CONV_NAIVE_TYPE
#error "CONV_NAIVE_TYPE must name the element type"
#endif

using ElementType = CONV_NAIVE_TYPE;

namespace
{

__device__ inline float load(const ElementType* data, long long offset)
{
    return static_cast<float>(data[offset]);
}

__device__ inline void store(ElementType* data, long long offset, float value)
{
    data[offset] = static_cast<ElementType>(value);
}

/// The filter element a tap reads: the tap itself for cross-correlation, its reflection for
/// convolution.
__device__ inline long long filterIndex(const ConvNaiveArgs& args, int axis, long long tap)
{
    return args.flipFilter != 0 ? args.filter[axis] - 1 - tap : tap;
}

__device__ inline long long firstThread()
{
    return static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
           + static_cast<long long>(threadIdx.x);
}

__device__ inline long long threadStride()
{
    return static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
}

} // namespace

/// y[n, k, o] = sum over c in k's group and every tap f of x[n, c, o * stride - pad + f * dil]
/// times w[k, c, f]. One thread per y element.
extern "C" __global__ void ConvNaiveFwd(ConvNaiveArgs args)
{
    const auto* x = static_cast<const ElementType*>(args.x);
    const auto* w = static_cast<const ElementType*>(args.w);
    auto* y = static_cast<ElementType*>(args.y);

    const long long cPerGroup = args.c / args.groups;
    const long long kPerGroup = args.k / args.groups;

    for(long long index = firstThread(); index < args.total; index += threadStride())
    {
        long long rest = index;
        const long long ow = rest % args.outSpatial[2];
        rest /= args.outSpatial[2];
        const long long oh = rest % args.outSpatial[1];
        rest /= args.outSpatial[1];
        const long long od = rest % args.outSpatial[0];
        rest /= args.outSpatial[0];
        const long long k = rest % args.k;
        const long long n = rest / args.k;

        const long long firstChannel = (k / kPerGroup) * cPerGroup;

        float accumulator = 0.0f;
        for(long long c = 0; c < cPerGroup; ++c)
        {
            const long long xChannel = n * args.xStrides[0] + (firstChannel + c) * args.xStrides[1];
            const long long wChannel = k * args.wStrides[0] + c * args.wStrides[1];
            for(long long fd = 0; fd < args.filter[0]; ++fd)
            {
                const long long id
                    = od * args.stride[0] - args.prePadding[0] + fd * args.dilation[0];
                if(id < 0 || id >= args.inSpatial[0])
                {
                    continue;
                }
                for(long long fh = 0; fh < args.filter[1]; ++fh)
                {
                    const long long ih
                        = oh * args.stride[1] - args.prePadding[1] + fh * args.dilation[1];
                    if(ih < 0 || ih >= args.inSpatial[1])
                    {
                        continue;
                    }
                    for(long long fw = 0; fw < args.filter[2]; ++fw)
                    {
                        const long long iw
                            = ow * args.stride[2] - args.prePadding[2] + fw * args.dilation[2];
                        if(iw < 0 || iw >= args.inSpatial[2])
                        {
                            continue;
                        }
                        const long long xOffset = xChannel + id * args.xStrides[2]
                                                  + ih * args.xStrides[3] + iw * args.xStrides[4];
                        const long long wOffset = wChannel
                                                  + filterIndex(args, 0, fd) * args.wStrides[2]
                                                  + filterIndex(args, 1, fh) * args.wStrides[3]
                                                  + filterIndex(args, 2, fw) * args.wStrides[4];
                        accumulator += load(x, xOffset) * load(w, wOffset);
                    }
                }
            }
        }

        store(y,
              n * args.yStrides[0] + k * args.yStrides[1] + od * args.yStrides[2]
                  + oh * args.yStrides[3] + ow * args.yStrides[4],
              accumulator);
    }
}

/// dx[n, c, i] = sum over k in c's group and every tap f with (i + pad - f * dil) a multiple
/// of stride, of dy[n, k, (i + pad - f * dil) / stride] times w[k, c, f]. One thread per dx
/// element, so every dx element is written, including those no output reaches.
extern "C" __global__ void ConvNaiveBwd(ConvNaiveArgs args)
{
    auto* dx = static_cast<ElementType*>(args.x);
    const auto* w = static_cast<const ElementType*>(args.w);
    const auto* dy = static_cast<const ElementType*>(args.y);

    const long long cPerGroup = args.c / args.groups;
    const long long kPerGroup = args.k / args.groups;

    for(long long index = firstThread(); index < args.total; index += threadStride())
    {
        long long rest = index;
        const long long iw = rest % args.inSpatial[2];
        rest /= args.inSpatial[2];
        const long long ih = rest % args.inSpatial[1];
        rest /= args.inSpatial[1];
        const long long id = rest % args.inSpatial[0];
        rest /= args.inSpatial[0];
        const long long c = rest % args.c;
        const long long n = rest / args.c;

        const long long group = c / cPerGroup;
        const long long localChannel = c - group * cPerGroup;

        float accumulator = 0.0f;
        for(long long kLocal = 0; kLocal < kPerGroup; ++kLocal)
        {
            const long long k = group * kPerGroup + kLocal;
            const long long dyChannel = n * args.yStrides[0] + k * args.yStrides[1];
            const long long wChannel = k * args.wStrides[0] + localChannel * args.wStrides[1];
            for(long long fd = 0; fd < args.filter[0]; ++fd)
            {
                const long long td = id + args.prePadding[0] - fd * args.dilation[0];
                if(td < 0 || td % args.stride[0] != 0 || td / args.stride[0] >= args.outSpatial[0])
                {
                    continue;
                }
                const long long od = td / args.stride[0];
                for(long long fh = 0; fh < args.filter[1]; ++fh)
                {
                    const long long th = ih + args.prePadding[1] - fh * args.dilation[1];
                    if(th < 0 || th % args.stride[1] != 0
                       || th / args.stride[1] >= args.outSpatial[1])
                    {
                        continue;
                    }
                    const long long oh = th / args.stride[1];
                    for(long long fw = 0; fw < args.filter[2]; ++fw)
                    {
                        const long long tw = iw + args.prePadding[2] - fw * args.dilation[2];
                        if(tw < 0 || tw % args.stride[2] != 0
                           || tw / args.stride[2] >= args.outSpatial[2])
                        {
                            continue;
                        }
                        const long long ow = tw / args.stride[2];
                        const long long dyOffset = dyChannel + od * args.yStrides[2]
                                                   + oh * args.yStrides[3] + ow * args.yStrides[4];
                        const long long wOffset = wChannel
                                                  + filterIndex(args, 0, fd) * args.wStrides[2]
                                                  + filterIndex(args, 1, fh) * args.wStrides[3]
                                                  + filterIndex(args, 2, fw) * args.wStrides[4];
                        accumulator += load(dy, dyOffset) * load(w, wOffset);
                    }
                }
            }
        }

        store(dx,
              n * args.xStrides[0] + c * args.xStrides[1] + id * args.xStrides[2]
                  + ih * args.xStrides[3] + iw * args.xStrides[4],
              accumulator);
    }
}

/// The backward-weights block: CONV_NAIVE_WRW_BLOCK threads share one dw element's reduction.
#define CONV_NAIVE_WRW_BLOCK 256

/// dw[k, c, f] = sum over n and every output position o of x[n, c', o * stride - pad + t * dil]
/// times dy[n, k, o], where c' is c's channel within k's group and t is the tap that reads
/// filter element f. One block per dw element: thread j sums the output positions (n, o) whose
/// flat index is j modulo the block, then the block adds the threads' sums in a fixed tree, so
/// the result is deterministic and a long reduction spreads over the block's lanes.
extern "C" __global__ void __launch_bounds__(CONV_NAIVE_WRW_BLOCK) ConvNaiveWrw(ConvNaiveArgs args)
{
    const auto* x = static_cast<const ElementType*>(args.x);
    auto* dw = static_cast<ElementType*>(args.w);
    const auto* dy = static_cast<const ElementType*>(args.y);
    __shared__ float sums[CONV_NAIVE_WRW_BLOCK];

    const long long cPerGroup = args.c / args.groups;
    const long long kPerGroup = args.k / args.groups;
    const long long positions
        = args.n * args.outSpatial[0] * args.outSpatial[1] * args.outSpatial[2];
    const int lane = static_cast<int>(threadIdx.x);

    for(long long index = blockIdx.x; index < args.total; index += gridDim.x)
    {
        long long rest = index;
        const long long fw = rest % args.filter[2];
        rest /= args.filter[2];
        const long long fh = rest % args.filter[1];
        rest /= args.filter[1];
        const long long fd = rest % args.filter[0];
        rest /= args.filter[0];
        const long long localChannel = rest % cPerGroup;
        const long long k = rest / cPerGroup;

        const long long c = (k / kPerGroup) * cPerGroup + localChannel;
        // filterIndex is its own inverse: the tap that reads element f is filterIndex(f).
        const long long td = filterIndex(args, 0, fd);
        const long long th = filterIndex(args, 1, fh);
        const long long tw = filterIndex(args, 2, fw);

        float accumulator = 0.0f;
        for(long long p = lane; p < positions; p += CONV_NAIVE_WRW_BLOCK)
        {
            long long at = p;
            const long long ow = at % args.outSpatial[2];
            at /= args.outSpatial[2];
            const long long oh = at % args.outSpatial[1];
            at /= args.outSpatial[1];
            const long long od = at % args.outSpatial[0];
            const long long n = at / args.outSpatial[0];
            const long long id = od * args.stride[0] - args.prePadding[0] + td * args.dilation[0];
            const long long ih = oh * args.stride[1] - args.prePadding[1] + th * args.dilation[1];
            const long long iw = ow * args.stride[2] - args.prePadding[2] + tw * args.dilation[2];
            if(id < 0 || id >= args.inSpatial[0] || ih < 0 || ih >= args.inSpatial[1] || iw < 0
               || iw >= args.inSpatial[2])
            {
                continue;
            }
            const long long xOffset = n * args.xStrides[0] + c * args.xStrides[1]
                                      + id * args.xStrides[2] + ih * args.xStrides[3]
                                      + iw * args.xStrides[4];
            const long long dyOffset = n * args.yStrides[0] + k * args.yStrides[1]
                                       + od * args.yStrides[2] + oh * args.yStrides[3]
                                       + ow * args.yStrides[4];
            accumulator += load(x, xOffset) * load(dy, dyOffset);
        }

        // The threads' sums in a fixed tree: lane j adds lane j + half at each level.
        sums[lane] = accumulator;
        __syncthreads();
        for(int half = CONV_NAIVE_WRW_BLOCK / 2; half > 0; half /= 2)
        {
            if(lane < half)
            {
                sums[lane] += sums[lane + half];
            }
            __syncthreads();
        }
        if(lane == 0)
        {
            store(dw,
                  k * args.wStrides[0] + localChannel * args.wStrides[1] + fd * args.wStrides[2]
                      + fh * args.wStrides[3] + fw * args.wStrides[4],
                  sums[0]);
        }
        __syncthreads(); // sums is reused by the block's next element
    }
}
