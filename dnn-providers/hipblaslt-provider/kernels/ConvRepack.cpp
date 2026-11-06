// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Repack kernels of the explicit-GEMM convolution engines: im2col, col2im and a strided N-D
// copy. They move a convolution's operands into and out of the layouts a single BLAS GEMM
// reads, for the shapes no direct GEMM covers.
//
// One HIPRTC program holds every entry point, and every shape is a runtime argument
// (ConvRepackArgs.h), so one compile per device serves every problem. im2col and the copy
// move bytes and have one entry per element size; col2im and the cast convert fp32 sums and have
// one entry per output type. Grid-stride loops with 64-bit indexing throughout; no vectorization
// or LDS tiling yet -- those come one at a time, each with a measured gain.

#include "ConvRepackArgs.h"

namespace
{

__device__ inline long long firstThread()
{
    return static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
           + static_cast<long long>(threadIdx.x);
}

__device__ inline long long threadStride()
{
    return static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
}

template <typename Bits>
__device__ void stridedCopy(const StridedCopyArgs& args)
{
    const auto* src = static_cast<const Bits*>(args.src);
    auto* dst = static_cast<Bits*>(args.dst);
    for(long long index = firstThread(); index < args.total; index += threadStride())
    {
        long long rest = index;
        long long srcOffset = 0;
        long long dstOffset = 0;
        for(int axis = args.rank - 1; axis >= 0; --axis)
        {
            const long long coordinate = rest % args.dims[axis];
            rest /= args.dims[axis];
            srcOffset += coordinate * args.srcStrides[axis];
            dstOffset += coordinate * args.dstStrides[axis];
        }
        dst[dstOffset] = src[srcOffset];
    }
}

template <typename OutputType>
__device__ void stridedCast(const StridedCastArgs& args)
{
    auto* dst = static_cast<OutputType*>(args.dst);
    for(long long index = firstThread(); index < args.total; index += threadStride())
    {
        long long rest = index;
        long long srcOffset = 0;
        long long dstOffset = 0;
        for(int axis = args.rank - 1; axis >= 0; --axis)
        {
            const long long coordinate = rest % args.dims[axis];
            rest /= args.dims[axis];
            srcOffset += coordinate * args.srcStrides[axis];
            dstOffset += coordinate * args.dstStrides[axis];
        }
        dst[dstOffset] = static_cast<OutputType>(args.src[srcOffset]);
    }
}

__device__ inline long long filterSize(const ConvColumnGeometry& g)
{
    return g.filter[0] * g.filter[1] * g.filter[2];
}

__device__ inline long long outputSize(const ConvColumnGeometry& g)
{
    return g.output[0] * g.output[1] * g.output[2];
}

/// The tap through which filter element @p element of @p axis reads the input.
__device__ inline long long tapOf(const ConvColumnGeometry& g, int axis, long long element)
{
    return g.flipFilter != 0 ? g.filter[axis] - 1 - element : element;
}

/// The column of (group, channel in group, flattened filter element) in the column matrix.
__device__ inline long long
    columnOf(const ConvColumnGeometry& g, long long group, long long channel, long long element)
{
    const long long inGroup = g.channelsInner != 0 ? element * g.cPerGroup + channel
                                                   : channel * filterSize(g) + element;
    return group * g.cPerGroup * filterSize(g) + inGroup;
}

template <typename Bits>
__device__ void im2col(const Im2ColArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const auto* x = static_cast<const Bits*>(args.x);
    auto* col = static_cast<Bits*>(args.col);
    const long long taps = filterSize(g);
    const long long columns = g.groups * g.cPerGroup * taps;
    const long long outputs = outputSize(g);

    for(long long index = firstThread(); index < args.total; index += threadStride())
    {
        const long long column = index % columns;
        const long long row = index / columns;

        const long long group = column / (g.cPerGroup * taps);
        const long long inGroup = column % (g.cPerGroup * taps);
        const long long channel = g.channelsInner != 0 ? inGroup % g.cPerGroup : inGroup / taps;
        const long long element = g.channelsInner != 0 ? inGroup / g.cPerGroup : inGroup % taps;

        const long long n = row / outputs;
        long long position = row % outputs;
        long long rest = element;
        long long offset = n * g.xStrides[0] + (group * g.cPerGroup + channel) * g.xStrides[1];
        bool inside = true;
        for(int axis = CONV_REPACK_SPATIAL_DIMS - 1; axis >= 0; --axis)
        {
            const long long o = position % g.output[axis];
            position /= g.output[axis];
            const long long e = rest % g.filter[axis];
            rest /= g.filter[axis];
            const long long i
                = o * g.stride[axis] - g.prePadding[axis] + tapOf(g, axis, e) * g.dilation[axis];
            inside = inside && i >= 0 && i < g.input[axis];
            offset += i * g.xStrides[2 + axis];
        }
        col[index] = inside ? x[offset] : Bits(0);
    }
}

template <typename OutputType>
__device__ void col2im(const Col2ImArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    auto* x = static_cast<OutputType*>(args.x);
    const long long columns = g.groups * g.cPerGroup * filterSize(g);
    const long long outputs = outputSize(g);
    const long long channels = g.groups * g.cPerGroup;

    for(long long index = firstThread(); index < args.total; index += threadStride())
    {
        // index walks dx as packed (n, c, d, h, w); dx itself is addressed through xStrides.
        long long rest = index;
        long long i[CONV_REPACK_SPATIAL_DIMS];
        for(int axis = CONV_REPACK_SPATIAL_DIMS - 1; axis >= 0; --axis)
        {
            i[axis] = rest % g.input[axis];
            rest /= g.input[axis];
        }
        const long long c = rest % channels;
        const long long n = rest / channels;
        const long long group = c / g.cPerGroup;
        const long long channel = c % g.cPerGroup;

        float sum = 0.0f;
        for(long long e0 = 0; e0 < g.filter[0]; ++e0)
        {
            const long long s0 = i[0] + g.prePadding[0] - tapOf(g, 0, e0) * g.dilation[0];
            if(s0 < 0 || s0 % g.stride[0] != 0 || s0 / g.stride[0] >= g.output[0])
            {
                continue;
            }
            for(long long e1 = 0; e1 < g.filter[1]; ++e1)
            {
                const long long s1 = i[1] + g.prePadding[1] - tapOf(g, 1, e1) * g.dilation[1];
                if(s1 < 0 || s1 % g.stride[1] != 0 || s1 / g.stride[1] >= g.output[1])
                {
                    continue;
                }
                for(long long e2 = 0; e2 < g.filter[2]; ++e2)
                {
                    const long long s2 = i[2] + g.prePadding[2] - tapOf(g, 2, e2) * g.dilation[2];
                    if(s2 < 0 || s2 % g.stride[2] != 0 || s2 / g.stride[2] >= g.output[2])
                    {
                        continue;
                    }
                    const long long position
                        = ((s0 / g.stride[0]) * g.output[1] + s1 / g.stride[1]) * g.output[2]
                          + s2 / g.stride[2];
                    const long long element = (e0 * g.filter[1] + e1) * g.filter[2] + e2;
                    sum += args.col[(n * outputs + position) * columns
                                    + columnOf(g, group, channel, element)];
                }
            }
        }
        const long long offset = n * g.xStrides[0] + c * g.xStrides[1] + i[0] * g.xStrides[2]
                                 + i[1] * g.xStrides[3] + i[2] * g.xStrides[4];
        x[offset] = static_cast<OutputType>(sum);
    }
}

} // namespace

extern "C" __global__ void StridedCopy2(StridedCopyArgs args)
{
    stridedCopy<unsigned short>(args);
}

extern "C" __global__ void StridedCopy4(StridedCopyArgs args)
{
    stridedCopy<unsigned int>(args);
}

extern "C" __global__ void StridedCastFloatToHalf(StridedCastArgs args)
{
    stridedCast<_Float16>(args);
}

extern "C" __global__ void StridedCastFloatToBfloat16(StridedCastArgs args)
{
    stridedCast<__bf16>(args);
}

extern "C" __global__ void Im2Col2(Im2ColArgs args)
{
    im2col<unsigned short>(args);
}

extern "C" __global__ void Im2Col4(Im2ColArgs args)
{
    im2col<unsigned int>(args);
}

extern "C" __global__ void Col2ImFloat(Col2ImArgs args)
{
    col2im<float>(args);
}

extern "C" __global__ void Col2ImHalf(Col2ImArgs args)
{
    col2im<_Float16>(args);
}

extern "C" __global__ void Col2ImBfloat16(Col2ImArgs args)
{
    col2im<__bf16>(args);
}
