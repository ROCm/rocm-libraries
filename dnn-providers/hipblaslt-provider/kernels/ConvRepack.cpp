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
        // Threads walk the column matrix in its own storage order, so writes are contiguous.
        const long long column
            = g.columnsFirst != 0 ? (index / outputs) % columns : index % columns;
        const long long row = g.columnsFirst != 0
                                  ? (index / (outputs * columns)) * outputs + index % outputs
                                  : index / columns;

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
                    const long long column = columnOf(g, group, channel, element);
                    sum += args.col[g.columnsFirst != 0
                                        ? (n * columns + column) * outputs + position
                                        : (n * outputs + position) * columns + column];
                }
            }
        }
        const long long offset = n * g.xStrides[0] + c * g.xStrides[1] + i[0] * g.xStrides[2]
                                 + i[1] * g.xStrides[3] + i[2] * g.xStrides[4];
        x[offset] = static_cast<OutputType>(sum);
    }
}

// ---- Specialized kernels --------------------------------------------------------------------
//
// Chosen by the host (ConvRepackLaunch.hpp) when every offset fits 32 bits and the layouts suit
// them; the generic kernels above serve everything else. Index math is 32-bit and decomposed
// once per row or tile rather than per element.

template <typename Bits, int VEC>
struct VectorOf
{
    using type = Bits __attribute__((ext_vector_type(VEC)));
};

template <typename Bits>
struct VectorOf<Bits, 1>
{
    using type = Bits;
};

__device__ inline int tapIndex(int flip, int extent, int element)
{
    return flip != 0 ? extent - 1 - element : element;
}

/// im2col into channels-last columns in the filter's KRSC order, from an input whose channels are
/// contiguous: `rows` column-matrix rows per block, `lanes` threads per row. A row's threads
/// share its index decomposition and copy VEC channels at a time under a uniform loop over group
/// and tap, so no element pays for a division.
template <typename Bits, int VEC>
__device__ void im2colRows(const Im2ColArgs& args)
{
    using Vector = typename VectorOf<Bits, VEC>::type;
    const ConvColumnGeometry& g = args.geometry;
    const int lanes = args.launch.lanes;
    const int lane = static_cast<int>(threadIdx.x) % lanes;
    const int row
        = static_cast<int>(blockIdx.x) * args.launch.rows + static_cast<int>(threadIdx.x) / lanes;
    const int od = static_cast<int>(g.output[0]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int outputs = od * oh * ow;
    if(row >= static_cast<int>(g.n) * outputs)
    {
        return;
    }
    const int n = row / outputs;
    int position = row - n * outputs;
    const int ox = position % ow;
    position /= ow;
    const int oy = position % oh;
    const int oz = position / oh;

    const int kd = static_cast<int>(g.filter[0]);
    const int kh = static_cast<int>(g.filter[1]);
    const int kw = static_cast<int>(g.filter[2]);
    const int taps = kd * kh * kw;
    const int cg = static_cast<int>(g.cPerGroup);
    const int columns = static_cast<int>(g.groups) * cg * taps;
    const int vectors = cg / VEC;
    const int xs0 = static_cast<int>(g.xStrides[0]);
    const int xs2 = static_cast<int>(g.xStrides[2]);
    const int xs3 = static_cast<int>(g.xStrides[3]);
    const int xs4 = static_cast<int>(g.xStrides[4]);
    const auto* x = static_cast<const Bits*>(args.x) + n * xs0;
    auto* col = static_cast<Bits*>(args.col) + row * columns;

    for(int group = 0; group < static_cast<int>(g.groups); ++group)
    {
        int element = 0;
        for(int ez = 0; ez < kd; ++ez)
        {
            const int iz = oz * static_cast<int>(g.stride[0]) - static_cast<int>(g.prePadding[0])
                           + tapIndex(g.flipFilter, kd, ez) * static_cast<int>(g.dilation[0]);
            for(int ey = 0; ey < kh; ++ey)
            {
                const int iy = oy * static_cast<int>(g.stride[1])
                               - static_cast<int>(g.prePadding[1])
                               + tapIndex(g.flipFilter, kh, ey) * static_cast<int>(g.dilation[1]);
                for(int ex = 0; ex < kw; ++ex, ++element)
                {
                    const int ix
                        = ox * static_cast<int>(g.stride[2]) - static_cast<int>(g.prePadding[2])
                          + tapIndex(g.flipFilter, kw, ex) * static_cast<int>(g.dilation[2]);
                    const bool inside = iz >= 0 && iz < static_cast<int>(g.input[0]) && iy >= 0
                                        && iy < static_cast<int>(g.input[1]) && ix >= 0
                                        && ix < static_cast<int>(g.input[2]);
                    const Bits* source = x + group * cg + iz * xs2 + iy * xs3 + ix * xs4;
                    Bits* target = col + group * cg * taps + element * cg;
                    for(int v = lane; v < vectors; v += lanes)
                    {
                        const Vector value
                            = inside ? *reinterpret_cast<const Vector*>(source + v * VEC)
                                     : Vector(0);
                        *reinterpret_cast<Vector*>(target + v * VEC) = value;
                    }
                }
            }
        }
    }
}

/// im2col into channels-last columns in KRSC order for few channels per tap: a row's `lanes`
/// threads walk its (group, tap, channel) elements in column order, so writes stay contiguous
/// even when a tap holds only a handful of channels (stems).
template <typename Bits>
__device__ void im2colRowsFlat(const Im2ColArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const int lanes = args.launch.lanes;
    const int lane = static_cast<int>(threadIdx.x) % lanes;
    const int row
        = static_cast<int>(blockIdx.x) * args.launch.rows + static_cast<int>(threadIdx.x) / lanes;
    const int od = static_cast<int>(g.output[0]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int outputs = od * oh * ow;
    if(row >= static_cast<int>(g.n) * outputs)
    {
        return;
    }
    const int n = row / outputs;
    int position = row - n * outputs;
    const int ox = position % ow;
    position /= ow;
    const int oy = position % oh;
    const int oz = position / oh;

    const int kh = static_cast<int>(g.filter[1]);
    const int kw = static_cast<int>(g.filter[2]);
    const int kd = static_cast<int>(g.filter[0]);
    const int taps = kd * kh * kw;
    const int cg = static_cast<int>(g.cPerGroup);
    const int columns = static_cast<int>(g.groups) * cg * taps;
    const auto* x = static_cast<const Bits*>(args.x) + n * static_cast<int>(g.xStrides[0]);
    auto* col = static_cast<Bits*>(args.col) + row * columns;
    for(int index = lane; index < columns; index += lanes)
    {
        const int c = index % cg;
        int rest = index / cg;
        const int element = rest % taps;
        const int group = rest / taps;
        const int ex = element % kw;
        const int ey = (element / kw) % kh;
        const int ez = element / (kw * kh);
        const int iz = oz * static_cast<int>(g.stride[0]) - static_cast<int>(g.prePadding[0])
                       + tapIndex(g.flipFilter, kd, ez) * static_cast<int>(g.dilation[0]);
        const int iy = oy * static_cast<int>(g.stride[1]) - static_cast<int>(g.prePadding[1])
                       + tapIndex(g.flipFilter, kh, ey) * static_cast<int>(g.dilation[1]);
        const int ix = ox * static_cast<int>(g.stride[2]) - static_cast<int>(g.prePadding[2])
                       + tapIndex(g.flipFilter, kw, ex) * static_cast<int>(g.dilation[2]);
        const bool inside = iz >= 0 && iz < static_cast<int>(g.input[0]) && iy >= 0
                            && iy < static_cast<int>(g.input[1]) && ix >= 0
                            && ix < static_cast<int>(g.input[2]);
        col[index]
            = inside
                  ? x[group * cg + c + iz * static_cast<int>(g.xStrides[2])
                      + iy * static_cast<int>(g.xStrides[3]) + ix * static_cast<int>(g.xStrides[4])]
                  : Bits(0);
    }
}

/// im2col into channels-first columns from a 2-D channels-first input, as MIOpen's NCHW kernel:
/// one block per (image, channel, tileH x tileW output tile) stages the tile's input halo in LDS
/// once, padding resolved as zeros at load, then every thread emits its output position's taps
/// from LDS. Writes run along output width, so they coalesce.
template <typename Bits>
__device__ void im2colTile(const Im2ColArgs& args)
{
    extern __shared__ unsigned char tileBytes[];
    Bits* halo = reinterpret_cast<Bits*>(tileBytes);
    const ConvColumnGeometry& g = args.geometry;
    const int tileH = args.launch.tileH;
    const int tileW = args.launch.tileW;
    const int haloH = args.launch.haloH;
    const int haloW = args.launch.haloW;
    const int ih = static_cast<int>(g.input[1]);
    const int iw = static_cast<int>(g.input[2]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int sh = static_cast<int>(g.stride[1]);
    const int sw = static_cast<int>(g.stride[2]);
    const int dh = static_cast<int>(g.dilation[1]);
    const int dw = static_cast<int>(g.dilation[2]);
    const int kh = static_cast<int>(g.filter[1]);
    const int kw = static_cast<int>(g.filter[2]);
    const int cg = static_cast<int>(g.cPerGroup);
    const int channels = static_cast<int>(g.groups) * cg;

    int block = static_cast<int>(blockIdx.x);
    const int tilesW = (ow + tileW - 1) / tileW;
    const int tilesH = (oh + tileH - 1) / tileH;
    const int tx = block % tilesW;
    block /= tilesW;
    const int ty = block % tilesH;
    block /= tilesH;
    const int c = block % channels;
    const int n = block / channels;

    const int oy0 = ty * tileH;
    const int ox0 = tx * tileW;
    const int iy0 = oy0 * sh - static_cast<int>(g.prePadding[1]);
    const int ix0 = ox0 * sw - static_cast<int>(g.prePadding[2]);
    const auto* x = static_cast<const Bits*>(args.x) + n * static_cast<int>(g.xStrides[0])
                    + c * static_cast<int>(g.xStrides[1]);
    const int xs3 = static_cast<int>(g.xStrides[3]);
    const int xs4 = static_cast<int>(g.xStrides[4]);
    for(int i = static_cast<int>(threadIdx.x); i < haloH * haloW; i += static_cast<int>(blockDim.x))
    {
        const int ly = i / haloW;
        const int lx = i - ly * haloW;
        const int iy = iy0 + ly;
        const int ix = ix0 + lx;
        halo[i] = iy >= 0 && iy < ih && ix >= 0 && ix < iw ? x[iy * xs3 + ix * xs4] : Bits(0);
    }
    __syncthreads();

    const int taps = kh * kw;
    const int outputs = oh * ow;
    const int columns = channels * taps;
    const int group = c / cg;
    const int inGroup = c - group * cg;
    auto* col = static_cast<Bits*>(args.col) + n * columns * outputs;
    for(int t = static_cast<int>(threadIdx.x); t < tileH * tileW; t += static_cast<int>(blockDim.x))
    {
        const int ly = t / tileW;
        const int lx = t - ly * tileW;
        const int oy = oy0 + ly;
        const int ox = ox0 + lx;
        if(oy >= oh || ox >= ow)
        {
            continue;
        }
        Bits* out = col + oy * ow + ox;
        int element = 0;
        for(int ey = 0; ey < kh; ++ey)
        {
            const int hy = ly * sh + tapIndex(g.flipFilter, kh, ey) * dh;
            for(int ex = 0; ex < kw; ++ex, ++element)
            {
                const int hx = lx * sw + tapIndex(g.flipFilter, kw, ex) * dw;
                const int column = g.channelsInner != 0 ? group * cg * taps + element * cg + inGroup
                                                        : c * taps + element;
                out[column * outputs] = halo[hy * haloW + hx];
            }
        }
    }
}

/// The output positions along one axis that read input position i through tap e: o with
/// o * stride = i + pad - tap * dilation, or -1.
__device__ inline int sourceOf(int i, int pad, int tap, int dilation, int stride, int extent)
{
    const int s = i + pad - tap * dilation;
    if(s < 0)
    {
        return -1;
    }
    int o = s;
    if(stride != 1)
    {
        if(s % stride != 0)
        {
            return -1;
        }
        o = s / stride;
    }
    return o < extent ? o : -1;
}

/// col2im from channels-last columns in KRSC order into an input whose channels are
/// contiguous: `rows` dx positions per block, `lanes` threads each, VEC channels per thread
/// (fp32 vector reads of the column matrix). A gather: fp32 sums, one rounding, no atomics.
template <typename OutputType, int VEC>
__device__ void col2imRows(const Col2ImArgs& args)
{
    using Sums = typename VectorOf<float, VEC>::type;
    const ConvColumnGeometry& g = args.geometry;
    const int lanes = args.launch.lanes;
    const int lane = static_cast<int>(threadIdx.x) % lanes;
    const int row
        = static_cast<int>(blockIdx.x) * args.launch.rows + static_cast<int>(threadIdx.x) / lanes;
    const int id = static_cast<int>(g.input[0]);
    const int ih = static_cast<int>(g.input[1]);
    const int iw = static_cast<int>(g.input[2]);
    const int inputs = id * ih * iw;
    if(row >= static_cast<int>(g.n) * inputs)
    {
        return;
    }
    const int n = row / inputs;
    int position = row - n * inputs;
    const int ix = position % iw;
    position /= iw;
    const int iy = position % ih;
    const int iz = position / ih;

    const int od = static_cast<int>(g.output[0]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int outputs = od * oh * ow;
    const int kd = static_cast<int>(g.filter[0]);
    const int kh = static_cast<int>(g.filter[1]);
    const int kw = static_cast<int>(g.filter[2]);
    const int taps = kd * kh * kw;
    const int cg = static_cast<int>(g.cPerGroup);
    const int channels = static_cast<int>(g.groups) * cg;
    const int columns = channels * taps;
    auto* x = static_cast<OutputType*>(args.x) + n * static_cast<int>(g.xStrides[0])
              + iz * static_cast<int>(g.xStrides[2]) + iy * static_cast<int>(g.xStrides[3])
              + ix * static_cast<int>(g.xStrides[4]);
    const float* col = args.col + n * outputs * columns;

    for(int v = lane; v < channels / VEC; v += lanes)
    {
        const int c = v * VEC;
        const int group = c / cg;
        const int inGroup = c - group * cg;
        const float* groupColumns = col + group * cg * taps + inGroup;
        Sums sum = Sums(0.0f);
        int element = 0;
        for(int ez = 0; ez < kd; ++ez)
        {
            const int sz = sourceOf(iz,
                                    static_cast<int>(g.prePadding[0]),
                                    tapIndex(g.flipFilter, kd, ez),
                                    static_cast<int>(g.dilation[0]),
                                    static_cast<int>(g.stride[0]),
                                    od);
            for(int ey = 0; ey < kh; ++ey)
            {
                const int sy = sourceOf(iy,
                                        static_cast<int>(g.prePadding[1]),
                                        tapIndex(g.flipFilter, kh, ey),
                                        static_cast<int>(g.dilation[1]),
                                        static_cast<int>(g.stride[1]),
                                        oh);
                for(int ex = 0; ex < kw; ++ex, ++element)
                {
                    const int sx = sourceOf(ix,
                                            static_cast<int>(g.prePadding[2]),
                                            tapIndex(g.flipFilter, kw, ex),
                                            static_cast<int>(g.dilation[2]),
                                            static_cast<int>(g.stride[2]),
                                            ow);
                    if(sz < 0 || sy < 0 || sx < 0)
                    {
                        continue;
                    }
                    const int o = (sz * oh + sy) * ow + sx;
                    sum += *reinterpret_cast<const Sums*>(groupColumns + o * columns
                                                          + element * cg);
                }
            }
        }
        if constexpr(VEC == 1)
        {
            x[c] = static_cast<OutputType>(sum);
        }
        else
        {
#pragma unroll
            for(int k = 0; k < VEC; ++k)
            {
                x[c + k] = static_cast<OutputType>(sum[k]);
            }
        }
    }
}

/// col2im from channels-first columns into a channels-first input: one thread per dx element,
/// width fastest, so reads of the column matrix run along output width. A gather as above.
template <typename OutputType>
__device__ void col2imChannelsFirst(const Col2ImArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                      + static_cast<int>(threadIdx.x);
    if(index >= static_cast<int>(args.total))
    {
        return;
    }
    const int id = static_cast<int>(g.input[0]);
    const int ih = static_cast<int>(g.input[1]);
    const int iw = static_cast<int>(g.input[2]);
    const int cg = static_cast<int>(g.cPerGroup);
    const int channels = static_cast<int>(g.groups) * cg;
    int rest = index;
    const int ix = rest % iw;
    rest /= iw;
    const int iy = rest % ih;
    rest /= ih;
    const int iz = rest % id;
    rest /= id;
    const int c = rest % channels;
    const int n = rest / channels;

    const int od = static_cast<int>(g.output[0]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int outputs = od * oh * ow;
    const int kd = static_cast<int>(g.filter[0]);
    const int kh = static_cast<int>(g.filter[1]);
    const int kw = static_cast<int>(g.filter[2]);
    const int taps = kd * kh * kw;
    const int columns = channels * taps;
    const int group = c / cg;
    const int inGroup = c - group * cg;
    const float* col = args.col + n * columns * outputs;

    float sum = 0.0f;
    int element = 0;
    for(int ez = 0; ez < kd; ++ez)
    {
        const int sz = sourceOf(iz,
                                static_cast<int>(g.prePadding[0]),
                                tapIndex(g.flipFilter, kd, ez),
                                static_cast<int>(g.dilation[0]),
                                static_cast<int>(g.stride[0]),
                                od);
        for(int ey = 0; ey < kh; ++ey)
        {
            const int sy = sourceOf(iy,
                                    static_cast<int>(g.prePadding[1]),
                                    tapIndex(g.flipFilter, kh, ey),
                                    static_cast<int>(g.dilation[1]),
                                    static_cast<int>(g.stride[1]),
                                    oh);
            for(int ex = 0; ex < kw; ++ex, ++element)
            {
                const int sx = sourceOf(ix,
                                        static_cast<int>(g.prePadding[2]),
                                        tapIndex(g.flipFilter, kw, ex),
                                        static_cast<int>(g.dilation[2]),
                                        static_cast<int>(g.stride[2]),
                                        ow);
                if(sz < 0 || sy < 0 || sx < 0)
                {
                    continue;
                }
                const int column = g.channelsInner != 0 ? group * cg * taps + element * cg + inGroup
                                                        : c * taps + element;
                sum += col[column * outputs + (sz * oh + sy) * ow + sx];
            }
        }
    }
    auto* x = static_cast<OutputType*>(args.x);
    x[n * static_cast<int>(g.xStrides[0]) + c * static_cast<int>(g.xStrides[1])
      + iz * static_cast<int>(g.xStrides[2]) + iy * static_cast<int>(g.xStrides[3])
      + ix * static_cast<int>(g.xStrides[4])]
        = static_cast<OutputType>(sum);
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

extern "C" __global__ void __launch_bounds__(256) Im2ColRows2V1(Im2ColArgs args)
{
    im2colRows<unsigned short, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColRows2V8(Im2ColArgs args)
{
    im2colRows<unsigned short, 8>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColRows4V1(Im2ColArgs args)
{
    im2colRows<unsigned int, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColRows4V4(Im2ColArgs args)
{
    im2colRows<unsigned int, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColTile2(Im2ColArgs args)
{
    im2colTile<unsigned short>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColTile4(Im2ColArgs args)
{
    im2colTile<unsigned int>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImRowsFloatV1(Col2ImArgs args)
{
    col2imRows<float, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImRowsFloatV4(Col2ImArgs args)
{
    col2imRows<float, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImRowsHalfV1(Col2ImArgs args)
{
    col2imRows<_Float16, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImRowsHalfV4(Col2ImArgs args)
{
    col2imRows<_Float16, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImRowsBfloat16V1(Col2ImArgs args)
{
    col2imRows<__bf16, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImRowsBfloat16V4(Col2ImArgs args)
{
    col2imRows<__bf16, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImChannelsFirstFloat(Col2ImArgs args)
{
    col2imChannelsFirst<float>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImChannelsFirstHalf(Col2ImArgs args)
{
    col2imChannelsFirst<_Float16>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImChannelsFirstBfloat16(Col2ImArgs args)
{
    col2imChannelsFirst<__bf16>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColRowsFlat2(Im2ColArgs args)
{
    im2colRowsFlat<unsigned short>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColRowsFlat4(Im2ColArgs args)
{
    im2colRowsFlat<unsigned int>(args);
}
