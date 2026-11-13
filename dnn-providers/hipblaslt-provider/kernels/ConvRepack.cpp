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

/// A strided copy that transposes: src is unit-stride along srcUnitAxis, dst along dstUnitAxis
/// (NCHW to NHWC staging, say). One block per 32 x 32 tile of those two axes and per entry of the
/// others, staged through LDS so both the loads and the stores run along a unit stride.
template <typename Bits>
__device__ void stridedTranspose(const StridedCopyArgs& args)
{
    constexpr int TILE = 32;
    constexpr int ROWS = 8;
    __shared__ Bits tile[TILE][TILE + 1];
    const int a = args.srcUnitAxis;
    const int b = args.dstUnitAxis;
    const long long extentA = args.dims[a];
    const long long extentB = args.dims[b];
    const long long tilesA = (extentA + TILE - 1) / TILE;
    const long long tilesB = (extentB + TILE - 1) / TILE;
    long long block = blockIdx.x;
    const long long a0 = (block % tilesA) * TILE;
    block /= tilesA;
    const long long b0 = (block % tilesB) * TILE;
    block /= tilesB;
    long long srcBase = 0;
    long long dstBase = 0;
    for(int axis = args.rank - 1; axis >= 0; --axis)
    {
        if(axis == a || axis == b)
        {
            continue;
        }
        const long long coordinate = block % args.dims[axis];
        block /= args.dims[axis];
        srcBase += coordinate * args.srcStrides[axis];
        dstBase += coordinate * args.dstStrides[axis];
    }
    const auto* src = static_cast<const Bits*>(args.src) + srcBase;
    auto* dst = static_cast<Bits*>(args.dst) + dstBase;
    const int tx = static_cast<int>(threadIdx.x) % TILE;
    const int ty = static_cast<int>(threadIdx.x) / TILE;
    for(int j = ty; j < TILE; j += ROWS)
    {
        const long long ia = a0 + tx;
        const long long ib = b0 + j;
        if(ia < extentA && ib < extentB)
        {
            tile[j][tx] = src[ia + ib * args.srcStrides[b]];
        }
    }
    __syncthreads();
    for(int j = ty; j < TILE; j += ROWS)
    {
        const long long ib = b0 + tx;
        const long long ia = a0 + j;
        if(ia < extentA && ib < extentB)
        {
            dst[ib + ia * args.dstStrides[a]] = tile[tx][j];
        }
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
        float sum = args.src[srcOffset];
        for(int batch = 1; batch < args.batches; ++batch)
        {
            sum += args.src[srcOffset + batch * args.batchStride];
        }
        dst[dstOffset] = static_cast<OutputType>(sum);
    }
}

/// The cast of many partials into few outputs: one 64-thread block per output element, each
/// thread summing every 64th partial in order, then a fixed tree in LDS. Deterministic (the same
/// order on every run and GPU), one rounding; for sums of many partials, which the per-element
/// loop above runs on a handful of threads.
template <typename OutputType>
__device__ void stridedCastWide(const StridedCastArgs& args)
{
    constexpr int LANES = 64;
    __shared__ float partial[LANES];
    const long long index = blockIdx.x;
    if(index >= args.total)
    {
        return;
    }
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
    const int lane = static_cast<int>(threadIdx.x);
    float sum = 0.0f;
    for(int batch = lane; batch < args.batches; batch += LANES)
    {
        sum += args.src[srcOffset + batch * args.batchStride];
    }
    partial[lane] = sum;
    __syncthreads();
#pragma unroll
    for(int width = LANES / 2; width > 0; width /= 2)
    {
        if(lane < width)
        {
            partial[lane] += partial[lane + width];
        }
        __syncthreads();
    }
    if(lane == 0)
    {
        static_cast<OutputType*>(args.dst)[dstOffset] = static_cast<OutputType>(partial[0]);
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

/// col2im of a filter whose taps never overlap (stride at least the dilated filter extent on
/// every axis, a single tap included): each dx element is at most one column entry, in dx's
/// type, or 0 where no output reads it. The one candidate output along an axis is
/// (i + pad) / stride, the tap the remainder over the dilation. Any layout; 64-bit offsets.
template <typename Bits>
__device__ void col2imDisjoint(const Col2ImArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const auto* col = reinterpret_cast<const Bits*>(args.col);
    auto* x = static_cast<Bits*>(args.x);
    const long long channels = g.groups * g.cPerGroup;
    const long long taps = g.filter[0] * g.filter[1] * g.filter[2];
    const long long columns = channels * taps;
    const long long outputs = g.output[0] * g.output[1] * g.output[2];
    for(long long index = firstThread(); index < args.total; index += threadStride())
    {
        long long rest = index;
        long long position[CONV_REPACK_SPATIAL_DIMS];
        for(int axis = CONV_REPACK_SPATIAL_DIMS - 1; axis >= 0; --axis)
        {
            position[axis] = rest % g.input[axis];
            rest /= g.input[axis];
        }
        const long long c = rest % channels;
        const long long n = rest / channels;
        long long o = 0;
        long long element = 0;
        bool read = true;
        for(int axis = 0; axis < CONV_REPACK_SPATIAL_DIMS; ++axis)
        {
            const long long s = position[axis] + g.prePadding[axis];
            const long long coordinate = s >= 0 ? s / g.stride[axis] : 0;
            const long long offset = s - coordinate * g.stride[axis];
            const long long tap = offset / g.dilation[axis];
            read = read && s >= 0 && coordinate < g.output[axis] && offset % g.dilation[axis] == 0
                   && tap < g.filter[axis];
            o = o * g.output[axis] + coordinate;
            element
                = element * g.filter[axis] + (g.flipFilter != 0 ? g.filter[axis] - 1 - tap : tap);
        }
        long long target = n * g.xStrides[0] + c * g.xStrides[1];
        for(int axis = 0; axis < CONV_REPACK_SPATIAL_DIMS; ++axis)
        {
            target += position[axis] * g.xStrides[2 + axis];
        }
        const long long group = c / g.cPerGroup;
        const long long column = g.channelsInner != 0 ? (group * taps + element) * g.cPerGroup
                                                            + (c - group * g.cPerGroup)
                                                      : c * taps + element;
        x[target] = read ? col[g.columnsFirst != 0 ? (n * columns + column) * outputs + o
                                                   : (n * outputs + o) * columns + column]
                         : Bits(0);
    }
}

/// The strided transpose with every offset in 32 bits (args.narrow): the block's tile and base
/// found once with 32-bit divisions, and 16-bit elements moved in pairs -- a 64 x 64 tile whose
/// rows load and store as 4-byte pairs where aligned, one element at a time where not. Moves
/// bytes: the generic transpose's result bit for bit.
template <typename Bits>
__device__ void stridedTransposeNarrow(const StridedCopyArgs& args)
{
    constexpr int PAIR = sizeof(Bits) == 2 ? 2 : 1;
    constexpr int TILE = 32 * PAIR;
    constexpr int LANES = TILE / PAIR; // threads along a tile row
    constexpr int ROWS = 256 / LANES;
    __shared__ Bits tile[TILE][TILE + 1];
    const int a = args.srcUnitAxis;
    const int b = args.dstUnitAxis;
    const int extentA = static_cast<int>(args.dims[a]);
    const int extentB = static_cast<int>(args.dims[b]);
    const int tilesA = (extentA + TILE - 1) / TILE;
    const int tilesB = (extentB + TILE - 1) / TILE;
    // Tiles in diagonal order within each (tilesA x tilesB) plane: consecutive blocks start
    // their rows in different memory partitions instead of all a large stride apart.
    int block = static_cast<int>(blockIdx.x);
    const int tilesPlane = tilesA * tilesB;
    const int inPlane = block % tilesPlane;
    block /= tilesPlane;
    const int tb = inPlane % tilesB;
    const int ta = (inPlane / tilesB + tb) % tilesA;
    const int a0 = ta * TILE;
    const int b0 = tb * TILE;
    int srcBase = 0;
    int dstBase = 0;
    for(int axis = args.rank - 1; axis >= 0; --axis)
    {
        if(axis == a || axis == b)
        {
            continue;
        }
        const int extent = static_cast<int>(args.dims[axis]);
        const int coordinate = block % extent;
        block /= extent;
        srcBase += coordinate * static_cast<int>(args.srcStrides[axis]);
        dstBase += coordinate * static_cast<int>(args.dstStrides[axis]);
    }
    const auto* src = static_cast<const Bits*>(args.src) + srcBase;
    auto* dst = static_cast<Bits*>(args.dst) + dstBase;
    const int srcB = static_cast<int>(args.srcStrides[b]);
    const int dstA = static_cast<int>(args.dstStrides[a]);
    const int tx = static_cast<int>(threadIdx.x) % LANES;
    const int ty = static_cast<int>(threadIdx.x) / LANES;
    const int ia = a0 + PAIR * tx;
    for(int j = ty; j < TILE; j += ROWS)
    {
        const int ib = b0 + j;
        if(ib >= extentB)
        {
            continue;
        }
        const Bits* from = src + ia + ib * srcB;
        if constexpr(PAIR == 2)
        {
            if(ia + 1 < extentA && (reinterpret_cast<unsigned long long>(from) & 3) == 0)
            {
                const unsigned int pair = *reinterpret_cast<const unsigned int*>(from);
                tile[j][PAIR * tx] = static_cast<Bits>(pair & 0xffffu);
                tile[j][PAIR * tx + 1] = static_cast<Bits>(pair >> 16);
                continue;
            }
        }
#pragma unroll
        for(int q = 0; q < PAIR; ++q)
        {
            if(ia + q < extentA)
            {
                tile[j][PAIR * tx + q] = from[q];
            }
        }
    }
    __syncthreads();
    const int ib = b0 + PAIR * tx;
    for(int j = ty; j < TILE; j += ROWS)
    {
        const int ja = a0 + j;
        if(ja >= extentA)
        {
            continue;
        }
        Bits* to = dst + ib + ja * dstA;
        if constexpr(PAIR == 2)
        {
            if(ib + 1 < extentB && (reinterpret_cast<unsigned long long>(to) & 3) == 0)
            {
                *reinterpret_cast<unsigned int*>(to)
                    = static_cast<unsigned int>(tile[PAIR * tx][j])
                      | static_cast<unsigned int>(tile[PAIR * tx + 1][j]) << 16;
                continue;
            }
        }
#pragma unroll
        for(int q = 0; q < PAIR; ++q)
        {
            if(ib + q < extentB)
            {
                to[q] = tile[PAIR * tx + q][j];
            }
        }
    }
}

#include "ConvRepackDisjoint.h"

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

/// One column-matrix row of channels-last KRSC im2col for few channels per tap, written to
/// @p target: `lanes` threads, lane @p lane taking every lanes-th (group, tap) and copying its
/// channels. The tap is decomposed once, and consecutive lanes write consecutive taps.
template <typename Bits>
__device__ void im2colTapRow(const Im2ColArgs& args, int row, int lane, int lanes, Bits* target)
{
    const ConvColumnGeometry& g = args.geometry;
    const int od = static_cast<int>(g.output[0]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int outputs = od * oh * ow;
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
    const int groupTaps = static_cast<int>(g.groups) * taps;
    const int z0 = oz * static_cast<int>(g.stride[0]) - static_cast<int>(g.prePadding[0]);
    const int y0 = oy * static_cast<int>(g.stride[1]) - static_cast<int>(g.prePadding[1]);
    const int x0 = ox * static_cast<int>(g.stride[2]) - static_cast<int>(g.prePadding[2]);
    const auto* x = static_cast<const Bits*>(args.x) + n * static_cast<int>(g.xStrides[0]);
    for(int index = lane; index < groupTaps; index += lanes)
    {
        const int group = index / taps;
        const int element = index - group * taps;
        const int ex = element % kw;
        const int ey = (element / kw) % kh;
        const int ez = element / (kw * kh);
        const int iz = z0 + tapIndex(g.flipFilter, kd, ez) * static_cast<int>(g.dilation[0]);
        const int iy = y0 + tapIndex(g.flipFilter, kh, ey) * static_cast<int>(g.dilation[1]);
        const int ix = x0 + tapIndex(g.flipFilter, kw, ex) * static_cast<int>(g.dilation[2]);
        const bool inside = iz >= 0 && iz < static_cast<int>(g.input[0]) && iy >= 0
                            && iy < static_cast<int>(g.input[1]) && ix >= 0
                            && ix < static_cast<int>(g.input[2]);
        const Bits* source = x + group * cg + iz * static_cast<int>(g.xStrides[2])
                             + iy * static_cast<int>(g.xStrides[3])
                             + ix * static_cast<int>(g.xStrides[4]);
        Bits* out = target + index * cg;
        for(int c = 0; c < cg; ++c)
        {
            out[c] = inside ? source[c] : Bits(0);
        }
    }
}

/// im2col into channels-last columns in KRSC order for few channels per tap (stems): `rows`
/// rows per block, `lanes` threads per row, each row through im2colTapRow straight to memory.
template <typename Bits>
__device__ void im2colTaps(const Im2ColArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const int lanes = args.launch.lanes;
    const int row
        = static_cast<int>(blockIdx.x) * args.launch.rows + static_cast<int>(threadIdx.x) / lanes;
    if(row >= static_cast<int>(g.n * g.output[0] * g.output[1] * g.output[2]))
    {
        return;
    }
    const int columns
        = static_cast<int>(g.groups * g.filter[0] * g.filter[1] * g.filter[2] * g.cPerGroup);
    im2colTapRow(args,
                 row,
                 static_cast<int>(threadIdx.x) % lanes,
                 lanes,
                 static_cast<Bits*>(args.col) + row * columns);
}

/// The per-tap im2col with its writes widened: a block builds `rows` consecutive column-matrix
/// rows in LDS, a contiguous span of the column matrix, then stores the span in 16-byte vectors.
/// Sub-dword channel stores, which bound the per-tap kernel, become full-width ones. The host
/// picks `rows` so every span starts 16-byte aligned.
template <typename Bits>
__device__ void im2colTapsStaged(const Im2ColArgs& args)
{
    extern __shared__ unsigned char stagedBytes[];
    Bits* staged = reinterpret_cast<Bits*>(stagedBytes);
    const ConvColumnGeometry& g = args.geometry;
    const int lanes = args.launch.lanes;
    const int rowsPerBlock = args.launch.rows;
    const int totalRows = static_cast<int>(g.n * g.output[0] * g.output[1] * g.output[2]);
    const int columns
        = static_cast<int>(g.groups * g.filter[0] * g.filter[1] * g.filter[2] * g.cPerGroup);
    const int firstRow = static_cast<int>(blockIdx.x) * rowsPerBlock;
    const int rows = min(rowsPerBlock, totalRows - firstRow);
    const int tid = static_cast<int>(threadIdx.x);
    for(int r = tid / lanes; r < rows; r += static_cast<int>(blockDim.x) / lanes)
    {
        im2colTapRow(args, firstRow + r, tid % lanes, lanes, staged + r * columns);
    }
    __syncthreads();

    using Vector = unsigned int __attribute__((ext_vector_type(4)));
    constexpr int PER_VECTOR = static_cast<int>(sizeof(Vector) / sizeof(Bits));
    const int elements = rows * columns;
    const int vectors = elements / PER_VECTOR;
    auto* col = static_cast<Bits*>(args.col) + firstRow * columns;
    for(int v = tid; v < vectors; v += static_cast<int>(blockDim.x))
    {
        reinterpret_cast<Vector*>(col)[v] = reinterpret_cast<const Vector*>(staged)[v];
    }
    for(int e = vectors * PER_VECTOR + tid; e < elements; e += static_cast<int>(blockDim.x))
    {
        col[e] = staged[e];
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

    // A block takes channelsPerBlock channels of one tile (several when a whole output plane is
    // smaller than the block), each with its own halo.
    const int perBlock = max(1, args.launch.channelsPerBlock);
    int block = static_cast<int>(blockIdx.x);
    const int tilesW = (ow + tileW - 1) / tileW;
    const int tilesH = (oh + tileH - 1) / tileH;
    const int chunks = (channels + perBlock - 1) / perBlock;
    const int tx = block % tilesW;
    block /= tilesW;
    const int ty = block % tilesH;
    block /= tilesH;
    const int c0 = (block % chunks) * perBlock;
    const int n = block / chunks;

    const int oy0 = ty * tileH;
    const int ox0 = tx * tileW;
    const int iy0 = oy0 * sh - static_cast<int>(g.prePadding[1]);
    const int ix0 = ox0 * sw - static_cast<int>(g.prePadding[2]);
    const auto* x = static_cast<const Bits*>(args.x) + n * static_cast<int>(g.xStrides[0]);
    const int xs1 = static_cast<int>(g.xStrides[1]);
    const int xs3 = static_cast<int>(g.xStrides[3]);
    const int xs4 = static_cast<int>(g.xStrides[4]);
    const int haloSize = haloH * haloW;
    for(int i = static_cast<int>(threadIdx.x); i < perBlock * haloSize;
        i += static_cast<int>(blockDim.x))
    {
        const int kc = i / haloSize;
        const int within = i - kc * haloSize;
        const int ly = within / haloW;
        const int lx = within - ly * haloW;
        const int iy = iy0 + ly;
        const int ix = ix0 + lx;
        halo[i] = c0 + kc < channels && iy >= 0 && iy < ih && ix >= 0 && ix < iw
                      ? x[(c0 + kc) * xs1 + iy * xs3 + ix * xs4]
                      : Bits(0);
    }
    __syncthreads();

    const int taps = kh * kw;
    const int outputs = oh * ow;
    const int columns = channels * taps;
    const int tileSize = tileH * tileW;
    auto* col = static_cast<Bits*>(args.col) + n * columns * outputs;
    for(int t = static_cast<int>(threadIdx.x); t < perBlock * tileSize;
        t += static_cast<int>(blockDim.x))
    {
        const int kc = t / tileSize;
        const int within = t - kc * tileSize;
        const int c = c0 + kc;
        const int ly = within / tileW;
        const int lx = within - ly * tileW;
        const int oy = oy0 + ly;
        const int ox = ox0 + lx;
        if(c >= channels || oy >= oh || ox >= ow)
        {
            continue;
        }
        const int group = c / cg;
        const int inGroup = c - group * cg;
        const Bits* plane = halo + kc * haloSize;
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
                out[column * outputs] = plane[hy * haloW + hx];
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

/// col2im from channels-last columns in KRSC order into a channels-contiguous dx for groups of
/// few channels, where adjacent groups' columns sit a whole cg x taps block apart. One block per
/// (image, dx row, `rows` dx positions along width, `groupsPerBlock` groups): for each filter row
/// it stages the kw x cg column runs those positions read, for every output position they read
/// along width, in LDS with coalesced loads, then each thread sums its VEC channels from LDS.
/// A gather as above: fp32 sums, one rounding, no atomics.
template <typename OutputType, int VEC>
__device__ void col2imGroups(const Col2ImArgs& args)
{
    using Sums = typename VectorOf<float, VEC>::type;
    extern __shared__ unsigned char stageBytes[];
    float* stage = reinterpret_cast<float*>(stageBytes);
    const ConvColumnGeometry& g = args.geometry;
    const int lanes = args.launch.lanes;
    const int strip = args.launch.rows;
    const int groupsPerBlock = args.launch.groupsPerBlock;
    const int id = static_cast<int>(g.input[0]);
    const int ih = static_cast<int>(g.input[1]);
    const int iw = static_cast<int>(g.input[2]);
    const int od = static_cast<int>(g.output[0]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int kd = static_cast<int>(g.filter[0]);
    const int kh = static_cast<int>(g.filter[1]);
    const int kw = static_cast<int>(g.filter[2]);
    const int taps = kd * kh * kw;
    const int cg = static_cast<int>(g.cPerGroup);
    const int groups = static_cast<int>(g.groups);
    const int columns = groups * cg * taps;
    const int padW = static_cast<int>(g.prePadding[2]);
    const int sw = static_cast<int>(g.stride[2]);
    const int dw = static_cast<int>(g.dilation[2]);

    int block = static_cast<int>(blockIdx.x);
    const int chunks = (groups + groupsPerBlock - 1) / groupsPerBlock;
    const int strips = (iw + strip - 1) / strip;
    const int g0 = (block % chunks) * groupsPerBlock;
    block /= chunks;
    const int ix0 = (block % strips) * strip;
    block /= strips;
    const int iy = block % ih;
    block /= ih;
    const int iz = block % id;
    const int n = block / id;

    // The output positions along width any tap of the strip reads.
    const int lowest = ix0 + padW - (kw - 1) * dw;
    const int highest = ix0 + strip - 1 + padW;
    const int oxLo = lowest <= 0 ? 0 : (lowest + sw - 1) / sw;
    const int oxHi = highest < 0 ? -1 : min(ow - 1, highest / sw);
    const int width = oxHi - oxLo + 1;
    const int chunkGroups = min(groupsPerBlock, groups - g0);
    const int run = kw * cg;

    const int tid = static_cast<int>(threadIdx.x);
    const int position = tid / lanes;
    const int c = (tid - position * lanes) * VEC;
    const int localGroup = c / cg;
    const int inGroup = c - localGroup * cg;
    const int ix = ix0 + position;
    const bool active = position < strip && ix < iw && localGroup < chunkGroups;
    const float* col = args.col + n * (od * oh * ow) * columns;

    Sums sum = Sums(0.0f);
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
            if(sz < 0 || sy < 0 || width <= 0)
            {
                continue; // uniform across the block
            }
            __syncthreads();
            const float* rows = col + ((sz * oh + sy) * ow + oxLo) * columns + g0 * cg * taps
                                + (ez * kh + ey) * run;
            const int vectors = width * chunkGroups * run / VEC;
            for(int i = tid; i < vectors; i += static_cast<int>(blockDim.x))
            {
                const int k = i * VEC;
                const int runIndex = k / run;
                const int within = k - runIndex * run;
                const int ox = runIndex / chunkGroups;
                const int group = runIndex - ox * chunkGroups;
                *reinterpret_cast<Sums*>(stage + ((ox * groupsPerBlock + group) * run) + within)
                    = *reinterpret_cast<const Sums*>(rows + ox * columns + group * cg * taps
                                                     + within);
            }
            __syncthreads();
            if(active)
            {
                for(int ex = 0; ex < kw; ++ex)
                {
                    const int sx = sourceOf(ix, padW, tapIndex(g.flipFilter, kw, ex), dw, sw, ow);
                    if(sx < 0)
                    {
                        continue;
                    }
                    sum += *reinterpret_cast<const Sums*>(
                        stage + ((sx - oxLo) * groupsPerBlock + localGroup) * run + ex * cg
                        + inGroup);
                }
            }
        }
    }
    if(active)
    {
        auto* x = static_cast<OutputType*>(args.x) + n * static_cast<int>(g.xStrides[0])
                  + iz * static_cast<int>(g.xStrides[2]) + iy * static_cast<int>(g.xStrides[3])
                  + ix * static_cast<int>(g.xStrides[4]) + (g0 + localGroup) * cg + inGroup;
        if constexpr(VEC == 1)
        {
            x[0] = static_cast<OutputType>(sum);
        }
        else
        {
#pragma unroll
            for(int k = 0; k < VEC; ++k)
            {
                x[k] = static_cast<OutputType>(sum[k]);
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

extern "C" __global__ void Col2ImDisjoint2(Col2ImArgs args)
{
    col2imDisjoint<unsigned short>(args);
}

extern "C" __global__ void Col2ImDisjoint4(Col2ImArgs args)
{
    col2imDisjoint<unsigned int>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointWidth2V1(Col2ImArgs args)
{
    col2imDisjointWidth<unsigned short, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointWidth2V2(Col2ImArgs args)
{
    col2imDisjointWidth<unsigned short, 2>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointWidth2V4(Col2ImArgs args)
{
    col2imDisjointWidth<unsigned short, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointWidth2V8(Col2ImArgs args)
{
    col2imDisjointWidth<unsigned short, 8>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointChannels2V1(Col2ImArgs args)
{
    col2imDisjointChannels<unsigned short, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointChannels2V2(Col2ImArgs args)
{
    col2imDisjointChannels<unsigned short, 2>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointChannels2V4(Col2ImArgs args)
{
    col2imDisjointChannels<unsigned short, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointChannels2V8(Col2ImArgs args)
{
    col2imDisjointChannels<unsigned short, 8>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointWidth4V1(Col2ImArgs args)
{
    col2imDisjointWidth<unsigned int, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointWidth4V2(Col2ImArgs args)
{
    col2imDisjointWidth<unsigned int, 2>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointWidth4V4(Col2ImArgs args)
{
    col2imDisjointWidth<unsigned int, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointChannels4V1(Col2ImArgs args)
{
    col2imDisjointChannels<unsigned int, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointChannels4V2(Col2ImArgs args)
{
    col2imDisjointChannels<unsigned int, 2>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImDisjointChannels4V4(Col2ImArgs args)
{
    col2imDisjointChannels<unsigned int, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColDisjointOutputs2V1(Im2ColArgs args)
{
    im2colDisjointOutputs<unsigned short, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColDisjointOutputs2V2(Im2ColArgs args)
{
    im2colDisjointOutputs<unsigned short, 2>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColDisjointOutputs2V4(Im2ColArgs args)
{
    im2colDisjointOutputs<unsigned short, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColDisjointOutputs2V8(Im2ColArgs args)
{
    im2colDisjointOutputs<unsigned short, 8>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColDisjointOutputs4V1(Im2ColArgs args)
{
    im2colDisjointOutputs<unsigned int, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColDisjointOutputs4V2(Im2ColArgs args)
{
    im2colDisjointOutputs<unsigned int, 2>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColDisjointOutputs4V4(Im2ColArgs args)
{
    im2colDisjointOutputs<unsigned int, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) StridedTransposeNarrow2(StridedCopyArgs args)
{
    stridedTransposeNarrow<unsigned short>(args);
}

extern "C" __global__ void __launch_bounds__(256) StridedTransposeNarrow4(StridedCopyArgs args)
{
    stridedTransposeNarrow<unsigned int>(args);
}

extern "C" __global__ void __launch_bounds__(64) StridedCastWideFloatToFloat(StridedCastArgs args)
{
    stridedCastWide<float>(args);
}

extern "C" __global__ void __launch_bounds__(64) StridedCastWideFloatToHalf(StridedCastArgs args)
{
    stridedCastWide<_Float16>(args);
}

extern "C" __global__ void __launch_bounds__(64)
    StridedCastWideFloatToBfloat16(StridedCastArgs args)
{
    stridedCastWide<__bf16>(args);
}

extern "C" __global__ void __launch_bounds__(256) StridedTranspose2(StridedCopyArgs args)
{
    stridedTranspose<unsigned short>(args);
}

extern "C" __global__ void __launch_bounds__(256) StridedTranspose4(StridedCopyArgs args)
{
    stridedTranspose<unsigned int>(args);
}

extern "C" __global__ void StridedCastFloatToFloat(StridedCastArgs args)
{
    stridedCast<float>(args);
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

extern "C" __global__ void __launch_bounds__(256) Col2ImGroupsFloatV1(Col2ImArgs args)
{
    col2imGroups<float, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImGroupsFloatV4(Col2ImArgs args)
{
    col2imGroups<float, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImGroupsHalfV1(Col2ImArgs args)
{
    col2imGroups<_Float16, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImGroupsHalfV4(Col2ImArgs args)
{
    col2imGroups<_Float16, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImGroupsBfloat16V1(Col2ImArgs args)
{
    col2imGroups<__bf16, 1>(args);
}

extern "C" __global__ void __launch_bounds__(256) Col2ImGroupsBfloat16V4(Col2ImArgs args)
{
    col2imGroups<__bf16, 4>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColTaps2(Im2ColArgs args)
{
    im2colTaps<unsigned short>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColTaps4(Im2ColArgs args)
{
    im2colTaps<unsigned int>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColTapsStaged2(Im2ColArgs args)
{
    im2colTapsStaged<unsigned short>(args);
}

extern "C" __global__ void __launch_bounds__(256) Im2ColTapsStaged4(Im2ColArgs args)
{
    im2colTapsStaged<unsigned int>(args);
}
