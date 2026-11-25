// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The run kernels of the repack program for filters whose taps never overlap
// (convColumnsDisjoint): col2im and im2col along a run of dx's or a column's contiguous axis,
// 32-bit indices, sources stepped without dividing. Included by ConvRepack.cpp inside its
// anonymous namespace, after the helpers it uses (tapOf); embedded for HIPRTC as a header of
// its own, which keeps each embedded source under the 64 KiB a string literal may hold.

#pragma once

/// The one output position and filter element (flipped for true convolution) that read input
/// position @p i along one axis of a disjoint filter, or false when none does: 32-bit, for the
/// specialized disjoint kernels below.
__device__ inline bool disjointSource(const ConvColumnGeometry& g, int axis, int i, int& o, int& e)
{
    const int s = i + static_cast<int>(g.prePadding[axis]);
    const int stride = static_cast<int>(g.stride[axis]);
    const int dilation = static_cast<int>(g.dilation[axis]);
    const int filter = static_cast<int>(g.filter[axis]);
    if(s < 0)
    {
        return false;
    }
    o = s / stride;
    const int offset = s - o * stride;
    const int tap = offset / dilation;
    e = g.flipFilter != 0 ? filter - 1 - tap : tap;
    return o < static_cast<int>(g.output[axis]) && offset == tap * dilation && tap < filter;
}

/// The column matrix index of (image n, channel c, filter element e, output position o) as
/// base + e * elementStep + o * outputStep, base fixed by (n, c): the specialized disjoint
/// kernels find it once per channel and step e and o without dividing.
struct DisjointColumns
{
    int cg;
    int taps;
    int columns;
    int outputs;
    int elementStep;
    int outputStep;

    __device__ explicit DisjointColumns(const ConvColumnGeometry& g)
        : cg(static_cast<int>(g.cPerGroup))
        , taps(static_cast<int>(g.filter[0] * g.filter[1] * g.filter[2]))
        , columns(static_cast<int>(g.groups) * cg * taps)
        , outputs(static_cast<int>(g.output[0] * g.output[1] * g.output[2]))
    {
        const int columnStep = g.channelsInner != 0 ? cg : 1;
        elementStep = g.columnsFirst != 0 ? columnStep * outputs : columnStep;
        outputStep = g.columnsFirst != 0 ? 1 : columns;
    }

    /// The base of image n, channel (group, cIn).
    __device__ int base(const ConvColumnGeometry& g, int n, int group, int cIn) const
    {
        const int column
            = g.channelsInner != 0 ? group * taps * cg + cIn : (group * cg + cIn) * taps;
        return g.columnsFirst != 0 ? (n * columns + column) * outputs
                                   : n * outputs * columns + column;
    }
};

/// Filter tap t of an axis as the filter element it reads (flipped for true convolution).
__device__ inline int disjointElement(const ConvColumnGeometry& g, int axis, int tap)
{
    return g.flipFilter != 0 ? static_cast<int>(g.filter[axis]) - 1 - tap : tap;
}

/// A run of V consecutive elements, stored as one 4-, 8- or 16-byte vector when @p whole (the
/// host checks the run is aligned), else one at a time.
template <typename Bits, int V>
__device__ inline void storeRun(Bits* x, const Bits (&values)[V], int count, bool whole)
{
    constexpr int BYTES = static_cast<int>(sizeof(Bits)) * V;
    if constexpr(V > 1 && (BYTES == 4 || BYTES == 8 || BYTES == 16))
    {
        if(whole)
        {
            typedef unsigned int U2 __attribute__((ext_vector_type(2)));
            typedef unsigned int U4 __attribute__((ext_vector_type(4)));
            if constexpr(BYTES == 16)
            {
                U4 packed;
                __builtin_memcpy(&packed, values, sizeof(packed));
                *reinterpret_cast<U4*>(x) = packed;
            }
            else if constexpr(BYTES == 8)
            {
                U2 packed;
                __builtin_memcpy(&packed, values, sizeof(packed));
                *reinterpret_cast<U2*>(x) = packed;
            }
            else
            {
                unsigned int packed;
                __builtin_memcpy(&packed, values, sizeof(packed));
                *reinterpret_cast<unsigned int*>(x) = packed;
            }
            return;
        }
    }
    for(int j = 0; j < count; ++j)
    {
        x[j] = values[j];
    }
}

/// Disjoint col2im into a width-contiguous dx (channels-first): one thread per run of V dx
/// elements along width, or, when dx packs (height, width) (args.launch.rows = 1), along both --
/// so rows of any width still make whole runs. 32-bit indices (the host checks every offset
/// fits). Depth and height sources are found once per run row, width ones stepped along it
/// without dividing. Writes every dx element once -- a column entry or 0 -- and reads each column
/// entry once; moves bytes, so the result is the generic kernel's bit for bit.
template <typename Bits, int V>
__device__ void col2imDisjointWidth(const Col2ImArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const auto* col = reinterpret_cast<const Bits*>(args.col);
    auto* x = static_cast<Bits*>(args.x);
    const bool fused = args.launch.rows != 0;
    const int id = static_cast<int>(g.input[0]);
    const int ih = static_cast<int>(g.input[1]);
    const int iw = static_cast<int>(g.input[2]);
    const int cg = static_cast<int>(g.cPerGroup);
    const int channels = static_cast<int>(g.groups) * cg;
    const int lines = fused ? 1 : ih;
    const int length = fused ? ih * iw : iw;
    const int runs = (length + V - 1) / V;
    const int total = static_cast<int>(g.n) * channels * id * lines * runs;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                      + static_cast<int>(threadIdx.x);
    if(index >= total)
    {
        return;
    }
    int rest = index;
    const int run = rest % runs;
    rest /= runs;
    const int line = rest % lines;
    rest /= lines;
    const int iz = rest % id;
    rest /= id;
    const int c = rest % channels;
    const int n = rest / channels;

    const DisjointColumns columns(g);
    const int group = c / cg;
    const int kw = static_cast<int>(g.filter[2]);
    const int ow = static_cast<int>(g.output[2]);
    const int stride = static_cast<int>(g.stride[2]);
    const int dilation = static_cast<int>(g.dilation[2]);
    const int pad = static_cast<int>(g.prePadding[2]);
    int oz = 0;
    int ez = 0;
    const bool planeRead = disjointSource(g, 0, iz, oz, ez);
    // Element (ez, 0, 0) at output (oz, 0, 0) of this image and channel.
    const int planeIndex = columns.base(g, n, group, c - group * cg)
                           + ez * static_cast<int>(g.filter[1]) * kw * columns.elementStep
                           + oz * static_cast<int>(g.output[1]) * ow * columns.outputStep;

    const int f0 = run * V;
    const int count = length - f0 < V ? length - f0 : V;
    int iy = fused ? f0 / iw : line;
    int ix = fused ? f0 - iy * iw : f0;
    // The row's source along height, and the first element's along width, then stepped:
    // offset < 0 is the leading padding.
    int oy = 0;
    int ey = 0;
    bool rowRead = planeRead && disjointSource(g, 1, iy, oy, ey);
    int rowIndex = planeIndex + ey * kw * columns.elementStep + oy * ow * columns.outputStep;
    int s = ix + pad;
    int ox = s >= 0 ? s / stride : 0;
    int offset = s >= 0 ? s - ox * stride : s;
    Bits values[V];
#pragma unroll
    for(int j = 0; j < V; ++j)
    {
        Bits value = Bits(0);
        if(j < count && rowRead && offset >= 0 && ox < ow)
        {
            int tap = offset;
            bool read = tap < kw;
            if(dilation != 1)
            {
                tap = offset / dilation;
                read = offset == tap * dilation && tap < kw;
            }
            if(read)
            {
                value = col[rowIndex + disjointElement(g, 2, tap) * columns.elementStep
                            + ox * columns.outputStep];
            }
        }
        values[j] = value;
        if(++offset == stride)
        {
            offset = 0;
            ++ox;
        }
        if(++ix == iw && j + 1 < count)
        {
            // Fused runs: on to the next row's first element.
            ix = 0;
            ++iy;
            rowRead = planeRead && disjointSource(g, 1, iy, oy, ey);
            rowIndex = planeIndex + ey * kw * columns.elementStep + oy * ow * columns.outputStep;
            ox = pad >= 0 ? pad / stride : 0;
            offset = pad >= 0 ? pad - ox * stride : pad;
        }
    }
    const int target = n * static_cast<int>(g.xStrides[0]) + c * static_cast<int>(g.xStrides[1])
                       + iz * static_cast<int>(g.xStrides[2])
                       + (fused ? f0 : line * static_cast<int>(g.xStrides[3]) + f0);
    storeRun<Bits, V>(x + target, values, count, count == V);
}

/// Disjoint col2im into a channel-contiguous dx (channels-last): one thread per run of V
/// elements along channels, or, when dx packs (width, channels) (args.launch.rows = 1), along
/// both -- so few channels still make whole runs. Sources along depth and height are found once
/// per run; width and channels step without dividing. As above, bit for
/// bit the generic kernel's result.
template <typename Bits, int V>
__device__ void col2imDisjointChannels(const Col2ImArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const auto* col = reinterpret_cast<const Bits*>(args.col);
    auto* x = static_cast<Bits*>(args.x);
    const bool fused = args.launch.rows != 0;
    const int id = static_cast<int>(g.input[0]);
    const int ih = static_cast<int>(g.input[1]);
    const int iw = static_cast<int>(g.input[2]);
    const int cg = static_cast<int>(g.cPerGroup);
    const int channels = static_cast<int>(g.groups) * cg;
    const int positions = fused ? 1 : iw;
    const int length = fused ? iw * channels : channels;
    const int runs = (length + V - 1) / V;
    const int total = static_cast<int>(g.n) * id * ih * positions * runs;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                      + static_cast<int>(threadIdx.x);
    if(index >= total)
    {
        return;
    }
    int rest = index;
    const int run = rest % runs;
    rest /= runs;
    const int position = rest % positions;
    rest /= positions;
    const int iy = rest % ih;
    rest /= ih;
    const int iz = rest % id;
    const int n = rest / id;

    int oz = 0;
    int ez = 0;
    int oy = 0;
    int ey = 0;
    const bool rowRead = disjointSource(g, 0, iz, oz, ez) && disjointSource(g, 1, iy, oy, ey);
    const DisjointColumns columns(g);
    const int kw = static_cast<int>(g.filter[2]);
    const int ow = static_cast<int>(g.output[2]);
    // Element (ez, ey, 0) at output (oz, oy, 0) of this image, channel 0.
    const int rowOffset = (ez * static_cast<int>(g.filter[1]) + ey) * kw * columns.elementStep
                          + (oz * static_cast<int>(g.output[1]) + oy) * ow * columns.outputStep;
    const int f0 = run * V;
    const int count = length - f0 < V ? length - f0 : V;
    int ix = fused ? f0 / channels : position;
    const int c0 = fused ? f0 - ix * channels : f0;
    int group = c0 / cg;
    int cIn = c0 - group * cg;
    // The position's source along width, stepped with the position (offset < 0: the leading
    // padding), as in the width kernel.
    const int stride = static_cast<int>(g.stride[2]);
    const int dilation = static_cast<int>(g.dilation[2]);
    const int s0 = ix + static_cast<int>(g.prePadding[2]);
    int ox = s0 >= 0 ? s0 / stride : 0;
    int offset = s0 >= 0 ? s0 - ox * stride : s0;
    bool read = false;
    int positionIndex = 0;
    const auto source = [&]() {
        int tap = offset;
        read = rowRead && offset >= 0 && ox < ow && tap < kw;
        if(dilation != 1)
        {
            tap = offset / dilation;
            read = rowRead && offset >= 0 && ox < ow && offset == tap * dilation && tap < kw;
        }
        positionIndex = rowOffset + disjointElement(g, 2, tap) * columns.elementStep
                        + ox * columns.outputStep;
    };
    source();
    int channelIndex = columns.base(g, n, group, cIn);
    Bits values[V];
#pragma unroll
    for(int j = 0; j < V; ++j)
    {
        values[j] = read && j < count ? col[channelIndex + positionIndex] : Bits(0);
        if(++cIn == cg)
        {
            cIn = 0;
            if(++group == static_cast<int>(g.groups) && j + 1 < count)
            {
                // Fused runs: on to the next position's channel 0.
                group = 0;
                if(++offset == stride)
                {
                    offset = 0;
                    ++ox;
                }
                source();
            }
        }
        channelIndex = columns.base(g, n, group, cIn);
    }
    const int target = n * static_cast<int>(g.xStrides[0]) + iz * static_cast<int>(g.xStrides[2])
                       + iy * static_cast<int>(g.xStrides[3])
                       + (fused ? f0 : position * static_cast<int>(g.xStrides[4]) + f0);
    storeRun<Bits, V>(x + target, values, count, count == V);
}

/// im2col of a filter whose taps never overlap into channels-first columns: each input element
/// feeds at most one column entry, so there is no reuse to stage in LDS. One thread per run of
/// V entries of a column along its (flattened) output positions, 32-bit indices (the host checks
/// every offset fits); the input position steps with the output without dividing, and the run is
/// stored as one vector when aligned. Moves bytes: the generic kernel's result bit for bit.
template <typename Bits, int V>
__device__ void im2colDisjointOutputs(const Im2ColArgs& args)
{
    const ConvColumnGeometry& g = args.geometry;
    const auto* x = static_cast<const Bits*>(args.x);
    auto* col = static_cast<Bits*>(args.col);
    const int cg = static_cast<int>(g.cPerGroup);
    const int taps = static_cast<int>(g.filter[0] * g.filter[1] * g.filter[2]);
    const int columns = static_cast<int>(g.groups) * cg * taps;
    const int od = static_cast<int>(g.output[0]);
    const int oh = static_cast<int>(g.output[1]);
    const int ow = static_cast<int>(g.output[2]);
    const int outputs = od * oh * ow;
    const int runs = (outputs + V - 1) / V;
    const int total = static_cast<int>(g.n) * columns * runs;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                      + static_cast<int>(threadIdx.x);
    if(index >= total)
    {
        return;
    }
    const int run = index % runs;
    const int line = index / runs; // n * columns + column
    const int column = line % columns;
    const int n = line / columns;
    const int group = column / (cg * taps);
    const int inGroup = column - group * cg * taps;
    const int channel = g.channelsInner != 0 ? inGroup % cg : inGroup / taps;
    const int element = g.channelsInner != 0 ? inGroup / cg : inGroup % taps;
    const int kh = static_cast<int>(g.filter[1]);
    const int kw = static_cast<int>(g.filter[2]);
    const int ez = element / (kh * kw);
    const int ey = element / kw - ez * kh;
    const int ex = element - (ez * kh + ey) * kw;
    // Input position of output 0 along each axis, and its step per output.
    const int sz = static_cast<int>(g.stride[0]);
    const int sy = static_cast<int>(g.stride[1]);
    const int sx = static_cast<int>(g.stride[2]);
    const int z0 = static_cast<int>(tapOf(g, 0, ez) * g.dilation[0] - g.prePadding[0]);
    const int y0 = static_cast<int>(tapOf(g, 1, ey) * g.dilation[1] - g.prePadding[1]);
    const int x0 = static_cast<int>(tapOf(g, 2, ex) * g.dilation[2] - g.prePadding[2]);
    const int id = static_cast<int>(g.input[0]);
    const int ih = static_cast<int>(g.input[1]);
    const int iw = static_cast<int>(g.input[2]);
    const int xsz = static_cast<int>(g.xStrides[2]);
    const int xsy = static_cast<int>(g.xStrides[3]);
    const int xsx = static_cast<int>(g.xStrides[4]);
    const int base = n * static_cast<int>(g.xStrides[0])
                     + (group * cg + channel) * static_cast<int>(g.xStrides[1]);

    const int f0 = run * V;
    const int count = outputs - f0 < V ? outputs - f0 : V;
    int oz = f0 / (oh * ow);
    int oy = f0 / ow - oz * oh;
    int ox = f0 - (oz * oh + oy) * ow;
    int iz = z0 + oz * sz;
    int iy = y0 + oy * sy;
    int ix = x0 + ox * sx;
    Bits values[V];
#pragma unroll
    for(int j = 0; j < V; ++j)
    {
        const bool inside
            = j < count && iz >= 0 && iz < id && iy >= 0 && iy < ih && ix >= 0 && ix < iw;
        values[j] = inside ? x[base + iz * xsz + iy * xsy + ix * xsx] : Bits(0);
        ix += sx;
        if(++ox == ow)
        {
            ox = 0;
            ix = x0;
            iy += sy;
            if(++oy == oh)
            {
                oy = 0;
                iy = y0;
                iz += sz;
            }
        }
    }
    storeRun<Bits, V>(col + line * outputs + f0, values, count, count == V);
}
