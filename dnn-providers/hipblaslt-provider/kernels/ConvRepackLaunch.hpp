// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "kernels/ConvRepackArgs.h"

namespace hipblaslt_plugin
{

/// One launch of a repack kernel: its entry point, grid and block, and dynamic LDS.
struct RepackLaunch
{
    std::string entry;
    unsigned int grid = 1;
    unsigned int block = 256;
    size_t ldsBytes = 0;
};

/// The disjoint run kernels narrow their runs until they have at least this many threads: wide
/// runs of a small problem leave most of the device idle.
constexpr long long MIN_RUN_THREADS = 1LL << 16;

namespace repack_launch_detail
{

constexpr int BLOCK = 256;
/// Grid-stride kernels cap their grid; the specialized ones cover the problem exactly.
constexpr long long MAX_GENERIC_BLOCKS = 1LL << 20;
constexpr long long INT_LIMIT = std::numeric_limits<int>::max();
/// LDS one tile may stage: within every target's 64 KiB.
constexpr long long MAX_TILE_LDS_BYTES = 64 * 1024;
/// Grouped col2im stages groups under this many channels; wider groups read whole vectors.
constexpr long long MAX_STAGED_GROUP_CHANNELS = 64;
/// Rows the staged per-tap im2col builds in LDS per block: at most 64 and 32 KiB, at least 4.
constexpr long long MAX_STAGED_ROWS = 64;
constexpr long long MIN_STAGED_ROWS = 4;
constexpr long long MAX_STAGED_BYTES = 32 * 1024;
/// Threads per dx position the grouped col2im aims for, so a block spans 16+ positions.
constexpr int STAGED_LANES = 16;

inline long long product3(const long long (&values)[CONV_REPACK_SPATIAL_DIMS])
{
    return values[0] * values[1] * values[2];
}

/// One past the largest element offset @p g addresses in x.
inline long long xSpan(const ConvColumnGeometry& g)
{
    long long span = 1 + (g.n - 1) * g.xStrides[0] + (g.groups * g.cPerGroup - 1) * g.xStrides[1];
    for(int axis = 0; axis < CONV_REPACK_SPATIAL_DIMS; ++axis)
    {
        span += (g.input[axis] - 1) * g.xStrides[2 + axis];
    }
    return span;
}

inline long long columnElements(const ConvColumnGeometry& g)
{
    return g.n * product3(g.output) * g.groups * g.cPerGroup * product3(g.filter);
}

/// Every offset the specialized kernels compute fits a 32-bit int.
inline bool fitsInt(const ConvColumnGeometry& g)
{
    return xSpan(g) <= INT_LIMIT && columnElements(g) <= INT_LIMIT
           && g.n * product3(g.input) * g.groups * g.cPerGroup <= INT_LIMIT;
}

inline bool aligned(const void* pointer, int bytes)
{
    return reinterpret_cast<std::uintptr_t>(pointer) % static_cast<std::uintptr_t>(bytes) == 0;
}

/// Lanes per row for @p vectors vectors: the next power of two, at most a block.
inline int lanesFor(long long vectors)
{
    int lanes = 1;
    while(lanes < vectors && lanes < BLOCK)
    {
        lanes *= 2;
    }
    return lanes;
}

inline unsigned int genericBlocks(long long total)
{
    return static_cast<unsigned int>(
        std::clamp<long long>((total + BLOCK - 1) / BLOCK, 1, MAX_GENERIC_BLOCKS));
}

} // namespace repack_launch_detail

/**
 * @brief The launch for strided copy @p args of @p bytes-wide elements; first merges, in place,
 *        the axes contiguous in both src and dst (and drops extent-1 axes).
 *
 * When src and dst are unit-stride along different axes (a channels-first to channels-last
 * staging, say): the LDS tile transpose, one block per 32 x 32 tile of those axes and per entry
 * of the rest. Otherwise the generic kernel.
 */
inline RepackLaunch stridedCopyLaunch(StridedCopyArgs& args, int bytes)
{
    using namespace repack_launch_detail;
    // Merge outer axis i into inner axis i + 1 when both strides chain; drop extent-1 axes.
    int rank = 0;
    for(int axis = 0; axis < args.rank; ++axis)
    {
        if(args.dims[axis] == 1)
        {
            continue;
        }
        if(rank > 0 && args.srcStrides[rank - 1] == args.srcStrides[axis] * args.dims[axis]
           && args.dstStrides[rank - 1] == args.dstStrides[axis] * args.dims[axis])
        {
            args.dims[rank - 1] *= args.dims[axis];
            args.srcStrides[rank - 1] = args.srcStrides[axis];
            args.dstStrides[rank - 1] = args.dstStrides[axis];
            continue;
        }
        args.dims[rank] = args.dims[axis];
        args.srcStrides[rank] = args.srcStrides[axis];
        args.dstStrides[rank] = args.dstStrides[axis];
        ++rank;
    }
    if(rank == 0)
    {
        args.dims[0] = 1;
        args.srcStrides[0] = args.dstStrides[0] = 1;
        rank = 1;
    }
    args.rank = rank;
    args.srcUnitAxis = args.dstUnitAxis = -1;
    for(int axis = 0; axis < rank; ++axis)
    {
        if(args.srcStrides[axis] == 1)
        {
            args.srcUnitAxis = axis;
        }
        if(args.dstStrides[axis] == 1)
        {
            args.dstUnitAxis = axis;
        }
    }

    RepackLaunch launch;
    const std::string width = std::to_string(bytes);
    if(args.srcUnitAxis >= 0 && args.dstUnitAxis >= 0 && args.srcUnitAxis != args.dstUnitAxis)
    {
        long long blocks = 1;
        for(int axis = 0; axis < rank; ++axis)
        {
            blocks *= axis == args.srcUnitAxis || axis == args.dstUnitAxis
                          ? (args.dims[axis] + 31) / 32
                          : args.dims[axis];
        }
        // Every offset in 32 bits: the narrow transpose (64 x 64 tiles of 16-bit pairs).
        long long srcSpan = 1;
        long long dstSpan = 1;
        for(int axis = 0; axis < rank; ++axis)
        {
            srcSpan += (args.dims[axis] - 1) * args.srcStrides[axis];
            dstSpan += (args.dims[axis] - 1) * args.dstStrides[axis];
        }
        args.narrow = srcSpan <= INT_LIMIT && dstSpan <= INT_LIMIT ? 1 : 0;
        if(args.narrow != 0)
        {
            const long long tile = bytes == 2 ? 64 : 32;
            long long narrowBlocks = 1;
            for(int axis = 0; axis < rank; ++axis)
            {
                narrowBlocks *= axis == args.srcUnitAxis || axis == args.dstUnitAxis
                                    ? (args.dims[axis] + tile - 1) / tile
                                    : args.dims[axis];
            }
            if(narrowBlocks <= INT_LIMIT)
            {
                launch.entry = "StridedTransposeNarrow" + width;
                launch.grid = static_cast<unsigned int>(narrowBlocks);
                return launch;
            }
        }
        if(blocks <= INT_LIMIT)
        {
            launch.entry = "StridedTranspose" + width;
            launch.grid = static_cast<unsigned int>(blocks);
            return launch;
        }
    }
    launch.entry = "StridedCopy" + width;
    launch.grid = genericBlocks(args.total);
    return launch;
}

/**
 * @brief The launch for im2col @p args moving @p bytes-wide elements; fills args.launch.
 *
 * Channels-last columns in KRSC order from channels-contiguous x: the rows kernel, 16-byte
 * vectors when the channels and every stride allow, or one thread per tap when a tap holds under
 * 32 bytes of channels (staged in LDS for 16-byte stores when aligned). Channels-first 2-D
 * columns of a filter whose taps never overlap: one thread per run of a column's outputs, at least
 * @p minRunThreads of them. Other channels-first 2-D columns from a width-contiguous x: the LDS
 * tile kernel, 32 x 8 outputs (narrower and taller for narrow outputs), its longer side halved
 * until the halo fits. Anything else, or offsets past 32 bits: the generic kernel.
 */
inline RepackLaunch
    im2colLaunch(Im2ColArgs& args, int bytes, long long minRunThreads = MIN_RUN_THREADS)
{
    using namespace repack_launch_detail;
    const auto& g = args.geometry;
    const std::string width = std::to_string(bytes);
    RepackLaunch launch;
    args.launch = RepackLaunchShape{};

    if(fitsInt(g) && g.columnsFirst == 0 && g.channelsInner != 0 && g.xStrides[1] == 1
       && g.cPerGroup * bytes < 32)
    {
        // A tap holds less than two vectors of channels: one thread per tap, staging up to 64
        // rows in LDS to store them as 16-byte vectors when the spans stay aligned.
        args.launch.lanes = lanesFor(g.groups * product3(g.filter));
        const long long rows = g.n * product3(g.output);
        const long long rowBytes = g.groups * product3(g.filter) * g.cPerGroup * bytes;
        long long staged = MAX_STAGED_ROWS;
        while(staged >= MIN_STAGED_ROWS
              && (staged * rowBytes > MAX_STAGED_BYTES || (staged * rowBytes) % 16 != 0))
        {
            staged /= 2;
        }
        if(staged >= MIN_STAGED_ROWS && aligned(args.col, 16))
        {
            args.launch.rows = static_cast<int>(staged);
            launch.entry = "Im2ColTapsStaged" + width;
            launch.grid = static_cast<unsigned int>((rows + staged - 1) / staged);
            launch.ldsBytes = static_cast<size_t>(staged * rowBytes);
            return launch;
        }
        args.launch.rows = BLOCK / args.launch.lanes;
        launch.entry = "Im2ColTaps" + width;
        launch.grid = static_cast<unsigned int>((rows + args.launch.rows - 1) / args.launch.rows);
        return launch;
    }
    if(fitsInt(g) && g.columnsFirst == 0 && g.channelsInner != 0 && g.xStrides[1] == 1)
    {
        const int vec = 16 / bytes;
        const bool vectorized = g.cPerGroup % vec == 0 && g.xStrides[0] % vec == 0
                                && g.xStrides[2] % vec == 0 && g.xStrides[3] % vec == 0
                                && g.xStrides[4] % vec == 0 && aligned(args.x, 16)
                                && aligned(args.col, 16);
        const int width_ = vectorized ? vec : 1;
        args.launch.lanes = lanesFor(g.cPerGroup / width_);
        args.launch.rows = BLOCK / args.launch.lanes;
        const long long rows = g.n * product3(g.output);
        launch.entry = "Im2ColRows" + width + "V" + std::to_string(width_);
        launch.grid = static_cast<unsigned int>((rows + args.launch.rows - 1) / args.launch.rows);
        return launch;
    }

    if(fitsInt(g) && g.columnsFirst != 0 && convColumnsDisjoint(g))
    {
        // Disjoint taps into channels-first columns: no input element feeds two entries, so no
        // halo to stage. One thread per run of a column along its output positions, the widest
        // of 16, 8, 4 bytes or one element that keeps every run aligned and leaves
        // minRunThreads threads.
        const long long outputs = product3(g.output);
        const long long lines = g.n * g.groups * g.cPerGroup * product3(g.filter);
        int vectorWidth = 16 / bytes;
        while(vectorWidth > 1
              && (outputs % vectorWidth != 0 || !aligned(args.col, vectorWidth * bytes)
                  || lines * ((outputs + vectorWidth - 1) / vectorWidth) < minRunThreads))
        {
            vectorWidth /= 2;
        }
        const long long runs = (outputs + vectorWidth - 1) / vectorWidth;
        const long long threads = lines * runs;
        launch.entry = "Im2ColDisjointOutputs" + width + "V" + std::to_string(vectorWidth);
        launch.grid = static_cast<unsigned int>((threads + BLOCK - 1) / BLOCK);
        return launch;
    }
    if(fitsInt(g) && g.columnsFirst != 0 && g.input[0] == 1 && g.filter[0] == 1
       && g.xStrides[4] == 1)
    {
        // 32 x 8 outputs, narrowed (and made taller) for an output narrower than 32 so no lane
        // of a tile row idles. Wider tiles store fuller runs but load a larger halo per output,
        // which costs more for tall filters than it saves.
        int tileW = 8;
        while(tileW < g.output[2] && tileW < 32)
        {
            tileW *= 2;
        }
        int tileH = BLOCK / tileW;
        const auto halo = [&](int tile, int axis) {
            return static_cast<long long>(tile - 1) * g.stride[axis]
                   + (g.filter[axis] - 1) * g.dilation[axis] + 1;
        };
        while(halo(tileH, 1) * halo(tileW, 2) * bytes > MAX_TILE_LDS_BYTES
              && (tileH > 1 || tileW > 1))
        {
            // The longer side first: the halo shrinks most for the outputs it gives up.
            if(tileW >= tileH && tileW > 1)
            {
                tileW /= 2;
            }
            else
            {
                tileH /= 2;
            }
        }
        // An output plane of at most half a block is one tile, and the block takes as many
        // channels as fill it and their halos fit.
        const long long plane = g.output[1] * g.output[2];
        long long perBlock = 1;
        if(plane * 2 <= BLOCK)
        {
            const long long whole = halo(static_cast<int>(g.output[1]), 1)
                                    * halo(static_cast<int>(g.output[2]), 2) * bytes;
            perBlock = std::min<long long>(BLOCK / plane, g.groups * g.cPerGroup);
            while(perBlock > 1 && perBlock * whole > MAX_TILE_LDS_BYTES)
            {
                perBlock /= 2;
            }
            if(perBlock > 1)
            {
                tileH = static_cast<int>(g.output[1]);
                tileW = static_cast<int>(g.output[2]);
            }
        }
        if(perBlock * halo(tileH, 1) * halo(tileW, 2) * bytes <= MAX_TILE_LDS_BYTES)
        {
            args.launch.tileH = tileH;
            args.launch.tileW = tileW;
            args.launch.haloH = static_cast<int>(halo(tileH, 1));
            args.launch.haloW = static_cast<int>(halo(tileW, 2));
            args.launch.channelsPerBlock = static_cast<int>(perBlock);
            const long long tiles
                = ((g.output[1] + tileH - 1) / tileH) * ((g.output[2] + tileW - 1) / tileW);
            const long long grid
                = tiles * g.n * ((g.groups * g.cPerGroup + perBlock - 1) / perBlock);
            if(grid <= INT_LIMIT)
            {
                launch.entry = "Im2ColTile" + width;
                launch.grid = static_cast<unsigned int>(grid);
                launch.ldsBytes
                    = static_cast<size_t>(perBlock) * static_cast<size_t>(args.launch.haloH)
                      * static_cast<size_t>(args.launch.haloW) * static_cast<size_t>(bytes);
                return launch;
            }
        }
        args.launch = RepackLaunchShape{};
    }

    launch.entry = "Im2Col" + width;
    launch.grid = genericBlocks(args.total);
    return launch;
}

/// Partials per output from which the cast sums with a 64-thread block per output.
constexpr long long WIDE_CAST_BATCHES = 64;
/// Outputs up to which it does (one block each).
constexpr long long WIDE_CAST_OUTPUTS = 1LL << 18;

/**
 * @brief The launch for cast @p args, whose generic entry is @p genericEntry
 *        ("StridedCastFloatTo<Type>"): the wide kernel (one 64-thread block per output, a fixed
 *        tree over its partials) for at least WIDE_CAST_BATCHES partials into at most
 *        WIDE_CAST_OUTPUTS outputs, else an empty entry (the generic kernel, its grid by the
 *        caller).
 */
inline RepackLaunch castLaunch(const StridedCastArgs& args, const std::string& genericEntry)
{
    RepackLaunch launch;
    if(args.batches >= WIDE_CAST_BATCHES && args.total <= WIDE_CAST_OUTPUTS)
    {
        const std::string prefix = "StridedCast";
        launch.entry = prefix + "Wide" + genericEntry.substr(prefix.size());
        launch.grid = static_cast<unsigned int>(args.total);
        launch.block = 64;
    }
    return launch;
}

/**
 * @brief The launch for col2im @p args writing @p typeName ("Float", "Half", "Bfloat16") of
 *        @p bytes bytes; fills args.launch.
 *
 * Channels-last columns in KRSC order into channels-contiguous dx: for groups under 64 channels
 * the staged-groups kernel, as many groups per block as fit 16 lanes and the LDS, halved until
 * the stage fits; otherwise the rows kernel. Both read 4-wide fp32 when the channels allow.
 * Channels-first columns into width-contiguous dx: one thread per element, width fastest.
 * Disjoint taps: one thread per run of dx, at least @p minRunThreads of them. Anything else, or
 * offsets past 32 bits: the generic kernel.
 */
inline RepackLaunch col2imLaunch(Col2ImArgs& args,
                                 const std::string& typeName,
                                 int bytes,
                                 long long minRunThreads = MIN_RUN_THREADS)
{
    using namespace repack_launch_detail;
    const auto& g = args.geometry;
    RepackLaunch launch;
    args.launch = RepackLaunchShape{};

    // Disjoint taps (a single one included): the columns are in dx's type (ConvRepackArgs.h).
    // Within 32-bit offsets, one thread per run of dx along its contiguous axis -- width (with
    // height too when dx packs rows), or channels (with width too when dx packs them), the longer
    // when both are unit --
    // the run the widest of 16, 8, 4 bytes or one element that every run start keeps aligned,
    // the run's axis holds and leaves minRunThreads threads; otherwise the generic kernel.
    if(convColumnsDisjoint(g))
    {
        const long long channels = g.groups * g.cPerGroup;
        const bool channelsUnit = g.xStrides[1] == 1;
        const bool widthUnit = g.xStrides[4] == 1;
        if(fitsInt(g) && (channelsUnit || widthUnit))
        {
            const bool byChannels = channelsUnit && (!widthUnit || channels >= g.input[2]);
            // Fused runs: channels with width (channels-last), or rows (channels-first), where
            // dx packs them.
            const bool fused = byChannels ? g.xStrides[4] == channels : g.xStrides[3] == g.input[2];
            // The axes whose strides place a run's start: all but the run's own.
            long long starts[5]
                = {g.xStrides[0], g.xStrides[1], g.xStrides[2], g.xStrides[3], g.xStrides[4]};
            starts[byChannels ? 1 : 4] = 0;
            if(fused)
            {
                starts[byChannels ? 4 : 3] = 0;
            }
            const long long length = byChannels ? (fused ? channels * g.input[2] : channels)
                                                : (fused ? g.input[1] * g.input[2] : g.input[2]);
            const long long lines = byChannels
                                        ? g.n * g.input[0] * g.input[1] * (fused ? 1 : g.input[2])
                                        : g.n * channels * g.input[0] * (fused ? 1 : g.input[1]);
            int width = 16 / bytes;
            while(width > 1)
            {
                bool fits = width <= length && aligned(args.x, width * bytes)
                            && lines * ((length + width - 1) / width) >= minRunThreads;
                for(const long long stride : starts)
                {
                    fits = fits && stride % width == 0;
                }
                if(fits)
                {
                    break;
                }
                width /= 2;
            }
            const long long runs = (length + width - 1) / width;
            args.launch.rows = fused ? 1 : 0;
            launch.entry = std::string("Col2ImDisjoint") + (byChannels ? "Channels" : "Width")
                           + std::to_string(bytes) + "V" + std::to_string(width);
            launch.grid = static_cast<unsigned int>((lines * runs + BLOCK - 1) / BLOCK);
            return launch;
        }
        launch.entry = "Col2ImDisjoint" + std::to_string(bytes);
        launch.grid = genericBlocks(args.total);
        return launch;
    }
    if(fitsInt(g) && g.columnsFirst == 0 && g.channelsInner != 0 && g.xStrides[1] == 1
       && g.groups > 1 && g.cPerGroup < MAX_STAGED_GROUP_CHANNELS)
    {
        // Few channels per group: stage each filter row's column runs in LDS.
        const int vec = g.cPerGroup % 4 == 0 && aligned(args.col, 16) ? 4 : 1;
        const int cg = static_cast<int>(g.cPerGroup);
        int groupsPerBlock
            = static_cast<int>(std::min<long long>(g.groups, std::max(1, STAGED_LANES * vec / cg)));
        const auto stageBytes = [&](int groupsInBlock, int& lanes, int& strip, int& halo) {
            lanes = groupsInBlock * cg / vec;
            strip = BLOCK / lanes;
            halo = static_cast<int>((strip - 1 + (g.filter[2] - 1) * g.dilation[2]) / g.stride[2]
                                    + 2);
            return static_cast<long long>(halo) * groupsInBlock * g.filter[2] * cg * 4;
        };
        int lanes = 0;
        int strip = 0;
        int halo = 0;
        while(stageBytes(groupsPerBlock, lanes, strip, halo) > MAX_TILE_LDS_BYTES
              && groupsPerBlock > 1)
        {
            groupsPerBlock /= 2;
        }
        const long long grid = g.n * g.input[0] * g.input[1] * ((g.input[2] + strip - 1) / strip)
                               * ((g.groups + groupsPerBlock - 1) / groupsPerBlock);
        const long long lds = stageBytes(groupsPerBlock, lanes, strip, halo);
        if(lds <= MAX_TILE_LDS_BYTES && grid <= INT_LIMIT)
        {
            args.launch.lanes = lanes;
            args.launch.rows = strip;
            args.launch.haloW = halo;
            args.launch.groupsPerBlock = groupsPerBlock;
            launch.entry = "Col2ImGroups" + typeName + "V" + std::to_string(vec);
            launch.grid = static_cast<unsigned int>(grid);
            launch.ldsBytes = static_cast<size_t>(lds);
            return launch;
        }
        args.launch = RepackLaunchShape{};
    }
    if(fitsInt(g) && g.columnsFirst == 0 && g.channelsInner != 0 && g.xStrides[1] == 1)
    {
        const bool vectorized
            = g.cPerGroup % 4 == 0 && aligned(args.col, 16) && aligned(args.x, 4 * bytes);
        const int width = vectorized ? 4 : 1;
        args.launch.lanes = lanesFor(g.groups * g.cPerGroup / width);
        args.launch.rows = BLOCK / args.launch.lanes;
        const long long rows = g.n * product3(g.input);
        launch.entry = "Col2ImRows" + typeName + "V" + std::to_string(width);
        launch.grid = static_cast<unsigned int>((rows + args.launch.rows - 1) / args.launch.rows);
        return launch;
    }
    if(fitsInt(g) && g.columnsFirst != 0 && g.xStrides[4] == 1)
    {
        launch.entry = "Col2ImChannelsFirst" + typeName;
        launch.grid = static_cast<unsigned int>((args.total + BLOCK - 1) / BLOCK);
        return launch;
    }
    launch.entry = "Col2Im" + typeName;
    launch.grid = genericBlocks(args.total);
    return launch;
}

} // namespace hipblaslt_plugin
