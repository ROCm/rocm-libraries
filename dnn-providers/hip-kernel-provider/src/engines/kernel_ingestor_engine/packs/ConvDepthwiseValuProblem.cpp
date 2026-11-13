// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"

#include <array>
#include <limits>

namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_valu
{

namespace
{

using conv_naive::ConvDirection;

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
constexpr int SPATIAL = CONV_NAIVE_SPATIAL_DIMS;

/// True when @p strides (ordered n, c, d, h, w) are the packed strides of @p extents in
/// @p layout. Extent-1 axes may carry any stride: nothing is ever read through them.
bool isPacked(const long long* strides,
              const std::array<long long, 2 + SPATIAL>& extents,
              DepthwiseLayout layout)
{
    // Axes innermost first.
    std::array<int, 2 + SPATIAL> order{};
    if(layout == DepthwiseLayout::CHANNELS_LAST)
    {
        order = {1, 4, 3, 2, 0};
    }
    else
    {
        order = {4, 3, 2, 1, 0};
    }
    long long expected = 1;
    for(const int axis : order)
    {
        if(extents[static_cast<size_t>(axis)] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= extents[static_cast<size_t>(axis)];
    }
    return true;
}

std::array<long long, 2 + SPATIAL>
    extentsOf(long long n, long long c, const long long (&spatial)[SPATIAL])
{
    return {n, c, spatial[0], spatial[1], spatial[2]};
}

bool fitsInt(long long value)
{
    return value >= -INT_LIMIT && value <= INT_LIMIT;
}

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

} // namespace

std::optional<DepthwiseLayout> depthwiseLayoutOf(const ConvNaiveArgs& a)
{
    // x and y share a packed layout; the filter, (c, 1, spatial...), is packed either way.
    if(!isPacked(a.wStrides, extentsOf(a.k, 1, a.filter), DepthwiseLayout::CHANNELS_FIRST))
    {
        return std::nullopt;
    }
    for(const auto layout : {DepthwiseLayout::CHANNELS_LAST, DepthwiseLayout::CHANNELS_FIRST})
    {
        if(isPacked(a.xStrides, extentsOf(a.n, a.c, a.inSpatial), layout)
           && isPacked(a.yStrides, extentsOf(a.n, a.k, a.outSpatial), layout))
        {
            return layout;
        }
    }
    return std::nullopt;
}

std::string layoutName(DepthwiseLayout layout)
{
    return layout == DepthwiseLayout::CHANNELS_LAST ? "NHWC" : "NCHW";
}

std::optional<DepthwiseVariant> variantNamed(const std::string& name)
{
    if(name == "floor")
    {
        return DepthwiseVariant::FLOOR;
    }
    if(name == "microtile")
    {
        return DepthwiseVariant::MICROTILE;
    }
    if(name == "fused")
    {
        return DepthwiseVariant::FUSED;
    }
    if(name == "lds")
    {
        return DepthwiseVariant::LDS;
    }
    if(name == "phase")
    {
        return DepthwiseVariant::PHASE;
    }
    if(name == "gather")
    {
        return DepthwiseVariant::GATHER;
    }
    return std::nullopt;
}

std::optional<DepthwiseValuProblem>
    parseDepthwiseValuProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                              ConvDirection direction)
{
    if(direction != ConvDirection::FORWARD && direction != ConvDirection::BACKWARD_DATA)
    {
        return std::nullopt;
    }
    auto base = conv_naive::parseConvNaiveProblem(context, direction);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    const auto& a = base->args;
    // Depthwise: one input and one output channel per group.
    if(a.groups < 1 || a.c != a.groups || a.k != a.groups)
    {
        return std::nullopt;
    }

    DepthwiseValuProblem problem;
    problem.ndims = base->spatialRank == 3 ? 3 : 2;

    const auto layout = depthwiseLayoutOf(a);
    if(!layout.has_value())
    {
        return std::nullopt;
    }
    problem.layout = *layout;

    // The kernels index in 32-bit ints (offsets are widened before they multiply).
    for(const long long value : {a.n, a.c})
    {
        if(!fitsInt(value))
        {
            return std::nullopt;
        }
    }
    for(int axis = 0; axis < SPATIAL; ++axis)
    {
        for(const long long value : {a.inSpatial[axis],
                                     a.outSpatial[axis],
                                     a.filter[axis],
                                     a.prePadding[axis],
                                     a.stride[axis],
                                     a.dilation[axis]})
        {
            if(!fitsInt(value))
            {
                return std::nullopt;
            }
        }
    }

    const bool forward = direction == ConvDirection::FORWARD;
    // Backward data runs as a forward convolution only for unit stride; strided, only the
    // gather variant serves it.
    if(!forward)
    {
        for(int axis = 0; axis < SPATIAL; ++axis)
        {
            if(a.stride[axis] != 1)
            {
                problem.gatherOnly = true;
            }
        }
    }

    // The run convolution, axes (d, h, w): forward reads x into y; backward data dy into dx.
    std::array<long long, SPATIAL> in{};
    std::array<long long, SPATIAL> out{};
    std::array<long long, SPATIAL> pad{};
    for(int axis = 0; axis < SPATIAL; ++axis)
    {
        const auto i = static_cast<size_t>(axis);
        in[i] = forward ? a.inSpatial[axis] : a.outSpatial[axis];
        out[i] = forward ? a.outSpatial[axis] : a.inSpatial[axis];
        pad[i] = forward ? a.prePadding[axis]
                         : a.dilation[axis] * (a.filter[axis] - 1) - a.prePadding[axis];
        if(!fitsInt(pad[i]))
        {
            return std::nullopt;
        }
    }
    problem.unitStrideAndDilation = true;
    for(int axis = 0; axis < SPATIAL; ++axis)
    {
        if(a.stride[axis] != 1 || a.dilation[axis] != 1)
        {
            problem.unitStrideAndDilation = false;
        }
    }
    // Reflected for true convolution, and again for backward data: they cancel.
    const bool convolutionMode = a.flipFilter != 0;
    problem.convolutionMode = convolutionMode;
    problem.flipFilter = forward ? convolutionMode : !convolutionMode;

    problem.inUid = forward ? base->operands.x : base->operands.y;
    problem.filterUid = base->operands.w;
    problem.outUid = forward ? base->operands.y : base->operands.x;

    auto& args = problem.args;
    args.n = static_cast<int>(a.n);
    args.c = static_cast<int>(a.c);
    args.di = static_cast<int>(in[0]);
    args.hi = static_cast<int>(in[1]);
    args.wi = static_cast<int>(in[2]);
    args.dOut = static_cast<int>(out[0]);
    args.ho = static_cast<int>(out[1]);
    args.wo = static_cast<int>(out[2]);
    args.kd = static_cast<int>(a.filter[0]);
    args.kh = static_cast<int>(a.filter[1]);
    args.kw = static_cast<int>(a.filter[2]);
    args.pd = static_cast<int>(pad[0]);
    args.ph = static_cast<int>(pad[1]);
    args.pw = static_cast<int>(pad[2]);
    args.sd = static_cast<int>(a.stride[0]);
    args.sh = static_cast<int>(a.stride[1]);
    args.sw = static_cast<int>(a.stride[2]);
    args.dd = static_cast<int>(a.dilation[0]);
    args.dh = static_cast<int>(a.dilation[1]);
    args.dw = static_cast<int>(a.dilation[2]);

    if(!forward)
    {
        auto& g = problem.gatherArgs;
        g = args;
        g.di = static_cast<int>(a.outSpatial[0]);
        g.hi = static_cast<int>(a.outSpatial[1]);
        g.wi = static_cast<int>(a.outSpatial[2]);
        g.dOut = static_cast<int>(a.inSpatial[0]);
        g.ho = static_cast<int>(a.inSpatial[1]);
        g.wo = static_cast<int>(a.inSpatial[2]);
        g.pd = static_cast<int>(a.prePadding[0]);
        g.ph = static_cast<int>(a.prePadding[1]);
        g.pw = static_cast<int>(a.prePadding[2]);
    }

    problem.base = std::move(*base);
    return problem;
}

std::optional<DepthwiseLaunch> launchFor(const DepthwiseValuProblem& problem,
                                         const DepthwiseTile& tile)
{
    const auto& a = problem.args;
    const int64_t n = a.n;
    const int64_t c = a.c;
    const int64_t depth = a.dOut;
    const int64_t ho = a.ho;
    const int64_t wo = a.wo;
    DepthwiseLaunch launch;
    switch(tile.variant)
    {
    case DepthwiseVariant::FLOOR:
    {
        launch.block = 256;
        // Channels-first: one thread per output element; channels-last: one per (image, depth,
        // row, W-strip, channel).
        const int64_t threads = problem.layout == DepthwiseLayout::CHANNELS_FIRST
                                    ? n * c * depth * ho * wo
                                    : n * depth * ho * divUp(wo, tile.wstrip) * c;
        launch.grid = divUp(threads, launch.block);
        break;
    }
    case DepthwiseVariant::GATHER:
    {
        // One thread per dx element.
        const auto& g = problem.gatherArgs;
        launch.block = 256;
        launch.grid = divUp(int64_t{g.n} * g.c * g.dOut * g.ho * g.wo, launch.block);
        break;
    }
    case DepthwiseVariant::PHASE:
    {
        // A thread per dy position (qy, qx) of the phase grid covering dx: channels-last, VEC
        // channels each and bk channel vectors per block; channels-first, one channel each.
        const auto& g = problem.gatherArgs;
        if(tile.stride < 1
           || (problem.layout == DepthwiseLayout::CHANNELS_LAST
               && (tile.vec < 1 || tile.bk < 1 || 256 % tile.bk != 0)))
        {
            return std::nullopt;
        }
        const int64_t qh = (int64_t{g.ho} - 1 + g.ph) / tile.stride + 1;
        const int64_t qw = (int64_t{g.wo} - 1 + g.pw) / tile.stride + 1;
        const int64_t positions = int64_t{g.n} * qh * qw;
        launch.block = 256;
        launch.grid
            = problem.layout == DepthwiseLayout::CHANNELS_LAST
                  ? divUp(positions, 256 / tile.bk) * divUp(int64_t{g.c}, tile.bk * tile.vec)
                  : divUp(positions * g.c, launch.block);
        // The kernel indexes positions (times channels, channels-first) in 32 bits.
        if(positions * g.c > INT_LIMIT)
        {
            return std::nullopt;
        }
        break;
    }
    case DepthwiseVariant::MICROTILE:
        launch.block = 256;
        launch.grid = divUp(n * divUp(ho, tile.th) * divUp(wo, tile.tw) * c, launch.block);
        break;
    case DepthwiseVariant::FUSED:
    case DepthwiseVariant::LDS:
        if(problem.ndims == 3)
        {
            launch.block
                = (tile.bd / tile.rd) * (tile.bh / tile.rh) * (tile.bw / tile.rw) * tile.bk;
            launch.grid = n * divUp(depth, tile.bd) * divUp(ho, tile.bh) * divUp(wo, tile.bw)
                          * divUp(c, tile.bk);
        }
        else
        {
            launch.block = (tile.bh / tile.rh) * (tile.bw / tile.rw) * tile.bk;
            launch.grid = n * divUp(ho, tile.bh) * divUp(wo, tile.bw) * divUp(c, tile.bk);
        }
        break;
    default:
        return std::nullopt;
    }
    if(launch.grid < 1 || launch.grid > INT_LIMIT || launch.block < 1 || launch.block > 1024)
    {
        return std::nullopt;
    }
    return launch;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_valu

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
