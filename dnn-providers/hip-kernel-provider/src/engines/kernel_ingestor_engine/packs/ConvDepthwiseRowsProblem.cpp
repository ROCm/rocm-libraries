// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseRowsProblem.hpp"

#include <initializer_list>
#include <limits>

namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_rows
{

namespace
{

using conv_depthwise_valu::DepthwiseLayout;
using conv_depthwise_valu::DepthwiseValuProblem;
using conv_naive::ConvDirection;

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
constexpr int64_t MAX_BLOCK = 1024;
/// LDS a block may claim, in fp32 elements (64 KiB).
constexpr int64_t LDS_FLOATS = 16384;

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

bool allZero(std::initializer_list<int64_t> fields)
{
    for(const int64_t field : fields)
    {
        if(field != 0)
        {
            return false;
        }
    }
    return true;
}

/// A compile-time extent of 0 reads the problem's; any other must equal it.
bool filterFits(int64_t configured, int problemExtent)
{
    return configured == 0 || configured == problemExtent;
}

/// The phase index range of a dx row: rowsPhaseRange in the kernel.
void phaseRange(int64_t dxLength, int64_t pad, int64_t stride, int64_t& qmin, int64_t& numQ)
{
    qmin = pad / stride;
    numQ = (dxLength - 1 + pad) / stride - qmin + 1;
}

/// The forward form a tile or band runs: forward, or unit-stride backward data as the forward
/// convolution of dy with the filter reflected.
std::optional<RowsLaunch> forwardForm(const DepthwiseValuProblem& problem,
                                      ConvDirection direction,
                                      const RowsConfig& config)
{
    if(problem.layout != DepthwiseLayout::CHANNELS_FIRST || config.layout != "NCHW")
    {
        return std::nullopt;
    }
    if(direction == ConvDirection::BACKWARD_DATA && problem.gatherOnly)
    {
        return std::nullopt;
    }
    const auto& a = problem.args;
    if(!filterFits(config.kh, a.kh) || !filterFits(config.kw, a.kw))
    {
        return std::nullopt;
    }
    if(config.lds < 1 || config.lds > LDS_FLOATS)
    {
        return std::nullopt;
    }
    RowsLaunch launch;
    launch.args = a;
    launch.flip = problem.flipFilter;
    return launch;
}

/// The gather form a phase or lanes config runs: backward data of a convolution whose height
/// axis is the identity and whose width has unit dilation, at the config's stride.
std::optional<RowsLaunch> gatherForm(const DepthwiseValuProblem& problem,
                                     ConvDirection direction,
                                     const RowsConfig& config)
{
    if(direction != ConvDirection::BACKWARD_DATA)
    {
        return std::nullopt;
    }
    const auto& g = problem.gatherArgs;
    if(g.kh != 1 || g.sh != 1 || g.ph != 0 || g.hi != g.ho || g.dw != 1 || g.pw < 0)
    {
        return std::nullopt;
    }
    if(config.stride < 1 || config.stride != g.sw || config.kh != 0 || !filterFits(config.kw, g.kw))
    {
        return std::nullopt;
    }
    if(config.threads < 1 || config.threads > MAX_BLOCK)
    {
        return std::nullopt;
    }
    RowsLaunch launch;
    launch.args = g;
    launch.flip = problem.convolutionMode;
    launch.block = config.threads;
    return launch;
}

} // namespace

std::optional<RowsVariant> rowsVariantOf(int64_t value)
{
    switch(value)
    {
    case static_cast<int64_t>(RowsVariant::TILE):
        return RowsVariant::TILE;
    case static_cast<int64_t>(RowsVariant::BAND):
        return RowsVariant::BAND;
    case static_cast<int64_t>(RowsVariant::PHASE):
        return RowsVariant::PHASE;
    case static_cast<int64_t>(RowsVariant::LANES):
        return RowsVariant::LANES;
    default:
        return std::nullopt;
    }
}

int64_t rowsHaloExtent(int64_t outputs, int64_t stride, int64_t filter, int64_t dilation)
{
    return (outputs - 1) * stride + (filter - 1) * dilation + 1;
}

int64_t rowsBandThreads(int64_t groups, int64_t wo, int64_t maxThreads)
{
    const int64_t rounded = divUp(groups * wo, 64) * 64;
    return rounded < maxThreads ? rounded : maxThreads;
}

std::optional<RowsLaunch> rowsLaunchFor(const DepthwiseValuProblem& problem,
                                        ConvDirection direction,
                                        const RowsConfig& config)
{
    const auto variant = rowsVariantOf(config.variant);
    if(!variant.has_value() || problem.ndims != 2)
    {
        return std::nullopt;
    }
    std::optional<RowsLaunch> launch;
    switch(*variant)
    {
    case RowsVariant::TILE:
    {
        if(!allZero({config.bh, config.stride, config.q, config.mmax}) || config.tx < 1
           || config.ty < 1 || config.rw < 1 || config.rh < 1
           || config.threads != config.tx * config.ty || config.threads > MAX_BLOCK)
        {
            return std::nullopt;
        }
        launch = forwardForm(problem, direction, config);
        if(!launch.has_value())
        {
            return std::nullopt;
        }
        const auto& a = launch->args;
        const int64_t bh = config.ty * config.rh;
        const int64_t bw = config.tx * config.rw;
        if(rowsHaloExtent(bh, a.sh, a.kh, a.dh) * rowsHaloExtent(bw, a.sw, a.kw, a.dw) > config.lds)
        {
            return std::nullopt;
        }
        launch->block = config.threads;
        launch->grid = int64_t{a.n} * a.c * divUp(a.ho, bh) * divUp(a.wo, bw);
        break;
    }
    case RowsVariant::BAND:
    {
        if(!allZero({config.tx, config.ty, config.rw, config.stride, config.q, config.mmax})
           || config.rh < 1 || config.bh < 1 || config.bh % config.rh != 0 || config.threads < 1
           || config.threads > MAX_BLOCK)
        {
            return std::nullopt;
        }
        launch = forwardForm(problem, direction, config);
        if(!launch.has_value())
        {
            return std::nullopt;
        }
        const auto& a = launch->args;
        if(rowsHaloExtent(config.bh, a.sh, a.kh, a.dh) * rowsHaloExtent(a.wo, a.sw, a.kw, a.dw)
           > config.lds)
        {
            return std::nullopt;
        }
        launch->block = rowsBandThreads(config.bh / config.rh, a.wo, config.threads);
        launch->grid = int64_t{a.n} * a.c * divUp(a.ho, config.bh);
        break;
    }
    case RowsVariant::PHASE:
    {
        if(problem.layout
               != (config.layout == "NCHW" ? DepthwiseLayout::CHANNELS_FIRST
                                           : DepthwiseLayout::CHANNELS_LAST)
           || (config.layout != "NCHW" && config.layout != "NHWC")
           || !allZero({config.tx, config.ty, config.rw, config.rh, config.bh, config.lds})
           || config.q < 1 || config.mmax < 1)
        {
            return std::nullopt;
        }
        launch = gatherForm(problem, direction, config);
        if(!launch.has_value())
        {
            return std::nullopt;
        }
        const auto& g = launch->args;
        // The dy window holds mmax taps per phase; a compile-time filter has exactly its own.
        const int64_t taps = divUp(g.kw, config.stride);
        if(taps > config.mmax || (config.kw != 0 && taps != config.mmax)
           || config.threads * config.q + config.mmax - 1 > LDS_FLOATS)
        {
            return std::nullopt;
        }
        int64_t qmin = 0;
        int64_t numQ = 0;
        phaseRange(g.wo, g.pw, config.stride, qmin, numQ);
        launch->grid = int64_t{g.n} * g.c * g.ho * divUp(numQ, config.threads * config.q);
        break;
    }
    case RowsVariant::LANES:
    {
        if(problem.layout != DepthwiseLayout::CHANNELS_LAST || config.layout != "NHWC"
           || !allZero({config.tx,
                        config.ty,
                        config.rw,
                        config.rh,
                        config.bh,
                        config.q,
                        config.lds,
                        config.mmax}))
        {
            return std::nullopt;
        }
        launch = gatherForm(problem, direction, config);
        if(!launch.has_value())
        {
            return std::nullopt;
        }
        const auto& g = launch->args;
        int64_t qmin = 0;
        int64_t numQ = 0;
        phaseRange(g.wo, g.pw, config.stride, qmin, numQ);
        launch->grid = divUp(int64_t{g.n} * g.ho * numQ * g.c, config.threads);
        break;
    }
    default:
        return std::nullopt;
    }
    if(launch->grid < 1 || launch->grid > INT_LIMIT || launch->block < 1
       || launch->block > MAX_BLOCK)
    {
        return std::nullopt;
    }
    return launch;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_rows

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
