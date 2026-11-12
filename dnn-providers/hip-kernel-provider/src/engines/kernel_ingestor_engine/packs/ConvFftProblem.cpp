// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvFftProblem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_fft
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace
{

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
/// Global work-items of one launch (grid x threads) stay below 2^32.
constexpr int64_t WORK_ITEM_LIMIT = (int64_t{1} << 32) - 1;
constexpr size_t ALIGN = 256;
/// LDS a block may declare on every target (64 KiB; gfx950 has 160).
constexpr int64_t LDS_LIMIT_BYTES = 64 * 1024;

bool fitsInt(long long value)
{
    return value >= 0 && value <= INT_LIMIT;
}

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

size_t alignUp(size_t value)
{
    return (value + ALIGN - 1) / ALIGN * ALIGN;
}

bool isPow2(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

} // namespace

std::optional<FftProblem> parseFftProblem(const MatchContext& context, ConvDirection direction)
{
    auto base = conv_naive::parseConvNaiveProblem(context, direction);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    // The transforms and products are fp32: a compute type of fp32 (or none) only.
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    // 1-D and 2-D: the transforms are 2-D (a 1-D problem runs with height 1).
    if(base->spatialRank != 1 && base->spatialRank != 2)
    {
        return std::nullopt;
    }
    const auto& a = base->args;
    // Unit stride: a strided convolution reads one point in s of every window, which the
    // circular correlation of whole windows cannot skip.
    if(a.stride[0] != 1 || a.stride[1] != 1 || a.stride[2] != 1)
    {
        return std::nullopt;
    }
    for(const long long value : {a.n, a.c, a.k, a.groups})
    {
        if(!fitsInt(value) || value < 1)
        {
            return std::nullopt;
        }
    }
    for(int axis = 1; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        for(const long long value : {a.inSpatial[axis],
                                     a.outSpatial[axis],
                                     a.filter[axis],
                                     a.prePadding[axis],
                                     a.dilation[axis]})
        {
            if(!fitsInt(value))
            {
                return std::nullopt;
            }
        }
        // The dilated filter's span within an int as well (both factors are ints here, so the
        // product cannot overflow 64 bits; a one-tap axis has span 1).
        const long long span = (a.filter[axis] - 1) * a.dilation[axis] + 1;
        if(a.filter[axis] < 1 || !fitsInt(span))
        {
            return std::nullopt;
        }
    }

    FftProblem problem;
    problem.n = static_cast<int>(a.n);
    problem.groups = static_cast<int>(a.groups);
    problem.cg = static_cast<int>(a.c / a.groups);
    problem.kg = static_cast<int>(a.k / a.groups);
    problem.h = static_cast<int>(a.inSpatial[1]);
    problem.w = static_cast<int>(a.inSpatial[2]);
    problem.ho = static_cast<int>(a.outSpatial[1]);
    problem.wo = static_cast<int>(a.outSpatial[2]);
    problem.r = static_cast<int>(a.filter[1]);
    problem.s = static_cast<int>(a.filter[2]);
    problem.padH = static_cast<int>(a.prePadding[1]);
    problem.padW = static_cast<int>(a.prePadding[2]);
    problem.dilH = static_cast<int>(a.dilation[1]);
    problem.dilW = static_cast<int>(a.dilation[2]);
    problem.flip = a.flipFilter != 0 ? 1 : 0;
    // (n, c, d, h, w) -> (n, c, h, w); depth is 1.
    const long long* strides[3] = {a.xStrides, a.wStrides, a.yStrides};
    long long* out[3] = {problem.xStr, problem.wStr, problem.yStr};
    for(int t = 0; t < 3; ++t)
    {
        out[t][0] = strides[t][0];
        out[t][1] = strides[t][1];
        out[t][2] = strides[t][3];
        out[t][3] = strides[t][4];
    }
    problem.base = std::move(*base);
    return problem;
}

std::optional<Config> configOf(const KernelDefinition& kernel)
{
    Config config;
    config.th = static_cast<int>(kernel.getIntMetadata("th"));
    config.tw = static_cast<int>(kernel.getIntMetadata("tw"));
    config.pb = static_cast<int>(kernel.getIntMetadata("pb"));
    config.tb = static_cast<int>(kernel.getIntMetadata("tb"));
    config.gn = static_cast<int>(kernel.getIntMetadata("gn"));
    config.tm = static_cast<int>(kernel.getIntMetadata("tm"));
    const bool thOk = config.th == 1 || (isPow2(config.th) && config.th >= 8 && config.th <= 64);
    const bool twOk = isPow2(config.tw) && config.tw >= 8 && config.tw <= 64;
    if(!thOk || !twOk || config.pb < 1 || (config.th == 1 && config.pb % 2 != 0))
    {
        return std::nullopt;
    }
    if(static_cast<int64_t>(config.pb) * config.bins() * 8 > LDS_LIMIT_BYTES)
    {
        return std::nullopt;
    }
    // One thread a row pair: a pass reads every row before any thread writes over it.
    const int pairs = config.th == 1 ? config.pb / 2 : config.pb * config.th / 2;
    if(config.tb < 64 || config.tb > 256 || config.tb % 64 != 0 || config.tb < pairs)
    {
        return std::nullopt;
    }
    if(config.gn != 16 && config.gn != 32 && config.gn != 64)
    {
        return std::nullopt;
    }
    if(config.tm != 4 && !(config.tm == 8 && config.gn == 64))
    {
        return std::nullopt;
    }
    return config;
}

size_t workspaceCapBytes()
{
    const char* value = std::getenv("HIP_KERNEL_PROVIDER_CONV_FFT_WORKSPACE_CAP");
    if(value != nullptr)
    {
        char* end = nullptr;
        const unsigned long long bytes = std::strtoull(value, &end, 10);
        if(end != value && *end == '\0' && bytes >= 4096)
        {
            return static_cast<size_t>(bytes);
        }
    }
    return WORKSPACE_CAP_BYTES;
}

bool admissionFloorEnabled()
{
    const char* value = std::getenv("HIP_KERNEL_PROVIDER_CONV_FFT_FLOOR");
    return value == nullptr || std::string_view(value) != "0";
}

double admissionRatio(const FftProblem& problem)
{
    if(isDepthwise(problem))
    {
        return ADMIT_RATIO_DEPTHWISE;
    }
    return problem.base.dataType == data_objects::DataType::FLOAT ? ADMIT_RATIO : ADMIT_RATIO_16BIT;
}

double fftAdvantage(const FftProblem& problem, const Config& config)
{
    const int oh = config.th == 1 ? 1 : config.th - problem.spanH() + 1;
    const int ow = config.tw - problem.spanW() + 1;
    if(oh < 1 || ow < 1)
    {
        return 0.0;
    }
    // Per output point (and per contracted channel pair): direct r s, FFT 4 F / (oh ow) per
    // tile, counting the partial edge tiles.
    const double tiles
        = static_cast<double>(divUp(problem.ho, oh)) * static_cast<double>(divUp(problem.wo, ow));
    const double direct = static_cast<double>(problem.ho) * problem.wo * problem.r * problem.s;
    double perTile = 4.0 * config.bins();
    if(isDepthwise(problem))
    {
        const double points = static_cast<double>(config.th) * config.tw;
        perTile += 2.0 * 2.5 * points * std::log2(points);
    }
    return direct / (perTile * tiles);
}

namespace
{

/// The depthwise path's workspace and grids (planFor's tail for one-channel groups).
std::optional<Plan> depthwisePlan(Plan plan, bool wrw, int64_t computeUnits)
{
    // The backward-weights kernel holds the x and the dy spectra of its tiles at once.
    if(wrw && 2 * static_cast<int64_t>(plan.config.pb) * plan.bins * 8 > LDS_LIMIT_BYTES)
    {
        return std::nullopt;
    }
    auto& a = plan.args;
    const int64_t groups = a.groups;
    const int64_t bins = plan.bins;
    const int64_t pb = plan.config.pb;
    plan.chunks = 1;
    plan.chunk = plan.tiles;
    a.p0 = 0;
    a.pc = plan.tiles;
    a.pcGrid = plan.tiles;
    const auto spectra = pack_features::checkedProduct({groups, bins, 8});
    if(!spectra.has_value())
    {
        return std::nullopt;
    }
    plan.fixedOffset = 0;
    plan.fixedBytes = static_cast<size_t>(*spectra);
    plan.aOffset = alignUp(plan.fixedBytes);
    plan.aBytes = 0;
    plan.filterGrid = groups; // fwd / bwd filter transform; wrw inverse (one filter a group)
    if(wrw)
    {
        // Enough partitions of the tiles that groups x partitions blocks fill the device four
        // times over; each partition a whole number of transform blocks.
        const int64_t units = computeUnits > 0 ? computeUnits : 64;
        const int64_t blocks = divUp(plan.tiles, pb);
        const int64_t parts = std::clamp<int64_t>(divUp(4 * units, groups), 1, blocks);
        const int64_t perPart = divUp(blocks, parts) * pb;
        a.tilesPerPart = static_cast<int>(perPart);
        a.parts = static_cast<int>(divUp(plan.tiles, perPart));
        plan.aBytes = static_cast<size_t>(a.parts) * plan.fixedBytes;
        plan.productGridPerChunk = groups * a.parts; // the partial-sum launch
    }
    else
    {
        plan.productGridPerChunk = groups * divUp(plan.tiles, pb); // the fused launch
    }
    plan.bOffset = alignUp(plan.aOffset + plan.aBytes);
    plan.bBytes = 0;
    plan.workspaceBytes = plan.bOffset;
    if(plan.workspaceBytes > workspaceCapBytes())
    {
        return std::nullopt;
    }
    for(const long long grid : {plan.filterGrid, plan.productGridPerChunk})
    {
        if(grid < 1 || grid > INT_LIMIT || grid * 256 > WORK_ITEM_LIMIT)
        {
            return std::nullopt;
        }
    }
    return plan;
}

} // namespace

long long Plan::dataGrid(int tilesInChunk, int channels) const
{
    return divUp(tilesInChunk, config.pb) * static_cast<long long>(args.groups) * channels;
}

std::optional<Plan> planFor(const FftProblem& problem,
                            ConvDirection direction,
                            const Config& config,
                            int64_t computeUnits)
{
    const bool depthwise = isDepthwise(problem);
    // One product tile for the depthwise path (it has no product): the gn twins are not offered.
    if(depthwise && (config.gn != 64 || config.tm != 4))
    {
        return std::nullopt;
    }
    if(config.th == 1 && problem.spanH() != 1)
    {
        return std::nullopt;
    }
    const int oh = config.th == 1 ? 1 : config.th - problem.spanH() + 1;
    const int ow = config.tw - problem.spanW() + 1;
    if(oh < 1 || ow < 1)
    {
        return std::nullopt;
    }
    if(admissionFloorEnabled()
       && (problem.base.multiplyAccumulates < ADMIT_MIN_MACS
           || fftAdvantage(problem, config) < admissionRatio(problem)))
    {
        return std::nullopt;
    }

    Plan plan;
    plan.direction = direction;
    plan.config = config;
    plan.bins = config.bins();
    auto& a = plan.args;
    a.n = problem.n;
    a.groups = problem.groups;
    a.r = problem.r;
    a.s = problem.s;
    a.dilH = problem.dilH;
    a.dilW = problem.dilW;
    a.oh = oh;
    a.ow = ow;
    for(int i = 0; i < 4; ++i)
    {
        a.filtStr[i] = problem.wStr[i];
    }
    const bool bwd = direction == ConvDirection::BACKWARD_DATA;
    const bool wrw = direction == ConvDirection::BACKWARD_WEIGHTS;
    a.dstH = bwd ? problem.h : problem.ho;
    a.dstW = bwd ? problem.w : problem.wo;
    a.ca = bwd ? problem.kg : problem.cg;
    a.cb = bwd ? problem.cg : problem.kg;
    a.filtMode = bwd ? 1 : 0;
    // Backward data correlates dy with the reflected filter; convolution mode reflects again.
    a.flip = bwd ? 1 - problem.flip : problem.flip;
    a.tilesH = static_cast<int>(divUp(a.dstH, oh));
    a.tilesW = static_cast<int>(divUp(a.dstW, ow));
    const auto tiles = pack_features::checkedProduct({problem.n, a.tilesH, a.tilesW});
    if(!tiles.has_value() || !fitsInt(*tiles))
    {
        return std::nullopt;
    }
    plan.tiles = static_cast<int>(*tiles);
    plan.depthwise = depthwise;
    plan.transformThreads = config.tb;
    if(depthwise)
    {
        return depthwisePlan(std::move(plan), wrw, computeUnits);
    }

    const int64_t groups = problem.groups;
    const int64_t bins = plan.bins;
    const auto fixedElems = pack_features::checkedProduct({groups, bins, a.ca, a.cb});
    const auto perTileElems = pack_features::checkedProduct({groups, bins, a.ca + a.cb});
    if(!fixedElems.has_value() || !perTileElems.has_value())
    {
        return std::nullopt;
    }
    plan.fixedBytes = static_cast<size_t>(*fixedElems) * 8;
    const size_t cap = workspaceCapBytes();
    if(plan.fixedBytes > cap / 2)
    {
        return std::nullopt;
    }
    const size_t perTile = static_cast<size_t>(*perTileElems) * 8;
    const size_t room = cap - alignUp(plan.fixedBytes) - 2 * ALIGN;
    const int64_t fit = static_cast<int64_t>(room / perTile);
    // Chunks of fewer than 64 tiles (or the whole problem) would launch too often to pay.
    if(fit < std::min<int64_t>(plan.tiles, 64))
    {
        return std::nullopt;
    }
    plan.chunks = static_cast<int>(divUp(plan.tiles, std::min<int64_t>(fit, plan.tiles)));
    plan.chunk = static_cast<int>(divUp(plan.tiles, plan.chunks));
    a.pcGrid = plan.chunk;

    plan.fixedOffset = 0;
    plan.aOffset = alignUp(plan.fixedBytes);
    plan.aBytes = static_cast<size_t>(groups * bins * a.ca) * static_cast<size_t>(plan.chunk) * 8;
    plan.bOffset = alignUp(plan.aOffset + plan.aBytes);
    plan.bBytes = static_cast<size_t>(groups * bins * a.cb) * static_cast<size_t>(plan.chunk) * 8;
    plan.workspaceBytes = plan.bOffset + plan.bBytes;
    plan.transformThreads = config.tb;

    if(wrw)
    {
        // ConvFftCore.h fftWgradProduct: 4096 / gn x channels by gn dy channels a block.
        plan.filterGrid = groups * a.cb * divUp(a.ca, config.pb);
        plan.productGridPerChunk
            = groups * bins * divUp(a.cb, config.gn) * divUp(a.ca, 4096 / config.gn);
    }
    else
    {
        // ConvFftCore.h FftProductTile: tm tiles a thread, 256 / (gn / 4) threads along the tiles
        // (64 x 64 or 128 x 64, 128 x 32, 256 x 16).
        const int64_t bm = (256 / (config.gn / 4)) * config.tm;
        plan.filterGrid = groups * a.ca * divUp(a.cb, config.pb);
        plan.productGridPerChunk = groups * bins * divUp(a.cb, config.gn) * divUp(plan.chunk, bm);
    }
    const int maxChannels = std::max(a.ca, a.cb);
    for(const long long grid :
        {plan.filterGrid, plan.productGridPerChunk, plan.dataGrid(plan.chunk, maxChannels)})
    {
        if(grid < 1 || grid > INT_LIMIT || grid * 256 > WORK_ITEM_LIMIT)
        {
            return std::nullopt;
        }
    }
    return plan;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_fft

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
