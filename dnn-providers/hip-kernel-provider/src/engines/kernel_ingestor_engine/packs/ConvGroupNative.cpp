// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvGroupProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvGroupNative.cpp
 * @brief The small-group convolution engines (CONV_GROUP_FWD, CONV_GROUP_BWD, CONV_GROUP_WRW):
 *        graph matching, kernel matching, workspace sizing and dispatch.
 *
 * Grouped convolutions of 2 to 16 channels a group (ResNeXt, RegNet), where an implicit GEMM
 * per group leaves its tiles mostly empty and reads 4- to 32-byte channel slices. The kernels
 * (descriptors/conv/group/kernels) run each 16-channel block on a 16x16 matrix-core atom with
 * the block-diagonal expansion of its groups' filters, from an input halo staged in LDS: 16 /
 * C/g groups per multiply, every input read once a workgroup. Forward gathers x into y; backward
 * data gathers dy into dx with the filter reflected and its channel roles swapped, which the
 * expansion does, by stride phases when strided. 3-D convolutions loop the depth taps over planes. The activations are staged channels-last by the implicit-GEMM
 * engine's own parse and staging kernels when the graph's are not, or (cf configs) read and
 * written in place when both are packed channels-first. Backward weights reads x and dy in place
 * (both packed channels-first or both channels-last), multiplies each 16-channel block's dy by
 * its x at every tap's shift, keeps the in-group (diagonal) entries as fp32 partials of a split
 * of the dy tiles, and a fixed-order reduce writes dw. Every config is admitted on correctness
 * alone; autotune ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace conv_group
{

namespace
{

constexpr long long MAX_GRID = std::numeric_limits<int>::max();
constexpr size_t WORKSPACE_ALIGNMENT = 256;

size_t alignUp(size_t bytes)
{
    return (bytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

long long divUp(long long a, long long b)
{
    return (a + b - 1) / b;
}

} // namespace

long long groupLdsBytes(int elementBytes, int r, int s, int stride, int th, int tw, int cb)
{
    const long long vec = 16 / elementBytes;
    const long long rows = static_cast<long long>(th - 1) * stride + r;
    const long long cols = static_cast<long long>(tw - 1) * stride + s;
    return rows * cols * (16LL * cb + vec) * elementBytes;
}

std::optional<unsigned int> phaseBlocksWide(const K4Args& a, int th, int tw, int nbi, int nbo)
{
    if(th < 1 || tw < 1 || nbo < 1 || nbi % nbo != 0 || a.cg < 1 || a.c % a.cg != 0)
    {
        return std::nullopt;
    }
    const long long blocks = static_cast<long long>(a.n) * a.dout * divUp(a.ho, th)
                             * divUp(a.wo, tw) * (a.c / a.cg) * (nbi / nbo);
    if(blocks < 1 || blocks > MAX_GRID)
    {
        return std::nullopt;
    }
    return static_cast<unsigned int>(blocks);
}

std::optional<unsigned int> phaseBlocks(const K4Args& a, int th, int tw, int cb)
{
    if(th < 1 || tw < 1 || cb < 1 || a.c % (16 * cb) != 0)
    {
        return std::nullopt;
    }
    const long long blocks = static_cast<long long>(a.n) * a.dout * divUp(a.ho, th)
                             * divUp(a.wo, tw) * (a.c / (16 * cb));
    if(blocks < 1 || blocks > MAX_GRID)
    {
        return std::nullopt;
    }
    return static_cast<unsigned int>(blocks);
}

size_t expandedElements(const GroupProblem& problem)
{
    const auto& a = problem.igemm.args;
    const size_t taps = static_cast<size_t>(problem.kernelT * problem.kernelR * problem.kernelS);
    const size_t nbi = static_cast<size_t>(problem.nbi);
    return problem.nbi > 0 ? static_cast<size_t>(a.groups) * nbi * nbi * taps * 256
                           : static_cast<size_t>(a.c / 16) * taps * 256;
}

namespace
{

/// Workspace: the staged activations (the implicit-GEMM layout of them), then each phase's
/// expanded filter.
void layOutWorkspace(GroupProblem& problem)
{
    auto& igemm = problem.igemm;
    size_t offset = 0;
    for(auto* staging : {&igemm.x, &igemm.y})
    {
        if(staging->needed)
        {
            staging->workspaceOffset = offset;
            offset = alignUp(offset + staging->bytes);
        }
    }
    const size_t expandBytes = expandedElements(problem) * static_cast<size_t>(igemm.elementBytes);
    for(auto& phase : problem.phases)
    {
        phase.expandOffset = offset;
        offset = alignUp(offset + expandBytes);
    }
    problem.workspaceBytes = offset;
}

/// One axis of a backward-data stride phase: dx[st j + p] = sum over q < ceil(F / st) of
/// dy[j + dmin + q] w[p + pad - st (dmin + q)], dmin = ceil((p + pad - F + 1) / st), taps past
/// the filter zero.
struct PhaseAxis
{
    int out = 0; ///< dx positions the phase writes
    int pad = 0; ///< the gather's leading padding (-dmin)
    int tapBase = 0; ///< filter tap of compact tap 0
};

PhaseAxis phaseAxis(long long in, long long pad, long long filter, int st, int p)
{
    const auto ceilDiv
        = [](long long a, long long b) { return a >= 0 ? (a + b - 1) / b : -((-a) / b); };
    const long long dmin = ceilDiv(p + pad - filter + 1, st);
    PhaseAxis axis;
    axis.out = static_cast<int>(divUp(in - p, st));
    axis.pad = static_cast<int>(-dmin);
    axis.tapBase = static_cast<int>(p + pad - st * dmin);
    return axis;
}

} // namespace

std::optional<GroupProblem> parseGroupProblem(const MatchContext& context, ConvDirection direction)
{
    auto igemm = conv_igemm::parseIgemmProblem(context, direction);
    if(!igemm.has_value())
    {
        return std::nullopt;
    }
    const auto& k = igemm->args;
    const auto& base = igemm->base.args;
    const bool forward = direction == ConvDirection::FORWARD;
    // Groups the same each way: small (2 to 16 channels, dividing 16, whole 16-channel blocks)
    // or wide (17 to 64 channels, whole 16-byte vectors: a multiple of 8, each group padded to
    // ceil(cg / 16) blocks).
    if(k.groups < 2 || k.c != k.k || k.c % k.groups != 0)
    {
        return std::nullopt;
    }
    const int cg = k.c / k.groups;
    const bool small = cg >= 2 && cg <= 16 && 16 % cg == 0 && k.c % 16 == 0;
    const bool wide = cg > 16 && cg <= 64 && cg % 8 == 0;
    if(!small && !wide)
    {
        return std::nullopt;
    }
    // 2-D or 3-D, unit dilation, one stride across rows and columns (the depth stride its own);
    // backward data: strides 1 or 2.
    if(k.dd != 1 || k.dh != 1 || k.dw != 1 || k.sh != k.sw || (!forward && (k.sh > 2 || k.sd > 2)))
    {
        return std::nullopt;
    }
    GroupProblem problem;
    problem.nbi = wide ? (cg + 15) / 16 : 0;
    problem.filterT = k.ft;
    problem.filterR = k.fr;
    problem.filterS = k.fs;
    problem.stride = k.sh;
    const int st = k.sh;
    // The filter, any strides; true convolution reflected by the expansion.
    K4ExpandArgs expand{};
    expand.sK = base.wStrides[0];
    expand.sC = base.wStrides[1];
    expand.sT = base.wStrides[2];
    expand.sR = base.wStrides[3];
    expand.sS = base.wStrides[4];
    expand.c = k.c;
    expand.cg = cg;
    expand.t = k.ft;
    expand.r = k.fr;
    expand.s = k.fs;
    expand.reflect = base.flipFilter != 0 ? 1 : 0;
    expand.elementBytes = igemm->elementBytes;
    expand.nbi = problem.nbi;
    K4Args args{};
    args.n = k.n;
    args.c = k.c;
    args.cg = cg;
    if(forward)
    {
        problem.kernelT = k.ft;
        problem.kernelR = k.fr;
        problem.kernelS = k.fs;
        problem.kernelStride = st;
        GroupPhase phase;
        args.di = k.di;
        args.hi = k.hi;
        args.wi = k.wi;
        args.dout = k.dout;
        args.ho = k.ho;
        args.wo = k.wo;
        args.sd = k.sd;
        args.pd = k.pd;
        args.ph = k.ph;
        args.pw = k.pw;
        args.outD = k.dout;
        args.outH = k.ho;
        args.outW = k.wo;
        args.osd = args.osh = args.osw = 1;
        phase.args = args;
        expand.tc = k.ft;
        expand.rc = k.fr;
        expand.sc = k.fs;
        expand.tStep = expand.rStep = expand.sStep = 1;
        phase.expand = expand;
        problem.phases.push_back(phase);
    }
    else
    {
        // Phases over depth, rows and columns, each a unit-stride gather of the compact taps.
        const int sd = k.sd;
        problem.kernelT = static_cast<int>(divUp(k.ft, sd));
        problem.kernelR = static_cast<int>(divUp(k.fr, st));
        problem.kernelS = static_cast<int>(divUp(k.fs, st));
        problem.kernelStride = 1;
        for(int pt = 0; pt < sd; ++pt)
        {
            for(int pr = 0; pr < st; ++pr)
            {
                for(int pc = 0; pc < st; ++pc)
                {
                    const auto d = phaseAxis(k.di, k.pd, k.ft, sd, pt);
                    const auto h = phaseAxis(k.hi, k.ph, k.fr, st, pr);
                    const auto w = phaseAxis(k.wi, k.pw, k.fs, st, pc);
                    if(d.out < 1 || h.out < 1 || w.out < 1)
                    {
                        continue;
                    }
                    GroupPhase phase;
                    phase.args = args;
                    auto& p = phase.args;
                    p.di = k.dout;
                    p.hi = k.ho;
                    p.wi = k.wo;
                    p.dout = d.out;
                    p.ho = h.out;
                    p.wo = w.out;
                    p.sd = 1;
                    p.pd = d.pad;
                    p.ph = h.pad;
                    p.pw = w.pad;
                    p.outD = k.di;
                    p.outH = k.hi;
                    p.outW = k.wi;
                    p.osd = sd;
                    p.osh = p.osw = st;
                    p.ood = pt;
                    p.ooh = pr;
                    p.oow = pc;
                    phase.expand = expand;
                    auto& e = phase.expand;
                    e.tc = problem.kernelT;
                    e.rc = problem.kernelR;
                    e.sc = problem.kernelS;
                    e.tBase = d.tapBase;
                    e.tStep = -sd;
                    e.rBase = h.tapBase;
                    e.rStep = -st;
                    e.sBase = w.tapBase;
                    e.sStep = -st;
                    e.swap = 1;
                    problem.phases.push_back(phase);
                }
            }
        }
    }
    if(problem.phases.empty())
    {
        return std::nullopt;
    }
    // The implicit-GEMM filter staging is not used.
    igemm->w = conv_igemm::Staging{};
    problem.igemm = std::move(*igemm);
    for(const auto& phase : problem.phases)
    {
        // The smallest tile's grid within the launch limits.
        if(!(wide ? phaseBlocksWide(phase.args, 1, 16, problem.nbi, problem.nbi)
                  : phaseBlocks(phase.args, 1, 16, 1))
                .has_value())
        {
            return std::nullopt;
        }
    }
    layOutWorkspace(problem);
    return problem;
}

std::optional<std::vector<GroupLaunch>> groupLaunches(const GroupProblem& problem, bool phased)
{
    std::vector<GroupLaunch> launches;
    const auto& phases = problem.phases;
    if(!phased)
    {
        for(size_t i = 0; i < phases.size(); ++i)
        {
            launches.push_back({phases[i].args, i});
        }
        return launches;
    }
    if(problem.igemm.direction != ConvDirection::BACKWARD_DATA || problem.stride != 2
       || phases.size() % 4 != 0 || phases.size() < 4)
    {
        return std::nullopt;
    }
    const size_t elementBytes = static_cast<size_t>(problem.igemm.elementBytes);
    for(size_t i = 0; i < phases.size(); i += 4)
    {
        // Phases run depth, row, column: the four of a depth phase are (0,0) (0,1) (1,0) (1,1).
        for(size_t q = 0; q < 4; ++q)
        {
            const auto& p = phases[i + q].args;
            if(p.ooh != static_cast<int>(q / 2) || p.oow != static_cast<int>(q % 2)
               || p.ood != phases[i].args.ood)
            {
                return std::nullopt;
            }
        }
        GroupLaunch launch{phases[i].args, i};
        launch.args.fph[0] = phases[i].args.ph;
        launch.args.fph[1] = phases[i + 2].args.ph;
        launch.args.fpw[0] = phases[i].args.pw;
        launch.args.fpw[1] = phases[i + 1].args.pw;
        launch.args.wps = static_cast<long long>(
            (phases[i + 1].expandOffset - phases[i].expandOffset) / elementBytes);
        launches.push_back(launch);
    }
    return launches;
}

long long
    groupPhasedLdsBytes(int elementBytes, int r, int s, int th, int tw, int cb, bool cf, int ob)
{
    const long long vec = 16 / elementBytes;
    const long long pitch = 16LL * cb + vec;
    const long long out = ob > 0 ? ob : cb; // output blocks the tile holds
    const long long halo = static_cast<long long>(th + r) * (tw + s) * pitch;
    const long long tile
        = cf ? 2LL * th * 16 * out * (2 * tw + 1) : 4LL * th * tw * (16 * out + vec);
    return (halo + tile) * elementBytes;
}

std::optional<GroupProblem> channelsFirstInPlace(const GroupProblem& problem)
{
    const auto& igemm = problem.igemm;
    if(!igemm.x.needed || !igemm.x.transpose || !igemm.y.needed || !igemm.y.transpose)
    {
        return std::nullopt;
    }
    GroupProblem inPlace = problem;
    inPlace.igemm.x = conv_igemm::Staging{};
    inPlace.igemm.y = conv_igemm::Staging{};
    layOutWorkspace(inPlace);
    return inPlace;
}

std::optional<GroupWrwProblem> parseGroupWrwProblem(const MatchContext& context)
{
    auto wrw = conv_wrw::parseWrwProblem(context);
    // The K4 wrw kernels read x and dy in place, both packed channels-first or both packed
    // channels-last; parseWrwProblem also serves other strides (K3's generic tier).
    if(!wrw.has_value() || !(wrw->channelsFirst || wrw->channelsLast))
    {
        return std::nullopt;
    }
    const auto& a = wrw->base.args;
    // Groups the same each way: small (2 to 16 channels, dividing 16, whole 16-channel blocks;
    // not depthwise) or wide (17 to 32, a multiple of 8, each padded to two blocks). Wider groups
    // are K3's: its per-group GEMM is dense there, and the wide kernel measured behind it at 56
    // and 64 channels on every arch.
    if(a.groups < 2 || a.c != a.k || a.c % a.groups != 0)
    {
        return std::nullopt;
    }
    const long long cg = a.c / a.groups;
    const bool small = cg >= 2 && cg <= 16 && 16 % cg == 0 && a.c % 16 == 0;
    const bool wide = cg > 16 && cg <= 32 && cg % 8 == 0;
    if(!small && !wide)
    {
        return std::nullopt;
    }
    // Unit dilation; one stride over rows and columns (the depth stride is the kernel's runtime).
    if(a.dilation[0] != 1 || a.dilation[1] != 1 || a.dilation[2] != 1 || a.stride[1] != a.stride[2])
    {
        return std::nullopt;
    }
    GroupWrwProblem problem;
    problem.cg = static_cast<int>(cg);
    problem.nbi = wide ? static_cast<int>((cg + 15) / 16) : 0;
    problem.filterT = static_cast<int>(a.filter[0]);
    problem.filterR = static_cast<int>(a.filter[1]);
    problem.filterS = static_cast<int>(a.filter[2]);
    problem.stride = static_cast<int>(a.stride[1]);
    problem.wrw = std::move(*wrw);
    return problem;
}

long long
    groupWrwLdsBytes(int elementBytes, int r, int s, int stride, int th, int tw, int cb, int nbi)
{
    const long long vec = 16 / elementBytes;
    const long long rows = static_cast<long long>(th - 1) * stride + r;
    const long long cols = static_cast<long long>(tw - 1) * stride + s;
    const long long dy = static_cast<long long>(th) * tw;
    if(nbi == 0)
    {
        return (rows * cols + dy + 1) * (16LL * cb + vec) * elementBytes;
    }
    // Wide: x pixels (and the zero pixel) of nbi blocks, dy pixels of nbo (cb) blocks; nbi 2:
    // two waves an x block add their sums through LDS, a 16 x 16 fp32 block product of each at
    // least, more taps a round as the staging's LDS holds them.
    const long long stage
        = ((rows * cols + 1) * (16LL * nbi + vec) + dy * (16LL * cb + vec)) * elementBytes;
    const long long red = nbi == 2 ? nbi * cb * 256LL * 4 : 0;
    return std::max(stage, red);
}

std::optional<GroupWrwLaunch>
    groupWrwLaunch(const GroupWrwProblem& problem,
                   int th,
                   int tw,
                   int blocks,
                   int bpu,
                   const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    const auto& a = problem.wrw.base.args;
    const int64_t units = device_units::dispatchUnits(device);
    // Channel tiles: small groups 16 cb channels of any groups; wide groups a group's nbo of its
    // nbi blocks.
    const bool wide = problem.nbi > 0;
    if(th < 1 || tw < 1 || blocks < 1 || bpu < 1 || units < 1
       || (wide ? problem.nbi % blocks != 0 : a.c % (16LL * blocks) != 0))
    {
        return std::nullopt;
    }
    GroupWrwLaunch launch;
    auto& k = launch.args;
    k.n = static_cast<int>(a.n);
    k.c = static_cast<int>(a.c);
    k.cg = problem.cg;
    k.di = static_cast<int>(a.inSpatial[0]);
    k.hi = static_cast<int>(a.inSpatial[1]);
    k.wi = static_cast<int>(a.inSpatial[2]);
    k.dout = static_cast<int>(a.outSpatial[0]);
    k.ho = static_cast<int>(a.outSpatial[1]);
    k.wo = static_cast<int>(a.outSpatial[2]);
    k.sd = static_cast<int>(a.stride[0]);
    k.pd = static_cast<int>(a.prePadding[0]);
    k.ph = static_cast<int>(a.prePadding[1]);
    k.pw = static_cast<int>(a.prePadding[2]);
    k.ft = problem.filterT;
    k.tilesW = static_cast<int>(divUp(k.wo, tw));
    k.tilesH = static_cast<int>(divUp(k.ho, th));
    const auto tiles = pack_features::checkedProduct(
        {a.n, a.outSpatial[0], static_cast<int64_t>(k.tilesH), static_cast<int64_t>(k.tilesW)});
    const long long taps
        = static_cast<long long>(problem.filterT) * problem.filterR * problem.filterS;
    k.partialElems = a.c * problem.cg * taps;
    if(!tiles.has_value() || *tiles < 1)
    {
        return std::nullopt;
    }
    k.tiles = *tiles;
    const long long blocksXY
        = (wide ? a.groups * (problem.nbi / blocks) : a.c / (16LL * blocks)) * problem.filterT;
    // At least 4 tiles a split; the partials (a slot a split) within 1 GiB.
    const int64_t splits = conv_wrw::splitsWanted(blocksXY, bpu, *tiles, k.partialElems, units);
    const long long grid = blocksXY * splits;
    if(splits < 1 || splits > MAX_GRID || grid > MAX_GRID)
    {
        return std::nullopt;
    }
    k.splits = static_cast<int>(splits);
    const long long slots = splits;
    auto& r = launch.reduce;
    r.partialElems = k.partialElems;
    r.slots = static_cast<int>(slots);
    r.cg = problem.cg;
    r.t = problem.filterT;
    r.r = problem.filterR;
    r.s = problem.filterS;
    r.sK = a.wStrides[0];
    r.sC = a.wStrides[1];
    r.sT = a.wStrides[2];
    r.sR = a.wStrides[3];
    r.sS = a.wStrides[4];
    r.reflect = a.flipFilter != 0 ? 1 : 0;
    const long long reduceGrid
        = divUp(k.partialElems, 256 / k4WrwReduceGroups(static_cast<int>(slots)));
    if(reduceGrid > MAX_GRID)
    {
        return std::nullopt;
    }
    launch.grid = static_cast<unsigned int>(grid);
    launch.reduceGrid = static_cast<unsigned int>(reduceGrid);
    launch.workspaceBytes = alignUp(static_cast<size_t>(slots * k.partialElems) * sizeof(float));
    return launch;
}

GroupSplit groupSplitOf(const GroupProblem& problem)
{
    const auto& in = problem.phases.front().args;
    const auto inElems = static_cast<int64_t>(in.n) * in.c * in.di * in.hi * in.wi;
    const size_t first = problem.phases.front().expandOffset;
    const size_t expandBytes
        = expandedElements(problem) * static_cast<size_t>(problem.igemm.elementBytes);
    const auto filterElems
        = static_cast<int64_t>((problem.phases.back().expandOffset + expandBytes - first)
                               / static_cast<size_t>(problem.igemm.elementBytes));
    conv_tf32::SplitTensor x;
    x.inner = inElems;
    x.cg = 1;
    x.pattern = conv_tf32::PATTERN_A;
    conv_tf32::SplitTensor w;
    w.inner = filterElems;
    w.cg = 1;
    w.pattern = conv_tf32::PATTERN_B;
    x.outer = w.outer = 1;
    GroupSplit split;
    split.plan = conv_tf32::splitPlanFor(x, w);
    split.offset = alignUp(problem.workspaceBytes);
    split.workspaceBytes = split.offset + split.plan.bytes;
    return split;
}

bool threeBf16Pays(const GroupProblem& problem, std::string_view gcnArchName)
{
    // Wide groups only: small groups multiply the block-diagonal expansion 16 / cg times over,
    // and measured 0.3-0.8x of exact on gfx950 and gfx1201.
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    return problem.nbi > 0
           && (arch == "gfx90a" || arch == "gfx942" || arch == "gfx950"
               || arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0);
}

} // namespace conv_group

namespace
{

using conv_group::GroupProblem;

constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_group_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_group_bwd.graph_match";
constexpr std::string_view WRW_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_group_wrw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_group.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_group.dispatch";

constexpr std::string_view EXPAND_ENTRY_POINT = "ConvGroupExpand";
constexpr std::string_view COPY_ENTRY_POINT = "ConvGroupCopy";
constexpr std::string_view TRANSPOSE_ENTRY_POINT = "ConvGroupTranspose";
constexpr std::string_view WRW_REDUCE_ENTRY_POINT = "ConvGroupWrwReduce";
constexpr unsigned int COPY_BLOCK = 256;
constexpr long long COPY_MAX_BLOCKS = 1LL << 20;
/// LDS a workgroup may take (every target arch has at least 64 KB).
constexpr long long LDS_LIMIT = 64 * 1024;

constexpr std::string_view X_TOKEN = "conv_group.x.uid";
constexpr std::string_view W_TOKEN = "conv_group.w.uid";
constexpr std::string_view Y_TOKEN = "conv_group.y.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_group.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_group.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_group.flops";

std::optional<ConvDirection> directionFor(const KernelDefinition& kernel)
{
    const auto& direction = kernel.getStringMetadata("direction");
    if(direction == "FWD")
    {
        return ConvDirection::FORWARD;
    }
    if(direction == "BWD")
    {
        return ConvDirection::BACKWARD_DATA;
    }
    if(direction == "WRW")
    {
        return ConvDirection::BACKWARD_WEIGHTS;
    }
    return std::nullopt;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_group::parseGroupProblem(context, direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    const auto& igemm = problem->igemm;
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = igemm.xUid;
    bound[std::string(W_TOKEN)] = igemm.wUid;
    bound[std::string(Y_TOKEN)] = igemm.yUid;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(igemm.base.dataType));
    bound[std::string(DIRECTION_TOKEN)]
        = std::string(direction == ConvDirection::FORWARD ? "FWD" : "BWD");
    const auto flops = pack_features::checkedMultiply(2, igemm.base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

std::optional<BoundTokens> fwdGraphMatches(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::FORWARD);
}

std::optional<BoundTokens> bwdGraphMatches(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::BACKWARD_DATA);
}

/// Backward weights: x, dw (the w token) and dy (the y token).
std::optional<BoundTokens> wrwGraphMatches(const MatchContext& context)
{
    const auto problem = conv_group::parseGroupWrwProblem(context);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    const auto& wrw = problem->wrw;
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = wrw.xUid;
    bound[std::string(W_TOKEN)] = wrw.dwUid;
    bound[std::string(Y_TOKEN)] = wrw.dyUid;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(wrw.base.dataType));
    bound[std::string(DIRECTION_TOKEN)] = std::string("WRW");
    const auto flops = pack_features::checkedMultiply(2, wrw.base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

/// A config's compile-time choices, read from its metadata. Matrix-core configs: wp 16-column
/// fragments a row (cg, oc, px 0). FMA configs (atom "fma"): the graph's channels a group cg, oc
/// output channels and px columns a thread (wp 0).
struct Tile
{
    std::string atom;
    conv_igemm::AtomInfo info;
    int64_t t = 0, r = 0, s = 0, stride = 0, th = 0, wp = 0, cb = 0, cf = 0;
    int64_t cg = 0, oc = 0, px = 0;
    int64_t pf = 0; ///< phase-fused backward data at stride 2
    /// Wide groups: the 16-channel blocks a group spans (the graph's ceil(cg / 16)) and the
    /// output blocks a workgroup; 0 for small-group configs.
    int64_t nbi = 0, nbo = 0;
    bool fma = false;
    int64_t tw = 0; ///< output columns a workgroup
    /// Backward weights: the dy tile's columns (the tw knob; th its rows) and the split
    /// planner's target workgroups a dispatch unit (1 to 16). 0 for the other directions'
    /// configs.
    int64_t bpu = 0;
    /// 1: the three-bf16 form (TF32 permission) of a bf16 matrix-core config, for fp32 graphs.
    int64_t math = 0;
};

/// The 16x16 atoms the wrw kernel runs (its chunks flatten the tile's positions, so the k-step
/// need not divide a row).
bool wrwAtom(std::string_view atom)
{
    static constexpr std::array<std::string_view, 10> ATOMS{"mfma16_f16",
                                                            "mfma16_bf16",
                                                            "mfma16_bf16_908",
                                                            "mfma16_f32",
                                                            "mfma16x32_f16",
                                                            "mfma16x32_bf16",
                                                            "wmma11_f16",
                                                            "wmma11_bf16",
                                                            "wmma12_f16",
                                                            "wmma12_bf16"};
    return std::find(ATOMS.begin(), ATOMS.end(), atom) != ATOMS.end();
}

constexpr int64_t FMA_THREADS = 256;

std::optional<Tile> tileOf(const KernelDefinition& kernel, ConvDirection direction)
{
    Tile tile;
    tile.atom = kernel.getStringMetadata("atom");
    const auto info = conv_igemm::atomNamed(tile.atom);
    if(!info.has_value())
    {
        return std::nullopt;
    }
    tile.info = *info;
    tile.fma = tile.atom == "fma";
    tile.t = kernel.getIntMetadata("t");
    tile.r = kernel.getIntMetadata("r");
    tile.s = kernel.getIntMetadata("s");
    tile.stride = kernel.getIntMetadata("stride");
    tile.th = kernel.getIntMetadata("th");
    tile.wp = kernel.getIntMetadata("wp");
    tile.cb = kernel.getIntMetadata("cb");
    tile.cf = kernel.getIntMetadata("cf");
    tile.cg = kernel.getIntMetadata("cg");
    tile.oc = kernel.getIntMetadata("oc");
    tile.px = kernel.getIntMetadata("px");
    tile.pf = kernel.getIntMetadata("pf");
    tile.nbi = kernel.getIntMetadata("nbi");
    tile.nbo = kernel.getIntMetadata("nbo");
    const int64_t tw = kernel.getIntMetadata("tw");
    tile.bpu = kernel.getIntMetadata("bpu");
    tile.math = kernel.getIntMetadata("math");
    // Three-bf16: a bf16 wide-group config, forward or backward data, not phase-fused.
    if(tile.math < 0 || tile.math > 1
       || (tile.math == 1
           && (tile.fma || tile.pf != 0 || tile.nbi == 0
               || direction == ConvDirection::BACKWARD_WEIGHTS
               || conv_igemm::atomElement(tile.atom) != "bf16")))
    {
        return std::nullopt;
    }
    if(direction == ConvDirection::BACKWARD_WEIGHTS)
    {
        // A split target of at most 16 workgroups a unit, no other direction's knobs (the
        // filter, stride, th, tw and bpu are the problem's and the launch plan's to gate). A
        // 16x16 atom: cb blocks 1, 2 or a multiple of 4 up to 16, no FMA knobs. Wide groups: a
        // 16x16 atom, nbi 2 to 4 blocks a group, nbo dy blocks a workgroup dividing them, no
        // channel blocks. FMA: compiled for cg channels a group, the 16 cb x cg (k, c) units
        // dividing the 256 threads, row segments of px positions dividing tw.
        if((tile.cf != 0 && tile.cf != 1) || tile.pf != 0 || tile.wp != 0 || tile.oc != 0
           || tile.bpu > 16)
        {
            return std::nullopt;
        }
        if(tile.nbi != 0)
        {
            // (nbo dividing the group's blocks is the launch plan's gate.)
            if(!wrwAtom(tile.atom) || tile.cb != 0 || tile.cg != 0 || tile.px != 0)
            {
                return std::nullopt;
            }
            tile.tw = tw;
            return tile;
        }
        if(tile.nbo != 0)
        {
            return std::nullopt;
        }
        if(tile.fma)
        {
            if(tile.cg < 1 || tile.cb < 1 || FMA_THREADS % (16 * tile.cb * tile.cg) != 0
               || tile.px < 1 || tw % tile.px != 0)
            {
                return std::nullopt;
            }
        }
        else if(!wrwAtom(tile.atom) || tile.cg != 0 || tile.px != 0
                || !(tile.cb == 1 || tile.cb == 2 || (tile.cb % 4 == 0 && tile.cb <= 16)))
        {
            return std::nullopt;
        }
        tile.tw = tw;
        return tile;
    }
    if(tw != 0 || tile.bpu != 0)
    {
        return std::nullopt;
    }
    if(tile.t < 1 || tile.r < 1 || tile.s < 1 || tile.stride < 1 || tile.th < 1
       || (tile.cf != 0 && tile.cf != 1) || (tile.pf != 0 && tile.pf != 1))
    {
        return std::nullopt;
    }
    if(tile.nbi != 0)
    {
        // Wide groups: a matrix-core atom, 4 waves of whole rows, whole output-block tiles, no
        // channel blocks or FMA knobs (phase fusion allowed).
        if(tile.fma || info->wave == 0 || tile.nbi < 2 || tile.nbi > 4 || tile.nbo < 1
           || tile.nbi % tile.nbo != 0 || tile.th % 4 != 0 || tile.wp < 1 || tile.cb != 0
           || tile.cg != 0 || tile.oc != 0 || tile.px != 0)
        {
            return std::nullopt;
        }
        tile.tw = 16 * tile.wp;
        return tile;
    }
    if(tile.nbo != 0 || tile.cb < 1 || tile.cb > 16)
    {
        return std::nullopt;
    }
    if(!tile.fma)
    {
        // A 16x16 atom: 4 waves of fragments.
        if(info->wave == 0 || tile.wp < 1 || tile.cg != 0 || tile.oc != 0 || tile.px != 0)
        {
            return std::nullopt;
        }
        tile.tw = 16 * tile.wp;
        return tile;
    }
    // FMA: oc output channels of whole groups (or whole groups of them), a chunk's pixels a
    // whole number of waves (64 lanes) and of rows.
    const auto divides16 = [](int64_t v) { return v >= 1 && v <= 16 && 16 % v == 0; };
    if(tile.pf != 0 || tile.wp != 0 || !divides16(tile.cg) || tile.cg < 2 || !divides16(tile.oc)
       || (tile.cg % tile.oc != 0 && tile.oc % tile.cg != 0) || tile.px < 1)
    {
        return std::nullopt;
    }
    const int64_t pixels = FMA_THREADS * tile.oc / (16 * tile.cb);
    if(pixels < 64 || pixels % 64 != 0 || pixels % tile.th != 0)
    {
        return std::nullopt;
    }
    tile.tw = pixels / tile.th * tile.px;
    return tile;
}

/// A launch's workgroups for a config's tile.
std::optional<unsigned int> blocksFor(const K4Args& args, const Tile& tile)
{
    const auto th = static_cast<int>(tile.th), tw = static_cast<int>(tile.tw);
    return tile.nbi != 0 ? conv_group::phaseBlocksWide(
                               args, th, tw, static_cast<int>(tile.nbi), static_cast<int>(tile.nbo))
                         : conv_group::phaseBlocks(args, th, tw, static_cast<int>(tile.cb));
}

/// The problem a config runs: a channels-first config reads and writes the graph's packed
/// channels-first activations in place (nullopt where they are not), others the staged ones.
std::optional<GroupProblem>
    problemFor(const MatchContext& context, ConvDirection direction, const Tile& tile)
{
    auto problem = conv_group::parseGroupProblem(context, direction);
    if(problem.has_value() && tile.cf != 0)
    {
        return conv_group::channelsFirstInPlace(*problem);
    }
    return problem;
}

/// Kernel-scoped applicability: the config's atom runs on this device and multiplies the graph's
/// element type (the 16x16 atoms whose k-step divides 16), its compile-time filter (depth, rows,
/// columns) and stride are the graph's, its channel tile divides the channels, its halo fits the
/// LDS, a channels-first (cf) config's graph has both activations packed channels-first, and a
/// phase-fused (pf) config's graph is backward data at row and column stride 2.
/// The wrw problem and launch of a backward-weights config on @p context's graph: the config's
/// atom runs on the device and multiplies the graph's element type, its filter (depth, rows,
/// columns) and stride are the graph's, its form (cf) the graph's activation layout, its tile's
/// LDS within 64 KB, and the launch within the limits; nullopt otherwise.
std::optional<std::pair<conv_group::GroupWrwProblem, conv_group::GroupWrwLaunch>>
    wrwFor(const MatchContext& context, const KernelDefinition& kernel, const Tile& tile)
{
    auto problem = conv_group::parseGroupWrwProblem(context);
    // The FMA kernel takes any element type (the catalog ships it for fp32) at the graph's
    // channels a group.
    if(!problem.has_value()
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->wrw.base.dataType)
       || !conv_igemm::atomRunsOn(tile.atom, context.deviceProperties.gcnArchName)
       || (tile.fma ? tile.cg != problem->cg
                    : conv_igemm::atomElement(tile.atom)
                          != conv_igemm::dataElement(problem->wrw.base.dataType)))
    {
        return std::nullopt;
    }
    // A wide config for the blocks the graph's groups span, small configs for small groups.
    if(tile.nbi != problem->nbi)
    {
        return std::nullopt;
    }
    if(tile.t != problem->filterT || tile.r != problem->filterR || tile.s != problem->filterS
       || tile.stride != problem->stride
       || !(tile.cf != 0 ? problem->wrw.channelsFirst : problem->wrw.channelsLast))
    {
        return std::nullopt;
    }
    const auto th = static_cast<int>(tile.th), tw = static_cast<int>(tile.tw);
    // Channel blocks a workgroup: cb, or a wide group's nbo dy blocks.
    const auto blocks = static_cast<int>(tile.nbi != 0 ? tile.nbo : tile.cb);
    if(tile.th > 4096 || tile.tw > 4096
       || conv_group::groupWrwLdsBytes(problem->wrw.elementBytes,
                                       problem->filterR,
                                       problem->filterS,
                                       problem->stride,
                                       th,
                                       tw,
                                       blocks,
                                       static_cast<int>(tile.nbi))
              > LDS_LIMIT)
    {
        return std::nullopt;
    }
    auto launch = conv_group::groupWrwLaunch(
        *problem, th, tw, blocks, static_cast<int>(tile.bpu), context.deviceProperties);
    if(!launch.has_value())
    {
        return std::nullopt;
    }
    return std::make_pair(std::move(*problem), std::move(*launch));
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto direction = directionFor(kernel);
    if(!direction.has_value())
    {
        return false;
    }
    const auto tile = tileOf(kernel, *direction);
    if(!tile.has_value())
    {
        return false;
    }
    if(*direction == ConvDirection::BACKWARD_WEIGHTS)
    {
        return wrwFor(context, kernel, *tile).has_value();
    }
    const auto problem = problemFor(context, *direction, *tile);
    // Three-bf16 (math 1): a bf16 atom for an fp32 graph, under TF32 permission only.
    const bool x3 = tile->math == 1;
    if(!problem.has_value()
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->igemm.base.dataType)
       || !conv_igemm::atomRunsOn(tile->atom, context.deviceProperties.gcnArchName)
       || (!tile->fma && !x3
           && conv_igemm::atomElement(tile->atom)
                  != conv_igemm::dataElement(problem->igemm.base.dataType))
       || tile->info.tf32)
    {
        return false;
    }
    if(x3
       && (problem->igemm.base.dataType != data_objects::DataType::FLOAT
           || !conv_igemm::tf32Permitted()
           || !conv_group::threeBf16Pays(*problem, context.deviceProperties.gcnArchName)))
    {
        return false;
    }
    // An FMA config is compiled for the graph's channels a group; a wide-group config for the
    // blocks its groups span (small-group configs for small groups).
    if((tile->fma && tile->cg != problem->phases.front().args.cg) || tile->nbi != problem->nbi)
    {
        return false;
    }
    if(tile->t != problem->filterT || tile->r != problem->filterR || tile->s != problem->filterS
       || tile->stride != problem->stride)
    {
        return false;
    }
    // A phase-fused config: backward data at row and column stride 2 with all four phases.
    const auto launches = conv_group::groupLaunches(*problem, tile->pf != 0);
    if(!launches.has_value())
    {
        return false;
    }
    // The LDS the kernel takes: its compile-time filter and gather stride (backward data's
    // stride phases gather at unit stride over the taps a phase reaches); the phase-fused kernel
    // adds its output tile.
    // Three-bf16 stages bf16 and writes fp32.
    const int bytes = x3 ? 2 : problem->igemm.elementBytes;
    const int outBytes = x3 ? 4 : bytes;
    const auto th = static_cast<int>(tile->th), tw = static_cast<int>(tile->tw);
    // Wide groups stage nbi blocks of a group's channels (cb's role).
    const auto cb = static_cast<int>(tile->nbi != 0 ? tile->nbi : tile->cb);
    const long long lds
        = tile->pf != 0
              ? conv_group::groupPhasedLdsBytes(bytes,
                                                problem->kernelR,
                                                problem->kernelS,
                                                th,
                                                tw,
                                                cb,
                                                tile->cf != 0,
                                                static_cast<int>(tile->nbo))
              : conv_group::groupLdsBytes(
                    bytes, problem->kernelR, problem->kernelS, problem->kernelStride, th, tw, cb);
    if(lds > LDS_LIMIT)
    {
        return false;
    }
    // A channels-first config parks its output tile in the halo.
    if(tile->cf != 0 && tile->pf == 0
       && static_cast<long long>(th) * 16 * (tile->nbi != 0 ? tile->nbo : tile->cb) * (tw + 1)
                  * outBytes
              > lds)
    {
        return false;
    }
    return std::all_of(launches->begin(), launches->end(), [&](const auto& launch) {
        return blocksFor(launch.args, *tile).has_value();
    });
}

std::string elementTypeFor(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return "float";
    case data_objects::DataType::HALF:
        return "_Float16";
    case data_objects::DataType::BFLOAT16:
        return "unsigned short";
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "conv_group: unsupported element type");
    }
}

template <class Args>
const std::vector<KernelArgument>& signatureOf()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(Args)), 0, ""}};
    return s_signature;
}

class PreparedConvGroup : public IngestorPreparedDispatch
{
public:
    PreparedConvGroup(IngestorKernelCode conv,
                      IngestorKernelCode expand,
                      std::array<std::optional<IngestorKernelCode>, 2> copies,
                      GroupProblem problem,
                      std::vector<conv_group::GroupLaunch> launches,
                      std::vector<unsigned int> blocks,
                      std::optional<IngestorKernelCode> split,
                      conv_group::GroupSplit splitPlan)
        : _conv(std::move(conv))
        , _expand(std::move(expand))
        , _copies(std::move(copies))
        , _problem(std::move(problem))
        , _launches(std::move(launches))
        , _blocks(std::move(blocks))
        , _split(std::move(split))
        , _splitPlan(splitPlan)
    {
    }

    /// Three-bf16: the split kernel (nullopt otherwise) and its plan.
    bool split() const
    {
        return _split.has_value();
    }

    compilation::IRunnableKernel& splitForStream(hipStream_t stream) const
    {
        return _split->kernelForStream(stream);
    }

    const conv_group::GroupSplit& splitPlan() const
    {
        return _splitPlan;
    }

    /// The kernel's launches and each one's workgroups.
    const std::vector<conv_group::GroupLaunch>& launches() const
    {
        return _launches;
    }

    const std::vector<unsigned int>& blocks() const
    {
        return _blocks;
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _conv;
    }

    compilation::IRunnableKernel& convForStream(hipStream_t stream) const
    {
        return _conv.kernelForStream(stream);
    }

    compilation::IRunnableKernel& expandForStream(hipStream_t stream) const
    {
        return _expand.kernelForStream(stream);
    }

    /// The staging of x (0) or y (1).
    compilation::IRunnableKernel& copyForStream(size_t operand, hipStream_t stream) const
    {
        return _copies[operand]->kernelForStream(stream);
    }

    const GroupProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _conv;
    IngestorKernelCode _expand;
    std::array<std::optional<IngestorKernelCode>, 2> _copies;
    GroupProblem _problem;
    std::vector<conv_group::GroupLaunch> _launches;
    std::vector<unsigned int> _blocks;
    std::optional<IngestorKernelCode> _split;
    conv_group::GroupSplit _splitPlan;
};

/// Backward weights: the kernel (partials) and the reduce (dw).
class PreparedConvGroupWrw : public IngestorPreparedDispatch
{
public:
    PreparedConvGroupWrw(IngestorKernelCode conv,
                         IngestorKernelCode reduce,
                         conv_group::GroupWrwProblem problem,
                         conv_group::GroupWrwLaunch launch)
        : _conv(std::move(conv))
        , _reduce(std::move(reduce))
        , _problem(std::move(problem))
        , _launch(launch)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _conv;
    }

    compilation::IRunnableKernel& convForStream(hipStream_t stream) const
    {
        return _conv.kernelForStream(stream);
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce.kernelForStream(stream);
    }

    const conv_group::GroupWrwProblem& problem() const
    {
        return _problem;
    }

    const conv_group::GroupWrwLaunch& wrwLaunch() const
    {
        return _launch;
    }

private:
    IngestorKernelCode _conv;
    IngestorKernelCode _reduce;
    conv_group::GroupWrwProblem _problem;
    conv_group::GroupWrwLaunch _launch;
};

class ConvGroupDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvGroupDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                             const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The staged activations and the expanded filter.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto direction = directionFor(kernel);
        const auto tile
            = direction.has_value() ? tileOf(kernel, *direction) : std::optional<Tile>{};
        if(!direction.has_value() || !tile.has_value())
        {
            return 0;
        }
        if(*direction == ConvDirection::BACKWARD_WEIGHTS)
        {
            const auto wrw = wrwFor(context, kernel, *tile);
            return wrw.has_value() ? wrw->second.workspaceBytes : 0;
        }
        const auto problem = problemFor(context, *direction, *tile);
        if(!problem.has_value())
        {
            return 0;
        }
        return tile->math == 1 ? conv_group::groupSplitOf(*problem).workspaceBytes
                               : problem->workspaceBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto direction = directionFor(kernel);
        const auto tile
            = direction.has_value() ? tileOf(kernel, *direction) : std::optional<Tile>{};
        if(direction == ConvDirection::BACKWARD_WEIGHTS && tile.has_value())
        {
            return prepareWrw(context, bound, kernel, *tile);
        }
        std::optional<GroupProblem> problem;
        if(direction.has_value() && tile.has_value())
        {
            problem = problemFor(context, *direction, *tile);
        }
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!problem.has_value() || !tile.has_value() || !x.has_value() || !w.has_value()
           || !y.has_value() || *x != problem->igemm.xUid || *w != problem->igemm.wUid
           || *y != problem->igemm.yUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_group dispatch was given a graph its matcher did not admit");
        }
        const auto launches = conv_group::groupLaunches(*problem, tile->pf != 0);
        if(!launches.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_group dispatch was given a config its kernel match declines");
        }
        std::vector<unsigned int> blocks;
        for(const auto& launch : *launches)
        {
            const auto phaseGrid = blocksFor(launch.args, *tile);
            if(!phaseGrid.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "conv_group dispatch was given a config its kernel match declines");
            }
            blocks.push_back(*phaseGrid);
        }

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        // Three-bf16: the bf16 kernel over three parts writing fp32, plus the split. The staging
        // and the filter expansion stay fp32 (their element size is a runtime argument).
        const bool x3 = tile->math == 1;
        if(x3)
        {
            options.add("CONV_GROUP_T", std::string("unsigned short"));
            options.add("CONV_GROUP_TO", std::string("float"));
            options.add("CONV_GROUP_P", std::string("3"));
            options.add(std::string(conv_tf32::SPLIT_DEFINE), std::string("1"));
        }
        else
        {
            options.add("CONV_GROUP_T", elementTypeFor(problem->igemm.base.dataType));
        }
        if(tile->fma)
        {
            options.add("CONV_GROUP_FMA", std::string("1"));
            options.add("CONV_GROUP_CG", std::to_string(tile->cg));
            options.add("CONV_GROUP_OC", std::to_string(tile->oc));
            options.add("CONV_GROUP_PX", std::to_string(tile->px));
        }
        else if(tile->nbi != 0)
        {
            options.add("CONV_GROUP_WIDE", std::string("1"));
            options.add("CONV_GROUP_ATOM", std::string(tile->info.kernelType));
            options.add("CONV_GROUP_WP", std::to_string(tile->wp));
            options.add("CONV_GROUP_NBI", std::to_string(tile->nbi));
            options.add("CONV_GROUP_NBO", std::to_string(tile->nbo));
            options.add("CONV_GROUP_PF", std::to_string(tile->pf));
        }
        else
        {
            options.add("CONV_GROUP_ATOM", std::string(tile->info.kernelType));
            options.add("CONV_GROUP_PF", std::to_string(tile->pf));
            options.add("CONV_GROUP_WP", std::to_string(tile->wp));
        }
        options.add("CONV_GROUP_FT", std::to_string(problem->kernelT));
        options.add("CONV_GROUP_R", std::to_string(problem->kernelR));
        options.add("CONV_GROUP_S", std::to_string(problem->kernelS));
        options.add("CONV_GROUP_ST", std::to_string(problem->kernelStride));
        options.add("CONV_GROUP_TH", std::to_string(tile->th));
        options.add("CONV_GROUP_CB", std::to_string(tile->cb));
        options.add("CONV_GROUP_CF", std::to_string(tile->cf));

        auto conv = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, signatureOf<K4Args>());
        conv.setBlockSize(
            static_cast<unsigned int>(tile->fma ? FMA_THREADS : 4 * tile->info.wave), 1, 1);
        conv.setGridSize(blocks.front(), 1, 1);

        KernelDefinition expandKernel = kernel;
        expandKernel.source.entryPoint = std::string(EXPAND_ENTRY_POINT);
        auto expand = buildIngestorKernelCode(_kernelCompiler,
                                              _kpackLoader,
                                              context,
                                              expandKernel,
                                              options,
                                              signatureOf<K4ExpandArgs>());
        const auto expandElems = static_cast<long long>(conv_group::expandedElements(*problem));
        expand.setBlockSize(COPY_BLOCK, 1, 1);
        expand.setGridSize(static_cast<unsigned int>(std::clamp<long long>(
                               (expandElems + COPY_BLOCK - 1) / COPY_BLOCK, 1, COPY_MAX_BLOCKS)),
                           1,
                           1);

        std::array<std::optional<IngestorKernelCode>, 2> copies;
        const std::array<const conv_igemm::Staging*, 2> stagings{&problem->igemm.x,
                                                                 &problem->igemm.y};
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(!stagings[i]->needed)
            {
                continue;
            }
            KernelDefinition copyKernel = kernel;
            const bool transpose = stagings[i]->transpose;
            copyKernel.source.entryPoint
                = std::string(transpose ? TRANSPOSE_ENTRY_POINT : COPY_ENTRY_POINT);
            auto copy = buildIngestorKernelCode(_kernelCompiler,
                                                _kpackLoader,
                                                context,
                                                copyKernel,
                                                options,
                                                transpose ? signatureOf<ConvIgemmTransposeArgs>()
                                                          : signatureOf<ConvIgemmCopyArgs>());
            long long grid = 0;
            if(transpose)
            {
                const auto& t = stagings[i]->transposeArgs;
                constexpr long long TILE = CONV_IGEMM_TRANSPOSE_TILE;
                grid = (t.cols + TILE - 1) / TILE * ((t.rows + TILE - 1) / TILE) * t.batches;
                if(grid > MAX_GRID_BLOCKS)
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                        "conv_group staging transpose exceeds the launch grid");
                }
            }
            else
            {
                grid = std::clamp<long long>(
                    (stagings[i]->copy.total + COPY_BLOCK - 1) / COPY_BLOCK, 1, COPY_MAX_BLOCKS);
            }
            copy.setBlockSize(COPY_BLOCK, 1, 1);
            copy.setGridSize(static_cast<unsigned int>(grid), 1, 1);
            copies[i].emplace(std::move(copy));
        }
        std::optional<IngestorKernelCode> split;
        conv_group::GroupSplit splitPlan;
        if(x3)
        {
            splitPlan = conv_group::groupSplitOf(*problem);
            split = splitKernelCode(context, kernel, options, splitPlan);
        }
        return std::make_unique<PreparedConvGroup>(std::move(conv),
                                                   std::move(expand),
                                                   std::move(copies),
                                                   *problem,
                                                   *launches,
                                                   std::move(blocks),
                                                   std::move(split),
                                                   splitPlan);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        if(const auto* wrw = dynamic_cast<const PreparedConvGroupWrw*>(&prepared))
        {
            launchWrw(handle, *wrw, deviceBuffers, numDeviceBuffers, workspace);
            return;
        }
        const auto& dispatch = dynamic_cast<const PreparedConvGroup&>(prepared);
        const auto& problem = dispatch.problem();
        const auto& igemm = problem.igemm;
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "conv_group needs its workspace");
        }
        auto* x
            = hipdnn_plugin_sdk::findDeviceBuffer(igemm.xUid, deviceBuffers, numDeviceBuffers).ptr;
        auto* w
            = hipdnn_plugin_sdk::findDeviceBuffer(igemm.wUid, deviceBuffers, numDeviceBuffers).ptr;
        auto* y
            = hipdnn_plugin_sdk::findDeviceBuffer(igemm.yUid, deviceBuffers, numDeviceBuffers).ptr;
        const auto stream = handle.getStream();
        const bool forward = igemm.direction == ConvDirection::FORWARD;
        auto* base = static_cast<char*>(workspace);
        const std::array<void*, 2> graph{x, y};
        const std::array<const conv_igemm::Staging*, 2> stagings{&igemm.x, &igemm.y};
        // x is read forward and written backward; y the other way round.
        const std::array<bool, 2> read{forward, !forward};
        std::array<void*, 2> operand = graph;
        auto copy = [&](size_t i, bool in) {
            char* staged = base + stagings[i]->workspaceOffset;
            void* src = in ? graph[i] : staged;
            void* dst = in ? staged : graph[i];
            if(stagings[i]->transpose)
            {
                ConvIgemmTransposeArgs args = stagings[i]->transposeArgs;
                args.src = src;
                args.dst = dst;
                dispatch.copyForStream(i, stream).launch(stream, args);
            }
            else
            {
                ConvIgemmCopyArgs args = stagings[i]->copy;
                args.src = src;
                args.dst = dst;
                dispatch.copyForStream(i, stream).launch(stream, args);
            }
        };
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(stagings[i]->needed)
            {
                operand[i] = base + stagings[i]->workspaceOffset;
                if(read[i])
                {
                    copy(i, true);
                }
            }
        }
        // Every phase's filter expansion, then the gathers (one a phase, or one a depth phase for
        // the phase-fused kernel, reading its four phases' filters from the first one's).
        for(const auto& phase : problem.phases)
        {
            K4ExpandArgs expand = phase.expand;
            expand.src = w;
            expand.dst = base + phase.expandOffset;
            dispatch.expandForStream(stream).launch(stream, expand);
        }
        // Three-bf16: the kernel's input and every phase's expanded filter split into parts.
        const void* in = forward ? operand[0] : operand[1];
        const size_t firstExpand = problem.phases.front().expandOffset;
        char* splitFilter = nullptr;
        if(dispatch.split())
        {
            const auto& plan = dispatch.splitPlan();
            const auto split
                = conv_tf32::splitArgsFor(plan.plan, in, base + firstExpand, base + plan.offset);
            dispatch.splitForStream(stream).launch(stream, split);
            in = split.t[0].dst;
            splitFilter = reinterpret_cast<char*>(split.t[1].dst);
        }
        for(size_t i = 0; i < dispatch.launches().size(); ++i)
        {
            const auto& launch = dispatch.launches()[i];
            K4Args args = launch.args;
            args.in = in;
            args.w = base + problem.phases[launch.phase].expandOffset;
            if(splitFilter != nullptr)
            {
                // The phase's filter at its fp32 element offset, in bf16 parts.
                const auto& plan = dispatch.splitPlan().plan;
                const size_t element = (problem.phases[launch.phase].expandOffset - firstExpand)
                                       / static_cast<size_t>(igemm.elementBytes);
                args.w = splitFilter + element * 2;
                args.xps = plan.a.inner;
                args.wpt = plan.b.inner;
            }
            args.out = forward ? operand[1] : operand[0];
            auto& conv = dispatch.convForStream(stream);
            conv.setGridSize(dispatch.blocks()[i], 1, 1);
            conv.launch(stream, args);
        }
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(stagings[i]->needed && !read[i])
            {
                copy(i, false);
            }
        }
    }

private:
    std::unique_ptr<PreparedDispatch> prepareWrw(const MatchContext& context,
                                                 const BoundTokens& bound,
                                                 const KernelDefinition& kernel,
                                                 const Tile& tile) const
    {
        auto wrw = wrwFor(context, kernel, tile);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!wrw.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != wrw->first.wrw.xUid || *w != wrw->first.wrw.dwUid || *y != wrw->first.wrw.dyUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_group wrw dispatch was given a graph or config its matchers decline");
        }
        const auto& problem = wrw->first;
        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_GROUP_WRW", std::string("1"));
        options.add("CONV_GROUP_T", elementTypeFor(problem.wrw.base.dataType));
        if(tile.fma)
        {
            options.add("CONV_GROUP_FMA", std::string("1"));
            options.add("CONV_GROUP_CG", std::to_string(tile.cg));
            options.add("CONV_GROUP_PX", std::to_string(tile.px));
        }
        else
        {
            options.add("CONV_GROUP_ATOM", std::string(tile.info.kernelType));
        }
        if(tile.nbi != 0)
        {
            options.add("CONV_GROUP_WIDE", std::string("1"));
            options.add("CONV_GROUP_NBI", std::to_string(tile.nbi));
            options.add("CONV_GROUP_NBO", std::to_string(tile.nbo));
        }
        options.add("CONV_GROUP_R", std::to_string(problem.filterR));
        options.add("CONV_GROUP_S", std::to_string(problem.filterS));
        options.add("CONV_GROUP_ST", std::to_string(problem.stride));
        options.add("CONV_GROUP_TH", std::to_string(tile.th));
        options.add("CONV_GROUP_TW", std::to_string(tile.tw));
        options.add("CONV_GROUP_CB", std::to_string(tile.cb));
        options.add("CONV_GROUP_CF", std::to_string(tile.cf));
        auto conv = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, signatureOf<K4WrwArgs>());
        conv.setBlockSize(
            static_cast<unsigned int>(tile.fma ? FMA_THREADS : 4 * tile.info.wave), 1, 1);
        conv.setGridSize(wrw->second.grid, 1, 1);
        KernelDefinition reduceKernel = kernel;
        reduceKernel.source.entryPoint = std::string(WRW_REDUCE_ENTRY_POINT);
        auto reduce = buildIngestorKernelCode(_kernelCompiler,
                                              _kpackLoader,
                                              context,
                                              reduceKernel,
                                              options,
                                              signatureOf<K4WrwReduceArgs>());
        reduce.setBlockSize(COPY_BLOCK, 1, 1);
        reduce.setGridSize(wrw->second.reduceGrid, 1, 1);
        return std::make_unique<PreparedConvGroupWrw>(
            std::move(conv), std::move(reduce), std::move(wrw->first), wrw->second);
    }

    static void launchWrw(const Handle& handle,
                          const PreparedConvGroupWrw& dispatch,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace)
    {
        const auto& problem = dispatch.problem().wrw;
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "conv_group wrw needs its workspace");
        }
        const auto stream = handle.getStream();
        K4WrwArgs args = dispatch.wrwLaunch().args;
        args.x = hipdnn_plugin_sdk::findDeviceBuffer(problem.xUid, deviceBuffers, numDeviceBuffers)
                     .ptr;
        args.dy
            = hipdnn_plugin_sdk::findDeviceBuffer(problem.dyUid, deviceBuffers, numDeviceBuffers)
                  .ptr;
        args.partial = static_cast<float*>(workspace);
        dispatch.convForStream(stream).launch(stream, args);
        K4WrwReduceArgs reduce = dispatch.wrwLaunch().reduce;
        reduce.partial = static_cast<const float*>(workspace);
        reduce.dw
            = hipdnn_plugin_sdk::findDeviceBuffer(problem.dwUid, deviceBuffers, numDeviceBuffers)
                  .ptr;
        dispatch.reduceForStream(stream).launch(stream, reduce);
    }

    /// The three-bf16 split kernel of @p kernel's module (same options) for @p plan.
    IngestorKernelCode splitKernelCode(const MatchContext& context,
                                       const KernelDefinition& kernel,
                                       const compilation::KernelCompileOptions& options,
                                       const conv_group::GroupSplit& plan) const
    {
        KernelDefinition splitKernel = kernel;
        splitKernel.source.entryPoint = std::string(conv_tf32::SPLIT_ENTRY_POINT);
        auto split = buildIngestorKernelCode(_kernelCompiler,
                                             _kpackLoader,
                                             context,
                                             splitKernel,
                                             options,
                                             signatureOf<K1Split3Args>());
        split.setBlockSize(COPY_BLOCK, 1, 1);
        split.setGridSize(plan.plan.grid, 1, 1);
        return split;
    }

    static constexpr long long MAX_GRID_BLOCKS = std::numeric_limits<int>::max();
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convGroupKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvGroupDispatchHandler& convGroupDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convGroupKpackModuleCache());
    static const ConvGroupDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvGroupSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(WRW_GRAPH_MATCHER_SYMBOL), &wrwGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convGroupDispatchHandler());
}

void resetConvGroupModuleCache()
{
    convGroupKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
