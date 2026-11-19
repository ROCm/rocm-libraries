// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "engines/kernel_ingestor_engine/packs/ConvWinogradNdProblem.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <limits>
namespace hip_kernel_provider::kernel_ingestor_engine::conv_winograd_nd
{
namespace
{
bool mul(long long a, long long b, long long& out)
{
    if(a < 0 || b < 0 || (b != 0 && a > LLONG_MAX / b))
    {
        return false;
    }
    out = a * b;
    return true;
}
long long ceilDiv(long long a, long long b)
{
    return a / b + (a % b != 0 ? 1 : 0);
}
size_t aligned(size_t a)
{
    return (a + 255) / 256 * 256;
}
// Generic views are admitted when positive strides prove a nonoverlapping rectangular view.
// This conservative sorted-stride proof supports packed, padded and mixed layouts, not broadcasts.
bool safeView(const long long* strides, const long long* dims, int bytes)
{
    std::array<int, 5> order = {0, 1, 2, 3, 4};
    std::sort(order.begin(), order.end(), [&](int x, int y) { return strides[x] < strides[y]; });
    long long span = 1;
    for(const int j : order)
    {
        if(dims[j] == 1)
        {
            continue;
        }
        if(strides[j] < span)
        {
            return false;
        }
        long long extent = 0;
        if(!mul(dims[j] - 1, strides[j], extent) || extent > LLONG_MAX - span)
        {
            return false;
        }
        span += extent;
    }
    return span <= LLONG_MAX / bytes;
}
}
std::optional<Problem> parseProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                                    conv_naive::ConvDirection direction)
{
    if(direction != conv_naive::ConvDirection::FORWARD)
    {
        return std::nullopt;
    }
    auto base = conv_naive::parseConvNaiveProblem(context, direction);
    if(!base || base->spatialRank != 2)
    {
        return std::nullopt;
    }
    using D = hipdnn_flatbuffers_sdk::data_objects::DataType;
    if(base->dataType != D::FLOAT)
    {
        return std::nullopt;
    }
    const auto& b = base->args;
    if(b.dilation[1] == 1 && b.dilation[2] == 1)
    {
        return std::nullopt;
    }
    if(b.c > INT_MAX - 128 || b.k > INT_MAX - 128 || b.groups > 65535)
    {
        return std::nullopt;
    }
    for(int j = 0; j < 3; ++j)
    {
        if(b.stride[j] != 1 || b.dilation[j] <= 0)
        {
            return std::nullopt;
        }
        if(b.filter[j] != (j == 0 ? 1 : 3))
        {
            return std::nullopt;
        }
        // Origins include padded partial tiles and their last input tap. Bound every signed gather.
        if(b.inSpatial[j] > LLONG_MAX / 8 || b.outSpatial[j] > LLONG_MAX / 8
           || b.dilation[j] > LLONG_MAX / 8 || b.prePadding[j] > LLONG_MAX / 8)
        {
            return std::nullopt;
        }
    }
    const std::array<long long, 5> xd = {b.n, b.c, b.inSpatial[0], b.inSpatial[1], b.inSpatial[2]};
    const std::array<long long, 5> yd
        = {b.n, b.k, b.outSpatial[0], b.outSpatial[1], b.outSpatial[2]};
    const std::array<long long, 5> wd
        = {b.k, b.c / b.groups, b.filter[0], b.filter[1], b.filter[2]};
    const int bytes = 4;
    if(!safeView(b.xStrides, xd.data(), bytes) || !safeView(b.yStrides, yd.data(), bytes)
       || !safeView(b.wStrides, wd.data(), bytes))
    {
        return std::nullopt;
    }
    Problem p;
    p.base = *base;
    auto& a = p.args;
    a.n = b.n;
    a.ci = b.c;
    a.co = b.k;
    a.groups = b.groups;
    a.rotate = b.flipFilter != 0 ? 1 : 0;
    for(int j = 0; j < 2; ++j)
    {
        a.input[j] = b.inSpatial[j + 1];
        a.output[j] = b.outSpatial[j + 1];
        a.dilation[j] = b.dilation[j + 1];
        a.pad[j] = b.prePadding[j + 1];
    }
    for(int j = 0; j < 4; ++j)
    {
        const int role = j < 2 ? j : j + 1;
        a.inStride[j] = b.xStrides[role];
        a.outStride[j] = b.yStrides[role];
        a.wStride[j] = b.wStrides[role];
    }
    return p;
}
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& k)
{
    try
    {
        Config c;
        const auto get = [&](const char* name, int& value) {
            const auto v = k.getIntMetadata(name);
            if(v < 0 || v > INT_MAX)
            {
                return false;
            }
            value = static_cast<int>(v);
            return true;
        };
        if(!get("atom", c.atom) || !get("gemm_bm", c.bm) || !get("gemm_bn", c.bn)
           || !get("gemm_bk", c.bk) || !get("chunk_tiles", c.chunkTiles))
        {
            return std::nullopt;
        }
        c.arch = k.getStringMetadata("arch");
        if(k.getIntMetadata("precision") != 0 || c.chunkTiles < 1 || c.chunkTiles > INT_MAX - 128)
        {
            return std::nullopt;
        }
        if(c.atom == 5 && c.arch == "gfx1250" && (c.bm == 64 || c.bm == 128) && c.bn == 64
           && c.bk == 16)
        {
            return c;
        }
    }
    catch(const std::exception&)
    {
        // A malformed record (a missing or mistyped field) is not a config.
        return std::nullopt;
    }
    return std::nullopt;
}
bool configRunsOn(const Config& c, std::string_view arch)
{
    return c.atom == 5 && c.arch == "gfx1250"
           && hipdnn_plugin_sdk::stripArchFeatures(arch) == "gfx1250";
}
std::optional<Plan> planFor(const Problem& p, const Config& c, size_t limit)
{
    if(p.base.spatialRank != 2 || p.base.direction != conv_naive::ConvDirection::FORWARD
       || p.base.dataType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
       || (p.args.dilation[0] == 1 && p.args.dilation[1] == 1) || c.atom != 5 || c.arch != "gfx1250"
       || (c.bm != 64 && c.bm != 128) || c.bn != 64 || c.bk != 16 || c.chunkTiles < 1
       || c.chunkTiles > INT_MAX - 128)
    {
        return std::nullopt;
    }
    Plan plan;
    plan.args = p.args;
    auto& a = plan.args;
    plan.bins = 16;
    long long total = a.n;
    for(int j = 0; j < 2; ++j)
    {
        a.residues[j] = std::min(a.dilation[j], a.output[j]);
        if(!mul(a.residues[j], ceilDiv(ceilDiv(a.output[j], a.dilation[j]), 2), a.axisTiles[j])
           || !mul(total, a.axisTiles[j], total))
        {
            return std::nullopt;
        }
    }
    plan.totalTiles = total;
    if(a.groups > 65535 / plan.bins)
    {
        return std::nullopt;
    }
    const long long binBytes = static_cast<long long>(plan.bins) * 4;
    long long u = 0;
    long long per = 0;
    if(!mul(a.co, a.ci / a.groups, u) || !mul(u, binBytes, u) || !mul(a.ci + a.co, binBytes, per))
    {
        return std::nullopt;
    }
    if(static_cast<unsigned long long>(u) > std::numeric_limits<size_t>::max() - 255 || limit < 512
       || static_cast<unsigned long long>(u) > limit - 512)
    {
        return std::nullopt;
    }
    plan.vOffset = aligned(static_cast<size_t>(u));
    const size_t capacity = (limit - plan.vOffset - 512) / static_cast<size_t>(per);
    if(capacity < 1)
    {
        return std::nullopt;
    }
    a.tiles = static_cast<int>(
        std::min<unsigned long long>({static_cast<unsigned long long>(total),
                                      static_cast<unsigned long long>(c.chunkTiles),
                                      capacity}));
    const auto v = static_cast<size_t>(a.ci * plan.bins * a.tiles * 4);
    const auto m = static_cast<size_t>(a.co * plan.bins * a.tiles * 4);
    plan.mOffset = plan.vOffset + aligned(v);
    plan.workspaceBytes = plan.mOffset + aligned(m);
    if(plan.workspaceBytes > limit)
    {
        return std::nullopt;
    }
    const auto blocks = [](long long work) {
        return static_cast<unsigned int>(std::min(1048576LL, ceilDiv(work, 256)));
    };
    plan.filterBlocks = blocks(a.co * (a.ci / a.groups));
    plan.inputBlocks = blocks(a.ci * a.tiles);
    plan.outputBlocks = blocks(a.co * a.tiles);
    plan.gemmBlocks[0] = static_cast<unsigned int>(ceilDiv(a.tiles, c.bn));
    plan.gemmBlocks[1] = static_cast<unsigned int>(ceilDiv(a.co / a.groups, c.bm));
    plan.gemmBlocks[2] = static_cast<unsigned int>(a.groups * plan.bins);
    if(plan.gemmBlocks[1] > 65535)
    {
        return std::nullopt;
    }
    return plan;
}
}
#endif
