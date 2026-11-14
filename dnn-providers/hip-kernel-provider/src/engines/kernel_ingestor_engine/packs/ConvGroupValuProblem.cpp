// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvGroupValuProblem.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace
{

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
/// Byte offsets within the images a pass spans stay below 2^31 (32-bit offsets and their
/// out-of-range marker 2^31).
constexpr int64_t OFFSET_LIMIT = int64_t{1} << 31;
constexpr int64_t GRID_Y_LIMIT = 65535;

bool fitsInt(long long value)
{
    return value >= 0 && value <= INT_LIMIT;
}

/// @p left * @p right for non-negative factors (zero allowed), or nullopt on overflow.
std::optional<int64_t> multiply(int64_t left, int64_t right)
{
    if(left < 0 || right < 0 || (right != 0 && left > std::numeric_limits<int64_t>::max() / right))
    {
        return std::nullopt;
    }
    return left * right;
}

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

/// Bytes from a tensor's first element to one past its last, or nullopt.
std::optional<long long> extentBytes(const std::array<long long, 5>& dims,
                                     const std::array<long long, 5>& strides,
                                     int elementBytes)
{
    int64_t last = 0;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        if(strides[axis] < 0)
        {
            return std::nullopt;
        }
        const auto term = multiply(dims[axis] - 1, strides[axis]);
        if(!term.has_value() || *term > std::numeric_limits<int64_t>::max() - last)
        {
            return std::nullopt;
        }
        last += *term;
    }
    const auto bytes = multiply(last + 1, elementBytes);
    if(!bytes.has_value())
    {
        return std::nullopt;
    }
    return static_cast<long long>(*bytes);
}

/// The byte alignment the graph declares for tensor @p uid's pointer, at most 16; 0 if missing.
int pointerAlignment(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto found = tensors.find(uid);
    if(found == tensors.end() || found->second == nullptr)
    {
        return 0;
    }
    const int64_t declared = found->second->alignment();
    int align = 16;
    while(align > 1 && declared % align != 0)
    {
        align /= 2;
    }
    return align;
}

/// The widest access (16, 8, 4 bytes or the element) every lane's run is aligned to.
int accessBytes(int64_t runBytes,
                std::initializer_list<int64_t> strideBytes,
                int pointerAlign,
                int elementBytes)
{
    int vb = 16;
    while(vb > elementBytes)
    {
        bool ok = runBytes % vb == 0 && pointerAlign % vb == 0;
        for(const int64_t s : strideBytes)
        {
            ok = ok && s % vb == 0;
        }
        if(ok)
        {
            break;
        }
        vb /= 2;
    }
    return vb;
}

int waveOf(std::string_view gcnArchName, int reportedWarpSize)
{
    if(reportedWarpSize == 32 || reportedWarpSize == 64)
    {
        return reportedWarpSize;
    }
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    return arch.size() >= 5 && arch.substr(0, 4) == "gfx1" ? 32 : 64;
}

} // namespace

std::optional<GvProblem> parseGvProblem(const MatchContext& context, ConvDirection direction)
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
    // fp32 sums: a compute type of fp32 (or none) only.
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    if(computeType != data_objects::DataType::UNSET && computeType != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    if(base->spatialRank < 1 || base->spatialRank > 3)
    {
        return std::nullopt;
    }
    const auto& a = base->args;
    for(const long long value : {a.n, a.c, a.k, a.groups})
    {
        if(!fitsInt(value) || value < 1)
        {
            return std::nullopt;
        }
    }
    GvProblem problem;
    problem.direction = direction;
    for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
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
        const auto i = static_cast<size_t>(axis);
        problem.in.at(i) = static_cast<int>(a.inSpatial[axis]);
        problem.out.at(i) = static_cast<int>(a.outSpatial[axis]);
        problem.filter.at(i) = static_cast<int>(a.filter[axis]);
        problem.pad.at(i) = static_cast<int>(a.prePadding[axis]);
        problem.stride.at(i) = static_cast<int>(a.stride[axis]);
        problem.dilation.at(i) = static_cast<int>(a.dilation[axis]);
        if(problem.stride.at(i) < 1 || problem.dilation.at(i) < 1 || problem.filter.at(i) < 1
           || problem.out.at(i) < 1 || problem.in.at(i) < 1)
        {
            return std::nullopt;
        }
    }
    problem.n = static_cast<int>(a.n);
    problem.groups = static_cast<int>(a.groups);
    problem.cpg = static_cast<int>(a.c / a.groups);
    problem.kpg = static_cast<int>(a.k / a.groups);
    problem.flip = a.flipFilter != 0 ? 1 : 0;
    problem.elementBytes = base->dataType == data_objects::DataType::FLOAT ? 4 : 2;
    for(size_t i = 0; i < 5; ++i)
    {
        problem.xStr.at(i) = a.xStrides[i];
        problem.yStr.at(i) = a.yStrides[i];
        problem.wStr.at(i) = a.wStrides[i];
    }
    const std::array<long long, 5> xDims{a.n, a.c, a.inSpatial[0], a.inSpatial[1], a.inSpatial[2]};
    const std::array<long long, 5> yDims{
        a.n, a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]};
    const auto xBytes = extentBytes(xDims, problem.xStr, problem.elementBytes);
    const auto yBytes = extentBytes(yDims, problem.yStr, problem.elementBytes);
    if(!xBytes.has_value() || !yBytes.has_value())
    {
        return std::nullopt;
    }
    problem.xBytes = *xBytes;
    problem.yBytes = *yBytes;
    problem.channelsFirst = problem.xStr[4] == 1 && problem.yStr[4] == 1;
    problem.channelsLast = problem.xStr[1] == 1 && problem.yStr[1] == 1;
    if(!problem.channelsFirst && !problem.channelsLast)
    {
        return std::nullopt;
    }
    problem.xAlign = pointerAlignment(context, base->operands.x);
    problem.yAlign = pointerAlignment(context, base->operands.y);
    problem.base = *base;
    return problem;
}

std::optional<Config> configOf(const KernelDefinition& kernel)
{
    Config config;
    config.direction = kernel.getStringMetadata("direction");
    config.layout = kernel.getStringMetadata("layout");
    config.kb = static_cast<int>(kernel.getIntMetadata("kb"));
    config.cb = static_cast<int>(kernel.getIntMetadata("cb"));
    config.p = static_cast<int>(kernel.getIntMetadata("p"));
    config.fd = static_cast<int>(kernel.getIntMetadata("fd"));
    config.fr = static_cast<int>(kernel.getIntMetadata("fr"));
    config.fs = static_cast<int>(kernel.getIntMetadata("fs"));
    config.sw = static_cast<int>(kernel.getIntMetadata("sw"));
    config.dw = static_cast<int>(kernel.getIntMetadata("dw"));
    config.pw = static_cast<int>(kernel.getIntMetadata("pw"));
    config.nw = static_cast<int>(kernel.getIntMetadata("nw"));
    if((config.direction != "FWD" && config.direction != "BWD")
       || (config.layout != "nchw" && config.layout != "nhwc"))
    {
        return std::nullopt;
    }
    const auto pow2 = [](int v, int most) { return v >= 1 && v <= most && (v & (v - 1)) == 0; };
    if(!pow2(config.kb, 16) || !pow2(config.cb, 4) || config.p < 1 || config.p > 16
       || config.kb * config.p > MAX_ACCUMULATORS || config.fd < 1 || config.fd > 7 || config.fr < 1
       || config.fr > 7 || config.fs < 1 || config.fs > 7 || config.sw < 1 || config.sw > 2
       || (config.direction == "BWD" && config.p % config.sw != 0) || config.dw < 1 || config.dw > 4
       || config.pw < 0 || !pow2(config.nw, 8))
    {
        return std::nullopt;
    }
    return config;
}

std::optional<Plan>
    planFor(const GvProblem& problem, const Config& config, int wave, int64_t computeUnits)
{
    const bool bwd = problem.direction == ConvDirection::BACKWARD_DATA;
    if((config.direction == "BWD") != bwd)
    {
        return std::nullopt;
    }
    if(problem.filter[0] != config.fd || problem.filter[1] != config.fr
       || problem.filter[2] != config.fs || problem.stride[2] != config.sw
       || problem.dilation[2] != config.dw || problem.pad[2] != config.pw)
    {
        return std::nullopt;
    }
    const bool nchw = config.layout == "nchw";
    if(nchw ? !problem.channelsFirst : !problem.channelsLast)
    {
        return std::nullopt;
    }
    if(wave != 32 && wave != 64)
    {
        return std::nullopt;
    }
    // Kernel roles.
    const int ci = bwd ? problem.kpg : problem.cpg;
    const int ko = bwd ? problem.cpg : problem.kpg;
    if(ko % config.kb != 0 || ci % config.cb != 0)
    {
        return std::nullopt;
    }
    const auto& kin = bwd ? problem.out : problem.in;
    const auto& kout = bwd ? problem.in : problem.out;
    std::array<int, 3> kpad{};
    for(size_t i = 0; i < 3; ++i)
    {
        const int span = (problem.filter.at(i) - 1) * problem.dilation.at(i) + 1;
        kpad.at(i) = bwd ? span - 1 - problem.pad.at(i) : problem.pad.at(i);
        if(kpad.at(i) < 0)
        {
            return std::nullopt;
        }
    }
    // The kernel's width: forward strides through in by sw; backward data reads dy upsampled by
    // sw at unit stride. A lane's own input columns, and the window's reach to either side (each
    // within one neighbouring lane).
    const int kernelStride = bwd ? 1 : config.sw;
    const int upsample = bwd ? config.sw : 1;
    const auto floorDiv = [](int v, int d) { return v >= 0 ? v / d : -((-v + d - 1) / d); };
    const int own = config.p * kernelStride / upsample;
    const int left = -floorDiv(-kpad[2], upsample);
    const int right
        = floorDiv((config.p - 1) * kernelStride + (config.fs - 1) * config.dw - kpad[2], upsample)
          + 1 - own;
    if(own < 1 || left > own || right > own)
    {
        return std::nullopt;
    }
    const int64_t need
        = std::max(divUp(kin[2], own), divUp(kout[2], static_cast<int64_t>(config.p)));
    int64_t lanes = 1;
    while(lanes < need)
    {
        lanes *= 2;
    }
    if(lanes > wave)
    {
        return std::nullopt;
    }
    const int64_t rowsPerPass = wave / lanes;
    const int64_t rowsAnImage = static_cast<int64_t>(kout[0]) * kout[1];
    const int64_t rows = problem.n * rowsAnImage;
    if(rows > INT_LIMIT)
    {
        return std::nullopt;
    }
    const auto& inStr = bwd ? problem.yStr : problem.xStr;
    const auto& outStr = bwd ? problem.xStr : problem.yStr;
    const int es = problem.elementBytes;
    // Offsets within the images one pass spans stay within 31 bits.
    const int64_t images = std::min<int64_t>(problem.n, divUp(rowsPerPass - 1, rowsAnImage) + 1);
    const auto reach = [&](const std::array<long long, 5>& str,
                           int64_t channels,
                           const std::array<int, 3>& ext,
                           int64_t w) -> std::optional<int64_t> {
        int64_t sum = 0;
        for(const auto& t : {multiply(images, str[0]),
                             multiply(channels, str[1]),
                             multiply(ext[0], str[2]),
                             multiply(ext[1], str[3]),
                             multiply(w, str[4])})
        {
            if(!t.has_value())
            {
                return std::nullopt;
            }
            sum += *t;
        }
        return multiply(sum, es);
    };
    const auto inReach = reach(inStr, ci, kin, lanes * own + own);
    const auto outReach = reach(outStr, ko, kout, lanes * config.p + config.p);
    if(!inReach.has_value() || !outReach.has_value() || *inReach >= OFFSET_LIMIT
       || *outReach >= OFFSET_LIMIT)
    {
        return std::nullopt;
    }
    const int inAlign = bwd ? problem.yAlign : problem.xAlign;
    const int outAlign = bwd ? problem.xAlign : problem.yAlign;
    if(inAlign < es || outAlign < es)
    {
        return std::nullopt;
    }

    Plan plan;
    plan.config = config;
    plan.wave = wave;
    plan.kernelPw = kpad[2];
    plan.ragged = nchw && (kin[2] % own != 0 || kout[2] % config.p != 0);
    plan.colmask = kout[2] % config.p != 0;
    if(nchw)
    {
        plan.vbx = plan.ragged
                       ? es
                       : accessBytes(static_cast<int64_t>(own) * es,
                                     {inStr[0] * es, inStr[1] * es, inStr[2] * es, inStr[3] * es},
                                     inAlign,
                                     es);
        plan.vby
            = plan.colmask
                  ? es
                  : accessBytes(static_cast<int64_t>(config.p) * es,
                                {outStr[0] * es, outStr[1] * es, outStr[2] * es, outStr[3] * es},
                                outAlign,
                                es);
    }
    else
    {
        plan.vbx = accessBytes(static_cast<int64_t>(config.cb) * es,
                               {inStr[0] * es,
                                inStr[2] * es,
                                inStr[3] * es,
                                inStr[4] * es,
                                static_cast<int64_t>(ci) * es},
                               inAlign,
                               es);
        plan.vby = accessBytes(static_cast<int64_t>(config.kb) * es,
                               {outStr[0] * es,
                                outStr[2] * es,
                                outStr[3] * es,
                                outStr[4] * es,
                                static_cast<int64_t>(ko) * es},
                               outAlign,
                               es);
    }

    auto& a = plan.args;
    for(size_t i = 0; i < 5; ++i)
    {
        a.is[i] = inStr.at(i);
        a.os[i] = outStr.at(i);
        a.ws[i] = problem.wStr.at(i);
    }
    a.inBytes = bwd ? problem.yBytes : problem.xBytes;
    a.outBytes = bwd ? problem.xBytes : problem.yBytes;
    a.n = problem.n;
    a.groups = problem.groups;
    a.ci = ci;
    a.ko = ko;
    a.di = kin[0];
    a.hi = kin[1];
    a.wi = kin[2];
    a.dOut = kout[0];
    a.ho = kout[1];
    a.wo = kout[2];
    a.pd = kpad[0];
    a.ph = kpad[1];
    a.sd = bwd ? 1 : problem.stride[0];
    a.sh = bwd ? 1 : problem.stride[1];
    a.ud = bwd ? problem.stride[0] : 1;
    a.uh = bwd ? problem.stride[1] : 1;
    a.dd = problem.dilation[0];
    a.dh = problem.dilation[1];
    a.lanes = static_cast<int>(lanes);
    a.rows = static_cast<int>(rows);
    a.kBlocks = ko / config.kb;
    a.wGraphK = problem.kpg;
    a.wGraphC = problem.cpg;
    a.backward = bwd ? 1 : 0;
    a.flip = problem.flip;
    const int64_t blocksY = static_cast<int64_t>(problem.groups) * a.kBlocks;
    if(blocksY > GRID_Y_LIMIT)
    {
        return std::nullopt;
    }
    const auto prepared = multiply(
        blocksY * ci, static_cast<int64_t>(config.fd) * config.fr * config.fs * config.kb);
    if(!prepared.has_value() || *prepared > INT_LIMIT)
    {
        return std::nullopt;
    }
    a.prepared = static_cast<int>(*prepared);
    plan.prepareGrid = divUp(*prepared, 256);
    plan.workspaceBytes = static_cast<size_t>(*prepared) * sizeof(float);
    // Row blocks: about four workgroups a dispatch unit, each wave at least one pass.
    const int64_t units = computeUnits > 0 ? computeUnits : 64;
    const int64_t perBlock = config.nw * rowsPerPass;
    int64_t rowBlocks = std::clamp<int64_t>(
        divUp(4 * units, blocksY), 1, std::max<int64_t>(1, divUp(rows, perBlock)));
    const int64_t rowsPerBlock = divUp(divUp(rows, rowBlocks), rowsPerPass) * rowsPerPass;
    rowBlocks = divUp(rows, rowsPerBlock);
    if(rowBlocks > INT_LIMIT)
    {
        return std::nullopt;
    }
    a.rowsPerBlock = static_cast<int>(rowsPerBlock);
    a.rowBlocks = static_cast<int>(rowBlocks);
    return plan;
}

int planWave(std::string_view gcnArchName, int reportedWarpSize)
{
    return waveOf(gcnArchName, reportedWarpSize);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
