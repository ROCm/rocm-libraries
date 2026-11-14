// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvGroupValuWrwProblem.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu_wrw
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace
{

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
/// Byte offsets within the images a pass spans stay below 2^31 (the kernels' 32-bit offsets and
/// their out-of-range marker 2^31).
constexpr int64_t OFFSET_LIMIT = int64_t{1} << 31;
/// Grid dimensions: tiles along x (2^31 - 1), splits along y (65535).
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

/// Bytes from a tensor's first element to one past its last: sum of (extent - 1) x stride over
/// n, c, h, w and the depth, plus one element. nullopt for a negative stride or an overflow.
// NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
std::optional<long long> extentBytes(const long long (&dims)[4],
                                     // NOLINTNEXTLINE(modernize-avoid-c-arrays) -- as above
                                     const long long (&strides)[4],
                                     long long depth,
                                     long long depthStride,
                                     int elementBytes)
{
    if(depthStride < 0)
    {
        return std::nullopt;
    }
    const auto depthTerm = multiply(depth - 1, depthStride);
    if(!depthTerm.has_value())
    {
        return std::nullopt;
    }
    int64_t last = *depthTerm;
    for(int axis = 0; axis < 4; ++axis)
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

/// The byte alignment the graph guarantees for tensor @p uid's pointer (its declared alignment),
/// at most 16; 0 when the tensor is missing.
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

/// The widest load (16, 8, 4 bytes, or the element) every lane's window start is aligned to:
/// it divides the window's bytes, every stride a pass offset is built from, and the pointer's
/// alignment.
int loadBytes(int64_t windowBytes,
              std::initializer_list<int64_t> strideBytes,
              int pointerAlign,
              int elementBytes)
{
    int vb = 16;
    while(vb > elementBytes)
    {
        bool ok = windowBytes % vb == 0 && pointerAlign % vb == 0;
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

} // namespace

std::optional<GvwProblem> parseGvwProblem(const MatchContext& context)
{
    auto base = conv_naive::parseConvNaiveProblem(context, ConvDirection::BACKWARD_WEIGHTS);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    // The sums are fp32: a compute type of fp32 (or none) only.
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
    }

    GvwProblem problem;
    problem.n = static_cast<int>(a.n);
    problem.groups = static_cast<int>(a.groups);
    problem.cpg = static_cast<int>(a.c / a.groups);
    problem.kpg = static_cast<int>(a.k / a.groups);
    problem.hi = static_cast<int>(a.inSpatial[1]);
    problem.wi = static_cast<int>(a.inSpatial[2]);
    problem.ho = static_cast<int>(a.outSpatial[1]);
    problem.wo = static_cast<int>(a.outSpatial[2]);
    problem.r = static_cast<int>(a.filter[1]);
    problem.s = static_cast<int>(a.filter[2]);
    problem.ph = static_cast<int>(a.prePadding[1]);
    problem.pw = static_cast<int>(a.prePadding[2]);
    problem.sh = static_cast<int>(a.stride[1]);
    problem.sw = static_cast<int>(a.stride[2]);
    problem.dh = static_cast<int>(a.dilation[1]);
    problem.dw = static_cast<int>(a.dilation[2]);
    problem.di = static_cast<int>(a.inSpatial[0]);
    problem.dOut = static_cast<int>(a.outSpatial[0]);
    problem.t = static_cast<int>(a.filter[0]);
    problem.pd = static_cast<int>(a.prePadding[0]);
    problem.sd = static_cast<int>(a.stride[0]);
    problem.dd = static_cast<int>(a.dilation[0]);
    // Below 3-D the depth is 1 and its strides are never multiplied by more than 0.
    const bool threeD = base->spatialRank == 3;
    problem.xsd = threeD ? a.xStrides[2] : 0;
    problem.ysd = threeD ? a.yStrides[2] : 0;
    problem.wsd = threeD ? a.wStrides[2] : 0;
    problem.flip = a.flipFilter != 0 ? 1 : 0;
    problem.elementBytes = base->dataType == data_objects::DataType::FLOAT ? 4 : 2;
    if(problem.sh < 1 || problem.sw < 1 || problem.dh < 1 || problem.dw < 1 || problem.r < 1
       || problem.s < 1 || problem.ho < 1 || problem.wo < 1 || problem.sd < 1 || problem.dd < 1
       || problem.t < 1 || problem.di < 1 || problem.dOut < 1)
    {
        return std::nullopt;
    }
    // (n, c, d, h, w) -> (n, c, h, w); the depth strides are apart (xsd, ysd, wsd).
    // NOLINTBEGIN(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
    const long long* strides[3] = {a.xStrides, a.wStrides, a.yStrides};
    long long* out[3] = {problem.xStr, problem.wStr, problem.yStr};
    // NOLINTEND(modernize-avoid-c-arrays)
    for(int t = 0; t < 3; ++t)
    {
        out[t][0] = strides[t][0];
        out[t][1] = strides[t][1];
        out[t][2] = strides[t][3];
        out[t][3] = strides[t][4];
    }
    // NOLINTBEGIN(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
    const long long xDims[4] = {a.n, a.c, problem.hi, problem.wi};
    const long long yDims[4] = {a.n, a.k, problem.ho, problem.wo};
    // NOLINTEND(modernize-avoid-c-arrays)
    const auto xBytes
        = extentBytes(xDims, problem.xStr, problem.di, problem.xsd, problem.elementBytes);
    const auto yBytes
        = extentBytes(yDims, problem.yStr, problem.dOut, problem.ysd, problem.elementBytes);
    if(!xBytes.has_value() || !yBytes.has_value())
    {
        return std::nullopt;
    }
    problem.xBytes = *xBytes;
    problem.yBytes = *yBytes;
    problem.xAlign = pointerAlignment(context, base->operands.x);
    problem.yAlign = pointerAlignment(context, base->operands.y);
    // The axis a lane's loads run along must be packed in both x and dy.
    problem.channelsFirst = problem.xStr[3] == 1 && problem.yStr[3] == 1;
    problem.channelsLast = problem.xStr[1] == 1 && problem.yStr[1] == 1;
    if(!problem.channelsFirst && !problem.channelsLast)
    {
        return std::nullopt;
    }
    problem.base = *base;
    return problem;
}

std::optional<Config> configOf(const KernelDefinition& kernel)
{
    Config config;
    config.layout = kernel.getStringMetadata("layout");
    config.cb = static_cast<int>(kernel.getIntMetadata("cb"));
    config.kb = static_cast<int>(kernel.getIntMetadata("kb"));
    config.p = static_cast<int>(kernel.getIntMetadata("p"));
    config.fr = static_cast<int>(kernel.getIntMetadata("fr"));
    config.fs = static_cast<int>(kernel.getIntMetadata("fs"));
    config.sw = static_cast<int>(kernel.getIntMetadata("sw"));
    config.dw = static_cast<int>(kernel.getIntMetadata("dw"));
    config.pw = static_cast<int>(kernel.getIntMetadata("pw"));
    config.nw = static_cast<int>(kernel.getIntMetadata("nw"));
    config.dot2 = static_cast<int>(kernel.getIntMetadata("dot2"));
    config.ring = static_cast<int>(kernel.getIntMetadata("ring"));
    if(config.layout != "nchw" && config.layout != "nhwc")
    {
        return std::nullopt;
    }
    const auto tileOk = [](int v) { return v == 1 || v == 2 || v == 4 || v == 8 || v == 16; };
    if(!tileOk(config.cb) || !tileOk(config.kb) || config.p < 1 || config.p > 16 || config.fr < 1
       || config.fr > 7 || config.fs < 1 || config.fs > 7 || config.sw < 1 || config.sw > 2
       || config.dw < 1 || config.dw > 4 || config.pw < 0
       || (config.nw != 1 && config.nw != 2 && config.nw != 4 && config.nw != 8))
    {
        return std::nullopt;
    }
    // The window a lane reads: pw to its left, its own p sw columns, and what the filter reaches
    // past them to its right, each within one neighbouring lane.
    const int own = config.p * config.sw;
    const int right
        = (config.p - 1) * config.sw + (config.fs - 1) * config.dw + 1 - config.pw - own;
    if(config.pw > own || right > own)
    {
        return std::nullopt;
    }
    if(config.cb * config.kb * config.fr * config.fs > MAX_ACCUMULATORS)
    {
        return std::nullopt;
    }
    if(config.dot2 != 0
       && (config.dot2 != 1 || config.layout != "nchw" || config.sw != 1 || config.p % 2 != 0))
    {
        return std::nullopt;
    }
    if(config.ring != 0 && (config.ring != 1 || config.layout != "nchw"))
    {
        return std::nullopt;
    }
    return config;
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

std::optional<Plan>
    planFor(const GvwProblem& problem, const Config& config, int wave, int64_t computeUnits)
{
    if(problem.r != config.fr || problem.s != config.fs || problem.sw != config.sw
       || problem.dw != config.dw || problem.pw != config.pw)
    {
        return std::nullopt;
    }
    const bool nchw = config.layout == "nchw";
    if(nchw ? !problem.channelsFirst : !problem.channelsLast)
    {
        return std::nullopt;
    }
    if(problem.cpg % config.cb != 0 || problem.kpg % config.kb != 0)
    {
        return std::nullopt;
    }
    if(wave != 32 && wave != 64)
    {
        return std::nullopt;
    }
    // Lanes a row: the lanes' own input columns cover the input row and their output columns the
    // output row, rounded up to a power of two (rows of a pass never straddle a lane group).
    const int own = config.p * config.sw;
    const int64_t need
        = std::max(divUp(problem.wi, own), divUp(problem.wo, static_cast<int64_t>(config.p)));
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
    const int64_t rowsAnImage = static_cast<int64_t>(problem.dOut) * problem.ho;
    const int64_t rows = static_cast<int64_t>(problem.n) * rowsAnImage;
    if(rows > INT_LIMIT)
    {
        return std::nullopt;
    }

    // Offsets within the images one pass spans (its rows reach at most this many images) stay
    // within 31 bits; the pass's first image is a 64-bit base.
    const int64_t images = std::min<int64_t>(problem.n, divUp(rowsPerPass - 1, rowsAnImage) + 1);
    const int es = problem.elementBytes;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
    const auto reach = [&](const long long(&str)[4],
                           int64_t channels,
                           int64_t depth,
                           long long depthStride,
                           int64_t h,
                           int64_t w) -> std::optional<int64_t> {
        const auto terms = {multiply(images, str[0]),
                            multiply(channels, str[1]),
                            multiply(depth, depthStride),
                            multiply(h, str[2]),
                            multiply(w, str[3])};
        int64_t sum = 0;
        for(const auto& t : terms)
        {
            if(!t.has_value())
            {
                return std::nullopt;
            }
            sum += *t;
        }
        return multiply(sum, es);
    };
    // Lanes past the row read a whole window (out of range: zeros) at most one window further.
    const auto xReach
        = reach(problem.xStr, config.cb, problem.di, problem.xsd, problem.hi, lanes * own + own);
    const auto yReach = reach(problem.yStr,
                              config.kb,
                              problem.dOut,
                              problem.ysd,
                              problem.ho,
                              lanes * config.p + config.p);
    if(!xReach.has_value() || !yReach.has_value() || *xReach >= OFFSET_LIMIT
       || *yReach >= OFFSET_LIMIT)
    {
        return std::nullopt;
    }

    Plan plan;
    plan.config = config;
    plan.wave = wave;
    plan.accumulators = config.cb * config.kb * config.fr * config.fs;
    // NCHW windows can run past a row's end when the width is not whole windows: those configs
    // load element by element (a wider load could straddle the tensor's end) and zero the
    // elements past the row. NHWC loads are per column, each checked.
    plan.ragged = nchw && (problem.wi % own != 0 || problem.wo % config.p != 0);
    plan.colmask = problem.wo % config.p != 0;
    if(nchw)
    {
        plan.vbx
            = plan.ragged
                  ? es
                  : loadBytes(static_cast<int64_t>(own) * es,
                              {problem.xStr[0] * es, problem.xStr[1] * es, problem.xStr[2] * es},
                              problem.xAlign,
                              es);
        plan.vby
            = plan.ragged
                  ? es
                  : loadBytes(static_cast<int64_t>(config.p) * es,
                              {problem.yStr[0] * es, problem.yStr[1] * es, problem.yStr[2] * es},
                              problem.yAlign,
                              es);
    }
    else
    {
        // A column's cb (kb) channels: the tile's first channel is a multiple of cb (kb) past
        // its group's, which is a multiple of cpg (kpg).
        plan.vbx = loadBytes(static_cast<int64_t>(config.cb) * es,
                             {problem.xStr[0] * es,
                              problem.xStr[2] * es,
                              problem.xStr[3] * es,
                              static_cast<int64_t>(problem.cpg) * es},
                             problem.xAlign,
                             es);
        plan.vby = loadBytes(static_cast<int64_t>(config.kb) * es,
                             {problem.yStr[0] * es,
                              problem.yStr[2] * es,
                              problem.yStr[3] * es,
                              static_cast<int64_t>(problem.kpg) * es},
                             problem.yAlign,
                             es);
    }
    if(problem.xAlign < es || problem.yAlign < es)
    {
        return std::nullopt;
    }
    // dot2 multiplies fp16 dword pairs: whole-dword windows, no ragged edge.
    if(config.dot2 != 0
       && (problem.base.dataType != data_objects::DataType::HALF || plan.ragged || plan.vbx < 4
           || plan.vby < 4))
    {
        return std::nullopt;
    }

    auto& a = plan.args;
    for(int i = 0; i < 4; ++i)
    {
        a.xs[i] = problem.xStr[i];
        a.ys[i] = problem.yStr[i];
        a.ws[i] = problem.wStr[i];
    }
    a.xBytes = problem.xBytes;
    a.yBytes = problem.yBytes;
    a.n = problem.n;
    a.groups = problem.groups;
    a.cpg = problem.cpg;
    a.kpg = problem.kpg;
    a.hi = problem.hi;
    a.wi = problem.wi;
    a.ho = problem.ho;
    a.wo = problem.wo;
    a.ph = problem.ph;
    a.sh = problem.sh;
    a.dh = problem.dh;
    a.lanes = static_cast<int>(lanes);
    a.rows = static_cast<int>(rows);
    a.cBlocks = problem.cpg / config.cb;
    a.kBlocks = problem.kpg / config.kb;
    // A tile per depth tap too: the grid's tiles are fd x the plane's.
    const int64_t tilesPlane = static_cast<int64_t>(problem.groups) * a.cBlocks * a.kBlocks;
    const int64_t tiles = tilesPlane * problem.t;
    if(tiles > INT_LIMIT)
    {
        return std::nullopt;
    }
    a.tiles = static_cast<int>(tiles);
    a.tilesPlane = static_cast<int>(tilesPlane);
    a.flip = problem.flip;
    a.di = problem.di;
    a.dOut = problem.dOut;
    a.pd = problem.pd;
    a.sd = problem.sd;
    a.dd = problem.dd;
    a.fd = problem.t;
    a.rowsAnImage = static_cast<int>(rowsAnImage);
    a.xsd = problem.xsd;
    a.ysd = problem.ysd;
    a.wsd = problem.wsd;

    if(config.ring != 0)
    {
        // The ring walks the rows of 2-D images at unit height stride and dilation.
        if(problem.sh != 1 || problem.dh != 1 || problem.t != 1 || problem.dOut != 1
           || problem.di != 1)
        {
            return std::nullopt;
        }
        const int64_t units = computeUnits > 0 ? computeUnits : 64;
        // Items (a lane group's run of rows of one image): about two lane groups a SIMD lane
        // slot of the device, runs of at least 8 rows.
        const int64_t wantGroups = units * 4 * rowsPerPass * 2;
        int64_t perImage = std::max<int64_t>(1, divUp(divUp(wantGroups, tiles), problem.n));
        perImage = std::min<int64_t>(perImage, std::max<int64_t>(1, problem.ho / 8));
        const int64_t chunk = divUp(problem.ho, perImage);
        perImage = divUp(problem.ho, chunk);
        const int64_t items = static_cast<int64_t>(problem.n) * perImage;
        if(items > INT_LIMIT)
        {
            return std::nullopt;
        }
        a.chunk = static_cast<int>(chunk);
        a.itemsPerImage = static_cast<int>(perImage);
        a.items = static_cast<int>(items);
    }

    // Splits: enough workgroups to fill the device twice over, each wave at least four passes.
    const int64_t units = computeUnits > 0 ? computeUnits : 64;
    const int64_t rowsPerWorkgroupMin = config.nw * rowsPerPass * 4;
    const int64_t maxSplits = std::max<int64_t>(1, divUp(rows, rowsPerWorkgroupMin));
    int64_t splits = std::clamp<int64_t>(divUp(2 * units, tiles), 1, maxSplits);
    const auto workspaceOf = [&](int64_t s) {
        return static_cast<size_t>(s) * static_cast<size_t>(tiles)
               * static_cast<size_t>(plan.accumulators) * sizeof(float);
    };
    while(splits > 1 && workspaceOf(splits) > WORKSPACE_CAP_BYTES)
    {
        splits = std::max<int64_t>(1, splits / 2);
    }
    splits = std::min(splits, GRID_Y_LIMIT);
    if(config.ring != 0)
    {
        // Splits over items: each workgroup a whole number of passes (rowsPerPass items each).
        const int64_t perWorkgroup = config.nw * rowsPerPass;
        splits = std::clamp<int64_t>(
            divUp(2 * units, tiles), 1, std::max<int64_t>(1, divUp(a.items, perWorkgroup)));
        while(splits > 1 && workspaceOf(splits) > WORKSPACE_CAP_BYTES)
        {
            splits = std::max<int64_t>(1, splits / 2);
        }
        splits = std::min(splits, GRID_Y_LIMIT);
        const int64_t itemsPerSplit = divUp(divUp(a.items, splits), rowsPerPass) * rowsPerPass;
        splits = divUp(a.items, itemsPerSplit);
        a.itemsPerSplit = static_cast<int>(itemsPerSplit);
    }
    // Whole passes a split, then the splits those take.
    int64_t const rowsPerSplit = divUp(divUp(rows, splits), rowsPerPass) * rowsPerPass;
    if(config.ring == 0)
    {
        splits = divUp(rows, rowsPerSplit);
    }
    a.rowsPerSplit = static_cast<int>(std::min<int64_t>(rowsPerSplit, INT_LIMIT));
    a.splits = static_cast<int>(splits);
    if(splits > 1)
    {
        plan.workspaceBytes = workspaceOf(splits);
        plan.foldGrid = divUp(tiles * plan.accumulators, 256);
        if(plan.foldGrid > INT_LIMIT)
        {
            return std::nullopt;
        }
    }
    return plan;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu_wrw

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
