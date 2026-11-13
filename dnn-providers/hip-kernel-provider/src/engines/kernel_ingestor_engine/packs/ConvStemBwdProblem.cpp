// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvStemBwdProblem.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_stem_bwd
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
constexpr int64_t MAX_GRID_YZ = 65535;
constexpr int64_t MAX_BLOCK = 1024;
/// The strides, filter extents and padding the window arithmetic takes.
constexpr int64_t MAX_GEOMETRY = 64;
/// The FMA loop's threads per super-row (SbFma::WAVE).
constexpr int64_t FMA_ROW_THREADS = 16;

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

/// floor(a / b) for b > 0.
int64_t floorDiv(int64_t a, int64_t b)
{
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

/// True when @p strides are packed for @p extents with the axes in @p order (slowest first);
/// extent-1 axes may carry any stride.
bool packedInOrder(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                   const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents,
                   const std::array<size_t, CONV_NAIVE_TENSOR_DIMS>& order)
{
    long long expected = 1;
    for(size_t i = CONV_NAIVE_TENSOR_DIMS; i-- > 0;)
    {
        const size_t axis = order[i];
        if(extents[axis] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= extents[axis];
    }
    return true;
}

constexpr std::array<size_t, CONV_NAIVE_TENSOR_DIMS> CHANNELS_FIRST_ORDER{0, 1, 2, 3, 4};
constexpr std::array<size_t, CONV_NAIVE_TENSOR_DIMS> CHANNELS_LAST_ORDER{0, 2, 3, 4, 1};

/// The atoms' shapes (DIM, KA), in ATOMS order.
constexpr std::array<AtomShape, ATOMS.size()> SHAPES{{
    {16, 16}, // mfma16_f16
    {16, 16}, // mfma16_bf16
    {16, 8}, // mfma16_bf16_908
    {16, 4}, // mfma16_f32
    {32, 8}, // mfma32_f16
    {32, 8}, // mfma32_bf16
    {32, 4}, // mfma32_bf16_908
    {32, 2}, // mfma32_f32
    {16, 32}, // mfma16x32_f16 (gfx950)
    {16, 32}, // mfma16x32_bf16 (gfx950)
    {32, 16}, // mfma32x16_f16 (gfx950)
    {32, 16}, // mfma32x16_bf16 (gfx950)
    {16, 16}, // wmma11_f16
    {16, 16}, // wmma11_bf16
    {16, 16}, // wmma12_f16
    {16, 16}, // wmma12_bf16
    {16, 4}, // fma: SbFma
}};

} // namespace

int64_t atomIdOf(std::string_view atom)
{
    const auto it = std::find(ATOMS.begin(), ATOMS.end(), atom);
    return it == ATOMS.end() ? -1 : static_cast<int64_t>(it - ATOMS.begin());
}

bool runsMatrixAtomFor(std::string_view element, std::string_view gcnArchName)
{
    return std::any_of(ATOMS.begin(), ATOMS.end(), [&](std::string_view atom) {
        return atom != "fma" && conv_igemm::atomElement(atom) == element
               && conv_igemm::atomRunsOn(atom, gcnArchName);
    });
}

AtomShape atomShapeOf(std::string_view atom)
{
    const auto id = atomIdOf(atom);
    return id < 0 ? AtomShape{} : SHAPES[static_cast<size_t>(id)];
}

Window windowOf(int64_t filter, int64_t stride, int64_t padding)
{
    Window window;
    window.offset = floorDiv(stride - 1 + padding, stride);
    // ceil((padding - filter + 1) / stride) = -floor((filter - 1 - padding) / stride).
    window.extent = window.offset + floorDiv(filter - 1 - padding, stride) + 1;
    return window;
}

SpaceToDepthWindow spaceToDepthWindowOf(int64_t filter, int64_t padding)
{
    const int64_t lo = padding - filter + 1;
    SpaceToDepthWindow window;
    window.shift = lo - floorDiv(lo, SPACE_TO_DEPTH) * SPACE_TO_DEPTH;
    window.offset = floorDiv(padding + 1 - window.shift, SPACE_TO_DEPTH);
    window.extent = window.offset - floorDiv(lo, SPACE_TO_DEPTH) + 1;
    return window;
}

std::optional<StemBwdProblem>
    parseStemBwdProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    auto base = conv_naive::parseConvNaiveProblem(context, ConvDirection::BACKWARD_DATA);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    int elementBytes = 0;
    switch(base->dataType)
    {
    case data_objects::DataType::FLOAT:
        elementBytes = 4;
        break;
    case data_objects::DataType::HALF:
    case data_objects::DataType::BFLOAT16:
        elementBytes = 2;
        break;
    default:
        return std::nullopt;
    }
    const auto& a = base->args;
    // One group of at most MAX_CHANNELS input channels.
    if(a.groups != 1 || a.c < 1 || a.c > MAX_CHANNELS)
    {
        return std::nullopt;
    }
    // The depth axis: trivial (its stride and dilation then change nothing; 2-D configs), or
    // unit stride and dilation with at most MAX_DEPTH_TAPS taps (3-D configs: the dx planes in
    // the grid, the taps in the reduction).
    const bool trivialDepth
        = a.inSpatial[0] == 1 && a.outSpatial[0] == 1 && a.filter[0] == 1 && a.prePadding[0] == 0;
    if(!trivialDepth
       && (a.stride[0] != 1 || a.dilation[0] != 1 || a.filter[0] < 1 || a.filter[0] > MAX_DEPTH_TAPS
           || a.prePadding[0] > MAX_GEOMETRY))
    {
        return std::nullopt;
    }
    for(int axis = 1; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        if(a.dilation[axis] != 1 || a.stride[axis] < 1 || a.stride[axis] > MAX_GEOMETRY
           || a.filter[axis] > MAX_GEOMETRY || a.prePadding[axis] > MAX_GEOMETRY)
        {
            return std::nullopt;
        }
    }
    // Extents and channels as the kernel's ints; the batch times dx's depth as the grid's z
    // (which, at unit depth stride, bounds both depths too).
    for(const long long value :
        {a.k, a.inSpatial[1], a.inSpatial[2], a.outSpatial[1], a.outSpatial[2]})
    {
        if(value > INT_LIMIT)
        {
            return std::nullopt;
        }
    }
    const auto images = pack_features::checkedProduct({a.n, a.inSpatial[0]});
    if(!images.has_value() || *images > MAX_GRID_YZ)
    {
        return std::nullopt;
    }
    const std::array<long long, CONV_NAIVE_TENSOR_DIMS> dxExtents{
        a.n, a.c, a.inSpatial[0], a.inSpatial[1], a.inSpatial[2]};
    const std::array<long long, CONV_NAIVE_TENSOR_DIMS> dyExtents{
        a.n, a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]};
    const std::array<long long, CONV_NAIVE_TENSOR_DIMS> wExtents{
        a.k, a.c, a.filter[0], a.filter[1], a.filter[2]};
    const bool dyChannelsFirst = packedInOrder(a.yStrides, dyExtents, CHANNELS_FIRST_ORDER);
    const bool dyChannelsLast = packedInOrder(a.yStrides, dyExtents, CHANNELS_LAST_ORDER);
    if(!dyChannelsFirst && !dyChannelsLast)
    {
        return std::nullopt;
    }
    if(!packedInOrder(a.xStrides, dxExtents, CHANNELS_FIRST_ORDER)
       && !packedInOrder(a.xStrides, dxExtents, CHANNELS_LAST_ORDER))
    {
        return std::nullopt;
    }
    if(!packedInOrder(a.wStrides, wExtents, CHANNELS_FIRST_ORDER)
       && !packedInOrder(a.wStrides, wExtents, CHANNELS_LAST_ORDER))
    {
        return std::nullopt;
    }
    // One image of dy is one 32-bit-addressed buffer.
    const auto dyImage
        = pack_features::checkedProduct({a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]});
    if(!dyImage.has_value() || *dyImage > INT_LIMIT / elementBytes)
    {
        return std::nullopt;
    }
    // Extents of at least 1: ceil(padding / stride) + floor((filter - 1 - padding) / stride)
    // >= 0 for any filter of at least one tap.
    const auto rows = windowOf(a.filter[1], a.stride[1], a.prePadding[1]);
    const auto cols = windowOf(a.filter[2], a.stride[2], a.prePadding[2]);
    StemBwdProblem problem;
    problem.elementBytes = elementBytes;
    problem.dyChannelsFirst = dyChannelsFirst;
    problem.dyChannelsLast = dyChannelsLast;
    problem.depthTaps = trivialDepth ? 0 : a.filter[0];
    if(a.stride[1] == 1 && a.stride[2] == 1)
    {
        const auto sdRows = spaceToDepthWindowOf(a.filter[1], a.prePadding[1]);
        const auto sdCols = spaceToDepthWindowOf(a.filter[2], a.prePadding[2]);
        problem.sdOh = sdRows.offset;
        problem.sdU = sdRows.extent;
        problem.sdShiftH = sdRows.shift;
        problem.sdOw = sdCols.offset;
        problem.sdV = sdCols.extent;
        problem.sdShiftW = sdCols.shift;
    }
    problem.oh = rows.offset;
    problem.u = rows.extent;
    problem.ow = cols.offset;
    problem.v = cols.extent;
    problem.dxUid = base->operands.x;
    problem.wUid = base->operands.w;
    problem.dyUid = base->operands.y;
    problem.base = std::move(*base);
    return problem;
}

int64_t ldsBytesOf(const StemBwdConfig& config, int elementBytes)
{
    // Every field as int, as the kernel instantiates it; a field outside a small range
    // declines first.
    for(const int64_t value : {config.u, config.v, config.nb, config.kc, config.th, config.fm})
    {
        if(value < 1 || value > 4096)
        {
            return std::numeric_limits<int64_t>::max();
        }
    }
    const int64_t dim = atomShapeOf(config.atom).dim;
    if(dim < 1)
    {
        return std::numeric_limits<int64_t>::max();
    }
    return sbLdsBytes(elementBytes,
                      static_cast<int>(config.u),
                      static_cast<int>(config.v),
                      static_cast<int>(config.nb),
                      static_cast<int>(config.kc),
                      static_cast<int>(config.th),
                      static_cast<int>(config.fm * dim));
}

int64_t ldsLimitOf(std::string_view gcnArchName)
{
    return hipdnn_plugin_sdk::stripArchFeatures(gcnArchName) == "gfx950" ? 160 * 1024 : 64 * 1024;
}

std::optional<StemBwdLaunch>
    stemBwdLaunchFor(const StemBwdProblem& problem,
                     const StemBwdConfig& config,
                     const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    // The atom: known, its id, on this device, taking the graph's element type. "fma" takes
    // fp32 only (16-bit runs on the matrix cores everywhere).
    const auto atomId = atomIdOf(config.atom);
    const auto info = conv_igemm::atomNamed(config.atom);
    if(atomId < 0 || atomId != config.atomId || !info.has_value())
    {
        return std::nullopt;
    }
    const bool fma = info->wave == 0;
    const std::string_view arch = device.gcnArchName;
    const auto element = conv_igemm::dataElement(problem.base.dataType);
    // FMA: fp32 everywhere, 16-bit only where no matrix atom takes the type (gfx906).
    if(!conv_igemm::atomRunsOn(config.atom, arch)
       || (fma ? element != "f32" && runsMatrixAtomFor(element, arch)
               : conv_igemm::atomElement(config.atom) != element))
    {
        return std::nullopt;
    }
    const auto shape = atomShapeOf(config.atom);
    const auto& a = problem.base.args;
    const int eb = problem.elementBytes;
    const int64_t ev = 16 / eb;

    // The layout and its vector.
    if(config.layout == LAYOUT_CHANNELS_LAST)
    {
        if(!problem.dyChannelsLast || config.vec != ev || a.k % ev != 0)
        {
            return std::nullopt;
        }
    }
    else if(config.layout == LAYOUT_CHANNELS_FIRST)
    {
        // Vectors of aligned columns: dy rows whole vectors. Space-to-depth reads them as
        // vectors wherever it can (1.07-1.19x over single elements, gfx1201): its scalar configs
        // serve only rows that are not whole vectors.
        const bool wholeRows = a.outSpatial[2] % ev == 0;
        if(!problem.dyChannelsFirst || (config.vec != 1 && config.vec != ev)
           || (config.vec == ev && !wholeRows) || (config.sd != 0 && config.vec == 1 && wholeRows))
        {
            return std::nullopt;
        }
    }
    else
    {
        return std::nullopt;
    }
    // The window, the strides, the depth taps and space-to-depth are compile-time. Space-to-
    // depth: stride 1 (the configs' sh and sw), no depth taps, its own window.
    if(config.sd != 0 && config.sd != SPACE_TO_DEPTH)
    {
        return std::nullopt;
    }
    const bool sd = config.sd != 0;
    if(sd && (problem.sdU == 0 || config.ft != 0))
    {
        return std::nullopt;
    }
    if(config.u != (sd ? problem.sdU : problem.u) || config.v != (sd ? problem.sdV : problem.v)
       || config.sh != a.stride[1] || config.sw != a.stride[2] || config.ft != problem.depthTaps)
    {
        return std::nullopt;
    }
    // The phases of a super-pixel.
    const int64_t phH = sd ? config.sd : a.stride[1];
    const int64_t phW = sd ? config.sd : a.stride[2];
    // The GEMM columns: every (channel, phase), whole fragments on a matrix atom.
    const int64_t columns = a.c * phH * phW;
    if(config.nb < columns || config.nb > MAX_COLUMNS || (!fma && config.nb % shape.dim != 0))
    {
        return std::nullopt;
    }
    // The chunk: whole atom steps and 16-byte units.
    if(config.kc < 1 || config.kc % shape.ka != 0 || config.kc % ev != 0)
    {
        return std::nullopt;
    }
    if(config.th < 1 || config.fm < 1)
    {
        return std::nullopt;
    }
    const int64_t threads = config.th * (fma ? FMA_ROW_THREADS : info->wave);
    if(threads > MAX_BLOCK || ldsBytesOf(config, eb) > ldsLimitOf(arch))
    {
        return std::nullopt;
    }
    // The grid: super-columns, super-rows, images.
    const int64_t tw = config.fm * shape.dim;
    const int64_t gridX = divUp(divUp(a.inSpatial[2], phW), tw);
    const int64_t gridY = divUp(divUp(a.inSpatial[1], phH), config.th);
    // 3-D: z is (image, dx plane).
    const int64_t planes = config.ft != 0 ? a.inSpatial[0] : 1;
    const auto gridZ = pack_features::checkedProduct({a.n, planes});
    // One dispatch addresses at most 2^32 - 1 work items along x.
    if(gridX > INT_LIMIT
       || gridX * threads > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())
       || gridY > MAX_GRID_YZ || !gridZ.has_value() || *gridZ > MAX_GRID_YZ)
    {
        return std::nullopt;
    }
    // The transformed filter: nb x u v x kp, one 32-bit-addressed buffer; 3-D: kp = taps x kpr;
    // space-to-depth: kp = sub-pixels x kpr. Space-to-depth from channels-first column vectors:
    // a chunk takes one sub-row and both sub-columns of kc / sd channels, so kp = sub-rows x
    // chunks of kc / sd channels x kc.
    // (kc is whole 16-byte units, so sd = 2 divides it.)
    const bool sdVectors = sd && config.layout == LAYOUT_CHANNELS_FIRST && config.vec > 1;
    const int64_t chunkChannels = sdVectors ? config.kc / config.sd : config.kc;
    const int64_t kpr = divUp(a.k, chunkChannels) * chunkChannels;
    const int64_t kp = sdVectors ? phH * divUp(a.k, chunkChannels) * config.kc
                                 : kpr * std::max<int64_t>(config.ft, 1) * (sd ? phH * phW : 1);
    const auto bElems = pack_features::checkedProduct({config.nb, config.u, config.v, kp});
    if(kp > INT_LIMIT || !bElems.has_value() || *bElems > INT_LIMIT / eb)
    {
        return std::nullopt;
    }

    StemBwdLaunch launch;
    launch.gridX = static_cast<unsigned int>(gridX);
    launch.gridY = static_cast<unsigned int>(gridY);
    launch.gridZ = static_cast<unsigned int>(*gridZ);
    launch.block = static_cast<unsigned int>(threads);
    launch.transformGrid = static_cast<unsigned int>(
        std::min(divUp(*bElems, int64_t{launch.transformBlock}), MAX_TRANSFORM_BLOCKS));
    launch.kernelType = info->kernelType;
    launch.workspaceBytes = static_cast<size_t>(*bElems * eb);

    auto& m = launch.args;
    m.n = static_cast<int>(a.n);
    m.c = static_cast<int>(a.c);
    m.k = static_cast<int>(a.k);
    m.kp = static_cast<int>(kp);
    m.hi = static_cast<int>(a.inSpatial[1]);
    m.wi = static_cast<int>(a.inSpatial[2]);
    m.ho = static_cast<int>(a.outSpatial[1]);
    m.wo = static_cast<int>(a.outSpatial[2]);
    m.oh = static_cast<int>(sd ? problem.sdOh : problem.oh);
    m.ow = static_cast<int>(sd ? problem.sdOw : problem.ow);
    // (n, k, h, w) and (n, c, h, w); the depth axis apart (2-D configs: trivial).
    const long long dyAxes[4] = {a.yStrides[0], a.yStrides[1], a.yStrides[3], a.yStrides[4]};
    const long long dxAxes[4] = {a.xStrides[0], a.xStrides[1], a.xStrides[3], a.xStrides[4]};
    std::copy(std::begin(dyAxes), std::end(dyAxes), m.dys);
    std::copy(std::begin(dxAxes), std::end(dxAxes), m.dxs);
    m.dyImage = a.k * a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2];
    m.d = static_cast<int>(planes);
    m.dout = static_cast<int>(a.outSpatial[0]);
    m.pd = static_cast<int>(a.prePadding[0]);
    m.kpr = static_cast<int>(kpr);
    m.dyd = a.yStrides[2];
    m.dxd = a.xStrides[2];
    m.sdh = static_cast<int>(sd ? problem.sdShiftH : 0);
    m.sdw = static_cast<int>(sd ? problem.sdShiftW : 0);

    auto& t = launch.transform;
    t.nb = static_cast<int>(config.nb);
    t.u = static_cast<int>(config.u);
    t.v = static_cast<int>(config.v);
    t.kp = static_cast<int>(kp);
    t.c = static_cast<int>(a.c);
    t.k = static_cast<int>(a.k);
    t.r = static_cast<int>(a.filter[1]);
    t.s = static_cast<int>(a.filter[2]);
    t.sh = static_cast<int>(phH);
    t.sw = static_cast<int>(phW);
    t.ph = static_cast<int>(a.prePadding[1]);
    t.pw = static_cast<int>(a.prePadding[2]);
    t.oh = m.oh;
    t.ow = m.ow;
    t.flip = a.flipFilter != 0 ? 1 : 0;
    const long long wAxes[4] = {a.wStrides[0], a.wStrides[1], a.wStrides[3], a.wStrides[4]};
    std::copy(std::begin(wAxes), std::end(wAxes), t.ws);
    t.ft = static_cast<int>(config.ft);
    t.kpr = static_cast<int>(kpr);
    t.wsd = a.wStrides[2];
    t.dh = m.sdh;
    t.dw = m.sdw;
    return launch;
}

std::vector<std::pair<std::string, std::string>> compileDefinesFor(data_objects::DataType dataType,
                                                                   const StemBwdConfig& config,
                                                                   std::string_view kernelType)
{
    std::string element;
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        element = "float";
        break;
    case data_objects::DataType::HALF:
        element = "_Float16";
        break;
    case data_objects::DataType::BFLOAT16:
        element = "unsigned short"; // bf16 bits
        break;
    default:
        throw std::invalid_argument("conv_stem_bwd: unsupported element type");
    }
    std::vector<std::pair<std::string, std::string>> defines{{"CONV_SB_T", element}};
    if(!kernelType.empty())
    {
        defines.emplace_back("CONV_SB_ATOM", std::string(kernelType));
    }
    const std::array<std::pair<const char*, int64_t>, 10> fields{{{"LAYOUT", config.layout},
                                                                  {"VEC", config.vec},
                                                                  {"U", config.u},
                                                                  {"V", config.v},
                                                                  {"NB", config.nb},
                                                                  {"KC", config.kc},
                                                                  {"TH", config.th},
                                                                  {"FM", config.fm},
                                                                  {"SH", config.sh},
                                                                  {"SW", config.sw}}};
    for(const auto& [name, value] : fields)
    {
        defines.emplace_back(std::string("CONV_SB_") + name, std::to_string(value));
    }
    // Only when on (3-D, space-to-depth): every other config compiles from the definitions it
    // always had.
    if(config.ft != 0)
    {
        defines.emplace_back("CONV_SB_FT", std::to_string(config.ft));
    }
    if(config.sd != 0)
    {
        defines.emplace_back("CONV_SB_SD", std::to_string(config.sd));
    }
    return defines;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_stem_bwd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
