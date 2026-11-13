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
#include <variant>
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
#include "engines/kernel_ingestor_engine/descriptors/conv/direct/kernels/ConvDirectArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvDirectProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvDirectNative.cpp
 * @brief The native half of the stride-1 halo-tile direct convolution engines (CONV_DIRECT_FWD,
 *        CONV_DIRECT_BWD): graph matching, kernel matching, the tile planner, workspace sizing
 *        and dispatch.
 *
 * One HIPRTC source serves every architecture: a block stages its input halo in LDS once per
 * channel chunk and runs each filter tap as an MMA (MFMA on gfx908 to gfx950, WMMA on gfx11 and
 * gfx12; the implicit-GEMM core's atoms) or as FMA micro-tiles anywhere, in fp32, fp16 and bf16,
 * 2-D and 3-D, stride 1. Shapes are runtime; a catalog config fixes the filter extent, the atom
 * and the tile. A weight pre-pass formats the graph's filter (any layout, reflected for true
 * convolution, rotated and channel-swapped for backward data) into the fragment order a config
 * reads; activations that are not channels-last are staged through the workspace. Every config is
 * admitted on correctness and its ship list (the arch groups whose sweeps it won); autotune (UHD
 * later) ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace conv_direct
{

namespace
{

constexpr long long INT_LIMIT = std::numeric_limits<int>::max();
constexpr long long BUFFER_LIMIT = std::numeric_limits<int>::max();
constexpr size_t WORKSPACE_ALIGNMENT = 256;
constexpr long long MAX_GRID_YZ = 65535;
constexpr long long MAX_SPLITS = 16;
constexpr int SPATIAL = CONV_NAIVE_SPATIAL_DIMS;

long long divUp(long long value, long long divisor)
{
    return (value + divisor - 1) / divisor;
}

size_t alignUp(size_t value)
{
    return (value + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

bool fitsInt(long long value)
{
    return value >= -INT_LIMIT && value <= INT_LIMIT;
}

/// True when @p strides (n, c, d, h, w) are packed channels-last for @p extents; extent-1 axes
/// may carry any stride.
bool channelsLast(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                  const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents)
{
    long long expected = 1;
    for(const int axis : {1, 4, 3, 2, 0})
    {
        if(extents[static_cast<size_t>(axis)] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= extents[static_cast<size_t>(axis)];
    }
    return true;
}

/// True when @p strides (n, c, d, h, w) are packed channels-first for @p extents.
bool channelsFirst(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                   const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents)
{
    long long expected = 1;
    for(const int axis : {4, 3, 2, 1, 0})
    {
        if(extents[static_cast<size_t>(axis)] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= extents[static_cast<size_t>(axis)];
    }
    return true;
}

ConvIgemmCopyArgs copyOf5(const std::array<long long, 5>& dims,
                          const std::array<long long, 5>& srcStrides,
                          const std::array<long long, 5>& dstStrides,
                          int elementBytes)
{
    ConvIgemmCopyArgs copy{};
    copy.rank = 5;
    copy.total = 1;
    for(size_t i = 0; i < 5; ++i)
    {
        copy.dims[i] = dims[i];
        copy.srcStrides[i] = srcStrides[i];
        copy.dstStrides[i] = dstStrides[i];
        copy.total *= dims[i];
    }
    copy.elementBytes = elementBytes;
    return copy;
}

/// The activation @p strides (n, c, d, h, w) staged into (in) or out of (out) the workspace's
/// packed channels-last copy: by a tiled transpose when packed channels-first, else by the
/// strided copy.
Staging activationStaging(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                          long long n,
                          long long c,
                          const long long (&spatial)[SPATIAL],
                          bool in,
                          int elementBytes)
{
    Staging staging;
    staging.needed = true;
    const long long d = spatial[0], h = spatial[1], w = spatial[2];
    staging.bytes = static_cast<size_t>(n * d * h * w * c) * static_cast<size_t>(elementBytes);
    if(channelsFirst(strides, {n, c, d, h, w}))
    {
        // Per image a [C][DHW] <-> [DHW][C] transpose.
        staging.transpose = true;
        auto& t = staging.transposeArgs;
        t.batches = n;
        t.rows = in ? c : d * h * w;
        t.cols = in ? d * h * w : c;
        t.elementBytes = elementBytes;
        staging.copy.total = n * c * d * h * w;
        return staging;
    }
    const std::array<long long, 5> packed{d * h * w * c, h * w * c, w * c, c, 1};
    if(in)
    {
        // Destination order (n, d, h, w, c): coalesced on the packed side.
        staging.copy = copyOf5({n, d, h, w, c},
                               {strides[0], strides[2], strides[3], strides[4], strides[1]},
                               packed,
                               elementBytes);
    }
    else
    {
        // Destination order (n, c, d, h, w), the graph's order.
        staging.copy = copyOf5({n, c, d, h, w},
                               {d * h * w * c, 1, h * w * c, w * c, c},
                               {strides[0], strides[1], strides[2], strides[3], strides[4]},
                               elementBytes);
    }
    return staging;
}

/// The reduction step (k per MMA) of each conv_igemm atom the catalog uses: the weight format
/// granule.
int atomReductionStep(std::string_view atom)
{
    struct Entry
    {
        std::string_view name;
        int ka;
    };
    static constexpr std::array<Entry, 16> STEPS{{
        {"mfma16_f16", 16},
        {"mfma32_f16", 8},
        {"mfma16_bf16", 16},
        {"mfma32_bf16", 8},
        {"mfma16_bf16_908", 8},
        {"mfma32_bf16_908", 4},
        {"mfma16_f32", 4},
        {"mfma32_f32", 2},
        {"mfma16x32_f16", 32},
        {"mfma16x32_bf16", 32},
        {"mfma32x16_f16", 16},
        {"mfma32x16_bf16", 16},
        {"wmma11_f16", 16},
        {"wmma11_bf16", 16},
        {"wmma12_f16", 16},
        {"wmma12_bf16", 16},
    }};
    for(const auto& entry : STEPS)
    {
        if(entry.name == atom)
        {
            return entry.ka;
        }
    }
    return 0;
}

std::optional<int> intField(const KernelDefinition& kernel, const char* name)
{
    const auto value = kernel.tryGetMetadata(name);
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* number = std::get_if<int64_t>(&*value);
    if(number == nullptr || !fitsInt(*number))
    {
        return std::nullopt;
    }
    return static_cast<int>(*number);
}

std::optional<std::string> stringField(const KernelDefinition& kernel, const char* name)
{
    const auto value = kernel.tryGetMetadata(name);
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* text = std::get_if<std::string>(&*value);
    if(text == nullptr)
    {
        return std::nullopt;
    }
    return *text;
}

/// The tile: th x tw output pixels (tw a multiple of @p run) of ds planes.
struct Tile
{
    int th = 0, tw = 0, ds = 0, tilesW = 0, tilesH = 0, tilesD = 0;
    long long blocks = 0;
};

std::optional<Tile>
    tileFor(const K2Args& a, const Config& config, long long columnBlocks, long long computeUnits)
{
    const int pixels = config.blockPixels();
    const int run = config.fma ? config.tm : 1;
    const int haloMax = k2HaloMax(pixels, config.r, config.s);
    Tile best;
    double bestCost = -1.0;
    const int widest = std::min(pixels, static_cast<int>(divUp(a.wo, run) * run));
    for(int tw = run; tw <= widest; tw += run)
    {
        int th = std::min(pixels / tw, a.ho);
        while(th > 0 && (th + config.r - 1) * (tw + config.s - 1) > haloMax)
        {
            --th;
        }
        if(th < 1)
        {
            continue;
        }
        const double tiles
            = static_cast<double>(divUp(a.ho, th)) * static_cast<double>(divUp(a.wo, tw));
        const double cost = tiles * (pixels + 0.5 * (th + config.r - 1) * (tw + config.s - 1));
        if(bestCost < 0 || cost < bestCost)
        {
            bestCost = cost;
            best.th = th;
            best.tw = tw;
        }
    }
    if(best.th < 1)
    {
        return std::nullopt;
    }
    best.tilesW = static_cast<int>(divUp(a.wo, best.tw));
    best.tilesH = static_cast<int>(divUp(a.ho, best.th));
    // Split depth until the grid has 4 blocks per CU when streaming (down to KD planes per block;
    // each block re-reads KD - 1 extra planes), 8 per CU otherwise (down to one plane).
    const int minDs = config.ring != 0 ? config.kd : 1;
    const long long spatial
        = static_cast<long long>(best.tilesW) * best.tilesH * a.n * columnBlocks;
    const long long target = (minDs == 1 ? 8 : 4) * std::max<long long>(1, computeUnits);
    best.ds = a.dout;
    while(best.ds > minDs && spatial * divUp(a.dout, best.ds) < target)
    {
        best.ds = std::max(minDs, (best.ds + 1) / 2);
    }
    best.tilesD = static_cast<int>(divUp(a.dout, best.ds));
    best.blocks = spatial * best.tilesD;
    return best;
}

} // namespace

std::optional<DirectProblem> parseDirectProblem(const MatchContext& context,
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
    if(base->spatialRank != 2 && base->spatialRank != 3)
    {
        return std::nullopt;
    }
    if(a.groups != 1)
    {
        return std::nullopt;
    }
    for(const long long value : {a.n, a.c, a.k})
    {
        if(value < 1 || value > INT_LIMIT)
        {
            return std::nullopt;
        }
    }
    for(int axis = 0; axis < SPATIAL; ++axis)
    {
        if(a.stride[axis] != 1 || a.dilation[axis] != 1)
        {
            return std::nullopt;
        }
        for(const long long value :
            {a.inSpatial[axis], a.outSpatial[axis], a.filter[axis], a.prePadding[axis]})
        {
            if(!fitsInt(value))
            {
                return std::nullopt;
            }
        }
    }
    const bool knownFilter
        = std::any_of(FILTERS.begin(), FILTERS.end(), [&](const std::array<int, 3>& f) {
              return f[0] == a.filter[0] && f[1] == a.filter[1] && f[2] == a.filter[2];
          });
    if(!knownFilter)
    {
        return std::nullopt;
    }

    DirectProblem problem;
    problem.direction = direction;
    problem.elementBytes = elementBytes;
    problem.xUid = base->operands.x;
    problem.wUid = base->operands.w;
    problem.yUid = base->operands.y;
    for(int i = 0; i < 5; ++i)
    {
        problem.wStrides[i] = a.wStrides[i];
    }
    problem.flip = a.flipFilter != 0 ? 1 : 0;

    const bool forward = direction == ConvDirection::FORWARD;
    // The forward form: backward data convolves dy (k channels, output extents) into dx.
    const long long inC = forward ? a.c : a.k;
    const long long outC = forward ? a.k : a.c;
    const auto& inExt = forward ? a.inSpatial : a.outSpatial;
    const auto& outExt = forward ? a.outSpatial : a.inSpatial;
    auto& k2 = problem.args;
    k2.n = static_cast<int>(a.n);
    k2.c = static_cast<int>(inC);
    k2.k = static_cast<int>(outC);
    k2.di = static_cast<int>(inExt[0]);
    k2.hi = static_cast<int>(inExt[1]);
    k2.wi = static_cast<int>(inExt[2]);
    k2.dout = static_cast<int>(outExt[0]);
    k2.ho = static_cast<int>(outExt[1]);
    k2.wo = static_cast<int>(outExt[2]);
    const long long pads[3] = {forward ? a.prePadding[0] : a.filter[0] - 1 - a.prePadding[0],
                               forward ? a.prePadding[1] : a.filter[1] - 1 - a.prePadding[1],
                               forward ? a.prePadding[2] : a.filter[2] - 1 - a.prePadding[2]};
    k2.pd = static_cast<int>(pads[0]);
    k2.ph = static_cast<int>(pads[1]);
    k2.pw = static_cast<int>(pads[2]);
    k2.splits = 1;

    // One image of the forward-form input within the kernels' 32-bit buffer offsets.
    const long long imageBytes = inExt[0] * inExt[1] * inExt[2] * inC * elementBytes;
    if(imageBytes >= BUFFER_LIMIT || static_cast<long long>(a.n) * k2.dout * k2.ho * k2.wo < 1)
    {
        return std::nullopt;
    }

    // Activations: forward reads x and writes y, backward data reads dy and writes dx.
    const auto& inStrides = forward ? a.xStrides : a.yStrides;
    const auto& outStrides = forward ? a.yStrides : a.xStrides;
    if(!channelsLast(inStrides, {a.n, inC, inExt[0], inExt[1], inExt[2]}))
    {
        problem.in = activationStaging(inStrides, a.n, inC, inExt, true, elementBytes);
    }
    if(!channelsLast(outStrides, {a.n, outC, outExt[0], outExt[1], outExt[2]}))
    {
        problem.out = activationStaging(outStrides, a.n, outC, outExt, false, elementBytes);
    }
    size_t offset = 0;
    for(auto* staging : {&problem.in, &problem.out})
    {
        if(staging->needed)
        {
            staging->workspaceOffset = offset;
            offset = alignUp(offset + staging->bytes);
        }
    }
    problem.stagingBytes = offset;
    problem.base = std::move(*base);
    return problem;
}

std::optional<Config> configOf(const KernelDefinition& kernel)
{
    Config config;
    const auto kind = stringField(kernel, "kind");
    const auto atom = stringField(kernel, "atom");
    const auto archs = stringField(kernel, "archs");
    if(!kind.has_value() || !atom.has_value() || !archs.has_value()
       || (*kind != "mma" && *kind != "fma" && *kind != "rows"))
    {
        return std::nullopt;
    }
    config.fma = *kind == "fma";
    config.rows = *kind == "rows";
    config.atom = *atom;
    config.archs = *archs;
    const auto info = conv_igemm::atomNamed(config.atom);
    if(!info.has_value() || (info->wave == 0) != config.fma)
    {
        return std::nullopt;
    }
    config.kernelType = info->kernelType;
    config.wave = info->wave;
    int math = 0;
    int atomId = 0;
    int rowsKnob = 0;
    const std::array<std::pair<const char*, int*>, 27> ints{
        {{"atom_id", &atomId},   {"math", &math},          {"kd", &config.kd},
         {"r", &config.r},       {"s", &config.s},         {"bm", &config.bm},
         {"bn", &config.bn},     {"bc", &config.bc},       {"wm", &config.wm},
         {"wn", &config.wn},     {"vec", &config.vec},     {"ring", &config.ring},
         {"bl", &config.bl},     {"split", &config.split}, {"tm", &config.tm},
         {"tn", &config.tn},     {"nt", &config.nt},       {"threads", &config.threads},
         {"pipe", &config.pipe}, {"rows", &rowsKnob},      {"pw", &config.pw},
         {"fn", &config.fn},     {"wp", &config.wp},       {"wq", &config.wq},
         {"wk", &config.wk},     {"pp", &config.pp},       {"epi", &config.epi}}};
    for(const auto& [name, target] : ints)
    {
        const auto value = intField(kernel, name);
        if(!value.has_value())
        {
            return std::nullopt;
        }
        *target = *value;
    }
    if((math != 0 && math != 1) || rowsKnob != (config.rows ? 1 : 0))
    {
        return std::nullopt;
    }
    // The rows knobs belong to rows configs only; the halo-tile knobs to the others.
    const bool rowsFields = config.pw != 0 || config.fn != 0 || config.wp != 0 || config.wq != 0
                            || config.wk != 0 || config.pp != 0 || config.epi != 0;
    if(config.rows)
    {
        if(config.pw < 1 || config.fn < 1 || config.wp < 1 || config.wq < 1 || config.wk < 1
           || (config.pp != 0 && config.pp != 1) || (config.pp == 1 && config.wk != 2)
           || (config.epi != 0 && config.epi != 1) || config.ring != 0 || config.bl != 0
           || config.split != 0 || config.wm != 0 || config.wn != 0
           || config.bm != config.wp * config.pw || config.bn != config.wk * config.fn * 16)
        {
            return std::nullopt;
        }
    }
    else if(rowsFields)
    {
        return std::nullopt;
    }
    config.math = math == 1;
    const bool known
        = std::any_of(FILTERS.begin(), FILTERS.end(), [&](const std::array<int, 3>& f) {
              return f[0] == config.kd && f[1] == config.r && f[2] == config.s;
          });
    if(!known || config.bc < 1 || config.vec < 1 || config.bc % config.vec != 0
       || (config.ring != 0 && config.ring != 1) || (config.bl != 0 && config.bl != 1)
       || (config.split != 0 && config.split != 1) || (config.pipe != 0 && config.pipe != 1))
    {
        return std::nullopt;
    }
    if(config.fma)
    {
        if(config.tm < 1 || config.tn < 1 || config.nt < 1 || config.threads < 1
           || config.threads % config.nt != 0 || config.math || config.bl != 0 || config.split != 0)
        {
            return std::nullopt;
        }
        config.ka = 1;
    }
    else
    {
        const int step = atomReductionStep(config.atom);
        // A rows config's waves are wp x wq x wk (checked above), not wm x wn.
        if(step == 0 || config.bm < 1 || config.bn < 1
           || (!config.rows && (config.wm < 1 || config.wn < 1)) || config.bc % step != 0)
        {
            return std::nullopt;
        }
        config.ka = config.bl != 0 ? config.bc : step;
    }
    return config;
}

bool shipsTo(std::string_view archs, std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    if(arch.empty())
    {
        return false;
    }
    size_t start = 0;
    while(start <= archs.size())
    {
        const size_t end = std::min(archs.find(',', start), archs.size());
        const auto group = archs.substr(start, end - start);
        if(group == "any" || group == arch
           || ((group == "gfx11" || group == "gfx12") && arch.rfind(group, 0) == 0))
        {
            return true;
        }
        start = end + 1;
    }
    return false;
}

int64_t ldsLimitOf(std::string_view gcnArchName)
{
    // CDNA4 is gfx950 alone; the stripped name keys the family.
    return hipdnn_plugin_sdk::stripArchFeatures(gcnArchName).rfind("gfx95", 0) == 0 ? 160 * 1024
                                                                                    : 64 * 1024;
}

int64_t rowsLdsBytes(const Config& config)
{
    if(!config.rows)
    {
        return 0;
    }
    // The operand is the atom's element (three-bf16 runs on bf16 atoms).
    const int64_t operandBytes = conv_igemm::atomElement(config.atom) == "f32" ? 4 : 2;
    const int64_t haloRows = static_cast<int64_t>(config.wp) * config.pw + config.r - 1;
    const int64_t haloColumns = static_cast<int64_t>(config.wq) * 16 + config.s - 1;
    return 2 * haloRows * haloColumns * config.bc * operandBytes;
}

bool configRunsOn(const Config& config, std::string_view gcnArchName)
{
    return conv_igemm::atomRunsOn(config.atom, gcnArchName) && shipsTo(config.archs, gcnArchName)
           && rowsLdsBytes(config) <= ldsLimitOf(gcnArchName);
}

std::optional<Plan>
    planFor(const DirectProblem& problem, const Config& config, int64_t computeUnits)
{
    const auto& a = problem.args;
    if(computeUnits < 1)
    {
        return std::nullopt;
    }
    Plan plan;
    plan.args = a;
    const long long reduction = static_cast<long long>(a.c) * (config.math ? 3 : 1);
    const long long columns = config.blockColumns();
    const long long columnBlocks = divUp(a.k, columns);
    const long long chunks = divUp(reduction, config.bc);
    const long long cpad = chunks * config.bc;
    const long long kpad = columnBlocks * columns;
    const long long taps = static_cast<long long>(config.kd) * config.r * config.s;
    plan.cpad = static_cast<int>(std::min(cpad, INT_LIMIT));
    plan.weightElems = taps * cpad * kpad;
    // The kernels read the formatted weights through one buffer of 31-bit byte offsets.
    const long long weightOperandBytes = config.math ? 2 : problem.elementBytes;
    if(reduction % config.vec != 0 || columnBlocks > MAX_GRID_YZ || cpad > INT_LIMIT
       || plan.weightElems >= INT_LIMIT || plan.weightElems * weightOperandBytes >= BUFFER_LIMIT)
    {
        return std::nullopt;
    }
    std::optional<Tile> tile;
    if(config.rows)
    {
        // Fixed tile of one output plane; each block walks tiles, one block per compute unit at
        // most. A 2-D config (kd 1) takes 2-D problems; a 3-D one walks every output plane.
        Tile t;
        t.th = config.wp * config.pw;
        t.tw = config.wq * 16;
        t.ds = 1;
        t.tilesW = static_cast<int>(divUp(a.wo, t.tw));
        t.tilesH = static_cast<int>(divUp(a.ho, t.th));
        t.tilesD = 1;
        t.blocks = static_cast<long long>(t.tilesW) * t.tilesH * a.n * a.dout * columnBlocks;
        // The halo moves in 16-byte granules of whole channels; 32-bit offsets within one input
        // image; the tile count fits an int.
        const int operandBytes = config.math ? 2 : problem.elementBytes;
        const long long imageBytes
            = static_cast<long long>(a.di) * a.hi * a.wi * reduction * operandBytes;
        if(config.vec * operandBytes != 16 || (config.kd == 1 && (a.dout != 1 || a.di != 1))
           || imageBytes >= BUFFER_LIMIT || t.blocks > INT_LIMIT)
        {
            return std::nullopt;
        }
        tile = t;
    }
    else
    {
        tile = tileFor(a, config, columnBlocks, computeUnits);
    }
    if(!tile.has_value())
    {
        return std::nullopt;
    }
    const long long gridX
        = static_cast<long long>(tile->tilesW) * tile->tilesH * tile->tilesD * a.n;
    // A staging transpose is one launch of a 256-thread block per tile.
    for(const Staging* staging : {&problem.in, &problem.out})
    {
        if(staging->needed && staging->transpose)
        {
            const auto& t = staging->transposeArgs;
            constexpr long long TILE = CONV_IGEMM_TRANSPOSE_TILE;
            if((t.cols + TILE - 1) / TILE * ((t.rows + TILE - 1) / TILE) * t.batches * 256
               > static_cast<long long>(std::numeric_limits<uint32_t>::max()))
            {
                return std::nullopt;
            }
        }
    }
    // One dispatch addresses at most 2^32 - 1 work items along x (rows configs launch at
    // most one block per unit).
    if(gridX > INT_LIMIT
       || (!config.rows
           && gridX * config.blockThreads()
                  > static_cast<long long>(std::numeric_limits<uint32_t>::max())))
    {
        return std::nullopt;
    }
    auto& k2 = plan.args;
    k2.th = tile->th;
    k2.tw = tile->tw;
    k2.ds = tile->ds;
    k2.tilesW = tile->tilesW;
    k2.tilesH = tile->tilesH;
    k2.tilesD = tile->tilesD;
    k2.kpad = static_cast<int>(kpad);
    k2.ca = static_cast<int>(cpad / config.ka);
    k2.c = static_cast<int>(reduction);
    k2.splits = 1;
    k2.cps = static_cast<int>(chunks);
    plan.outElems = static_cast<long long>(a.n) * a.dout * a.ho * a.wo * a.k;
    if(config.split != 0)
    {
        // Split only an under-filled grid: below 2 blocks per CU, 2+ chunks per split, <= 16.
        const long long wanted = 2 * std::max<int64_t>(1, computeUnits);
        if(tile->blocks < wanted && chunks >= 4)
        {
            const long long splits
                = std::min({MAX_SPLITS, chunks / 2, divUp(wanted, tile->blocks)});
            if(splits > 1)
            {
                k2.cps = static_cast<int>(divUp(chunks, splits));
                k2.splits = static_cast<int>(divUp(chunks, k2.cps));
            }
        }
    }
    plan.grid[0] = static_cast<unsigned int>(gridX);
    plan.grid[1] = static_cast<unsigned int>(columnBlocks);
    plan.grid[2] = static_cast<unsigned int>(k2.splits);
    if(config.rows)
    {
        // One 1-D grid of at most one block per unit; the kernel walks the tiles.
        plan.grid[0] = static_cast<unsigned int>(std::min<long long>(tile->blocks, computeUnits));
        plan.grid[1] = 1;
        plan.grid[2] = 1;
    }

    // Workspace: [staged activations][formatted weights][three-bf16 input][split partials].
    size_t offset = problem.stagingBytes;
    plan.weightOffset = offset;
    plan.weightBytes = static_cast<size_t>(plan.weightElems)
                       * static_cast<size_t>(config.math ? 2 : problem.elementBytes);
    offset = alignUp(offset + plan.weightBytes);
    if(config.math)
    {
        const long long pixels = static_cast<long long>(a.n) * a.di * a.hi * a.wi;
        const long long imageBytes = static_cast<long long>(a.di) * a.hi * a.wi * reduction * 2;
        if(imageBytes >= BUFFER_LIMIT)
        {
            return std::nullopt;
        }
        plan.splitInputOffset = offset;
        plan.splitInputBytes = static_cast<size_t>(pixels * reduction * 2);
        offset = alignUp(offset + plan.splitInputBytes);
    }
    if(config.split != 0)
    {
        plan.partialOffset = offset;
        plan.partialBytes = static_cast<size_t>(k2.splits * plan.outElems) * sizeof(float);
        offset = alignUp(offset + plan.partialBytes);
    }
    plan.workspaceBytes = offset;
    return plan;
}

} // namespace conv_direct

namespace
{

using conv_direct::Config;
using conv_direct::DirectProblem;
using conv_direct::Plan;
using conv_direct::Staging;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_direct_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_direct_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_direct.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_direct.dispatch";
/// The other kernels of the same source, under their own entry points.
constexpr std::string_view FORMAT_ENTRY_POINT = "ConvDirectFormat";
constexpr std::string_view SPLIT_INPUT_ENTRY_POINT = "ConvDirectSplitInput";
constexpr std::string_view REDUCE_ENTRY_POINT = "ConvDirectReduce";
constexpr std::string_view COPY_ENTRY_POINT = "ConvDirectCopy";
constexpr std::string_view TRANSPOSE_ENTRY_POINT = "ConvDirectTranspose";
constexpr unsigned int AUX_BLOCK = 256;
constexpr long long AUX_MAX_BLOCKS = 1LL << 20;

constexpr std::string_view X_TOKEN = "conv_direct.x.uid";
constexpr std::string_view W_TOKEN = "conv_direct.w.uid";
constexpr std::string_view Y_TOKEN = "conv_direct.y.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_direct.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_direct.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_direct.flops";

std::optional<ConvDirection> directionFor(const KernelDefinition& kernel)
{
    const auto value = kernel.tryGetMetadata("direction");
    const auto* direction = value.has_value() ? std::get_if<std::string>(&*value) : nullptr;
    if(direction == nullptr)
    {
        return std::nullopt;
    }
    if(*direction == "FWD")
    {
        return ConvDirection::FORWARD;
    }
    if(*direction == "BWD")
    {
        return ConvDirection::BACKWARD_DATA;
    }
    return std::nullopt;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_direct::parseDirectProblem(context, direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = problem->xUid;
    bound[std::string(W_TOKEN)] = problem->wUid;
    bound[std::string(Y_TOKEN)] = problem->yUid;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(problem->base.dataType));
    bound[std::string(DIRECTION_TOKEN)]
        = std::string(direction == ConvDirection::FORWARD ? "FWD" : "BWD");
    const auto flops = pack_features::checkedMultiply(2, problem->base.multiplyAccumulates);
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

/// The problem and config a kernel definition runs on @p context, or nullopt when the kernel
/// match declines.
struct Admitted
{
    DirectProblem problem;
    Config config;
    Plan plan;
};

std::optional<Admitted> admitted(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto direction = directionFor(kernel);
    const auto config = conv_direct::configOf(kernel);
    const auto dtype = kernel.tryGetMetadata("dtype");
    const auto* dtypeName = dtype.has_value() ? std::get_if<std::string>(&*dtype) : nullptr;
    if(!direction.has_value() || !config.has_value() || dtypeName == nullptr)
    {
        return std::nullopt;
    }
    auto problem = conv_direct::parseDirectProblem(context, *direction);
    if(!problem.has_value() || *dtypeName != data_objects::EnumNameDataType(problem->base.dataType)
       || !conv_direct::configRunsOn(*config, context.deviceProperties.gcnArchName))
    {
        return std::nullopt;
    }
    const auto& filter = problem->base.args.filter;
    if(config->kd != filter[0] || config->r != filter[1] || config->s != filter[2])
    {
        return std::nullopt;
    }
    // The atom takes the graph's element type; three-bf16 runs fp32 graphs on bf16 atoms, and
    // only with TF32 permission.
    const auto element = conv_igemm::atomElement(config->atom);
    const auto wanted = conv_igemm::dataElement(problem->base.dataType);
    if(config->math)
    {
        if(wanted != "f32" || element != "bf16" || !conv_igemm::tf32Permitted())
        {
            return std::nullopt;
        }
    }
    else if(element != "any" && element != wanted)
    {
        return std::nullopt;
    }
    const auto plan = conv_direct::planFor(
        *problem, *config, device_units::dispatchUnits(context.deviceProperties));
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return Admitted{std::move(*problem), *config, *plan};
}

/// Kernel-scoped applicability: the config's direction, dtype and filter are the graph's; its
/// atom runs on this device and its ship list names it; its channel vector divides the reduction
/// channels; and the planner fits the problem in its halo and launch limits.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return admitted(context, kernel).has_value();
}

template <class Args>
const std::vector<KernelArgument>& signatureOf()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(Args)), 0, ""}};
    return s_signature;
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
                                                       "conv_direct: unsupported element type");
    }
}

unsigned int auxBlocks(long long elements)
{
    return static_cast<unsigned int>(
        std::clamp<long long>((elements + AUX_BLOCK - 1) / AUX_BLOCK, 1, AUX_MAX_BLOCKS));
}

class PreparedConvDirect : public IngestorPreparedDispatch
{
public:
    PreparedConvDirect(IngestorKernelCode conv,
                       IngestorKernelCode format,
                       std::optional<IngestorKernelCode> splitInput,
                       std::optional<IngestorKernelCode> reduce,
                       std::array<std::optional<IngestorKernelCode>, 2> copies,
                       Admitted admitted)
        : _conv(std::move(conv))
        , _format(std::move(format))
        , _splitInput(std::move(splitInput))
        , _reduce(std::move(reduce))
        , _copies(std::move(copies))
        , _admitted(std::move(admitted))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _conv;
    }

    compilation::IRunnableKernel& conv(hipStream_t stream) const
    {
        return _conv.kernelForStream(stream);
    }
    compilation::IRunnableKernel& format(hipStream_t stream) const
    {
        return _format.kernelForStream(stream);
    }
    compilation::IRunnableKernel& splitInput(hipStream_t stream) const
    {
        return _splitInput->kernelForStream(stream);
    }
    compilation::IRunnableKernel& reduce(hipStream_t stream) const
    {
        return _reduce->kernelForStream(stream);
    }
    /// The staging copy of the forward-form input (0) or output (1).
    compilation::IRunnableKernel& copy(size_t operand, hipStream_t stream) const
    {
        return _copies[operand]->kernelForStream(stream);
    }
    const Admitted& admittedRun() const
    {
        return _admitted;
    }

private:
    IngestorKernelCode _conv;
    IngestorKernelCode _format;
    std::optional<IngestorKernelCode> _splitInput;
    std::optional<IngestorKernelCode> _reduce;
    std::array<std::optional<IngestorKernelCode>, 2> _copies;
    Admitted _admitted;
};

class ConvDirectDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvDirectDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                              const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The staged activations, the formatted weights, the three-bf16 input and the split-C
    /// partials of this config.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto run = admitted(context, kernel);
        return run.has_value() ? run->plan.workspaceBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        auto run = admitted(context, kernel);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!run.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != run->problem.xUid || *w != run->problem.wUid || *y != run->problem.yUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_direct dispatch was given a graph or config its matchers did not admit");
        }
        const auto& config = run->config;
        const auto& plan = run->plan;

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        const std::string element = elementTypeFor(run->problem.base.dataType);
        options.add("CONV_DIRECT_T", config.math ? std::string("unsigned short") : element);
        options.add("CONV_DIRECT_TO", config.math ? std::string("float") : element);
        options.add("CONV_DIRECT_MATH", std::string(config.math ? "1" : "0"));
        options.add("CONV_DIRECT_KD", std::to_string(config.kd));
        options.add("CONV_DIRECT_R", std::to_string(config.r));
        options.add("CONV_DIRECT_S", std::to_string(config.s));
        options.add("CONV_DIRECT_FMA", std::string(config.fma ? "1" : "0"));
        options.add("CONV_DIRECT_BC", std::to_string(config.bc));
        options.add("CONV_DIRECT_VEC", std::to_string(config.vec));
        options.add("CONV_DIRECT_RING", std::to_string(config.ring));
        if(config.fma)
        {
            options.add("CONV_DIRECT_TM", std::to_string(config.tm));
            options.add("CONV_DIRECT_TN", std::to_string(config.tn));
            options.add("CONV_DIRECT_NT", std::to_string(config.nt));
            options.add("CONV_DIRECT_THREADS", std::to_string(config.threads));
            options.add("CONV_DIRECT_PIPE", std::to_string(config.pipe));
        }
        else if(config.rows)
        {
            options.add("CONV_DIRECT_ROWS", std::string("1"));
            options.add("CONV_DIRECT_ATOM", std::string(config.kernelType));
            options.add("CONV_DIRECT_PW", std::to_string(config.pw));
            options.add("CONV_DIRECT_FN", std::to_string(config.fn));
            options.add("CONV_DIRECT_WP", std::to_string(config.wp));
            options.add("CONV_DIRECT_WQ", std::to_string(config.wq));
            options.add("CONV_DIRECT_WK", std::to_string(config.wk));
            options.add("CONV_DIRECT_PP", std::to_string(config.pp));
            options.add("CONV_DIRECT_EPI", std::to_string(config.epi));
        }
        else
        {
            options.add("CONV_DIRECT_ATOM", std::string(config.kernelType));
            options.add("CONV_DIRECT_BM", std::to_string(config.bm));
            options.add("CONV_DIRECT_BN", std::to_string(config.bn));
            options.add("CONV_DIRECT_WM", std::to_string(config.wm));
            options.add("CONV_DIRECT_WN", std::to_string(config.wn));
            options.add("CONV_DIRECT_BL", std::to_string(config.bl));
            options.add("CONV_DIRECT_SPLIT", std::to_string(config.split));
        }

        auto build = [&](std::string_view entry, const std::vector<KernelArgument>& signature) {
            KernelDefinition aux = kernel;
            aux.source.entryPoint = std::string(entry);
            return buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, aux, options, signature);
        };
        auto conv = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, signatureOf<K2Args>());
        conv.setBlockSize(static_cast<unsigned int>(config.blockThreads()), 1, 1);
        conv.setGridSize(plan.grid[0], plan.grid[1], plan.grid[2]);

        auto format = build(FORMAT_ENTRY_POINT, signatureOf<K2FormatArgs>());
        format.setBlockSize(AUX_BLOCK, 1, 1);
        // The halo-tile format (k2FormatWeights) takes a run of ka elements per thread; the rows
        // and three-bf16 formats one element per thread.
        format.setGridSize(
            auxBlocks(config.rows || config.math ? plan.weightElems : plan.weightElems / config.ka),
            1,
            1);

        std::optional<IngestorKernelCode> splitInput;
        if(config.math)
        {
            const auto& a = run->problem.args;
            splitInput.emplace(build(SPLIT_INPUT_ENTRY_POINT, signatureOf<K2SplitArgs>()));
            splitInput->setBlockSize(AUX_BLOCK, 1, 1);
            splitInput->setGridSize(
                auxBlocks(static_cast<long long>(a.n) * a.di * a.hi * a.wi * a.c), 1, 1);
        }
        std::optional<IngestorKernelCode> reduce;
        if(config.split != 0)
        {
            reduce.emplace(build(REDUCE_ENTRY_POINT, signatureOf<K2ReduceArgs>()));
            reduce->setBlockSize(AUX_BLOCK, 1, 1);
            reduce->setGridSize(auxBlocks(plan.outElems), 1, 1);
        }
        std::array<std::optional<IngestorKernelCode>, 2> copies;
        const std::array<const Staging*, 2> stagings{&run->problem.in, &run->problem.out};
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(!stagings[i]->needed)
            {
                continue;
            }
            const bool transpose = stagings[i]->transpose;
            auto copy = transpose
                            ? build(TRANSPOSE_ENTRY_POINT, signatureOf<ConvIgemmTransposeArgs>())
                            : build(COPY_ENTRY_POINT, signatureOf<ConvIgemmCopyArgs>());
            long long blocks = 0;
            if(transpose)
            {
                // One block per tile of each image.
                const auto& t = stagings[i]->transposeArgs;
                constexpr long long TILE = CONV_IGEMM_TRANSPOSE_TILE;
                blocks = (t.cols + TILE - 1) / TILE * ((t.rows + TILE - 1) / TILE) * t.batches;
                if(blocks * AUX_BLOCK > std::numeric_limits<uint32_t>::max())
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                        "conv_direct staging transpose exceeds the launch grid");
                }
            }
            else
            {
                blocks = auxBlocks(stagings[i]->copy.total);
            }
            copy.setBlockSize(AUX_BLOCK, 1, 1);
            copy.setGridSize(static_cast<unsigned int>(blocks), 1, 1);
            copies[i].emplace(std::move(copy));
        }
        return std::make_unique<PreparedConvDirect>(std::move(conv),
                                                    std::move(format),
                                                    std::move(splitInput),
                                                    std::move(reduce),
                                                    std::move(copies),
                                                    std::move(*run));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvDirect&>(prepared);
        const auto& run = dispatch.admittedRun();
        const auto& problem = run.problem;
        const auto& plan = run.plan;
        const auto& config = run.config;
        if(plan.workspaceBytes > 0 && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "conv_direct needs its workspace");
        }
        auto* x = hipdnn_plugin_sdk::findDeviceBuffer(problem.xUid, deviceBuffers, numDeviceBuffers)
                      .ptr;
        auto* w = hipdnn_plugin_sdk::findDeviceBuffer(problem.wUid, deviceBuffers, numDeviceBuffers)
                      .ptr;
        auto* y = hipdnn_plugin_sdk::findDeviceBuffer(problem.yUid, deviceBuffers, numDeviceBuffers)
                      .ptr;
        const auto stream = handle.getStream();
        const bool forward = problem.direction == ConvDirection::FORWARD;
        auto* base = static_cast<char*>(workspace);
        // The forward form's input (x, or dy) and output (y, or dx), as the graph holds them.
        const std::array<void*, 2> graph{forward ? x : y, forward ? y : x};
        const std::array<const Staging*, 2> stagings{&problem.in, &problem.out};
        std::array<void*, 2> operand = graph;
        auto stage = [&](size_t i, bool in) {
            char* staged = base + stagings[i]->workspaceOffset;
            void* src = in ? graph[i] : staged;
            void* dst = in ? staged : graph[i];
            if(stagings[i]->transpose)
            {
                ConvIgemmTransposeArgs args = stagings[i]->transposeArgs;
                args.src = src;
                args.dst = dst;
                dispatch.copy(i, stream).launch(stream, args);
            }
            else
            {
                ConvIgemmCopyArgs args = stagings[i]->copy;
                args.src = src;
                args.dst = dst;
                dispatch.copy(i, stream).launch(stream, args);
            }
        };
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(stagings[i]->needed)
            {
                operand[i] = base + stagings[i]->workspaceOffset;
            }
        }
        if(stagings[0]->needed)
        {
            stage(0, true);
        }

        // The weights, in the order this config reads.
        K2FormatArgs fmt{};
        fmt.w = w;
        fmt.wf = base + plan.weightOffset;
        for(int i = 0; i < 5; ++i)
        {
            fmt.wStrides[i] = problem.wStrides[i];
        }
        const auto& a = problem.base.args;
        fmt.k0 = static_cast<int>(a.k);
        fmt.c0 = static_cast<int>(a.c);
        fmt.kd = config.kd;
        fmt.r = config.r;
        fmt.s = config.s;
        fmt.kpad = plan.args.kpad;
        fmt.cpad = plan.cpad;
        fmt.ka = config.ka;
        fmt.backward = forward ? 0 : 1;
        fmt.flip = problem.flip;
        dispatch.format(stream).launch(stream, fmt);

        K2Args args = plan.args;
        args.x = operand[0];
        args.w = base + plan.weightOffset;
        args.y = operand[1];
        if(config.math)
        {
            K2SplitArgs split{};
            split.x = operand[0];
            split.xs = base + plan.splitInputOffset;
            split.pixels = static_cast<long long>(problem.args.n) * problem.args.di
                           * problem.args.hi * problem.args.wi;
            split.c = problem.args.c;
            dispatch.splitInput(stream).launch(stream, split);
            args.x = split.xs;
        }
        args.ws = config.split != 0 ? reinterpret_cast<float*>(base + plan.partialOffset) : nullptr;
        dispatch.conv(stream).launch(stream, args);
        if(config.split != 0)
        {
            K2ReduceArgs reduce{args.ws, operand[1], plan.outElems, args.splits, 0};
            dispatch.reduce(stream).launch(stream, reduce);
        }
        if(stagings[1]->needed)
        {
            stage(1, false);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convDirectKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvDirectDispatchHandler& convDirectDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convDirectKpackModuleCache());
    static const ConvDirectDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvDirectSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convDirectDispatchHandler());
}

void resetConvDirectModuleCache()
{
    convDirectKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
