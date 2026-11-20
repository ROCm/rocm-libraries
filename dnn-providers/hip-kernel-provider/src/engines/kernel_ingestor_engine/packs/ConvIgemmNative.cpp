// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
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
#include "engines/kernel_ingestor_engine/descriptors/conv/igemm/kernels/ConvIgemmArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvIgemmNative.cpp
 * @brief The native half of the implicit-GEMM convolution engines (CONV_IGEMM_FWD,
 *        CONV_IGEMM_BWD): graph matching, kernel matching, workspace sizing and dispatch.
 *
 * One HIPRTC source serves every architecture: matrix-core atoms per family (MFMA 16x16 and
 * 32x32 on gfx908 to gfx950, WMMA on gfx11 and gfx12) and an FMA kernel everywhere, in fp32,
 * fp16 and bf16, any stride, dilation, padding and group count, 1-D to 3-D. Shapes are runtime;
 * a catalog config fixes only the atom and the tile. The kernels read channels-last activations
 * and a K T R S C/g filter (backward data: transposed per group); operands in any other layout,
 * and a reflected filter (true convolution), are staged through the workspace by a strided-copy
 * kernel of the same source. Every config is admitted on correctness alone; autotune (UHD later)
 * ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace conv_igemm
{

namespace
{

constexpr long long INT_LIMIT = std::numeric_limits<int>::max();
constexpr size_t WORKSPACE_ALIGNMENT = 256;
/// The largest catalog row tile: bounds how many images one block's rows span.
constexpr long long MAX_BM = 256;
/// The smallest catalog row tile: bounds the grid's first axis.
constexpr long long MIN_BM = 64;
constexpr long long MAX_GRID_YZ = 65535;
constexpr int SPATIAL = CONV_NAIVE_SPATIAL_DIMS;

bool fitsInt(long long value)
{
    return value >= -INT_LIMIT && value <= INT_LIMIT;
}

long long divUp(long long value, long long divisor)
{
    return (value + divisor - 1) / divisor;
}

size_t alignUp(size_t value)
{
    return (value + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

/// True when @p strides (n, c, d, h, w) are packed channels-last for @p extents; extent-1 axes
/// may carry any stride.
// NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
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
// NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
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

/// Packed strides of @p dims (outermost first).
template <size_t N>
std::array<long long, N> packedStrides(const std::array<long long, N>& dims)
{
    std::array<long long, N> strides{};
    long long running = 1;
    for(size_t i = N; i-- > 0;)
    {
        strides[i] = running;
        running *= dims[i];
    }
    return strides;
}

template <size_t N>
ConvIgemmCopyArgs copyOf(const std::array<long long, N>& dims,
                         const std::array<long long, N>& srcStrides,
                         const std::array<long long, N>& dstStrides,
                         int elementBytes)
{
    static_assert(N <= CONV_IGEMM_COPY_RANK);
    ConvIgemmCopyArgs copy{};
    copy.rank = static_cast<int>(N);
    copy.total = 1;
    for(size_t i = 0; i < N; ++i)
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
/// packed channels-last copy.
Staging activationStaging(
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
    const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
    long long n,
    long long c,
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
    const long long (&spatial)[SPATIAL],
    bool in,
    int elementBytes)
{
    Staging staging;
    staging.needed = true;
    const long long d = spatial[0];
    const long long h = spatial[1];
    const long long w = spatial[2];
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
    if(in)
    {
        // Destination order (n, d, h, w, c): coalesced on the packed side.
        const std::array<long long, 5> dims{n, d, h, w, c};
        staging.copy = copyOf<5>(dims,
                                 {strides[0], strides[2], strides[3], strides[4], strides[1]},
                                 packedStrides(dims),
                                 elementBytes);
    }
    else
    {
        // Destination order (n, c, d, h, w), the graph's order for channels-first.
        const std::array<long long, 5> dims{n, c, d, h, w};
        staging.copy = copyOf<5>(dims,
                                 {d * h * w * c, 1, h * w * c, w * c, c},
                                 {strides[0], strides[1], strides[2], strides[3], strides[4]},
                                 elementBytes);
    }
    return staging;
}

struct AtomEntry
{
    AtomInfo info;
    std::string_view element; ///< "f16", "bf16", "f32", or "any" (FMA)
    std::array<std::string_view, 7> archs;
};

constexpr std::array<std::string_view, 7> CDNA{"gfx908", "gfx90a", "gfx942", "gfx950"};
constexpr std::array<std::string_view, 7> CDNA_BF16_1K{"gfx90a", "gfx942", "gfx950"};
constexpr std::array<std::string_view, 7> GFX908{"gfx908"};
constexpr std::array<std::string_view, 7> GFX942{"gfx942"};
constexpr std::array<std::string_view, 7> GFX950{"gfx950"};
constexpr std::array<std::string_view, 7> GFX11{
    "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152"};
constexpr std::array<std::string_view, 7> GFX12{"gfx1200", "gfx1201"};
constexpr std::array<std::string_view, 7> GFX125{"gfx1250"};

constexpr std::array<AtomEntry, 22> ATOMS{{
    {{"mfma16_f16", "K1AtomMfmaF16", 64, false, 32}, "f16", CDNA},
    {{"mfma32_f16", "K1AtomMfma32F16", 64, false, 16}, "f16", CDNA},
    {{"mfma16_bf16", "K1AtomMfmaBf16", 64, false, 32}, "bf16", CDNA_BF16_1K},
    {{"mfma32_bf16", "K1AtomMfma32Bf16", 64, false, 16}, "bf16", CDNA_BF16_1K},
    {{"mfma16_bf16_908", "K1AtomMfmaBf16Gfx908", 64, false}, "bf16", GFX908},
    {{"mfma32_bf16_908", "K1AtomMfma32Bf16Gfx908", 64, false}, "bf16", GFX908},
    {{"mfma16_f32", "K1AtomMfmaF32", 64, false}, "f32", CDNA},
    {{"mfma32_f32", "K1AtomMfma32F32", 64, false}, "f32", CDNA},
    {{"mfma16_xf32", "K1AtomMfmaXf32", 64, true}, "f32", GFX942},
    {{"mfma32_xf32", "K1AtomMfma32Xf32", 64, true}, "f32", GFX942},
    {{"mfma16x32_f16", "K1AtomMfmaF16X32", 64, false, 32}, "f16", GFX950},
    {{"mfma16x32_bf16", "K1AtomMfmaBf16X32", 64, false, 32}, "bf16", GFX950},
    {{"mfma32x16_f16", "K1AtomMfma32F16X16", 64, false, 16}, "f16", GFX950},
    {{"mfma32x16_bf16", "K1AtomMfma32Bf16X16", 64, false, 16}, "bf16", GFX950},
    {{"wmma11_f16", "K1AtomWmma11F16", 32, false, 16}, "f16", GFX11},
    {{"wmma11_bf16", "K1AtomWmma11Bf16", 32, false, 16}, "bf16", GFX11},
    {{"wmma12_f16", "K1AtomWmma12F16", 32, false, 16}, "f16", GFX12},
    {{"wmma12_bf16", "K1AtomWmma12Bf16", 32, false, 16}, "bf16", GFX12},
    {{"fma", "", 0, false}, "any", {}},
    {{"wmma125_f16", "K1AtomWmma125F16", 32, false, 32}, "f16", GFX125},
    {{"wmma125_bf16", "K1AtomWmma125Bf16", 32, false, 32}, "bf16", GFX125},
    {{"wmma125_f32", "K1AtomWmma125F32", 32, false}, "f32", GFX125},
}};

/// The staged operands' places in the workspace, x then w then y, and its size.
void layOutWorkspace(IgemmProblem& problem)
{
    size_t offset = 0;
    for(auto* staging : {&problem.x, &problem.w, &problem.y})
    {
        if(staging->needed)
        {
            staging->workspaceOffset = offset;
            offset = alignUp(offset + staging->bytes);
        }
    }
    if(problem.col2imBytes > 0)
    {
        problem.col2imOffset = offset;
        offset = alignUp(offset + problem.col2imBytes);
    }
    problem.workspaceBytes = offset;
}

const AtomEntry* entryNamed(std::string_view name)
{
    if(name.empty())
    {
        return nullptr;
    }
    for(const auto& entry : ATOMS)
    {
        if(entry.info.name == name)
        {
            return &entry;
        }
    }
    return nullptr;
}

} // namespace

std::string_view dataElement(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return "f32";
    case data_objects::DataType::HALF:
        return "f16";
    case data_objects::DataType::BFLOAT16:
        return "bf16";
    default:
        return "";
    }
}

std::string_view atomElement(std::string_view atom)
{
    const auto* entry = entryNamed(atom);
    return entry == nullptr ? std::string_view{} : entry->element;
}

std::optional<AtomInfo> atomNamed(std::string_view name)
{
    const auto* entry = entryNamed(name);
    if(entry == nullptr)
    {
        return std::nullopt;
    }
    return entry->info;
}

bool atomRunsOn(std::string_view atom, std::string_view gcnArchName)
{
    const auto* entry = entryNamed(atom);
    if(entry == nullptr)
    {
        return false;
    }
    if(entry->element == "any")
    {
        return true;
    }
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    return std::find(entry->archs.begin(), entry->archs.end(), arch) != entry->archs.end()
           && !arch.empty();
}

std::optional<IgemmProblem> parseIgemmProblem(const MatchContext& context,
                                              ConvDirection direction,
                                              conv_naive::Fusion fusion)
{
    if(direction != ConvDirection::FORWARD && direction != ConvDirection::BACKWARD_DATA)
    {
        return std::nullopt;
    }
    auto base = conv_naive::parseConvNaiveProblem(context, direction, fusion);
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
    if(a.groups < 1 || a.c % a.groups != 0 || a.k % a.groups != 0)
    {
        return std::nullopt;
    }
    for(const long long value : {a.n, a.c, a.k, a.groups})
    {
        if(value < 1 || value > INT_LIMIT)
        {
            return std::nullopt;
        }
    }
    for(int axis = 0; axis < SPATIAL; ++axis)
    {
        for(const long long value :
            {a.inSpatial[axis], a.outSpatial[axis], a.filter[axis], a.prePadding[axis]})
        {
            if(!fitsInt(value))
            {
                return std::nullopt;
            }
        }
        if(a.stride[axis] < 1 || a.stride[axis] > INT_LIMIT || a.dilation[axis] < 1
           || a.dilation[axis] > INT_LIMIT || a.filter[axis] < 1)
        {
            return std::nullopt;
        }
    }

    IgemmProblem problem;
    problem.direction = direction;
    problem.elementBytes = elementBytes;
    problem.xUid = base->operands.x;
    problem.wUid = base->operands.w;
    problem.yUid = base->operands.y;
    auto& k1 = problem.args;
    k1.n = static_cast<int>(a.n);
    k1.c = static_cast<int>(a.c);
    k1.k = static_cast<int>(a.k);
    k1.groups = static_cast<int>(a.groups);
    k1.di = static_cast<int>(a.inSpatial[0]);
    k1.hi = static_cast<int>(a.inSpatial[1]);
    k1.wi = static_cast<int>(a.inSpatial[2]);
    k1.dout = static_cast<int>(a.outSpatial[0]);
    k1.ho = static_cast<int>(a.outSpatial[1]);
    k1.wo = static_cast<int>(a.outSpatial[2]);
    k1.ft = static_cast<int>(a.filter[0]);
    k1.fr = static_cast<int>(a.filter[1]);
    k1.fs = static_cast<int>(a.filter[2]);
    k1.sd = static_cast<int>(a.stride[0]);
    k1.sh = static_cast<int>(a.stride[1]);
    k1.sw = static_cast<int>(a.stride[2]);
    k1.pd = static_cast<int>(a.prePadding[0]);
    k1.ph = static_cast<int>(a.prePadding[1]);
    k1.pw = static_cast<int>(a.prePadding[2]);
    k1.dd = static_cast<int>(a.dilation[0]);
    k1.dh = static_cast<int>(a.dilation[1]);
    k1.dw = static_cast<int>(a.dilation[2]);
    k1.m = a.n * a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2];

    const bool forward = direction == ConvDirection::FORWARD;
    const long long cg = a.c / a.groups;
    const long long kg = a.k / a.groups;
    // Activations: forward stages x in and y out, backward data dy in and dx out.
    if(!channelsLast(a.xStrides, {a.n, a.c, a.inSpatial[0], a.inSpatial[1], a.inSpatial[2]}))
    {
        problem.x = activationStaging(a.xStrides, a.n, a.c, a.inSpatial, forward, elementBytes);
    }
    if(!channelsLast(a.yStrides, {a.n, a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]}))
    {
        problem.y = activationStaging(a.yStrides, a.n, a.k, a.outSpatial, !forward, elementBytes);
    }
    // The filter: reflected through negative strides for true convolution.
    const bool flip = a.flipFilter != 0;
    const long long t = a.filter[0];
    const long long r = a.filter[1];
    const long long s = a.filter[2];
    const long long st = flip ? -a.wStrides[2] : a.wStrides[2];
    const long long sr = flip ? -a.wStrides[3] : a.wStrides[3];
    const long long ss = flip ? -a.wStrides[4] : a.wStrides[4];
    const int64_t reflectedStart
        = flip ? (t - 1) * a.wStrides[2] + (r - 1) * a.wStrides[3] + (s - 1) * a.wStrides[4] : 0;
    const std::array<long long, 5> ktrsc{a.k, t, r, s, cg};
    // The filter's packed orders: K T R S C/g (channels-last) or K C/g T R S (channels-first).
    const long long taps = t * r * s;
    const auto packedAs = [&](bool channelsLastFilter) {
        if(flip)
        {
            return false;
        }
        const long long cStride = channelsLastFilter ? 1 : taps;
        const long long sStride = channelsLastFilter ? cg : 1;
        return (a.k == 1 || a.wStrides[0] == cg * taps) && (cg == 1 || a.wStrides[1] == cStride)
               && (t == 1 || a.wStrides[2] == sStride * s * r)
               && (r == 1 || a.wStrides[3] == sStride * s) && (s == 1 || a.wStrides[4] == sStride);
    };
    const bool krsc = packedAs(true);
    const bool kcrs = !krsc && packedAs(false);
    // Each row (output channel) is a packed [C/g][taps] or [taps][C/g] block, so one batched
    // tiled transpose reaches the layout the kernel reads.
    const auto transposeTo = [&](long long batches, long long rows, long long cols) {
        problem.w.needed = true;
        problem.w.transpose = true;
        auto& tr = problem.w.transposeArgs;
        tr.batches = batches;
        tr.rows = rows;
        tr.cols = cols;
        tr.elementBytes = elementBytes;
        problem.w.copy.total = batches * rows * cols;
    };
    if(forward && !krsc)
    {
        if(kcrs)
        {
            // Per output channel [C/g][taps] -> [taps][C/g]: K T R S C/g.
            transposeTo(a.k, cg, taps);
        }
        else
        {
            problem.w.needed = true;
            problem.w.copy = copyOf<5>(ktrsc,
                                       {a.wStrides[0], st, sr, ss, a.wStrides[1]},
                                       packedStrides(ktrsc),
                                       elementBytes);
        }
    }
    else if(!forward)
    {
        if(krsc)
        {
            // Per group [K/g][taps C/g] -> [taps C/g][K/g]: the default [g][tap][c][k].
            transposeTo(a.groups, kg, taps * cg);
        }
        else if(kcrs)
        {
            // Per group [K/g][C/g taps] -> [C/g taps][K/g]: [g][c][tap][k].
            transposeTo(a.groups, kg, cg * taps);
            problem.args.wRowStride = taps * kg;
            problem.args.wTapStride = kg;
        }
        else
        {
            // [g][t][r][s][c][k]
            const std::array<long long, 6> dims{a.groups, t, r, s, cg, kg};
            problem.w.needed = true;
            problem.w.copy
                = copyOf<6>(dims,
                            {kg * a.wStrides[0], st, sr, ss, a.wStrides[1], a.wStrides[0]},
                            packedStrides(dims),
                            elementBytes);
        }
    }
    if(problem.w.needed)
    {
        problem.w.graphOffset = reflectedStart;
        problem.w.bytes
            = static_cast<size_t>(problem.w.copy.total) * static_cast<size_t>(elementBytes);
    }
    problem.filterKtrsc = krsc;
    problem.filterExpandable = krsc || kcrs;
    problem.wSrcK = a.wStrides[0];
    problem.wSrcC = a.wStrides[1];
    problem.wSrcTap = krsc ? cg : 1;
    layOutWorkspace(problem);

    // The kernels index the positions of one image (input and output) in 32 bits; an image of
    // more than 2^31 - 1 pixels (#14424 X3: 46341 x 46341) takes the large-image configs.
    problem.largeImage = a.inSpatial[0] * a.inSpatial[1] * a.inSpatial[2] > INT_LIMIT
                         || a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2] > INT_LIMIT;

    // Rows of the GEMM, and how far one block's rows reach in the gathered tensor.
    long long imageBytes = 0;
    long long rowsPerImage = 1;
    if(forward)
    {
        problem.rows = k1.m;
        rowsPerImage = a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2];
        imageBytes = a.inSpatial[0] * a.inSpatial[1] * a.inSpatial[2] * a.c * elementBytes;
    }
    else
    {
        long long rowsMax = a.n;
        rowsPerImage = 1;
        for(int axis = 0; axis < SPATIAL; ++axis)
        {
            rowsMax *= divUp(a.inSpatial[axis], a.stride[axis]);
            rowsPerImage *= std::max<long long>(1, a.inSpatial[axis] / a.stride[axis]);
        }
        problem.rows = rowsMax;
        problem.phases = a.stride[0] * a.stride[1] * a.stride[2];
        imageBytes = a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2] * a.k * elementBytes;
    }
    problem.blockSpanBytes = (divUp(MAX_BM, std::max<long long>(1, rowsPerImage)) + 1) * imageBytes;
    if(problem.rows < 1 || divUp(problem.rows, MIN_BM) > INT_LIMIT)
    {
        return std::nullopt;
    }
    problem.base = *base;
    return problem;
}

IgemmProblem withoutFilterStaging(const IgemmProblem& problem)
{
    IgemmProblem direct = problem;
    direct.w = Staging{};
    direct.args.wRowStride = 0;
    direct.args.wTapStride = 0;
    layOutWorkspace(direct);
    return direct;
}

std::optional<IgemmProblem> withFusedGroups(const IgemmProblem& problem, int64_t fuse)
{
    const auto& a = problem.args;
    if(fuse < 2 || a.groups % fuse != 0 || !problem.filterExpandable)
    {
        return std::nullopt;
    }
    IgemmProblem fused = problem;
    fused.args.groups = static_cast<int>(a.groups / fuse);
    fused.args.wRowStride = 0;
    fused.args.wTapStride = 0;
    const long long cg = a.c / a.groups;
    const long long kg = a.k / a.groups;
    const long long taps = static_cast<long long>(a.ft) * a.fr * a.fs;
    Staging w;
    w.needed = true;
    w.expand = true;
    auto& e = w.expandArgs;
    e.k = a.k;
    e.taps = taps;
    e.cg = cg;
    e.kg = kg;
    e.f = fuse;
    e.srcK = problem.wSrcK;
    e.srcC = problem.wSrcC;
    e.srcTap = problem.wSrcTap;
    e.elementBytes = problem.elementBytes;
    w.copy.total = a.k * taps * fuse * cg;
    w.bytes = static_cast<size_t>(w.copy.total) * static_cast<size_t>(problem.elementBytes);
    fused.w = w;
    // The expanded filter is packed K T R S C/g (per fused group): direct-filter readable.
    fused.filterKtrsc = true;
    layOutWorkspace(fused);
    return fused;
}

bool outputChannelsFirst(const IgemmProblem& problem)
{
    const auto& output = problem.direction == ConvDirection::FORWARD ? problem.y : problem.x;
    return output.needed && output.transpose;
}

IgemmProblem withoutActivationStaging(const IgemmProblem& problem, bool removeInput)
{
    IgemmProblem direct = problem;
    (problem.direction == ConvDirection::FORWARD ? direct.y : direct.x) = Staging{};
    if(removeInput)
    {
        (problem.direction == ConvDirection::FORWARD ? direct.x : direct.y) = Staging{};
    }
    layOutWorkspace(direct);
    return direct;
}

std::optional<IgemmProblem> withPatchView(const IgemmProblem& problem)
{
    const auto& b = problem.base.args;
    const bool forward = problem.direction == ConvDirection::FORWARD;
    // An axis folds when the filter tiles it exactly: extent equal to the stride, no padding,
    // unit dilation, and the input a whole number of tiles.
    const auto folds = [&](int axis) {
        return b.filter[axis] == b.stride[axis] && b.prePadding[axis] == 0
               && (b.filter[axis] == 1 || b.dilation[axis] == 1)
               && b.outSpatial[axis] * b.filter[axis] == b.inSpatial[axis];
    };
    if(b.groups != 1 || problem.largeImage)
    {
        return std::nullopt;
    }
    IgemmProblem view = problem;
    auto& v = view.args;
    if(forward)
    {
        if(!folds(2) || b.filter[2] < 2 || b.c * b.filter[2] > INT_LIMIT)
        {
            return std::nullopt;
        }
        // W / S pixels of S C channels; the filter row's S taps become its channels.
        v.c = static_cast<int>(b.c * b.filter[2]);
        v.wi = static_cast<int>(b.inSpatial[2] / b.filter[2]);
        v.fs = 1;
        v.sw = 1;
        v.pw = 0;
        v.dw = 1;
        return view;
    }
    const long long taps = b.filter[0] * b.filter[1] * b.filter[2];
    if(!folds(0) || !folds(1) || !folds(2) || taps < 2 || b.c * taps > INT_LIMIT)
    {
        return std::nullopt;
    }
    // A 1x1, unit-stride problem over dy's positions whose T R S C columns scatter to dx.
    v.c = static_cast<int>(b.c * taps);
    v.di = v.dout;
    v.hi = v.ho;
    v.wi = v.wo;
    v.ft = v.fr = v.fs = 1;
    v.sd = v.sh = v.sw = 1;
    v.pd = v.ph = v.pw = 0;
    v.dd = v.dh = v.dw = 1;
    v.patchT = static_cast<int>(b.filter[0]);
    v.patchR = static_cast<int>(b.filter[1]);
    v.wRowStride = 0;
    v.wTapStride = 0;
    view.phases = 1;
    // The transposed filter is [(t, r, s, c)][k]: the parse's own transpose for a packed
    // K T R S C filter; any other filter (K C T R S, reflected, strided) by the strided copy.
    if(!problem.filterKtrsc)
    {
        const bool flip = b.flipFilter != 0;
        const long long t = b.filter[0];
        const long long r = b.filter[1];
        const long long s = b.filter[2];
        const std::array<long long, 5> dims{t, r, s, b.c, b.k};
        Staging w;
        w.needed = true;
        w.copy = copyOf<5>(dims,
                           {flip ? -b.wStrides[2] : b.wStrides[2],
                            flip ? -b.wStrides[3] : b.wStrides[3],
                            flip ? -b.wStrides[4] : b.wStrides[4],
                            b.wStrides[1],
                            b.wStrides[0]},
                           packedStrides(dims),
                           problem.elementBytes);
        w.graphOffset
            = flip ? (t - 1) * b.wStrides[2] + (r - 1) * b.wStrides[3] + (s - 1) * b.wStrides[4]
                   : 0;
        w.bytes = static_cast<size_t>(w.copy.total) * static_cast<size_t>(problem.elementBytes);
        view.w = w;
    }
    layOutWorkspace(view);
    return view;
}

std::optional<IgemmProblem> withCol2imView(const IgemmProblem& problem)
{
    const auto& b = problem.base.args;
    // The filter packed K T R S C or K C T R S and unreflected (filterExpandable): K-major rows
    // of taps x channels in either order, the GEMM's B read in place.
    if(problem.direction != ConvDirection::BACKWARD_DATA || b.groups != 1 || problem.largeImage
       || !problem.filterExpandable || withPatchView(problem).has_value())
    {
        return std::nullopt;
    }
    const long long taps = b.filter[0] * b.filter[1] * b.filter[2];
    const long long dyPositions = b.n * b.outSpatial[0] * b.outSpatial[1] * b.outSpatial[2];
    const long long dxPositions = b.n * b.inSpatial[0] * b.inSpatial[1] * b.inSpatial[2];
    const long long phaseTaps = divUp(taps, std::max<long long>(1, problem.phases));
    constexpr long long MAX_COLUMN_BYTES = 1LL << 30;
    if(taps < 2 || b.c * taps > INT_LIMIT || dyPositions * taps >= dxPositions * phaseTaps
       || dyPositions * b.c * taps > MAX_COLUMN_BYTES / static_cast<long long>(sizeof(float)))
    {
        return std::nullopt;
    }
    // A 1x1, unit-stride problem over dy's positions with T R S C columns.
    IgemmProblem view = problem;
    auto& v = view.args;
    v.c = static_cast<int>(b.c * taps);
    v.di = v.dout;
    v.hi = v.ho;
    v.wi = v.wo;
    v.ft = v.fr = v.fs = 1;
    v.sd = v.sh = v.sw = 1;
    v.pd = v.ph = v.pw = 0;
    v.dd = v.dh = v.dw = 1;
    v.wRowStride = 0;
    v.wTapStride = 0;
    view.phases = 1;
    view.rows = dyPositions;
    view.w = Staging{};
    // The direct-filter loads read rows of the view's columns, whichever their order.
    view.filterKtrsc = true;
    view.col2imTapMajor = problem.wSrcC == 1;
    view.col2imBytes = static_cast<size_t>(dyPositions * b.c * taps) * sizeof(float);
    layOutWorkspace(view);
    return view;
}

bool packFits(const IgemmProblem& problem, int64_t bk)
{
    const auto& a = problem.args;
    const long long reduction
        = problem.direction == ConvDirection::FORWARD ? a.c / a.groups : a.k / a.groups;
    return reduction < bk && bk % reduction == 0;
}

bool flatFits(const IgemmProblem& problem, int64_t bk)
{
    const auto& a = problem.args;
    const long long reduction
        = problem.direction == ConvDirection::FORWARD ? a.c / a.groups : a.k / a.groups;
    return reduction % bk != 0;
}

bool inputChannelsFirst(const IgemmProblem& problem)
{
    const auto& input = problem.direction == ConvDirection::FORWARD ? problem.x : problem.y;
    return input.needed && input.transpose;
}

namespace
{

/// Whether the activation x (@p onX) or y of @p problem is packed channels-first: staged by the
/// transpose, or not staged (packed channels-last) with every spatial extent 1, where the two
/// layouts coincide.
bool packedChannelsFirst(const IgemmProblem& problem, bool onX)
{
    const auto& staging = onX ? problem.x : problem.y;
    if(staging.needed)
    {
        return staging.transpose;
    }
    const auto& b = problem.base.args;
    const auto& spatial = onX ? b.inSpatial : b.outSpatial;
    return channelsFirst(onX ? b.xStrides : b.yStrides,
                         {b.n, onX ? b.c : b.k, spatial[0], spatial[1], spatial[2]});
}

} // namespace

bool outputPackedChannelsFirst(const IgemmProblem& problem)
{
    return packedChannelsFirst(problem, problem.direction != ConvDirection::FORWARD);
}

bool inputPackedChannelsFirst(const IgemmProblem& problem)
{
    return packedChannelsFirst(problem, problem.direction == ConvDirection::FORWARD);
}

IgemmProblem withoutInputStaging(const IgemmProblem& problem)
{
    IgemmProblem direct = problem;
    (problem.direction == ConvDirection::FORWARD ? direct.x : direct.y) = Staging{};
    layOutWorkspace(direct);
    return direct;
}

IgemmProblem withChannelsFirstLineView(const IgemmProblem& problem)
{
    const auto& b = problem.base.args;
    if(b.inSpatial[0] == 1 && b.inSpatial[1] == 1)
    {
        return problem; // already 1-D
    }
    long long pixels = 1;
    for(int axis = 0; axis < SPATIAL; ++axis)
    {
        if(b.filter[axis] != 1 || b.stride[axis] != 1 || b.prePadding[axis] != 0
           || b.outSpatial[axis] != b.inSpatial[axis])
        {
            return problem;
        }
        pixels *= b.inSpatial[axis];
    }
    if(problem.largeImage || pixels > INT_LIMIT || !inputPackedChannelsFirst(problem)
       || !outputPackedChannelsFirst(problem))
    {
        return problem;
    }
    IgemmProblem view = problem;
    auto& v = view.base.args;
    for(int axis = 0; axis < SPATIAL - 1; ++axis)
    {
        v.inSpatial[axis] = 1;
        v.outSpatial[axis] = 1;
    }
    v.inSpatial[SPATIAL - 1] = pixels;
    v.outSpatial[SPATIAL - 1] = pixels;
    // Packed channels-first strides of the line (the axes of extent 1 take any stride).
    for(auto* strides : {v.xStrides, v.yStrides})
    {
        strides[2] = pixels;
        strides[3] = pixels;
        strides[4] = 1;
    }
    auto& a = view.args;
    a.di = 1;
    a.hi = 1;
    a.wi = static_cast<int>(pixels);
    a.dout = 1;
    a.ho = 1;
    a.wo = static_cast<int>(pixels);
    return view;
}

namespace
{

/// One gather of the channels-first loaders along the width (k1Setup): the gathered row's
/// pixels, the GEMM rows per image, the row stride, the taps' reach and the padding's magnitude.
struct CfxGather
{
    long long in = 0, rows = 0, stride = 1, reach = 0, pad = 0;
};

/// The width gathers a 1-D problem's blocks run: forward one, backward data one per stride phase
/// (k1PhaseAxis: dy at unit stride, the phase's taps a dilation-multiple apart).
std::vector<CfxGather> cfxGathers(const IgemmProblem& problem)
{
    const auto& a = problem.args;
    if(problem.direction == ConvDirection::FORWARD)
    {
        return {{a.wi, a.wo, a.sw, static_cast<long long>(a.fs - 1) * a.dw, a.pw}};
    }
    std::vector<CfxGather> gathers;
    for(long long phi = 0; phi < a.sw; ++phi)
    {
        const long long first = ((phi - a.pw) % a.sw + a.sw) % a.sw;
        const long long count = first < a.wi ? (a.wi - 1 - first) / a.sw + 1 : 0;
        const long long tapStep = a.sw / std::gcd<long long>(a.sw, a.dw);
        long long tapFirst = -1;
        for(long long r = 0; r < a.fs && r < tapStep; ++r)
        {
            if(((r * a.dw - phi) % a.sw + a.sw) % a.sw == 0)
            {
                tapFirst = r;
                break;
            }
        }
        if(count < 1 || tapFirst < 0)
        {
            continue; // the phase's blocks read nothing
        }
        const long long taps = (a.fs - 1 - tapFirst) / tapStep + 1;
        const long long o0 = (first + a.pw - tapFirst * a.dw) / a.sw;
        gathers.push_back({a.wo, count, 1, (taps - 1) * (tapStep * a.dw / a.sw), std::abs(o0)});
    }
    return gathers;
}

} // namespace

bool cfxFits(const IgemmProblem& problem, const CfxTile& tile)
{
    const auto& b = problem.base.args;
    if(problem.largeImage || problem.phases < 1)
    {
        return false;
    }
    for(int axis = 0; axis < 2; ++axis)
    {
        if(b.inSpatial[axis] != 1 || b.outSpatial[axis] != 1 || b.filter[axis] != 1)
        {
            return false;
        }
    }
    const auto& a = problem.args;
    const long long chans
        = problem.direction == ConvDirection::FORWARD ? static_cast<long long>(a.c) : a.k;
    const long long eb = problem.elementBytes;
    const long long mv = 16 / eb; // pixels per channel vector (K1HaloCf, K1TilesCf)
    const auto gathers = cfxGathers(problem);
    for(const auto& g : gathers)
    {
        long long span = g.reach + 1;
        if(tile.halo != 0)
        {
            // The pixels one block's rows reach (K1HaloCf's run, clamped to the images of its
            // first and last row): bm - 1 row steps, plus the gap at each image boundary they
            // cross; where an image's rows reach past the next image's first pixel (a negative
            // gap) the run is bounded by the steps, the taps and the padding instead. Plus the
            // aligned base's vector.
            const long long steps = tile.bm - 1;
            const long long gap = g.in - g.rows * g.stride;
            const long long crossings = (steps + g.rows - 1) / g.rows;
            span = (gap >= 0 ? steps * g.stride + crossings * gap + g.reach + 1
                             : steps * g.stride + g.reach + 1 + g.pad)
                   + (mv - 1);
            const long long hrows = (tile.halo * tile.bm + HALO_CAP + 2 * mv - 1) / mv * mv;
            if(span > hrows)
            {
                return false;
            }
            // LDS: the halo rows, a zero row and the B tile, or the epilogue strips.
            const long long ld = tile.bk + 16 / eb;
            const long long main = (hrows + 1 + tile.bn) * ld * eb;
            const long long cfr = std::min<long long>(tile.bm / tile.wm, 64);
            const long long out = tile.wm * tile.wn * tile.dim * (cfr + 16 / eb) * eb;
            if(std::max(main, out) > tile.ldsBytes)
            {
                return false;
            }
        }
        // 32-bit byte offsets over the images a block's pixels span (from its first image).
        const long long rowsSpan = tile.halo != 0 ? span : tile.bm * g.stride + g.reach;
        if((rowsSpan + 2 * g.in) * chans * eb >= std::numeric_limits<int>::max())
        {
            return false;
        }
    }
    return true;
}

bool haloFits(const IgemmProblem& problem, int64_t bm)
{
    const auto& a = problem.args;
    const bool forward = problem.direction == ConvDirection::FORWARD;
    // The gather the kernels build (k1Setup): forward reads x at the filter's dilation and
    // padding; backward data at stride 1 reads dy with the taps reversed.
    const std::array<long long, 3> in = forward ? std::array<long long, 3>{a.di, a.hi, a.wi}
                                                : std::array<long long, 3>{a.dout, a.ho, a.wo};
    const std::array<long long, 3> out = forward ? std::array<long long, 3>{a.dout, a.ho, a.wo}
                                                 : std::array<long long, 3>{a.di, a.hi, a.wi};
    const std::array<long long, 3> stride{a.sd, a.sh, a.sw};
    const std::array<long long, 3> pad{a.pd, a.ph, a.pw};
    const std::array<long long, 3> dilation{a.dd, a.dh, a.dw};
    const std::array<long long, 3> taps{a.ft, a.fr, a.fs};
    long long lo = 0;
    long long hi = 0;
    long long pixelStride = 1;
    for(size_t i = 3; i-- > 0;)
    {
        if(stride[i] != 1 || in[i] != out[i])
        {
            return false;
        }
        const long long step = forward ? dilation[i] : -dilation[i];
        const long long first = forward ? -pad[i] : pad[i];
        const long long reach = (taps[i] - 1) * step;
        lo += (first + std::min<long long>(reach, 0)) * pixelStride;
        hi += (first + std::max<long long>(reach, 0)) * pixelStride;
        pixelStride *= in[i];
    }
    if(hi - lo > HALO_CAP)
    {
        return false;
    }
    // The staged rows (at most bm + HALO_CAP, rounded up to a loader pass of at most 512 rows)
    // are addressed from the block's first pixel by 32-bit byte offsets.
    const long long chans = forward ? a.c : a.k;
    const long long rows = bm + HALO_CAP + 512;
    return rows * chans * problem.elementBytes < std::numeric_limits<int>::max();
}

bool hasXcds(std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    return arch == "gfx942" || arch == "gfx950";
}

bool blockOrderServes(int64_t bm, int64_t bn, int64_t xcd, std::string_view gcnArchName)
{
    const bool xcds = hasXcds(gcnArchName);
    return xcd != 0 ? xcds : !xcds || bm != 256 || bn != 256;
}

int occupancyTarget(std::string_view atom,
                    int64_t bm,
                    int64_t bn,
                    int64_t bk,
                    int64_t wm,
                    int64_t wn,
                    int64_t swz,
                    std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    const bool bigVgprFile = arch == "gfx1100" || arch == "gfx1101" || arch == "gfx1151";
    return bigVgprFile && atom.rfind("wmma11_", 0) == 0 && bm == 128 && bn == 128 && bk == 64
                   && wm == 4 && wn == 2 && swz != 0
               ? 4
               : 0;
}

long long launchTilesFor(long long rowTiles, long long threads)
{
    constexpr long long DISPATCH_ITEMS = std::numeric_limits<uint32_t>::max();
    return std::min<long long>(rowTiles, DISPATCH_ITEMS / std::max<long long>(1, threads));
}

std::optional<Grid> gridFor(const IgemmProblem& problem, int64_t bm, int64_t bn)
{
    const auto& a = problem.args;
    const long long columns
        = problem.direction == ConvDirection::FORWARD ? a.k / a.groups : a.c / a.groups;
    const long long x = divUp(problem.rows, bm);
    const long long y = divUp(columns, bn);
    const long long z = static_cast<long long>(a.groups) * problem.phases;
    if(x < 1 || x > INT_LIMIT || y > MAX_GRID_YZ || z > MAX_GRID_YZ)
    {
        return std::nullopt;
    }
    return Grid{
        static_cast<unsigned int>(x), static_cast<unsigned int>(y), static_cast<unsigned int>(z)};
}

FlexTailSplit flexTailSplit(long long rowTiles,
                            long long rowWidth,
                            long long kSteps,
                            long long units,
                            bool tailAlone,
                            long long splitCost)
{
    constexpr long long MAX_SPLITS = 16;
    FlexTailSplit best;
    if(rowTiles < 1 || rowWidth < 1 || units < 1)
    {
        return best;
    }
    const long long tiles = rowTiles * rowWidth;
    long long bestCost = divUp(tiles, units) * kSteps;
    const auto consider = [&](long long first) {
        const long long tail = (rowTiles - first) * rowWidth;
        for(long long splits = 2; splits <= MAX_SPLITS; ++splits)
        {
            const long long cost
                = divUp(first * rowWidth, units) * kSteps
                  + divUp(tail * splits, units) * (divUp(kSteps, splits) + splitCost);
            if(cost < bestCost)
            {
                bestCost = cost;
                best.splitRowTile = first;
                best.splits = splits;
            }
        }
    };
    // The row tiles whose tiles fill whole rounds (none when they fill less than one).
    const long long full = tiles / units * units / rowWidth;
    if(tailAlone && full > 0 && full < rowTiles)
    {
        consider(full);
    }
    consider(0);
    return best;
}

int64_t residentBlocksOf(int64_t bm,
                         int64_t bn,
                         int64_t bk,
                         int64_t threads,
                         int64_t wave,
                         int64_t stages,
                         int64_t dma,
                         int elementBytes,
                         std::string_view gcnArchName)
{
    if(wave < 1 || threads < 1 || elementBytes < 1
       || device_units::reportsWorkgroupProcessors(gcnArchName)
       || hipdnn_plugin_sdk::stripArchFeatures(gcnArchName) == "gfx1250")
    {
        return 0;
    }
    const bool gfx950 = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName) == "gfx950";
    const long long wavesPerBlock = divUp(threads, wave);
    // Blocks of fewer than 8 waves keep the default's 2: their split plans measured faster at 2
    // than at an occupancy-derived 1 (wide and DMA tiles on gfx950 and MI325X, 6-8%).
    if(wavesPerBlock < 8)
    {
        return 0;
    }
    const long long lds = std::max<long long>(1, dma != 0 ? dma : stages) * (bm + bn)
                          * (bk + 16 / elementBytes) * elementBytes;
    const long long ldsPerCu = gfx950 ? 160 * 1024 : 64 * 1024;
    const long long registers = divUp(bm * bn / threads + 64, 8) * 8;
    const long long wavesPerSimd = std::clamp<long long>(512 / registers, 1, 8);
    const long long byWaves = 4 * wavesPerSimd / wavesPerBlock;
    const long long byLds = ldsPerCu / std::max<long long>(lds, 1);
    return std::max<long long>(1, std::min(byWaves, byLds));
}

SplitPlan splitPlanFor(const IgemmProblem& problem,
                       int64_t bm,
                       int64_t bn,
                       int64_t bk,
                       const hipdnn_plugin_sdk::ingestor::DeviceProperties& device,
                       int64_t pack,
                       bool noSplit,
                       int64_t splitPolicy,
                       int64_t residentBlocks,
                       int64_t flex)
{
    if(splitPolicy < 0 || splitPolicy > MAX_SPLIT_POLICY)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "conv_igemm: unsupported split policy");
    }
    // Blocks per dispatch unit below which the grid splits, and k-iterations each split keeps.
    // gfx9: 2 per CU, 2 iterations. Legacy WGP-reporting targets: 1 per WGP, 64
    // iterations; their partial store costs about 50 k-steps of a 128-row tile, and
    // splitting grids at least half a block per WGP measured slower (gfx1100/gfx1201).
    // gfx1250 keeps one block per reported CU and eight iterations per split. Matched
    // trials improved underfilled grids; two iterations or two blocks per CU regressed
    // short-reduction controls. This policy is independent of physical CU accounting.
    const bool wgps = device_units::reportsWorkgroupProcessors(device.gcnArchName);
    const bool gfx1250 = hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName) == "gfx1250";
    // gfx9: the config's resident blocks a CU where known (residentBlocksOf), at most 2.
    long long blocksPerUnit = 2;
    if(wgps || gfx1250)
    {
        blocksPerUnit = 1;
    }
    else if(residentBlocks > 0)
    {
        blocksPerUnit = std::min<long long>(residentBlocks, 2);
    }
    const long long minIterations = [&]() -> long long {
        if(gfx1250)
        {
            return 8;
        }
        return wgps ? 64 : 2;
    }();
    constexpr long long MAX_SPLITS = 16;
    constexpr long long MAX_PARTIAL_BYTES = 1LL << 30;
    const auto& a = problem.args;
    const bool forward = problem.direction == ConvDirection::FORWARD;
    SplitPlan plan;
    plan.workspaceBytes = problem.workspaceBytes;
    plan.outElems = forward ? a.m * a.k : static_cast<long long>(a.n) * a.di * a.hi * a.wi * a.c;
    if(splitPolicy == 1)
    {
        return plan;
    }
    const auto grid = gridFor(problem, bm, bn);
    if(!grid.has_value() || noSplit)
    {
        return plan;
    }
    const long long tiles = static_cast<long long>(grid->x) * grid->y * grid->z;
    const long long reduction = forward ? a.c / a.groups : a.k / a.groups;
    const long long taps = static_cast<long long>(a.ft) * a.fr * a.fs;
    const long long phaseTaps = std::max<long long>(1, divUp(taps, problem.phases));
    long long iterations = phaseTaps * divUp(reduction, bk);
    if(pack == 1 && reduction < bk)
    {
        iterations = divUp(phaseTaps, bk / reduction);
    }
    else if(pack == 2)
    {
        iterations = divUp(phaseTaps * reduction, bk);
    }
    const long long wanted = device_units::dispatchUnits(device) * blocksPerUnit;
    const bool underFilled = tiles < wanted;
    if(splitPolicy >= 2)
    {
        // Exactly splitPolicy splits of an under-filled grid, each of at least one k-iteration,
        // within the partials' and the grid's limits; otherwise none (the kernel match declines
        // the config).
        if(!underFilled || splitPolicy > iterations
           || splitPolicy * plan.outElems * static_cast<long long>(sizeof(float))
                  > MAX_PARTIAL_BYTES
           || grid->z * splitPolicy > MAX_GRID_YZ)
        {
            return plan;
        }
        plan.splits = static_cast<int>(splitPolicy);
        plan.partialElems = plan.outElems;
        plan.partialOffset = alignUp(problem.workspaceBytes);
        plan.workspaceBytes
            = plan.partialOffset + static_cast<size_t>(splitPolicy * plan.outElems) * sizeof(float);
        return plan;
    }
    long long splits = 1;
    long long splitRowTile = 0;
    if(flex != 0)
    {
        // One launch of the whole grid (rows within one dispatch) for the tail split.
        const long long rowWidth = static_cast<long long>(grid->y) * grid->z;
        const auto split = flexTailSplit(grid->x,
                                         rowWidth,
                                         iterations,
                                         wanted,
                                         problem.base.epilogue.empty()
                                             && launchTilesFor(grid->x, 512) == grid->x,
                                         flexSplitCost(flex));
        splits = split.splits;
        splitRowTile = split.splitRowTile;
    }
    else if(underFilled)
    {
        splits = std::min({divUp(wanted, tiles), iterations / minIterations, MAX_SPLITS});
    }
    // Rows (output positions forward, input positions backward) of bm a row tile.
    const long long outChans = forward ? a.k : a.c;
    const long long partialElems = plan.outElems - splitRowTile * bm * outChans;
    while(splits > 1
          && (splits * partialElems * static_cast<long long>(sizeof(float)) > MAX_PARTIAL_BYTES
              || grid->z * splits > MAX_GRID_YZ))
    {
        --splits;
    }
    if(splits > 1)
    {
        plan.splits = static_cast<int>(splits);
        plan.splitRowTile = splitRowTile;
        plan.partialElems = partialElems;
        plan.partialOffset = alignUp(problem.workspaceBytes);
        plan.workspaceBytes
            = plan.partialOffset + static_cast<size_t>(splits * partialElems) * sizeof(float);
    }
    return plan;
}

} // namespace conv_igemm

namespace
{

using conv_igemm::IgemmProblem;
using conv_igemm::Staging;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_igemm_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_igemm_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_igemm.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_igemm.dispatch";
/// The staging copy: the same source as the convolution, under another entry point.
constexpr std::string_view COPY_ENTRY_POINT = "ConvIgemmCopy";
constexpr std::string_view TRANSPOSE_ENTRY_POINT = "ConvIgemmTranspose";
constexpr std::string_view EXPAND_ENTRY_POINT = "ConvIgemmExpand";
constexpr std::string_view REDUCE_ENTRY_POINT = "ConvIgemmReduce";
constexpr std::string_view COL2IM_ENTRY_POINT = "ConvIgemmCol2im";
constexpr unsigned int COPY_BLOCK = 256;
/// Bytes a 32-bit buffer offset reaches.
constexpr long long BUFFER_LIMIT = std::numeric_limits<int>::max();
constexpr long long COPY_MAX_BLOCKS = 1LL << 20;
/// Work items one dispatch addresses per grid dimension (the packet's 32-bit grid size).
constexpr long long MAX_DISPATCH_ITEMS = std::numeric_limits<uint32_t>::max();
/// Filter taps along one axis the flex tile serves (one in-image bit per tap of each axis).
constexpr long long FLEX_MAX_TAPS = 10;

constexpr std::string_view X_TOKEN = "conv_igemm.x.uid";
constexpr std::string_view W_TOKEN = "conv_igemm.w.uid";
constexpr std::string_view Y_TOKEN = "conv_igemm.y.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_igemm.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_igemm.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_igemm.flops";

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
    return std::nullopt;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem
        = conv_igemm::parseIgemmProblem(context, direction, conv_naive::Fusion::EPILOGUE);
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

/// A config's compile-time choices, read from its metadata.
struct Tile
{
    std::string atom;
    conv_igemm::AtomInfo info;
    bool split = false; ///< three-bf16
    int64_t bm = 0, bn = 0, bk = 0, wm = 0, wn = 0, threads = 0, vec = 0, stages = 0;
    int64_t xcd = 0; ///< XCD-contiguous block order
    int64_t halo = 0; ///< the halo A loop
    int64_t wdirect = 0; ///< backward data reads the K T R S C/g filter directly
    int64_t pack = 0; ///< 1 k-steps of whole taps, 2 the flat reduction
    int64_t cf = 0; ///< 1 channels-first output; 2 pointwise channels-first input and output
    int64_t swz = 0; ///< XOR-swizzled LDS tiles read in 16-byte chunks
    int64_t dma = 0; ///< gfx950: buffer loads straight into this many swizzled LDS stages
    int64_t fuse = 0; ///< groups run as one block-diagonal group (group fusion), 0 none
    int64_t large = 0; ///< 64-bit positions within an image (K1_LARGE): images past 2^31 - 1 px
    int64_t patch = 0; ///< patch convolutions as one GEMM (withPatchView, K1_PATCH)
    int64_t cfx = 0; ///< 1-D channels-first activations read in place (CONV_IGEMM_CFX)
    int64_t splitPolicy = 0; ///< 0 automatic split-K, 1 none, N >= 2 exactly N (splitPlanFor)
    int64_t flex = 0; ///< the flex tile (CONV_IGEMM_FLEX): gfx950, flattened positions, ping-pong

    int64_t blockThreads() const
    {
        return info.wave == 0 ? threads : wm * wn * info.wave;
    }
};

/// Whether the kernels address @p problem with 32-bit buffer offsets (K1_BUFFER): a block's rows
/// and the filter stay under 2^31 bytes.
bool bufferLoads(const IgemmProblem& problem)
{
    const auto& a = problem.args;
    const long long filterBytes = static_cast<long long>(a.k) * a.ft * a.fr * a.fs
                                  * (a.c / a.groups) * problem.elementBytes;
    return problem.blockSpanBytes < BUFFER_LIMIT && filterBytes < BUFFER_LIMIT;
}

/// The atoms the patch tile runs: 16x16 matrix-core atoms of 16-bit operands.
bool patchTileAtom(std::string_view atom)
{
    const bool family = atom.rfind("mfma16", 0) == 0 || atom.rfind("wmma1", 0) == 0;
    return family && atom.find("f32") == std::string_view::npos;
}

/// LDS bytes of the patch tile (k1PatchMma): two stages of the A tile and the B tile (K-major
/// for gfx950 backward data), or the epilogue strips if larger.
long long patchTileLds(const Tile& tile, bool backward, std::string_view arch)
{
    const long long ld = tile.vec == 8 ? tile.bk : tile.bk + 8;
    const bool kMajor
        = backward && tile.wdirect != 0 && arch == "gfx950" && tile.atom.rfind("mfma16x32", 0) == 0;
    const long long b = kMajor ? tile.bk * (tile.bn + 8) : tile.bn * ld;
    const long long stages = 2 * (tile.bm * ld + b) * 2;
    const long long out = tile.wm * tile.wn * 16 * (tile.bn / tile.wn + 8) * 2;
    return std::max(stages, out);
}

/// The flex tile's form by its atom: 0 none, 1 LDS DMA (16x16x32, gfx950), 2 register-staged
/// (gfx942).
int64_t flexForm(const Tile& tile)
{
    if(tile.flex == 0)
    {
        return 0;
    }
    return tile.atom.rfind("mfma16x32_", 0) == 0 ? 1 : 2;
}

std::optional<Tile> tileOf(const KernelDefinition& kernel)
{
    Tile tile;
    tile.atom = kernel.getStringMetadata("atom");
    const auto info = conv_igemm::atomNamed(tile.atom);
    const auto& math = kernel.getStringMetadata("math");
    if(!info.has_value() || (math != "native" && math != "bf16x3"))
    {
        return std::nullopt;
    }
    tile.info = *info;
    tile.split = math == "bf16x3";
    tile.bm = kernel.getIntMetadata("bm");
    tile.bn = kernel.getIntMetadata("bn");
    tile.bk = kernel.getIntMetadata("bk");
    tile.wm = kernel.getIntMetadata("wm");
    tile.wn = kernel.getIntMetadata("wn");
    tile.threads = kernel.getIntMetadata("threads");
    tile.vec = kernel.getIntMetadata("vec");
    tile.stages = kernel.getIntMetadata("stages");
    tile.xcd = kernel.getIntMetadata("xcd");
    tile.halo = kernel.getIntMetadata("halo");
    tile.wdirect = kernel.getIntMetadata("wdirect");
    tile.pack = kernel.getIntMetadata("pack");
    tile.cf = kernel.getIntMetadata("cf");
    tile.swz = kernel.getIntMetadata("swz");
    tile.dma = kernel.getIntMetadata("dma");
    tile.fuse = kernel.getIntMetadata("fuse");
    tile.large = kernel.getIntMetadata("large");
    tile.patch = kernel.getIntMetadata("patch");
    tile.splitPolicy = kernel.getIntMetadata("split_policy");
    tile.cfx = kernel.getIntMetadata("cfx");
    tile.flex = kernel.metadata.count("flex") != 0 ? kernel.getIntMetadata("flex") : 0;
    // The flex tile: 256 positions x 256, 192 or 128 channels of 16-bit 16x16x32 atoms on 4 x 2
    // waves, 64-channel k-steps, 8-wide vectors, XCD order; any split policy; every other variant
    // knob off. The register-staged form (16x16x16 / 16x16x4 atoms, gfx942): one 64 KiB-class LDS
    // stage, BN 128, 128-byte reduction steps (64 16-bit or 32 fp32 channels, 16-byte vectors),
    // and backward data may read the graph's filter in place (wdirect).
    const bool flexReg
        = tile.atom == "mfma16_f16" || tile.atom == "mfma16_bf16" || tile.atom == "mfma16_f32";
    const int64_t flexElem = tile.atom == "mfma16_f32" ? 4 : 2;
    if(tile.flex != 0
       && (tile.flex != 1
           || (tile.atom != "mfma16x32_f16" && tile.atom != "mfma16x32_bf16" && !flexReg)
           || tile.split || tile.bm != 256 || (tile.bn != 256 && tile.bn != 192 && tile.bn != 128)
           || (flexReg && tile.bn != 128) || tile.bk != 128 / flexElem || tile.wm != 4
           || tile.wn != 2 || tile.vec != 16 / flexElem || tile.stages != 1 || tile.xcd != 1
           || tile.halo != 0 || (tile.wdirect != 0 && !flexReg) || tile.pack != 0 || tile.cf != 0
           || tile.swz != 0 || tile.dma != 0 || tile.fuse != 0 || tile.large != 0 || tile.patch != 0
           || tile.cfx != 0))
    {
        return std::nullopt;
    }
    // Channels-first input: the matrix-core kernel's halo loop (halo h: h input pixels a row) with
    // the channels-first output, or the FMA kernel without a halo; native math, plain tiles.
    if(tile.cfx != 0
       && (tile.cfx != 1 || tile.split || tile.info.tf32 || tile.swz != 0 || tile.dma != 0
           || tile.pack != 0 || tile.wdirect != 0 || tile.fuse != 0 || tile.large != 0
           || tile.patch != 0 || tile.stages != 1
           || (tile.info.wave != 0 ? tile.cf != 1 || tile.halo < 1 || tile.halo > 2
                                   : tile.halo != 0)))
    {
        return std::nullopt;
    }
    if((tile.large != 0
        && (tile.large != 1 || tile.xcd != 0 || tile.halo != 0 || tile.wdirect != 0
            || tile.pack != 0 || tile.cf != 0 || tile.swz != 0 || tile.dma != 0 || tile.fuse != 0
            || tile.split))
       || tile.bm < 1 || tile.bn < 1 || tile.bk < 1 || tile.vec < 1 || tile.blockThreads() < 1
       || (tile.xcd != 0 && tile.xcd != 1) || (tile.cfx == 0 && tile.halo != 0 && tile.halo != 1)
       || tile.splitPolicy < 0 || tile.splitPolicy > conv_igemm::MAX_SPLIT_POLICY
       || (tile.halo != 0 && (tile.info.wave == 0 || tile.stages != 1))
       || (tile.wdirect != 0 && tile.wdirect != 1) || tile.pack < 0 || tile.pack > 2
       || (tile.pack != 0 && (tile.halo != 0 || tile.wdirect != 0)) || tile.cf < 0 || tile.cf > 2
       || (tile.cf != 0 && tile.info.wave == 0)
       || (tile.cf == 2
           && (tile.atom != "wmma125_f16" || tile.split || tile.bm != 64 || tile.bn != 64
               || tile.bk != 32 || tile.wm != 2 || tile.wn != 2 || tile.vec != 8 || tile.stages != 1
               || tile.halo != 0 || tile.wdirect != 0 || tile.pack != 0 || tile.swz != 0
               || tile.dma != 0 || tile.fuse != 0 || tile.splitPolicy != 0))
       || (tile.swz != 0 && tile.swz != 1)
       || (tile.swz != 0
           && (tile.info.wave == 0 || tile.split || tile.halo != 0 || tile.wdirect != 0
               || tile.pack == 1 || tile.vec != 8
               || (tile.bk != 16 && tile.bk != 32 && tile.bk != 64) || tile.info.swizzleK == 0
               || tile.bk % tile.info.swizzleK != 0))
       || (tile.dma != 0 && (tile.dma < 2 || tile.dma > 4 || tile.swz == 0 || tile.pack == 1))
       || (tile.fuse != 0 && tile.fuse != 2 && tile.fuse != 4 && tile.fuse != 8 && tile.fuse != 16)
       || (tile.fuse != 0 && (tile.halo != 0 || tile.pack != 0))
       || (tile.patch != 0
           && (tile.patch < 0 || tile.patch > 3 || tile.large != 0 || tile.fuse != 0))
       // Col2im (withCol2imView): the general loop reading the graph's filter in place, plain
       // tiles (no halo, tap packing, channels-first output or input, DMA), native math.
       || (tile.patch == 3
           && (tile.wdirect != 1 || tile.split || tile.halo != 0 || tile.pack != 0 || tile.cf != 0
               || tile.dma != 0 || tile.cfx != 0 || tile.flex != 0))
       // The patch tile (k1PatchMma): 16-bit 16x16 matrix-core atoms, every variant knob off.
       || (tile.patch == 2
           && (!patchTileAtom(tile.atom) || tile.split || tile.stages != 1 || tile.xcd != 0
               || tile.halo != 0 || tile.pack != 0 || tile.cf != 0 || tile.swz != 0 || tile.dma != 0
               || (tile.vec != 4 && tile.vec != 8) || tile.bm % (tile.wm * 16) != 0
               || tile.bn % (tile.wn * 16) != 0 || (tile.bk != 16 && tile.bk != 32 && tile.bk != 64)
               || tile.info.swizzleK == 0 || tile.bk % tile.info.swizzleK != 0)))
    {
        return std::nullopt;
    }
    return tile;
}

/// Kernel-scoped applicability: the config runs on this device and computes this problem
/// correctly. A large-image config (64-bit positions within an image) serves exactly the images
/// past 2^31 - 1 pixels, the other configs exactly the rest. Its atom must exist on the device and take the graph's element type (fp32 also
/// through rounded operands -- xf32, three-bf16 -- the TF32 configs, which only the
/// CONV_IGEMM_*_TF32 engines list), and its channel vector
/// must divide the reduction channels. An XCD-order config is served only where the device has
/// several XCDs (elsewhere it is its plain twin), a halo config only where the problem fits
/// the halo loop. A direct-filter config serves backward data whose filter is packed K T R S C/g
/// with its input channels a multiple of the vector (16-bit: output channels even); a tap-packing
/// config only reductions shorter than its k-step that divide it, a flat-reduction config (pack
/// 2) only reductions its k-step does not divide. A channels-first-output config
/// serves outputs (forward y, backward data dx) that are packed channels-first. A swizzled-tile
/// config (16-bit atoms with 8-, 16- or 32-byte fragments and a k-step the chunked reads divide,
/// vector 8, no halo, direct filter or tap packing; checked in tileOf) serves whatever its plain
/// twin serves. A DMA config (swizzled tiles loaded straight into LDS) serves gfx950 only, and only
/// problems the 32-bit buffer offsets address. A patch config serves only patch convolutions
/// (withPatchView), and every check above applies to the problem as it runs it; in backward data
/// it writes channels-last dx (no channels-first output) with a channel vector dividing S C.
/// The config's resident blocks a CU for the split plan (conv_igemm::residentBlocksOf).
int64_t residentBlocksFor(const Tile& tile,
                          const conv_igemm::IgemmProblem& problem,
                          const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    // The flex tile runs one block a CU: the LDS-DMA form's two activation and three filter
    // stages fill a gfx950 CU's LDS, the register-staged form's 48 KiB stage a gfx942 one's.
    if(tile.flex != 0)
    {
        return 1;
    }
    return conv_igemm::residentBlocksOf(tile.bm,
                                        tile.bn,
                                        tile.bk,
                                        tile.blockThreads(),
                                        tile.info.wave,
                                        tile.stages,
                                        tile.dma,
                                        problem.elementBytes,
                                        device.gcnArchName);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto direction = directionFor(kernel);
    const auto tile = tileOf(kernel);
    if(!direction.has_value() || !tile.has_value())
    {
        return false;
    }
    auto problem = conv_igemm::parseIgemmProblem(context, *direction, conv_naive::Fusion::EPILOGUE);
    if(problem.has_value() && tile->fuse != 0)
    {
        // Group fusion serves small groups (fused groups of at most 64 channels each way) whose
        // expanded filter forward or the direct-filter backward-data configs read.
        const auto& p = problem->args;
        const long long cg = p.c / p.groups;
        const long long kg = p.k / p.groups;
        if((*direction == ConvDirection::BACKWARD_DATA && tile->wdirect == 0)
           || cg * tile->fuse > 64 || kg * tile->fuse > 64)
        {
            return false;
        }
        problem = conv_igemm::withFusedGroups(*problem, tile->fuse);
    }
    if(problem.has_value() && tile->patch == 3)
    {
        // Col2im configs: backward data its view serves (withCol2imView).
        if(*direction != ConvDirection::BACKWARD_DATA)
        {
            return false;
        }
        problem = conv_igemm::withCol2imView(*problem);
    }
    else if(problem.has_value() && tile->patch != 0)
    {
        // Patch configs serve exactly the patch convolutions; backward data writes dx
        // channels-last, in runs of S C columns its channel vector divides.
        problem = conv_igemm::withPatchView(*problem);
        if(problem.has_value() && *direction == ConvDirection::BACKWARD_DATA)
        {
            const auto& v = problem->args;
            const long long run = v.c / (static_cast<long long>(v.patchT) * v.patchR);
            if(tile->cf != 0 || run % tile->vec != 0)
            {
                return false;
            }
        }
        if(problem.has_value() && tile->patch == 2)
        {
            // The patch tile: 32-bit buffer offsets over the whole read operands, its LDS within
            // the device's, and the filter's columns whole vectors (backward data).
            const auto& v = problem->args;
            const bool forward = *direction == ConvDirection::FORWARD;
            const long long readElems
                = forward ? static_cast<long long>(v.n) * v.di * v.hi * v.wi * v.c : v.m * v.k;
            const long long filterElems = static_cast<long long>(v.k) * v.c * v.ft * v.fr * v.fs;
            const auto arch
                = hipdnn_plugin_sdk::stripArchFeatures(context.deviceProperties.gcnArchName);
            const long long ldsLimit = arch == "gfx950" ? 160 * 1024 : 64 * 1024;
            if(readElems * problem->elementBytes >= BUFFER_LIMIT
               || filterElems * problem->elementBytes >= BUFFER_LIMIT
               || patchTileLds(*tile, !forward, arch) > ldsLimit
               || (!forward && v.c % tile->vec != 0))
            {
                return false;
            }
        }
    }
    // The split-K plan is sized from the device's units: none reported, nothing to size.
    if(!problem.has_value() || device_units::dispatchUnits(context.deviceProperties) < 1
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->base.dataType)
       || !conv_igemm::atomRunsOn(tile->atom, context.deviceProperties.gcnArchName)
       // Large-image configs serve exactly the images past 2^31 - 1 pixels.
       || (tile->large != 0) != problem->largeImage)
    {
        return false;
    }
    // An explicit split policy: not on the three-bf16 split form or the patch configs, which
    // never split.
    if(tile->splitPolicy != 0 && (tile->split || tile->patch != 0))
    {
        return false;
    }
    const auto element = conv_igemm::atomElement(tile->atom);
    const auto wanted = conv_igemm::dataElement(problem->base.dataType);
    if(tile->split)
    {
        if(wanted != "f32" || element != "bf16")
        {
            return false;
        }
    }
    else if(element != "any" && element != wanted)
    {
        return false;
    }
    const auto& a = problem->args;
    const long long reduction
        = *direction == ConvDirection::FORWARD ? a.c / a.groups : a.k / a.groups;
    if(reduction % tile->vec != 0)
    {
        return false;
    }
    if(!conv_igemm::blockOrderServes(
           tile->bm, tile->bn, tile->xcd, context.deviceProperties.gcnArchName))
    {
        return false;
    }
    if(tile->halo != 0 && tile->cfx == 0 && !conv_igemm::haloFits(*problem, tile->bm))
    {
        return false;
    }
    if(tile->cfx != 0)
    {
        // Channels-first input read in place: a packed channels-first activation and output (the
        // FMA kernel writes it itself), a 1-D problem (or a pointwise one, as its 1-D line) the
        // channels-first loaders fit, and 32-bit buffer offsets.
        problem = conv_igemm::withChannelsFirstLineView(*problem);
        const auto arch
            = hipdnn_plugin_sdk::stripArchFeatures(context.deviceProperties.gcnArchName);
        conv_igemm::CfxTile fit;
        fit.bm = tile->bm;
        fit.bn = tile->bn;
        fit.bk = tile->bk;
        fit.wm = tile->info.wave != 0 ? tile->wm : 1;
        fit.wn = tile->info.wave != 0 ? tile->wn : 1;
        fit.halo = tile->halo;
        fit.dim = tile->atom.rfind("mfma32", 0) == 0 ? 32 : 16;
        fit.ldsBytes = arch == "gfx950" ? 160 * 1024 : 64 * 1024;
        if(!conv_igemm::inputPackedChannelsFirst(*problem)
           || !conv_igemm::outputPackedChannelsFirst(*problem)
           || !conv_igemm::cfxFits(*problem, fit) || !bufferLoads(*problem))
        {
            return false;
        }
    }
    if(tile->wdirect != 0
       && (*direction != ConvDirection::BACKWARD_DATA || !problem->filterKtrsc
           || (a.c / a.groups) % tile->vec != 0
           || (problem->elementBytes == 2 && (a.k / a.groups) % 2 != 0)))
    {
        return false;
    }
    if(tile->pack == 1 && !conv_igemm::packFits(*problem, tile->bk))
    {
        return false;
    }
    if(tile->pack == 2 && !conv_igemm::flatFits(*problem, tile->bk))
    {
        return false;
    }
    if(tile->cf != 0 && tile->cfx == 0 && !conv_igemm::outputChannelsFirst(*problem))
    {
        return false;
    }
    if(tile->cf == 2)
    {
        const long long spatial = static_cast<long long>(a.di) * a.hi * a.wi;
        if(*direction != ConvDirection::FORWARD || !problem->x.needed || !problem->x.transpose
           || a.groups != 1 || a.c > conv_igemm::INT_LIMIT - tile->bk
           || spatial > conv_igemm::INT_LIMIT || spatial % tile->vec != 0
           || context.graph.getTensorMap().at(problem->xUid)->alignment() < 16
           || (!problem->w.needed && !bufferLoads(*problem)
               && context.graph.getTensorMap().at(problem->wUid)->alignment() < 16))
        {
            return false;
        }
        for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
        {
            if(problem->base.args.filter[axis] != 1 || problem->base.args.stride[axis] != 1
               || problem->base.args.prePadding[axis] != 0
               || problem->base.args.inSpatial[axis] != problem->base.args.outSpatial[axis])
            {
                return false;
            }
        }
    }
    if(tile->dma != 0
       && (hipdnn_plugin_sdk::stripArchFeatures(context.deviceProperties.gcnArchName) != "gfx950"
           || !bufferLoads(*problem)))
    {
        return false;
    }
    // The flex tile: gfx950 (LDS-DMA form, 16x16x32 atoms) or gfx942 (register-staged form),
    // 32-bit buffer offsets, backward data at unit stride (one phase), and at most FLEX_MAX_TAPS
    // taps along each filter axis (its per-row in-image tap bits).
    if(tile->flex != 0
       && (hipdnn_plugin_sdk::stripArchFeatures(context.deviceProperties.gcnArchName)
               != (tile->atom.rfind("mfma16x32_", 0) == 0 ? "gfx950" : "gfx942")
           || !bufferLoads(*problem) || problem->phases != 1 || a.ft > FLEX_MAX_TAPS
           || a.fr > FLEX_MAX_TAPS || a.fs > FLEX_MAX_TAPS))
    {
        return false;
    }
    const auto grid = conv_igemm::gridFor(*problem, tile->bm, tile->bn);
    if(!grid.has_value())
    {
        return false;
    }
    // The XCD block order numbers a launch's blocks in 32 bits.
    const long long launchTiles = conv_igemm::launchTilesFor(grid->x, tile->blockThreads());
    if(tile->xcd != 0
       && launchTiles * grid->y * static_cast<long long>(grid->z) > MAX_DISPATCH_ITEMS)
    {
        return false;
    }
    // A staging transpose is one launch of a block per tile.
    for(const Staging* staging : {&problem->x, &problem->w, &problem->y})
    {
        // Check only copies this config dispatches: channels-first output/input and direct
        // filters bypass staging, including its single-dispatch transpose grid limit.
        if((tile->cf != 0
            && staging == (*direction == ConvDirection::FORWARD ? &problem->y : &problem->x))
           || (tile->cf == 2 && staging == &problem->x)
           || (tile->fuse == 0 && tile->wdirect != 0 && staging == &problem->w))
        {
            continue;
        }
        if(staging->needed && staging->transpose)
        {
            const auto& t = staging->transposeArgs;
            constexpr long long TILE = CONV_IGEMM_TRANSPOSE_TILE;
            if((t.cols + TILE - 1) / TILE * ((t.rows + TILE - 1) / TILE) * t.batches
               > MAX_DISPATCH_ITEMS / COPY_BLOCK)
            {
                return false;
            }
        }
    }
    // An explicit split policy only where its plan differs from the default's (and, for N
    // splits, is exactly N): autotune then times distinct plans only.
    if(tile->splitPolicy != 0)
    {
        const auto resident = residentBlocksFor(*tile, *problem, context.deviceProperties);
        const auto automatic = conv_igemm::splitPlanFor(*problem,
                                                        tile->bm,
                                                        tile->bn,
                                                        tile->bk,
                                                        context.deviceProperties,
                                                        tile->pack,
                                                        false,
                                                        0,
                                                        resident,
                                                        flexForm(*tile));
        const auto chosen = conv_igemm::splitPlanFor(*problem,
                                                     tile->bm,
                                                     tile->bn,
                                                     tile->bk,
                                                     context.deviceProperties,
                                                     tile->pack,
                                                     false,
                                                     tile->splitPolicy,
                                                     resident,
                                                     flexForm(*tile));
        if((chosen.splits == automatic.splits && chosen.splitRowTile == automatic.splitRowTile)
           || (tile->splitPolicy >= 2 && chosen.splits != tile->splitPolicy))
        {
            return false;
        }
    }
    return true;
}

/// The problem a config runs: a direct-filter config reads the graph's filter, and a
/// channels-first-output config writes the graph's output, so nothing stages them; a group-fusion
/// config stages the expanded filter; a patch config runs the patch view.
std::optional<IgemmProblem>
    problemFor(const MatchContext& context, ConvDirection direction, const Tile& tile)
{
    auto problem = conv_igemm::parseIgemmProblem(context, direction, conv_naive::Fusion::EPILOGUE);
    if(problem.has_value() && tile.fuse != 0)
    {
        // The expanded filter is the staged operand, direct filter or not.
        problem = conv_igemm::withFusedGroups(*problem, tile.fuse);
    }
    if(problem.has_value() && tile.patch == 3)
    {
        problem = conv_igemm::withCol2imView(*problem);
    }
    else if(problem.has_value() && tile.patch != 0)
    {
        problem = conv_igemm::withPatchView(*problem);
    }
    if(problem.has_value() && tile.fuse == 0 && tile.wdirect != 0)
    {
        problem = conv_igemm::withoutFilterStaging(*problem);
    }
    if(problem.has_value() && (tile.cf != 0 || tile.cfx != 0))
    {
        problem = conv_igemm::withoutActivationStaging(*problem, tile.cf == 2);
    }
    if(problem.has_value() && tile.cfx != 0)
    {
        problem = conv_igemm::withoutInputStaging(conv_igemm::withChannelsFirstLineView(*problem));
    }
    return problem;
}

/// The argument the kernels take, and a ConvEpiArgs after it when an epilogue is fused. Only
/// consulted for a pre-built kernel.
const std::vector<KernelArgument>& convSignature(bool fused)
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K1Args)), 0, ""}};
    static const std::vector<KernelArgument> s_fusedSignature{
        {"by_value", static_cast<uint32_t>(sizeof(K1Args)), 0, ""},
        {"by_value", static_cast<uint32_t>(sizeof(ConvEpiArgs)), 0, ""}};
    return fused ? s_fusedSignature : s_signature;
}

const std::vector<KernelArgument>& reduceSignature(bool fused)
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K1ReduceArgs)), 0, ""}};
    static const std::vector<KernelArgument> s_fusedSignature{
        {"by_value", static_cast<uint32_t>(sizeof(K1ReduceArgs)), 0, ""},
        {"by_value", static_cast<uint32_t>(sizeof(ConvEpiArgs)), 0, ""}};
    return fused ? s_fusedSignature : s_signature;
}

long long divUpBlocks(long long elements)
{
    return (elements + COPY_BLOCK - 1) / COPY_BLOCK;
}

const std::vector<KernelArgument>& transposeSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvIgemmTransposeArgs)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& expandSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvIgemmExpandArgs)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& col2imSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K1Col2imArgs)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& copySignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvIgemmCopyArgs)), 0, ""}};
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
                                                       "conv_igemm: unsupported element type");
    }
}

class PreparedConvIgemm : public IngestorPreparedDispatch
{
public:
    PreparedConvIgemm(IngestorKernelCode conv,
                      std::optional<IngestorKernelCode> convTail,
                      long long launchTiles,
                      long long rowTiles,
                      std::optional<IngestorKernelCode> reduce,
                      std::array<std::optional<IngestorKernelCode>, 3> copies,
                      IgemmProblem problem,
                      const conv_igemm::SplitPlan& plan)
        : _conv(std::move(conv))
        , _convTail(std::move(convTail))
        , _launchTiles(launchTiles)
        , _rowTiles(rowTiles)
        , _reduce(std::move(reduce))
        , _copies(std::move(copies))
        , _problem(std::move(problem))
        , _plan(plan)
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

    /// The last launch's kernel when the row tiles do not divide into whole launches.
    compilation::IRunnableKernel& convTailForStream(hipStream_t stream) const
    {
        return _convTail.has_value() ? _convTail->kernelForStream(stream)
                                     : _conv.kernelForStream(stream);
    }

    /// Row tiles per launch, and in all: a grid past one dispatch's 2^32 work items along x
    /// runs as consecutive launches of launchTiles() row tiles (the last one the remainder).
    long long launchTiles() const
    {
        return _launchTiles;
    }
    long long rowTiles() const
    {
        return _rowTiles;
    }

    /// The staging copy of x (0), w (1) or y (2).
    compilation::IRunnableKernel& copyForStream(size_t operand, hipStream_t stream) const
    {
        return _copies[operand]->kernelForStream(stream);
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce->kernelForStream(stream);
    }

    const IgemmProblem& problem() const
    {
        return _problem;
    }

    const conv_igemm::SplitPlan& plan() const
    {
        return _plan;
    }

private:
    IngestorKernelCode _conv;
    std::optional<IngestorKernelCode> _convTail;
    long long _launchTiles;
    long long _rowTiles;
    std::optional<IngestorKernelCode> _reduce;
    std::array<std::optional<IngestorKernelCode>, 3> _copies;
    IgemmProblem _problem;
    conv_igemm::SplitPlan _plan;
};

class ConvIgemmDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvIgemmDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                             const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The staged operands, and the split-K partials of this config.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto direction = directionFor(kernel);
        if(!direction.has_value())
        {
            return 0;
        }
        const auto tile = tileOf(kernel);
        if(!tile.has_value())
        {
            return 0;
        }
        const auto problem = problemFor(context, *direction, *tile);
        if(!problem.has_value())
        {
            return 0;
        }
        return conv_igemm::splitPlanFor(
                   *problem,
                   tile->bm,
                   tile->bn,
                   tile->bk,
                   context.deviceProperties,
                   tile->pack,
                   tile->patch != 0,
                   tile->splitPolicy,
                   residentBlocksFor(*tile, *problem, context.deviceProperties),
                   flexForm(*tile))
            .workspaceBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto direction = directionFor(kernel);
        const auto tile = tileOf(kernel);
        std::optional<IgemmProblem> problem;
        if(direction.has_value() && tile.has_value())
        {
            problem = problemFor(context, *direction, *tile);
        }
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!problem.has_value() || !tile.has_value() || !x.has_value() || !w.has_value()
           || !y.has_value() || *x != problem->xUid || *w != problem->wUid || *y != problem->yUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_igemm dispatch was given a graph its matcher did not admit");
        }
        const auto grid = conv_igemm::gridFor(*problem, tile->bm, tile->bn);
        if(!grid.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_igemm dispatch was given a config its kernel match declines");
        }
        problem->args.xcds = device_units::xcdCount(context.deviceProperties);

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_IGEMM_T", elementTypeFor(problem->base.dataType));
        options.add("CONV_IGEMM_FMA", std::string(tile->info.wave == 0 ? "1" : "0"));
        if(tile->info.wave != 0)
        {
            options.add("CONV_IGEMM_ATOM", std::string(tile->info.kernelType));
        }
        options.add("CONV_IGEMM_MATH", std::string(tile->split ? "1" : "0"));
        options.add("CONV_IGEMM_BM", std::to_string(tile->bm));
        options.add("CONV_IGEMM_BN", std::to_string(tile->bn));
        options.add("CONV_IGEMM_BK", std::to_string(tile->bk));
        options.add("CONV_IGEMM_WM", std::to_string(tile->wm));
        options.add("CONV_IGEMM_WN", std::to_string(tile->wn));
        options.add("CONV_IGEMM_THREADS", std::to_string(tile->threads));
        options.add("CONV_IGEMM_VEC", std::to_string(tile->vec));
        options.add("CONV_IGEMM_STAGES", std::to_string(tile->stages));
        options.add("CONV_IGEMM_XCD", std::to_string(tile->xcd));
        options.add("CONV_IGEMM_HALO", std::to_string(tile->halo));
        options.add("CONV_IGEMM_WD", std::to_string(tile->wdirect));
        options.add("CONV_IGEMM_PACK", std::to_string(tile->pack));
        options.add("CONV_IGEMM_CF", std::to_string(tile->cf));
        options.add("CONV_IGEMM_SWZ", std::to_string(tile->swz));
        options.add("CONV_IGEMM_DMA", std::to_string(tile->dma));
        options.add("CONV_IGEMM_DIR",
                    std::string(*direction == ConvDirection::FORWARD ? "0" : "1"));
        options.add("K1_BUFFER", std::string(bufferLoads(*problem) ? "1" : "0"));
        options.add("K1_LARGE", std::to_string(tile->large));
        options.add("K1_PATCH", std::to_string(tile->patch));
        if(tile->cfx != 0)
        {
            options.add("CONV_IGEMM_CFX", std::to_string(tile->cfx));
        }
        if(tile->flex != 0)
        {
            options.add("CONV_IGEMM_FLEX", std::to_string(flexForm(*tile)));
        }
        const int waves = conv_igemm::occupancyTarget(tile->atom,
                                                      tile->bm,
                                                      tile->bn,
                                                      tile->bk,
                                                      tile->wm,
                                                      tile->wn,
                                                      tile->swz,
                                                      context.deviceProperties.gcnArchName);
        if(waves > 0)
        {
            options.add("CONV_IGEMM_WAVES_MIN", std::to_string(waves));
            options.add("CONV_IGEMM_WAVES_MAX", std::to_string(waves));
        }

        const auto plan
            = conv_igemm::splitPlanFor(*problem,
                                       tile->bm,
                                       tile->bn,
                                       tile->bk,
                                       context.deviceProperties,
                                       tile->pack,
                                       tile->patch != 0,
                                       tile->splitPolicy,
                                       residentBlocksFor(*tile, *problem, context.deviceProperties),
                                       flexForm(*tile));
        // The fused epilogue: in the convolution, and in the split-K reduce that writes the
        // output when the reduction is split. Fusion off adds no option, so the code is that of
        // an unfused build.
        const auto& epilogue = problem->base.epilogue;
        const bool fused = !epilogue.empty();
        conv_epilogue::addEpilogueDefines(options, epilogue);
        auto conv = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, convSignature(fused));
        const auto threads = static_cast<unsigned int>(tile->blockThreads());
        const unsigned int gridZ = grid->z * static_cast<unsigned int>(plan.splits);
        // One dispatch addresses at most 2^32 - 1 work items along x: more row tiles than that
        // run as consecutive launches (K1Args::tileBase), the last one taking the remainder.
        const long long rowTiles = grid->x;
        const long long launchTiles = conv_igemm::launchTilesFor(rowTiles, threads);
        conv.setBlockSize(threads, 1, 1);
        conv.setGridSize(static_cast<unsigned int>(launchTiles), grid->y, gridZ);
        std::optional<IngestorKernelCode> convTail;
        if(plan.splitRowTile > 0)
        {
            // The tail split: the leading row tiles unsplit, then the rest split (convTail).
            conv.setGridSize(static_cast<unsigned int>(plan.splitRowTile), grid->y, grid->z);
            convTail.emplace(buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, kernel, options, convSignature(fused)));
            convTail->setBlockSize(threads, 1, 1);
            convTail->setGridSize(
                static_cast<unsigned int>(rowTiles - plan.splitRowTile), grid->y, gridZ);
        }
        else if(rowTiles % launchTiles != 0)
        {
            convTail.emplace(buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, kernel, options, convSignature(fused)));
            convTail->setBlockSize(threads, 1, 1);
            convTail->setGridSize(
                static_cast<unsigned int>(rowTiles % launchTiles), grid->y, gridZ);
        }
        std::optional<IngestorKernelCode> reduce;
        if(tile->patch == 3)
        {
            // Col2im: ConvIgemmCol2im in the reduce's place, four channels of a dx position a
            // thread.
            const auto& b = problem->base.args;
            KernelDefinition col2imKernel = kernel;
            col2imKernel.source.entryPoint = std::string(COL2IM_ENTRY_POINT);
            reduce.emplace(buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, col2imKernel, options, col2imSignature()));
            reduce->setBlockSize(COPY_BLOCK, 1, 1);
            reduce->setGridSize(static_cast<unsigned int>(std::clamp<long long>(
                                    divUpBlocks(b.n * b.inSpatial[0] * b.inSpatial[1]
                                                * b.inSpatial[2] * ((b.c + 3) / 4)),
                                    1,
                                    COPY_MAX_BLOCKS)),
                                1,
                                1);
        }
        else if(plan.splits > 1)
        {
            KernelDefinition reduceKernel = kernel;
            reduceKernel.source.entryPoint = std::string(REDUCE_ENTRY_POINT);
            reduce.emplace(buildIngestorKernelCode(_kernelCompiler,
                                                   _kpackLoader,
                                                   context,
                                                   reduceKernel,
                                                   options,
                                                   reduceSignature(fused)));
            reduce->setBlockSize(COPY_BLOCK, 1, 1);
            reduce->setGridSize(static_cast<unsigned int>(std::clamp<long long>(
                                    divUpBlocks(plan.partialElems), 1, COPY_MAX_BLOCKS)),
                                1,
                                1);
        }

        // The staging copies are bitwise and take no epilogue.
        conv_epilogue::removeEpilogueDefines(options);
        std::array<std::optional<IngestorKernelCode>, 3> copies;
        const std::array<const Staging*, 3> stagings{&problem->x, &problem->w, &problem->y};
        for(size_t i = 0; i < stagings.size(); ++i)
        {
            if(!stagings[i]->needed)
            {
                continue;
            }
            KernelDefinition copyKernel = kernel;
            const bool transpose = stagings[i]->transpose;
            const bool expand = stagings[i]->expand;
            std::string_view entryPoint = COPY_ENTRY_POINT;
            const std::vector<KernelArgument>* signature = &copySignature();
            if(expand)
            {
                entryPoint = EXPAND_ENTRY_POINT;
                signature = &expandSignature();
            }
            else if(transpose)
            {
                entryPoint = TRANSPOSE_ENTRY_POINT;
                signature = &transposeSignature();
            }
            copyKernel.source.entryPoint = std::string(entryPoint);
            auto copy = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, copyKernel, options, *signature);
            long long blocks = 0;
            if(transpose)
            {
                // One block per tile of each image.
                const auto& t = stagings[i]->transposeArgs;
                constexpr long long TILE = CONV_IGEMM_TRANSPOSE_TILE;
                blocks = (t.cols + TILE - 1) / TILE * ((t.rows + TILE - 1) / TILE) * t.batches;
                if(blocks > MAX_DISPATCH_ITEMS / COPY_BLOCK)
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                        "conv_igemm staging transpose exceeds the launch grid");
                }
            }
            else
            {
                blocks = std::clamp<long long>(
                    (stagings[i]->copy.total + COPY_BLOCK - 1) / COPY_BLOCK, 1, COPY_MAX_BLOCKS);
            }
            copy.setBlockSize(COPY_BLOCK, 1, 1);
            copy.setGridSize(static_cast<unsigned int>(blocks), 1, 1);
            copies[i].emplace(std::move(copy));
        }
        return std::make_unique<PreparedConvIgemm>(std::move(conv),
                                                   std::move(convTail),
                                                   launchTiles,
                                                   rowTiles,
                                                   std::move(reduce),
                                                   std::move(copies),
                                                   *problem,
                                                   plan);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvIgemm&>(prepared);
        const auto& problem = dispatch.problem();
        const auto& plan = dispatch.plan();
        if(plan.workspaceBytes > 0 && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_igemm needs its staging workspace");
        }
        auto* x = hipdnn_plugin_sdk::findDeviceBuffer(problem.xUid, deviceBuffers, numDeviceBuffers)
                      .ptr;
        auto* w = hipdnn_plugin_sdk::findDeviceBuffer(problem.wUid, deviceBuffers, numDeviceBuffers)
                      .ptr;
        auto* y = hipdnn_plugin_sdk::findDeviceBuffer(problem.yUid, deviceBuffers, numDeviceBuffers)
                      .ptr;
        const auto stream = handle.getStream();
        const bool forward = problem.direction == ConvDirection::FORWARD;
        const std::array<void*, 3> graph{x, w, y};
        const std::array<const Staging*, 3> stagings{&problem.x, &problem.w, &problem.y};
        // x is read forward and written backward; w is always read; y is written forward.
        const std::array<bool, 3> read{forward, true, !forward};
        std::array<void*, 3> operand = graph;
        auto* base = static_cast<char*>(workspace);
        auto copy = [&](size_t i, bool in) {
            char* graphSide
                = static_cast<char*>(graph[i])
                  + stagings[i]->graphOffset * static_cast<int64_t>(problem.elementBytes);
            char* staged = base + stagings[i]->workspaceOffset;
            void* src = in ? graphSide : staged;
            void* dst = in ? staged : graphSide;
            if(stagings[i]->expand)
            {
                ConvIgemmExpandArgs args = stagings[i]->expandArgs;
                args.src = src;
                args.dst = dst;
                dispatch.copyForStream(i, stream).launch(stream, args);
            }
            else if(stagings[i]->transpose)
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
        K1Args args = problem.args;
        args.x = operand[0];
        args.w = operand[1];
        args.y = operand[2];
        args.splits = plan.splits;
        args.outElems = plan.outElems;
        args.partial
            = plan.splits > 1 ? reinterpret_cast<float*>(base + plan.partialOffset) : nullptr;
        const bool col2im = problem.col2imBytes > 0;
        if(col2im)
        {
            // Col2im: the GEMM's columns (one split) in the workspace, for ConvIgemmCol2im.
            args.partial = reinterpret_cast<float*>(base + problem.col2imOffset);
            args.outElems = static_cast<long long>(problem.col2imBytes / sizeof(float));
        }
        const bool fused = !problem.base.epilogue.empty();
        ConvEpiArgs epiArgs{};
        if(fused)
        {
            // The split-K reduce walks the output flat: packed channels-last unless the config
            // writes the graph's channels-first output in place.
            const auto& a = problem.args;
            const bool channelsFirstOutput
                = !problem.y.needed && problem.base.args.yStrides[1] != 1 && a.k > 1;
            const long long pixels = static_cast<long long>(a.dout) * a.ho * a.wo;
            epiArgs = conv_epilogue::epilogueArgs(problem.base.epilogue,
                                                  deviceBuffers,
                                                  numDeviceBuffers,
                                                  channelsFirstOutput ? pixels : 1,
                                                  a.k);
        }
        const auto launchConv = [&](compilation::IRunnableKernel& conv) {
            if(fused)
            {
                conv.launch(stream, args, epiArgs);
            }
            else
            {
                conv.launch(stream, args);
            }
        };
        args.partialTile = 0;
        if(plan.splitRowTile > 0)
        {
            // The tail split: the leading row tiles unsplit, then the rest split into partials
            // of their rows only.
            args.splits = 1;
            args.tileBase = 0;
            launchConv(dispatch.convForStream(stream));
            args.splits = plan.splits;
            args.tileBase = static_cast<int>(plan.splitRowTile);
            args.partialTile = static_cast<int>(plan.splitRowTile);
            args.outElems = plan.partialElems;
            launchConv(dispatch.convTailForStream(stream));
        }
        else
        {
            for(long long first = 0; first < dispatch.rowTiles(); first += dispatch.launchTiles())
            {
                args.tileBase = static_cast<int>(first);
                const bool whole = dispatch.rowTiles() - first >= dispatch.launchTiles();
                launchConv(whole ? dispatch.convForStream(stream)
                                 : dispatch.convTailForStream(stream));
            }
        }
        if(col2im)
        {
            const auto& b = problem.base.args;
            K1Col2imArgs c2{};
            c2.partial = args.partial;
            c2.dx = operand[0];
            c2.splitElems = args.outElems;
            c2.splits = 1;
            c2.n = static_cast<int>(b.n);
            c2.c = static_cast<int>(b.c);
            c2.di = static_cast<int>(b.inSpatial[0]);
            c2.hi = static_cast<int>(b.inSpatial[1]);
            c2.wi = static_cast<int>(b.inSpatial[2]);
            c2.dout = static_cast<int>(b.outSpatial[0]);
            c2.ho = static_cast<int>(b.outSpatial[1]);
            c2.wo = static_cast<int>(b.outSpatial[2]);
            c2.ft = static_cast<int>(b.filter[0]);
            c2.fr = static_cast<int>(b.filter[1]);
            c2.fs = static_cast<int>(b.filter[2]);
            c2.sd = static_cast<int>(b.stride[0]);
            c2.sh = static_cast<int>(b.stride[1]);
            c2.sw = static_cast<int>(b.stride[2]);
            c2.pd = static_cast<int>(b.prePadding[0]);
            c2.ph = static_cast<int>(b.prePadding[1]);
            c2.pw = static_cast<int>(b.prePadding[2]);
            c2.dd = static_cast<int>(b.dilation[0]);
            c2.dh = static_cast<int>(b.dilation[1]);
            c2.dw = static_cast<int>(b.dilation[2]);
            c2.tapMajor = problem.col2imTapMajor ? 1 : 0;
            dispatch.reduceForStream(stream).launch(stream, c2);
        }
        else if(plan.splits > 1)
        {
            // The split rows' output: past the unsplit rows' elements.
            char* out = static_cast<char*>(forward ? operand[2] : operand[0])
                        + (plan.outElems - plan.partialElems)
                              * static_cast<long long>(problem.elementBytes);
            K1ReduceArgs reduce{args.partial, out, plan.partialElems, plan.splits, 0};
            if(fused)
            {
                dispatch.reduceForStream(stream).launch(stream, reduce, epiArgs);
            }
            else
            {
                dispatch.reduceForStream(stream).launch(stream, reduce);
            }
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
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convIgemmKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvIgemmDispatchHandler& convIgemmDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convIgemmKpackModuleCache());
    static const ConvIgemmDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvIgemmSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convIgemmDispatchHandler());
}

void resetConvIgemmModuleCache()
{
    convIgemmKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
