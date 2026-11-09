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

constexpr std::array<AtomEntry, 21> ATOMS{{
    {{"mfma16_f16", "K1AtomMfmaF16", 64, false}, "f16", CDNA},
    {{"mfma32_f16", "K1AtomMfma32F16", 64, false}, "f16", CDNA},
    {{"mfma16_bf16", "K1AtomMfmaBf16", 64, false}, "bf16", CDNA_BF16_1K},
    {{"mfma32_bf16", "K1AtomMfma32Bf16", 64, false}, "bf16", CDNA_BF16_1K},
    {{"mfma16_bf16_908", "K1AtomMfmaBf16Gfx908", 64, false}, "bf16", GFX908},
    {{"mfma32_bf16_908", "K1AtomMfma32Bf16Gfx908", 64, false}, "bf16", GFX908},
    {{"mfma16_f32", "K1AtomMfmaF32", 64, false}, "f32", CDNA},
    {{"mfma32_f32", "K1AtomMfma32F32", 64, false}, "f32", CDNA},
    {{"mfma16_xf32", "K1AtomMfmaXf32", 64, true}, "f32", GFX942},
    {{"mfma32_xf32", "K1AtomMfma32Xf32", 64, true}, "f32", GFX942},
    {{"mfma16x32_f16", "K1AtomMfmaF16X32", 64, false}, "f16", GFX950},
    {{"mfma16x32_bf16", "K1AtomMfmaBf16X32", 64, false}, "bf16", GFX950},
    {{"mfma32x16_f16", "K1AtomMfma32F16X16", 64, false}, "f16", GFX950},
    {{"mfma32x16_bf16", "K1AtomMfma32Bf16X16", 64, false}, "bf16", GFX950},
    {{"wmma11_f16", "K1AtomWmma11F16", 32, false}, "f16", GFX11},
    {{"wmma11_bf16", "K1AtomWmma11Bf16", 32, false}, "bf16", GFX11},
    {{"wmma12_f16", "K1AtomWmma12F16", 32, false}, "f16", GFX12},
    {{"wmma12_bf16", "K1AtomWmma12Bf16", 32, false}, "bf16", GFX12},
    {{"fma", "", 0, false}, "any", {}},
    // Unused slots keep the table's size fixed while atoms are added.
    {{"", "", 0, false}, "", {}},
    {{"", "", 0, false}, "", {}},
}};

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

bool tf32Permitted()
{
    const auto raw = hipdnn_data_sdk::utilities::toLower(hipdnn_data_sdk::utilities::trim(
        hipdnn_data_sdk::utilities::getEnv("HIP_KERNEL_PROVIDER_CONV_TF32")));
    return raw == "1" || raw == "true" || raw == "on" || raw == "yes";
}

std::optional<IgemmProblem> parseIgemmProblem(const MatchContext& context, ConvDirection direction)
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
    const long long cg = a.c / a.groups, kg = a.k / a.groups;
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
    const long long t = a.filter[0], r = a.filter[1], s = a.filter[2];
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
    size_t offset = 0;
    for(auto* staging : {&problem.x, &problem.w, &problem.y})
    {
        if(staging->needed)
        {
            staging->workspaceOffset = offset;
            offset = alignUp(offset + staging->bytes);
        }
    }
    problem.workspaceBytes = offset;

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
    problem.base = std::move(*base);
    return problem;
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
    long long lo = 0, hi = 0, pixelStride = 1;
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

SplitPlan splitPlanFor(
    const IgemmProblem& problem, int64_t bm, int64_t bn, int64_t bk, int64_t computeUnits)
{
    constexpr long long BLOCKS_PER_CU = 2;
    constexpr long long MAX_SPLITS = 16;
    constexpr long long MAX_PARTIAL_BYTES = 1LL << 30;
    const auto& a = problem.args;
    const bool forward = problem.direction == ConvDirection::FORWARD;
    SplitPlan plan;
    plan.workspaceBytes = problem.workspaceBytes;
    plan.outElems = forward ? a.m * a.k : static_cast<long long>(a.n) * a.di * a.hi * a.wi * a.c;
    const auto grid = gridFor(problem, bm, bn);
    if(!grid.has_value())
    {
        return plan;
    }
    const long long tiles = static_cast<long long>(grid->x) * grid->y * grid->z;
    const long long reduction = forward ? a.c / a.groups : a.k / a.groups;
    const long long taps = static_cast<long long>(a.ft) * a.fr * a.fs;
    const long long iterations
        = std::max<long long>(1, divUp(taps, problem.phases)) * divUp(reduction, bk);
    const long long wanted = std::max<int64_t>(1, computeUnits) * BLOCKS_PER_CU;
    long long splits
        = tiles >= wanted ? 1 : std::min({divUp(wanted, tiles), iterations / 2, MAX_SPLITS});
    while(splits > 1
          && (splits * plan.outElems * static_cast<long long>(sizeof(float)) > MAX_PARTIAL_BYTES
              || grid->z * splits > MAX_GRID_YZ))
    {
        --splits;
    }
    if(splits > 1)
    {
        plan.splits = static_cast<int>(splits);
        plan.partialOffset = alignUp(problem.workspaceBytes);
        plan.workspaceBytes
            = plan.partialOffset + static_cast<size_t>(splits * plan.outElems) * sizeof(float);
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
constexpr std::string_view REDUCE_ENTRY_POINT = "ConvIgemmReduce";
constexpr unsigned int COPY_BLOCK = 256;
/// Bytes a 32-bit buffer offset reaches.
constexpr long long BUFFER_LIMIT = std::numeric_limits<int>::max();
constexpr long long COPY_MAX_BLOCKS = 1LL << 20;

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
    const auto problem = conv_igemm::parseIgemmProblem(context, direction);
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

    int64_t blockThreads() const
    {
        return info.wave == 0 ? threads : wm * wn * info.wave;
    }
};

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
    if(tile.bm < 1 || tile.bn < 1 || tile.bk < 1 || tile.vec < 1 || tile.blockThreads() < 1
       || (tile.xcd != 0 && tile.xcd != 1) || (tile.halo != 0 && tile.halo != 1)
       || (tile.halo != 0 && (tile.info.wave == 0 || tile.stages != 1)))
    {
        return std::nullopt;
    }
    return tile;
}

/// Kernel-scoped applicability: the config runs on this device and computes this problem
/// correctly. Its atom must exist on the device and take the graph's element type (fp32 through
/// rounded operands -- xf32, three-bf16 -- only with TF32 permission), and its channel vector
/// must divide the reduction channels. An XCD-order config is served only where the device has
/// several XCDs (elsewhere it is its plain twin), a halo config only where the problem fits
/// the halo loop.
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
    const auto problem = conv_igemm::parseIgemmProblem(context, *direction);
    if(!problem.has_value()
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->base.dataType)
       || !conv_igemm::atomRunsOn(tile->atom, context.deviceProperties.gcnArchName))
    {
        return false;
    }
    const auto element = conv_igemm::atomElement(tile->atom);
    const auto wanted = conv_igemm::dataElement(problem->base.dataType);
    if(tile->split)
    {
        if(wanted != "f32" || element != "bf16" || !conv_igemm::tf32Permitted())
        {
            return false;
        }
    }
    else if(element != "any"
            && (element != wanted || (tile->info.tf32 && !conv_igemm::tf32Permitted())))
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
    if(tile->xcd != 0 && !conv_igemm::hasXcds(context.deviceProperties.gcnArchName))
    {
        return false;
    }
    if(tile->halo != 0 && !conv_igemm::haloFits(*problem, tile->bm))
    {
        return false;
    }
    return conv_igemm::gridFor(*problem, tile->bm, tile->bn).has_value();
}

/// The single argument both kernels take. Only consulted for a pre-built kernel.
const std::vector<KernelArgument>& convSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K1Args)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& reduceSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K1ReduceArgs)), 0, ""}};
    return s_signature;
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
                      std::optional<IngestorKernelCode> reduce,
                      std::array<std::optional<IngestorKernelCode>, 3> copies,
                      const IgemmProblem& problem,
                      const conv_igemm::SplitPlan& plan)
        : _conv(std::move(conv))
        , _reduce(std::move(reduce))
        , _copies(std::move(copies))
        , _problem(problem)
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
        const auto problem = conv_igemm::parseIgemmProblem(context, *direction);
        const auto tile = tileOf(kernel);
        if(!problem.has_value() || !tile.has_value())
        {
            return 0;
        }
        return conv_igemm::splitPlanFor(*problem,
                                        tile->bm,
                                        tile->bn,
                                        tile->bk,
                                        context.deviceProperties.multiProcessorCount)
            .workspaceBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto direction = directionFor(kernel);
        const auto tile = tileOf(kernel);
        std::optional<IgemmProblem> problem;
        if(direction.has_value())
        {
            problem = conv_igemm::parseIgemmProblem(context, *direction);
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
        options.add("CONV_IGEMM_DIR",
                    std::string(*direction == ConvDirection::FORWARD ? "0" : "1"));
        // 32-bit buffer loads while a block's rows and the filter stay under 2^31 bytes.
        const long long filterBytes
            = static_cast<long long>(problem->args.k) * problem->args.ft * problem->args.fr
              * problem->args.fs * (problem->args.c / problem->args.groups) * problem->elementBytes;
        const bool buffer = problem->blockSpanBytes < BUFFER_LIMIT && filterBytes < BUFFER_LIMIT;
        options.add("K1_BUFFER", std::string(buffer ? "1" : "0"));

        const auto plan = conv_igemm::splitPlanFor(
            *problem, tile->bm, tile->bn, tile->bk, context.deviceProperties.multiProcessorCount);
        auto conv = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, convSignature());
        conv.setBlockSize(static_cast<unsigned int>(tile->blockThreads()), 1, 1);
        conv.setGridSize(grid->x, grid->y, grid->z * static_cast<unsigned int>(plan.splits));
        std::optional<IngestorKernelCode> reduce;
        if(plan.splits > 1)
        {
            KernelDefinition reduceKernel = kernel;
            reduceKernel.source.entryPoint = std::string(REDUCE_ENTRY_POINT);
            reduce.emplace(buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, reduceKernel, options, reduceSignature()));
            reduce->setBlockSize(COPY_BLOCK, 1, 1);
            reduce->setGridSize(static_cast<unsigned int>(std::clamp<long long>(
                                    divUpBlocks(plan.outElems), 1, COPY_MAX_BLOCKS)),
                                1,
                                1);
        }

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
            copyKernel.source.entryPoint
                = std::string(transpose ? TRANSPOSE_ENTRY_POINT : COPY_ENTRY_POINT);
            auto copy = buildIngestorKernelCode(_kernelCompiler,
                                                _kpackLoader,
                                                context,
                                                copyKernel,
                                                options,
                                                transpose ? transposeSignature() : copySignature());
            long long blocks = 0;
            if(transpose)
            {
                // One block per 32 x 32 tile of each image.
                const auto& t = stagings[i]->transposeArgs;
                blocks = (t.cols + 31) / 32 * ((t.rows + 31) / 32) * t.batches;
                if(blocks > std::numeric_limits<int>::max())
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
        return std::make_unique<PreparedConvIgemm>(
            std::move(conv), std::move(reduce), std::move(copies), *problem, plan);
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
        K1Args args = problem.args;
        args.x = operand[0];
        args.w = operand[1];
        args.y = operand[2];
        args.splits = plan.splits;
        args.outElems = plan.outElems;
        args.partial
            = plan.splits > 1 ? reinterpret_cast<float*>(base + plan.partialOffset) : nullptr;
        dispatch.convForStream(stream).launch(stream, args);
        if(plan.splits > 1)
        {
            K1ReduceArgs reduce{
                args.partial, forward ? operand[2] : operand[0], plan.outElems, plan.splits, 0};
            dispatch.reduceForStream(stream).launch(stream, reduce);
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
