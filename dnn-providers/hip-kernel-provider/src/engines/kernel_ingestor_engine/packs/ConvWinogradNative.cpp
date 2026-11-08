// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
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
#include "engines/kernel_ingestor_engine/descriptors/conv/winograd/kernels/ConvWinogradArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvWinogradProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvWinogradNative.cpp
 * @brief The native half of the Winograd convolution engines (CONV_WINOGRAD_FWD,
 *        CONV_WINOGRAD_BWD): graph matching, kernel matching, workspace sizing and dispatch.
 *
 * Winograd F(2,3) (every dtype) and F(4,3) (fp32) for 3x3, stride-1, dilation-1 2-D
 * convolution, forward and backward data (the filter rotated and its K/C roles swapped in the
 * filter pre-pass), both convolution modes (a reflected filter, in the same pre-pass), groups,
 * and any padding. Three variants, each one HIPRTC source with runtime shapes:
 * - multipass: filter transform, input transform, a batched GEMM per Winograd position, output
 *   transform; NCHW and NHWC read natively; the batch is sliced to keep the transformed-input
 *   and product buffers under the engine's workspace limit (or, as a separate config, inside the
 *   last-level cache of an RDNA device);
 * - fused VALU: one persistent kernel, all architectures (NCHW);
 * - fused WMMA: one persistent kernel on gfx11 / gfx12, fp16 and bf16 (NCHW).
 * Operands in a layout the variant does not read are staged through the workspace. All
 * transform arithmetic and accumulation are fp32; 16-bit operands are rounded once after their
 * transform. An output whose 3x3 window lies entirely in the padding is exactly 0. Every config
 * is admitted on correctness alone; autotune ranks them. Winograd mixes neighbouring inputs (an
 * inf input poisons its tile), hence the engines' WINOGRAD numerical note.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

namespace conv_winograd
{

namespace
{

constexpr long long INT_LIMIT = std::numeric_limits<int>::max();
constexpr long long GRID_YZ_LIMIT = 65535;
constexpr size_t WORKSPACE_ALIGNMENT = 256;
/// Bytes the fused kernels' 32-bit buffer offsets reach: K5_OOB in the kernels.
constexpr long long FUSED_BUFFER_LIMIT = 1LL << 30;
constexpr long long DEFAULT_WORKSPACE_MB = 2048;
constexpr long long FALLBACK_COMPUTE_UNITS = 16;
/// Persistent workgroups per compute unit for the fused kernels.
constexpr long long PERSISTENT_PER_CU = 2;

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

/// True when @p strides (in n, c, d, h, w slots) are packed in @p order (slowest axis first,
/// as slot indices) for @p extents (in the same slots); extent-1 axes may carry any stride.
bool packedIn(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
              const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents,
              const std::array<int, 4>& order)
{
    long long expected = 1;
    for(size_t i = order.size(); i-- > 0;)
    {
        const auto axis = static_cast<size_t>(order[i]);
        if(extents[axis] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= extents[axis];
    }
    return true;
}

constexpr std::array<int, 4> NCHW_ORDER{0, 1, 3, 4};
constexpr std::array<int, 4> NHWC_ORDER{0, 3, 4, 1};

Layout layoutOf(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents)
{
    if(packedIn(strides, extents, NCHW_ORDER))
    {
        return Layout::NCHW;
    }
    if(packedIn(strides, extents, NHWC_ORDER))
    {
        return Layout::NHWC;
    }
    return Layout::OTHER;
}

long long envLongLong(const char* name, long long fallback)
{
    const auto raw = hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv(name));
    if(raw.empty())
    {
        return fallback;
    }
    char* end = nullptr;
    const long long value = std::strtoll(raw.c_str(), &end, 10);
    return end != nullptr && *end == '\0' && value > 0 ? value : fallback;
}

/// A rank-4 copy over @p dims (destination order) with the given element strides.
K5CopyArgs copyOf(const std::array<long long, 4>& dims,
                  const std::array<long long, 4>& srcStrides,
                  const std::array<long long, 4>& dstStrides,
                  int elementBytes)
{
    K5CopyArgs copy{};
    copy.rank = 4;
    copy.total = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        copy.dims[i] = dims[i];
        copy.srcStrides[i] = srcStrides[i];
        copy.dstStrides[i] = dstStrides[i];
        copy.total *= dims[i];
    }
    copy.elementBytes = elementBytes;
    return copy;
}

std::array<long long, 4> packedStrides(const std::array<long long, 4>& dims)
{
    return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

/// An activation (n, c, h, w) between the graph's @p strides (n, c, d, h, w slots) and a packed
/// @p layout copy; @p in copies graph -> packed, else packed -> graph.
K5CopyArgs activationCopy(const long long* strides,
                          long long n,
                          long long c,
                          long long h,
                          long long w,
                          Layout layout,
                          bool in,
                          int elementBytes)
{
    // Packed strides of the staged copy, in (n, c, h, w) roles.
    const std::array<long long, 4> packed = layout == Layout::NCHW
                                                ? std::array<long long, 4>{c * h * w, h * w, w, 1}
                                                : std::array<long long, 4>{h * w * c, 1, w * c, c};
    const std::array<long long, 4> graph{strides[0], strides[1], strides[3], strides[4]};
    if(in)
    {
        // Destination order, so the packed side is written contiguously.
        if(layout == Layout::NCHW)
        {
            return copyOf({n, c, h, w}, graph, packed, elementBytes);
        }
        return copyOf({n, h, w, c},
                      {graph[0], graph[2], graph[3], graph[1]},
                      {packed[0], packed[2], packed[3], packed[1]},
                      elementBytes);
    }
    return copyOf({n, c, h, w}, packed, graph, elementBytes);
}

} // namespace

size_t workspaceLimit()
{
    return static_cast<size_t>(
               envLongLong("HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB", DEFAULT_WORKSPACE_MB))
           << 20;
}

size_t llcBytes(std::string_view gcnArchName)
{
    const long long override = envLongLong("HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES", 0);
    if(override > 0)
    {
        return static_cast<size_t>(override);
    }
    // MALL ("Infinity Cache") sizes; HIP does not report them.
    struct Entry
    {
        std::string_view arch;
        size_t mib;
    };
    static constexpr std::array<Entry, 6> TABLE{{{"gfx1100", 96},
                                                 {"gfx1101", 64},
                                                 {"gfx1102", 32},
                                                 {"gfx1151", 32},
                                                 {"gfx1200", 32},
                                                 {"gfx1201", 64}}};
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    for(const auto& entry : TABLE)
    {
        if(entry.arch == arch)
        {
            return entry.mib << 20;
        }
    }
    return 0;
}

std::optional<WinogradProblem> parseWinogradProblem(const MatchContext& context,
                                                    ConvDirection direction)
{
    if(direction != ConvDirection::FORWARD && direction != ConvDirection::BACKWARD_DATA)
    {
        return std::nullopt;
    }
    auto base = conv_naive::parseConvNaiveProblem(context, direction);
    if(!base.has_value() || base->spatialRank != 2)
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
    // 3x3, stride 1, dilation 1 on both spatial axes (slot 0 is the padded depth axis).
    for(int axis = 1; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        if(a.filter[axis] != 3 || a.stride[axis] != 1 || a.dilation[axis] != 1)
        {
            return std::nullopt;
        }
    }
    for(const long long value : {a.n, a.c, a.k, a.groups})
    {
        if(value < 1 || value > INT_LIMIT)
        {
            return std::nullopt;
        }
    }
    for(int axis = 1; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        // Backward data pads by 2 - pre in the forward form.
        if(!fitsInt(a.inSpatial[axis]) || !fitsInt(a.outSpatial[axis])
           || !fitsInt(a.prePadding[axis]) || !fitsInt(2 - a.prePadding[axis]))
        {
            return std::nullopt;
        }
    }

    WinogradProblem problem;
    problem.direction = direction;
    problem.elementBytes = elementBytes;
    problem.xUid = base->operands.x;
    problem.wUid = base->operands.w;
    problem.yUid = base->operands.y;
    const bool forward = direction == ConvDirection::FORWARD;
    auto& k5 = problem.args;
    k5.n = static_cast<int>(a.n);
    k5.groups = static_cast<int>(a.groups);
    k5.ci = static_cast<int>(forward ? a.c : a.k);
    k5.co = static_cast<int>(forward ? a.k : a.c);
    const long long* inExtent = forward ? a.inSpatial : a.outSpatial;
    const long long* outExtent = forward ? a.outSpatial : a.inSpatial;
    k5.hi = static_cast<int>(inExtent[1]);
    k5.wi = static_cast<int>(inExtent[2]);
    k5.ho = static_cast<int>(outExtent[1]);
    k5.wo = static_cast<int>(outExtent[2]);
    k5.ph = static_cast<int>(forward ? a.prePadding[1] : 2 - a.prePadding[1]);
    k5.pw = static_cast<int>(forward ? a.prePadding[2] : 2 - a.prePadding[2]);
    k5.flip = a.flipFilter;

    const Layout x = layoutOf(a.xStrides, {a.n, a.c, 1, a.inSpatial[1], a.inSpatial[2]});
    const Layout y = layoutOf(a.yStrides, {a.n, a.k, 1, a.outSpatial[1], a.outSpatial[2]});
    problem.activations = x == y ? x : Layout::OTHER;
    // Filter (k, c/g, r, s): KCRS packed is the channels-first order, KRSC the channels-last.
    problem.filter = layoutOf(a.wStrides, {a.k, a.c / a.groups, 1, 3, 3});
    problem.base = std::move(*base);
    return problem;
}

std::optional<Config> configOf(const KernelDefinition& kernel)
{
    try
    {
        Config config;
        const auto& variant = kernel.getStringMetadata("variant");
        if(variant == "multipass")
        {
            config.variant = Variant::MULTIPASS;
        }
        else if(variant == "fused")
        {
            config.variant = Variant::FUSED;
        }
        else if(variant == "wmma")
        {
            config.variant = Variant::WMMA;
        }
        else
        {
            return std::nullopt;
        }
        const auto& layout = kernel.getStringMetadata("layout");
        if(layout != "NCHW" && layout != "NHWC")
        {
            return std::nullopt;
        }
        config.layout = layout == "NCHW" ? Layout::NCHW : Layout::NHWC;
        config.arch = kernel.getStringMetadata("arch");
        const auto intField = [&kernel](const char* name) {
            const int64_t value = kernel.getIntMetadata(name);
            return value < 0 || value > 4096 ? -1 : static_cast<int>(value);
        };
        config.m = intField("m");
        config.gemmBm = intField("gemm_bm");
        config.gemmBn = intField("gemm_bn");
        config.gemmBk = intField("gemm_bk");
        config.bk = intField("bk");
        config.bt = intField("bt");
        config.ccv = intField("ccv");
        config.tm = intField("tm");
        config.tn = intField("tn");
        config.waves = intField("waves");
        const int llc = intField("llc");
        if(llc != 0 && llc != 1)
        {
            return std::nullopt;
        }
        config.llcPlan = llc == 1;
        switch(config.variant)
        {
        case Variant::MULTIPASS:
            // The GEMM's 16 x 16 threads own runs of 4 rows and columns; its tile loads split
            // evenly over 256 threads.
            if((config.m != 2 && config.m != 4) || config.gemmBm < 64 || config.gemmBm % 64 != 0
               || config.gemmBn < 64 || config.gemmBn % 64 != 0 || config.gemmBk < 1
               || (config.gemmBm * config.gemmBk) % 256 != 0
               || (config.gemmBn * config.gemmBk) % 256 != 0 || config.arch != "any")
            {
                return std::nullopt;
            }
            break;
        case Variant::FUSED:
        {
            if(config.m != 2 || config.layout != Layout::NCHW || config.llcPlan
               || config.arch != "any" || config.tm < 4 || config.tn < 4 || config.tm % 4 != 0
               || config.tn % 4 != 0 || config.bk < config.tm || config.bt < config.tn
               || config.bk % config.tm != 0 || config.bt % config.tn != 0 || config.ccv < 1)
            {
                return std::nullopt;
            }
            const int lanesPerBin = (config.bk / config.tm) * (config.bt / config.tn);
            if(lanesPerBin < 1 || 64 % lanesPerBin != 0 || 16 % (64 / lanesPerBin) != 0)
            {
                return std::nullopt;
            }
            break;
        }
        case Variant::WMMA:
            if(config.m != 2 || config.layout != Layout::NCHW || config.llcPlan
               || (config.arch != "gfx11" && config.arch != "gfx12")
               || (config.bk != 32 && config.bk != 64) || (config.waves != 4 && config.waves != 8))
            {
                return std::nullopt;
            }
            break;
        default:
            return std::nullopt;
        }
        return config;
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
}

bool configRunsOn(const Config& config, std::string_view gcnArchName)
{
    if(config.variant != Variant::WMMA)
    {
        return true;
    }
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    return arch.size() > 5 && arch.substr(0, 5) == config.arch;
}

std::optional<Plan> planFor(const WinogradProblem& problem,
                            const Config& config,
                            const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    const auto& a = problem.args;
    const auto& base = problem.base.args;
    const bool forward = problem.direction == ConvDirection::FORWARD;
    const bool multipass = config.variant == Variant::MULTIPASS;
    const Layout kernelLayout = multipass ? config.layout : Layout::NCHW;
    // A multipass config reads its own layout natively; anything but the other packed layout
    // is staged to channels-first. The fused kernels read channels-first and stage the rest.
    if(multipass && problem.activations != kernelLayout
       && !(problem.activations == Layout::OTHER && kernelLayout == Layout::NCHW))
    {
        return std::nullopt;
    }
    const long long eb = problem.elementBytes;
    const long long n = a.n, groups = a.groups, cgi = a.ci / groups, cgo = a.co / groups;
    const long long m = config.m, positions = (m + 2) * (m + 2);
    const long long th = divUp(a.ho, m), tw = divUp(a.wo, m), tilesPerImage = th * tw;
    const long long inImage = static_cast<long long>(a.ci) * a.hi * a.wi;
    const long long outImage = static_cast<long long>(a.co) * a.ho * a.wo;
    if(tilesPerImage < 1 || tilesPerImage > INT_LIMIT)
    {
        return std::nullopt;
    }

    Plan plan;
    plan.args = a;
    plan.args.th = static_cast<int>(th);
    plan.args.tw = static_cast<int>(tw);
    size_t offset = 0;
    const auto place = [&offset](Staging& staging) {
        staging.workspaceOffset = offset;
        offset = alignUp(offset + staging.bytes);
    };
    if(problem.activations != kernelLayout)
    {
        const long long* inStrides = forward ? base.xStrides : base.yStrides;
        const long long* outStrides = forward ? base.yStrides : base.xStrides;
        plan.in.needed = true;
        plan.in.copy = activationCopy(
            inStrides, n, a.ci, a.hi, a.wi, kernelLayout, true, problem.elementBytes);
        plan.in.bytes = static_cast<size_t>(n * inImage * eb);
        place(plan.in);
        plan.out.needed = true;
        plan.out.copy = activationCopy(
            outStrides, n, a.co, a.ho, a.wo, kernelLayout, false, problem.elementBytes);
        plan.out.bytes = static_cast<size_t>(n * outImage * eb);
        place(plan.out);
    }
    if(problem.filter != kernelLayout)
    {
        // The graph's filter (k, c/g, r, s) to packed KCRS or KRSC.
        const long long k = base.k, cg = base.c / base.groups;
        const std::array<long long, 4> graph{
            base.wStrides[0], base.wStrides[1], base.wStrides[3], base.wStrides[4]};
        plan.filter.needed = true;
        if(kernelLayout == Layout::NCHW)
        {
            plan.filter.copy
                = copyOf({k, cg, 3, 3}, graph, packedStrides({k, cg, 3, 3}), problem.elementBytes);
        }
        else
        {
            plan.filter.copy = copyOf({k, 3, 3, cg},
                                      {graph[0], graph[2], graph[3], graph[1]},
                                      packedStrides({k, 3, 3, cg}),
                                      problem.elementBytes);
        }
        plan.filter.bytes = static_cast<size_t>(k * cg * 9 * eb);
        place(plan.filter);
    }

    long long uBytes = 0;
    switch(config.variant)
    {
    case Variant::MULTIPASS:
        uBytes = groups * positions * cgo * cgi * eb;
        break;
    case Variant::FUSED:
    {
        const long long vw = problem.base.dataType == data_objects::DataType::HALF ? 2 : 1;
        uBytes = groups * 16 * divUp(cgi, vw) * cgo * 4;
        break;
    }
    case Variant::WMMA:
        uBytes = groups * 16 * cgo * divUp(cgi, 16) * 16 * eb;
        break;
    default:
        return std::nullopt;
    }
    plan.uOffset = offset;
    plan.uBytes = static_cast<size_t>(uBytes);
    offset = alignUp(offset + plan.uBytes);

    long long slice = n;
    if(multipass)
    {
        const long long v1 = groups * positions * cgi * tilesPerImage * eb;
        const long long m1 = groups * positions * cgo * tilesPerImage * 4;
        const size_t cap = config.llcPlan ? llcBytes(device.gcnArchName) / 4 * 3 : workspaceLimit();
        if(config.llcPlan && cap == 0)
        {
            return std::nullopt;
        }
        const size_t fixed = config.llcPlan ? plan.uBytes : offset;
        const long long room = cap > fixed ? static_cast<long long>(cap - fixed) : 0;
        slice = std::clamp<long long>(room / (v1 + m1), 1, n);
        // The LLC plan is only a distinct config where it slices.
        if(config.llcPlan && slice >= n)
        {
            return std::nullopt;
        }
        plan.vOffset = offset;
        plan.vBytes = static_cast<size_t>(v1 * slice);
        offset = alignUp(offset + plan.vBytes);
        plan.mOffset = offset;
        plan.mBytes = static_cast<size_t>(m1 * slice);
        offset = alignUp(offset + plan.mBytes);
    }
    else
    {
        // The fused kernels address the input and U through 32-bit buffer offsets.
        if(uBytes > FUSED_BUFFER_LIMIT || inImage * eb > FUSED_BUFFER_LIMIT)
        {
            return std::nullopt;
        }
        slice = std::min(n, FUSED_BUFFER_LIMIT / (inImage * eb));
    }
    const long long tiles = slice * tilesPerImage;
    if(tiles > INT_LIMIT)
    {
        return std::nullopt;
    }
    plan.sliceImages = static_cast<int>(slice);
    plan.slices = static_cast<int>(divUp(n, slice));
    plan.args.n = static_cast<int>(slice);
    plan.args.tiles = static_cast<int>(tiles);
    plan.workspaceBytes = offset;

    const auto grid = [](unsigned int(&blocks)[3], long long x, long long y, long long z) {
        if(x < 1 || x > INT_LIMIT || y > GRID_YZ_LIMIT || z > GRID_YZ_LIMIT)
        {
            return false;
        }
        blocks[0] = static_cast<unsigned int>(x);
        blocks[1] = static_cast<unsigned int>(std::max<long long>(1, y));
        blocks[2] = static_cast<unsigned int>(std::max<long long>(1, z));
        return true;
    };
    const long long computeUnits = device.multiProcessorCount > 0
                                       ? static_cast<long long>(device.multiProcessorCount)
                                       : FALLBACK_COMPUTE_UNITS;
    switch(config.variant)
    {
    case Variant::MULTIPASS:
    {
        // K5XformBlock: NCHW 64 tiles x 4 channels, NHWC 16 x 16.
        const long long tb = kernelLayout == Layout::NCHW ? 64 : 16;
        const long long cb = 256 / tb;
        if(!grid(plan.filterBlocks, divUp(cgi, 16), divUp(cgo, 16), groups)
           || !grid(plan.inputBlocks, divUp(tiles, tb), divUp(cgi, cb), groups)
           || !grid(plan.outputBlocks, divUp(tiles, tb), divUp(cgo, cb), groups)
           || !grid(plan.gemmBlocks,
                    divUp(tiles, config.gemmBn),
                    divUp(cgo, config.gemmBm),
                    groups * positions))
        {
            return std::nullopt;
        }
        break;
    }
    case Variant::FUSED:
    {
        const long long vw = problem.base.dataType == data_objects::DataType::HALF ? 2 : 1;
        const long long lanesPerBin = (config.bk / config.tm) * (config.bt / config.tn);
        plan.mainThreads = static_cast<unsigned int>(16 / (64 / lanesPerBin) * 64);
        const long long items = groups * divUp(cgo, config.bk) * divUp(tiles, config.bt);
        if(!grid(plan.filterBlocks, divUp(groups * divUp(cgi, vw) * cgo, 256), 1, 1))
        {
            return std::nullopt;
        }
        plan.mainBlocks = static_cast<unsigned int>(
            std::clamp<long long>(items, 1, computeUnits * PERSISTENT_PER_CU));
        break;
    }
    case Variant::WMMA:
    {
        plan.mainThreads = static_cast<unsigned int>(32 * config.waves);
        const long long items = groups * divUp(cgo, config.bk) * divUp(tiles, 32);
        if(!grid(plan.filterBlocks, divUp(groups * cgo * divUp(cgi, 16) * 16, 256), 1, 1))
        {
            return std::nullopt;
        }
        plan.mainBlocks = static_cast<unsigned int>(
            std::clamp<long long>(items, 1, computeUnits * PERSISTENT_PER_CU));
        break;
    }
    default:
        return std::nullopt;
    }
    return plan;
}

} // namespace conv_winograd

namespace
{

using conv_winograd::Config;
using conv_winograd::Plan;
using conv_winograd::Variant;
using conv_winograd::WinogradProblem;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_winograd_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_winograd_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_winograd.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_winograd.dispatch";
constexpr unsigned int COPY_BLOCK = 256;
constexpr long long COPY_MAX_BLOCKS = 1LL << 20;

constexpr std::string_view X_TOKEN = "conv_winograd.x.uid";
constexpr std::string_view W_TOKEN = "conv_winograd.w.uid";
constexpr std::string_view Y_TOKEN = "conv_winograd.y.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_winograd.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_winograd.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_winograd.flops";

std::optional<ConvDirection> directionFor(const KernelDefinition& kernel)
{
    try
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
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_winograd::parseWinogradProblem(context, direction);
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

struct Resolved
{
    WinogradProblem problem;
    Config config;
    Plan plan;
};

/// The problem, config and plan of @p kernel on @p context, or nullopt where it does not apply:
/// its direction and dtype are the graph's, its instructions exist on the device (WMMA
/// family), F(4,3) only for fp32, WMMA only for fp16 / bf16, and it has a plan.
std::optional<Resolved> resolve(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto direction = directionFor(kernel);
    const auto config = conv_winograd::configOf(kernel);
    if(!direction.has_value() || !config.has_value())
    {
        return std::nullopt;
    }
    auto problem = conv_winograd::parseWinogradProblem(context, *direction);
    if(!problem.has_value()
       || !conv_winograd::configRunsOn(*config, context.deviceProperties.gcnArchName))
    {
        return std::nullopt;
    }
    try
    {
        if(kernel.getStringMetadata("dtype")
           != data_objects::EnumNameDataType(problem->base.dataType))
        {
            return std::nullopt;
        }
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
    const bool fp32 = problem->base.dataType == data_objects::DataType::FLOAT;
    if((config->m == 4 && !fp32) || (config->variant == Variant::WMMA && fp32))
    {
        return std::nullopt;
    }
    auto plan = conv_winograd::planFor(*problem, *config, context.deviceProperties);
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return Resolved{std::move(*problem), *config, *plan};
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return resolve(context, kernel).has_value();
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
                                                       "conv_winograd: unsupported element type");
    }
}

template <typename Args>
const std::vector<KernelArgument>& signatureOf()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(Args)), 0, ""}};
    return s_signature;
}

/// The launches of one config: the filter pre-pass, the per-slice kernels, and the copies.
class PreparedConvWinograd : public IngestorPreparedDispatch
{
public:
    PreparedConvWinograd(std::vector<IngestorKernelCode> kernels,
                         std::optional<IngestorKernelCode> copy,
                         const Resolved& resolved)
        : _kernels(std::move(kernels))
        , _copy(std::move(copy))
        , _resolved(resolved)
    {
    }

    /// The main kernel: the GEMM (multipass) or the fused convolution.
    const IngestorKernelCode& kernelCode() const override
    {
        return _kernels[1];
    }

    /// 0 filter, 1 main (GEMM / fused), 2 input transform, 3 output transform.
    compilation::IRunnableKernel& kernelForStream(size_t index, hipStream_t stream) const
    {
        return _kernels[index].kernelForStream(stream);
    }

    compilation::IRunnableKernel& copyForStream(hipStream_t stream) const
    {
        return _copy->kernelForStream(stream);
    }

    const Resolved& resolved() const
    {
        return _resolved;
    }

private:
    std::vector<IngestorKernelCode> _kernels;
    std::optional<IngestorKernelCode> _copy;
    Resolved _resolved;
};

class ConvWinogradDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvWinogradDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto resolved = resolve(context, kernel);
        return resolved.has_value() ? resolved->plan.workspaceBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto resolved = resolve(context, kernel);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!resolved.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != resolved->problem.xUid || *w != resolved->problem.wUid
           || *y != resolved->problem.yUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_winograd dispatch was given a graph or config its matchers did not admit");
        }
        const auto& config = resolved->config;
        const auto& plan = resolved->plan;
        const bool multipass = config.variant == Variant::MULTIPASS;

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_WINO_T", elementTypeFor(resolved->problem.base.dataType));
        options.add("CONV_WINO_VARIANT",
                    std::to_string(multipass ? 0 : (config.variant == Variant::FUSED ? 1 : 2)));
        options.add("CONV_WINO_DIR",
                    std::string(resolved->problem.direction == ConvDirection::FORWARD ? "0" : "1"));
        options.add(
            "CONV_WINO_LAYOUT",
            std::string(multipass && config.layout == conv_winograd::Layout::NHWC ? "1" : "0"));
        options.add("CONV_WINO_M", std::to_string(config.m));
        options.add("CONV_WINO_GEMM_BM", std::to_string(config.gemmBm));
        options.add("CONV_WINO_GEMM_BN", std::to_string(config.gemmBn));
        options.add("CONV_WINO_GEMM_BK", std::to_string(config.gemmBk));
        options.add("CONV_WINO_BK", std::to_string(config.bk));
        options.add("CONV_WINO_BT", std::to_string(config.bt));
        options.add("CONV_WINO_CCV", std::to_string(config.ccv));
        options.add("CONV_WINO_TM", std::to_string(config.tm));
        options.add("CONV_WINO_TN", std::to_string(config.tn));
        options.add("CONV_WINO_WAVES", std::to_string(config.waves));
        options.add("CONV_WINO_GFX12", std::string(config.arch == "gfx12" ? "1" : "0"));

        const auto build = [&](std::string_view entryPoint,
                               const std::vector<KernelArgument>& signature,
                               unsigned int threads,
                               const unsigned int(&blocks)[3]) {
            KernelDefinition definition = kernel;
            definition.source.entryPoint = std::string(entryPoint);
            auto code = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, definition, options, signature);
            code.setBlockSize(threads, 1, 1);
            code.setGridSize(blocks[0], blocks[1], blocks[2]);
            return code;
        };
        std::vector<IngestorKernelCode> kernels;
        const unsigned int mainGrid[3] = {plan.mainBlocks, 1, 1};
        switch(config.variant)
        {
        case Variant::MULTIPASS:
            kernels.push_back(
                build("ConvWinogradFilter", signatureOf<K5Args>(), 256, plan.filterBlocks));
            kernels.push_back(
                build("ConvWinogradGemm", signatureOf<K5GemmArgs>(), 256, plan.gemmBlocks));
            kernels.push_back(
                build("ConvWinogradInput", signatureOf<K5Args>(), 256, plan.inputBlocks));
            kernels.push_back(
                build("ConvWinogradOutput", signatureOf<K5Args>(), 256, plan.outputBlocks));
            break;
        case Variant::FUSED:
            kernels.push_back(
                build("ConvWinogradFusedFilter", signatureOf<K5Args>(), 256, plan.filterBlocks));
            kernels.push_back(
                build("ConvWinogradFused", signatureOf<K5Args>(), plan.mainThreads, mainGrid));
            break;
        case Variant::WMMA:
            kernels.push_back(
                build("ConvWinogradWmmaFilter", signatureOf<K5Args>(), 256, plan.filterBlocks));
            kernels.push_back(
                build("ConvWinogradWmma", signatureOf<K5Args>(), plan.mainThreads, mainGrid));
            break;
        default:
            break;
        }
        std::optional<IngestorKernelCode> copy;
        if(plan.in.needed || plan.out.needed || plan.filter.needed)
        {
            // Grid-stride: one grid sized for the largest staged operand serves all three.
            long long total = 1;
            for(const auto* staging : {&plan.in, &plan.out, &plan.filter})
            {
                if(staging->needed)
                {
                    total = std::max(total, staging->copy.total);
                }
            }
            const unsigned int copyGrid[3]
                = {static_cast<unsigned int>(std::clamp<long long>(
                       (total + COPY_BLOCK - 1) / COPY_BLOCK, 1, COPY_MAX_BLOCKS)),
                   1,
                   1};
            copy.emplace(
                build("ConvWinogradCopy", signatureOf<K5CopyArgs>(), COPY_BLOCK, copyGrid));
        }
        return std::make_unique<PreparedConvWinograd>(
            std::move(kernels), std::move(copy), *resolved);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvWinograd&>(prepared);
        const auto& resolved = dispatch.resolved();
        const auto& problem = resolved.problem;
        const auto& plan = resolved.plan;
        if(plan.workspaceBytes > 0 && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           "conv_winograd needs its workspace");
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
        const auto copy = [&](const conv_winograd::Staging& staging, const void* src, void* dst) {
            K5CopyArgs args = staging.copy;
            args.src = src;
            args.dst = dst;
            dispatch.copyForStream(stream).launch(stream, args);
        };
        void* graphIn = forward ? x : y;
        void* graphOut = forward ? y : x;
        const void* in = graphIn;
        void* out = graphOut;
        const void* filter = w;
        if(plan.in.needed)
        {
            in = base + plan.in.workspaceOffset;
            copy(plan.in, graphIn, base + plan.in.workspaceOffset);
        }
        if(plan.out.needed)
        {
            out = base + plan.out.workspaceOffset;
        }
        if(plan.filter.needed)
        {
            filter = base + plan.filter.workspaceOffset;
            copy(plan.filter, w, base + plan.filter.workspaceOffset);
        }
        K5Args args = plan.args;
        args.in = in;
        args.w = filter;
        args.out = out;
        args.u = base + plan.uOffset;
        args.v = plan.vBytes > 0 ? base + plan.vOffset : nullptr;
        args.mw = plan.mBytes > 0 ? reinterpret_cast<float*>(base + plan.mOffset) : nullptr;
        // The pre-pass reads the whole filter; its tile counts do not matter.
        dispatch.kernelForStream(0, stream).launch(stream, args);

        const long long n = problem.args.n;
        const long long eb = problem.elementBytes;
        const long long inImage = static_cast<long long>(args.ci) * args.hi * args.wi;
        const long long outImage = static_cast<long long>(args.co) * args.ho * args.wo;
        const long long tilesPerImage = static_cast<long long>(args.th) * args.tw;
        const bool multipass = resolved.config.variant == Variant::MULTIPASS;
        for(long long n0 = 0; n0 < n; n0 += plan.sliceImages)
        {
            const long long images = std::min<long long>(plan.sliceImages, n - n0);
            K5Args slice = args;
            slice.n = static_cast<int>(images);
            slice.tiles = static_cast<int>(images * tilesPerImage);
            slice.in = static_cast<const char*>(in) + n0 * inImage * eb;
            slice.out = static_cast<char*>(out) + n0 * outImage * eb;
            if(multipass)
            {
                const int cgi = slice.ci / slice.groups, cgo = slice.co / slice.groups;
                K5GemmArgs gemm{};
                gemm.a = slice.u;
                gemm.b = slice.v;
                gemm.c = slice.mw;
                gemm.m = cgo;
                gemm.n = slice.tiles;
                gemm.k = cgi;
                gemm.lda = cgi;
                gemm.ldb = slice.tiles;
                gemm.ldc = slice.tiles;
                gemm.sa = static_cast<long long>(cgo) * cgi;
                gemm.sb = static_cast<long long>(cgi) * slice.tiles;
                gemm.sc = static_cast<long long>(cgo) * slice.tiles;
                dispatch.kernelForStream(2, stream).launch(stream, slice);
                dispatch.kernelForStream(1, stream).launch(stream, gemm);
                dispatch.kernelForStream(3, stream).launch(stream, slice);
            }
            else
            {
                dispatch.kernelForStream(1, stream).launch(stream, slice);
            }
        }
        if(plan.out.needed)
        {
            copy(plan.out, base + plan.out.workspaceOffset, graphOut);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convWinogradKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvWinogradDispatchHandler& convWinogradDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convWinogradKpackModuleCache());
    static const ConvWinogradDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvWinogradSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convWinogradDispatchHandler());
}

void resetConvWinogradModuleCache()
{
    convWinogradKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
