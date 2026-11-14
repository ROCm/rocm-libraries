// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
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
#include "engines/kernel_ingestor_engine/descriptors/conv/depthwise_valu_wrw/kernels/ConvDepthwiseValuWrwArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuWrwNative.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvDepthwiseValuWrwNative.cpp
 * @brief The native half of the VALU depthwise backward-weights engine
 *        (CONV_DEPTHWISE_VALU_WRW): graph matching, kernel matching, workspace sizing and
 *        dispatch of the partial and fold kernels.
 *
 * The schedule is hipconv's Hankel depthwise wgrad (partial sums per partition of the
 * reduction, folded in a fixed order), reimplemented on vector ALU so it runs on every
 * architecture, both layouts, 1-D to 3-D and every element type in exact fp32 arithmetic. On
 * CDNA the hankel configs run the construction itself on 4x4x4 MFMA (16-bit, channels-last,
 * stride 1 or 2; ConvDepthwiseHankelWrw.h). Every kernel config is admitted on correctness alone;
 * autotune (UHD later) ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_depthwise_valu::DepthwiseLayout;
using conv_naive::ConvDirection;

namespace conv_depthwise_valu_wrw
{

namespace
{

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();

/// Threads per block of both kernels, and channels per block of the channels-last partial.
constexpr int64_t BLOCK = 256;
constexpr int64_t CHANNELS_PER_BLOCK = 64;
/// Blocks per compute unit the partial kernel's partitions aim for.
constexpr int64_t BLOCKS_PER_CU = 4;

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

} // namespace

std::optional<WrwProblem> parseWrwProblem(const MatchContext& context)
{
    auto base = conv_naive::parseConvNaiveProblem(context, ConvDirection::BACKWARD_WEIGHTS);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    const auto& a = base->args;
    // One input channel a group; the base parse makes the output channels a whole number per
    // group (the multiplier).
    if(a.groups < 1 || a.c != a.groups)
    {
        return std::nullopt;
    }
    const auto layout = conv_depthwise_valu::depthwiseLayoutOf(a);
    if(!layout.has_value())
    {
        return std::nullopt;
    }
    // Shapes in 32-bit ints; element offsets are formed in 64 bits by the kernels.
    // (The output channels are bounded with their taps below.)
    for(const long long value : {a.n, a.c})
    {
        if(value < 1 || value > INT_LIMIT)
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
            if(value < -INT_LIMIT || value > INT_LIMIT)
            {
                return std::nullopt;
            }
        }
    }
    const long long rows = a.n * a.outSpatial[0] * a.outSpatial[1];
    const long long taps = a.filter[0] * a.filter[1] * a.filter[2];
    if(rows > INT_LIMIT || taps > INT_LIMIT || a.k * taps > INT_LIMIT)
    {
        return std::nullopt;
    }

    WrwProblem problem;
    problem.mult = a.k / a.c;
    problem.layout = *layout;
    problem.ndims = base->spatialRank == 3 ? 3 : 2;
    problem.flipFilter = a.flipFilter != 0;
    problem.xUid = base->operands.x;
    problem.dyUid = base->operands.y;
    problem.dwUid = base->operands.w;
    auto& w = problem.args;
    w.n = static_cast<int>(a.n);
    w.c = static_cast<int>(a.c);
    w.di = static_cast<int>(a.inSpatial[0]);
    w.hi = static_cast<int>(a.inSpatial[1]);
    w.wi = static_cast<int>(a.inSpatial[2]);
    w.dOut = static_cast<int>(a.outSpatial[0]);
    w.ho = static_cast<int>(a.outSpatial[1]);
    w.wo = static_cast<int>(a.outSpatial[2]);
    w.kd = static_cast<int>(a.filter[0]);
    w.kh = static_cast<int>(a.filter[1]);
    w.kw = static_cast<int>(a.filter[2]);
    w.pd = static_cast<int>(a.prePadding[0]);
    w.ph = static_cast<int>(a.prePadding[1]);
    w.pw = static_cast<int>(a.prePadding[2]);
    w.sd = static_cast<int>(a.stride[0]);
    w.sh = static_cast<int>(a.stride[1]);
    w.sw = static_cast<int>(a.stride[2]);
    w.dd = static_cast<int>(a.dilation[0]);
    w.dh = static_cast<int>(a.dilation[1]);
    w.dilW = static_cast<int>(a.dilation[2]);
    w.rows = static_cast<int>(rows);
    w.taps = static_cast<int>(taps);
    problem.base = *base;
    return problem;
}

namespace
{

bool isHankel(const WrwConfig& config)
{
    return config.variant == "hankel";
}

/// The vector tile's lanes along the channels (ConvDepthwiseValuWrw.cpp, vtile::channelLanes).
int64_t vtileChannelLanes(int64_t channels, int64_t cv)
{
    const int64_t need = divUp(channels, cv);
    int64_t lanes = 1;
    while(lanes < need && lanes < 64)
    {
        lanes *= 2;
    }
    return lanes;
}

/// Partial blocks per partition: channel blocks.
int64_t blocksPerPartitionOf(const WrwProblem& problem, const WrwConfig& config)
{
    const auto& a = problem.args;
    if(isHankel(config))
    {
        return divUp(a.c, 64 * config.nwg);
    }
    if(config.variant == "vtile")
    {
        return divUp(a.c, vtileChannelLanes(a.c, config.cv) * config.cv);
    }
    // The VALU kernels walk output channels.
    const int64_t channels = a.c * problem.mult;
    return problem.layout == DepthwiseLayout::CHANNELS_LAST ? divUp(channels, CHANNELS_PER_BLOCK)
                                                            : channels;
}

} // namespace

int64_t hankelLdsBytes(const WrwConfig& config)
{
    // ConvDepthwiseHankelWrw.h: G x-row buffers (s phase rows each) of 64 nwg channel rows, and
    // KH + G - 1 ring slots plus a zero slot of 64 nwg dy rows (+ 16 halves to spread the slots
    // over banks).
    const int64_t sg = divUp(config.kw, 4);
    const int64_t stride = std::max<int64_t>(config.s, 1);
    const int64_t xg = stride == 1 ? config.q / 4 + sg : config.q / 4 + (sg - 1) / 2 + 1;
    // Row pitches: whole dwords, an odd number of dword pairs (pitchOf).
    const auto pitchOf
        = [](int64_t dwords) { return 2 * ((dwords + 2) / 2 % 2 == 1 ? dwords + 2 : dwords + 4); };
    const int64_t channels = 64 * config.nwg;
    const int64_t xPitch = pitchOf(2 * xg);
    const int64_t dyPitch = pitchOf(2 * (config.q / 4));
    // Plus half a row's halves for every 8 rows' skew (skewOf).
    const int64_t slot = divUp(channels * dyPitch + channels / 2, 64) * 64 + 16;
    const int64_t slots = config.kh + config.g - 1;
    return (config.g * stride * (channels * xPitch + channels / 2) + (slots + 1) * slot) * 2;
}

bool hankelAdmits(const WrwProblem& problem,
                  const WrwConfig& config,
                  std::string_view gcnArchName,
                  int xAlign,
                  int dyAlign)
{
    // 16-byte loads of x and dy (8 channels a column).
    if(xAlign % 16 != 0 || dyAlign % 16 != 0)
    {
        return false;
    }
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    const bool f16 = problem.base.dataType == data_objects::DataType::HALF;
    const bool bf16 = problem.base.dataType == data_objects::DataType::BFLOAT16;
    // 4x4x4 MFMA: fp16 from gfx908, bf16 (the _1k form) from gfx90a.
    const bool cdna = arch == "gfx90a" || arch == "gfx942" || arch == "gfx950";
    if(!((f16 && (cdna || arch == "gfx908")) || (bf16 && cdna)))
    {
        return false;
    }
    const auto& a = problem.args;
    // No multiplier (the matrix-core kernel walks input channels).
    if(config.mult != 1 || problem.layout != DepthwiseLayout::CHANNELS_LAST || problem.ndims != 2
       || a.kd != 1 || config.kd != 1 || a.kh != config.kh || a.kw != config.kw)
    {
        return false;
    }
    // Stride 2 reads at most two words a tap group: filters up to 8 wide.
    if((config.s != 1 && config.s != 2) || (config.s == 2 && config.kw > 8))
    {
        return false;
    }
    if(a.sh != config.s || a.sw != config.s || a.dh != 1 || a.dilW != 1 || a.c % 8 != 0)
    {
        return false;
    }
    // The kernel addresses an image of x and of dy through a buffer descriptor: under 2 GiB.
    const auto maxImage = int64_t{0x7fffffff};
    if(int64_t{a.hi} * a.wi * a.c * 2 > maxImage || int64_t{a.ho} * a.wo * a.c * 2 > maxImage)
    {
        return false;
    }
    if(config.q < 4 || config.q > 64 || config.q % 4 != 0 || config.g < 1 || config.g > 4
       || (config.nwg != 1 && config.nwg != 2 && config.nwg != 4)
       || (config.bpc != 1 && config.bpc != 2 && config.bpc != 4 && config.bpc != 8))
    {
        return false;
    }
    const int64_t ldsLimit = arch == "gfx950" ? 160 * 1024 : 64 * 1024;
    return hankelLdsBytes(config) <= ldsLimit;
}

bool vtileAdmits(const WrwProblem& problem, const WrwConfig& config, int xAlign, int dyAlign)
{
    const auto& a = problem.args;
    if(config.mult != 1 || problem.mult != 1 || problem.layout != DepthwiseLayout::CHANNELS_LAST
       || problem.ndims != 2 || a.kd != 1 || config.kd != 1 || a.kh != config.kh
       || a.kw != config.kw || a.sw != 1 || a.dilW != 1)
    {
        return false;
    }
    if((config.cv != 1 && config.cv != 2 && config.cv != 4 && config.cv != 8)
       || (config.strip != 2 && config.strip != 4 && config.strip != 8) || a.c % config.cv != 0)
    {
        return false;
    }
    const int64_t vectorBytes
        = config.cv * (problem.base.dataType == data_objects::DataType::FLOAT ? 4 : 2);
    return xAlign % vectorBytes == 0 && dyAlign % vectorBytes == 0;
}

void partitionFor(WrwProblem& problem, int64_t computeUnits, const WrwConfig& config)
{
    auto& a = problem.args;
    // Enough partitions that the partial kernel fills the device, each at least one row.
    const int64_t blocksPerPartition = blocksPerPartitionOf(problem, config);
    const int64_t wanted
        = std::max<int64_t>(1, computeUnits) * (isHankel(config) ? config.bpc : BLOCKS_PER_CU);
    // A vector tile's block splits a partition's rows over its row lanes: at least four rows a
    // lane, or the lanes idle and the partial sums outgrow the work.
    const int64_t rowsAtLeast
        = config.variant == "vtile"
              ? 4 * (BLOCK / vtileChannelLanes(a.c, std::max<int64_t>(config.cv, 1)))
              : 1;
    const int64_t partitions = std::clamp<int64_t>(
        divUp(wanted, blocksPerPartition), 1, std::max<int64_t>(1, a.rows / rowsAtLeast));
    a.rowsPerPartition = static_cast<int>(divUp(a.rows, partitions));
    a.partitions = static_cast<int>(divUp(a.rows, a.rowsPerPartition));
}

size_t workspaceBytesFor(const WrwProblem& problem)
{
    const auto& a = problem.args;
    return static_cast<size_t>(a.partitions) * static_cast<size_t>(a.c * problem.mult)
           * static_cast<size_t>(a.taps) * sizeof(float);
}

std::optional<WrwLaunch> launchFor(const WrwProblem& problem, const WrwConfig& config)
{
    const auto& a = problem.args;
    WrwLaunch launch;
    launch.block = BLOCK;
    launch.partialGrid = int64_t{a.partitions} * blocksPerPartitionOf(problem, config);
    launch.foldGrid = divUp(int64_t{a.c} * problem.mult * a.taps, BLOCK);
    if(launch.partialGrid < 1 || launch.partialGrid > INT_LIMIT || launch.foldGrid > INT_LIMIT)
    {
        return std::nullopt;
    }
    return launch;
}

} // namespace conv_depthwise_valu_wrw

namespace
{

using conv_depthwise_valu_wrw::WrwProblem;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.conv_depthwise_valu_wrw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_depthwise_valu_wrw.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_depthwise_valu_wrw.dispatch";
/// The fold kernel: the same source as the partial kernel, under another entry point.
constexpr std::string_view FOLD_ENTRY_POINT = "ConvDepthwiseValuWrwFold";

// Tokens the graph match binds.
constexpr std::string_view X_TOKEN = "conv_depthwise_valu_wrw.x.uid";
constexpr std::string_view DY_TOKEN = "conv_depthwise_valu_wrw.dy.uid";
constexpr std::string_view DW_TOKEN = "conv_depthwise_valu_wrw.dw.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_depthwise_valu_wrw.dtype";
constexpr std::string_view FLOPS_TOKEN = "conv_depthwise_valu_wrw.flops";

/// The byte alignment the graph declares for tensor @p uid's pointer, at most 16; 0 when the
/// tensor is missing.
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

std::string elementTypeFor(const std::string& dtype)
{
    if(dtype == "FLOAT")
    {
        return "float";
    }
    if(dtype == "HALF")
    {
        return "_Float16";
    }
    if(dtype == "BFLOAT16")
    {
        return "__bf16";
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "conv_depthwise_valu_wrw kernel declares unsupported dtype '" + dtype + "'");
}

/// A kernel's config, from its metadata.
conv_depthwise_valu_wrw::WrwConfig configOf(const KernelDefinition& kernel)
{
    conv_depthwise_valu_wrw::WrwConfig config;
    config.variant = kernel.getStringMetadata("variant");
    config.kd = kernel.getIntMetadata("kd");
    config.kh = kernel.getIntMetadata("kh");
    config.kw = kernel.getIntMetadata("kw");
    config.q = kernel.getIntMetadata("q");
    config.g = kernel.getIntMetadata("g");
    config.nwg = kernel.getIntMetadata("nwg");
    config.bpc = kernel.getIntMetadata("bpc");
    config.mult = kernel.getIntMetadata("mult");
    config.s = kernel.getIntMetadata("s");
    config.cv = kernel.getIntMetadata("cv");
    config.strip = kernel.getIntMetadata("strip");
    return config;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto problem = conv_depthwise_valu_wrw::parseWrwProblem(context);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = problem->xUid;
    bound[std::string(DY_TOKEN)] = problem->dyUid;
    bound[std::string(DW_TOKEN)] = problem->dwUid;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(problem->base.dataType));
    const auto flops = pack_features::checkedMultiply(2, problem->base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

/// Kernel-scoped applicability: the config computes this problem correctly and launches. The
/// floor serves every problem; a register tile needs its compile-time filter (depth 1 for 1-D
/// and 2-D); the matrix-core (hankel) configs pass hankelAdmits, and only they may set the
/// q, g, nwg, bpc and s fields; the vector tiles pass vtileAdmits, and only they may set cv and
/// strip. The partitions are sized from the device's units, so a device that
/// reports none is declined.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    auto problem = conv_depthwise_valu_wrw::parseWrwProblem(context);
    if(!problem.has_value() || device_units::dispatchUnits(context.deviceProperties) < 1)
    {
        return false;
    }
    if(kernel.getStringMetadata("dtype") != data_objects::EnumNameDataType(problem->base.dataType)
       || kernel.getStringMetadata("layout") != conv_depthwise_valu::layoutName(problem->layout))
    {
        return false;
    }
    const auto config = configOf(kernel);
    const bool hankel = config.variant == "hankel";
    // The kernel's multiplier is the graph's.
    if(config.mult != problem->mult)
    {
        return false;
    }
    // The matrix-core fields are the hankel variant's alone.
    if(!hankel
       && (config.q != 0 || config.g != 0 || config.nwg != 0 || config.bpc != 0 || config.s != 0))
    {
        return false;
    }
    // The vector-tile fields are the vtile variant's alone.
    const bool vtile = config.variant == "vtile";
    if(!vtile && (config.cv != 0 || config.strip != 0))
    {
        return false;
    }
    const auto& a = problem->args;
    if(config.variant == "tile")
    {
        if(a.kd != config.kd || a.kh != config.kh || a.kw != config.kw)
        {
            return false;
        }
    }
    else if(hankel)
    {
        if(!conv_depthwise_valu_wrw::hankelAdmits(*problem,
                                                  config,
                                                  context.deviceProperties.gcnArchName,
                                                  pointerAlignment(context, problem->xUid),
                                                  pointerAlignment(context, problem->dyUid)))
        {
            return false;
        }
    }
    else if(vtile)
    {
        if(!conv_depthwise_valu_wrw::vtileAdmits(*problem,
                                                 config,
                                                 pointerAlignment(context, problem->xUid),
                                                 pointerAlignment(context, problem->dyUid)))
        {
            return false;
        }
    }
    else if(config.variant != "floor")
    {
        return false;
    }
    conv_depthwise_valu_wrw::partitionFor(
        *problem, device_units::dispatchUnits(context.deviceProperties), config);
    return conv_depthwise_valu_wrw::launchFor(*problem, config).has_value();
}

/// The single argument both kernels take. Only consulted for a pre-built kernel.
const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvDepthwiseValuWrwArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvDepthwiseValuWrw : public IngestorPreparedDispatch
{
public:
    PreparedConvDepthwiseValuWrw(IngestorKernelCode partial,
                                 IngestorKernelCode fold,
                                 const WrwProblem& problem)
        : _partial(std::move(partial))
        , _fold(std::move(fold))
        , _problem(problem)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _partial;
    }

    compilation::IRunnableKernel& partialForStream(hipStream_t stream) const
    {
        return _partial.kernelForStream(stream);
    }

    compilation::IRunnableKernel& foldForStream(hipStream_t stream) const
    {
        return _fold.kernelForStream(stream);
    }

    const WrwProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _partial;
    IngestorKernelCode _fold;
    WrwProblem _problem;
};

class ConvDepthwiseValuWrwDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvDepthwiseValuWrwDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                        const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// One fp32 partial sum per (partition, channel, tap).
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        auto problem = conv_depthwise_valu_wrw::parseWrwProblem(context);
        if(!problem.has_value())
        {
            return 0;
        }
        conv_depthwise_valu_wrw::partitionFor(
            *problem, device_units::dispatchUnits(context.deviceProperties), configOf(kernel));
        return conv_depthwise_valu_wrw::workspaceBytesFor(*problem);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        auto problem = conv_depthwise_valu_wrw::parseWrwProblem(context);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto dy = tryGetBoundInt(bound, DY_TOKEN);
        const auto dw = tryGetBoundInt(bound, DW_TOKEN);
        if(!problem.has_value() || !x.has_value() || !dy.has_value() || !dw.has_value()
           || *x != problem->xUid || *dy != problem->dyUid || *dw != problem->dwUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_depthwise_valu_wrw dispatch was given a graph its matcher did not admit");
        }
        const auto config = configOf(kernel);
        conv_depthwise_valu_wrw::partitionFor(
            *problem, device_units::dispatchUnits(context.deviceProperties), config);
        const auto launch = conv_depthwise_valu_wrw::launchFor(*problem, config);
        if(!launch.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_depthwise_valu_wrw dispatch was given a config its kernel match declines");
        }

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_DW_TYPE", elementTypeFor(kernel.getStringMetadata("dtype")));
        options.add("CONV_DW_LAYOUT",
                    std::string(problem->layout == DepthwiseLayout::CHANNELS_FIRST ? "1" : "0"));
        const bool tile = config.variant == "tile";
        const bool hankel = config.variant == "hankel";
        const bool vtile = config.variant == "vtile";
        std::string variant = "0";
        if(vtile)
        {
            variant = "3";
        }
        else if(hankel)
        {
            variant = "2";
        }
        else if(tile)
        {
            variant = "1";
        }
        options.add("CONV_DW_VARIANT", variant);
        options.add("CONV_DW_FLIP", std::string(problem->flipFilter ? "1" : "0"));
        // Defined only for a multiplier, so the multiplier-1 programs compile as before.
        if(config.mult != 1)
        {
            options.add("CONV_DW_MULT", std::to_string(config.mult));
        }
        if(tile || hankel || vtile)
        {
            options.add("CONV_DW_KD", std::to_string(config.kd));
            options.add("CONV_DW_KH", std::to_string(config.kh));
            options.add("CONV_DW_KW", std::to_string(config.kw));
        }
        if(hankel)
        {
            options.add("CONV_DW_Q", std::to_string(config.q));
            options.add("CONV_DW_G", std::to_string(config.g));
            options.add("CONV_DW_NWG", std::to_string(config.nwg));
            // Defined only for stride 2, so the unit-stride programs compile as before.
            if(config.s != 1)
            {
                options.add("CONV_DW_S", std::to_string(config.s));
            }
            // Only where the tiles pass wo, so the other programs compile as before.
            if(problem->args.wo % config.q != 0)
            {
                options.add("CONV_DW_COLMASK", std::string("1"));
            }
        }
        if(vtile)
        {
            options.add("CONV_DW_CV", std::to_string(config.cv));
            options.add("CONV_DW_STRIP", std::to_string(config.strip));
        }

        auto partial = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, kernelSignature());
        KernelDefinition foldKernel = kernel;
        // The fold kernel is the same source under another entry point.
        foldKernel.source.entryPoint = std::string(FOLD_ENTRY_POINT);
        auto fold = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, foldKernel, options, kernelSignature());
        partial.setBlockSize(static_cast<unsigned int>(launch->block), 1, 1);
        partial.setGridSize(static_cast<unsigned int>(launch->partialGrid), 1, 1);
        fold.setBlockSize(static_cast<unsigned int>(launch->block), 1, 1);
        fold.setGridSize(static_cast<unsigned int>(launch->foldGrid), 1, 1);
        return std::make_unique<PreparedConvDepthwiseValuWrw>(
            std::move(partial), std::move(fold), *problem);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "conv_depthwise_valu_wrw needs its partial-sum workspace");
        }
        const auto& dispatch = dynamic_cast<const PreparedConvDepthwiseValuWrw&>(prepared);
        const auto& problem = dispatch.problem();
        ConvDepthwiseValuWrwArgs args = problem.args;
        args.x = hipdnn_plugin_sdk::findDeviceBuffer(problem.xUid, deviceBuffers, numDeviceBuffers)
                     .ptr;
        args.dy
            = hipdnn_plugin_sdk::findDeviceBuffer(problem.dyUid, deviceBuffers, numDeviceBuffers)
                  .ptr;
        args.dw
            = hipdnn_plugin_sdk::findDeviceBuffer(problem.dwUid, deviceBuffers, numDeviceBuffers)
                  .ptr;
        args.partial = static_cast<float*>(workspace);
        const auto stream = handle.getStream();
        dispatch.partialForStream(stream).launch(stream, args);
        dispatch.foldForStream(stream).launch(stream, args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convDepthwiseValuWrwKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvDepthwiseValuWrwDispatchHandler& convDepthwiseValuWrwDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        convDepthwiseValuWrwKpackModuleCache());
    static const ConvDepthwiseValuWrwDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                       s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvDepthwiseValuWrwSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convDepthwiseValuWrwDispatchHandler());
}

void resetConvDepthwiseValuWrwModuleCache()
{
    convDepthwiseValuWrwKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
