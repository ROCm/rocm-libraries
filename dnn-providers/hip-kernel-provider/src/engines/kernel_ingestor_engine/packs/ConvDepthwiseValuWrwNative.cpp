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
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvDepthwiseValuWrwNative.cpp
 * @brief The native half of the VALU depthwise backward-weights engine
 *        (CONV_DEPTHWISE_VALU_WRW): graph matching, kernel matching, workspace sizing and
 *        dispatch of the partial and fold kernels.
 *
 * The schedule is hipconv's Hankel depthwise wgrad (partial sums per partition of the
 * reduction, folded in a fixed order), reimplemented on vector ALU so it runs on every
 * architecture, both layouts, 1-D to 3-D and every element type in exact fp32 arithmetic. Every
 * kernel config is admitted on correctness alone; autotune (UHD later) ranks them.
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
    if(a.groups < 1 || a.c != a.groups || a.k != a.groups)
    {
        return std::nullopt;
    }
    const auto layout = conv_depthwise_valu::depthwiseLayoutOf(a);
    if(!layout.has_value())
    {
        return std::nullopt;
    }
    // Shapes in 32-bit ints; element offsets are formed in 64 bits by the kernels.
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
    if(rows > INT_LIMIT || taps > INT_LIMIT || a.c * taps > INT_LIMIT)
    {
        return std::nullopt;
    }

    WrwProblem problem;
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
    w.dw_ = static_cast<int>(a.dilation[2]);
    w.rows = static_cast<int>(rows);
    w.taps = static_cast<int>(taps);
    problem.base = std::move(*base);
    return problem;
}

void partitionFor(WrwProblem& problem, int64_t computeUnits)
{
    auto& a = problem.args;
    // Enough partitions that the partial kernel fills the device, each at least one row.
    const int64_t blocksPerPartition = problem.layout == DepthwiseLayout::CHANNELS_LAST
                                           ? divUp(a.c, CHANNELS_PER_BLOCK)
                                           : static_cast<int64_t>(a.c);
    const int64_t wanted = std::max<int64_t>(1, computeUnits) * BLOCKS_PER_CU;
    const int64_t partitions
        = std::clamp<int64_t>(divUp(wanted, blocksPerPartition), 1, static_cast<int64_t>(a.rows));
    a.rowsPerPartition = static_cast<int>(divUp(a.rows, partitions));
    a.partitions = static_cast<int>(divUp(a.rows, a.rowsPerPartition));
}

size_t workspaceBytesFor(const WrwProblem& problem)
{
    const auto& a = problem.args;
    return static_cast<size_t>(a.partitions) * static_cast<size_t>(a.c)
           * static_cast<size_t>(a.taps) * sizeof(float);
}

std::optional<WrwLaunch> launchFor(const WrwProblem& problem)
{
    const auto& a = problem.args;
    WrwLaunch launch;
    launch.block = BLOCK;
    launch.partialGrid = problem.layout == DepthwiseLayout::CHANNELS_LAST
                             ? int64_t{a.partitions} * divUp(a.c, CHANNELS_PER_BLOCK)
                             : int64_t{a.partitions} * a.c;
    launch.foldGrid = divUp(int64_t{a.c} * a.taps, BLOCK);
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
/// and 2-D).
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    auto problem = conv_depthwise_valu_wrw::parseWrwProblem(context);
    if(!problem.has_value())
    {
        return false;
    }
    if(kernel.getStringMetadata("dtype") != data_objects::EnumNameDataType(problem->base.dataType)
       || kernel.getStringMetadata("layout") != conv_depthwise_valu::layoutName(problem->layout))
    {
        return false;
    }
    const auto& variant = kernel.getStringMetadata("variant");
    if(variant == "tile")
    {
        const auto& a = problem->args;
        if(a.kd != kernel.getIntMetadata("kd") || a.kh != kernel.getIntMetadata("kh")
           || a.kw != kernel.getIntMetadata("kw"))
        {
            return false;
        }
    }
    else if(variant != "floor")
    {
        return false;
    }
    conv_depthwise_valu_wrw::partitionFor(*problem, context.deviceProperties.multiProcessorCount);
    return conv_depthwise_valu_wrw::launchFor(*problem).has_value();
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
                          const KernelDefinition& /*kernel*/) const override
    {
        auto problem = conv_depthwise_valu_wrw::parseWrwProblem(context);
        if(!problem.has_value())
        {
            return 0;
        }
        conv_depthwise_valu_wrw::partitionFor(*problem,
                                              context.deviceProperties.multiProcessorCount);
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
        conv_depthwise_valu_wrw::partitionFor(*problem,
                                              context.deviceProperties.multiProcessorCount);
        const auto launch = conv_depthwise_valu_wrw::launchFor(*problem);
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
        const bool tile = kernel.getStringMetadata("variant") == "tile";
        options.add("CONV_DW_VARIANT", std::string(tile ? "1" : "0"));
        options.add("CONV_DW_FLIP", std::string(problem->flipFilter ? "1" : "0"));
        if(tile)
        {
            options.add("CONV_DW_KD", std::to_string(kernel.getIntMetadata("kd")));
            options.add("CONV_DW_KH", std::to_string(kernel.getIntMetadata("kh")));
            options.add("CONV_DW_KW", std::to_string(kernel.getIntMetadata("kw")));
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
