// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
#include "engines/kernel_ingestor_engine/packs/ConvWrwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvWrwNative.cpp
 * @brief The native half of the matrix-core backward-weights engine (CONV_WRW, K3): graph
 *        matching, kernel matching, workspace sizing and dispatch of the partial and reduce
 *        kernels.
 *
 * The kernel (descriptors/conv/wrw/kernels) is a row-direct implicit GEMM on K1's matrix-core
 * atoms (MFMA on gfx908 to gfx950, WMMA on gfx11 and gfx12), with FMA loops for exact fp32
 * where there is no fp32 matrix core and, on gfx950, a direct loop for 16-bit channels-last
 * unit-stride layers (LDS DMA, swizzled transpose reads, an 8-wave ping-pong, a persistent
 * grid): x and dy are read in place
 * (packed channels-first or channels-last; the generic tier, loop 8, through any strides and
 * at runtime filter widths, strides and dilations, where no specialized config's class serves),
 * every filter-width tap of a block's filter rows
 * accumulates in registers, and the reduction over images and output positions is split over
 * blocks into fp32 partials in the workspace, which ConvWrwReduce (the same source under a
 * second entry point) adds in split order into dw: deterministic, no atomics. At one split
 * with dw's channels contiguous the kernel writes dw itself and neither the workspace nor the
 * reduce is used (WrwLaunch::dwStore). fp32, fp16 and
 * bf16; shapes are runtime, a config fixes the tile, the loop and the filter width, its stride
 * and dilation. Every config is admitted on correctness alone (wrwLaunchFor); autotune, and UHD
 * once trained, ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_wrw::WrwConfig;
using conv_wrw::WrwLaunch;
using conv_wrw::WrwProblem;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.conv_wrw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_wrw.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_wrw.dispatch";
/// The reduce kernels: the same source as ConvWrw, under other entry points; the launch says
/// which (the row reduce for K C T R S at few splits).
constexpr std::string_view REDUCE_ENTRY_POINT = "ConvWrwReduce";
constexpr std::string_view REDUCE_ROWS_ENTRY_POINT = "ConvWrwReduceRows";
/// Three-bf16's operand split (math 1), the same source again.
constexpr std::string_view SPLIT_ENTRY_POINT = "ConvWrwSplit";

// Tokens the graph match binds.
constexpr std::string_view X_TOKEN = "conv_wrw.x.uid";
constexpr std::string_view DY_TOKEN = "conv_wrw.dy.uid";
constexpr std::string_view DW_TOKEN = "conv_wrw.dw.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_wrw.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_wrw.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_wrw.flops";

/// The config @p kernel's metadata declares. Throws when a field is missing or mistyped.
WrwConfig configOf(const KernelDefinition& kernel)
{
    WrwConfig config;
    config.atom = kernel.getStringMetadata("atom");
    config.atomId = kernel.getIntMetadata("atom_id");
    config.loop = kernel.getIntMetadata("loop");
    config.layout = kernel.getIntMetadata("layout");
    config.bk = kernel.getIntMetadata("bk");
    config.bc = kernel.getIntMetadata("bc");
    config.bq = kernel.getIntMetadata("bq");
    config.fr = kernel.getIntMetadata("fr");
    config.s = kernel.getIntMetadata("s");
    config.sw = kernel.getIntMetadata("sw");
    config.dw = kernel.getIntMetadata("dw");
    config.wk = kernel.getIntMetadata("wk");
    config.wc = kernel.getIntMetadata("wc");
    config.tk = kernel.getIntMetadata("tk");
    config.tc = kernel.getIntMetadata("tc");
    config.vec = kernel.getIntMetadata("vec");
    config.xcd = kernel.getIntMetadata("xcd");
    config.align = kernel.getIntMetadata("align");
    config.ns = kernel.getIntMetadata("ns");
    config.wq = kernel.getIntMetadata("wq");
    config.dws = kernel.getIntMetadata("dws");
    config.pb = kernel.getIntMetadata("pb");
    config.bpu = kernel.getIntMetadata("bpu");
    // The generic loop's specialized classes (the two tiers); no other loop carries them.
    if(config.loop == conv_wrw::LOOP_GENERIC)
    {
        config.spec = kernel.getStringMetadata("spec");
    }
    config.math = kernel.getIntMetadata("math");
    return config;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto problem = conv_wrw::parseWrwProblem(context);
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
    bound[std::string(DIRECTION_TOKEN)] = std::string("WRW");
    const auto flops = pack_features::checkedMultiply(2, problem->base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

/// The launch @p kernel runs @p context's graph with, or nullopt when the config does not serve
/// it on this device (or its metadata is malformed).
std::optional<WrwLaunch> launchOf(const MatchContext& context, const KernelDefinition& kernel)
{
    try
    {
        if(kernel.getStringMetadata("direction") != "WRW")
        {
            return std::nullopt;
        }
        const auto problem = conv_wrw::parseWrwProblem(context);
        if(!problem.has_value()
           || kernel.getStringMetadata("dtype")
                  != data_objects::EnumNameDataType(problem->base.dataType))
        {
            return std::nullopt;
        }
        return conv_wrw::wrwLaunchFor(*problem, configOf(kernel), context.deviceProperties);
    }
    catch(const std::out_of_range&)
    {
        return std::nullopt;
    }
    catch(const std::invalid_argument&)
    {
        return std::nullopt;
    }
}

/// Kernel-scoped applicability: the config runs on this device and computes this problem
/// correctly.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return launchOf(context, kernel).has_value();
}

/// The single argument each kernel takes. Only consulted for a pre-built kernel.
const std::vector<KernelArgument>& wrwSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K3Args)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& reduceSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K3ReduceArgs)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& splitSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K3SplitArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvWrw : public IngestorPreparedDispatch
{
public:
    /// @p reduce is empty when the kernel writes dw itself (WrwLaunch::dwStore); @p split is
    /// set for three-bf16 (WrwLaunch::split) only.
    PreparedConvWrw(IngestorKernelCode wrw,
                    std::optional<IngestorKernelCode> reduce,
                    std::optional<IngestorKernelCode> split,
                    const WrwLaunch& launch,
                    const WrwProblem& problem)
        : _wrw(std::move(wrw))
        , _reduce(std::move(reduce))
        , _split(std::move(split))
        , _args(launch.args)
        , _reduceArgs(launch.reduce)
        , _splitArgs(launch.splitArgs)
        , _splitYOffset(launch.splitYOffset)
        , _splitBytes(launch.splitBytes)
        , _xUid(problem.xUid)
        , _dyUid(problem.dyUid)
        , _dwUid(problem.dwUid)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _wrw;
    }

    compilation::IRunnableKernel& wrwForStream(hipStream_t stream) const
    {
        return _wrw.kernelForStream(stream);
    }

    bool hasReduce() const
    {
        return _reduce.has_value();
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce->kernelForStream(stream);
    }

    bool hasSplit() const
    {
        return _split.has_value();
    }

    compilation::IRunnableKernel& splitForStream(hipStream_t stream) const
    {
        return _split->kernelForStream(stream);
    }

    void argsFor(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace,
                 K3Args& args,
                 K3ReduceArgs& reduce,
                 K3SplitArgs& split) const
    {
        args = _args;
        reduce = _reduceArgs;
        split = _splitArgs;
        args.x = hipdnn_plugin_sdk::findDeviceBuffer(_xUid, deviceBuffers, numDeviceBuffers).ptr;
        args.dy = hipdnn_plugin_sdk::findDeviceBuffer(_dyUid, deviceBuffers, numDeviceBuffers).ptr;
        // Three-bf16: the kernel reads the split operands; the partials follow them.
        auto* partial = static_cast<char*>(workspace);
        if(_split.has_value())
        {
            split.x = static_cast<const float*>(args.x);
            split.dy = static_cast<const float*>(args.dy);
            split.xs = reinterpret_cast<unsigned short*>(partial);
            split.ys = reinterpret_cast<unsigned short*>(partial + _splitYOffset);
            args.x = split.xs;
            args.dy = split.ys;
            partial += _splitBytes;
        }
        args.partial = reinterpret_cast<float*>(partial);
        reduce.partial = reinterpret_cast<const float*>(partial);
        reduce.dw
            = hipdnn_plugin_sdk::findDeviceBuffer(_dwUid, deviceBuffers, numDeviceBuffers).ptr;
        args.dwOut = reduce.dw;
    }

private:
    IngestorKernelCode _wrw;
    std::optional<IngestorKernelCode> _reduce;
    std::optional<IngestorKernelCode> _split;
    K3Args _args;
    K3ReduceArgs _reduceArgs;
    K3SplitArgs _splitArgs;
    size_t _splitYOffset;
    size_t _splitBytes;
    int64_t _xUid;
    int64_t _dyUid;
    int64_t _dwUid;
};

class ConvWrwDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvWrwDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                           const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The splits' fp32 partials.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto launch = launchOf(context, kernel);
        return launch.has_value() ? launch->workspaceBytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto launch = launchOf(context, kernel);
        const auto problem = conv_wrw::parseWrwProblem(context);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto dy = tryGetBoundInt(bound, DY_TOKEN);
        const auto dw = tryGetBoundInt(bound, DW_TOKEN);
        if(!launch.has_value() || !problem.has_value() || !x.has_value() || !dy.has_value()
           || !dw.has_value() || *x != problem->xUid || *dy != problem->dyUid
           || *dw != problem->dwUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_wrw dispatch was given a graph or config its matchers did not admit");
        }
        const auto config = configOf(kernel);

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        for(const auto& [name, value] :
            conv_wrw::compileDefinesFor(problem->base.dataType, config, launch->kernelType))
        {
            options.add(name, value);
        }

        auto wrw = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, wrwSignature());
        std::optional<IngestorKernelCode> reduce;
        if(!launch->dwStore)
        {
            KernelDefinition reduceKernel = kernel;
            // The reduce is the same source under another entry point.
            reduceKernel.source.entryPoint = std::string(
                launch->reduce.rows != 0 ? REDUCE_ROWS_ENTRY_POINT : REDUCE_ENTRY_POINT);
            reduce = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, reduceKernel, options, reduceSignature());
            reduce->setBlockSize(launch->reduceBlock, 1, 1);
            reduce->setGridSize(launch->reduceGrid, 1, 1);
        }
        std::optional<IngestorKernelCode> split;
        if(launch->split)
        {
            KernelDefinition splitKernel = kernel;
            splitKernel.source.entryPoint = std::string(SPLIT_ENTRY_POINT);
            split = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, splitKernel, options, splitSignature());
            split->setBlockSize(256, 1, 1);
            split->setGridSize(launch->splitGrid, 1, 1);
        }
        wrw.setBlockSize(launch->block, 1, 1);
        wrw.setGridSize(launch->gridX, launch->gridY, launch->gridZ);
        return std::make_unique<PreparedConvWrw>(
            std::move(wrw), std::move(reduce), std::move(split), *launch, *problem);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvWrw&>(prepared);
        if((dispatch.hasReduce() || dispatch.hasSplit()) && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_wrw needs its partial-sum workspace");
        }
        K3Args args{};
        K3ReduceArgs reduce{};
        K3SplitArgs split{};
        dispatch.argsFor(deviceBuffers, numDeviceBuffers, workspace, args, reduce, split);
        const auto stream = handle.getStream();
        if(dispatch.hasSplit())
        {
            dispatch.splitForStream(stream).launch(stream, split);
        }
        dispatch.wrwForStream(stream).launch(stream, args);
        if(dispatch.hasReduce())
        {
            dispatch.reduceForStream(stream).launch(stream, reduce);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convWrwKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvWrwDispatchHandler& convWrwDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convWrwKpackModuleCache());
    static const ConvWrwDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvWrwSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convWrwDispatchHandler());
}

void resetConvWrwModuleCache()
{
    convWrwKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
