// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
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
#include "engines/kernel_ingestor_engine/packs/ConvStemBwdProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvStemBwdNative.cpp
 * @brief The native half of the stem backward-data engine (CONV_STEM_BWD): graph matching,
 *        kernel matching, workspace sizing and dispatch of the filter transform and the main
 *        kernel.
 *
 * dx for at most 4 input channels (ConvStemBwdProblem.hpp): the stride phases of dx join the
 * input channels on the GEMM's N axis, so a 7x7 stride-2 stem's 3 channels fill 12 of 16
 * matrix-core columns instead of 3. The filter is rearranged into the workspace on every launch
 * (ConvStemBwdTransform, the same source under a second entry point); the main kernel reads
 * dy in place and writes every dx element once. fp32, fp16 and bf16, on K1's matrix-core atoms
 * or (fp32) the FMA loop; shapes are runtime, a config fixes the strides, the window and the
 * tile. Every config is admitted on correctness alone (stemBwdLaunchFor); autotune, and UHD once
 * trained, ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_stem_bwd::StemBwdConfig;
using conv_stem_bwd::StemBwdLaunch;
using conv_stem_bwd::StemBwdProblem;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.conv_stem_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_stem_bwd.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_stem_bwd.dispatch";
/// The filter transform: the same source as ConvStemBwd, under another entry point.
constexpr std::string_view TRANSFORM_ENTRY_POINT = "ConvStemBwdTransform";

// Tokens the graph match binds.
constexpr std::string_view DX_TOKEN = "conv_stem_bwd.dx.uid";
constexpr std::string_view W_TOKEN = "conv_stem_bwd.w.uid";
constexpr std::string_view DY_TOKEN = "conv_stem_bwd.dy.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_stem_bwd.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_stem_bwd.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_stem_bwd.flops";

/// The config @p kernel's metadata declares. Throws when a field is missing or mistyped.
StemBwdConfig configOf(const KernelDefinition& kernel)
{
    StemBwdConfig config;
    config.atom = kernel.getStringMetadata("atom");
    config.atomId = kernel.getIntMetadata("atom_id");
    config.layout = kernel.getIntMetadata("layout");
    config.vec = kernel.getIntMetadata("vec");
    config.u = kernel.getIntMetadata("u");
    config.v = kernel.getIntMetadata("v");
    config.nb = kernel.getIntMetadata("nb");
    config.kc = kernel.getIntMetadata("kc");
    config.th = kernel.getIntMetadata("th");
    config.fm = kernel.getIntMetadata("fm");
    config.sh = kernel.getIntMetadata("sh");
    config.sw = kernel.getIntMetadata("sw");
    config.ft = kernel.getIntMetadata("ft");
    config.sd = kernel.getIntMetadata("sd");
    return config;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto problem = conv_stem_bwd::parseStemBwdProblem(context);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(DX_TOKEN)] = problem->dxUid;
    bound[std::string(W_TOKEN)] = problem->wUid;
    bound[std::string(DY_TOKEN)] = problem->dyUid;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(problem->base.dataType));
    bound[std::string(DIRECTION_TOKEN)] = std::string("BWD");
    const auto flops = pack_features::checkedMultiply(2, problem->base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

/// The launch @p kernel runs @p context's graph with, or nullopt when the config does not serve
/// it on this device (or its metadata is malformed).
std::optional<StemBwdLaunch> launchOf(const MatchContext& context, const KernelDefinition& kernel)
{
    try
    {
        if(kernel.getStringMetadata("direction") != "BWD")
        {
            return std::nullopt;
        }
        const auto problem = conv_stem_bwd::parseStemBwdProblem(context);
        if(!problem.has_value()
           || kernel.getStringMetadata("dtype")
                  != data_objects::EnumNameDataType(problem->base.dataType))
        {
            return std::nullopt;
        }
        return conv_stem_bwd::stemBwdLaunchFor(
            *problem, configOf(kernel), context.deviceProperties);
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
const std::vector<KernelArgument>& mainSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvStemBwdArgs)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& transformSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvStemBwdTransformArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvStemBwd : public IngestorPreparedDispatch
{
public:
    PreparedConvStemBwd(IngestorKernelCode main,
                        IngestorKernelCode transform,
                        const StemBwdLaunch& launch,
                        const StemBwdProblem& problem)
        : _main(std::move(main))
        , _transform(std::move(transform))
        , _args(launch.args)
        , _transformArgs(launch.transform)
        , _dxUid(problem.dxUid)
        , _wUid(problem.wUid)
        , _dyUid(problem.dyUid)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _main;
    }

    compilation::IRunnableKernel& mainForStream(hipStream_t stream) const
    {
        return _main.kernelForStream(stream);
    }

    compilation::IRunnableKernel& transformForStream(hipStream_t stream) const
    {
        return _transform.kernelForStream(stream);
    }

    void argsFor(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace,
                 ConvStemBwdArgs& args,
                 ConvStemBwdTransformArgs& transform) const
    {
        args = _args;
        transform = _transformArgs;
        transform.w
            = hipdnn_plugin_sdk::findDeviceBuffer(_wUid, deviceBuffers, numDeviceBuffers).ptr;
        transform.b = workspace;
        args.b = workspace;
        args.dy = hipdnn_plugin_sdk::findDeviceBuffer(_dyUid, deviceBuffers, numDeviceBuffers).ptr;
        args.dx = hipdnn_plugin_sdk::findDeviceBuffer(_dxUid, deviceBuffers, numDeviceBuffers).ptr;
    }

private:
    IngestorKernelCode _main;
    IngestorKernelCode _transform;
    ConvStemBwdArgs _args;
    ConvStemBwdTransformArgs _transformArgs;
    int64_t _dxUid;
    int64_t _wUid;
    int64_t _dyUid;
};

class ConvStemBwdDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvStemBwdDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                               const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The transformed filter.
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
        const auto problem = conv_stem_bwd::parseStemBwdProblem(context);
        const auto dx = tryGetBoundInt(bound, DX_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto dy = tryGetBoundInt(bound, DY_TOKEN);
        if(!launch.has_value() || !problem.has_value() || !dx.has_value() || !w.has_value()
           || !dy.has_value() || *dx != problem->dxUid || *w != problem->wUid
           || *dy != problem->dyUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_stem_bwd dispatch was given a graph or config its matchers did not admit");
        }
        const auto config = configOf(kernel);

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        for(const auto& [name, value] :
            conv_stem_bwd::compileDefinesFor(problem->base.dataType, config, launch->kernelType))
        {
            options.add(name, value);
        }

        auto main = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, mainSignature());
        KernelDefinition transformKernel = kernel;
        // The transform is the same source under another entry point.
        transformKernel.source.entryPoint = std::string(TRANSFORM_ENTRY_POINT);
        auto transform = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, transformKernel, options, transformSignature());
        main.setBlockSize(launch->block, 1, 1);
        main.setGridSize(launch->gridX, launch->gridY, launch->gridZ);
        transform.setBlockSize(launch->transformBlock, 1, 1);
        transform.setGridSize(launch->transformGrid, 1, 1);
        return std::make_unique<PreparedConvStemBwd>(
            std::move(main), std::move(transform), *launch, *problem);
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
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_stem_bwd needs its filter workspace");
        }
        const auto& dispatch = dynamic_cast<const PreparedConvStemBwd&>(prepared);
        ConvStemBwdArgs args{};
        ConvStemBwdTransformArgs transform{};
        dispatch.argsFor(deviceBuffers, numDeviceBuffers, workspace, args, transform);
        const auto stream = handle.getStream();
        dispatch.transformForStream(stream).launch(stream, transform);
        dispatch.mainForStream(stream).launch(stream, args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convStemBwdKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvStemBwdDispatchHandler& convStemBwdDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convStemBwdKpackModuleCache());
    static const ConvStemBwdDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvStemBwdSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convStemBwdDispatchHandler());
}

void resetConvStemBwdModuleCache()
{
    convStemBwdKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
