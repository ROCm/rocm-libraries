// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
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
#include "engines/kernel_ingestor_engine/descriptors/conv/group_valu_wrw/kernels/ConvGroupValuWrwArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvGroupValuWrwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvGroupValuWrwNative.cpp
 * @brief The native half of CONV_GROUP_VALU_WRW: graph matching, kernel matching, workspace
 *        sizing and dispatch of the main and fold kernels (ConvGroupValuWrw.cpp).
 *
 * One launch with one split (the main kernel writes dw); otherwise the main kernel writes each
 * split's fp32 sums to the workspace and the fold kernel adds them in split order. Every launch
 * is in stream order and no sum is shared between threads, so the result is the same on every
 * run.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace gvw = conv_group_valu_wrw;

namespace
{

constexpr std::string_view GRAPH_SYMBOL = "hipkernel.conv_group_valu_wrw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_group_valu_wrw.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_group_valu_wrw.dispatch";

constexpr std::string_view X_TOKEN = "conv_group_valu_wrw.x.uid";
constexpr std::string_view W_TOKEN = "conv_group_valu_wrw.w.uid";
constexpr std::string_view Y_TOKEN = "conv_group_valu_wrw.y.uid";
constexpr std::string_view FLOPS_TOKEN = "conv_group_valu_wrw.flops";

int dtypeCode(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return 0;
    case data_objects::DataType::HALF:
        return 1;
    case data_objects::DataType::BFLOAT16:
        return 2;
    default:
        break;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "conv_group_valu_wrw: unsupported element type");
}

std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    const auto problem = gvw::parseGvwProblem(context);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = problem->base.operands.x;
    bound[std::string(W_TOKEN)] = problem->base.operands.w;
    bound[std::string(Y_TOKEN)] = problem->base.operands.y;
    const auto flops = pack_features::checkedMultiply(2, problem->base.multiplyAccumulates);
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }
    return bound;
}

/// The problem and plan a kernel runs, or nullopt when it does not serve the graph.
struct Resolved
{
    gvw::GvwProblem problem;
    gvw::Plan plan;
};

std::optional<Resolved> resolve(const MatchContext& context, const KernelDefinition& kernel)
{
    auto problem = gvw::parseGvwProblem(context);
    if(!problem.has_value()
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->base.dataType))
    {
        return std::nullopt;
    }
    const auto config = gvw::configOf(kernel);
    if(!config.has_value())
    {
        return std::nullopt;
    }
    const int wave
        = gvw::waveOf(context.deviceProperties.gcnArchName, context.deviceProperties.warpSize);
    auto plan = gvw::planFor(
        *problem, *config, wave, device_units::dispatchUnits(context.deviceProperties));
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    return Resolved{*problem, std::move(*plan)};
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return resolve(context, kernel).has_value();
}

const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvGroupValuWrwArgs)), 0, ""}};
    return s_signature;
}

/// The main launch and, with several splits, the fold launch.
class PreparedConvGroupValuWrw : public IngestorPreparedDispatch
{
public:
    PreparedConvGroupValuWrw(std::vector<IngestorKernelCode> kernels, Resolved resolved)
        : _kernels(std::move(kernels))
        , _resolved(std::move(resolved))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _kernels.front();
    }

    compilation::IRunnableKernel& kernel(size_t index, hipStream_t stream) const
    {
        return _kernels.at(index).kernelForStream(stream);
    }

    const Resolved& resolved() const
    {
        return _resolved;
    }

private:
    std::vector<IngestorKernelCode> _kernels;
    Resolved _resolved;
};

class ConvGroupValuWrwDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvGroupValuWrwDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
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
        auto resolved = resolve(context, kernel);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!resolved.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != resolved->problem.base.operands.x || *w != resolved->problem.base.operands.w
           || *y != resolved->problem.base.operands.y)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_group_valu_wrw dispatch was given a graph or config its matchers did not "
                "admit");
        }
        const auto& plan = resolved->plan;
        const auto& config = plan.config;
        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("GVW_DT", dtypeCode(resolved->problem.base.dataType));
        options.add("GVW_LAYOUT", config.layout == "nchw" ? 1 : 0);
        options.add("GVW_CB", config.cb);
        options.add("GVW_KB", config.kb);
        options.add("GVW_P", config.p);
        options.add("GVW_FR", config.fr);
        options.add("GVW_FS", config.fs);
        options.add("GVW_SW", config.sw);
        options.add("GVW_DW", config.dw);
        options.add("GVW_PW", config.pw);
        options.add("GVW_NW", config.nw);
        options.add("GVW_WAVE", plan.wave);
        options.add("GVW_VBX", plan.vbx);
        options.add("GVW_VBY", plan.vby);
        options.add("GVW_RAGGED", plan.ragged ? 1 : 0);
        options.add("GVW_COLMASK", plan.colmask ? 1 : 0);
        options.add("GVW_DOT2", config.dot2);
        options.add("GVW_RING", config.ring);
        // The depth code only where there is a depth: 2-D problems compile without it.
        const auto& problem = resolved->problem;
        options.add("GVW_3D", problem.t > 1 || problem.di > 1 || problem.dOut > 1 ? 1 : 0);

        struct Launch
        {
            const char* entry;
            unsigned gridX;
            unsigned gridY;
            unsigned threads;
        };
        std::vector<Launch> launches = {{"ConvGroupValuWrw",
                                         static_cast<unsigned>(plan.args.tiles),
                                         static_cast<unsigned>(plan.args.splits),
                                         static_cast<unsigned>(config.nw * plan.wave)}};
        if(plan.args.splits > 1)
        {
            launches.push_back(
                {"ConvGroupValuWrwFold", static_cast<unsigned>(plan.foldGrid), 1, 256});
        }
        std::vector<IngestorKernelCode> kernels;
        for(const auto& launch : launches)
        {
            KernelDefinition entry = kernel;
            entry.source.entryPoint = launch.entry;
            auto code = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, entry, options, kernelSignature());
            code.setBlockSize(launch.threads, 1, 1);
            code.setGridSize(launch.gridX, launch.gridY, 1);
            kernels.push_back(std::move(code));
        }
        return std::make_unique<PreparedConvGroupValuWrw>(std::move(kernels), std::move(*resolved));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvGroupValuWrw&>(prepared);
        const auto& resolved = dispatch.resolved();
        const auto& plan = resolved.plan;
        const auto& uids = resolved.problem.base.operands;
        if(plan.args.splits > 1 && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "conv_group_valu_wrw needs its workspace for the splits' sums");
        }
        ConvGroupValuWrwArgs args = plan.args;
        args.x = hipdnn_plugin_sdk::findDeviceBuffer(uids.x, deviceBuffers, numDeviceBuffers).ptr;
        args.dw = hipdnn_plugin_sdk::findDeviceBuffer(uids.w, deviceBuffers, numDeviceBuffers).ptr;
        args.dy = hipdnn_plugin_sdk::findDeviceBuffer(uids.y, deviceBuffers, numDeviceBuffers).ptr;
        args.partial = static_cast<float*>(workspace);
        const auto stream = handle.getStream();
        dispatch.kernel(0, stream).launch(stream, args);
        if(plan.args.splits > 1)
        {
            dispatch.kernel(1, stream).launch(stream, args);
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convGroupValuWrwKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvGroupValuWrwDispatchHandler& convGroupValuWrwDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convGroupValuWrwKpackModuleCache());
    static const ConvGroupValuWrwDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvGroupValuWrwSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_SYMBOL), &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convGroupValuWrwDispatchHandler());
}

void resetConvGroupValuWrwModuleCache()
{
    convGroupValuWrwKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
