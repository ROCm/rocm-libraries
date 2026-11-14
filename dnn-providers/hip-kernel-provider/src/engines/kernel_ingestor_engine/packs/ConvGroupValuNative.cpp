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
#include "engines/kernel_ingestor_engine/descriptors/conv/group_valu/kernels/ConvGroupValuArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvGroupValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvGroupValuNative.cpp
 * @brief The native half of CONV_GROUP_VALU_FWD / CONV_GROUP_VALU_BWD: graph matching, kernel
 *        matching, workspace sizing and dispatch of the prepare and main kernels
 *        (ConvGroupValu.cpp). The workspace holds the prepared fp32 filter; both launches are in
 *        stream order and every output element is written by one lane: deterministic.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace gv = conv_group_valu;
using conv_naive::ConvDirection;

namespace
{

constexpr std::string_view FWD_GRAPH_SYMBOL = "hipkernel.conv_group_valu_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_SYMBOL = "hipkernel.conv_group_valu_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_group_valu.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_group_valu.dispatch";

constexpr std::string_view X_TOKEN = "conv_group_valu.x.uid";
constexpr std::string_view W_TOKEN = "conv_group_valu.w.uid";
constexpr std::string_view Y_TOKEN = "conv_group_valu.y.uid";
constexpr std::string_view FLOPS_TOKEN = "conv_group_valu.flops";

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
                                                   "conv_group_valu: unsupported element type");
}

std::optional<ConvDirection> directionOf(const std::string& name)
{
    if(name == "FWD")
    {
        return ConvDirection::FORWARD;
    }
    if(name == "BWD")
    {
        return ConvDirection::BACKWARD_DATA;
    }
    return std::nullopt;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = gv::parseGvProblem(context, direction);
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

struct Resolved
{
    gv::GvProblem problem;
    gv::Plan plan;
};

std::optional<Resolved> resolve(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto direction = directionOf(kernel.getStringMetadata("direction"));
    if(!direction.has_value())
    {
        return std::nullopt;
    }
    auto problem = gv::parseGvProblem(context, *direction);
    if(!problem.has_value()
       || kernel.getStringMetadata("dtype")
              != data_objects::EnumNameDataType(problem->base.dataType))
    {
        return std::nullopt;
    }
    const auto config = gv::configOf(kernel);
    if(!config.has_value())
    {
        return std::nullopt;
    }
    const int wave
        = gv::planWave(context.deviceProperties.gcnArchName, context.deviceProperties.warpSize);
    auto plan = gv::planFor(
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
        {"by_value", static_cast<uint32_t>(sizeof(ConvGroupValuArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvGroupValu : public IngestorPreparedDispatch
{
public:
    PreparedConvGroupValu(std::vector<IngestorKernelCode> kernels, Resolved resolved)
        : _kernels(std::move(kernels))
        , _resolved(std::move(resolved))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _kernels.back();
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

class ConvGroupValuDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvGroupValuDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
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
                "conv_group_valu dispatch was given a graph or config its matchers did not "
                "admit");
        }
        const auto& plan = resolved->plan;
        const auto& config = plan.config;
        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("GV_DT", dtypeCode(resolved->problem.base.dataType));
        options.add("GV_LAYOUT", config.layout == "nchw" ? 1 : 0);
        options.add("GV_KB", config.kb);
        options.add("GV_CB", config.cb);
        options.add("GV_P", config.p);
        options.add("GV_FD", config.fd);
        options.add("GV_FR", config.fr);
        options.add("GV_FS", config.fs);
        // Forward strides through x along the width; backward data reads dy upsampled.
        const bool bwd = config.direction == "BWD";
        options.add("GV_SW", bwd ? 1 : config.sw);
        options.add("GV_US", bwd ? config.sw : 1);
        options.add("GV_UPROWS", plan.args.uh > 1 || plan.args.ud > 1 ? 1 : 0);
        options.add("GV_DW", config.dw);
        options.add("GV_PW", plan.kernelPw);
        options.add("GV_NW", config.nw);
        options.add("GV_WAVE", plan.wave);
        options.add("GV_VBX", plan.vbx);
        options.add("GV_VBY", plan.vby);
        options.add("GV_RAGGED", plan.ragged ? 1 : 0);
        options.add("GV_COLMASK", plan.colmask ? 1 : 0);

        struct Launch
        {
            const char* entry;
            unsigned gridX;
            unsigned gridY;
            unsigned threads;
        };
        const std::vector<Launch> launches
            = {{"ConvGroupValuPrepare", static_cast<unsigned>(plan.prepareGrid), 1, 256},
               {"ConvGroupValu",
                static_cast<unsigned>(plan.args.rowBlocks),
                static_cast<unsigned>(plan.args.groups * plan.args.kBlocks),
                static_cast<unsigned>(config.nw * plan.wave)}};
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
        return std::make_unique<PreparedConvGroupValu>(std::move(kernels), std::move(*resolved));
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
                "conv_group_valu needs its prepared-filter workspace");
        }
        const auto& dispatch = dynamic_cast<const PreparedConvGroupValu&>(prepared);
        const auto& resolved = dispatch.resolved();
        const auto& uids = resolved.problem.base.operands;
        const bool bwd = resolved.problem.direction == ConvDirection::BACKWARD_DATA;
        void* xPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.x, deviceBuffers, numDeviceBuffers).ptr;
        void* wPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.w, deviceBuffers, numDeviceBuffers).ptr;
        void* yPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(uids.y, deviceBuffers, numDeviceBuffers).ptr;
        ConvGroupValuArgs args = resolved.plan.args;
        args.in = bwd ? yPtr : xPtr;
        args.out = bwd ? xPtr : yPtr;
        args.w = wPtr;
        args.wt = static_cast<float*>(workspace);
        const auto stream = handle.getStream();
        dispatch.kernel(0, stream).launch(stream, args);
        dispatch.kernel(1, stream).launch(stream, args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convGroupValuKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvGroupValuDispatchHandler& convGroupValuDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convGroupValuKpackModuleCache());
    static const ConvGroupValuDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

std::optional<BoundTokens> graphMatchesFwd(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::FORWARD);
}
std::optional<BoundTokens> graphMatchesBwd(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::BACKWARD_DATA);
}

} // namespace

void registerConvGroupValuSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_SYMBOL), &graphMatchesFwd);
    scope.add(std::string(BWD_GRAPH_SYMBOL), &graphMatchesBwd);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convGroupValuDispatchHandler());
}

void resetConvGroupValuModuleCache()
{
    convGroupValuKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
