// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
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
#include "engines/kernel_ingestor_engine/packs/ConvIgemmNchwProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvIgemmNchwNative.cpp
 * @brief The native half of the channels-first backward-data implicit-GEMM engine
 *        (CONV_IGEMM_NCHW_BWD, K7): graph matching, kernel matching and dispatch.
 *
 * The kernel (descriptors/conv/igemm_nchw/kernels) is an implicit GEMM with the col2im fused:
 * a block owns a run of dx positions covering every stride phase, so each product lands in its
 * own accumulator and dy, the K C S filter and dx are read and written in place, with no
 * staging and no workspace. Matrix-core configs reuse K1's atoms (MFMA on gfx908 to gfx950, WMMA
 * on gfx11 and gfx12); the FMA kernel runs everywhere. fp32, fp16 and bf16; shapes are runtime,
 * a config fixes the tile, the stride and the taps per phase. Every config is admitted on
 * correctness alone (nchwLaunchFor); autotune, and UHD once trained, ranks them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_igemm_nchw::NchwConfig;
using conv_igemm_nchw::NchwLaunch;
using conv_igemm_nchw::NchwProblem;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_igemm_nchw_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_igemm_nchw.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_igemm_nchw.dispatch";

// Tokens the graph match binds.
constexpr std::string_view DX_TOKEN = "conv_igemm_nchw.dx.uid";
constexpr std::string_view W_TOKEN = "conv_igemm_nchw.w.uid";
constexpr std::string_view DY_TOKEN = "conv_igemm_nchw.dy.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_igemm_nchw.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_igemm_nchw.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_igemm_nchw.flops";

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
                                                       "conv_igemm_nchw: unsupported element type");
    }
}

/// The config @p kernel's metadata declares. Throws when a field is missing or mistyped.
NchwConfig configOf(const KernelDefinition& kernel)
{
    NchwConfig config;
    config.atom = kernel.getStringMetadata("atom");
    config.atomId = kernel.getIntMetadata("atom_id");
    config.bmq = kernel.getIntMetadata("bmq");
    config.bn = kernel.getIntMetadata("bn");
    config.bk = kernel.getIntMetadata("bk");
    config.wm = kernel.getIntMetadata("wm");
    config.wn = kernel.getIntMetadata("wn");
    config.wp = kernel.getIntMetadata("wp");
    config.tq = kernel.getIntMetadata("tq");
    config.tc = kernel.getIntMetadata("tc");
    config.vec = kernel.getIntMetadata("vec");
    config.stride = kernel.getIntMetadata("stride");
    config.taps = kernel.getIntMetadata("taps");
    config.stages = kernel.getIntMetadata("stages");
    config.flat = kernel.getIntMetadata("flat");
    config.halo = kernel.getIntMetadata("halo");
    config.math = kernel.getIntMetadata("math");
    return config;
}

std::optional<BoundTokens> bwdGraphMatches(const MatchContext& context)
{
    const auto problem = conv_igemm_nchw::parseNchwProblem(context);
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
std::optional<NchwLaunch> launchOf(const MatchContext& context, const KernelDefinition& kernel)
{
    try
    {
        if(kernel.getStringMetadata("direction") != "BWD")
        {
            return std::nullopt;
        }
        const auto problem = conv_igemm_nchw::parseNchwProblem(context);
        if(!problem.has_value()
           || kernel.getStringMetadata("dtype")
                  != data_objects::EnumNameDataType(problem->base.dataType))
        {
            return std::nullopt;
        }
        const auto config = configOf(kernel);
        // The atom must run on this device and take the graph's element type (the FMA kernel
        // takes any).
        if(!conv_igemm::atomRunsOn(config.atom, context.deviceProperties.gcnArchName))
        {
            return std::nullopt;
        }
        // Three-bf16 (math 1) runs a bf16 atom on an fp32 graph: nchwLaunchFor gates it.
        const auto element = conv_igemm::atomElement(config.atom);
        if(config.math == 0 && element != "any"
           && element != conv_igemm::dataElement(problem->base.dataType))
        {
            return std::nullopt;
        }
        if(config.math != 0
           && !conv_igemm_nchw::threeBf16Pays(*problem, context.deviceProperties.gcnArchName))
        {
            return std::nullopt;
        }
        return conv_igemm_nchw::nchwLaunchFor(*problem, config);
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

/// The single argument the kernel takes. Only consulted for a pre-built kernel.
const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K7Args)), 0, ""}};
    return s_signature;
}

const std::vector<KernelArgument>& splitSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(K1Split3Args)), 0, ""}};
    return s_signature;
}

class PreparedConvIgemmNchw : public IngestorPreparedDispatch
{
public:
    /// @p split is set for three-bf16 (NchwLaunch::split) only.
    PreparedConvIgemmNchw(IngestorKernelCode code,
                          std::optional<IngestorKernelCode> split,
                          const NchwLaunch& launch,
                          const NchwProblem& problem)
        : _code(std::move(code))
        , _split(std::move(split))
        , _plan(launch.plan)
        , _args(launch.args)
        , _dxUid(problem.dxUid)
        , _wUid(problem.wUid)
        , _dyUid(problem.dyUid)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    K7Args argsFor(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                   uint32_t numDeviceBuffers,
                   void* workspace,
                   K1Split3Args& split) const
    {
        K7Args args = _args;
        args.dx = hipdnn_plugin_sdk::findDeviceBuffer(_dxUid, deviceBuffers, numDeviceBuffers).ptr;
        args.w = hipdnn_plugin_sdk::findDeviceBuffer(_wUid, deviceBuffers, numDeviceBuffers).ptr;
        args.dy = hipdnn_plugin_sdk::findDeviceBuffer(_dyUid, deviceBuffers, numDeviceBuffers).ptr;
        if(_split.has_value())
        {
            // Three-bf16: the kernel reads the split dy (3K channels) and w from the workspace.
            split = conv_tf32::splitArgsFor(_plan, args.dy, args.w, workspace);
            args.dy = split.t[0].dst;
            args.w = split.t[1].dst;
        }
        return args;
    }

    bool hasSplit() const
    {
        return _split.has_value();
    }

    compilation::IRunnableKernel& splitForStream(hipStream_t stream) const
    {
        return _split->kernelForStream(stream);
    }

private:
    IngestorKernelCode _code;
    std::optional<IngestorKernelCode> _split;
    conv_tf32::SplitPlan _plan;
    K7Args _args;
    int64_t _dxUid;
    int64_t _wUid;
    int64_t _dyUid;
};

class ConvIgemmNchwDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvIgemmNchwDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                 const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// None, or (three-bf16) the split dy and w.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto launch = launchOf(context, kernel);
        return launch.has_value() && launch->split ? launch->plan.bytes : 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto launch = launchOf(context, kernel);
        const auto problem = conv_igemm_nchw::parseNchwProblem(context);
        const auto dx = tryGetBoundInt(bound, DX_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto dy = tryGetBoundInt(bound, DY_TOKEN);
        if(!launch.has_value() || !problem.has_value() || !dx.has_value() || !w.has_value()
           || !dy.has_value() || *dx != problem->dxUid || *w != problem->wUid
           || *dy != problem->dyUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_igemm_nchw dispatch was given a graph or config its matchers did not "
                "admit");
        }
        const auto config = configOf(kernel);
        const bool fma = launch->kernelType.empty();

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        if(launch->split)
        {
            // Three-bf16: the bf16 kernel writing fp32 dx, plus the split pre-pass.
            options.add("CONV_K7_T", std::string("unsigned short"));
            options.add("CONV_K7_TO", std::string("float"));
            options.add(std::string(conv_tf32::SPLIT_DEFINE), std::string("1"));
        }
        else
        {
            options.add("CONV_K7_T", elementTypeFor(problem->base.dataType));
        }
        options.add("CONV_K7_FMA", std::string(fma ? "1" : "0"));
        if(!fma)
        {
            options.add("CONV_K7_ATOM", std::string(launch->kernelType));
        }
        const std::array<std::pair<const char*, int64_t>, 14> fields{{{"STAGES", config.stages},
                                                                      {"FLAT", config.flat},
                                                                      {"HALO", config.halo},
                                                                      {"BMQ", config.bmq},
                                                                      {"BN", config.bn},
                                                                      {"BK", config.bk},
                                                                      {"WM", config.wm},
                                                                      {"WN", config.wn},
                                                                      {"WP", config.wp},
                                                                      {"TQ", config.tq},
                                                                      {"TC", config.tc},
                                                                      {"VEC", config.vec},
                                                                      {"STRIDE", config.stride},
                                                                      {"TAPS", config.taps}}};
        for(const auto& [name, value] : fields)
        {
            options.add(std::string("CONV_K7_") + name, std::to_string(value));
        }

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, kernelSignature());
        code.setBlockSize(launch->block, 1, 1);
        code.setGridSize(launch->gridX, launch->gridY, launch->gridZ);
        std::optional<IngestorKernelCode> split;
        if(launch->split)
        {
            KernelDefinition splitKernel = kernel;
            splitKernel.source.entryPoint = std::string(conv_tf32::SPLIT_ENTRY_POINT);
            split = buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, splitKernel, options, splitSignature());
            split->setBlockSize(256, 1, 1);
            split->setGridSize(launch->plan.grid, 1, 1);
        }
        return std::make_unique<PreparedConvIgemmNchw>(
            std::move(code), std::move(split), *launch, *problem);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvIgemmNchw&>(prepared);
        if(dispatch.hasSplit() && workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "conv_igemm_nchw three-bf16 needs its workspace");
        }
        K1Split3Args split{};
        const auto args = dispatch.argsFor(deviceBuffers, numDeviceBuffers, workspace, split);
        const auto stream = handle.getStream();
        if(dispatch.hasSplit())
        {
            dispatch.splitForStream(stream).launch(stream, split);
        }
        dispatch.kernelForStream(stream).launch(stream, args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convIgemmNchwKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvIgemmNchwDispatchHandler& convIgemmNchwDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convIgemmNchwKpackModuleCache());
    static const ConvIgemmNchwDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvIgemmNchwSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convIgemmNchwDispatchHandler());
}

void resetConvIgemmNchwModuleCache()
{
    convIgemmNchwKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
