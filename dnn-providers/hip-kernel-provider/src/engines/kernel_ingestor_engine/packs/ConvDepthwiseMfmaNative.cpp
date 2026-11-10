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
#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseMfmaProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvDepthwiseMfmaNative.cpp
 * @brief The native half of the matrix-core depthwise engines (CONV_DEPTHWISE_MFMA_FWD,
 *        CONV_DEPTHWISE_MFMA_BWD): graph matching, kernel matching and dispatch.
 *
 * The kernel (descriptors/conv/depthwise_mfma/kernels) runs 2-D depthwise convolution at stride
 * 1 or 2, fp16 and bf16, on CDNA matrix cores, with a ring of per-filter-row accumulators
 * streaming down the image. Channels-last: the filter row as a one-hot diagonal MFMA operand over
 * 8 channels, the input row staged once per block through LDS. Channels-first: channel pairs
 * along the MFMA's K, each wave loading its own rows. Backward data at unit stride is the forward
 * convolution of dy with the filter reflected. Every catalog config is admitted on correctness
 * alone (mfmaLaunchFor, mfmaRunsOn); which one runs is the ranking's: autotune, and
 * UHD once trained.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_depthwise_mfma::MfmaConfig;
using conv_depthwise_mfma::MfmaLaunch;
using conv_depthwise_valu::DepthwiseLayout;
using conv_depthwise_valu::DepthwiseValuProblem;
using conv_naive::ConvDirection;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL
    = "hipkernel.conv_depthwise_mfma_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL
    = "hipkernel.conv_depthwise_mfma_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_depthwise_mfma.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_depthwise_mfma.dispatch";

// Tokens the graph match binds.
constexpr std::string_view IN_TOKEN = "conv_depthwise_mfma.in.uid";
constexpr std::string_view FILTER_TOKEN = "conv_depthwise_mfma.filter.uid";
constexpr std::string_view OUT_TOKEN = "conv_depthwise_mfma.out.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_depthwise_mfma.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_depthwise_mfma.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_depthwise_mfma.flops";

std::string elementTypeFor(const std::string& dtype)
{
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
        "conv_depthwise_mfma kernel declares unsupported dtype '" + dtype + "'");
}

std::optional<ConvDirection> directionOf(const KernelDefinition& kernel)
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

/// The config @p kernel's metadata declares. Throws when a field is missing or mistyped.
MfmaConfig configOf(const KernelDefinition& kernel)
{
    MfmaConfig config;
    config.layout = kernel.getStringMetadata("layout");
    config.atom = kernel.getIntMetadata("atom");
    config.kh = kernel.getIntMetadata("kh");
    config.kw = kernel.getIntMetadata("kw");
    config.stride = kernel.getIntMetadata("stride");
    config.waves = kernel.getIntMetadata("waves");
    config.depth = kernel.getIntMetadata("depth");
    config.fold = kernel.getIntMetadata("fold");
    config.rows = kernel.getIntMetadata("rows");
    return config;
}

std::optional<DepthwiseValuProblem> problemFor(const MatchContext& context, ConvDirection direction)
{
    auto problem = conv_depthwise_valu::parseDepthwiseValuProblem(context, direction);
    // 1-D (as height 1) and 2-D, either layout, unit dilation, stride 1 on both axes or (forward)
    // 2 on both (backward data: its unit-stride forward form), 16-bit.
    const auto stridesServed = [](const ConvDepthwiseValuArgs& a) {
        return a.dh == 1 && a.dw == 1 && a.sh == a.sw && (a.sh == 1 || a.sh == 2);
    };
    if(!problem.has_value() || problem->ndims != 2 || problem->gatherOnly
       || !stridesServed(problem->args)
       || (problem->base.dataType != data_objects::DataType::HALF
           && problem->base.dataType != data_objects::DataType::BFLOAT16))
    {
        return std::nullopt;
    }
    return problem;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = problemFor(context, direction);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    BoundTokens bound;
    bound[std::string(IN_TOKEN)] = problem->inUid;
    bound[std::string(FILTER_TOKEN)] = problem->filterUid;
    bound[std::string(OUT_TOKEN)] = problem->outUid;
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

/// The launch @p kernel runs @p context's graph with, or nullopt when the config does not
/// serve it (or its metadata is malformed).
std::optional<MfmaLaunch> launchOf(const MatchContext& context, const KernelDefinition& kernel)
{
    try
    {
        const auto direction = directionOf(kernel);
        if(!direction.has_value())
        {
            return std::nullopt;
        }
        const auto problem = problemFor(context, *direction);
        if(!problem.has_value()
           || kernel.getStringMetadata("dtype")
                  != data_objects::EnumNameDataType(problem->base.dataType))
        {
            return std::nullopt;
        }
        const auto config = configOf(kernel);
        if(!conv_depthwise_mfma::mfmaRunsOn(config.atom,
                                            context.deviceProperties.gcnArchName,
                                            problem->base.dataType
                                                == data_objects::DataType::BFLOAT16,
                                            config.layout))
        {
            return std::nullopt;
        }
        return conv_depthwise_mfma::mfmaLaunchFor(*problem, config);
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

/// Kernel-scoped applicability: the config computes this problem correctly and launches.
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
        {"by_value", static_cast<uint32_t>(sizeof(ConvDepthwiseMfmaArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvDepthwiseMfma : public IngestorPreparedDispatch
{
public:
    PreparedConvDepthwiseMfma(IngestorKernelCode code,
                              ConvDepthwiseMfmaArgs args,
                              int64_t inUid,
                              int64_t filterUid,
                              int64_t outUid)
        : _code(std::move(code))
        , _args(args)
        , _inUid(inUid)
        , _filterUid(filterUid)
        , _outUid(outUid)
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

    ConvDepthwiseMfmaArgs argsFor(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                  uint32_t numDeviceBuffers) const
    {
        ConvDepthwiseMfmaArgs args = _args;
        args.in = hipdnn_plugin_sdk::findDeviceBuffer(_inUid, deviceBuffers, numDeviceBuffers).ptr;
        args.filter
            = hipdnn_plugin_sdk::findDeviceBuffer(_filterUid, deviceBuffers, numDeviceBuffers).ptr;
        args.out
            = hipdnn_plugin_sdk::findDeviceBuffer(_outUid, deviceBuffers, numDeviceBuffers).ptr;
        return args;
    }

private:
    IngestorKernelCode _code;
    ConvDepthwiseMfmaArgs _args;
    int64_t _inUid;
    int64_t _filterUid;
    int64_t _outUid;
};

class ConvDepthwiseMfmaDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvDepthwiseMfmaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                     const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto launch = launchOf(context, kernel);
        const auto direction = directionOf(kernel);
        const auto problem = direction.has_value() ? problemFor(context, *direction)
                                                   : std::optional<DepthwiseValuProblem>{};
        const auto in = tryGetBoundInt(bound, IN_TOKEN);
        const auto filter = tryGetBoundInt(bound, FILTER_TOKEN);
        const auto out = tryGetBoundInt(bound, OUT_TOKEN);
        if(!launch.has_value() || !problem.has_value() || !in.has_value() || !filter.has_value()
           || !out.has_value() || *in != problem->inUid || *filter != problem->filterUid
           || *out != problem->outUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_depthwise_mfma dispatch was given a graph or config its matchers did not "
                "admit");
        }
        const auto config = configOf(kernel);

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_DWM_TYPE", elementTypeFor(kernel.getStringMetadata("dtype")));
        options.add("CONV_DWM_FLIP", std::string(launch->flip ? "1" : "0"));
        options.add("CONV_DWM_LAYOUT", std::string(config.layout == "NHWC" ? "0" : "1"));
        const std::array<std::pair<const char*, int64_t>, 7> fields{{{"ATOM", config.atom},
                                                                     {"KH", config.kh},
                                                                     {"KW", config.kw},
                                                                     {"STRIDE", config.stride},
                                                                     {"WAVES", config.waves},
                                                                     {"DEPTH", config.depth},
                                                                     {"FOLD", config.fold}}};
        for(const auto& [name, value] : fields)
        {
            options.add(std::string("CONV_DWM_") + name, std::to_string(value));
        }

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, kernelSignature());
        code.setBlockSize(static_cast<unsigned int>(launch->block), 1, 1);
        code.setGridSize(static_cast<unsigned int>(launch->grid), 1, 1);
        return std::make_unique<PreparedConvDepthwiseMfma>(
            std::move(code), launch->args, problem->inUid, problem->filterUid, problem->outUid);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvDepthwiseMfma&>(prepared);
        const auto args = dispatch.argsFor(deviceBuffers, numDeviceBuffers);
        dispatch.kernelForStream(handle.getStream()).launch(handle.getStream(), args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convDepthwiseMfmaKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvDepthwiseMfmaDispatchHandler& convDepthwiseMfmaDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convDepthwiseMfmaKpackModuleCache());
    static const ConvDepthwiseMfmaDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                    s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvDepthwiseMfmaSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convDepthwiseMfmaDispatchHandler());
}

void resetConvDepthwiseMfmaModuleCache()
{
    convDepthwiseMfmaKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
