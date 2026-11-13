// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvDepthwiseValuNative.cpp
 * @brief The native half of the VALU depthwise convolution engines (CONV_DEPTHWISE_VALU_FWD,
 *        CONV_DEPTHWISE_VALU_BWD): graph matching, kernel matching and dispatch.
 *
 * The kernels are hipconv's hand-written VALU depthwise cores (descriptors/conv/depthwise_valu/
 * kernels). Every kernel config in the catalog is a candidate: its kernel match admits it on
 * correctness alone -- layout, rank, element type and, for the halo and LDS variants, unit
 * stride and dilation and their compile-time filter size. Which config runs is the ranking's:
 * catalog order (the floors first) until autotune measures, and UHD once trained. No
 * shape-to-config routing is built in.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_depthwise_valu::DepthwiseLayout;
using conv_depthwise_valu::DepthwiseTile;
using conv_depthwise_valu::DepthwiseValuProblem;
using conv_depthwise_valu::DepthwiseVariant;
using conv_naive::ConvDirection;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL
    = "hipkernel.conv_depthwise_valu_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL
    = "hipkernel.conv_depthwise_valu_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_depthwise_valu.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_depthwise_valu.dispatch";

// Tokens the graph match binds.
constexpr std::string_view IN_TOKEN = "conv_depthwise_valu.in.uid";
constexpr std::string_view FILTER_TOKEN = "conv_depthwise_valu.filter.uid";
constexpr std::string_view OUT_TOKEN = "conv_depthwise_valu.out.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_depthwise_valu.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_depthwise_valu.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_depthwise_valu.flops";

std::string_view directionName(ConvDirection direction)
{
    return direction == ConvDirection::FORWARD ? "FWD" : "BWD";
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
        "conv_depthwise_valu kernel declares unsupported dtype '" + dtype + "'");
}

ConvDirection directionFor(const KernelDefinition& kernel)
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
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "conv_depthwise_valu kernel declares unsupported direction '" + direction + "'");
}

/// Bytes per element of @p dataType (the dtypes the depthwise kernels take).
int64_t elementBytes(data_objects::DataType dataType)
{
    return dataType == data_objects::DataType::FLOAT ? 4 : 2;
}

/// The tile @p kernel's metadata declares, or nullopt for an unknown variant.
std::optional<DepthwiseTile> tileOf(const KernelDefinition& kernel)
{
    const auto variant = conv_depthwise_valu::variantNamed(kernel.getStringMetadata("variant"));
    if(!variant.has_value())
    {
        return std::nullopt;
    }
    DepthwiseTile tile;
    tile.variant = *variant;
    tile.kd = kernel.getIntMetadata("kd");
    tile.kh = kernel.getIntMetadata("kh");
    tile.kw = kernel.getIntMetadata("kw");
    tile.wstrip = kernel.getIntMetadata("wstrip");
    tile.th = kernel.getIntMetadata("th");
    tile.tw = kernel.getIntMetadata("tw");
    tile.bd = kernel.getIntMetadata("bd");
    tile.bh = kernel.getIntMetadata("bh");
    tile.bw = kernel.getIntMetadata("bw");
    tile.bk = kernel.getIntMetadata("bk");
    tile.rd = kernel.getIntMetadata("rd");
    tile.rh = kernel.getIntMetadata("rh");
    tile.rw = kernel.getIntMetadata("rw");
    tile.stride = kernel.getIntMetadata("stride");
    tile.vec = kernel.getIntMetadata("vec");
    return tile;
}

std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_depthwise_valu::parseDepthwiseValuProblem(context, direction);
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
    bound[std::string(DIRECTION_TOKEN)] = std::string(directionName(direction));
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

/// Kernel-scoped applicability: the config computes this problem correctly and launches.
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    ConvDirection direction = ConvDirection::FORWARD;
    try
    {
        direction = directionFor(kernel);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return false;
    }
    const auto problem = conv_depthwise_valu::parseDepthwiseValuProblem(context, direction);
    const auto tile = tileOf(kernel);
    if(!problem.has_value() || !tile.has_value())
    {
        return false;
    }
    if(kernel.getStringMetadata("dtype") != data_objects::EnumNameDataType(problem->base.dataType)
       || kernel.getStringMetadata("layout") != conv_depthwise_valu::layoutName(problem->layout)
       || kernel.getIntMetadata("ndims") != problem->ndims)
    {
        return false;
    }
    // The phase blocks serve 2-D backward data at their stride (height and width) and filter,
    // unit dilation; channels-last vectors of at most 16 bytes, whole 32-bit words.
    if(tile->variant == DepthwiseVariant::PHASE)
    {
        const auto& g = problem->gatherArgs;
        const int64_t bytes = tile->vec * elementBytes(problem->base.dataType);
        const bool vecOk = problem->layout == DepthwiseLayout::CHANNELS_FIRST
                               ? tile->vec == 0 && tile->bk == 0
                               : tile->vec > 0 && bytes <= 16 && bytes % 4 == 0 && tile->bk > 0;
        return direction == ConvDirection::BACKWARD_DATA && problem->ndims == 2 && tile->stride >= 2
               && g.sh == tile->stride && g.sw == tile->stride && g.dh == 1 && g.dw == 1
               && g.kh == tile->kh && g.kw == tile->kw && vecOk
               && conv_depthwise_valu::launchFor(*problem, *tile).has_value();
    }
    // The gather serves backward data at any stride, and only it (and the phase blocks) a
    // strided one.
    if(tile->variant == DepthwiseVariant::GATHER || problem->gatherOnly)
    {
        return tile->variant == DepthwiseVariant::GATHER
               && direction == ConvDirection::BACKWARD_DATA
               && conv_depthwise_valu::launchFor(*problem, *tile).has_value();
    }
    if(tile->variant != DepthwiseVariant::FLOOR)
    {
        // The halo and LDS variants bake the filter and assume unit stride and dilation.
        const auto& a = problem->args;
        if(!problem->unitStrideAndDilation || tile->kh != a.kh || tile->kw != a.kw
           || (problem->ndims == 3 && tile->kd != a.kd))
        {
            return false;
        }
    }
    return conv_depthwise_valu::launchFor(*problem, *tile).has_value();
}

/// The single argument the kernel takes. Only consulted for a pre-built kernel.
const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvDepthwiseValuArgs)), 0, ""}};
    return s_signature;
}

class PreparedConvDepthwiseValu : public IngestorPreparedDispatch
{
public:
    PreparedConvDepthwiseValu(IngestorKernelCode code, const DepthwiseValuProblem& problem)
        : _code(std::move(code))
        , _problem(problem)
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

    const DepthwiseValuProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _code;
    DepthwiseValuProblem _problem;
};

class ConvDepthwiseValuDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    ConvDepthwiseValuDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
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
        const auto problem
            = conv_depthwise_valu::parseDepthwiseValuProblem(context, directionFor(kernel));
        const auto tile = tileOf(kernel);
        const auto in = tryGetBoundInt(bound, IN_TOKEN);
        const auto filter = tryGetBoundInt(bound, FILTER_TOKEN);
        const auto out = tryGetBoundInt(bound, OUT_TOKEN);
        if(!problem.has_value() || !tile.has_value() || !in.has_value() || !filter.has_value()
           || !out.has_value() || *in != problem->inUid || *filter != problem->filterUid
           || *out != problem->outUid)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_depthwise_valu dispatch was given a graph its matcher did not admit");
        }
        const auto launch = conv_depthwise_valu::launchFor(*problem, *tile);
        if(!launch.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_depthwise_valu dispatch was given a config its kernel match declines");
        }

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_DW_TYPE", elementTypeFor(kernel.getStringMetadata("dtype")));
        options.add("CONV_DW_VARIANT", std::to_string(static_cast<int>(tile->variant)));
        options.add("CONV_DW_LAYOUT",
                    std::string(problem->layout == DepthwiseLayout::CHANNELS_FIRST ? "1" : "0"));
        options.add("CONV_DW_NDIMS", std::to_string(problem->ndims));
        // The gather and the phase blocks run in gather terms (in = dy, out = dx, the forward
        // convolution's padding, stride and dilation).
        const bool gather
            = tile->variant == DepthwiseVariant::GATHER || tile->variant == DepthwiseVariant::PHASE;
        // The gather reads the filter as the forward convolution does: reflected only for
        // true convolution.
        options.add(
            "CONV_DW_FLIP",
            std::string((gather ? problem->convolutionMode : problem->flipFilter) ? "1" : "0"));
        const std::array<std::pair<const char*, int64_t>, 15> tileDefines{{{"WSTRIP", tile->wstrip},
                                                                           {"STRIDE", tile->stride},
                                                                           {"VEC", tile->vec},
                                                                           {"KD", tile->kd},
                                                                           {"KH", tile->kh},
                                                                           {"KW", tile->kw},
                                                                           {"TH", tile->th},
                                                                           {"TW", tile->tw},
                                                                           {"BD", tile->bd},
                                                                           {"BH", tile->bh},
                                                                           {"BW", tile->bw},
                                                                           {"BK", tile->bk},
                                                                           {"RD", tile->rd},
                                                                           {"RH", tile->rh},
                                                                           {"RW", tile->rw}}};
        // Only the fields the variant uses are set; the rest are 0 and keep the kernel defaults.
        for(const auto& [name, value] : tileDefines)
        {
            if(value > 0)
            {
                options.add(std::string("CONV_DW_") + name, std::to_string(value));
            }
        }

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, kernelSignature());
        code.setBlockSize(static_cast<unsigned int>(launch->block), 1, 1);
        code.setGridSize(static_cast<unsigned int>(launch->grid), 1, 1);
        auto prepared = *problem;
        if(gather)
        {
            prepared.args = prepared.gatherArgs;
        }
        return std::make_unique<PreparedConvDepthwiseValu>(std::move(code), prepared);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvDepthwiseValu&>(prepared);
        const auto& problem = dispatch.problem();
        ConvDepthwiseValuArgs args = problem.args;
        args.in
            = hipdnn_plugin_sdk::findDeviceBuffer(problem.inUid, deviceBuffers, numDeviceBuffers)
                  .ptr;
        args.filter = hipdnn_plugin_sdk::findDeviceBuffer(
                          problem.filterUid, deviceBuffers, numDeviceBuffers)
                          .ptr;
        args.out
            = hipdnn_plugin_sdk::findDeviceBuffer(problem.outUid, deviceBuffers, numDeviceBuffers)
                  .ptr;
        dispatch.kernelForStream(handle.getStream()).launch(handle.getStream(), args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convDepthwiseValuKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const ConvDepthwiseValuDispatchHandler& convDepthwiseValuDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convDepthwiseValuKpackModuleCache());
    static const ConvDepthwiseValuDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                    s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvDepthwiseValuSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convDepthwiseValuDispatchHandler());
}

void resetConvDepthwiseValuModuleCache()
{
    convDepthwiseValuKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
