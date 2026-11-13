// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
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
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file ConvNaiveNative.cpp
 * @brief The native half of the naive convolution engines (CONV_NAIVE_FWD, CONV_NAIVE_BWD,
 *        CONV_NAIVE_WRW): graph matching, kernel matching and dispatch.
 *
 * The engines are the coverage floor of the conv engine group: one HIPRTC-compiled kernel
 * per direction and element type, correct for every layout, group count, padding, stride,
 * dilation and spatial rank, and accepted only for problems small enough that its speed
 * does not matter (conv_naive::withinWorkloadBudget).
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;
using conv_naive::ConvNaiveProblem;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view FWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_naive_fwd.graph_match";
constexpr std::string_view BWD_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_naive_bwd.graph_match";
constexpr std::string_view WRW_GRAPH_MATCHER_SYMBOL = "hipkernel.conv_naive_wrw.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_naive.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_naive.dispatch";

// KMD fields: every kernel is baked for one element type and one direction.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view DIRECTION_FIELD = "direction";

// Tokens the graph match binds: the operand uids in forward roles, the graph's element type
// and direction in the KMD's vocabulary, and the op-intrinsic cost fields.
constexpr std::string_view X_TOKEN = "conv_naive.x.uid";
constexpr std::string_view W_TOKEN = "conv_naive.w.uid";
constexpr std::string_view Y_TOKEN = "conv_naive.y.uid";
constexpr std::string_view DTYPE_TOKEN = "conv_naive.dtype";
constexpr std::string_view DIRECTION_TOKEN = "conv_naive.direction";
constexpr std::string_view FLOPS_TOKEN = "conv_naive.flops";

constexpr unsigned int BLOCK_SIZE = 256;
/// Blocks per launch at most; the kernel strides over the rest of the elements.
constexpr int64_t MAX_GRID_BLOCKS = int64_t{1} << 20;

std::string_view directionName(ConvDirection direction)
{
    switch(direction)
    {
    case ConvDirection::FORWARD:
        return "FWD";
    case ConvDirection::BACKWARD_DATA:
        return "BWD";
    case ConvDirection::BACKWARD_WEIGHTS:
        return "WRW";
    default:
        break;
    }
    return "";
}

/// The element type's name in the kernel source, from the KMD's dtype spelling.
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
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "conv_naive kernel declares unsupported dtype '"
                                                       + dtype + "'");
}

/// Graph-scoped applicability for one direction: a single convolution node of that
/// direction that the naive kernel computes correctly and quickly enough.
std::optional<BoundTokens> graphMatches(const MatchContext& context, ConvDirection direction)
{
    const auto problem = conv_naive::parseConvNaiveProblem(context, direction);
    if(!problem.has_value()
       || !conv_naive::withinWorkloadBudget(*problem, context.deviceProperties))
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(X_TOKEN)] = problem->operands.x;
    bound[std::string(W_TOKEN)] = problem->operands.w;
    bound[std::string(Y_TOKEN)] = problem->operands.y;
    bound[std::string(DTYPE_TOKEN)]
        = std::string(data_objects::EnumNameDataType(problem->dataType));
    bound[std::string(DIRECTION_TOKEN)] = std::string(directionName(direction));

    const auto flops = pack_features::checkedMultiply(2, problem->multiplyAccumulates);
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

std::optional<BoundTokens> wrwGraphMatches(const MatchContext& context)
{
    return graphMatches(context, ConvDirection::BACKWARD_WEIGHTS);
}

std::optional<std::string> boundString(const BoundTokens& bound, std::string_view token)
{
    const auto found = bound.find(std::string(token));
    if(found == bound.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&found->second);
    return value == nullptr ? std::nullopt : std::make_optional(*value);
}

/// Kernel-scoped applicability: the kernel is baked for the graph's element type and
/// direction. Without it an f32 graph could reach an f16 kernel, or a forward graph a
/// backward entry point.
bool kernelMatches(const MatchContext& /*context*/,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto dtype = boundString(bound, DTYPE_TOKEN);
    const auto direction = boundString(bound, DIRECTION_TOKEN);
    return dtype.has_value() && direction.has_value()
           && kernel.getStringMetadata(std::string(DTYPE_FIELD)) == *dtype
           && kernel.getStringMetadata(std::string(DIRECTION_FIELD)) == *direction;
}

ConvDirection directionFor(const KernelDefinition& kernel)
{
    const auto& direction = kernel.getStringMetadata(std::string(DIRECTION_FIELD));
    for(const auto candidate :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        if(direction == directionName(candidate))
        {
            return candidate;
        }
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "conv_naive kernel declares unsupported direction '" + direction + "'");
}

/// The argument list the kernels take: one ConvNaiveArgs by value. Only consulted for a
/// pre-built kernel; a HIPRTC-compiled one carries no recorded signature.
const std::vector<KernelArgument>& kernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", static_cast<uint32_t>(sizeof(ConvNaiveArgs)), 0, ""}};
    return s_signature;
}

/// The compiled kernel and its argument block, minus the three pointers, which change per
/// execution. Immutable once built, so concurrent launches are safe.
class PreparedConvNaive : public IngestorPreparedDispatch
{
public:
    PreparedConvNaive(IngestorKernelCode code, const ConvNaiveProblem& problem)
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

    const ConvNaiveProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _code;
    ConvNaiveProblem _problem;
};

class ConvNaiveDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    ConvNaiveDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                             const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Every output element is accumulated in a register and written once.
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
        // Re-derived from the graph, which the match admitted; the bound uids confirm it is
        // the same graph.
        const auto direction = directionFor(kernel);
        const auto problem = conv_naive::parseConvNaiveProblem(context, direction);
        const auto x = tryGetBoundInt(bound, X_TOKEN);
        const auto w = tryGetBoundInt(bound, W_TOKEN);
        const auto y = tryGetBoundInt(bound, Y_TOKEN);
        if(!problem.has_value() || !x.has_value() || !w.has_value() || !y.has_value()
           || *x != problem->operands.x || *w != problem->operands.w || *y != problem->operands.y)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_naive dispatch was given a graph its matcher did not admit");
        }

        compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        options.add("CONV_NAIVE_TYPE",
                    elementTypeFor(kernel.getStringMetadata(std::string(DTYPE_FIELD))));

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, kernelSignature());

        // Backward weights: one block per dw element (the block shares its reduction); else one
        // thread per output element.
        const int64_t blocks
            = std::clamp<int64_t>(direction == ConvDirection::BACKWARD_WEIGHTS
                                      ? problem->args.total
                                      : (problem->args.total + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                  1,
                                  MAX_GRID_BLOCKS);
        code.setBlockSize(BLOCK_SIZE, 1, 1);
        code.setGridSize(static_cast<unsigned int>(blocks), 1, 1);

        return std::make_unique<PreparedConvNaive>(std::move(code), *problem);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedConvNaive&>(prepared);
        const auto& problem = dispatch.problem();

        ConvNaiveArgs args = problem.args;
        args.x = hipdnn_plugin_sdk::findDeviceBuffer(
                     problem.operands.x, deviceBuffers, numDeviceBuffers)
                     .ptr;
        args.w = hipdnn_plugin_sdk::findDeviceBuffer(
                     problem.operands.w, deviceBuffers, numDeviceBuffers)
                     .ptr;
        args.y = hipdnn_plugin_sdk::findDeviceBuffer(
                     problem.operands.y, deviceBuffers, numDeviceBuffers)
                     .ptr;

        dispatch.kernelForStream(handle.getStream()).launch(handle.getStream(), args);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& convNaiveKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

/// Process-lifetime: the registry holds a non-owning pointer to the handler, and a
/// provider's Container is created and destroyed per handle.
const ConvNaiveDispatchHandler& convNaiveDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convNaiveKpackModuleCache());
    static const ConvNaiveDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvNaiveSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(FWD_GRAPH_MATCHER_SYMBOL), &fwdGraphMatches);
    scope.add(std::string(BWD_GRAPH_MATCHER_SYMBOL), &bwdGraphMatches);
    scope.add(std::string(WRW_GRAPH_MATCHER_SYMBOL), &wrwGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &convNaiveDispatchHandler());
}

void resetConvNaiveModuleCache()
{
    convNaiveKpackModuleCache().clear();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
