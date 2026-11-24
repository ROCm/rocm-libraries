// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/HiprtcKernelCompiler.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleFwdPlan.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"

/**
 * @file HipMlopsNative.cpp
 * @brief The HIP MLOps engine on the kernel ingestor (ALMIOPEN-3066): the engine's graph
 *        match, its per-family operation matchers, and one dispatch handler per family.
 *
 * The descriptor engine `hipkernel:HipMlops` takes over HIP_MLOPS_ENGINE's name and id
 * (`builtin_engine`). Each family accepts exactly the graphs the hand-written engine's plan
 * builder accepts, through that builder's own validator, and compiles the same embedded
 * source with the same compile options (buildResampleFwdKernel() and friends), so the two
 * engines serve a graph with one program. The hand-written engine's engine-level checks
 * (ragged and misaligned tensors) run first, here as there.
 *
 * Deliberate difference: a graph with override shapes is declined. Every family bakes its
 * shapes into the compiled program, and the hand-written Resample builders were the only
 * ones without this guard.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.hip_mlops.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.hip_mlops.kernel_match";
constexpr std::string_view RESAMPLE_FWD_MATCHER_SYMBOL = "hipkernel.hip_mlops.resample_fwd_match";
constexpr std::string_view RESAMPLE_BWD_MATCHER_SYMBOL = "hipkernel.hip_mlops.resample_bwd_match";
constexpr std::string_view RESAMPLE_FWD_DISPATCH_SYMBOL
    = "hipkernel.hip_mlops.resample_fwd_dispatch";
constexpr std::string_view RESAMPLE_BWD_DISPATCH_SYMBOL
    = "hipkernel.hip_mlops.resample_bwd_dispatch";

constexpr std::string_view OPERATION_FIELD = "operation";
constexpr std::string_view OPERATION_TOKEN = "hip_mlops.operation";

// The values of the kernels' `operation` metadata, one per pack.
constexpr std::string_view RESAMPLE_FWD = "RESAMPLE_FWD";
constexpr std::string_view RESAMPLE_BWD = "RESAMPLE_BWD";

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The hand-written engine's checks before any plan builder: no ragged tensor, no stray
/// ragged multiplier, every tensor 16-byte aligned.
bool passesEngineChecks(const MatchContext& context)
{
    const auto& tensorMap = context.graph.getTensorMap();
    return hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorMap)
           && core::utils::hasNoStrayRaggedOffsetMultiplier(tensorMap)
           && core::utils::hasOnly16ByteAlignedTensors(tensorMap);
}

/// The family that serves @p context's graph, or nullopt. Mirrors the hand-written
/// Resample builders' isApplicable(): one node of the family's attributes, fp32 compute,
/// and the family's validator.
std::optional<std::string_view> resampleFamily(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.computeDataType() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }

    try
    {
        resample::ResampleValidator validator(context.graph.getTensorMap());
        switch(node.attributesType())
        {
        case data_objects::NodeAttributes::ResampleFwdAttributes:
            validator.checkTensorConfigSupported(
                node.attributesAs<data_objects::ResampleFwdAttributes>());
            return RESAMPLE_FWD;
        case data_objects::NodeAttributes::ResampleBwdAttributes:
            validator.checkBwdTensorConfigSupported(
                node.attributesAs<data_objects::ResampleBwdAttributes>());
            return RESAMPLE_BWD;
        default:
            return std::nullopt;
        }
    }
    catch(const std::exception& error)
    {
        HIPDNN_PLUGIN_LOG_INFO("hipkernel:HipMlops declines the graph: " << error.what());
        return std::nullopt;
    }
}

/**
 * @brief Graph-scoped applicability for the whole engine: binds the operation of the one
 *        family that serves the graph. Families are disjoint by node type, so at most one
 *        binds and the per-pack operation matchers select its pack.
 */
std::optional<BoundTokens> hipMlopsGraphMatches(const MatchContext& context)
{
    if(!passesEngineChecks(context) || context.graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    const auto family = resampleFamily(context);
    if(!family.has_value())
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(OPERATION_TOKEN)] = std::string(*family);
    return bound;
}

/// The operation the graph match bound, or nullopt.
std::optional<std::string> boundOperation(const BoundTokens& bound)
{
    const auto entry = bound.find(std::string(OPERATION_TOKEN));
    if(entry == bound.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&entry->second);
    return value == nullptr ? std::nullopt : std::make_optional(*value);
}

bool resampleFwdOperationMatches(const MatchContext& /*context*/, const BoundTokens& bound)
{
    return boundOperation(bound) == RESAMPLE_FWD;
}

bool resampleBwdOperationMatches(const MatchContext& /*context*/, const BoundTokens& bound)
{
    return boundOperation(bound) == RESAMPLE_BWD;
}

/// Kernel-scoped: the kernel compiles the operation the graph match bound.
bool kernelMatches(const MatchContext& /*context*/,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    return boundOperation(bound) == kernel.getStringMetadata(std::string(OPERATION_FIELD));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// A compiled resample kernel and the uids of its three buffer arguments, in launch order.
/// The index buffer is optional; launch() passes a null pointer when the graph has none.
class PreparedResample : public IngestorPreparedDispatch
{
public:
    PreparedResample(IngestorKernelCode code,
                     int64_t firstUid,
                     std::optional<int64_t> indexUid,
                     int64_t lastUid,
                     bool indexInMiddle)
        : _code(std::move(code))
        , _firstUid(firstUid)
        , _indexUid(indexUid)
        , _lastUid(lastUid)
        , _indexInMiddle(indexInMiddle)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers) const
    {
        const auto buffer = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
        };
        void* first = buffer(_firstUid);
        void* last = buffer(_lastUid);
        void* index = _indexUid.has_value() ? buffer(*_indexUid) : nullptr;

        // Changing these argument lists means changing the kernel sources with them:
        // ResampleFwd(x, y, index) and ResampleBwd(dy, index, dx).
        auto& kernel = _code.kernelForStream(stream);
        if(_indexInMiddle)
        {
            kernel.launch(stream, first, index, last);
        }
        else
        {
            kernel.launch(stream, first, last, index);
        }
    }

private:
    IngestorKernelCode _code;
    int64_t _firstUid;
    std::optional<int64_t> _indexUid;
    int64_t _lastUid;
    bool _indexInMiddle;
};

/// A flatbuffer optional uid as a std::optional.
std::optional<int64_t> optionalUid(const flatbuffers::Optional<int64_t>& uid)
{
    return uid.has_value() ? std::optional<int64_t>(uid.value()) : std::nullopt;
}

/// Compiles @p kernel with @p build and applies its geometry.
IngestorKernelCode compileResample(const compilation::IKernelCompiler& compiler,
                                   const compilation::KpackKernelLoader& loader,
                                   const MatchContext& context,
                                   const KernelDefinition& kernel,
                                   const resample::ResampleKernelBuild& build)
{
    auto code = buildIngestorKernelCode(compiler, loader, context, kernel, build.options, {});
    code.setBlockSize(resample::ResampleKernelBuild::BLOCK_SIZE, 1, 1);
    code.setGridSize(build.gridSize, 1, 1);
    return code;
}

/// Shared by both families' handlers: no workspace, and launch() is the prepared
/// dispatch's. Plans compile embedded source, so they cannot be saved (the default).
class ResampleDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param compiler and @p loader must outlive this handler; all are process-lifetime.
    ResampleDispatchHandler(const compilation::IKernelCompiler& compiler,
                            const compilation::KpackKernelLoader& loader)
        : _compiler(compiler)
        , _loader(loader)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        dynamic_cast<const PreparedResample&>(prepared).launch(
            handle.getStream(), deviceBuffers, numDeviceBuffers);
    }

protected:
    const compilation::IKernelCompiler& _compiler;
    const compilation::KpackKernelLoader& _loader;
};

class ResampleFwdDispatchHandler : public ResampleDispatchHandler
{
public:
    using ResampleDispatchHandler::ResampleDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& node = context.graph.getNodeWrapper(0);
        const auto& attributes = node.attributesAs<data_objects::ResampleFwdAttributes>();
        const resample::ResampleFwdParams params(
            attributes, context.graph.getTensorMap(), node.computeDataType());
        auto code = compileResample(
            _compiler,
            _loader,
            context,
            kernel,
            resample::buildResampleFwdKernel(params, context.deviceProperties.gcnArchName));

        return std::make_unique<PreparedResample>(std::move(code),
                                                  attributes.x_tensor_uid(),
                                                  optionalUid(attributes.index_tensor_uid()),
                                                  attributes.y_tensor_uid(),
                                                  /*indexInMiddle=*/false);
    }
};

class ResampleBwdDispatchHandler : public ResampleDispatchHandler
{
public:
    using ResampleDispatchHandler::ResampleDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& node = context.graph.getNodeWrapper(0);
        const auto& attributes = node.attributesAs<data_objects::ResampleBwdAttributes>();
        const resample::ResampleBwdParams params(
            attributes, context.graph.getTensorMap(), node.computeDataType());
        auto code = compileResample(
            _compiler,
            _loader,
            context,
            kernel,
            resample::buildResampleBwdKernel(params, context.deviceProperties.gcnArchName));

        return std::make_unique<PreparedResample>(std::move(code),
                                                  attributes.dy_tensor_uid(),
                                                  optionalUid(attributes.index_tensor_uid()),
                                                  attributes.dx_tensor_uid(),
                                                  /*indexInMiddle=*/true);
    }
};

/// Process-lifetime state the registry's non-owning pointers need to outlive every
/// per-handle Container. The loader is never reached -- every kernel is embedded source --
/// but buildIngestorKernelCode takes one.
const compilation::HiprtcKernelCompiler& kernelCompiler()
{
    static const compilation::HiprtcKernelCompiler s_compiler;
    return s_compiler;
}

const compilation::KpackKernelLoader& kpackLoader()
{
    static compilation::KpackModuleCache s_cache;
    static const compilation::KpackKernelLoader s_loader(s_cache);
    return s_loader;
}

const ResampleFwdDispatchHandler& resampleFwdDispatchHandler()
{
    static const ResampleFwdDispatchHandler s_handler(kernelCompiler(), kpackLoader());
    return s_handler;
}

const ResampleBwdDispatchHandler& resampleBwdDispatchHandler()
{
    static const ResampleBwdDispatchHandler s_handler(kernelCompiler(), kpackLoader());
    return s_handler;
}

} // namespace

void registerHipMlopsSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &hipMlopsGraphMatches);
    scope.add(std::string(RESAMPLE_FWD_MATCHER_SYMBOL), &resampleFwdOperationMatches);
    scope.add(std::string(RESAMPLE_BWD_MATCHER_SYMBOL), &resampleBwdOperationMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(RESAMPLE_FWD_DISPATCH_SYMBOL), &resampleFwdDispatchHandler());
    scope.add(std::string(RESAMPLE_BWD_DISPATCH_SYMBOL), &resampleBwdDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
