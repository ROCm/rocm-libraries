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
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
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
#include "engines/hip_mlops_engine/plans/layernorm/LayernormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormFwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleFwdPlan.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"

/**
 * @file HipMlopsNative.cpp
 * @brief The HIP MLOps engine on the kernel ingestor (ALMIOPEN-3066): the engine's graph
 *        match, its per-family operation matchers, and one dispatch handler per family.
 *
 * The descriptor engine `hipkernel:HipMlops` takes over HIP_MLOPS_ENGINE's name and id
 * (`builtin_engine`). Each family accepts exactly the graphs the hand-written engine's plan
 * builder accepts, through that builder's own validator, and compiles the same embedded
 * source with the same compile options and launches (buildResampleFwdKernel(),
 * buildLayernormBwdKernel() and friends), so the two engines serve a graph with one program.
 * The hand-written engine's engine-level checks (ragged and misaligned tensors) run first,
 * here as there.
 *
 * Deliberate difference: a graph with override shapes is declined by every family. Each
 * bakes its shapes into the compiled program; the hand-written Resample builders were the
 * only ones without this guard.
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

constexpr std::string_view OPERATION_FIELD = "operation";
constexpr std::string_view OPERATION_TOKEN = "hip_mlops.operation";

// The values of the kernels' `operation` metadata, one per pack.
constexpr std::string_view RESAMPLE_FWD = "RESAMPLE_FWD";
constexpr std::string_view RESAMPLE_BWD = "RESAMPLE_BWD";
constexpr std::string_view LAYERNORM_FWD = "LAYERNORM_FWD";
constexpr std::string_view LAYERNORM_BWD = "LAYERNORM_BWD";

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

/// The operation of the family that serves @p context's graph, or nullopt. Mirrors the
/// hand-written plan builders' isApplicable(): one node of the family's attributes, fp32
/// compute, and the family's validator. Families are disjoint by node type.
std::optional<std::string_view> familyOf(const MatchContext& context)
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

    const auto& tensorMap = context.graph.getTensorMap();
    try
    {
        switch(node.attributesType())
        {
        case data_objects::NodeAttributes::ResampleFwdAttributes:
            resample::ResampleValidator(tensorMap).checkTensorConfigSupported(
                node.attributesAs<data_objects::ResampleFwdAttributes>());
            return RESAMPLE_FWD;
        case data_objects::NodeAttributes::ResampleBwdAttributes:
            resample::ResampleValidator(tensorMap).checkBwdTensorConfigSupported(
                node.attributesAs<data_objects::ResampleBwdAttributes>());
            return RESAMPLE_BWD;
        case data_objects::NodeAttributes::LayernormAttributes:
            layernorm::LayernormValidator(tensorMap).checkTensorConfigSupported(
                node.attributesAs<data_objects::LayernormAttributes>());
            return LAYERNORM_FWD;
        case data_objects::NodeAttributes::LayernormBackwardAttributes:
            layernorm::LayernormValidator(tensorMap).checkTensorConfigSupported(
                node.attributesAs<data_objects::LayernormBackwardAttributes>());
            return LAYERNORM_BWD;
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
 *        family that serves the graph, which the per-pack operation matchers select on.
 */
std::optional<BoundTokens> hipMlopsGraphMatches(const MatchContext& context)
{
    if(!passesEngineChecks(context) || context.graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    const auto family = familyOf(context);
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

template <const std::string_view& Operation>
bool operationMatches(const MatchContext& /*context*/, const BoundTokens& bound)
{
    return boundOperation(bound) == Operation;
}

/// Kernel-scoped: the kernel compiles the operation the graph match bound.
bool kernelMatches(const MatchContext& /*context*/,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    return boundOperation(bound) == kernel.getStringMetadata(std::string(OPERATION_FIELD));
}

// ---------------------------------------------------------------------------
// Dispatch: shared
// ---------------------------------------------------------------------------

/// A family's prepared launch: the compiled program(s) and the uids and scalars its launches
/// read, copied out of the graph at prepare() so nothing points back into it.
class PreparedHipMlops : public IngestorPreparedDispatch
{
public:
    virtual void launch(hipStream_t stream,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers,
                        void* workspace) const
        = 0;
};

/// A flatbuffer optional uid as a std::optional.
std::optional<int64_t> optionalUid(const flatbuffers::Optional<int64_t>& uid)
{
    return uid.has_value() ? std::optional<int64_t>(uid.value()) : std::nullopt;
}

/// The device pointer bound to @p uid, or null for an absent optional operand.
class Buffers
{
public:
    Buffers(const hipdnnPluginDeviceBuffer_t* deviceBuffers, uint32_t numDeviceBuffers)
        : _deviceBuffers(deviceBuffers)
        , _numDeviceBuffers(numDeviceBuffers)
    {
    }

    void* operator()(int64_t uid) const
    {
        return hipdnn_plugin_sdk::findDeviceBuffer(uid, _deviceBuffers, _numDeviceBuffers).ptr;
    }

    void* operator()(const std::optional<int64_t>& uid) const
    {
        return uid.has_value() ? (*this)(*uid) : nullptr;
    }

    float scalar(const hipdnn_plugin_sdk::ScalarOperand& operand) const
    {
        return static_cast<float>(hipdnn_plugin_sdk::toDouble(
            hipdnn_plugin_sdk::resolveScalarOperand(operand, _deviceBuffers, _numDeviceBuffers)));
    }

private:
    const hipdnnPluginDeviceBuffer_t* _deviceBuffers;
    uint32_t _numDeviceBuffers;
};

/// Compiles entry point @p entryPoint of @p kernel's source with @p options and applies a
/// 1D geometry. Entry points after the first come out of the program the first compiled.
IngestorKernelCode compileEntryPoint(const compilation::IKernelCompiler& compiler,
                                     const compilation::KpackKernelLoader& loader,
                                     const MatchContext& context,
                                     const KernelDefinition& kernel,
                                     const compilation::KernelCompileOptions& options,
                                     const std::string& entryPoint,
                                     unsigned int blockSize,
                                     unsigned int gridSize)
{
    auto code = buildIngestorKernelCode(
        compiler, loader, context, entryPointKernelDefinition(kernel, entryPoint), options, {});
    code.setBlockSize(blockSize, 1, 1);
    code.setGridSize(gridSize, 1, 1);
    return code;
}

/// Shared by every family's handler: no workspace unless the family says otherwise, and
/// launch() is the prepared dispatch's. Plans compile embedded source, so they cannot be
/// saved (the default).
class HipMlopsDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param compiler and @p loader must outlive this handler; all are process-lifetime.
    HipMlopsDispatchHandler(const compilation::IKernelCompiler& compiler,
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
                void* workspace) const override
    {
        dynamic_cast<const PreparedHipMlops&>(prepared).launch(
            handle.getStream(), deviceBuffers, numDeviceBuffers, workspace);
    }

protected:
    const compilation::IKernelCompiler& _compiler;
    const compilation::KpackKernelLoader& _loader;
};

// ---------------------------------------------------------------------------
// Dispatch: Resample
// ---------------------------------------------------------------------------

/// ResampleFwd(x, y, index) or ResampleBwd(dy, index, dx); the index is optional.
class PreparedResample : public PreparedHipMlops
{
public:
    PreparedResample(IngestorKernelCode code,
                     int64_t sourceUid,
                     std::optional<int64_t> indexUid,
                     int64_t resultUid,
                     bool backward)
        : _code(std::move(code))
        , _sourceUid(sourceUid)
        , _indexUid(indexUid)
        , _resultUid(resultUid)
        , _backward(backward)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const Buffers buffer(deviceBuffers, numDeviceBuffers);
        void* source = buffer(_sourceUid);
        void* index = buffer(_indexUid);
        void* result = buffer(_resultUid);

        // Changing these argument lists means changing the kernel sources with them.
        auto& kernel = _code.kernelForStream(stream);
        if(_backward)
        {
            kernel.launch(stream, source, index, result);
        }
        else
        {
            kernel.launch(stream, source, result, index);
        }
    }

private:
    IngestorKernelCode _code;
    int64_t _sourceUid;
    std::optional<int64_t> _indexUid;
    int64_t _resultUid;
    bool _backward;
};

class ResampleFwdDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& node = context.graph.getNodeWrapper(0);
        const auto& attributes = node.attributesAs<data_objects::ResampleFwdAttributes>();
        const resample::ResampleFwdParams params(
            attributes, context.graph.getTensorMap(), node.computeDataType());
        const auto build
            = resample::buildResampleFwdKernel(params, context.deviceProperties.gcnArchName);
        auto code = compileEntryPoint(_compiler,
                                      _loader,
                                      context,
                                      kernel,
                                      build.options,
                                      "ResampleFwd",
                                      resample::ResampleKernelBuild::BLOCK_SIZE,
                                      build.gridSize);

        return std::make_unique<PreparedResample>(std::move(code),
                                                  attributes.x_tensor_uid(),
                                                  optionalUid(attributes.index_tensor_uid()),
                                                  attributes.y_tensor_uid(),
                                                  /*backward=*/false);
    }
};

class ResampleBwdDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& node = context.graph.getNodeWrapper(0);
        const auto& attributes = node.attributesAs<data_objects::ResampleBwdAttributes>();
        const resample::ResampleBwdParams params(
            attributes, context.graph.getTensorMap(), node.computeDataType());
        const auto build
            = resample::buildResampleBwdKernel(params, context.deviceProperties.gcnArchName);
        auto code = compileEntryPoint(_compiler,
                                      _loader,
                                      context,
                                      kernel,
                                      build.options,
                                      "ResampleBwd",
                                      resample::ResampleKernelBuild::BLOCK_SIZE,
                                      build.gridSize);

        return std::make_unique<PreparedResample>(std::move(code),
                                                  attributes.dy_tensor_uid(),
                                                  optionalUid(attributes.index_tensor_uid()),
                                                  attributes.dx_tensor_uid(),
                                                  /*backward=*/true);
    }
};

// ---------------------------------------------------------------------------
// Dispatch: Layernorm
// ---------------------------------------------------------------------------

/// Compiles every launch of @p build, in order, out of one program.
std::vector<IngestorKernelCode> compileLaunches(const compilation::IKernelCompiler& compiler,
                                                const compilation::KpackKernelLoader& loader,
                                                const MatchContext& context,
                                                const KernelDefinition& kernel,
                                                const layernorm::LayernormKernelBuild& build)
{
    std::vector<IngestorKernelCode> codes;
    codes.reserve(build.launches.size());
    for(const auto& launch : build.launches)
    {
        codes.push_back(compileEntryPoint(compiler,
                                          loader,
                                          context,
                                          kernel,
                                          build.options,
                                          launch.entryPoint,
                                          launch.blockSize,
                                          launch.gridSize));
    }
    return codes;
}

/// LayernormFwd(x, y, scale, bias, mean, invVariance, epsilon); mean and inverse variance
/// are optional outputs.
class PreparedLayernormFwd : public PreparedHipMlops
{
public:
    PreparedLayernormFwd(IngestorKernelCode code,
                         const data_objects::LayernormAttributes& attributes,
                         hipdnn_plugin_sdk::ScalarOperand epsilon)
        : _code(std::move(code))
        , _x(attributes.x_tensor_uid())
        , _y(attributes.y_tensor_uid())
        , _scale(attributes.scale_tensor_uid())
        , _bias(attributes.bias_tensor_uid())
        , _mean(optionalUid(attributes.mean_tensor_uid()))
        , _invVariance(optionalUid(attributes.inv_variance_tensor_uid()))
        , _epsilon(epsilon)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const Buffers buffer(deviceBuffers, numDeviceBuffers);
        _code.kernelForStream(stream).launch(stream,
                                             buffer(_x),
                                             buffer(_y),
                                             buffer(_scale),
                                             buffer(_bias),
                                             buffer(_mean),
                                             buffer(_invVariance),
                                             buffer.scalar(_epsilon));
    }

private:
    IngestorKernelCode _code;
    int64_t _x;
    int64_t _y;
    int64_t _scale;
    int64_t _bias;
    std::optional<int64_t> _mean;
    std::optional<int64_t> _invVariance;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;
};

/// LayernormBwd, then LayernormBwdScaleBias, or LayernormBwdScaleBiasParallel and
/// LayernormBwdScaleBiasReduceSum, sharing the workspace in that order. Without an epsilon
/// in the graph the hand-written plan's default applies.
class PreparedLayernormBwd : public PreparedHipMlops
{
public:
    PreparedLayernormBwd(std::vector<IngestorKernelCode> codes,
                         const data_objects::LayernormBackwardAttributes& attributes,
                         std::optional<hipdnn_plugin_sdk::ScalarOperand> epsilon)
        : _codes(std::move(codes))
        , _dy(attributes.dy_tensor_uid())
        , _x(attributes.x_tensor_uid())
        , _scale(attributes.scale_tensor_uid())
        , _mean(optionalUid(attributes.mean_tensor_uid()))
        , _invVariance(optionalUid(attributes.inv_variance_tensor_uid()))
        , _dx(attributes.dx_tensor_uid())
        , _dscale(attributes.dscale_tensor_uid())
        , _dbias(attributes.dbias_tensor_uid())
        , _epsilon(epsilon)
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _codes.front();
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const Buffers buffer(deviceBuffers, numDeviceBuffers);
        const float epsilon
            = _epsilon.has_value()
                  ? buffer.scalar(*_epsilon)
                  : static_cast<float>(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
        void* dy = buffer(_dy);
        void* x = buffer(_x);
        void* mean = buffer(_mean);
        void* invVariance = buffer(_invVariance);

        // Changing these argument lists means changing LayernormBwd.cpp with them.
        _codes[0].kernelForStream(stream).launch(
            stream, dy, x, buffer(_scale), mean, invVariance, buffer(_dx), workspace, epsilon);
        if(_codes.size() == 3)
        {
            _codes[1].kernelForStream(stream).launch(stream, dy, x, mean, invVariance, workspace);
            _codes[2].kernelForStream(stream).launch(
                stream, workspace, buffer(_dscale), buffer(_dbias));
        }
        else
        {
            _codes[1].kernelForStream(stream).launch(
                stream, dy, x, mean, invVariance, buffer(_dscale), buffer(_dbias), workspace);
        }
    }

private:
    std::vector<IngestorKernelCode> _codes;
    int64_t _dy;
    int64_t _x;
    int64_t _scale;
    std::optional<int64_t> _mean;
    std::optional<int64_t> _invVariance;
    int64_t _dx;
    int64_t _dscale;
    int64_t _dbias;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _epsilon;
};

class LayernormFwdDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& attributes
            = context.graph.getNodeWrapper(0).attributesAs<data_objects::LayernormAttributes>();
        const auto& tensorMap = context.graph.getTensorMap();
        const layernorm::LayernormFwdParams params(attributes, tensorMap);
        const auto build
            = layernorm::buildLayernormFwdKernel(params, context.deviceProperties.gcnArchName);
        auto codes = compileLaunches(_compiler, _loader, context, kernel, build);

        return std::make_unique<PreparedLayernormFwd>(
            std::move(codes.front()),
            attributes,
            hipdnn_plugin_sdk::makeScalarOperand(
                tensorMap, attributes.epsilon_tensor_uid(), "Epsilon"));
    }
};

class LayernormBwdDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        const auto& attributes = context.graph.getNodeWrapper(0)
                                     .attributesAs<data_objects::LayernormBackwardAttributes>();
        const layernorm::LayernormBwdParams params(attributes, context.graph.getTensorMap());
        return layernorm::layernormBwdWorkspaceBytes(params,
                                                     context.deviceProperties.multiProcessorCount);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& attributes = context.graph.getNodeWrapper(0)
                                     .attributesAs<data_objects::LayernormBackwardAttributes>();
        const auto& tensorMap = context.graph.getTensorMap();
        const layernorm::LayernormBwdParams params(attributes, tensorMap);
        const auto build
            = layernorm::buildLayernormBwdKernel(params,
                                                 context.deviceProperties.gcnArchName,
                                                 context.deviceProperties.multiProcessorCount);

        std::optional<hipdnn_plugin_sdk::ScalarOperand> epsilon;
        if(attributes.epsilon_tensor_uid().has_value())
        {
            epsilon = hipdnn_plugin_sdk::makeScalarOperand(
                tensorMap, attributes.epsilon_tensor_uid().value(), "Epsilon");
        }
        return std::make_unique<PreparedLayernormBwd>(
            compileLaunches(_compiler, _loader, context, kernel, build), attributes, epsilon);
    }
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

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

template <typename Handler>
const Handler& dispatchHandler()
{
    static const Handler s_handler(kernelCompiler(), kpackLoader());
    return s_handler;
}

} // namespace

void registerHipMlopsSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &hipMlopsGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);

    scope.add("hipkernel.hip_mlops.resample_fwd_match", &operationMatches<RESAMPLE_FWD>);
    scope.add("hipkernel.hip_mlops.resample_bwd_match", &operationMatches<RESAMPLE_BWD>);
    scope.add("hipkernel.hip_mlops.layernorm_fwd_match", &operationMatches<LAYERNORM_FWD>);
    scope.add("hipkernel.hip_mlops.layernorm_bwd_match", &operationMatches<LAYERNORM_BWD>);

    scope.add("hipkernel.hip_mlops.resample_fwd_dispatch",
              &dispatchHandler<ResampleFwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.resample_bwd_dispatch",
              &dispatchHandler<ResampleBwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.layernorm_fwd_dispatch",
              &dispatchHandler<LayernormFwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.layernorm_bwd_dispatch",
              &dispatchHandler<LayernormBwdDispatchHandler>());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
