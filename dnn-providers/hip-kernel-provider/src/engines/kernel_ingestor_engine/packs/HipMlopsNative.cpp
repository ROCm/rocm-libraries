// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
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
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormFwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdInferencePlan.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdInferenceWithVariancePlan.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdTrainingPlan.hpp"
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
 * buildLayernormBwdKernel(), buildRMSnormFwdKernel() and friends), so the two engines serve a
 * graph with one program. RMSnorm and batchnorm also serve the hand-written builders'
 * fused graphs: an activation after the forward (two nodes), and before the backward (RMSnorm:
 * two nodes; batchnorm: three, from the inference that produced the activation's input).
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
constexpr std::string_view RMSNORM_FWD = "RMSNORM_FWD";
constexpr std::string_view RMSNORM_BWD = "RMSNORM_BWD";
constexpr std::string_view BATCHNORM_INFERENCE = "BATCHNORM_INFERENCE";
constexpr std::string_view BATCHNORM_INFERENCE_VARIANCE = "BATCHNORM_INFERENCE_VARIANCE";
constexpr std::string_view BATCHNORM_TRAINING = "BATCHNORM_TRAINING";
constexpr std::string_view BATCHNORM_BACKWARD = "BATCHNORM_BACKWARD";

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

/// The operation a two-node graph fuses with an activation, or nullopt: RMSnorm's forward
/// then the activation, or the activation's backward then RMSnorm's backward; or batchnorm
/// inference then the activation (the hand-written builders' two-node cases). Throws what
/// the checks throw.
std::optional<std::string_view> fusedFamilyOf(const MatchContext& context)
{
    const auto& first = context.graph.getNodeWrapper(0);
    const auto& second = context.graph.getNodeWrapper(1);
    const auto& tensorMap = context.graph.getTensorMap();
    if(first.attributesType() == data_objects::NodeAttributes::RMSNormAttributes
       && second.attributesType() == data_objects::NodeAttributes::PointwiseAttributes)
    {
        const auto& rmsnorm = first.attributesAs<data_objects::RMSNormAttributes>();
        const auto& activation = second.attributesAs<data_objects::PointwiseAttributes>();
        rmsnorm::checkFwdActivationTensors(rmsnorm, activation, tensorMap);
        rmsnorm::RMSnormValidator(tensorMap).checkFwdActivationTensorConfigSupported(rmsnorm,
                                                                                     activation);
        return RMSNORM_FWD;
    }
    if(first.attributesType() == data_objects::NodeAttributes::PointwiseAttributes
       && second.attributesType() == data_objects::NodeAttributes::RMSNormBackwardAttributes)
    {
        const auto& activation = first.attributesAs<data_objects::PointwiseAttributes>();
        const auto& rmsnorm = second.attributesAs<data_objects::RMSNormBackwardAttributes>();
        rmsnorm::checkBwdActivationTensors(activation, rmsnorm, tensorMap);
        rmsnorm::RMSnormValidator(tensorMap).checkBwdActivationTensorConfigSupported(activation,
                                                                                     rmsnorm);
        return RMSNORM_BWD;
    }
    if(first.attributesType() == data_objects::NodeAttributes::BatchnormInferenceAttributes
       && second.attributesType() == data_objects::NodeAttributes::PointwiseAttributes)
    {
        const auto& batchnorm = first.attributesAs<data_objects::BatchnormInferenceAttributes>();
        const auto& activation = second.attributesAs<data_objects::PointwiseAttributes>();
        batchnorm::checkInferenceActivationTensors(batchnorm, activation, tensorMap);
        BatchnormValidator(tensorMap).checkInferenceActivationTensorConfigSupported(batchnorm,
                                                                                    activation);
        return BATCHNORM_INFERENCE;
    }
    if(first.attributesType()
           == data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt
       && second.attributesType() == data_objects::NodeAttributes::PointwiseAttributes)
    {
        const auto& batchnorm
            = first.attributesAs<data_objects::BatchnormInferenceAttributesVarianceExt>();
        const auto& activation = second.attributesAs<data_objects::PointwiseAttributes>();
        batchnorm::checkInferenceVarianceExtActivationTensors(batchnorm, activation, tensorMap);
        BatchnormValidator(tensorMap).checkInferenceVarianceExtActivationTensorConfigSupported(
            batchnorm, activation);
        return BATCHNORM_INFERENCE_VARIANCE;
    }
    if(first.attributesType() == data_objects::NodeAttributes::BatchnormAttributes)
    {
        const auto& batchnorm = batchnorm::checkTrainingNode(first);
        const auto& activation = batchnorm::checkTrainingActivationNode(second, batchnorm);
        batchnorm::checkTrainingTensorVirtuality2Node(batchnorm, activation, tensorMap);
        BatchnormValidator(tensorMap).checkFwdTrainingActivationTensorConfigSupported(batchnorm,
                                                                                      activation);
        return BATCHNORM_TRAINING;
    }
    return std::nullopt;
}

/// The batchnorm backward a three-node graph fuses, or nullopt: the inference that produced
/// the activation's input, the activation's backward, then the backward (the hand-written
/// batchnorm builder's three-node case). Throws what the checks throw.
std::optional<std::string_view> threeNodeFamilyOf(const MatchContext& context)
{
    const auto& graph = context.graph;
    const auto& tensorMap = graph.getTensorMap();
    const auto& inference
        = graph.getNodeWrapper(0).attributesAs<data_objects::BatchnormInferenceAttributes>();
    const auto& activation
        = graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>();
    const auto& backward
        = graph.getNodeWrapper(2).attributesAs<data_objects::BatchnormBackwardAttributes>();
    batchnorm::checkBackwardActivationTensors(inference, activation, backward, tensorMap);
    BatchnormValidator(tensorMap).checkBwdActivationTensorConfigSupported(
        inference, activation, backward);
    return BATCHNORM_BACKWARD;
}

/// One-node batchnorm training. The hand-written engine serves it through
/// BatchnormPlanBuilder or, when that declines, BatchnormFwdTrainingPlanBuilder; both build the
/// same plan and run the same validator, and differ only in which operands may be virtual, so
/// either builder's operand check suffices.
void checkTrainingGraph(const MatchContext& context)
{
    const auto& tensorMap = context.graph.getTensorMap();
    const auto& attributes
        = context.graph.getNodeWrapper(0).attributesAs<data_objects::BatchnormAttributes>();
    try
    {
        batchnorm::checkOneNodeOperandsNotVirtual(context.graph.getNode(0), tensorMap);
    }
    catch(const std::exception&)
    {
        batchnorm::checkTrainingTensorVirtuality1Node(attributes, tensorMap);
    }
    BatchnormValidator(tensorMap).checkFwdTrainingTensorConfigSupported(attributes);
}

/// The operation of the family that serves @p context's graph, or nullopt. Mirrors the
/// hand-written plan builders' isApplicable(): fp32 compute throughout, one node of the
/// family's attributes (or a fused graph of two or three), and the family's validator.
/// Families are disjoint by their first node's type.
std::optional<std::string_view> familyOf(const MatchContext& context)
{
    const auto nodeCount = context.graph.nodeCount();
    if(nodeCount < 1 || nodeCount > 3)
    {
        return std::nullopt;
    }
    for(uint32_t i = 0; i < nodeCount; ++i)
    {
        if(context.graph.getNodeWrapper(i).computeDataType() != data_objects::DataType::FLOAT)
        {
            return std::nullopt;
        }
    }

    const auto& node = context.graph.getNodeWrapper(0);
    const auto& tensorMap = context.graph.getTensorMap();
    try
    {
        if(nodeCount == 2)
        {
            return fusedFamilyOf(context);
        }
        if(nodeCount == 3)
        {
            return threeNodeFamilyOf(context);
        }
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
        case data_objects::NodeAttributes::RMSNormAttributes:
            rmsnorm::RMSnormValidator(tensorMap).checkFwdTensorConfigSupported(
                node.attributesAs<data_objects::RMSNormAttributes>());
            return RMSNORM_FWD;
        case data_objects::NodeAttributes::RMSNormBackwardAttributes:
            rmsnorm::RMSnormValidator(tensorMap).checkBwdTensorConfigSupported(
                node.attributesAs<data_objects::RMSNormBackwardAttributes>());
            return RMSNORM_BWD;
        case data_objects::NodeAttributes::BatchnormInferenceAttributes:
            batchnorm::checkOneNodeOperandsNotVirtual(context.graph.getNode(0), tensorMap);
            BatchnormValidator(tensorMap).checkInferenceTensorConfigSupported(
                node.attributesAs<data_objects::BatchnormInferenceAttributes>());
            return BATCHNORM_INFERENCE;
        case data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt:
            batchnorm::checkOneNodeOperandsNotVirtual(context.graph.getNode(0), tensorMap);
            BatchnormValidator(tensorMap).checkInferenceVarianceExtTensorConfigSupported(
                node.attributesAs<data_objects::BatchnormInferenceAttributesVarianceExt>());
            return BATCHNORM_INFERENCE_VARIANCE;
        case data_objects::NodeAttributes::BatchnormAttributes:
            checkTrainingGraph(context);
            return BATCHNORM_TRAINING;
        case data_objects::NodeAttributes::BatchnormBackwardAttributes:
            batchnorm::checkOneNodeOperandsNotVirtual(context.graph.getNode(0), tensorMap);
            BatchnormValidator(tensorMap).checkBwdTensorConfigSupported(
                node.attributesAs<data_objects::BatchnormBackwardAttributes>());
            return BATCHNORM_BACKWARD;
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

/// The uid of an optional operand the plan params hold, or nullopt.
std::optional<int64_t> uidOf(const data_objects::TensorAttributes* tensor)
{
    return tensor == nullptr ? std::nullopt : std::optional<int64_t>(tensor->uid());
}

/// The alpha and beta launch arguments of a fused activation, zero without one.
std::pair<float, float>
    activationArguments(const std::optional<core::utils::ActivationParams>& activation)
{
    return activation.has_value() ? std::make_pair(static_cast<float>(activation->alpha),
                                                   static_cast<float>(activation->beta))
                                  : std::make_pair(0.0f, 0.0f);
}

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

/// Compiles every launch of @p build (a family's KernelBuild), in order, out of one program.
template <typename KernelBuild>
std::vector<IngestorKernelCode> compileLaunches(const compilation::IKernelCompiler& compiler,
                                                const compilation::KpackKernelLoader& loader,
                                                const MatchContext& context,
                                                const KernelDefinition& kernel,
                                                const KernelBuild& build)
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
// Dispatch: RMSnorm
// ---------------------------------------------------------------------------

/// RMSnormFwd(x, scale, bias, y, invRms, epsilon, alpha, beta). Bias and the inverse RMS are
/// optional; y is the fused activation's output when there is one.
class PreparedRmsnormFwd : public PreparedHipMlops
{
public:
    PreparedRmsnormFwd(IngestorKernelCode code,
                       const rmsnorm::RMSnormFwdParams& params,
                       hipdnn_plugin_sdk::ScalarOperand epsilon)
        : _code(std::move(code))
        , _x(params.x()->uid())
        , _scale(params.scale()->uid())
        , _bias(uidOf(params.bias()))
        , _y(params.optActivation().has_value() && params.activationOut() != nullptr
                 ? params.activationOut()->uid()
                 : params.y()->uid())
        , _invRms(uidOf(params.invRMS()))
        , _epsilon(epsilon)
        , _activation(activationArguments(params.optActivation()))
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
        // Changing this argument list means changing RMSNormFwd.cpp with it.
        _code.kernelForStream(stream).launch(stream,
                                             buffer(_x),
                                             buffer(_scale),
                                             buffer(_bias),
                                             buffer(_y),
                                             buffer(_invRms),
                                             buffer.scalar(_epsilon),
                                             _activation.first,
                                             _activation.second);
    }

private:
    IngestorKernelCode _code;
    int64_t _x;
    int64_t _scale;
    std::optional<int64_t> _bias;
    int64_t _y;
    std::optional<int64_t> _invRms;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;
    std::pair<float, float> _activation;
};

/// RMSnormBwdData(dy, x, scale, invRms, dx, y, alpha, beta), then
/// RMSnormBwdScaleBias(dy, x, invRms, dscale, dbias, y, alpha, beta). With a fused
/// activation, dy is the gradient the activation reads and y its forward input (x for
/// IDENTITY, which ignores it); dbias is optional.
class PreparedRmsnormBwd : public PreparedHipMlops
{
public:
    PreparedRmsnormBwd(std::vector<IngestorKernelCode> codes,
                       const rmsnorm::RMSnormBwdParams& params)
        : _codes(std::move(codes))
        , _dy(params.dy()->uid())
        , _x(params.x()->uid())
        , _scale(params.scale()->uid())
        , _invRms(params.invRMS()->uid())
        , _dx(params.dx()->uid())
        , _dscale(params.dscale()->uid())
        , _dbias(uidOf(params.dbias()))
        , _y(uidOf(params.y()))
        , _activation(activationArguments(params.optActivation()))
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _codes.front();
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const Buffers buffer(deviceBuffers, numDeviceBuffers);
        void* dy = buffer(_dy);
        void* x = buffer(_x);
        void* invRms = buffer(_invRms);
        void* y = buffer(_y);
        const auto [alpha, beta] = _activation;

        // Changing these argument lists means changing RMSNormBwd.cpp with them.
        _codes[0].kernelForStream(stream).launch(
            stream, dy, x, buffer(_scale), invRms, buffer(_dx), y, alpha, beta);
        _codes[1].kernelForStream(stream).launch(
            stream, dy, x, invRms, buffer(_dscale), buffer(_dbias), y, alpha, beta);
    }

private:
    std::vector<IngestorKernelCode> _codes;
    int64_t _dy;
    int64_t _x;
    int64_t _scale;
    int64_t _invRms;
    int64_t _dx;
    int64_t _dscale;
    std::optional<int64_t> _dbias;
    std::optional<int64_t> _y;
    std::pair<float, float> _activation;
};

class RmsnormFwdDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& graph = context.graph;
        const auto& tensorMap = graph.getTensorMap();
        const auto& attributes
            = graph.getNodeWrapper(0).attributesAs<data_objects::RMSNormAttributes>();
        // A fused graph's ReLU follows the RMSnorm.
        const auto params
            = graph.nodeCount() == 2
                  ? rmsnorm::RMSnormFwdParams(
                        attributes,
                        graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>(),
                        tensorMap)
                  : rmsnorm::RMSnormFwdParams(attributes, tensorMap);
        const auto build
            = rmsnorm::buildRMSnormFwdKernel(params, context.deviceProperties.gcnArchName);
        auto codes = compileLaunches(_compiler, _loader, context, kernel, build);

        return std::make_unique<PreparedRmsnormFwd>(
            std::move(codes.front()),
            params,
            hipdnn_plugin_sdk::makeScalarOperand(
                tensorMap, attributes.epsilon_tensor_uid(), "Epsilon"));
    }
};

class RmsnormBwdDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& graph = context.graph;
        const auto& tensorMap = graph.getTensorMap();
        // A fused graph's ReLU backward precedes the RMSnorm backward.
        const bool fused = graph.nodeCount() == 2;
        const auto& attributes = graph.getNodeWrapper(fused ? 1 : 0)
                                     .attributesAs<data_objects::RMSNormBackwardAttributes>();
        const auto params
            = fused ? rmsnorm::RMSnormBwdParams(
                          attributes,
                          graph.getNodeWrapper(0).attributesAs<data_objects::PointwiseAttributes>(),
                          tensorMap)
                    : rmsnorm::RMSnormBwdParams(attributes, tensorMap);
        const auto build
            = rmsnorm::buildRMSnormBwdKernel(params, context.deviceProperties.gcnArchName);

        return std::make_unique<PreparedRmsnormBwd>(
            compileLaunches(_compiler, _loader, context, kernel, build), params);
    }
};

// ---------------------------------------------------------------------------
// Dispatch: batchnorm inference
// ---------------------------------------------------------------------------

/// BatchNormFwdInferSpatialEstInvVar(x, y, mean, invVariance, scale, bias, strides..., alpha,
/// beta), or BatchNormFwdInferSpatialEst(x, y, mean, variance, scale, bias, epsilon,
/// strides..., alpha, beta) with a variance. y is the fused activation's output when there is
/// one.
class PreparedBatchnormInference : public PreparedHipMlops
{
public:
    template <typename Params>
    PreparedBatchnormInference(IngestorKernelCode code,
                               const Params& params,
                               int64_t variance,
                               std::optional<hipdnn_plugin_sdk::ScalarOperand> epsilon,
                               const batchnorm::BatchnormInferenceKernelBuild& build)
        : _code(std::move(code))
        , _x(params.x()->uid())
        , _y(params.optActivation().has_value() && params.activationOut() != nullptr
                 ? params.activationOut()->uid()
                 : params.y()->uid())
        , _mean(params.estMean()->uid())
        , _variance(variance)
        , _scale(params.scale()->uid())
        , _bias(params.bias()->uid())
        , _epsilon(std::move(epsilon))
        , _strides{build.channelStride, build.spatialStride, build.batchStride}
        , _activation(activationArguments(params.optActivation()))
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
        const auto [channelStride, spatialStride, batchStride] = _strides;
        const auto [alpha, beta] = _activation;
        auto& kernel = _code.kernelForStream(stream);
        // Changing these argument lists means changing BatchNormFwdInferSpatial.cpp with them.
        if(_epsilon.has_value())
        {
            const double epsilon
                = hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                    *_epsilon, deviceBuffers, numDeviceBuffers));
            kernel.launch(stream,
                          buffer(_x),
                          buffer(_y),
                          buffer(_mean),
                          buffer(_variance),
                          buffer(_scale),
                          buffer(_bias),
                          epsilon,
                          channelStride,
                          spatialStride,
                          batchStride,
                          alpha,
                          beta);
        }
        else
        {
            kernel.launch(stream,
                          buffer(_x),
                          buffer(_y),
                          buffer(_mean),
                          buffer(_variance),
                          buffer(_scale),
                          buffer(_bias),
                          channelStride,
                          spatialStride,
                          batchStride,
                          alpha,
                          beta);
        }
    }

private:
    IngestorKernelCode _code;
    int64_t _x;
    int64_t _y;
    int64_t _mean;
    int64_t _variance;
    int64_t _scale;
    int64_t _bias;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _epsilon;
    std::tuple<unsigned int, unsigned int, unsigned int> _strides;
    std::pair<float, float> _activation;
};

/// Compiles entry point @p entryPoint of @p kernel's source with @p options and a 3D geometry.
IngestorKernelCode compileBatchnormLaunch(const compilation::IKernelCompiler& compiler,
                                          const compilation::KpackKernelLoader& loader,
                                          const MatchContext& context,
                                          const KernelDefinition& kernel,
                                          const compilation::KernelCompileOptions& options,
                                          const char* entryPoint,
                                          const std::array<unsigned int, 3>& blockSize,
                                          const std::array<unsigned int, 3>& gridSize)
{
    auto code = buildIngestorKernelCode(
        compiler, loader, context, entryPointKernelDefinition(kernel, entryPoint), options, {});
    code.setBlockSize(blockSize[0], blockSize[1], blockSize[2]);
    code.setGridSize(gridSize[0], gridSize[1], gridSize[2]);
    return code;
}

IngestorKernelCode compileBatchnormInference(const compilation::IKernelCompiler& compiler,
                                             const compilation::KpackKernelLoader& loader,
                                             const MatchContext& context,
                                             const KernelDefinition& kernel,
                                             const batchnorm::BatchnormInferenceKernelBuild& build)
{
    return compileBatchnormLaunch(compiler,
                                  loader,
                                  context,
                                  kernel,
                                  build.options,
                                  build.entryPoint,
                                  build.blockSize,
                                  build.gridSize);
}

class BatchnormInferenceDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& graph = context.graph;
        const auto& tensorMap = graph.getTensorMap();
        const auto& attributes
            = graph.getNodeWrapper(0).attributesAs<data_objects::BatchnormInferenceAttributes>();
        // A fused graph's activation follows the batchnorm.
        const auto params
            = graph.nodeCount() == 2
                  ? batchnorm::BatchnormFwdInferenceParams(
                        attributes,
                        graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>(),
                        tensorMap)
                  : batchnorm::BatchnormFwdInferenceParams(attributes, tensorMap);
        const auto& properties = context.deviceProperties;
        const auto build = batchnorm::buildBatchnormFwdInferenceKernel(
            params, properties.gcnArchName, properties.multiProcessorCount, properties.warpSize);

        return std::make_unique<PreparedBatchnormInference>(
            compileBatchnormInference(_compiler, _loader, context, kernel, build),
            params,
            params.invVariance()->uid(),
            std::nullopt,
            build);
    }
};

class BatchnormInferenceVarianceDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& graph = context.graph;
        const auto& tensorMap = graph.getTensorMap();
        const auto& attributes
            = graph.getNodeWrapper(0)
                  .attributesAs<data_objects::BatchnormInferenceAttributesVarianceExt>();
        // A fused graph's activation follows the batchnorm.
        const auto params
            = graph.nodeCount() == 2
                  ? batchnorm::BatchnormFwdInferenceWithVarianceParams(
                        attributes,
                        graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>(),
                        tensorMap)
                  : batchnorm::BatchnormFwdInferenceWithVarianceParams(attributes, tensorMap);
        const auto& properties = context.deviceProperties;
        const auto build = batchnorm::buildBatchnormFwdInferenceWithVarianceKernel(
            params, properties.gcnArchName, properties.multiProcessorCount, properties.warpSize);

        return std::make_unique<PreparedBatchnormInference>(
            compileBatchnormInference(_compiler, _loader, context, kernel, build),
            params,
            params.estVariance()->uid(),
            hipdnn_plugin_sdk::makeScalarOperand(
                tensorMap, attributes.epsilon_tensor_uid(), "Epsilon"),
            build);
    }
};

// ---------------------------------------------------------------------------
// Dispatch: batchnorm training and backward
// ---------------------------------------------------------------------------

/// Compiles every launch of @p build, in order, out of one program.
std::vector<IngestorKernelCode>
    compileBatchnormLaunches(const compilation::IKernelCompiler& compiler,
                             const compilation::KpackKernelLoader& loader,
                             const MatchContext& context,
                             const KernelDefinition& kernel,
                             const batchnorm::BatchnormKernelBuild& build)
{
    std::vector<IngestorKernelCode> codes;
    codes.reserve(build.launches.size());
    for(const auto& launch : build.launches)
    {
        codes.push_back(compileBatchnormLaunch(compiler,
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

/// BatchNormFwdTrainSpatial, or for variant 2 the mean/variance, final mean/variance and
/// normalization launches, which stash their statistics in y. The saved statistics and the
/// running statistics are each optional, and choose the argument lists.
class PreparedBatchnormTraining : public PreparedHipMlops
{
public:
    PreparedBatchnormTraining(
        std::vector<IngestorKernelCode> codes,
        int variant,
        const batchnorm::BatchnormFwdTrainingParams& params,
        const data_objects::BatchnormAttributes& attributes,
        const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensorMap)
        : _codes(std::move(codes))
        , _variant(variant)
        , _x(params.x()->uid())
        , _y(params.optActivation().has_value() && params.activationOut() != nullptr
                 ? params.activationOut()->uid()
                 : params.y()->uid())
        , _scale(params.scale()->uid())
        , _bias(params.bias()->uid())
        , _epsilon(hipdnn_plugin_sdk::makeScalarOperand(
              tensorMap, attributes.epsilon_tensor_uid(), "Epsilon"))
        , _activation(activationArguments(params.optActivation()))
    {
        if(params.hasSaveMeanVariance())
        {
            _saved = std::array<int64_t, 2>{params.mean()->uid(), params.invVariance()->uid()};
        }
        if(params.hasRunningStats())
        {
            _running = std::array<int64_t, 4>{params.prevRunningMean()->uid(),
                                              params.prevRunningVariance()->uid(),
                                              params.nextRunningMean()->uid(),
                                              params.nextRunningVariance()->uid()};
            _momentum = hipdnn_plugin_sdk::makeScalarOperand(
                tensorMap, attributes.momentum_tensor_uid().value(), "Momentum");
        }
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _codes.front();
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const Buffers buffer(deviceBuffers, numDeviceBuffers);
        const auto resolve = [&](const hipdnn_plugin_sdk::ScalarOperand& operand) {
            return hipdnn_plugin_sdk::toDouble(
                hipdnn_plugin_sdk::resolveScalarOperand(operand, deviceBuffers, numDeviceBuffers));
        };
        void* x = buffer(_x);
        void* y = buffer(_y);
        void* scale = buffer(_scale);
        void* bias = buffer(_bias);
        const double epsilon = resolve(_epsilon);
        const double expAvgFactor = _momentum.has_value() ? resolve(*_momentum) : 0.0;
        void* saveMean = _saved.has_value() ? buffer((*_saved)[0]) : nullptr;
        void* saveInvVariance = _saved.has_value() ? buffer((*_saved)[1]) : nullptr;
        void* prevMean = _running.has_value() ? buffer((*_running)[0]) : nullptr;
        void* prevVariance = _running.has_value() ? buffer((*_running)[1]) : nullptr;
        void* nextMean = _running.has_value() ? buffer((*_running)[2]) : nullptr;
        void* nextVariance = _running.has_value() ? buffer((*_running)[3]) : nullptr;
        const auto [alpha, beta] = _activation;
        const bool saved = _saved.has_value();
        const bool running = _running.has_value();

        // Changing these argument lists means changing BatchNormFwdTrainSpatial.cpp with them.
        if(_variant != 2)
        {
            auto& kernel = _codes[0].kernelForStream(stream);
            if(saved && running)
            {
                kernel.launch(stream,
                              x,
                              y,
                              scale,
                              bias,
                              expAvgFactor,
                              prevMean,
                              prevVariance,
                              nextMean,
                              nextVariance,
                              epsilon,
                              saveMean,
                              saveInvVariance,
                              alpha,
                              beta);
            }
            else if(saved)
            {
                kernel.launch(
                    stream, x, y, scale, bias, epsilon, saveMean, saveInvVariance, alpha, beta);
            }
            else if(running)
            {
                kernel.launch(stream,
                              x,
                              y,
                              scale,
                              bias,
                              expAvgFactor,
                              prevMean,
                              prevVariance,
                              nextMean,
                              nextVariance,
                              epsilon,
                              alpha,
                              beta);
            }
            else
            {
                kernel.launch(stream, x, y, scale, bias, epsilon, alpha, beta);
            }
            return;
        }

        _codes[0].kernelForStream(stream).launch(stream, x, y);
        auto& finalKernel = _codes[1].kernelForStream(stream);
        if(saved && running)
        {
            finalKernel.launch(stream,
                               y,
                               expAvgFactor,
                               prevMean,
                               prevVariance,
                               nextMean,
                               nextVariance,
                               epsilon,
                               saveMean,
                               saveInvVariance);
        }
        else if(saved)
        {
            finalKernel.launch(stream, y, epsilon, saveMean, saveInvVariance);
        }
        else if(running)
        {
            finalKernel.launch(
                stream, y, expAvgFactor, prevMean, prevVariance, nextMean, nextVariance, epsilon);
        }
        else
        {
            finalKernel.launch(stream, y, epsilon);
        }
        _codes[2].kernelForStream(stream).launch(stream, x, y, scale, bias, alpha, beta);
    }

private:
    std::vector<IngestorKernelCode> _codes;
    int _variant;
    int64_t _x;
    int64_t _y;
    int64_t _scale;
    int64_t _bias;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _momentum;
    std::optional<std::array<int64_t, 2>> _saved;
    std::optional<std::array<int64_t, 4>> _running;
    std::pair<float, float> _activation;
};

/// BatchNormBwdSpatial, or for variant 2 the launches BatchnormBwdPlan::execute() indexes:
/// with saved statistics 0, 1 and 4, otherwise all five, stashing in dx. Without saved
/// statistics epsilon is the hand-written plan's default.
class PreparedBatchnormBackward : public PreparedHipMlops
{
public:
    PreparedBatchnormBackward(std::vector<IngestorKernelCode> codes,
                              int variant,
                              const batchnorm::BatchnormBwdParams& params)
        : _codes(std::move(codes))
        , _variant(variant)
        , _x(params.x()->uid())
        , _dy(params.dy()->uid())
        , _dx(params.dx()->uid())
        , _scale(params.scale()->uid())
        , _dscale(params.dscale()->uid())
        , _dbias(params.dbias()->uid())
        , _bias(uidOf(params.bias()))
        , _activation(activationArguments(params.optActivation()))
    {
        if(params.hasSavedStats())
        {
            _saved = std::array<int64_t, 2>{params.savedMean()->uid(),
                                            params.savedInvVariance()->uid()};
        }
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _codes.front();
    }

    void launch(hipStream_t stream,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const Buffers buffer(deviceBuffers, numDeviceBuffers);
        void* x = buffer(_x);
        void* dy = buffer(_dy);
        void* dx = buffer(_dx);
        void* scale = buffer(_scale);
        void* dscale = buffer(_dscale);
        void* dbias = buffer(_dbias);
        void* bias = buffer(_bias);
        void* savedMean = _saved.has_value() ? buffer((*_saved)[0]) : nullptr;
        void* savedInvVariance = _saved.has_value() ? buffer((*_saved)[1]) : nullptr;
        const double epsilon = hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON;
        const auto [alpha, beta] = _activation;
        const auto kernel
            = [&](size_t index) -> auto& { return _codes[index].kernelForStream(stream); };

        // Changing these argument lists means changing BatchNormBwdSpatial.cpp with them.
        if(_variant != 2)
        {
            if(_saved.has_value())
            {
                kernel(0).launch(stream,
                                 x,
                                 dy,
                                 dx,
                                 scale,
                                 bias,
                                 dscale,
                                 dbias,
                                 savedMean,
                                 savedInvVariance,
                                 alpha,
                                 beta);
            }
            else
            {
                kernel(0).launch(
                    stream, x, dy, dx, scale, bias, dscale, dbias, epsilon, alpha, beta);
            }
            return;
        }

        if(_saved.has_value())
        {
            kernel(0).launch(
                stream, x, dy, dx, scale, bias, savedMean, savedInvVariance, alpha, beta);
            kernel(1).launch(stream, dx, dscale, dbias);
            kernel(4).launch(stream,
                             x,
                             dy,
                             dx,
                             scale,
                             bias,
                             dscale,
                             dbias,
                             savedMean,
                             savedInvVariance,
                             alpha,
                             beta);
        }
        else
        {
            kernel(0).launch(stream, x, dx);
            kernel(1).launch(stream, dx, epsilon);
            kernel(2).launch(stream, x, dy, dx, scale, bias, alpha, beta);
            kernel(3).launch(stream, dx, dscale, dbias);
            kernel(4).launch(stream, x, dy, dx, scale, bias, dscale, dbias, alpha, beta);
        }
    }

private:
    std::vector<IngestorKernelCode> _codes;
    int _variant;
    int64_t _x;
    int64_t _dy;
    int64_t _dx;
    int64_t _scale;
    int64_t _dscale;
    int64_t _dbias;
    std::optional<int64_t> _bias;
    std::optional<std::array<int64_t, 2>> _saved;
    std::pair<float, float> _activation;
};

class BatchnormTrainingDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& graph = context.graph;
        const auto& tensorMap = graph.getTensorMap();
        const auto& attributes
            = graph.getNodeWrapper(0).attributesAs<data_objects::BatchnormAttributes>();
        // A fused graph's activation follows the batchnorm.
        const auto params
            = graph.nodeCount() == 2
                  ? batchnorm::BatchnormFwdTrainingParams(
                        attributes,
                        graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>(),
                        tensorMap)
                  : batchnorm::BatchnormFwdTrainingParams(attributes, tensorMap);
        const auto& properties = context.deviceProperties;
        const auto build = batchnorm::buildBatchnormFwdTrainingKernel(
            params, properties.gcnArchName, properties.multiProcessorCount, properties.warpSize);

        return std::make_unique<PreparedBatchnormTraining>(
            compileBatchnormLaunches(_compiler, _loader, context, kernel, build),
            build.variant,
            params,
            attributes,
            tensorMap);
    }
};

class BatchnormBackwardDispatchHandler : public HipMlopsDispatchHandler
{
public:
    using HipMlopsDispatchHandler::HipMlopsDispatchHandler;

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        const auto& graph = context.graph;
        const auto& tensorMap = graph.getTensorMap();
        // A fused graph is the inference, the activation's backward, then the backward.
        const bool fused = graph.nodeCount() == 3;
        const auto& backward = graph.getNodeWrapper(fused ? 2 : 0)
                                   .attributesAs<data_objects::BatchnormBackwardAttributes>();
        const auto params
            = fused ? batchnorm::BatchnormBwdParams(
                          backward,
                          graph.getNodeWrapper(1).attributesAs<data_objects::PointwiseAttributes>(),
                          graph.getNodeWrapper(0)
                              .attributesAs<data_objects::BatchnormInferenceAttributes>(),
                          tensorMap)
                    : batchnorm::BatchnormBwdParams(backward, tensorMap);
        const auto build
            = batchnorm::buildBatchnormBwdKernel(params, context.deviceProperties.gcnArchName);

        return std::make_unique<PreparedBatchnormBackward>(
            compileBatchnormLaunches(_compiler, _loader, context, kernel, build),
            build.variant,
            params);
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
    scope.add("hipkernel.hip_mlops.rmsnorm_fwd_match", &operationMatches<RMSNORM_FWD>);
    scope.add("hipkernel.hip_mlops.rmsnorm_bwd_match", &operationMatches<RMSNORM_BWD>);
    scope.add("hipkernel.hip_mlops.batchnorm_inference_match",
              &operationMatches<BATCHNORM_INFERENCE>);
    scope.add("hipkernel.hip_mlops.batchnorm_inference_variance_match",
              &operationMatches<BATCHNORM_INFERENCE_VARIANCE>);
    scope.add("hipkernel.hip_mlops.batchnorm_training_match",
              &operationMatches<BATCHNORM_TRAINING>);
    scope.add("hipkernel.hip_mlops.batchnorm_backward_match",
              &operationMatches<BATCHNORM_BACKWARD>);

    scope.add("hipkernel.hip_mlops.resample_fwd_dispatch",
              &dispatchHandler<ResampleFwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.resample_bwd_dispatch",
              &dispatchHandler<ResampleBwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.layernorm_fwd_dispatch",
              &dispatchHandler<LayernormFwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.layernorm_bwd_dispatch",
              &dispatchHandler<LayernormBwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.rmsnorm_fwd_dispatch",
              &dispatchHandler<RmsnormFwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.rmsnorm_bwd_dispatch",
              &dispatchHandler<RmsnormBwdDispatchHandler>());
    scope.add("hipkernel.hip_mlops.batchnorm_inference_dispatch",
              &dispatchHandler<BatchnormInferenceDispatchHandler>());
    scope.add("hipkernel.hip_mlops.batchnorm_inference_variance_dispatch",
              &dispatchHandler<BatchnormInferenceVarianceDispatchHandler>());
    scope.add("hipkernel.hip_mlops.batchnorm_training_dispatch",
              &dispatchHandler<BatchnormTrainingDispatchHandler>());
    scope.add("hipkernel.hip_mlops.batchnorm_backward_dispatch",
              &dispatchHandler<BatchnormBackwardDispatchHandler>());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
