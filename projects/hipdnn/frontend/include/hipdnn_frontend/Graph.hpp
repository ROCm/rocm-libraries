// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Graph.hpp
 * @brief Main Graph class for building and executing deep learning operations
 *
 * This header defines the Graph class — hipDNN's top-level API for describing,
 * compiling, and running DNN operations on AMD GPUs. It is included
 * automatically via `#include <hipdnn_frontend.hpp>`.
 *
 * @section graph_overview Overview
 *
 * The Graph class provides a fluent API for:
 * - Creating tensor descriptors (shape + dtype, no data yet)
 * - Adding operations (convolution forward/dgrad/wgrad, batchnorm
 *   forward/backward/inference, layernorm, rmsnorm, pointwise, matmul,
 *   scaled dot-product attention, block-scale quantize/dequantize)
 * - Building (compiling) an execution plan for the GPU
 * - Executing the plan with real device pointers
 *
 * @section graph_workflow Typical Workflow
 *
 * @code{.cpp}
 * using namespace hipdnn_frontend;
 * using namespace hipdnn_frontend::graph;
 *
 * // 1. Create and configure the graph
 * Graph graph;
 * graph.set_io_data_type(DataType::HALF)
 *      .set_compute_data_type(DataType::FLOAT)
 *      .set_name("my_conv_graph");
 *
 * // 2. Create input tensors
 * auto x = Graph::tensor(TensorAttributes()
 *              .set_dim({N, C, H, W})
 *              .set_stride({C*H*W, H*W, W, 1})
 *              .set_uid(0));
 * auto w = Graph::tensor(TensorAttributes()
 *              .set_dim({K, C, R, S})
 *              .set_uid(1));
 *
 * // 3. Add operations
 * auto y = graph.conv_fprop(x, w, ConvFpropAttributes()
 *              .set_padding({1, 1})
 *              .set_stride({1, 1}));
 * y->set_output(true).set_uid(2);
 *
 * // 4. Build and execute
 * hipdnnHandle_t handle;
 * hipdnnCreate(&handle);
 * graph.build(handle);
 *
 * int64_t workspaceSize;
 * graph.get_workspace_size(workspaceSize);
 * void* workspace;
 * hipMalloc(&workspace, workspaceSize);
 *
 * std::unordered_map<int64_t, void*> variantPack = {
 *     {0, d_input}, {1, d_weights}, {2, d_output}
 * };
 * graph.execute(handle, variantPack, workspace);
 * @endcode
 */

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <unordered_set>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#endif
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ConvolutionFpropUnpacker.hpp>
#include <hipdnn_frontend/detail/CreateBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/GraphDetail.hpp>
#include <hipdnn_frontend/detail/GraphOverrideValidation.hpp>
#include <hipdnn_frontend/detail/GraphPacker.hpp>
#include <hipdnn_frontend/detail/GraphUnpacker.hpp>
#include <hipdnn_frontend/detail/KnobPacker.hpp>
#include <hipdnn_frontend/detail/KnobUnpacker.hpp>
#include <hipdnn_frontend/detail/OperationUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/VariantPackHelpers.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNodeVarianceExt.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/BlockScaleDequantizeNode.hpp>
#include <hipdnn_frontend/node/BlockScaleQuantizeNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/CustomOpNode.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/RMSNormBackwardNode.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>
#endif
#include <hipdnn_frontend/node/detail/TopologicalSortingUtils.hpp>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <nlohmann/json.hpp>
#endif

namespace hipdnn_frontend::graph
{

#ifdef HIPDNN_ENABLE_SDPA
/// Runtime shape/stride override for one tensor in the map execute overload.
struct OverrideEntry
{
    /// Runtime shape values.
    std::vector<int64_t> shape;
    /// Runtime strides.
    std::vector<int64_t> stride;
};
#endif // HIPDNN_ENABLE_SDPA

/**
 * @class Graph
 * @brief The main class for building and executing hipDNN computational graphs
 *
 * You describe **what** operations to run (convolution, batchnorm,
 * pointwise, matmul, layernorm, rmsnorm) and the library figures out
 * **how** to execute them efficiently on AMD GPUs.
 *
 * **Typical workflow:**
 * | Step | What you do | hipDNN call |
 * |------|-------------|-------------|
 * | 1. Describe tensor shapes | Define dims, strides, dtype | `Graph::tensor(attrs)` |
 * | 2. Add operations | Wire inputs to outputs | `graph.conv_fprop(x, w, ...)` |
 * | 3. Compile for GPU | Select engine, build plan | `graph.build(handle)` |
 * | 4. Execute | Pass device pointers | `graph.execute(handle, ptrs, ws)` |
 *
 * The Graph uses a **fluent API** — setter methods return `*this` so
 * you can chain calls:
 * @code{.cpp}
 * graph.set_io_data_type(DataType::HALF)
 *      .set_compute_data_type(DataType::FLOAT)
 *      .set_name("my_graph");
 * @endcode
 *
 * @see hipdnn_frontend::graph::TensorAttributes, hipdnn_frontend::graph::ConvFpropAttributes,
 *      hipdnn_frontend::graph::BatchnormAttributes, hipdnn_frontend::graph::PointwiseAttributes
 */
class Graph : public INode
{
private:
    std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> _graphDesc;
    bool _graphDescFinalized = false;
    std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> _engineConfigDesc;

protected:
    /// Owns the built execution plan descriptor. Protected so test subclasses
    /// can assert plan presence.
    std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> _executionPlanDesc;

    /// The execution plan is finalized only after build_plans()/build(); a plan
    /// descriptor that has merely been created is valid but not yet finalized and
    /// carries no execution context. serialize() gates the combined graph+plan
    /// container on this so an unfinalized plan is never embedded.
    bool _executionPlanFinalized = false;

    /// Engine id backing the current execution plan: captured at plan build, or
    /// recovered from the plan on deserialize. Queried at serialize time for
    /// execution-plan-serialization support. Reset when the plan is dropped.
    std::optional<int64_t> _selectedEngineId;

private:
    std::optional<int64_t> _preferredEngineId;

    bool _isOverrideShapeEnabled = false;

    /// Apply validated knob settings to the engine config descriptor via
    /// the descriptor-based C API path.
    Error applyKnobSettingsToEngineConfig(const std::vector<KnobSetting>& validatedSettings)
    {
        if(validatedSettings.empty())
        {
            return {ErrorCode::OK, ""};
        }

        return detail::applyKnobSettingsViaDescriptors(_engineConfigDesc->get(), validatedSettings);
    }

    /// Assemble the serialized binary representation of the graph.
    ///
    /// A serializable built plan is emitted alongside the graph in one container
    /// blob; otherwise the bare graph blob is emitted, byte-identical to a graph
    /// that never had a plan built.
    Error assembleSerializedBlob(std::vector<uint8_t>& data) const
    {
        if(!hasValidGraphDesc())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Graph has no backend descriptor. "
                    "Call build_operation_graph() first, or use the non-const "
                    "serialize() overload for auto-lowering."};
        }

        const bool planFinalized
            = _executionPlanDesc && _executionPlanDesc->valid() && _executionPlanFinalized;
        const bool planSerializable = planFinalized && _selectedEngineId.has_value()
                                      && engineSupportsPlanSerialization(*_selectedEngineId);
        if(planSerializable)
        {
            size_t blobByteSize = 0;
            HIPDNN_RETURN_ON_BACKEND_FAILURE(
                detail::hipdnnBackend()->backendGetSerializedBinaryGraphAndPlanExt(
                    _graphDesc->get(), _executionPlanDesc->get(), 0, &blobByteSize, nullptr),
                "Failed to query serialized graph and plan size");

            if(blobByteSize == 0)
            {
                return {ErrorCode::HIPDNN_BACKEND_ERROR,
                        "Backend returned zero-length binary graph and plan"};
            }

            data.resize(blobByteSize);
            HIPDNN_RETURN_ON_BACKEND_FAILURE(
                detail::hipdnnBackend()->backendGetSerializedBinaryGraphAndPlanExt(
                    _graphDesc->get(),
                    _executionPlanDesc->get(),
                    blobByteSize,
                    &blobByteSize,
                    data.data()),
                "Failed to serialize graph and plan");

            return {};
        }

        if(planFinalized)
        {
            // Plan exists but its engine cannot serialize it; fall through to the
            // byte-identical legacy bare-graph blob so legacy consumers keep working.
            HIPDNN_FE_LOG_WARN(
                "Execution plan was not captured during serialization: the engine "
                "does not support execution plan serialization; serializing the graph only.");
        }
        else if(_executionPlanDesc && _executionPlanDesc->valid())
        {
            // Plan descriptor was created but never finalized, so it carries no
            // execution context to embed; serialize the graph only.
            HIPDNN_FE_LOG_INFO("Execution plan has not been finalized; call build_plans() (or "
                               "build()) before serialize() to embed it. Serializing the graph "
                               "only.");
        }

        size_t graphByteSize = 0;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetSerializedBinaryGraphExt(
                _graphDesc->get(), 0, &graphByteSize, nullptr),
            "Failed to query serialized graph size");

        if(graphByteSize == 0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Backend returned zero-length binary graph"};
        }

        data.resize(graphByteSize);
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetSerializedBinaryGraphExt(
                _graphDesc->get(), graphByteSize, &graphByteSize, data.data()),
            "Failed to serialize graph");

        return {};
    }

    Error validateAndFilterKnobSettings(const std::vector<KnobSetting>& settings,
                                        const std::unordered_map<KnobType_t, Knob>& existingKnobs,
                                        std::vector<KnobSetting>& validatedSettings)
    {
        validatedSettings.clear();
        validatedSettings.reserve(settings.size());

        for(const auto& setting : settings)
        {
            auto knobIt = existingKnobs.find(setting.knobId());
            if(knobIt == existingKnobs.end())
            {
                HIPDNN_FE_LOG_WARN("Ignoring knob " << setting.knobId()
                                                    << " when creating execution plan for graph "
                                                    << graph_attributes.get_name()
                                                    << ".  Engine doesn't support chosen knob.");
                continue;
            }

            const auto& knob = knobIt->second;

            if(knob.isDeprecated())
            {
                HIPDNN_FE_LOG_WARN("Knob " << knob.knobId() << " has been marked as deprecated.");
            }

            HIPDNN_CHECK_ERROR(knob.validate(setting));

            validatedSettings.emplace_back(setting);
        }

        return {ErrorCode::OK, ""};
    }

    /// Set the graph descriptor and its finalization state atomically
    void setGraphDesc(std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> desc, bool finalized)
    {
        _graphDesc = std::move(desc);
        _graphDescFinalized = finalized;
    }

    /// Clear the graph descriptor and finalization state
    void resetGraphDesc()
    {
        _graphDesc.reset();
        _graphDescFinalized = false;
    }

    /// Finalize an existing unfinalized descriptor by adding a handle and finalizing
    Error finalizeGraphDescWithHandle(hipdnnHandle_t handle)
    {
        auto status
            = detail::hipdnnBackend()->backendSetAttribute(_graphDesc->get(),
                                                           HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                                           HIPDNN_TYPE_HANDLE,
                                                           1,
                                                           static_cast<const void*>(&handle));
        HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to set handle on graph descriptor");

        status = detail::hipdnnBackend()->backendFinalize(_graphDesc->get());
        HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to finalize graph descriptor");

        _graphDescFinalized = true;
        return {};
    }

    Error ensureLowered()
    {
        if(!hasValidGraphDesc())
        {
            HIPDNN_FE_LOG_INFO("Graph not lowered — auto-lowering for serialization");
            HIPDNN_CHECK_ERROR(lower_to_backend());
        }
        return {};
    }

    /// Check if we have a valid graph descriptor (may or may not be finalized)
    bool hasValidGraphDesc() const
    {
        return _graphDesc && _graphDesc->valid();
    }

    /// Check if we have a usable (valid + finalized) graph descriptor
    bool hasReadyGraphDesc() const
    {
        return hasValidGraphDesc() && _graphDescFinalized;
    }

    Error finalizeExecutionPlanDescriptor()
    {
        // Finalize engine config after knobs have been set
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(_engineConfigDesc->get()),
            "Failed to finalize engine config descriptor");

        // Create execution plan descriptor
        _executionPlanDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        _executionPlanFinalized = false;

        if(!_executionPlanDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend execution descriptor."};
        }

        return {ErrorCode::OK, ""};
    }

    void assignUnsetTensorUids()
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);
        auto usedIds = getUsedIds(allTensors);
        populateHipdnnTensorIds(allTensors, usedIds);
    }

    /// Lower the frontend graph into a backend descriptor without a handle.
    /// The descriptor is serializable but cannot be used for engine selection
    /// or execution. Clears any existing descriptor before re-lowering.
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error lower_to_backend()
    {
        HIPDNN_FE_LOG_INFO("Lowering graph to backend descriptor " << graph_attributes.get_name());
        return lowerGraphToDescriptors();
    }

    /// Shared lowering logic for lower_to_backend() and
    /// build_operation_graph_via_descriptors(). Assigns UIDs, validates,
    /// creates backend operation descriptors, and assembles the graph
    /// descriptor. When @p handle has a value the descriptor is finalized
    /// with a backend handle (ready for engine selection / execution);
    /// otherwise it is left unfinalized (serializable only).
    Error lowerGraphToDescriptors(std::optional<hipdnnHandle_t> handle = std::nullopt)
    {
        assignUnsetTensorUids();

        // Validate before resetting _graphDesc so the existing descriptor
        // is preserved if validation fails.
        HIPDNN_CHECK_ERROR(validate());

        if(_graphDesc)
        {
            HIPDNN_FE_LOG_INFO("Purging existing graph descriptor before re-lowering");
        }
        resetGraphDesc();

        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor> tensorDescs;
        std::vector<detail::ScopedHipdnnBackendDescriptor> operations;

        for(const auto& node : _sub_nodes)
        {
            HIPDNN_CHECK_ERROR(node->create_operation(tensorDescs, operations));
        }

        if(operations.empty())
        {
            return {ErrorCode::INVALID_VALUE, "No operations created for graph"};
        }

        // Data types are optional: NOT_SET values produce nullopt from
        // toHipdnnDataType() and are skipped by assembleGraphDescriptor().
        // This is intentional -- graphs can have unset graph-level data types
        // as long as individual tensors have their types set.
        std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> desc;
        if(handle.has_value())
        {
            HIPDNN_CHECK_ERROR(detail::assembleGraphDescriptor(
                operations,
                handle.value(),
                toHipdnnDataType(graph_attributes.get_compute_data_type()),
                toHipdnnDataType(graph_attributes.get_intermediate_data_type()),
                toHipdnnDataType(graph_attributes.get_io_data_type()),
                _preferredEngineId,
                _isOverrideShapeEnabled,
                graph_attributes.get_name(),
                desc));
            setGraphDesc(std::move(desc), true);
        }
        else
        {
            HIPDNN_CHECK_ERROR(detail::assembleGraphDescriptor(
                operations,
                toHipdnnDataType(graph_attributes.get_compute_data_type()),
                toHipdnnDataType(graph_attributes.get_intermediate_data_type()),
                toHipdnnDataType(graph_attributes.get_io_data_type()),
                _preferredEngineId,
                _isOverrideShapeEnabled,
                graph_attributes.get_name(),
                desc));
            setGraphDesc(std::move(desc), false);
        }

        return {ErrorCode::OK, ""};
    }

    static std::shared_ptr<TensorAttributes> outputTensor(const std::string& name)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_name(name).set_is_virtual(true);
        return tensor;
    }

    /// Whether the engine advertises the SUPPORTS_EXECUTION_PLAN_SERIALIZATION
    /// behavior note. A query error or a missing note returns false, so probing
    /// the capability never fails a plan build.
    bool engineSupportsPlanSerialization(int64_t engineId) const
    {
        std::vector<BehaviorNote> notes;
        const Error err = get_behavior_notes_for_engine(engineId, notes);
        if(err.is_bad())
        {
            return false;
        }

        return std::find(
                   notes.begin(), notes.end(), BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION)
               != notes.end();
    }

    Error initializeEngineConfig(hipdnnBackendDescriptor_t engineHeuristicDesc)
    {
        // The backend's SelectionHeuristic::Config built-in honors
        // HIPDNN_HEUR_CONFIG_PATH inside the policy loop, so the
        // heuristic-ranked list already reflects env/config-file overrides.
        // The explicit Graph.preferred_engine_id setter is honored here as a
        // post-hoc reorder: if the user pinned an engine and it appears in
        // the ranked list, prefer it over index 0; otherwise log and fall
        // back to the heuristic's choice.
        std::vector<std::unique_ptr<detail::ScopedHipdnnBackendDescriptor>> engineConfigs;
        std::vector<int64_t> engineIds;
        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::getEngineConfigs(
            engineConfigs, engineIds, engineHeuristicDesc, _preferredEngineId.has_value()));

        size_t selectedIndex = 0;
        if(_preferredEngineId.has_value())
        {
            const int64_t preferredId = _preferredEngineId.value();
            auto it = std::find(engineIds.begin(), engineIds.end(), preferredId);
            if(it != engineIds.end())
            {
                selectedIndex = static_cast<size_t>(std::distance(engineIds.begin(), it));
                HIPDNN_FE_LOG_INFO("Preferred engine id "
                                   << preferredId << " found, using it for execution plan.");
            }
            else
            {
                HIPDNN_FE_LOG_INFO("Preferred engine id "
                                   << preferredId
                                   << " not found, using top engine config instead.");
            }
        }

        const int64_t selectedEngineId = engineIds[selectedIndex];
        HIPDNN_FE_LOG_INFO("Selected engine id " << selectedEngineId << " for execution plan.");
        _engineConfigDesc = std::move(engineConfigs[selectedIndex]);

        // Record the engine for the upcoming plan; serialize() queries it for
        // plan-serialization support.
        _selectedEngineId = selectedEngineId;

        return {ErrorCode::OK, ""};
    }

    /// Initialize engine config for a specific engine ID.
    /// @param engineId The engine to configure
    /// @note This method does NOT finalize the engine config. The caller must
    ///       finalize after setting any knobs on the config.
    Error initializeEngineConfig(int64_t engineId)
    {
        detail::ScopedHipdnnBackendDescriptor engineDesc;

        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineDescriptorForGraph(
            engineDesc, _graphDesc->get(), engineId));

        auto engineConfigDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);

        HIPDNN_RETURN_ON_BACKEND_FAILURE(detail::hipdnnBackend()->backendSetAttribute(
                                             engineConfigDesc->get(),
                                             HIPDNN_ATTR_ENGINECFG_ENGINE,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             1,
                                             static_cast<const void*>(&engineDesc.get())),
                                         "Failed to set engine on the engine config descriptor.");

        _engineConfigDesc = std::move(engineConfigDesc);

        // Record the engine for the upcoming plan; serialize() queries it for
        // plan-serialization support.
        _selectedEngineId = engineId;

        return {ErrorCode::OK, ""};
    }

    detail::GraphStructure buildAdjacencyList(
        const std::unordered_map<std::shared_ptr<TensorAttributes>, size_t>& tensorToOriginNode)
        const
    {
        const size_t nodeCount = _sub_nodes.size();
        detail::GraphStructure structure;
        structure.adjacencyList.resize(nodeCount);

        for(size_t inputNodeIndex = 0; inputNodeIndex < nodeCount; ++inputNodeIndex)
        {
            auto inputs = _sub_nodes[inputNodeIndex]->getNodeInputTensorAttributes();
            for(auto& input : inputs)
            {
                auto it = tensorToOriginNode.find(input);
                if(it != tensorToOriginNode.end())
                {
                    const size_t outputNodeIndex = it->second;
                    structure.adjacencyList[outputNodeIndex].push_back(inputNodeIndex);
                }
            }
        }

        return structure;
    }

    std::unordered_map<std::shared_ptr<TensorAttributes>, size_t> buildTensorToOriginNodeMap() const
    {
        std::unordered_map<std::shared_ptr<TensorAttributes>, size_t> tensorToOriginNode;
        const size_t nodeCount = _sub_nodes.size();

        for(size_t i = 0; i < nodeCount; ++i)
        {
            auto outputs = _sub_nodes[i]->getNodeOutputTensorAttributes();
            for(auto& output : outputs)
            {
                tensorToOriginNode[output] = i;
            }
        }

        return tensorToOriginNode;
    }

    void reorderNodesTopologically(const std::vector<size_t>& topologicalOrder)
    {
        std::vector<std::shared_ptr<INode>> reorderedNodes;
        reorderedNodes.reserve(topologicalOrder.size());

        for(auto idx : topologicalOrder)
        {
            reorderedNodes.push_back(_sub_nodes[idx]);
        }

        _sub_nodes = std::move(reorderedNodes);
    }

    static std::unordered_set<int64_t>
        getUsedIds(const std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors)
    {
        std::unordered_set<int64_t> usedIds;
        for(const auto& tensor : allTensors)
        {
            if(tensor && tensor->has_uid())
            {
                usedIds.insert(tensor->get_uid());
            }
        }
        return usedIds;
    }

    static int64_t getUnusedTensorUid(int64_t& currentTensorId,
                                      std::unordered_set<int64_t>& usedIds)
    {
        while(usedIds.find(currentTensorId) != usedIds.end())
        {
            ++currentTensorId;
        }
        usedIds.insert(currentTensorId);
        return currentTensorId++;
    }

    static void populateHipdnnTensorIds(
        const std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors,
        std::unordered_set<int64_t>& usedIds)
    {
        int64_t currentTensorId = 0;

        for(const auto& tensor : allTensors)
        {
            if(!tensor)
            {
                continue;
            }

            if(!tensor->has_uid())
            {
                tensor->set_uid(getUnusedTensorUid(currentTensorId, usedIds));
            }
        }
    }

    static Error checkTensorUidsSetImpl(
        const std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors)
    {
        std::vector<std::string> missingUidTensors;

        for(const auto& tensor : allTensors)
        {
            if(tensor && !tensor->has_uid())
            {
                auto name = tensor->get_name();
                missingUidTensors.push_back(name.empty() ? "(unnamed)" : name);
            }
        }

        if(!missingUidTensors.empty())
        {
            std::string errorMsg = "Tensors without UIDs: ";
            for(const auto& name : missingUidTensors)
            {
                errorMsg += name + ", ";
            }
            errorMsg.pop_back();
            errorMsg.pop_back();
            return {ErrorCode::ATTRIBUTE_NOT_SET, errorMsg};
        }

        return {ErrorCode::OK, ""};
    }

    static Error checkNoDuplicateTensorIdsImpl(
        const std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors)
    {
        std::unordered_set<int64_t> seenUids;
        std::unordered_set<int64_t> duplicateUids;

        for(const auto& tensor : allTensors)
        {
            if(tensor && tensor->has_uid())
            {
                auto uid = tensor->get_uid();
                if(!seenUids.insert(uid).second)
                {
                    duplicateUids.insert(uid);
                }
            }
        }

        if(!duplicateUids.empty())
        {
            std::string errorMsg = "Duplicate tensor UIDs found in the graph: ";
            for(const auto& uid : duplicateUids)
            {
                errorMsg += std::to_string(uid) + ", ";
            }
            errorMsg.erase(errorMsg.length() - 2);
            return {ErrorCode::INVALID_VALUE, errorMsg};
        }

        return {ErrorCode::OK, ""};
    }

    std::pair<std::unordered_set<std::shared_ptr<TensorAttributes>>,
              std::unordered_set<std::shared_ptr<TensorAttributes>>>
        getGraphInputTensorAttributesAndRemainder() const
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allNodeOutputs;
        std::unordered_set<std::shared_ptr<TensorAttributes>> graphInputs;

        auto collectNodeOutputs = [&](const INode& node) {
            auto nodeOutputs = node.getNodeOutputTensorAttributes();
            allNodeOutputs.insert(nodeOutputs.begin(), nodeOutputs.end());
        };
        auto collectGraphInputs = [&](const INode& node) {
            auto nodeInputs = node.getNodeInputTensorAttributes();
            std::copy_if(nodeInputs.begin(),
                         nodeInputs.end(),
                         std::inserter(graphInputs, graphInputs.end()),
                         [&](const auto& nodePtr) { return allNodeOutputs.count(nodePtr) == 0; });
        };

        visit(collectNodeOutputs);
        visit(collectGraphInputs);

        return {graphInputs, allNodeOutputs};
    }

public:
    /**
     * @brief Construct an empty Graph
     */
    Graph()
        : INode(GraphAttributes{})
    {
        HIPDNN_FE_LOG_INFO("Creating new Graph instance");
    }

    // Copy operations disabled via INode base class
    // Move operations defaulted - automatically handles all members
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    /**
     * @brief Validate the graph structure and tensor configurations
     *
     * Validates that:
     * - No duplicate tensor UIDs exist
     * - Graph is a valid DAG (no cycles)
     * - Graph is a single connected component
     * - All tensor attributes are set (dims, type, strides)
     * - All operation nodes have valid configurations
     *
     * @return Error with ErrorCode::INVALID_VALUE or ErrorCode::ATTRIBUTE_NOT_SET
     *         on failure. Call get_message() for the specific failure reason.
     */
    Error validate()
    {
        HIPDNN_FE_LOG_INFO("Validating graph " << graph_attributes.get_name());

        auto [inputTensors, remainingTensors] = getGraphInputTensorAttributesAndRemainder();

        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors = inputTensors;
        allTensors.insert(remainingTensors.begin(), remainingTensors.end());

        HIPDNN_CHECK_ERROR(checkNoDuplicateTensorIdsImpl(allTensors));

        HIPDNN_CHECK_ERROR(topologicallySortGraph());

        for(const auto& tensor : inputTensors)
        {
            tensor->fill_from_context(graph_attributes);
            HIPDNN_CHECK_ERROR(tensor->validate());
        }

        HIPDNN_CHECK_ERROR(validateSubtree());

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Verify that no two tensors in the graph share the same UID
     * @return ErrorCode::OK if all UIDs are unique, or ErrorCode::INVALID_VALUE
     *         if duplicates exist. Call get_message() for the duplicate UIDs.
     */
    Error checkNoDuplicateTensorIds()
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        return checkNoDuplicateTensorIdsImpl(allTensors);
    }

    /**
     * @brief Check that all tensors in the graph have UIDs assigned
     * @return ErrorCode::OK if all tensors have UIDs, or
     *         ErrorCode::ATTRIBUTE_NOT_SET if any are missing. Call
     *         get_message() for the affected tensors.
     */
    Error checkTensorUidsSet() const
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        return checkTensorUidsSetImpl(allTensors);
    }

    /**
     * @brief Get all tensors in the graph indexed by UID
     *
     * Tensors without UIDs are skipped.
     *
     * @return Map from tensor UID to tensor attributes
     */
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> getTensorsByUid() const
    {
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> result;

        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        for(const auto& tensor : allTensors)
        {
            if(tensor && tensor->has_uid())
            {
                result[tensor->get_uid()] = tensor;
            }
        }

        return result;
    }

    /**
     * @brief Get all tensors in the graph indexed by name
     *
     * Tensors without names are skipped.
     *
     * @return Map from tensor name to tensor attributes
     */
    std::unordered_map<std::string, std::shared_ptr<TensorAttributes>> getTensorsByName() const
    {
        std::unordered_map<std::string, std::shared_ptr<TensorAttributes>> result;

        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        for(const auto& tensor : allTensors)
        {
            if(tensor && !tensor->get_name().empty())
            {
                result[tensor->get_name()] = tensor;
            }
        }

        return result;
    }

    /**
     * @brief Topologically sort the graph nodes
     *
     * Reorders internal nodes so that every node appears after its
     * dependencies.
     *
     * @return ErrorCode::OK on success, or ErrorCode::INVALID_VALUE if the
     *         graph contains a cycle or multiple disconnected components. Call
     *         get_message() for the specific failure reason.
     */
    Error topologicallySortGraph()
    {
        const size_t nodeCount = _sub_nodes.size();

        if(nodeCount == 0)
        {
            return {ErrorCode::OK, ""};
        }

        auto tensorToOriginNode = buildTensorToOriginNodeMap();
        auto graphStructure = buildAdjacencyList(tensorToOriginNode);

        auto sortResult = detail::performTopologicalSortWithComponentDetection(graphStructure);

        if(sortResult.hasCycle)
        {
            return {ErrorCode::INVALID_VALUE, "Graph contains a cycle, cannot be sorted."};
        }

        if(sortResult.componentCount > 1)
        {
            return {ErrorCode::INVALID_VALUE,
                    "Graph contains multiple disconnected components, please split the graph into "
                    "individual graphs"};
        }

        reorderNodesTopologically(sortResult.order);

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Build the operation graph descriptor
     *
     * Creates the backend operation graph descriptor from the frontend graph
     * representation. Typically called internally by build().
     *
     * @param handle The hipDNN handle
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR /
     *         ErrorCode::INVALID_VALUE on failure. Call get_message() for the
     *         specific failure reason.
     */
    Error build_operation_graph(hipdnnHandle_t handle) // NOLINT(readability-identifier-naming)
    {
        return build_operation_graph_via_descriptors(handle);
    }

protected:
    /// Get knobs for a specific engine, always using the descriptor-based
    /// C-API path. Exposed as protected so tests can exercise this path
    /// directly without relying on the public `getKnobs()` method.
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error get_knobs_for_engine_via_descriptors(int64_t engineId, std::vector<Knob>& knobs) const
    {
        if(!hasValidGraphDesc())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot get knobs "
                    "for engine."};
        }

        detail::ScopedHipdnnBackendDescriptor engineDesc;

        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineDescriptorForGraph(
            engineDesc, _graphDesc->get(), engineId));

        return detail::unpackKnobsFromDescriptors(engineDesc.get(), knobs);
    }

    // Returns the raw backend graph descriptor, or nullptr if the graph has not been built.
    // NOLINTNEXTLINE(readability-identifier-naming)
    hipdnnBackendDescriptor_t get_raw_graph_descriptor() const
    {
        return hasValidGraphDesc() ? _graphDesc->get() : nullptr;
    }

    /// Builds the operation graph with a handle for engine selection and
    /// execution. Clears any existing descriptor and re-lowers from the
    /// frontend nodes.
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error build_operation_graph_via_descriptors(hipdnnHandle_t handle)
    {
        HIPDNN_FE_LOG_INFO("Building operation graph via descriptors "
                           << graph_attributes.get_name());
        return lowerGraphToDescriptors(handle);
    }

    /// Reconstruct the Graph from a finalized backend OperationGraph descriptor.
    ///
    /// Extracts operations and graph-level data types from a backend descriptor
    /// and rebuilds the frontend Graph representation. Tensors are shared across
    /// operations via UID-based lookup.
    ///
    /// @param graphDesc A finalized backend OperationGraph descriptor
    /// @return ErrorCode::OK on success, or ErrorCode::INVALID_VALUE /
    ///         ErrorCode::HIPDNN_BACKEND_ERROR on failure. Call get_message()
    ///         for the specific failure reason.
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error fromBackendDescriptor(hipdnnBackendDescriptor_t graphDesc)
    {
        std::vector<std::shared_ptr<graph::INode>> tempNodes;
        graph::GraphAttributes tempAttrs;
        std::optional<int64_t> tempEngineId;
        bool tempOverrideShapeEnabled = false;

        HIPDNN_CHECK_ERROR(detail::unpackGraphDescriptor(
            graphDesc, tempNodes, tempAttrs, tempEngineId, tempOverrideShapeEnabled));

        _sub_nodes = std::move(tempNodes);
        graph_attributes = std::move(tempAttrs);
        _preferredEngineId = tempEngineId;
        _isOverrideShapeEnabled = tempOverrideShapeEnabled;

        // The frontend state has been fully replaced from the backend descriptor.
        // Any cached backend descriptors are stale and must be cleared. The caller
        // must call build_operation_graph() to rebuild them.
        resetGraphDesc();
        _engineConfigDesc.reset();
        _executionPlanDesc.reset();
        _executionPlanFinalized = false;
        _selectedEngineId.reset();
        return {};
    }

public:
    /**
     * @brief Get available configuration knobs for a specific engine
     *
     * @param engineId The engine ID to query
     * @param knobs Output vector of available Knob objects
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR
     *         if the graph has not been built. Call get_message() for the
     *         specific failure reason.
     *
     * @see hipdnn_frontend::Knob, hipdnn_frontend::KnobSetting
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error get_knobs_for_engine(int64_t engineId, std::vector<Knob>& knobs) const
    {
        if(!hasValidGraphDesc())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot get knobs "
                    "for engine."};
        }

        detail::ScopedHipdnnBackendDescriptor engineDesc;

        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineDescriptorForGraph(
            engineDesc, _graphDesc->get(), engineId));

        return detail::unpackKnobsFromDescriptors(engineDesc.get(), knobs);
    }

    /**
     * @brief Get knobs for a specific engine, indexed by knob type
     *
     * Convenience wrapper around get_knobs_for_engine() that populates
     * a map keyed by KnobType_t for direct lookup.
     *
     * @param engineId The engine ID to query
     * @param knobs Output map populated with available knobs, keyed by type
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR
     *         if the graph has not been built. Call get_message() for the
     *         specific failure reason.
     *
     * @see get_knobs_for_engine(), hipdnn_frontend::Knob
     */
    // NOLINTNEXTLINE(readability-identifier-naming, readability-convert-member-functions-to-static)
    Error get_knob_lookup_for_engine(int64_t engineId,
                                     std::unordered_map<KnobType_t, Knob>& knobs) const
    {
        knobs.clear();

        std::vector<Knob> knobVector;
        HIPDNN_CHECK_ERROR(get_knobs_for_engine(engineId, knobVector));

        for(auto& knob : knobVector)
        {
            knobs.try_emplace(knob.knobId(), std::move(knob));
        }

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Get a ranked list of engine IDs based on heuristics
     *
     * @param rankedEngineIds Output vector of engine IDs, ranked by expected performance
     * @param modes Heuristic modes to use for ranking
     * @return ErrorCode::OK on success; ErrorCode::GRAPH_NOT_SUPPORTED if no
     *         engine has an applicable solution for this graph on the current
     *         device; ErrorCode::HIPDNN_BACKEND_ERROR on other backend failure.
     *         Call get_message() for the specific failure reason.
     */
    // NOLINTNEXTLINE(readability-identifier-naming, readability-convert-member-functions-to-static)
    Error get_ranked_engine_ids(std::vector<int64_t>& rankedEngineIds,
                                const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK})
    {
        if(!hasReadyGraphDesc())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot get "
                    "ranked engine ids."};
        }

        detail::ScopedHipdnnBackendDescriptor engineHeuristicDesc;
        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineHeuristicDescriptorForGraph(
            engineHeuristicDesc, _graphDesc->get(), modes));

        std::vector<std::unique_ptr<detail::ScopedHipdnnBackendDescriptor>> engineConfigs;
        HIPDNN_CHECK_ERROR(detail::getEngineConfigs(
            engineConfigs, rankedEngineIds, engineHeuristicDesc.get(), true));

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Get behavior notes for an engine applicable to this graph.
     *
     * @param engineId Backend global engine ID to query
     * @param notes Output behavior notes; cleared on entry
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR on failure
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error get_behavior_notes_for_engine(int64_t engineId, std::vector<BehaviorNote>& notes) const
    {
        notes.clear();

        if(!hasReadyGraphDesc())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot get "
                    "behavior notes for engine."};
        }

        detail::ScopedHipdnnBackendDescriptor engineDesc;
        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineDescriptorForGraph(
            engineDesc, _graphDesc->get(), engineId));

        int64_t noteCount = 0;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetAttribute(engineDesc.get(),
                                                         HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                                         HIPDNN_TYPE_BEHAVIOR_NOTE,
                                                         0,
                                                         &noteCount,
                                                         nullptr),
            "Failed to get behavior note count from engine descriptor.");

        if(noteCount < 0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Backend returned a negative behavior note count: "
                        + std::to_string(noteCount)};
        }

        if(noteCount == 0)
        {
            return {ErrorCode::OK, ""};
        }

        const auto expectedNoteCount = noteCount;
        std::vector<hipdnnBackendBehaviorNote_t> backendNotes(static_cast<size_t>(noteCount));
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetAttribute(engineDesc.get(),
                                                         HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                                         HIPDNN_TYPE_BEHAVIOR_NOTE,
                                                         noteCount,
                                                         &noteCount,
                                                         backendNotes.data()),
            "Failed to get behavior notes from engine descriptor.");

        if(noteCount != expectedNoteCount)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Backend returned a behavior note count of " + std::to_string(noteCount)
                        + " after reporting " + std::to_string(expectedNoteCount) + "."};
        }

        notes.reserve(backendNotes.size());
        for(auto note : backendNotes)
        {
            notes.push_back(fromHipdnnBehaviorNote(note));
        }

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Create execution plans using heuristics
     *
     * Queries the backend for available engines and selects based on the
     * specified heuristic modes.
     *
     * @param modes Heuristic modes to use for engine selection
     * @return ErrorCode::OK on success; ErrorCode::GRAPH_NOT_SUPPORTED if no
     *         engine has an applicable solution for this graph on the current
     *         device; ErrorCode::HIPDNN_BACKEND_ERROR if the graph has not
     *         been built or on other backend failure. Call get_message() for
     *         the specific failure reason.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_execution_plans(const std::vector<HeuristicMode>& modes
                                 = {HeuristicMode::FALLBACK})
    {
        HIPDNN_FE_LOG_INFO("Creating execution plans for graph " << graph_attributes.get_name());

        if(!hasReadyGraphDesc())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot create "
                    "execution plan."};
        }

        detail::ScopedHipdnnBackendDescriptor engineHeuristicDesc;
        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineHeuristicDescriptorForGraph(
            engineHeuristicDesc, _graphDesc->get(), modes));

        HIPDNN_CHECK_ERROR(initializeEngineConfig(engineHeuristicDesc.get()));

        _executionPlanDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        _executionPlanFinalized = false;

        if(!_executionPlanDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend execution descriptor."};
        }

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Create an execution plan with specific engine and knob settings
     *
     * Creates an execution plan for a specific engine, configured via knob
     * settings. Settings for deprecated knobs or knobs that are not supported
     * by the engine are skipped and a log message is added describing this.
     *
     * @param engineId The engine ID to use
     * @param settings Knob settings to apply to the engine
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR
     *         if the graph has not been built. Call get_message() for the
     *         specific failure reason.
     *
     * @see hipdnn_frontend::Knob, hipdnn_frontend::KnobSetting, get_knobs_for_engine()
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>& settings)
    {
        HIPDNN_FE_LOG_INFO("Creating execution plans for graph " << graph_attributes.get_name());

        if(!hasReadyGraphDesc())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot create "
                    "execution plan."};
        }

        std::unordered_map<KnobType_t, Knob> existingKnobs;
        HIPDNN_CHECK_ERROR(get_knob_lookup_for_engine(engineId, existingKnobs));
        HIPDNN_CHECK_ERROR(initializeEngineConfig(engineId));

        std::vector<KnobSetting> validatedSettings;
        HIPDNN_CHECK_ERROR(
            validateAndFilterKnobSettings(settings, existingKnobs, validatedSettings));
        HIPDNN_CHECK_ERROR(applyKnobSettingsToEngineConfig(validatedSettings));

        return finalizeExecutionPlanDescriptor();
    }

    /**
     * @brief Verify that the execution plan is valid and supported
     *
     * @return ErrorCode::OK if valid, or ErrorCode::HIPDNN_BACKEND_ERROR
     *         if the execution plan has not been created.
     */
    Error check_support() // NOLINT(readability-identifier-naming)
    {
        HIPDNN_FE_LOG_INFO("Checking execution plan support for graph "
                           << graph_attributes.get_name());

        if(!_executionPlanDesc || !_executionPlanDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Execution plan descriptor is not created or invalid."};
        }

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Check if the graph is supported by any available engine plugin
     * @param handle The hipDNN handle
     * @param modes Heuristic modes for engine ranking
     * @return Error with OK if supported; GRAPH_NOT_SUPPORTED if no engine
     *         has an applicable solution for this graph on the current device;
     *         HIPDNN_BACKEND_ERROR on other backend failure
     *
     * Performs a lightweight check to determine if any engine plugin can
     * handle this graph. If the graph has not yet been validated and built,
     * those steps are performed automatically. The graph's internal state
     * (operation graph descriptor) is preserved for subsequent operations.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error is_supported_ext(hipdnnHandle_t handle,
                           const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK})
    {
        HIPDNN_FE_LOG_INFO("Checking engine support for graph " << graph_attributes.get_name());

        if(!hasValidGraphDesc())
        {
            HIPDNN_CHECK_ERROR(build_operation_graph(handle));
        }
        else if(!hasReadyGraphDesc())
        {
            HIPDNN_CHECK_ERROR(finalizeGraphDescWithHandle(handle));
        }

        detail::ScopedHipdnnBackendDescriptor engineHeuristicDesc;
        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineHeuristicDescriptorForGraph(
            engineHeuristicDesc, _graphDesc->get(), modes, /*findFirst=*/true));

        HIPDNN_CHECK_ERROR(detail::hasEngineConfigs(engineHeuristicDesc.get()));

        return {ErrorCode::OK, ""};
    }

    // ── Binary serialization (always available) ─────────────────────────

    /** @brief Serialize a graph to a binary byte vector, auto-lowering if needed.
     *
     * If the graph has not been lowered to a backend descriptor, it will be
     * auto-lowered before serialization.
     *
     * @param[out] data The serialized binary data.
     * @return Error indicating success or failure.
     */
    Error serialize(std::vector<uint8_t>& data)
    {
        HIPDNN_CHECK_ERROR(ensureLowered());
        return std::as_const(*this).serialize(data);
    }

    /** @brief Serialize a previously built graph to a binary byte vector.
     *
     * Requires a valid backend descriptor (call build_operation_graph() first).
     *
     * @param[out] data The serialized binary data.
     * @return Error indicating success or failure.
     */
    Error serialize(std::vector<uint8_t>& data) const
    {
        return assembleSerializedBlob(data);
    }

    /** @brief Serialize the graph to a binary byte vector, auto-lowering if needed.
     *
     * @return A pair of the serialized data and an Error.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::pair<std::vector<uint8_t>, Error> to_binary()
    {
        std::vector<uint8_t> data;
        auto err = serialize(data);
        return {std::move(data), std::move(err)};
    }

    /** @brief Serialize a previously built graph to a binary byte vector.
     *
     * @return A pair of the serialized data and an Error.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::pair<std::vector<uint8_t>, Error> to_binary() const
    {
        std::vector<uint8_t> data;
        auto err = serialize(data);
        return {std::move(data), std::move(err)};
    }

    /** @brief Deserialize a graph from a binary byte vector, finalizing with the given handle.
     *
     * Convenience wrapper around deserialize() for API symmetry with to_binary().
     *
     * @param handle The hipDNN handle for finalization.
     * @param data The binary data to deserialize.
     * @return Error indicating success or failure.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error from_binary(hipdnnHandle_t handle, const std::vector<uint8_t>& data)
    {
        return deserialize(handle, data);
    }

    /** @brief Deserialize a graph from a binary byte vector (structure only).
     *
     * Convenience wrapper around deserialize() for API symmetry with to_binary().
     * The backend descriptor is not finalized. Call build_operation_graph()
     * afterwards to finalize for execution.
     *
     * Any embedded execution plan is dropped (with a warning), as deserializing
     * it needs a handle. Use from_binary(handle, data) to restore it.
     *
     * @param data The binary data to deserialize.
     * @return Error indicating success or failure.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error from_binary(const std::vector<uint8_t>& data)
    {
        return deserialize(data);
    }

    /** @brief Deserialize a graph from a binary byte vector with handle.
     *
     * Unpacks the serialized graph, reconstructs frontend nodes and attributes,
     * and finalizes the backend descriptor with the given handle.
     *
     * @param handle The hipDNN handle for finalization.
     * @param data The binary data to deserialize.
     * @return Error indicating success or failure.
     */
    Error deserialize(hipdnnHandle_t handle, const std::vector<uint8_t>& data)
    {
        std::vector<std::shared_ptr<graph::INode>> tempNodes;
        graph::GraphAttributes tempAttrs;
        std::optional<int64_t> tempEngineId;
        bool tempOverrideShapeEnabled = false;

        auto [graphDesc, err] = detail::deserializeAndUnpackGraph(
            handle, data, tempNodes, tempAttrs, tempEngineId, tempOverrideShapeEnabled);
        if(err.is_bad())
        {
            return err;
        }

        _sub_nodes = std::move(tempNodes);
        graph_attributes = std::move(tempAttrs);
        _preferredEngineId = tempEngineId;
        _isOverrideShapeEnabled = tempOverrideShapeEnabled;
        setGraphDesc(std::move(graphDesc), handle != nullptr);
        _engineConfigDesc.reset();
        _executionPlanDesc.reset();
        _executionPlanFinalized = false;
        _selectedEngineId.reset();

        int flags = 0;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetSerializedBinaryContentsExt(
                data.data(), data.size(), &flags),
            "Failed to query serialized blob contents");

        if((flags & HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN) != 0)
        {
            if(handle != nullptr)
            {
                hipdnnBackendDescriptor_t plan = nullptr;
                HIPDNN_RETURN_ON_BACKEND_FAILURE(
                    detail::hipdnnBackend()->backendCreateAndDeserializeExecutionPlanExt(
                        handle, &plan, data.data(), data.size()),
                    "Failed to deserialize embedded execution plan");
                _executionPlanDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(plan);
                // A deserialized compiled plan is finalized by construction and can be re-serialized.
                _executionPlanFinalized = true;
                // Recover the engine backing the attached plan for the serialize capability gate.
                _selectedEngineId = detail::getExecutionPlanEngineId(_executionPlanDesc->get());
            }
            else
            {
                HIPDNN_FE_LOG_WARN(
                    "Deserialized blob contains an execution plan, but no handle was "
                    "provided; the embedded plan was dropped. Use the handle overload "
                    "to restore the plan.");
            }
        }
        return {};
    }

    /** @brief Deserialize a graph from a binary byte vector (structure only).
     *
     * The backend descriptor is not finalized. Call build_operation_graph()
     * afterwards to finalize for execution.
     *
     * Any embedded execution plan is dropped (with a warning), as deserializing
     * it needs a handle. Use deserialize(handle, data) to restore it.
     *
     * @param data The binary data to deserialize.
     * @return Error indicating success or failure.
     */
    Error deserialize(const std::vector<uint8_t>& data)
    {
        return deserialize(nullptr, data);
    }

    /** @brief Serialize the compiled backend execution plan to a byte vector.
     *
     * Requires build_plans() or build() to have finalized the execution plan.
     * The returned data is intended for from_compiled_plan_binary().
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error serialize_compiled_plan(std::vector<uint8_t>& data) const
    {
        if(!_executionPlanDesc || !_executionPlanDesc->valid())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Graph has no compiled execution plan. Call build() or build_plans() first."};
        }

        size_t planByteSize = 0;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetSerializedExecutionPlanExt(
                _executionPlanDesc->get(), 0, &planByteSize, nullptr),
            "Failed to query serialized compiled plan size");

        if(planByteSize == 0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Backend returned zero-length compiled plan"};
        }

        data.resize(planByteSize);
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetSerializedExecutionPlanExt(
                _executionPlanDesc->get(), planByteSize, &planByteSize, data.data()),
            "Failed to serialize compiled plan");
        data.resize(planByteSize);

        return {};
    }

    /** @brief Serialize the compiled backend execution plan to a byte vector. */
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::pair<std::vector<uint8_t>, Error> to_compiled_plan_binary() const
    {
        std::vector<uint8_t> data;
        auto err = serialize_compiled_plan(data);
        return {std::move(data), std::move(err)};
    }

    /** @brief Deserialize a compiled backend execution plan for execution.
     *
     * This restores enough backend state to execute the compiled plan, but it
     * does not restore frontend graph details such as tensor attributes,
     * declared shapes, or declared strides. UID-based override execution is
     * allowed on this lightweight plan-only object; graph-aware override
     * validation is skipped, so callers must supply overrides that are
     * consistent with the deserialized plan.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error deserialize_compiled_plan(hipdnnHandle_t handle, const std::vector<uint8_t>& data)
    {
        hipdnnBackendDescriptor_t executionPlan = nullptr;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendCreateAndDeserializeExecutionPlanExt(
                handle, &executionPlan, data.data(), data.size()),
            "Failed to deserialize compiled plan");

        _executionPlanDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(executionPlan);
        // A deserialized compiled plan is finalized by construction and can be re-serialized.
        _executionPlanFinalized = true;
        // Recover the engine backing the attached plan for the serialize capability gate.
        _selectedEngineId = detail::getExecutionPlanEngineId(_executionPlanDesc->get());
        _engineConfigDesc.reset();
        resetGraphDesc();
        _sub_nodes.clear();
        _isOverrideShapeEnabled = false;

        return {};
    }

    /** @brief Deserialize a compiled backend execution plan for execution.
     *
     * This restores enough backend state to execute the compiled plan, but it
     * does not restore frontend graph details such as tensor attributes,
     * declared shapes, or declared strides. UID-based override execution is
     * allowed on this lightweight plan-only object; graph-aware override
     * validation is skipped, so callers must supply overrides that are
     * consistent with the deserialized plan.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error from_compiled_plan_binary(hipdnnHandle_t handle, const std::vector<uint8_t>& data)
    {
        return deserialize_compiled_plan(handle, data);
    }

    // ── JSON string serialization (always available) ────────────────────

    /** @brief Serialize a previously built graph to a JSON string.
     *
     * Requires a valid backend descriptor (call build_operation_graph() first).
     *
     * @param[out] jsonData The serialized JSON string.
     * @return Error indicating success or failure.
     */
    Error serialize(std::string& jsonData) const
    {
        if(!hasValidGraphDesc())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Graph has no backend descriptor. "
                    "Call build_operation_graph() first, or use the non-const "
                    "serialize() overload for auto-lowering."};
        }

        size_t graphByteSize = 0;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(detail::hipdnnBackend()->backendGetSerializedJsonGraphExt(
                                             _graphDesc->get(), 0, &graphByteSize, nullptr),
                                         "Failed to query JSON graph size");

        if(graphByteSize == 0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Backend returned zero-length JSON graph"};
        }

        // The backend C API reports graphByteSize including the null terminator
        // (standard C convention). We resize to the full size so the backend can
        // write into the buffer, then shrink by one to exclude the terminator
        // from the std::string's logical content.
        jsonData.resize(graphByteSize);
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetSerializedJsonGraphExt(
                _graphDesc->get(), graphByteSize, &graphByteSize, jsonData.data()),
            "Failed to serialize graph to JSON");
        jsonData.resize(graphByteSize - 1);

        return {};
    }

    /** @brief Serialize a graph to a JSON string, auto-lowering if needed.
     *
     * If the graph has not been lowered to a backend descriptor, it will be
     * auto-lowered before serialization.
     *
     * @param[out] jsonData The serialized JSON string.
     * @return Error indicating success or failure.
     */
    Error serialize(std::string& jsonData)
    {
        HIPDNN_CHECK_ERROR(ensureLowered());
        return std::as_const(*this).serialize(jsonData);
    }

    /** @brief Serialize the graph to a JSON string, auto-lowering if needed.
     *
     * @return A pair of the serialized JSON string and an Error.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::pair<std::string, Error> to_json()
    {
        std::string jsonData;
        auto err = serialize(jsonData);
        return {std::move(jsonData), std::move(err)};
    }

    /** @brief Serialize a previously built graph to a JSON string.
     *
     * @return A pair of the serialized JSON string and an Error.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::pair<std::string, Error> to_json() const
    {
        std::string jsonData;
        auto err = serialize(jsonData);
        return {std::move(jsonData), std::move(err)};
    }

    /** @brief Deserialize a graph from a JSON string, finalizing with the given handle.
     *
     * Convenience wrapper around deserialize() for API symmetry with to_json().
     *
     * @param handle The hipDNN handle for finalization.
     * @param json The JSON string to deserialize.
     * @return Error indicating success or failure.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error from_json(hipdnnHandle_t handle, const std::string& json)
    {
        return deserialize(handle, json);
    }

    /** @brief Deserialize a graph from a JSON string (structure only).
     *
     * Convenience wrapper around deserialize() for API symmetry with to_json().
     * The backend descriptor is not finalized. Call build_operation_graph()
     * afterwards to finalize for execution.
     *
     * @param json The JSON string to deserialize.
     * @return Error indicating success or failure.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error from_json(const std::string& json)
    {
        return deserialize(json);
    }

    /** @brief Deserialize a graph from a JSON string with handle.
     *
     * Unpacks the serialized graph, reconstructs frontend nodes and attributes,
     * and finalizes the backend descriptor with the given handle.
     *
     * @param handle The hipDNN handle for finalization.
     * @param jsonData The JSON string to deserialize.
     * @return Error indicating success or failure.
     */
    Error deserialize(hipdnnHandle_t handle, const std::string& jsonData)
    {
        std::vector<std::shared_ptr<graph::INode>> tempNodes;
        graph::GraphAttributes tempAttrs;
        std::optional<int64_t> tempEngineId;
        bool tempOverrideShapeEnabled = false;

        auto [graphDesc, err] = detail::deserializeAndUnpackJsonGraph(
            handle, jsonData, tempNodes, tempAttrs, tempEngineId, tempOverrideShapeEnabled);
        if(err.is_bad())
        {
            return err;
        }

        _sub_nodes = std::move(tempNodes);
        graph_attributes = std::move(tempAttrs);
        _preferredEngineId = tempEngineId;
        _isOverrideShapeEnabled = tempOverrideShapeEnabled;
        setGraphDesc(std::move(graphDesc), handle != nullptr);
        _engineConfigDesc.reset();
        _executionPlanDesc.reset();
        _executionPlanFinalized = false;
        _selectedEngineId.reset();
        return {};
    }

    /** @brief Deserialize a graph from a JSON string (structure only).
     *
     * The backend descriptor is not finalized. Call build_operation_graph()
     * afterwards to finalize for execution.
     *
     * @param jsonData The JSON string to deserialize.
     * @return Error indicating success or failure.
     */
    Error deserialize(const std::string& jsonData)
    {
        return deserialize(nullptr, jsonData);
    }

    // ── nlohmann::json serialization (requires JSON library) ────────────

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
    /// Serialize a previously built graph to a nlohmann::json object.
    Error serialize(nlohmann::json& j) const
    {
        std::string jsonData;
        HIPDNN_CHECK_ERROR(serialize(jsonData));
        try
        {
            j = nlohmann::json::parse(jsonData);
        }
        catch(const nlohmann::json::exception& e)
        {
            return {ErrorCode::INVALID_VALUE,
                    std::string("Failed to parse serialized JSON: ") + e.what()};
        }
        return {};
    }

    /// Serialize a graph to a nlohmann::json object, auto-lowering if needed.
    Error serialize(nlohmann::json& j)
    {
        HIPDNN_CHECK_ERROR(ensureLowered());
        return std::as_const(*this).serialize(j);
    }

    /// Deserialize a graph from a nlohmann::json object with handle (finalizes).
    Error deserialize(hipdnnHandle_t handle, const nlohmann::json& j)
    {
        try
        {
            return deserialize(handle, j.dump());
        }
        catch(const nlohmann::json::exception& e)
        {
            return {ErrorCode::INVALID_VALUE,
                    std::string("Failed to dump JSON for deserialization: ") + e.what()};
        }
    }

    /// Deserialize a graph from a nlohmann::json object (structure only).
    /// The backend descriptor is not finalized. Call build_operation_graph()
    /// afterwards to finalize for execution.
    Error deserialize(const nlohmann::json& j)
    {
        try
        {
            return deserialize(nullptr, j.dump());
        }
        catch(const nlohmann::json::exception& e)
        {
            return {ErrorCode::INVALID_VALUE,
                    std::string("Failed to dump JSON for deserialization: ") + e.what()};
        }
    }
#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB

    /**
     * @brief Finalize the execution plan
     *
     * Called internally by build() after create_execution_plans().
     *
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR
     *         on failure. Call get_message() for the specific failure reason.
     */
    Error build_plans() // NOLINT(readability-identifier-naming)
    {
        HIPDNN_FE_LOG_INFO("Building plans for graph " << graph_attributes.get_name());

        HIPDNN_RETURN_ON_BACKEND_FAILURE(detail::hipdnnBackend()->backendSetAttribute(
                                             _executionPlanDesc->get(),
                                             HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             1,
                                             static_cast<const void*>(&_engineConfigDesc->get())),
                                         "Failed to set the engine config on execution plan.");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(_executionPlanDesc->get()),
            "Failed to finalize execution plan descriptor");

        _executionPlanFinalized = true;

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Build the complete graph and create execution plans
     *
     * This is the main method to prepare a graph for execution. It performs:
     * 1. Graph validation
     * 2. Operation graph building
     * 3. Execution plan creation
     * 4. Execution plan support verification
     * 5. Plan finalization
     *
     * @note This method does not allow setting engine knobs. If you need
     * to configure knobs, use get_ranked_engine_ids(), get_knobs_for_engine(),
     * and create_execution_plan_ext() instead.
     *
     * @code{.cpp}
     * hipdnnHandle_t handle;
     * hipdnnCreate(&handle);
     * Error err = graph.build(handle);
     * if(err.is_bad()) { handleError(); }
     * @endcode
     *
     * @param handle The hipDNN handle
     * @param modes Heuristic modes for engine selection
     * @param policy Build plan policy (currently only HEURISTICS_CHOICE is used)
     * @param do_multithreaded_builds Reserved for future use
     * @return ErrorCode::OK on success, or ErrorCode::INVALID_VALUE /
     *         ErrorCode::ATTRIBUTE_NOT_SET / ErrorCode::HIPDNN_BACKEND_ERROR /
     *         ErrorCode::GRAPH_NOT_SUPPORTED (when no engine has an applicable
     *         solution on the current device) on failure. Call get_message() for
     *         the specific failure reason.
     */
    // NOLINTBEGIN(readability-identifier-naming)
    Error build(hipdnnHandle_t handle,
                const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK},
                [[maybe_unused]] BuildPlanPolicy policy = BuildPlanPolicy::HEURISTICS_CHOICE,
                [[maybe_unused]] bool do_multithreaded_builds = false)
    // NOLINTEND(readability-identifier-naming)
    {
        auto graphName
            = graph_attributes.get_name().empty() ? "unnamed" : graph_attributes.get_name();
        HIPDNN_FE_LOG_INFO("BUILD with handle for graph '"
                           << graphName << "', policy: " << static_cast<int>(policy)
                           << ", modes count: " << modes.size());

        HIPDNN_CHECK_ERROR(build_operation_graph(handle));
        HIPDNN_CHECK_ERROR(create_execution_plans(modes));
        HIPDNN_CHECK_ERROR(check_support());
        HIPDNN_CHECK_ERROR(build_plans());

        HIPDNN_FE_LOG_INFO("BUILD ALL OK for graph " << graphName);
        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Get the workspace memory size required for execution
     *
     * Call this after build() to determine how much workspace memory to allocate.
     *
     * @param workspaceSize Output parameter for the workspace size in bytes
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR
     *         on failure. Call get_message() for the specific failure reason.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error get_workspace_size(int64_t& workspaceSize) const
    {
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendGetAttribute(_executionPlanDesc->get(),
                                                         HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                         HIPDNN_TYPE_INT64,
                                                         1,
                                                         nullptr,
                                                         &workspaceSize),
            "Failed to get workspace size from the execution plan descriptor.");

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Execute the graph with tensor pointers mapped by tensor handles
     *
     * @param handle The hipDNN handle
     * @param tensorLookup Map from std::shared_ptr<TensorAttributes> (tensor handles) to device
     * memory pointers
     * @param workspace Pointer to workspace memory (can be nullptr if size is 0)
     * @return ErrorCode::OK on success, ErrorCode::INVALID_VALUE if a tensor
     *         in the lookup is null or missing a UID, or
     *         ErrorCode::HIPDNN_BACKEND_ERROR on backend failure. Call
     *         get_message() for the specific failure reason.
     *
     * @code{.cpp}
     * std::unordered_map<std::shared_ptr<TensorAttributes>, void*> tensorLookup = {
     *     {inputTensor, d_input},
     *     {outputTensor, d_output}
     * };
     * graph.execute(handle, tensorLookup, workspace);
     * @endcode
     */
    Error execute(hipdnnHandle_t handle,
                  std::unordered_map<std::shared_ptr<TensorAttributes>, void*>& tensorLookup,
                  void* workspace) const
    {
        std::unordered_map<int64_t, void*> variantPack;
        for(const auto& [tensor, ptr] : tensorLookup)
        {
            if(tensor && tensor->has_uid())
            {
                variantPack[tensor->get_uid()] = ptr;
            }
            else
            {
                return {ErrorCode::INVALID_VALUE,
                        "Tensor in tensor lookup is null or does not have a valid uid."};
            }
        }

        return execute(handle, variantPack, workspace);
    }

    /**
     * @brief Execute the graph with tensor pointers mapped by UID
     *
     * @param handle The hipDNN handle
     * @param variantPack Map from tensor UID to device memory pointers
     * @param workspace Pointer to workspace memory (can be nullptr if size is 0)
     * @return ErrorCode::OK on success, or ErrorCode::HIPDNN_BACKEND_ERROR
     *         on failure. Call get_message() for the specific failure reason.
     *
     * @code{.cpp}
     * std::unordered_map<int64_t, void*> variantPack = {
     *     {0, d_input},   // UID 0 -> input tensor
     *     {1, d_weights}, // UID 1 -> weights
     *     {2, d_output}   // UID 2 -> output tensor
     * };
     * graph.execute(handle, variantPack, workspace);
     * @endcode
     */
    Error execute(hipdnnHandle_t handle,
                  std::unordered_map<int64_t, void*>& variantPack,
                  void* workspace) const
    {
        HIPDNN_FE_LOG_INFO("Executing graph " << graph_attributes.get_name());

        if(!_executionPlanDesc || !_executionPlanDesc->valid())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Graph has no compiled execution plan. Call build() or "
                    "from_compiled_plan_binary() first."};
        }

        auto variantPackDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        if(!variantPackDesc || !variantPackDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create variant pack descriptor."};
        }

        HIPDNN_CHECK_ERROR(
            detail::populateBaseVariantPackDescriptor(*variantPackDesc, variantPack, workspace));

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(variantPackDesc->get()),
            "Failed to finalize variant pack descriptor");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendExecute(
                handle, _executionPlanDesc->get(), variantPackDesc->get()),
            "Execute failed.");

        return {ErrorCode::OK, ""};
    }

#ifdef HIPDNN_ENABLE_SDPA
    /**
     * @brief Execute with per-tensor runtime shape/stride overrides.
     *
     * Graph-backed objects require `set_override_shape_enabled(true)`. Objects
     * restored from compiled-plan bytes receive structural validation only.
     * Empty override arrays dispatch through the non-override path.
     */
    Error execute(hipdnnHandle_t handle,
                  std::unordered_map<int64_t, void*>& variantPack,
                  void* workspace,
                  const std::vector<int64_t>& overrideUids,
                  const std::vector<std::vector<int64_t>>& overrideShapes,
                  const std::vector<std::vector<int64_t>>& overrideStrides) const
    {
        if(!_executionPlanDesc || !_executionPlanDesc->valid())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Graph has no compiled execution plan. Call build() or "
                    "from_compiled_plan_binary() first."};
        }

        if(overrideUids.empty() && overrideShapes.empty() && overrideStrides.empty())
        {
            HIPDNN_FE_LOG_INFO("Override execute called on graph "
                               << graph_attributes.get_name()
                               << " with empty override vectors; falling through to "
                                  "non-override entry.");
            return execute(handle, variantPack, workspace);
        }

        const bool planOnly = _sub_nodes.empty();
        if(planOnly)
        {
            HIPDNN_CHECK_ERROR(detail::validatePlanOnlyOverrideArguments(
                overrideUids, overrideShapes, overrideStrides));
        }
        else
        {
            if(!_isOverrideShapeEnabled)
            {
                HIPDNN_FE_LOG_INFO("Override execute called on graph "
                                   << graph_attributes.get_name()
                                   << " without set_override_shape_enabled(true).");
                return {ErrorCode::INVALID_VALUE,
                        "Graph::execute override overload called on a graph that did "
                        "not call set_override_shape_enabled(true). The override flag "
                        "must be set at build time before per-execute overrides are "
                        "supplied."};
            }

            HIPDNN_CHECK_ERROR(detail::validateGraphBackedOverrideArguments(
                getTensorsByUid(), overrideUids, overrideShapes, overrideStrides));
        }

        for(const auto uid : overrideUids)
        {
            if(variantPack.find(uid) == variantPack.end())
            {
                return {ErrorCode::INVALID_VALUE,
                        "Override UID " + std::to_string(uid)
                            + " is not present in the variant pack."};
            }
        }

        HIPDNN_FE_LOG_INFO("Executing graph " << graph_attributes.get_name() << " with "
                                              << overrideUids.size() << " override entries.");

        auto variantPackDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        if(!variantPackDesc || !variantPackDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create variant pack descriptor."};
        }

        HIPDNN_CHECK_ERROR(
            detail::populateBaseVariantPackDescriptor(*variantPackDesc, variantPack, workspace));

        HIPDNN_CHECK_ERROR(detail::populateOverrideVariantPackDescriptor(
            *variantPackDesc, overrideUids, overrideShapes, overrideStrides));

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(variantPackDesc->get()),
            "Failed to finalize variant pack descriptor");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendExecute(
                handle, _executionPlanDesc->get(), variantPackDesc->get()),
            "Execute failed.");

        return {ErrorCode::OK, ""};
    }

    /// Execute with map-keyed runtime shape/stride overrides.
    Error execute(hipdnnHandle_t handle,
                  std::unordered_map<int64_t, void*>& variantPack,
                  void* workspace,
                  const std::unordered_map<int64_t, OverrideEntry>& overrides) const
    {
        std::vector<int64_t> overrideUids;
        std::vector<std::vector<int64_t>> overrideShapes;
        std::vector<std::vector<int64_t>> overrideStrides;
        overrideUids.reserve(overrides.size());
        overrideShapes.reserve(overrides.size());
        overrideStrides.reserve(overrides.size());
        for(const auto& [uid, entry] : overrides)
        {
            overrideUids.push_back(uid);
            overrideShapes.push_back(entry.shape);
            overrideStrides.push_back(entry.stride);
        }

        return execute(
            handle, variantPack, workspace, overrideUids, overrideShapes, overrideStrides);
    }
#endif // HIPDNN_ENABLE_SDPA

    /// @brief Get the graph name
    const std::string& get_name() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_name();
    }

    /// @brief Get the compute data type (precision used inside operations, e.g. accumulation)
    DataType get_compute_data_type() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_compute_data_type();
    }
    /// @brief Get the intermediate data type (precision of virtual tensors between fused ops)
    DataType get_intermediate_data_type() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_intermediate_data_type();
    }
    /// @brief Get the I/O data type (precision of graph input and output tensors)
    DataType get_io_data_type() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_io_data_type();
    }

    /// @brief Get the preferred engine ID, if set
    // NOLINTBEGIN(readability-identifier-naming)
    std::optional<int64_t> get_preferred_engine_id_ext() const
    // NOLINTEND(readability-identifier-naming)
    {
        return _preferredEngineId;
    }

    /// @brief Get the engine ID actually backing the current execution plan.
    ///
    /// Returns the engine cached when the plan's engine config was selected
    /// (_selectedEngineId): the engine that will execute, regardless of how it
    /// was chosen (heuristic, soft preferred-engine with fallback, hard
    /// create_execution_plan_ext, or a deserialized compiled plan). Unlike
    /// get_preferred_engine_id_ext (which returns the *request*), this returns
    /// what was *selected*, so callers can detect a silent fallback.
    ///
    /// Fails if no execution plan has been created (create_execution_plans /
    /// create_execution_plan_ext).
    // NOLINTBEGIN(readability-identifier-naming)
    Error get_execution_plan_engine_id(int64_t& engineId) const
    // NOLINTEND(readability-identifier-naming)
    {
        if(!_selectedEngineId.has_value())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "No execution plan available; build a plan before querying its engine."};
        }
        engineId = *_selectedEngineId;
        return {ErrorCode::OK, ""};
    }

    /// @brief Set the graph name
    Graph& set_name(const std::string& name) // NOLINT(readability-identifier-naming)
    {
        graph_attributes.set_name(name);
        return *this;
    }
    /**
     * @brief Set the compute data type (precision for internal math)
     *
     * Controls the accumulation precision inside operations — the dtype
     * used for arithmetic during execution. For mixed-precision training
     * you might store tensors in fp16 (`io_data_type = HALF`) but
     * accumulate in fp32 (`compute_data_type = FLOAT`) for numerical
     * stability.
     */
    Graph& set_compute_data_type(DataType computeType) // NOLINT(readability-identifier-naming)
    {
        graph_attributes.set_compute_data_type(computeType);
        return *this;
    }
    /**
     * @brief Set the intermediate data type for virtual tensors between fused ops
     *
     * When the backend fuses multiple operations, intermediate results
     * are stored in this precision. Usually matches compute_data_type.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Graph& set_intermediate_data_type(DataType intermediateType)
    {
        graph_attributes.set_intermediate_data_type(intermediateType);
        return *this;
    }
    /**
     * @brief Set the I/O data type — the default precision for graph inputs/outputs
     *
     * This is the dtype of the tensors you feed in and read out.
     * Individual tensors can override this by calling
     * TensorAttributes::set_data_type().
     */
    Graph& set_io_data_type(DataType ioType) // NOLINT(readability-identifier-naming)
    {
        graph_attributes.set_io_data_type(ioType);
        return *this;
    }

    /** @brief Batch normalization forward pass for training
     *
     * Normalizes the input across the batch dimension and computes statistics.
     *
     * Formula:
     * @code
     * mean[c]    = (1/m) * sum(x[n,c,h,w])        where m = N*H*W
     * var[c]     = (1/m) * sum((x[n,c,h,w] - mean[c])^2)
     * invVar[c]  = 1 / sqrt(var[c] + epsilon)
     * y[n,c,h,w] = scale[c] * (x[n,c,h,w] - mean[c]) * invVar[c] + bias[c]
     * @endcode
     *
     * When previous running statistics are provided:
     * @code
     * nextRunningMean = (1 - momentum) * prevRunningMean + momentum * mean
     * nextRunningVar  = (1 - momentum) * prevRunningVar  + momentum * var
     * @endcode
     *
     * @param x Input tensor with batch, channel, and spatial dimensions
     * @param scale Per-channel scale (gamma)
     * @param bias Per-channel bias (beta)
     * @param attributes Configuration including epsilon; optionally
     *        prev_running_mean, prev_running_variance, and momentum for
     *        exponential moving average of running statistics
     * @return Array of 5 output tensors:
     *         - [0] y: Normalized output (same shape as x)
     *         - [1] mean: Per-channel batch mean
     *         - [2] invVariance: Per-channel batch inverse variance
     *         - [3] nextRunningMean: Updated running mean (nullptr if not tracking)
     *         - [4] nextRunningVariance: Updated running variance (nullptr if not tracking)
     *
     * @see hipdnn_frontend::graph::BatchnormAttributes
     */
    std::array<std::shared_ptr<TensorAttributes>, 5>
        batchnorm(std::shared_ptr<TensorAttributes> x,
                  std::shared_ptr<TensorAttributes> scale,
                  std::shared_ptr<TensorAttributes> bias,
                  BatchnormAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Batchnorm_" + std::to_string(_sub_nodes.size()));
        }

        auto y = outputTensor(attributes.get_name() + "::Y");
        auto meanOut = outputTensor(attributes.get_name() + "::MEAN");
        auto invVarianceOut = outputTensor(attributes.get_name() + "::INV_VARIANCE");

        auto prevRunningMean = attributes.get_prev_running_mean();
        auto prevRunningVariance = attributes.get_prev_running_variance();
        auto momentum = attributes.get_momentum();

        std::shared_ptr<TensorAttributes> nextRunningMean;
        std::shared_ptr<TensorAttributes> nextRunningVariance;
        if(prevRunningMean && prevRunningVariance && momentum)
        {
            nextRunningMean = outputTensor(attributes.get_name() + "::NEXT_RUNNING_MEAN");
            nextRunningVariance = outputTensor(attributes.get_name() + "::NEXT_RUNNING_VARIANCE");
        }

        attributes.set_x(std::move(x));
        attributes.set_scale(std::move(scale));
        attributes.set_bias(std::move(bias));
        attributes.set_y(y);
        attributes.set_mean(meanOut);
        attributes.set_inv_variance(invVarianceOut);
        attributes.set_next_running_mean(nextRunningMean);
        attributes.set_next_running_variance(nextRunningVariance);

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormNode>(std::move(attributes), graph_attributes));

        return {y, meanOut, invVarianceOut, nextRunningMean, nextRunningVariance};
    }

    /** @brief Batch normalization backward pass
     *
     * Computes gradients with respect to input, scale, and bias.
     *
     * Formula (using per-channel indexing for illustration):
     * @code
     * x_hat[c]  = (x[c] - mean[c]) * invVariance[c]
     * dbias[c]  = sum(dy[c])                           // sum over batch and spatial dims
     * dscale[c] = sum(dy[c] * x_hat[c])                // sum over batch and spatial dims
     * dx[c]     = scale[c] * invVariance[c] * (dy[c] - (dbias[c] + x_hat[c] * dscale[c]) / m)
     * @endcode
     * where m = number of elements per channel (batch size * spatial dims).
     *
     * @param dy Upstream gradient (loss gradient w.r.t. output, same shape as x)
     * @param x Original input from forward pass
     * @param scale Per-channel scale (gamma)
     * @param attributes Configuration; optionally set saved mean and inverse
     *        variance from the forward pass via set_saved_mean_and_inv_variance()
     * @return Array of 3 output tensors:
     *         - [0] dx: Gradient w.r.t. input (same shape as x)
     *         - [1] dscale: Per-channel gradient w.r.t. scale
     *         - [2] dbias: Per-channel gradient w.r.t. bias
     *
     * @see hipdnn_frontend::graph::BatchnormBackwardAttributes
     */
    std::array<std::shared_ptr<TensorAttributes>, 3>
        batchnorm_backward(std::shared_ptr<TensorAttributes> dy, // NOLINT
                           std::shared_ptr<TensorAttributes> x,
                           std::shared_ptr<TensorAttributes> scale,
                           BatchnormBackwardAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("BatchnormBackward_" + std::to_string(_sub_nodes.size()));
        }

        auto dx = outputTensor(attributes.get_name() + "::DX");
        auto dscale = outputTensor(attributes.get_name() + "::DSCALE");
        auto dbias = outputTensor(attributes.get_name() + "::DBIAS");

        attributes.set_x(std::move(x));
        attributes.set_dy(std::move(dy));
        attributes.set_scale(std::move(scale));
        attributes.set_dx(dx);
        attributes.set_dscale(dscale);
        attributes.set_dbias(dbias);

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormBackwardNode>(std::move(attributes), graph_attributes));

        return {dx, dscale, dbias};
    }

    /** @brief Batch normalization inference
     *
     * Applies pre-computed normalization statistics for inference.
     *
     * Formula:
     * @code
     * y[n,c,h,w] = scale[c] * (x[n,c,h,w] - mean[c]) * invVariance[c] + bias[c]
     * @endcode
     *
     * @param x Input tensor with batch, channel, and spatial dimensions
     * @param mean Pre-computed per-channel mean
     * @param invVariance Pre-computed per-channel inverse variance (1/sqrt(var+epsilon))
     * @param scale Per-channel scale (gamma)
     * @param bias Per-channel bias (beta)
     * @param attributes Additional configuration
     * @return y: Normalized output tensor (same shape as x)
     *
     * @see hipdnn_frontend::graph::BatchnormInferenceAttributes
     */
    std::shared_ptr<TensorAttributes>
        batchnorm_inference(std::shared_ptr<TensorAttributes> x, // NOLINT
                            std::shared_ptr<TensorAttributes> mean,
                            std::shared_ptr<TensorAttributes> invVariance,
                            std::shared_ptr<TensorAttributes> scale,
                            std::shared_ptr<TensorAttributes> bias,
                            BatchnormInferenceAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("BatchnormInference_" + std::to_string(_sub_nodes.size()));
        }

        auto y = outputTensor(attributes.get_name() + "::Y");

        attributes.set_x(std::move(x));
        attributes.set_mean(std::move(mean));
        attributes.set_inv_variance(std::move(invVariance));
        attributes.set_scale(std::move(scale));
        attributes.set_bias(std::move(bias));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<BatchnormInferenceNode>(std::move(attributes), graph_attributes));

        return y;
    }

    /** @brief Batch normalization inference with variance and epsilon tensors
     *
     * Variant that accepts variance (instead of inverse variance) and epsilon
     * as separate input tensors, computing inverse variance internally.
     *
     * Formula:
     * @code
     * y[n,c,h,w] = scale[c] * (x[n,c,h,w] - mean[c]) / sqrt(variance[c] + epsilon) + bias[c]
     * @endcode
     *
     * @param x Input tensor with batch, channel, and spatial dimensions
     * @param mean Pre-computed per-channel mean
     * @param variance Pre-computed per-channel variance
     * @param scale Per-channel scale (gamma)
     * @param bias Per-channel bias (beta)
     * @param epsilon Epsilon tensor for numerical stability (pass-by-value scalar)
     * @param attributes Additional configuration
     * @return y: Normalized output tensor (same shape as x)
     *
     * @see hipdnn_frontend::graph::BatchnormInferenceAttributesVarianceExt
     */
    std::shared_ptr<TensorAttributes>
        batchnorm_inference_variance_ext(std::shared_ptr<TensorAttributes> x, // NOLINT
                                         std::shared_ptr<TensorAttributes> mean,
                                         std::shared_ptr<TensorAttributes> variance,
                                         std::shared_ptr<TensorAttributes> scale,
                                         std::shared_ptr<TensorAttributes> bias,
                                         std::shared_ptr<TensorAttributes> epsilon,
                                         BatchnormInferenceAttributesVarianceExt attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("BatchnormInferenceVarianceExt_"
                                + std::to_string(_sub_nodes.size()));
        }

        auto y = outputTensor(attributes.get_name() + "::Y");

        attributes.set_x(std::move(x));
        attributes.set_mean(std::move(mean));
        attributes.set_variance(std::move(variance));
        attributes.set_scale(std::move(scale));
        attributes.set_bias(std::move(bias));
        attributes.set_epsilon(std::move(epsilon));
        attributes.set_y(y);

        _sub_nodes.emplace_back(std::make_shared<BatchnormInferenceNodeVarianceExt>(
            std::move(attributes), graph_attributes));

        return y;
    }

    /** @brief Layer normalization forward pass
     *
     * Normalizes the input across the last k feature dimensions, where k
     * is inferred from the scale tensor shape. By default, all dimensions
     * except the first (batch) dimension are normalized.
     *
     * Common configurations:
     * - **Transformer**: x=[B, S, D], scale=[D] → normalizes over D (k=1)
     * - **Vision**: x=[N, C, H, W], scale=[1, C, H, W] → normalizes over C, H, W (k=3)
     *
     * Formula:
     * @code
     * mean    = (1/m) * sum(x) over normalized dims, where m = product of normalized dims
     * var     = (1/m) * sum((x - mean)^2) over normalized dims
     * xhat    = (x - mean) / sqrt(var + epsilon)
     * y       = scale * xhat + bias
     * @endcode
     *
     * In training phase, mean and inverse variance are also returned as outputs.
     *
     * @param x Input tensor [N, D1, D2, ..., Dk]
     * @param scale Per-feature scale (gamma) tensor, matching the normalized
     *        dimensions. Can be full-rank with batch dims set to 1
     *        (e.g. [1, C, H, W]) or reduced-rank with batch dims omitted
     *        (e.g. [C, H, W])
     * @param bias Per-feature bias (beta) tensor (same shape as scale)
     * @param attributes Configuration including epsilon and forward phase
     * @return Array of 3 output tensors:
     *         - [0] y: Normalized output (same shape as x)
     *         - [1] mean: Computed mean (nullptr in inference mode)
     *         - [2] invVariance: Computed inverse variance (nullptr in inference mode)
     *
     * @see hipdnn_frontend::graph::LayernormAttributes
     */
    std::array<std::shared_ptr<TensorAttributes>, 3>
        layernorm(std::shared_ptr<TensorAttributes> x,
                  std::shared_ptr<TensorAttributes> scale,
                  std::shared_ptr<TensorAttributes> bias,
                  LayernormAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Layernorm_" + std::to_string(_sub_nodes.size()));
        }

        if(x->get_name().empty())
        {
            x->set_name(attributes.get_name() + "::X");
        }
        if(scale->get_name().empty())
        {
            scale->set_name(attributes.get_name() + "::SCALE");
        }
        if(bias->get_name().empty())
        {
            bias->set_name(attributes.get_name() + "::BIAS");
        }

        auto epsilon = attributes.get_epsilon();
        if(epsilon && epsilon->get_name().empty())
        {
            epsilon->set_name(attributes.get_name() + "::EPSILON");
        }

        auto y = outputTensor(attributes.get_name() + "::Y");

        std::shared_ptr<TensorAttributes> mean = nullptr;
        std::shared_ptr<TensorAttributes> invVariance = nullptr;

        if(attributes.get_forward_phase() != NormFwdPhase::INFERENCE)
        {
            mean = outputTensor(attributes.get_name() + "::MEAN");
            invVariance = outputTensor(attributes.get_name() + "::INV_VARIANCE");
            attributes.set_mean(mean);
            attributes.set_inv_variance(invVariance);
        }

        attributes.set_x(std::move(x));
        attributes.set_scale(std::move(scale));
        attributes.set_bias(std::move(bias));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<LayerNormNode>(std::move(attributes), graph_attributes));

        return {y, mean, invVariance};
    }

    /** @brief RMS normalization forward pass
     *
     * Normalizes the input using the root mean square across the channel
     * dimension, without mean subtraction. Unlike layer normalization,
     * RMSNorm does not center the activations.
     *
     * Formula:
     * @code
     * rms[n,h,w]  = sqrt((1/C) * sum_c x[n,c,h,w]^2 + epsilon)
     * y[n,c,h,w]  = scale[c] * (x[n,c,h,w] / rms[n,h,w]) + bias[c]
     * @endcode
     * where C = number of channels.
     *
     * In training phase, the inverse RMS is also returned as an output for use
     * in the backward pass.
     *
     * @param x Input tensor [N, C, H, W, ...] (minimum 2 dimensions)
     * @param scale Per-channel scale (gamma) tensor [1, C, 1, 1, ...]
     * @param attributes Configuration including epsilon, forward phase,
     *        and optional bias [1, C, 1, 1, ...]
     * @return Array of 2 output tensors:
     *         - [0] y: Normalized output (same shape as x)
     *         - [1] invRms: Inverse RMS values (nullptr in inference mode)
     *
     * @see hipdnn_frontend::graph::RMSNormAttributes, hipdnn_frontend::graph::LayernormAttributes
     */
    std::array<std::shared_ptr<TensorAttributes>, 2>
        rmsnorm(std::shared_ptr<TensorAttributes> x,
                std::shared_ptr<TensorAttributes> scale,
                RMSNormAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("RMSNorm_" + std::to_string(_sub_nodes.size()));
        }

        auto y = outputTensor(attributes.get_name() + "::Y");

        std::shared_ptr<TensorAttributes> invRmsOut;
        if(attributes.get_forward_phase() == NormFwdPhase::TRAINING)
        {
            invRmsOut = outputTensor(attributes.get_name() + "::INV_RMS");
            attributes.set_inv_rms(invRmsOut);
        }

        attributes.set_x(std::move(x));
        attributes.set_scale(std::move(scale));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<RMSNormNode>(std::move(attributes), graph_attributes));

        return {y, invRmsOut};
    }

    /** @brief RMS normalization backward pass
     *
     * Computes gradients with respect to input, scale, and optionally bias.
     *
     * @param dy Upstream gradient (loss gradient w.r.t. output, same shape as x)
     * @param x Original input from forward pass
     * @param scale Per-channel scale (gamma)
     * @param inv_rms Saved inv_rms from the forward pass
     * @param attributes Configuration; optionally include dbias
     *        computation via set_compute_dbias(true)
     * @return Array of 3 output tensors:
     *         - [0] dx: Gradient w.r.t. input (same shape as x)
     *         - [1] dscale: Per-channel gradient w.r.t. scale
     *         - [2] dbias: Per-channel gradient w.r.t. bias; nullptr unless
     *           attributes.set_compute_dbias(true) was called before this
     *
     * @see hipdnn_frontend::graph::RMSNormBackwardAttributes
     */
    // NOLINTBEGIN(readability-identifier-naming)
    std::array<std::shared_ptr<TensorAttributes>, 3>
        rmsnorm_backward(std::shared_ptr<TensorAttributes> dy,
                         std::shared_ptr<TensorAttributes> x,
                         std::shared_ptr<TensorAttributes> scale,
                         std::shared_ptr<TensorAttributes> inv_rms,
                         RMSNormBackwardAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("RMSNormBackward_" + std::to_string(_sub_nodes.size()));
        }

        auto dx = outputTensor(attributes.get_name() + "::DX");
        auto dscale = outputTensor(attributes.get_name() + "::DSCALE");

        std::shared_ptr<TensorAttributes> dbias;
        if(attributes.get_compute_dbias())
        {
            dbias = outputTensor(attributes.get_name() + "::DBIAS");
            attributes.set_dbias(dbias);
        }

        attributes.set_dy(std::move(dy));
        attributes.set_x(std::move(x));
        attributes.set_scale(std::move(scale));
        attributes.set_inv_rms(std::move(inv_rms));
        attributes.set_dx(dx);
        attributes.set_dscale(dscale);

        _sub_nodes.emplace_back(
            std::make_shared<RMSNormBackwardNode>(std::move(attributes), graph_attributes));

        return {dx, dscale, dbias};
    }

    /** @brief Block-scale dequantization
     *
     * Dequantizes a blocked low-precision tensor using per-block scale factors.
     * Supports MX blocked data-types (mxfp8, mxbfp8, mxfp6, mxfp4).
     *
     * @param x Input blocked tensor to dequantize
     * @param scale Scale tensor for block dequantization
     * @param attributes Configuration: block_size, is_negative_scale
     * @return y: Dequantized output tensor
     *
     * @see hipdnn_frontend::graph::BlockScaleDequantizeAttributes
     */
    // NOLINTBEGIN(readability-identifier-naming)
    std::shared_ptr<TensorAttributes>
        block_scale_dequantize(std::shared_ptr<TensorAttributes> x,
                               std::shared_ptr<TensorAttributes> scale,
                               BlockScaleDequantizeAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("BlockScaleDequantize_" + std::to_string(_sub_nodes.size()));
        }

        auto y = outputTensor(attributes.get_name() + "::Y");

        attributes.set_x(std::move(x));
        attributes.set_scale(std::move(scale));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<BlockScaleDequantizeNode>(std::move(attributes), graph_attributes));

        return y;
    }

    /** @brief Block-scale quantization
     *
     * Quantizes an input tensor into a blocked low-precision representation
     * with per-block scale factors. Supports MX blocked data-types
     * (mxfp8, mxbfp8, mxfp6, mxfp4).
     *
     * @param x Input tensor to quantize
     * @param attributes Configuration: block_size, axis, transpose
     * @return [y, scale]: Quantized output tensor and computed scale tensor
     *
     * @see hipdnn_frontend::graph::BlockScaleQuantizeAttributes
     */
    // NOLINTBEGIN(readability-identifier-naming)
    std::array<std::shared_ptr<TensorAttributes>, 2>
        block_scale_quantize(std::shared_ptr<TensorAttributes> x,
                             BlockScaleQuantizeAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("BlockScaleQuantize_" + std::to_string(_sub_nodes.size()));
        }

        auto y = outputTensor(attributes.get_name() + "::Y");
        auto scale = outputTensor(attributes.get_name() + "::Scale");

        attributes.set_x(std::move(x));
        attributes.set_y(y);
        attributes.set_scale(scale);

        _sub_nodes.emplace_back(
            std::make_shared<BlockScaleQuantizeNode>(std::move(attributes), graph_attributes));

        return {y, scale};
    }

    /** @brief Unary element-wise operation
     *
     * Applies an element-wise function to a single input tensor. The operation
     * is specified by PointwiseAttributes::set_mode().
     *
     * @param in0 Input tensor (arbitrary shape)
     * @param attributes Configuration specifying the pointwise mode and any
     *        mode-specific parameters (e.g., relu_lower_clip, elu_alpha)
     * @return out0: Output tensor (same shape as in0)
     *
     * @see hipdnn_frontend::graph::PointwiseAttributes, hipdnn_frontend::PointwiseMode
     */
    std::shared_ptr<TensorAttributes> pointwise(std::shared_ptr<TensorAttributes> in0,
                                                PointwiseAttributes attributes)

    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Pointwise_" + std::to_string(_sub_nodes.size()));
        }
        if(in0->get_name().empty())
        {
            in0->set_name(attributes.get_name() + "::IN_0");
        }
        auto out0 = outputTensor(attributes.get_name() + "::OUT_0");

        attributes.set_input_0(std::move(in0));
        attributes.set_output_0(out0);

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out0;
    }

    /** @brief Binary element-wise operation
     *
     * Applies an element-wise function to two input tensors. Inputs support
     * broadcasting.
     *
     * @param in0 First input tensor
     * @param in1 Second input tensor (broadcastable to in0 shape)
     * @param attributes Configuration specifying the pointwise mode
     * @return out0: Output tensor (broadcast shape of in0 and in1)
     *
     * @see hipdnn_frontend::graph::PointwiseAttributes, hipdnn_frontend::PointwiseMode
     */
    std::shared_ptr<TensorAttributes> pointwise(std::shared_ptr<TensorAttributes> in0,
                                                std::shared_ptr<TensorAttributes> in1,
                                                PointwiseAttributes attributes)

    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Pointwise_" + std::to_string(_sub_nodes.size()));
        }
        if(in0->get_name().empty())
        {
            in0->set_name(attributes.get_name() + "::IN_0");
        }
        if(in1->get_name().empty())
        {
            in1->set_name(attributes.get_name() + "::IN_1");
        }
        auto out0 = outputTensor(attributes.get_name() + "::OUT_0");

        attributes.set_input_0(std::move(in0));
        attributes.set_input_1(std::move(in1));
        attributes.set_output_0(out0);

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out0;
    }

    /** @brief Ternary element-wise operation
     *
     * Applies an element-wise function to three input tensors.
     * Currently only BINARY_SELECT uses this overload:
     * `out[i] = in0[i] ? in1[i] : in2[i]`
     *
     * @param in0 Condition tensor (selector mask)
     * @param in1 Value selected where in0 is non-zero
     * @param in2 Value selected where in0 is zero
     * @param attributes Configuration specifying the pointwise mode
     * @return out0: Output tensor
     *
     * @see hipdnn_frontend::graph::PointwiseAttributes, hipdnn_frontend::PointwiseMode
     */
    std::shared_ptr<TensorAttributes> pointwise(std::shared_ptr<TensorAttributes> in0,
                                                std::shared_ptr<TensorAttributes> in1,
                                                std::shared_ptr<TensorAttributes> in2,
                                                PointwiseAttributes attributes)

    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Pointwise_" + std::to_string(_sub_nodes.size()));
        }
        if(in0->get_name().empty())
        {
            in0->set_name(attributes.get_name() + "::IN_0");
        }
        if(in1->get_name().empty())
        {
            in1->set_name(attributes.get_name() + "::IN_1");
        }
        if(in2->get_name().empty())
        {
            in2->set_name(attributes.get_name() + "::IN_2");
        }
        auto out0 = outputTensor(attributes.get_name() + "::OUT_0");

        attributes.set_input_0(std::move(in0));
        attributes.set_input_1(std::move(in1));
        attributes.set_input_2(std::move(in2));
        attributes.set_output_0(out0);

        _sub_nodes.emplace_back(
            std::make_shared<PointwiseNode>(std::move(attributes), graph_attributes));

        return out0;
    }

    /** @brief Reduction operation
     *
     * Reduces an input tensor along one or more dimensions using the specified
     * reduction mode. Creates a new output tensor managed by the graph.
     *
     * @param x Input tensor (arbitrary shape)
     * @param attributes Configuration specifying the reduction mode
     * @return y: Output tensor (graph-managed, shape inferred during build)
     *
     * @see ReductionAttributes, ReductionMode
     */
    std::shared_ptr<TensorAttributes> reduction(std::shared_ptr<TensorAttributes> x,
                                                ReductionAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Reduction_" + std::to_string(_sub_nodes.size()));
        }
        if(x->get_name().empty())
        {
            x->set_name(attributes.get_name() + "::X");
        }
        auto y = outputTensor(attributes.get_name() + "::Y");

        attributes.set_x(std::move(x));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<ReductionNode>(std::move(attributes), graph_attributes));

        return y;
    }

    /** @brief Reduction operation with explicit output tensor
     *
     * Reduces an input tensor along one or more dimensions using the specified
     * reduction mode. The caller provides the output tensor, allowing explicit
     * control over output shape for partial reductions.
     *
     * @param x Input tensor (arbitrary shape)
     * @param y Output tensor (caller-provided, reduced shape)
     * @param attributes Configuration specifying the reduction mode
     * @return y: The provided output tensor
     *
     * @see ReductionAttributes, ReductionMode
     */
    std::shared_ptr<TensorAttributes> reduction(std::shared_ptr<TensorAttributes> x,
                                                std::shared_ptr<TensorAttributes> y,
                                                ReductionAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Reduction_" + std::to_string(_sub_nodes.size()));
        }
        if(x->get_name().empty())
        {
            x->set_name(attributes.get_name() + "::X");
        }
        if(y->get_name().empty())
        {
            y->set_name(attributes.get_name() + "::Y");
        }

        attributes.set_x(std::move(x));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<ReductionNode>(std::move(attributes), graph_attributes));

        return y;
    }

    /** @brief Matrix multiplication
     *
     * Computes the matrix product of two tensors with optional batch dimensions.
     *
     * Formula:
     * @code
     * C[..., i, j] = sum_k( A[..., i, k] * B[..., k, j] )
     * @endcode
     *
     * Batch dimensions are broadcast when they differ (one must be divisible
     * by the other).
     *
     * @param a Left input matrix [..., M, K]
     * @param b Right input matrix [..., K, N]
     * @param attributes Additional configuration
     * @return c: Output matrix [..., M, N]
     *
     * @see hipdnn_frontend::graph::MatmulAttributes
     */
    std::shared_ptr<TensorAttributes> matmul(std::shared_ptr<TensorAttributes> a,
                                             std::shared_ptr<TensorAttributes> b,
                                             MatmulAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("Matmul_" + std::to_string(_sub_nodes.size()));
        }
        if(a->get_name().empty())
        {
            a->set_name(attributes.get_name() + "::A");
        }
        if(b->get_name().empty())
        {
            b->set_name(attributes.get_name() + "::B");
        }

        auto c = outputTensor(attributes.get_name() + "::C");

        attributes.set_a(std::move(a));
        attributes.set_b(std::move(b));
        attributes.set_c(c);

        _sub_nodes.emplace_back(
            std::make_shared<MatmulNode>(std::move(attributes), graph_attributes));

        return c;
    }

    /** @brief Add a custom operation to the graph
     *
     * Custom ops let users coordinate directly with plugins without requiring
     * hipDNN to understand the operation. hipDNN transports the tensor I/O
     * topology and an opaque byte payload, and the target plugin interprets
     * the payload.
     *
     * @param inputs Input tensors (variable length)
     * @param numOutputs Number of output tensors to create
     * @param attributes Custom op configuration including opaque payload
     * @return Vector of output tensors
     *
     * @note This operation requires a matching custom plugin to find an engine.
     *       It will fail engine selection unless a plugin is loaded that explicitly
     *       handles the specified `custom_op_id`.
     *
     * @see hipdnn_frontend::graph::CustomOpAttributes
     */
    // NOLINTBEGIN(readability-identifier-naming)
    std::vector<std::shared_ptr<TensorAttributes>>
        custom_op(std::vector<std::shared_ptr<TensorAttributes>> inputs,
                  size_t numOutputs,
                  CustomOpAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("CustomOp_" + attributes.get_custom_op_id() + "_"
                                + std::to_string(_sub_nodes.size()));
        }

        for(size_t i = 0; i < inputs.size(); ++i)
        {
            if(inputs[i] && inputs[i]->get_name().empty())
            {
                inputs[i]->set_name(attributes.get_name() + "::input_" + std::to_string(i));
            }
        }

        std::vector<std::shared_ptr<TensorAttributes>> outputTensors;
        outputTensors.reserve(numOutputs);
        for(size_t i = 0; i < numOutputs; ++i)
        {
            outputTensors.push_back(
                outputTensor(attributes.get_name() + "::output_" + std::to_string(i)));
        }

        attributes.set_inputs(std::move(inputs));
        attributes.set_outputs(outputTensors);

        _sub_nodes.emplace_back(
            std::make_shared<CustomOpNode>(std::move(attributes), graph_attributes));

        return outputTensors;
    }

#ifdef HIPDNN_ENABLE_SDPA
    /** @brief Scaled dot-product attention forward pass
     *
     * Computes scaled dot-product attention:
     * @code
     * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
     * @endcode
     *
     * Supports optional causal masking, attention bias, dropout, paged
     * attention, and FP8 quantization via descale/scale tensors.
     *
     * @param q Query tensor [B, H, S_q, D]
     * @param k Key tensor [B, H, S_kv, D]
     * @param v Value tensor [B, H, S_kv, D]
     * @param attributes Configuration: masking, dropout, attention scale,
     *        paged attention, and other SDPA options
     * @return [o, stats]: Output tensor [B, H, S_q, D] and optional softmax
     *         statistics (nullptr if generate_stats is not set)
     *
     * @see hipdnn_frontend::graph::SdpaAttributes
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::array<std::shared_ptr<TensorAttributes>, 2> sdpa(std::shared_ptr<TensorAttributes> q,
                                                          std::shared_ptr<TensorAttributes> k,
                                                          std::shared_ptr<TensorAttributes> v,
                                                          SdpaAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("SdpaFwd_" + std::to_string(_sub_nodes.size()));
        }
        if(q->get_name().empty())
        {
            q->set_name(attributes.get_name() + "::Q");
        }
        if(k->get_name().empty())
        {
            k->set_name(attributes.get_name() + "::K");
        }
        if(v->get_name().empty())
        {
            v->set_name(attributes.get_name() + "::V");
        }

        auto o = outputTensor(attributes.get_name() + "::O");
        std::array<std::shared_ptr<TensorAttributes>, 2> ret = {o, nullptr};

        attributes.set_q(std::move(q));
        attributes.set_k(std::move(k));
        attributes.set_v(std::move(v));
        attributes.set_o(o);

        if(attributes.generate_stats.has_value() && attributes.generate_stats.value())
        {
            HIPDNN_FE_LOG_INFO("SDPA node '" << attributes.get_name()
                                             << "' is configured to generate stats output.");
            auto stats = outputTensor(attributes.get_name() + "::STATS");
            attributes.set_stats(stats);
            ret[1] = stats;
        }

        _sub_nodes.emplace_back(
            std::make_shared<SdpaFwdNode>(std::move(attributes), graph_attributes));

        return ret;
    }

    /** @brief Scaled dot-product attention backward pass
     *
     * Computes gradients dQ, dK, dV for the backward pass of SDPA:
     * @code
     * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
     * @endcode
     *
     * Requires softmax statistics (logsumexp) from the forward pass, which
     * are generated when the forward pass is configured with
     * `set_generate_stats(true)`.
     *
     * @param q Query tensor from forward pass [B, H, S_q, D]
     * @param k Key tensor from forward pass [B, H, S_kv, D]
     * @param v Value tensor from forward pass [B, H, S_kv, D]
     * @param o Output tensor from forward pass [B, H, S_q, D]
     * @param dO Upstream gradient tensor [B, H, S_q, D]
     * @param stats Softmax statistics (logsumexp) from forward pass [B, H, S_q, 1]
     * @param attributes Configuration: masking, dropout, attention scale
     * @return Array of 3 output tensors:
     *         - [0] dQ: Gradient w.r.t. query [B, H, S_q, D]
     *         - [1] dK: Gradient w.r.t. key [B, H, S_kv, D]
     *         - [2] dV: Gradient w.r.t. value [B, H, S_kv, D]
     *
     * @see hipdnn_frontend::graph::SdpaBackwardAttributes, hipdnn_frontend::graph::SdpaAttributes
     */
    std::array<std::shared_ptr<TensorAttributes>, 3>
        sdpa_backward(std::shared_ptr<TensorAttributes> q, // NOLINT
                      std::shared_ptr<TensorAttributes> k,
                      std::shared_ptr<TensorAttributes> v,
                      std::shared_ptr<TensorAttributes> o,
                      std::shared_ptr<TensorAttributes> dO,
                      std::shared_ptr<TensorAttributes> stats,
                      SdpaBackwardAttributes attributes)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("SdpaBwd_" + std::to_string(_sub_nodes.size()));
        }
        if(q->get_name().empty())
        {
            q->set_name(attributes.get_name() + "::Q");
        }
        if(k->get_name().empty())
        {
            k->set_name(attributes.get_name() + "::K");
        }
        if(v->get_name().empty())
        {
            v->set_name(attributes.get_name() + "::V");
        }
        if(dO->get_name().empty())
        {
            dO->set_name(attributes.get_name() + "::dO");
        }

        auto dq = outputTensor(attributes.get_name() + "::dQ");
        auto dk = outputTensor(attributes.get_name() + "::dK");
        auto dv = outputTensor(attributes.get_name() + "::dV");

        attributes.set_q(std::move(q));
        attributes.set_k(std::move(k));
        attributes.set_v(std::move(v));
        attributes.set_o(std::move(o));
        attributes.set_do(std::move(dO));
        attributes.set_stats(std::move(stats));
        attributes.set_dq(dq);
        attributes.set_dk(dk);
        attributes.set_dv(dv);

        _sub_nodes.emplace_back(
            std::make_shared<SdpaBwdNode>(std::move(attributes), graph_attributes));

        return {dq, dk, dv};
    }
#endif // HIPDNN_ENABLE_SDPA

    /** @brief Convolution forward pass
     *
     * Computes a cross-correlation (or convolution) of the input with filters.
     *
     * Example for 2D (using NCHW notation for illustration):
     * @code
     * y[n,k,oh,ow] = sum_c,r,s  x[n, c, oh*stride_h + r*dilation_h - pad_h,
     *                                     ow*stride_w + s*dilation_w - pad_w]
     *                           * w[k, c, r, s]
     *
     * output_dim = floor((input + pad_before + pad_after
     *              - dilation * (kernel - 1) - 1) / stride) + 1
     * @endcode
     *
     * @param x Input activation tensor (batch, channels, spatial dimensions)
     * @param w Filter/weight tensor (output channels, input channels, filter spatial dims)
     * @param attributes Convolution parameters: padding, stride, dilation,
     *        convolution mode
     * @return y: Output activation tensor
     *
     * @see hipdnn_frontend::graph::ConvFpropAttributes
     */
    // NOLINTBEGIN(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> resample_fwd(std::shared_ptr<TensorAttributes> x,
                                                   ResampleFwdAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("ResampleFwd_" + std::to_string(_sub_nodes.size()));
        }
        if(x->get_name().empty())
        {
            x->set_name(attributes.get_name() + "::X");
        }

        auto y = outputTensor(attributes.get_name() + "::Y");

        attributes.set_x(std::move(x));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<ResampleFwdNode>(std::move(attributes), graph_attributes));

        return y;
    }

    // NOLINTBEGIN(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> conv_fprop(std::shared_ptr<TensorAttributes> x,
                                                 std::shared_ptr<TensorAttributes> w,
                                                 ConvFpropAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("ConvolutionFprop_" + std::to_string(_sub_nodes.size()));
        }
        if(x->get_name().empty())
        {
            x->set_name(attributes.get_name() + "::X");
        }
        if(w->get_name().empty())
        {
            w->set_name(attributes.get_name() + "::W");
        }

        auto y = outputTensor(attributes.get_name() + "::Y");

        attributes.set_x(std::move(x));
        attributes.set_w(std::move(w));
        attributes.set_y(y);

        _sub_nodes.emplace_back(
            std::make_shared<ConvolutionFpropNode>(std::move(attributes), graph_attributes));

        return y;
    }

    /** @brief Convolution data gradient (backward data)
     *
     * Computes the gradient of the loss with respect to the convolution input,
     * given the output gradient and the filter weights. Used during
     * backpropagation.
     *
     * Example for 2D (using NCHW notation for illustration):
     * @code
     * dx[n,c,h,w] = sum_k,r,s  dy[n, k, p, q] * w[k, c, r, s]
     *   where p = (h + pad_h - r*dilation_h) / stride_h  (integer, in [0, H_out))
     *         q = (w + pad_w - s*dilation_w) / stride_w  (integer, in [0, W_out))
     * @endcode
     *
     * @param dy Upstream gradient (loss gradient w.r.t. conv output)
     * @param w Filter/weight tensor
     * @param attributes Convolution parameters: padding, stride, dilation
     *        (must match forward pass)
     * @return dx: Gradient w.r.t. input (same shape as forward input)
     *
     * @note If `dx` dimensions are not provided, the channel count is
     *       inferred assuming `groups = 1`. For grouped convolutions,
     *       set dimensions on the returned `dx` tensor before graph
     *       validation/finalization to avoid an incorrect channel count
     *       on the inferred input-gradient tensor.
     *
     * @see hipdnn_frontend::graph::ConvDgradAttributes
     */
    // NOLINTBEGIN(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> conv_dgrad(std::shared_ptr<TensorAttributes> dy,
                                                 std::shared_ptr<TensorAttributes> w,
                                                 ConvDgradAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("ConvolutionDgrad_" + std::to_string(_sub_nodes.size()));
        }
        if(dy->get_name().empty())
        {
            dy->set_name(attributes.get_name() + "::DY");
        }
        if(w->get_name().empty())
        {
            w->set_name(attributes.get_name() + "::W");
        }

        auto dx = outputTensor(attributes.get_name() + "::DX");

        attributes.set_dy(std::move(dy));
        attributes.set_w(std::move(w));
        attributes.set_dx(dx);

        _sub_nodes.emplace_back(
            std::make_shared<ConvolutionDgradNode>(std::move(attributes), graph_attributes));

        return dx;
    }

    /** @brief Convolution weight gradient (backward weights)
     *
     * Computes the gradient of the loss with respect to the filter weights,
     * given the output gradient and the original input. Used during
     * backpropagation.
     *
     * Example for 2D (using NCHW notation for illustration):
     * @code
     * dw[k,c,r,s] = sum_n,p,q  dy[n, k, p, q] * x[n, c, h, w]
     *   where h = p*stride_h - pad_h + r*dilation_h
     *         w = q*stride_w - pad_w + s*dilation_w
     * @endcode
     *
     * @param dy Upstream gradient (loss gradient w.r.t. conv output)
     * @param x Original input activation tensor
     * @param attributes Convolution parameters: padding, stride, dilation
     *        (must match forward pass)
     * @return dw: Gradient w.r.t. filter weights (same shape as forward weights)
     *
     * @note If `dw` dimensions are not provided, the channel count is
     *       inferred assuming `groups = 1`. For grouped convolutions,
     *       set dimensions on the returned `dw` tensor before graph
     *       validation/finalization to avoid an incorrect channel count
     *       on the inferred weight tensor.
     *
     * @see hipdnn_frontend::graph::ConvWgradAttributes
     */
    // NOLINTBEGIN(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> conv_wgrad(std::shared_ptr<TensorAttributes> dy,
                                                 std::shared_ptr<TensorAttributes> x,
                                                 ConvWgradAttributes attributes)
    // NOLINTEND(readability-identifier-naming)
    {
        if(attributes.get_name().empty())
        {
            attributes.set_name("ConvolutionWgrad_" + std::to_string(_sub_nodes.size()));
        }
        if(x->get_name().empty())
        {
            x->set_name(attributes.get_name() + "::X");
        }
        if(dy->get_name().empty())
        {
            dy->set_name(attributes.get_name() + "::DY");
        }

        auto dw = outputTensor(attributes.get_name() + "::DW");

        attributes.set_x(std::move(x));
        attributes.set_dy(std::move(dy));
        attributes.set_dw(dw);

        _sub_nodes.emplace_back(
            std::make_shared<ConvolutionWgradNode>(std::move(attributes), graph_attributes));

        return dw;
    }

    /**
     * @brief Set the preferred engine ID for execution plan selection
     * @param engineId Engine ID to prefer, or std::nullopt to clear
     * @return Reference to this Graph for method chaining
     */
    // NOLINTBEGIN(readability-identifier-naming)
    Graph& set_preferred_engine_id_ext(std::optional<int64_t> engineId)
    // NOLINTEND(readability-identifier-naming)
    {
        _preferredEngineId = engineId;
        return *this;
    }

    /**
     * @brief Set the preferred engine by name
     * @param engineName Engine name to look up; empty string clears the preference
     * @return Reference to this Graph for method chaining
     */
    // NOLINTBEGIN(readability-identifier-naming)
    Graph& set_preferred_engine_id_ext(const std::string& engineName)
    // NOLINTEND(readability-identifier-naming)
    {
        if(engineName.empty())
        {
            _preferredEngineId = std::nullopt;
            HIPDNN_FE_LOG_INFO("Cleared preferred engine ID (empty string)");
            return *this;
        }

        auto engineId = hipdnn_data_sdk::utilities::engineNameToId(engineName);
        _preferredEngineId = engineId;

        HIPDNN_FE_LOG_INFO("Engine name '" << engineName << "' mapped to ID: " << engineId);
        return *this;
    }

#ifdef HIPDNN_ENABLE_SDPA
    /// Enable or disable runtime tensor-shape overrides for this graph.
    Graph& set_override_shape_enabled(bool enabled) // NOLINT(readability-identifier-naming)
    {
        if((_graphDesc && _graphDesc->valid())
           || (_executionPlanDesc && _executionPlanDesc->valid()))
        {
            HIPDNN_FE_LOG_WARN(
                "set_override_shape_enabled() called after graph descriptors or execution plans "
                "were created. Rebuild the graph for this flag to affect backend plugin "
                "selection and execution-plan override eligibility.");
        }
        _isOverrideShapeEnabled = enabled;
        return *this;
    }

    /// Whether this graph has opted into runtime tensor-shape overrides.
    bool is_override_shape_enabled() const // NOLINT(readability-identifier-naming)
    {
        return _isOverrideShapeEnabled;
    }
#endif // HIPDNN_ENABLE_SDPA

    /**
     * @brief Create a new tensor with similar properties to an existing tensor
     * @param tensor The tensor to copy properties from
     * @param name Optional name for the new tensor
     * @return Shared pointer to the newly created TensorAttributes
     *
     * Creates a new TensorAttributes object by copying properties from the provided
     * tensor, but clears the UID and optionally assigns a new name. This is useful
     * for creating tensors with similar dimensions and data types but representing
     * different data.
     *
     * @code{.cpp}
     * // Create a tensor similar to x but with a different UID
     * auto y = Graph::tensor_like(x, "output");
     * y->set_uid(2);
     * @endcode
     *
     * @see tensor() for creating a tensor with all properties preserved
     */
    // NOLINTBEGIN(readability-identifier-naming)
    static std::shared_ptr<TensorAttributes>
        tensor_like(const std::shared_ptr<TensorAttributes>& tensor, const std::string& name = "")
    // NOLINTEND(readability-identifier-naming)
    {
        auto newTensor = std::make_shared<TensorAttributes>(*tensor);

        newTensor->clear_uid();
        newTensor->set_name(name);

        return newTensor;
    }

    /**
     * @brief Create a new tensor from existing tensor attributes
     * @param tensor The tensor attributes to copy
     * @return Shared pointer to the newly created TensorAttributes
     *
     * Creates a new TensorAttributes object as a copy of the provided tensor,
     * preserving all properties including UID. This is the standard way to
     * create a tensor for use in graph operations.
     *
     * @code{.cpp}
     * // Create a tensor from attributes
     * auto x = Graph::tensor(TensorAttributes()
     *              .set_dim({1, 64, 28, 28})
     *              .set_stride({50176, 784, 28, 1})
     *              .set_data_type(DataType::HALF)
     *              .set_uid(0));
     * @endcode
     *
     * @note This creates a tensor descriptor (shape, type, strides) only.
     *       No device memory is allocated. Device pointers are provided
     *       at execution time via the variant pack.
     *
     * @see execute() for passing device memory pointers at execution time
     * @see tensor_like() for creating a tensor with cleared UID and custom name
     */
    static std::shared_ptr<TensorAttributes> tensor(const TensorAttributes& tensor)
    {
        auto newTensor = std::make_shared<TensorAttributes>(tensor);

        return newTensor;
    }
};

} // namespace hipdnn_frontend::graph
