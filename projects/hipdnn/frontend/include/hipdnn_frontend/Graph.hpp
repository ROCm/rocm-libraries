// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Graph.hpp
 * @brief Main Graph class for building and executing deep learning operations
 *
 * This file contains the Graph class, which is the primary interface for users
 * to construct computational graphs of deep learning operations and execute them
 * on AMD GPUs via the hipDNN backend.
 *
 * @section graph_overview Overview
 *
 * The Graph class provides a fluent API for:
 * - Creating tensors with specified dimensions and data types
 * - Adding operations (convolution, batch normalization, pointwise, matmul)
 * - Building and executing plans on GPU
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

#include <HipdnnBackendFlatbufferData.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/CreateBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/GraphDetail.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNodeVarianceExt.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/detail/TopologicalSortingUtils.hpp>
#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <hipdnn_data_sdk/utilities/json/Graph.hpp>
#endif
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace hipdnn_frontend::graph
{

/**
 * @class Graph
 * @brief The main class for building and executing hipDNN computational graphs
 *
 * Graph is the central class in hipDNN Frontend. It allows users to:
 * - Define tensors and their properties
 * - Add operation nodes (convolution, batchnorm, pointwise, matmul)
 * - Build execution plans
 * - Execute the graph on AMD GPUs
 *
 * The Graph class uses a fluent interface pattern, where setter methods return
 * a reference to the graph for method chaining.
 *
 * @see TensorAttributes, ConvFpropAttributes, BatchnormAttributes, PointwiseAttributes
 */
class Graph : public INode
{
private:
    std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> _graphDesc;
    std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> _engineConfigDesc;
    std::unique_ptr<detail::ScopedHipdnnBackendDescriptor> _executionPlanDesc;

    std::optional<int64_t> _preferredEngineId;

    static std::optional<int64_t> getDefaultEngineId()
    {
        static const std::optional<int64_t> s_defaultId = []() -> std::optional<int64_t> {
            auto envStr = hipdnn_data_sdk::utilities::trim(
                hipdnn_data_sdk::utilities::getEnv("HIPDNN_DEFAULT_ENGINE"));
            if(envStr.empty())
            {
                return std::nullopt;
            }
            auto engineId = hipdnn_data_sdk::utilities::engineNameToId(envStr);
            HIPDNN_FE_LOG_INFO("HIPDNN_DEFAULT_ENGINE='" << envStr
                                                         << "' mapped to engine ID: " << engineId);
            return engineId;
        }();
        return s_defaultId;
    }

    void assignUnsetTensorUids()
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);
        auto usedIds = getUsedIds(allTensors);
        populateHipdnnTensorIds(allTensors, usedIds);
    }

    static std::shared_ptr<TensorAttributes> outputTensor(const std::string& name)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_name(name).set_is_virtual(true);
        return tensor;
    }

    Error initializeEngineConfig(hipdnnBackendDescriptor_t engineHeuristicDesc)
    {
        std::vector<std::unique_ptr<detail::ScopedHipdnnBackendDescriptor>> engineConfigs;
        std::vector<int64_t> engineIds;
        auto defaultEngineId = getDefaultEngineId();
        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::getEngineConfigs(
            engineConfigs,
            engineIds,
            engineHeuristicDesc,
            _preferredEngineId.has_value() || defaultEngineId.has_value()));

        // Select engine config based on preferred ID or use first available
        size_t selectedIndex = 0;
        if(defaultEngineId)
        {
            auto defaultId = defaultEngineId.value();
            auto it = std::find(engineIds.begin(), engineIds.end(), defaultId);
            if(it != engineIds.end())
            {
                selectedIndex = static_cast<size_t>(std::distance(engineIds.begin(), it));
                HIPDNN_FE_LOG_INFO("Default engine id " << defaultId
                                                        << " found, using it for execution plan.");
            }
            else
            {
                HIPDNN_FE_LOG_INFO("Default engine id "
                                   << defaultId << " not found, using top engine config instead.");
            }
        }

        if(_preferredEngineId.has_value())
        {
            bool found = false;

            for(size_t i = 0; i < engineIds.size(); ++i)
            {

                if(engineIds[i] == _preferredEngineId.value())
                {
                    selectedIndex = i;
                    found = true;
                    break;
                }
            }

            if(!found)
            {
                HIPDNN_FE_LOG_WARN("Preferred engine id "
                                   << _preferredEngineId.value()
                                   << " not found, using top engine config instead.");
            }
        }

        HIPDNN_FE_LOG_INFO("Selected engine id " << engineIds[selectedIndex]
                                                 << " for execution plan.");
        _engineConfigDesc = std::move(engineConfigs[selectedIndex]);

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

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendSetAttribute(engineConfigDesc->get(),
                                                         HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         &engineDesc.get()),
            "Failed to set engine on the engine config descriptor.");

        _engineConfigDesc = std::move(engineConfigDesc);
        return {ErrorCode::OK, ""};
    }

    detail::GraphStructure buildAdjacencyList(
        const std::unordered_map<std::shared_ptr<TensorAttributes>, size_t>& tensorToOriginNode)
        const
    {
        size_t nodeCount = _sub_nodes.size();
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
                    size_t outputNodeIndex = it->second;
                    structure.adjacencyList[outputNodeIndex].push_back(inputNodeIndex);
                }
            }
        }

        return structure;
    }

    std::unordered_map<std::shared_ptr<TensorAttributes>, size_t> buildTensorToOriginNodeMap() const
    {
        std::unordered_map<std::shared_ptr<TensorAttributes>, size_t> tensorToOriginNode;
        size_t nodeCount = _sub_nodes.size();

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
        std::unordered_set<std::shared_ptr<TensorAttributes>> const& allTensors)
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
        std::unordered_set<std::shared_ptr<TensorAttributes>> const& allTensors)
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

    flatbuffers::DetachedBuffer buildFlatbufferOperationGraphConst() const
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        flatbuffers::FlatBufferBuilder builder;

        std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
            tensorAttributes;
        for(auto& tensor : allTensors)
        {
            if(tensor)
            {
                tensorAttributes.emplace_back(tensor->pack_attributes(builder));
            }
        }

        std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
        for(auto& node : _sub_nodes)
        {
            if(node)
            {
                nodes.emplace_back(node->pack_node(builder));
            }
        }
        auto graph = hipdnn_data_sdk::data_objects::CreateGraphDirect(
            builder,
            graph_attributes.get_name().c_str(),
            toSdkType(graph_attributes.get_compute_data_type()),
            toSdkType(graph_attributes.get_intermediate_data_type()),
            toSdkType(graph_attributes.get_io_data_type()),
            &tensorAttributes,
            &nodes,
            _preferredEngineId);

        builder.Finish(graph);
        return builder.Release();
    }

    Error deserializeFromFlatBuffer(const hipdnn_data_sdk::data_objects::Graph* fbGraph)
    {
        // Set graph attributes from FlatBuffer
        if(fbGraph->name() != nullptr)
        {
            set_name(fbGraph->name()->c_str());
        }
        set_compute_data_type(fromSdkType(fbGraph->compute_data_type()));
        set_intermediate_data_type(fromSdkType(fbGraph->intermediate_data_type()));
        set_io_data_type(fromSdkType(fbGraph->io_data_type()));

        if(fbGraph->preferred_engine_id().has_value())
        {
            _preferredEngineId = fbGraph->preferred_engine_id().value();
        }

        // Build tensorMap from FlatBuffer tensors
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
        if(fbGraph->tensors() != nullptr)
        {
            for(const auto* fbTensor : *fbGraph->tensors())
            {
                auto tensor = TensorAttributes::fromFlatBuffer(fbTensor);
                if(tensor != nullptr && tensor->has_uid())
                {
                    tensorMap[tensor->get_uid()] = tensor;
                }
            }
        }

        // Create nodes from FlatBuffer
        if(fbGraph->nodes() != nullptr)
        {
            for(const auto* fbNode : *fbGraph->nodes())
            {
                auto type = fbNode->attributes_type();

                switch(type)
                {
                case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes:
                {
                    auto attr = BatchnormAttributes::fromFlatBuffer(
                        fbNode->attributes_as_BatchnormAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(
                        std::make_shared<BatchnormNode>(std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
                {
                    auto attr = BatchnormBackwardAttributes::fromFlatBuffer(
                        fbNode->attributes_as_BatchnormBackwardAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(
                        std::make_shared<BatchnormBackwardNode>(std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
                {
                    auto attr = BatchnormInferenceAttributes::fromFlatBuffer(
                        fbNode->attributes_as_BatchnormInferenceAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(std::make_shared<BatchnormInferenceNode>(
                        std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::
                    BatchnormInferenceAttributesVarianceExt:
                {
                    auto attr = BatchnormInferenceAttributesVarianceExt::fromFlatBuffer(
                        fbNode->attributes_as_BatchnormInferenceAttributesVarianceExt(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(std::make_shared<BatchnormInferenceNodeVarianceExt>(
                        std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
                {
                    auto attr = ConvFpropAttributes::fromFlatBuffer(
                        fbNode->attributes_as_ConvolutionFwdAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(
                        std::make_shared<ConvolutionFpropNode>(std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
                {
                    auto attr = ConvDgradAttributes::fromFlatBuffer(
                        fbNode->attributes_as_ConvolutionBwdAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(
                        std::make_shared<ConvolutionDgradNode>(std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
                {
                    auto attr = ConvWgradAttributes::fromFlatBuffer(
                        fbNode->attributes_as_ConvolutionWrwAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(
                        std::make_shared<ConvolutionWgradNode>(std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes:
                {
                    auto attr = PointwiseAttributes::fromFlatBuffer(
                        fbNode->attributes_as_PointwiseAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(
                        std::make_shared<PointwiseNode>(std::move(attr), graph_attributes));
                    break;
                }
                case hipdnn_data_sdk::data_objects::NodeAttributes::MatmulAttributes:
                {
                    auto attr = MatmulAttributes::fromFlatBuffer(
                        fbNode->attributes_as_MatmulAttributes(), tensorMap);
                    if(fbNode->name() != nullptr)
                    {
                        attr.set_name(fbNode->name()->str());
                    }
                    _sub_nodes.emplace_back(
                        std::make_shared<MatmulNode>(std::move(attr), graph_attributes));
                    break;
                }
                default:
                    return {ErrorCode::INVALID_VALUE, "Unsupported node type in deserialization"};
                }
            }
        }

        return {ErrorCode::OK, ""};
    }

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
    Error deserializeImpl(const nlohmann::json& j)
    {
        // Convert JSON to FlatBuffer, then deserialize
        flatbuffers::FlatBufferBuilder builder;
        auto graphOffset
            = hipdnn_data_sdk::json::to<hipdnn_data_sdk::data_objects::Graph>(builder, j);
        builder.Finish(graphOffset);
        auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(builder.GetBufferPointer());

        return deserializeFromFlatBuffer(fbGraph);
    }
#endif

public:
    /**
     * @brief Construct an empty Graph
     */
    Graph()
        : INode(GraphAttributes{})
    {
        HIPDNN_FE_LOG_INFO("Creating new Graph instance");
    }

    /**
     * @brief Validate the graph structure and tensor configurations
     * @return Error indicating success or describing validation failures
     *
     * Validates that:
     * - All tensors have required attributes set
     * - No duplicate tensor UIDs exist
     * - Graph is a valid DAG (no cycles)
     * - Graph is connected (no orphaned nodes)
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

    Error checkNoDuplicateTensorIds()
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        return checkNoDuplicateTensorIdsImpl(allTensors);
    }

    /// Checks if all tensors in the graph have UIDs assigned.
    /// Returns an error if any tensor is missing a UID.
    Error checkTensorUidsSet() const
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        return checkTensorUidsSetImpl(allTensors);
    }

    /// Returns a map of UID -> TensorAttributes for all tensors in the graph.
    /// Tensors without UIDs are skipped (no error is generated).
    /// @return Map from tensor UID to tensor attributes
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

    /// Returns a map of name -> TensorAttributes for all tensors in the graph.
    /// Tensors without names are skipped (no error is generated).
    /// @return Map from tensor name to tensor attributes
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

    Error topologicallySortGraph()
    {
        size_t nodeCount = _sub_nodes.size();

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

    flatbuffers::DetachedBuffer buildFlatbufferOperationGraph()
    {
        assignUnsetTensorUids();

        return buildFlatbufferOperationGraphConst();
    }

    /**
     * @brief Build the operation graph descriptor
     * @param handle The hipDNN handle
     * @return Error indicating success or failure
     *
     * This is typically called internally by build(). It creates the backend
     * operation graph descriptor from the frontend graph representation.
     */
    Error build_operation_graph(hipdnnHandle_t handle) // NOLINT(readability-identifier-naming)
    {
        HIPDNN_FE_LOG_INFO("Building operation graph " << graph_attributes.get_name());

        auto serializedGraph = buildFlatbufferOperationGraph();
        _graphDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            serializedGraph.data(), serializedGraph.size());

        if(!_graphDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend graph descriptor for the graph."};
        }

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendSetAttribute(_graphDesc->get(),
                                                         HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                                         HIPDNN_TYPE_HANDLE,
                                                         1,
                                                         &handle),
            "Failed to set handle on the graph.");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(_graphDesc->get()),
            "Failed to finalize backend descriptor for the graph");

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Get available configuration knobs for a specific engine
     * @param engineId The engine ID to query
     * @param knobs Output vector of available Knob objects
     * @return Error indicating success or failure
     *
     * @see Knob, KnobSetting
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error get_knobs_for_engine(int64_t engineId, std::vector<Knob>& knobs) const
    {
        if(!_graphDesc || !_graphDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot get knobs "
                    "for engine."};
        }

        detail::ScopedHipdnnBackendDescriptor engineDesc;

        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineDescriptorForGraph(
            engineDesc, _graphDesc->get(), engineId));

        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::getKnobsForEngine(knobs, engineDesc.get()));

        return {ErrorCode::OK, ""};
    }

    // NOLINTNEXTLINE(readability-identifier-naming, readability-convert-member-functions-to-static)
    Error get_knob_lookup_for_engine(int64_t engineId,
                                     std::unordered_map<KnobType_t, Knob>& knobs) const
    {
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
     * @param rankedEngineIds Output vector of engine IDs, ranked by expected performance
     * @param modes Heuristic modes to use for ranking
     * @return Error indicating success or failure
     */
    // NOLINTNEXTLINE(readability-identifier-naming, readability-convert-member-functions-to-static)
    Error get_ranked_engine_ids(std::vector<int64_t>& rankedEngineIds,
                                const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK})
    {
        detail::ScopedHipdnnBackendDescriptor engineHeuristicDesc;
        HIPDNN_CHECK_ERROR(hipdnn_frontend::detail::createEngineHeuristicDescriptorForGraph(
            engineHeuristicDesc, _graphDesc->get(), modes));

        std::vector<std::unique_ptr<detail::ScopedHipdnnBackendDescriptor>> engineConfigs;
        HIPDNN_CHECK_ERROR(detail::getEngineConfigs(
            engineConfigs, rankedEngineIds, engineHeuristicDesc.get(), true));

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Create execution plans using heuristics
     * @param modes Heuristic modes to use for engine selection
     * @return Error indicating success or failure
     *
     * Creates execution plans by querying the backend for available engines
     * and selecting based on the specified heuristic modes.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_execution_plans(const std::vector<HeuristicMode>& modes
                                 = {HeuristicMode::FALLBACK})
    {
        HIPDNN_FE_LOG_INFO("Creating execution plans for graph " << graph_attributes.get_name());

        if(!_graphDesc || !_graphDesc->valid())
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

        if(!_executionPlanDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend execution descriptor."};
        }

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Create an execution plan with specific engine and knob settings
     * @param engineId The engine ID to use
     * @param settings Vector of KnobSetting objects to configure the engine
     * @return Error indicating success or failure
     *
     * This method allows fine-grained control over engine selection and
     * configuration through knob settings.
     *
     * @see Knob, KnobSetting, get_knobs_for_engine()
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>& settings)
    {
        HIPDNN_FE_LOG_INFO("Creating execution plans for graph " << graph_attributes.get_name());

        if(!_graphDesc || !_graphDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot create "
                    "execution plan."};
        }

        std::unordered_map<KnobType_t, Knob> existingKnobs;
        HIPDNN_CHECK_ERROR(get_knob_lookup_for_engine(engineId, existingKnobs));
        HIPDNN_CHECK_ERROR(initializeEngineConfig(engineId));

        std::vector<KnobSetting> validatedSettings;
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

        if(!validatedSettings.empty())
        {
            std::vector<flatbuffers::DetachedBuffer> knobBuffers;
            knobBuffers.reserve(validatedSettings.size());

            for(const auto& setting : validatedSettings)
            {
                flatbuffers::FlatBufferBuilder builder;
                auto knobSettingOffset = setting.packKnobSetting(builder);
                builder.Finish(knobSettingOffset);
                knobBuffers.push_back(builder.Release());
            }

            std::vector<hipdnnBackendFlatbufferData_t> flatbufferDataArray;
            flatbufferDataArray.reserve(knobBuffers.size());

            for(const auto& buffer : knobBuffers)
            {
                hipdnnBackendFlatbufferData_t fbData;
                fbData.ptr = buffer.data();
                fbData.size = buffer.size();
                flatbufferDataArray.push_back(fbData);
            }

            HIPDNN_RETURN_ON_BACKEND_FAILURE(detail::hipdnnBackend()->backendSetAttribute(
                                                 _engineConfigDesc->get(),
                                                 HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                                 HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                 static_cast<int64_t>(flatbufferDataArray.size()),
                                                 flatbufferDataArray.data()),
                                             "Failed to set knob settings on engine config.");
        }

        // Finalize engine config after knobs have been set
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(_engineConfigDesc->get()),
            "Failed to finalize engine config descriptor");

        // Create execution plan descriptor
        _executionPlanDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

        if(!_executionPlanDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend execution descriptor."};
        }

        return {ErrorCode::OK, ""};
    }

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

    /// Serialize to FlatBuffer DetachedBuffer (const version)
    /// Returns error if tensor UIDs are not set
    Error toFlatBuffer(flatbuffers::DetachedBuffer& buffer) const
    {
        HIPDNN_CHECK_ERROR(checkTensorUidsSet());
        buffer = buildFlatbufferOperationGraphConst();
        return {ErrorCode::OK, ""};
    }

    /// Serialize to FlatBuffer DetachedBuffer (non-const version)
    /// Assigns tensor UIDs if not already set
    flatbuffers::DetachedBuffer toFlatBuffer()
    {
        assignUnsetTensorUids();
        return buildFlatbufferOperationGraphConst();
    }

    /// Deserialize from FlatBuffer Graph object
    Error fromFlatBuffer(const hipdnn_data_sdk::data_objects::Graph* fbGraph)
    {
        try
        {
            return deserializeFromFlatBuffer(fbGraph);
        }
        catch(const std::out_of_range& e)
        {
            return {ErrorCode::INVALID_VALUE,
                    std::string("Deserialization failed - missing tensor or invalid reference: ")
                        + e.what()};
        }
        catch(const std::exception& e)
        {
            return {ErrorCode::INVALID_VALUE, std::string("Deserialization failed: ") + e.what()};
        }
    }

    /// Deserialize from FlatBuffer DetachedBuffer
    Error fromFlatBuffer(const flatbuffers::DetachedBuffer& buffer)
    {
        auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());
        return fromFlatBuffer(fbGraph);
    }

    /// Serialize to FlatBuffer DetachedBuffer (const version)
    /// Returns error if tensor UIDs are not set
    Error serialize(flatbuffers::DetachedBuffer& buffer) const
    {
        return toFlatBuffer(buffer);
    }

    /// Deserialize from FlatBuffer Graph object
    Error deserialize(const hipdnn_data_sdk::data_objects::Graph* fbGraph)
    {
        return fromFlatBuffer(fbGraph);
    }

    /// Deserialize from FlatBuffer DetachedBuffer
    Error deserialize(const flatbuffers::DetachedBuffer& buffer)
    {
        return fromFlatBuffer(buffer);
    }

    /// Serialize to binary (const version)
    /// Returns error if tensor UIDs are not set
    Error serialize(std::vector<uint8_t>& data) const
    {
        HIPDNN_CHECK_ERROR(checkTensorUidsSet());
        auto buffer = buildFlatbufferOperationGraphConst();
        data.assign(buffer.data(), buffer.data() + buffer.size());
        return {ErrorCode::OK, ""};
    }

    /// Serialize to binary (non-const version)
    /// Assigns tensor UIDs if not already set
    std::vector<uint8_t> toBinary()
    {
        assignUnsetTensorUids();
        auto buffer = buildFlatbufferOperationGraphConst();
        return {buffer.data(), buffer.data() + buffer.size()};
    }

    /// Deserialize from binary packed FlatBuffer
    Error deserialize([[maybe_unused]] hipdnnHandle_t handle, const std::vector<uint8_t>& data)
    {
        auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(data.data());
        return fromFlatBuffer(fbGraph);
    }

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
    /// Serialize to JSON (const version)
    /// Returns error if tensor UIDs are not set
    ///
    /// Flow: Frontend → FlatBuffer binary → JSON
    /// GetGraph() is zero-copy (just a pointer into the buffer), so the only
    /// serialization cost is buildFlatbufferOperationGraphConst(). This keeps
    /// JSON serialization logic centralized in data_sdk.
    Error serialize(nlohmann::json& j) const
    {
        HIPDNN_CHECK_ERROR(checkTensorUidsSet());
        auto buffer = buildFlatbufferOperationGraphConst();
        // GetGraph returns a pointer view into buffer (zero-copy, no unpacking)
        auto sdkGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

        j = *sdkGraph;

        return {ErrorCode::OK, ""};
    }

    /// Serialize to JSON (non-const version)
    /// Assigns tensor UIDs if not already set
    nlohmann::json toJson()
    {
        assignUnsetTensorUids();
        auto buffer = buildFlatbufferOperationGraphConst();
        auto sdkGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

        return *sdkGraph;
    }

    /// Deserialize from JSON
    Error deserialize(const nlohmann::json& j)
    {
        try
        {
            return deserializeImpl(j);
        }
        catch(const std::out_of_range& e)
        {
            return {ErrorCode::INVALID_VALUE,
                    std::string("Deserialization failed - missing tensor or invalid reference: ")
                        + e.what()};
        }
        catch(const nlohmann::json::exception& e)
        {
            return {ErrorCode::INVALID_VALUE,
                    std::string("Deserialization failed - malformed JSON: ") + e.what()};
        }
        catch(const std::exception& e)
        {
            return {ErrorCode::INVALID_VALUE, std::string("Deserialization failed: ") + e.what()};
        }
    }
#endif

    /**
     * @brief Finalize the execution plan
     * @return Error indicating success or failure
     *
     * Called internally by build() after create_execution_plans().
     */
    Error build_plans() // NOLINT(readability-identifier-naming)
    {
        HIPDNN_FE_LOG_INFO("Building plans for graph " << graph_attributes.get_name());

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendSetAttribute(_executionPlanDesc->get(),
                                                         HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         &_engineConfigDesc->get()),
            "Failed to set the engine config on execution plan.");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(_executionPlanDesc->get()),
            "Failed to finalize execution plan descriptor");

        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Build the complete graph and create execution plans
     * @param handle The hipDNN handle
     * @param modes Heuristic modes for engine selection
     * @param policy Build plan policy (currently only HEURISTICS_CHOICE is used)
     * @param do_multithreaded_builds Reserved for future use
     * @return Error indicating success or failure
     *
     * This is the main method to prepare a graph for execution. It performs:
     * 1. Graph validation
     * 2. Operation graph building
     * 3. Execution plan creation
     * 4. Plan building
     *
     * @code{.cpp}
     * hipdnnHandle_t handle;
     * hipdnnCreate(&handle);
     * Error err = graph.build(handle);
     * if(err.is_bad()) { handleError(); }
     * @endcode
     */
    // NOLINTBEGIN(readability-identifier-naming)
    Error build(hipdnnHandle_t handle,
                std::vector<HeuristicMode> const& modes = {HeuristicMode::FALLBACK},
                [[maybe_unused]] BuildPlanPolicy policy = BuildPlanPolicy::HEURISTICS_CHOICE,
                [[maybe_unused]] bool do_multithreaded_builds = false)
    // NOLINTEND(readability-identifier-naming)
    {
        auto graphName
            = graph_attributes.get_name().empty() ? "unnamed" : graph_attributes.get_name();
        HIPDNN_FE_LOG_INFO("BUILD with handle for graph '"
                           << graphName << "', policy: " << static_cast<int>(policy)
                           << ", modes count: " << modes.size());

        HIPDNN_CHECK_ERROR(validate());
        HIPDNN_CHECK_ERROR(build_operation_graph(handle));
        HIPDNN_CHECK_ERROR(create_execution_plans(modes));
        HIPDNN_CHECK_ERROR(check_support());
        HIPDNN_CHECK_ERROR(build_plans());

        HIPDNN_FE_LOG_INFO("BUILD ALL OK for graph " << graphName);
        return {ErrorCode::OK, ""};
    }

    /**
     * @brief Get the workspace memory size required for execution
     * @param workspaceSize Output parameter for the workspace size in bytes
     * @return Error indicating success or failure
     *
     * Call this after build() to determine how much workspace memory to allocate.
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
     * @param handle The hipDNN handle
     * @param tensorLookup Map from std::shared_ptr<TensorAttributes> (tensor handles) to device memory pointers
     * @param workspace Pointer to workspace memory (can be nullptr if size is 0)
     * @return Error indicating success or failure
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
     * @param handle The hipDNN handle
     * @param variantPack Map from tensor UID to device memory pointers
     * @param workspace Pointer to workspace memory (can be nullptr if size is 0)
     * @return Error indicating success or failure
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

        auto variantPackDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
            HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        if(!variantPackDesc || !variantPackDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create variant pack descriptor."};
        }

        //split variant_pack into vector of keys and vector of values
        std::vector<int64_t> variantPackKeys;
        std::vector<void*> variantPackValues;
        variantPackKeys.reserve(variantPack.size());
        variantPackValues.reserve(variantPack.size());
        for(const auto& [key, value] : variantPack)
        {
            variantPackKeys.push_back(key);
            variantPackValues.push_back(value);
        }

        HIPDNN_RETURN_ON_BACKEND_FAILURE(detail::hipdnnBackend()->backendSetAttribute(
                                             variantPackDesc->get(),
                                             HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                             HIPDNN_TYPE_VOID_PTR,
                                             static_cast<int64_t>(variantPackValues.size()),
                                             variantPackValues.data()),
                                         "failed to set the variant pack data pointers.");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(detail::hipdnnBackend()->backendSetAttribute(
                                             variantPackDesc->get(),
                                             HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                             HIPDNN_TYPE_INT64,
                                             static_cast<int64_t>(variantPackKeys.size()),
                                             variantPackKeys.data()),
                                         "failed to set the variant pack unique ids.");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendSetAttribute(variantPackDesc->get(),
                                                         HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                         HIPDNN_TYPE_VOID_PTR,
                                                         1,
                                                         &workspace),
            "failed to set the variant pack unique ids.");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendFinalize(variantPackDesc->get()),
            "Failed to finalize variant pack descriptor");

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            detail::hipdnnBackend()->backendExecute(
                handle, _executionPlanDesc->get(), variantPackDesc->get()),
            "Execute failed.");

        return {ErrorCode::OK, ""};
    }

    const std::string& get_name() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_name();
    }

    DataType get_compute_data_type() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_compute_data_type();
    }
    DataType get_intermediate_data_type() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_intermediate_data_type();
    }
    DataType get_io_data_type() const // NOLINT(readability-identifier-naming)
    {
        return graph_attributes.get_io_data_type();
    }

    // NOLINTBEGIN(readability-identifier-naming)
    std::optional<int64_t> get_preferred_engine_id_ext() const
    // NOLINTEND(readability-identifier-naming)
    {
        return _preferredEngineId;
    }

    // Forwarding setters
    Graph& set_name(const std::string& name) // NOLINT(readability-identifier-naming)
    {
        graph_attributes.set_name(name);
        return *this;
    }
    Graph& set_compute_data_type(DataType computeType) // NOLINT(readability-identifier-naming)
    {
        graph_attributes.set_compute_data_type(computeType);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    Graph& set_intermediate_data_type(DataType intermediateType)
    {
        graph_attributes.set_intermediate_data_type(intermediateType);
        return *this;
    }
    Graph& set_io_data_type(DataType ioType) // NOLINT(readability-identifier-naming)
    {
        graph_attributes.set_io_data_type(ioType);
        return *this;
    }

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

    // NOLINTBEGIN(readability-identifier-naming)
    Graph& set_preferred_engine_id_ext(std::optional<int64_t> engineId)
    // NOLINTEND(readability-identifier-naming)
    {
        _preferredEngineId = engineId;
        return *this;
    }

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
     * @see tensor_like() for creating a tensor with cleared UID and custom name
     */
    static std::shared_ptr<TensorAttributes> tensor(const TensorAttributes& tensor)
    {
        auto newTensor = std::make_shared<TensorAttributes>(tensor);

        return newTensor;
    }
};

} // namespace hipdnn_frontend::graph
