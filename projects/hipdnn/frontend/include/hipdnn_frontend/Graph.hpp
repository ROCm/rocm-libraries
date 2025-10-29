// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/backend/BackendWrapper.hpp>
#include <hipdnn_frontend/backend/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/TopologicalSortingUtils.hpp>

namespace hipdnn_frontend
{
namespace graph
{
// When an error occurs, get the backend error string and append it to the error_message.
#define RETURN_ON_BACKEND_FAILURE(backend_status, error_message)                        \
    if((backend_status) != HIPDNN_STATUS_SUCCESS)                                       \
    {                                                                                   \
        std::array<char, 256> backend_err_msg{};                                        \
        hipdnn_frontend::hipdnnBackend()->getLastErrorString(backend_err_msg.data(),    \
                                                             backend_err_msg.size());   \
        std::string full_error_msg                                                      \
            = std::string(error_message) + " Backend error: " + backend_err_msg.data(); \
        return Error(ErrorCode::HIPDNN_BACKEND_ERROR, full_error_msg);                  \
    }

class Graph : public INode
{
private:
    std::unique_ptr<ScopedHipdnnBackendDescriptor> _graphDesc;
    std::unique_ptr<ScopedHipdnnBackendDescriptor> _engineHeuristicDesc;
    std::unique_ptr<ScopedHipdnnBackendDescriptor> _engineConfigDesc;
    std::unique_ptr<ScopedHipdnnBackendDescriptor> _executionPlanDesc;

    static std::shared_ptr<TensorAttributes> outputTensor(const std::string& name)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_name(name).set_is_virtual(true);
        return tensor;
    }

    Error initializeHeuristicDescriptor(std::vector<HeuristicMode> const& modes)
    {
        _engineHeuristicDesc
            = std::make_unique<ScopedHipdnnBackendDescriptor>(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(_engineHeuristicDesc->get(),
                                                 HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &_graphDesc->get()),
            "Failed to set operation graph on the engine heuristic descriptor.");

        // TODO
        // Currently we only handle the first mode in the vector.  Once we add heuristics we will need
        // to handle using all modes that are passed in.  We currently only have 1 mode so there
        // is only 1 possibility.
        std::vector<hipdnnBackendHeurMode_t> backendModes;
        backendModes.reserve(modes.size());
        for(const auto& mode : modes)
        {
            backendModes.push_back(toBackendType(mode));
        }

        RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendSetAttribute(_engineHeuristicDesc->get(),
                                                                       HIPDNN_ATTR_ENGINEHEUR_MODE,
                                                                       HIPDNN_TYPE_HEUR_MODE,
                                                                       1,
                                                                       backendModes.data()),
                                  "Failed to set mode on the engine heuristic descriptor.");

        RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(_engineHeuristicDesc->get()),
                                  "Failed to finalize engine heuristic descriptor");

        return {ErrorCode::OK, ""};
    }

    Error initializeEngineConfig()
    {
        int64_t availableEngineCount = 0;
        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendGetAttribute(_engineHeuristicDesc->get(),
                                                 HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 0,
                                                 &availableEngineCount,
                                                 nullptr),
            "Failed to get attribue from the engine heuristic descriptor.");

        if(availableEngineCount == 0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "No engine configurations available for the graph."};
        }

        int requiredCount = 1;
        std::vector<std::unique_ptr<ScopedHipdnnBackendDescriptor>> engineConfigs;
        std::vector<hipdnnBackendDescriptor_t> engineConfigsShallow;
        for(size_t i = 0; i < static_cast<size_t>(requiredCount); ++i)
        {
            auto engineCfgDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
                HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);

            if(engineCfgDesc == nullptr || !engineCfgDesc->valid())
            {
                return {ErrorCode::HIPDNN_BACKEND_ERROR,
                        "Failed to create engine configuration descriptor."};
            }
            engineConfigs.push_back(std::move(engineCfgDesc));
            engineConfigsShallow.push_back(engineConfigs.back()->get());
        }

        int64_t count = 0;
        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendGetAttribute(_engineHeuristicDesc->get(),
                                                 HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 static_cast<int64_t>(engineConfigsShallow.size()),
                                                 &count,
                                                 engineConfigsShallow.data()),
            "Failed to get engine configurations from the heuristic descriptor.");

        if(count == 0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "No engine configurations retrieved from the heuristic desc."};
        }

        //TODO
        // Add filtering and logic to select the best engine configuration that meets the requirements.
        _engineConfigDesc = std::move(engineConfigs[0]);
        engineConfigs.erase(engineConfigs.begin(), engineConfigs.begin() + 1);

        return {ErrorCode::OK, ""};
    }

    GraphStructure buildAdjacencyList(const std::unordered_map<std::shared_ptr<TensorAttributes>,
                                                               size_t>& tensorToOriginNode) const
    {
        size_t nodeCount = _sub_nodes.size();
        GraphStructure structure;
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

public:
    Graph()
        : INode(GraphAttributes{})
    {
        HIPDNN_FE_LOG_INFO("Creating new Graph instance");
    }

    Error validate()
    {
        HIPDNN_FE_LOG_INFO("Validating graph {}", graph_attributes.get_name());

        auto result = checkNoDuplicateTensorIds();
        if(result.code != ErrorCode::OK)
        {
            return result;
        }

        result = topologicallySortGraph();
        if(result.code != ErrorCode::OK)
        {
            return result;
        }

        return validateSubtree();
    }

    Error checkNoDuplicateTensorIds()
    {
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

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

    Error topologicallySortGraph()
    {
        size_t nodeCount = _sub_nodes.size();

        if(nodeCount == 0)
        {
            return {ErrorCode::OK, ""};
        }

        auto tensorToOriginNode = buildTensorToOriginNodeMap();
        auto graphStructure = buildAdjacencyList(tensorToOriginNode);

        auto sortResult = performTopologicalSortWithComponentDetection(graphStructure);

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
        std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
        gatherHipdnnTensorsSubtree(allTensors);

        auto usedIds = getUsedIds(allTensors);

        populateHipdnnTensorIds(allTensors, usedIds);

        flatbuffers::FlatBufferBuilder builder;

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensorAttributes;
        for(auto& tensor : allTensors)
        {
            if(tensor)
            {
                tensorAttributes.emplace_back(tensor->pack_attributes(builder));
            }
        }

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        for(auto& node : _sub_nodes)
        {
            if(node)
            {
                nodes.emplace_back(node->pack_node(builder));
            }
        }
        auto graph = hipdnn_sdk::data_objects::CreateGraphDirect(
            builder,
            graph_attributes.get_name().c_str(),
            toSdkType(graph_attributes.get_compute_data_type()),
            toSdkType(graph_attributes.get_intermediate_data_type()),
            toSdkType(graph_attributes.get_io_data_type()),
            &tensorAttributes,
            &nodes);

        builder.Finish(graph);
        return builder.Release();
    }

    Error build_operation_graph(hipdnnHandle_t handle) // NOLINT(readability-identifier-naming)
    {
        HIPDNN_FE_LOG_INFO("Building operation graph {}", graph_attributes.get_name());

        auto serializedGraph = buildFlatbufferOperationGraph();
        _graphDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(serializedGraph.data(),
                                                                     serializedGraph.size());

        if(!_graphDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create backend graph descriptor for the graph."};
        }

        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(_graphDesc->get(),
                                                 HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                                 HIPDNN_TYPE_HANDLE,
                                                 1,
                                                 &handle),
            "Failed to set handle on the graph.");

        RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(_graphDesc->get()),
                                  "Failed to finalize backend descriptor for the graph");

        return {ErrorCode::OK, ""};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_execution_plans(std::vector<HeuristicMode> const& modes
                                 = {HeuristicMode::FALLBACK})
    {
        HIPDNN_FE_LOG_INFO("Creating execution plans for graph {}", graph_attributes.get_name());

        if(!_graphDesc || !_graphDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Graph has not been built, build the operation graph first. Cannot create "
                    "execution plan."};
        }

        Error status = initializeHeuristicDescriptor(modes);
        HIPDNN_CHECK_ERROR(status);

        status = initializeEngineConfig();
        HIPDNN_CHECK_ERROR(status);

        _executionPlanDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
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
        HIPDNN_FE_LOG_INFO("Checking execution plan support for graph {}",
                           graph_attributes.get_name());

        if(!_executionPlanDesc || !_executionPlanDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Execution plan descriptor is not created or invalid."};
        }

        return {ErrorCode::OK, ""};
    }

    Error build_plans() // NOLINT(readability-identifier-naming)
    {
        HIPDNN_FE_LOG_INFO("Building plans for graph {}", graph_attributes.get_name());

        RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(_engineConfigDesc->get()),
                                  "Failed to finalize engine config descriptor");

        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(_executionPlanDesc->get(),
                                                 HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &_engineConfigDesc->get()),
            "Failed to set the engine config on execution plan.");

        RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(_executionPlanDesc->get()),
                                  "Failed to finalize execution plan descriptor");

        return {ErrorCode::OK, ""};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error get_workspace_size(int64_t& workspaceSize) const
    {
        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendGetAttribute(_executionPlanDesc->get(),
                                                 HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                 HIPDNN_TYPE_INT64,
                                                 1,
                                                 nullptr,
                                                 &workspaceSize),
            "Failed to get workspace size from the execution plan descriptor.");

        return {ErrorCode::OK, ""};
    }

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

    Error execute(hipdnnHandle_t handle,
                  std::unordered_map<int64_t, void*>& variantPack,
                  void* workspace) const
    {
        HIPDNN_FE_LOG_INFO("Executing graph {}", graph_attributes.get_name());

        auto variantPackDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
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

        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(variantPackDesc->get(),
                                                 HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                 HIPDNN_TYPE_VOID_PTR,
                                                 static_cast<int64_t>(variantPackValues.size()),
                                                 variantPackValues.data()),
            "failed to set the variant pack data pointers.");

        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(variantPackDesc->get(),
                                                 HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                                 HIPDNN_TYPE_INT64,
                                                 static_cast<int64_t>(variantPackKeys.size()),
                                                 variantPackKeys.data()),
            "failed to set the variant pack unique ids.");

        RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(variantPackDesc->get(),
                                                 HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                 HIPDNN_TYPE_VOID_PTR,
                                                 1,
                                                 &workspace),
            "failed to set the variant pack unique ids.");

        RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(variantPackDesc->get()),
                                  "Failed to finalize variant pack descriptor");

        RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendExecute(
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
    static std::shared_ptr<TensorAttributes>
        tensor_like(const std::shared_ptr<TensorAttributes>& tensor, const std::string& name = "")
    // NOLINTEND(readability-identifier-naming)
    {
        auto newTensor = std::make_shared<TensorAttributes>(*tensor);

        newTensor->clear_uid();
        newTensor->set_name(name);

        return newTensor;
    }

    static std::shared_ptr<TensorAttributes> tensor(const TensorAttributes& tensor)
    {
        auto newTensor = std::make_shared<TensorAttributes>(tensor);

        return newTensor;
    }
};
}
}
