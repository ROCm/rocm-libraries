// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstring>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceNotApplicableError.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizeSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleQuantizeSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionBwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionFwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionWrwPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormBpropSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MatmulPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanBuilderRegistry.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwisePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaRaggedFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaDensify.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaFwdSupport.hpp>

namespace hipdnn_test_sdk::utilities
{

class CpuReferenceGraphExecutor
{

public:
    CpuReferenceGraphExecutor() = default;

    // A ragged SDPA graph no plan accepts as is runs as its dense padded equivalent
    // (RaggedSdpaDensify.hpp).
    bool isApplicable(void* graphBuffer, size_t size)
    {
        if(isApplicableAsIs(graphBuffer, size))
        {
            return true;
        }
        auto dense = detail::densifyRaggedSdpaGraph(graphBuffer, size);
        return dense.has_value() && isApplicableAsIs(dense->graph.data(), dense->graph.size());
    }

    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack)
    {
        if(!isApplicableAsIs(graphBuffer, size))
        {
            if(auto dense = detail::densifyRaggedSdpaGraph(graphBuffer, size); dense.has_value())
            {
                executeRaggedAsDense(*dense, variantPack);
                return;
            }
        }
        executeAsIs(graphBuffer, size, variantPack);
    }

private:
    bool isApplicableAsIs(void* graphBuffer, size_t size)
    {
        auto graphWrap
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                graphBuffer, size);

        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            if(!isNodeApplicable(node, graphWrap.getTensorMap(), node.compute_data_type()))
            {
                return false;
            }
        }
        return true;
    }

    // Unpacks the packed buffers into dense ones, runs the dense graph, and packs the outputs back.
    void executeRaggedAsDense(detail::DenseSdpaFromRagged& dense,
                              const std::unordered_map<int64_t, void*>& variantPack)
    {
        const auto& primaries = dense.primaries;
        std::vector<std::vector<int64_t>> offsets;
        std::vector<std::vector<int64_t>> lengths;
        for(const auto& primary : primaries)
        {
            offsets.push_back(detail::raggedElementOffsets(
                primary, variantPack.at(primary.offsetUid), dense.batch));
            lengths.push_back(detail::raggedLengths(primary, offsets.back()));
        }
        const auto [qLengths, kvLengths] = detail::raggedGroupLengths(dense, lengths);

        auto densePack = variantPack;
        std::vector<std::vector<uint8_t>> buffers(primaries.size());
        for(size_t i = 0; i < primaries.size(); ++i)
        {
            buffers[i].assign(detail::raggedDenseBytes(primaries[i]), 0);
            if(!primaries[i].output)
            {
                detail::copyRaggedTokens(primaries[i],
                                         offsets[i],
                                         lengths[i],
                                         static_cast<uint8_t*>(variantPack.at(primaries[i].uid)),
                                         buffers[i].data(),
                                         true);
            }
            densePack[primaries[i].uid] = buffers[i].data();
        }
        const auto bindSeqLens = [&](std::optional<int64_t> added,
                                     int64_t uid,
                                     const std::vector<int64_t>& packed,
                                     std::vector<int32_t>& storage,
                                     const char* name) {
            if(added.has_value())
            {
                storage.assign(packed.begin(), packed.end());
                densePack[uid] = storage.data();
            }
            else if(!packed.empty())
            {
                detail::requireSeqLensWithinPacked(
                    static_cast<const int32_t*>(variantPack.at(uid)), packed, name);
            }
        };
        std::vector<int32_t> seqLenQ;
        std::vector<int32_t> seqLenKv;
        bindSeqLens(dense.addedSeqLenQUid, dense.seqLenQUid, qLengths, seqLenQ, "seq_len_q");
        bindSeqLens(dense.addedSeqLenKvUid, dense.seqLenKvUid, kvLengths, seqLenKv, "seq_len_kv");

        executeAsIs(dense.graph.data(), dense.graph.size(), densePack);

        for(size_t i = 0; i < primaries.size(); ++i)
        {
            if(primaries[i].output)
            {
                detail::copyRaggedTokens(primaries[i],
                                         offsets[i],
                                         lengths[i],
                                         static_cast<uint8_t*>(variantPack.at(primaries[i].uid)),
                                         buffers[i].data(),
                                         false);
            }
        }
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void executeAsIs(void* graphBuffer,
                     size_t size,
                     const std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                graphBuffer, size);

        std::vector<std::unique_ptr<detail::IGraphNodePlanExecutor>> planExecutors;

        //The graph in graphBuffer is guaranteed to be topologically sorted.
        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            planExecutors.push_back(buildPlanForNode(graphWrap, node));
        }

        std::vector<std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>> virtualTensors;
        const std::unordered_map<int64_t, void*> variantPackWithVirtualTensorsAdded
            = populateVariantPackWithMissingVirtualTensors(
                variantPack, graphWrap.getTensorMap(), virtualTensors);

        for(auto& executor : planExecutors)
        {
            executor->execute(variantPackWithVirtualTensorsAdded);
        }
    }

    static std::unordered_map<int64_t, void*> populateVariantPackWithMissingVirtualTensors(
        const std::unordered_map<int64_t, void*>& variantPack,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        std::vector<std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>>& virtualTensors)
    {
        std::unordered_map<int64_t, void*> updatedVariantPack = variantPack;

        for(const auto& [id, attr] : tensorMap)
        {
            if(attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end())
            {
                auto tensor = detail::createTensorFromAttribute(*attr);
                tensor->fillWithSentinelValue();
                virtualTensors.push_back(std::move(tensor));
                updatedVariantPack[id] = virtualTensors.back()->rawHostData();
            }
        }
        return updatedVariantPack;
    }

    bool isNodeApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        try
        {
            auto key = buildSignatureKey(node, tensorMap, computeType);
            const auto& builder = _planRegistry.getPlanBuilder(key);
            return builder.isApplicable(node, tensorMap);
        }
        catch(...)
        {
            return false;
        }
    }

    std::unique_ptr<detail::IGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                         const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        auto key = buildSignatureKey(node, graph.getTensorMap(), node.compute_data_type());

        const auto& planBuilder = _planRegistry.getPlanBuilder(key);
        if(!planBuilder.isApplicable(node, graph.getTensorMap()))
        {
            const std::string nodeName = node.name() == nullptr ? "" : " " + node.name()->str();
            throw CpuReferenceNotApplicableError(
                "Plan builder is not applicable for the given node: " + nodeName);
        }

        return planBuilder.buildNodePlan(graph, node);
    }

    static detail::PlanRegistrySignatureKey buildSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        switch(node.attributes_type())
        {
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return detail::BatchnormFwdInferenceSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt:
            return detail::BatchnormFwdInferenceWithVarianceSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes:
            return detail::PointwiseSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return detail::BatchnormBwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            return detail::BatchnormTrainSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
            return detail::ConvolutionFwdSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
            return detail::ConvolutionBwdSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
            return detail::ConvolutionWrwSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes:
            return detail::LayernormFpropSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes:
            return detail::LayernormBpropSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes:
            return detail::MatmulSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MoeGroupedMatmulAttributes:
            return detail::MoeGroupedMatmulSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MoeGroupedMatmulBwdAttributes:
            return detail::MoeGroupedMatmulBwdSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes:
            return detail::RMSNormFwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes:
            return detail::RMSNormBwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes:
            return detail::ResampleFwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes:
            return detail::ResampleBwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes:
            return detail::BlockScaleDequantizeSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleQuantizeAttributes:
            return detail::BlockScaleQuantizeSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes:
        {
            // A node with any ragged primary is keyed as ragged. A partly ragged node is declined
            // either way: the ragged plan needs all four primaries ragged, and the dense plan
            // rejects any ragged tensor.
            const auto* sdpaAttributes = node.attributes_as_SdpaAttributes();
            if(sdpaAttributes != nullptr
               && detail::hasRaggedSdpaPrimary(*sdpaAttributes, tensorMap))
            {
                return detail::SdpaRaggedFwdSignatureKey(node, tensorMap);
            }
            return detail::SdpaFwdSignatureKey(node, tensorMap);
        }
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes:
            return detail::SdpaBwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ReductionAttributes:
            return detail::ReductionSignatureKey(node, tensorMap, computeType);
        default:
            throw CpuReferenceNotApplicableError(
                "Unsupported node type for signature key generation");
        }
    }

    detail::PlanBuilderRegistry _planRegistry;
};
}
