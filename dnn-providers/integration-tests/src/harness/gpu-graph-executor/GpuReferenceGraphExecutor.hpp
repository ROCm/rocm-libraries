// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaDensify.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaFwdSupport.hpp>

#include "detail/GpuPlanBuilderRegistry.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/ReferenceCapabilityError.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor
{

class GpuReferenceGraphExecutor : public IReferenceGraphExecutor
{
public:
    GpuReferenceGraphExecutor() = default;

    // A ragged SDPA graph no plan accepts as is runs as its dense padded equivalent
    // (RaggedSdpaDensify.hpp), as in the CPU executor.
    bool isApplicable(void* graphBuffer, size_t size) override
    {
        if(isApplicableAsIs(graphBuffer, size))
        {
            return true;
        }
        auto dense = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(graphBuffer, size);
        return dense.has_value() && isApplicableAsIs(dense->graph.data(), dense->graph.size());
    }

    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack) override
    {
        if(!isApplicableAsIs(graphBuffer, size))
        {
            if(auto dense = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(graphBuffer, size);
               dense.has_value())
            {
                executeRaggedAsDense(*dense, variantPack);
                return;
            }
        }
        executeAsIs(graphBuffer, size, variantPack);
    }

    bool requiresDeviceMemory() const override
    {
        return true;
    }

private:
    // Owns one device allocation; frees it after the device is idle.
    struct DeviceBuffer
    {
        void* ptr = nullptr;
        explicit DeviceBuffer(size_t bytes)
        {
            hipdnn_gpu_ref::detail::throwOnHipError(hipMalloc(&ptr, bytes == 0 ? 1 : bytes),
                                                    "ragged SDPA densify: hipMalloc");
        }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        ~DeviceBuffer()
        {
            (void)hipDeviceSynchronize();
            (void)hipFree(ptr);
        }
    };

    static void copy(void* to, const void* from, size_t bytes, hipMemcpyKind kind)
    {
        if(bytes > 0)
        {
            hipdnn_gpu_ref::detail::throwOnHipError(hipMemcpy(to, from, bytes, kind),
                                                    "ragged SDPA densify: hipMemcpy");
        }
    }

    // The CPU executor's ragged-as-dense run with device buffers: offsets and packed inputs are
    // read back, unpacked on the host and uploaded; dense outputs are read back, packed into a
    // host copy of the packed output (so bytes outside live tokens are kept) and uploaded.
    void executeRaggedAsDense(hipdnn_test_sdk::detail::DenseSdpaFromRagged& dense,
                              const std::unordered_map<int64_t, void*>& variantPack)
    {
        namespace rd = hipdnn_test_sdk::detail;
        const auto& primaries = dense.primaries;
        std::vector<std::vector<int64_t>> offsets;
        std::vector<std::vector<int64_t>> lengths;
        for(const auto& primary : primaries)
        {
            std::vector<uint8_t> table(static_cast<size_t>(dense.batch + 1)
                                       * primary.offsetElementSize);
            copy(table.data(),
                 variantPack.at(primary.offsetUid),
                 table.size(),
                 hipMemcpyDeviceToHost);
            offsets.push_back(rd::raggedElementOffsets(primary, table.data(), dense.batch));
            lengths.push_back(rd::raggedLengths(primary, offsets.back()));
        }
        const auto [qLengths, kvLengths] = rd::raggedGroupLengths(dense, lengths);

        auto densePack = variantPack;
        std::vector<std::unique_ptr<DeviceBuffer>> deviceBuffers;
        std::vector<std::vector<uint8_t>> hostDense(primaries.size());
        for(size_t i = 0; i < primaries.size(); ++i)
        {
            hostDense[i].assign(rd::raggedDenseBytes(primaries[i]), 0);
            if(!primaries[i].output)
            {
                std::vector<uint8_t> packed(
                    rd::raggedPackedBytes(primaries[i], offsets[i], lengths[i]));
                copy(packed.data(),
                     variantPack.at(primaries[i].uid),
                     packed.size(),
                     hipMemcpyDeviceToHost);
                rd::copyRaggedTokens(
                    primaries[i], offsets[i], lengths[i], packed.data(), hostDense[i].data(), true);
            }
            deviceBuffers.push_back(std::make_unique<DeviceBuffer>(hostDense[i].size()));
            copy(deviceBuffers.back()->ptr,
                 hostDense[i].data(),
                 hostDense[i].size(),
                 hipMemcpyHostToDevice);
            densePack[primaries[i].uid] = deviceBuffers.back()->ptr;
        }
        const auto bindSeqLens = [&](std::optional<int64_t> added,
                                     int64_t uid,
                                     const std::vector<int64_t>& packed,
                                     const char* name) {
            if(packed.empty())
            {
                return;
            }
            std::vector<int32_t> values(packed.size());
            if(added.has_value())
            {
                values.assign(packed.begin(), packed.end());
                deviceBuffers.push_back(
                    std::make_unique<DeviceBuffer>(values.size() * sizeof(int32_t)));
                copy(deviceBuffers.back()->ptr,
                     values.data(),
                     values.size() * sizeof(int32_t),
                     hipMemcpyHostToDevice);
                densePack[uid] = deviceBuffers.back()->ptr;
            }
            else
            {
                copy(values.data(),
                     variantPack.at(uid),
                     values.size() * sizeof(int32_t),
                     hipMemcpyDeviceToHost);
                rd::requireSeqLensWithinPacked(values.data(), packed, name);
            }
        };
        bindSeqLens(dense.addedSeqLenQUid, dense.seqLenQUid, qLengths, "seq_len_q");
        bindSeqLens(dense.addedSeqLenKvUid, dense.seqLenKvUid, kvLengths, "seq_len_kv");

        executeAsIs(dense.graph.data(), dense.graph.size(), densePack);

        for(size_t i = 0; i < primaries.size(); ++i)
        {
            if(!primaries[i].output)
            {
                continue;
            }
            copy(hostDense[i].data(),
                 densePack.at(primaries[i].uid),
                 hostDense[i].size(),
                 hipMemcpyDeviceToHost);
            std::vector<uint8_t> packed(
                rd::raggedPackedBytes(primaries[i], offsets[i], lengths[i]));
            copy(packed.data(),
                 variantPack.at(primaries[i].uid),
                 packed.size(),
                 hipMemcpyDeviceToHost);
            rd::copyRaggedTokens(
                primaries[i], offsets[i], lengths[i], packed.data(), hostDense[i].data(), false);
            copy(variantPack.at(primaries[i].uid),
                 packed.data(),
                 packed.size(),
                 hipMemcpyHostToDevice);
        }
    }

    bool isApplicableAsIs(void* graphBuffer, size_t size)
    {
        auto graphWrap
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                graphBuffer, size);

        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            if(!isNodeApplicable(node, graphWrap.getTensorMap()))
            {
                return false;
            }
        }
        return true;
    }

    void executeAsIs(void* graphBuffer,
                     size_t size,
                     const std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                graphBuffer, size);

        std::vector<std::unique_ptr<detail::IGpuGraphNodePlanExecutor>> planExecutors;

        // The graph in graphBuffer is guaranteed to be topologically sorted.
        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            planExecutors.push_back(buildPlanForNode(graphWrap, node));
        }

        std::vector<std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>> virtualTensors;
        const auto variantPackWithVirtualTensors = populateVariantPackWithMissingVirtualTensors(
            variantPack, graphWrap.getTensorMap(), virtualTensors);

        for(auto& executor : planExecutors)
        {
            executor->execute(variantPackWithVirtualTensors);
        }
    }

    static std::unordered_map<int64_t, void*> populateVariantPackWithMissingVirtualTensors(
        const std::unordered_map<int64_t, void*>& variantPack,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        std::vector<std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>>& virtualTensors)
    {
        auto updatedVariantPack = variantPack;

        for(const auto& [id, attr] : tensorMap)
        {
            if(attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end())
            {
                auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(*attr);
                virtualTensors.push_back(std::move(tensor));

                updatedVariantPack[id] = virtualTensors.back()->rawDeviceData();
            }
        }
        return updatedVariantPack;
    }

    bool isNodeApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        try
        {
            auto key = buildSignatureKey(node, tensorMap);
            const auto& builder = _planRegistry.getPlanBuilder(key);
            return builder.isApplicable(node, tensorMap);
        }
        catch(...)
        {
            return false;
        }
    }

    std::unique_ptr<detail::IGpuGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                         const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        auto key = buildSignatureKey(node, graph.getTensorMap());

        const auto& planBuilder = _planRegistry.getPlanBuilder(key);
        if(!planBuilder.isApplicable(node, graph.getTensorMap()))
        {
            const std::string nodeName
                = node.name() == nullptr ? " unknown" : " " + node.name()->str();
            throw ReferenceCapabilityError("GPU plan builder is not applicable for the given node:"
                                           + nodeName);
        }

        return planBuilder.buildNodePlan(graph, node);
    }

    static detail::GpuPlanRegistrySignatureKey buildSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        using NodeAttrs = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

        switch(node.attributes_type())
        {
        case NodeAttrs::BatchnormAttributes:
            return detail::GpuBatchnormFwdTrainSignatureKey(
                node, tensorMap, node.compute_data_type());
        case NodeAttrs::ConvolutionFwdAttributes:
            return detail::GpuConvolutionFwdSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::LayernormAttributes:
            return detail::GpuLayernormFwdSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::LayernormBackwardAttributes:
            return detail::GpuLayernormBwdSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::MatmulAttributes:
            return detail::GpuMatmulSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::PointwiseAttributes:
            return detail::GpuPointwiseSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::ReductionAttributes:
            return detail::GpuReductionSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::ResampleFwdAttributes:
            return detail::GpuResampleFwdSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::ResampleBwdAttributes:
            return detail::GpuResampleBwdSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::RMSNormAttributes:
            return detail::GpuRMSNormFwdSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::RMSNormBackwardAttributes:
            return detail::GpuRMSNormBwdSignatureKey(node, tensorMap, node.compute_data_type());
        case NodeAttrs::BatchnormInferenceAttributes:
            return detail::GpuBatchnormFwdInfSignatureKey(
                node, tensorMap, node.compute_data_type());
        case NodeAttrs::BatchnormInferenceAttributesVarianceExt:
            return detail::GpuBatchnormFwdInfVarianceSignatureKey(
                node, tensorMap, node.compute_data_type());
        case NodeAttrs::SdpaAttributes:
        {
            // A node with any ragged primary is keyed as ragged. A partly ragged node is declined
            // by both plans (the ragged plan needs all four primaries ragged, and the dense plan
            // rejects any ragged tensor); isApplicable then tries the dense rewrite.
            const auto* sdpaAttributes = node.attributes_as_SdpaAttributes();
            if(sdpaAttributes != nullptr
               && hipdnn_test_sdk::detail::hasRaggedSdpaPrimary(*sdpaAttributes, tensorMap))
            {
                return detail::GpuSdpaRaggedFwdSignatureKey(node, tensorMap);
            }
            return detail::GpuSdpaFwdSignatureKey(node, tensorMap);
        }
        case NodeAttrs::SdpaBackwardAttributes:
            return detail::GpuSdpaBwdSignatureKey(node, tensorMap);

        // Node types with no GPU plan yet - throw descriptive error
        case NodeAttrs::BatchnormBackwardAttributes:
        case NodeAttrs::ConvolutionBwdAttributes:
        case NodeAttrs::ConvolutionWrwAttributes:
        case NodeAttrs::BlockScaleDequantizeAttributes:
        case NodeAttrs::BlockScaleQuantizeAttributes:
        {
            const std::string nodeName = node.name() == nullptr ? "unknown" : node.name()->str();
            throw ReferenceCapabilityError("GPU plan not yet implemented for node '" + nodeName
                                           + "'. Register a GPU plan for this operation type.");
        }

        case NodeAttrs::CustomOpAttributes:
            throw ReferenceCapabilityError(
                "GPU reference executor does not support custom operations");

        default:
            throw ReferenceCapabilityError(
                "Unsupported node type for GPU signature key generation");
        }
    }

    detail::GpuPlanBuilderRegistry _planRegistry;
};

} // namespace hipdnn_integration_tests::gpu_graph_executor
