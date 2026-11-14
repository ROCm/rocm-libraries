// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaDensify.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaFwdSupport.hpp>

#include "harness/ReferenceCapabilityError.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor
{

namespace
{

std::unordered_map<int64_t, void*> populateVariantPackWithMissingVirtualTensors(
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

// Owns one device allocation; frees it after the device is idle.
struct DeviceBuffer
{
    void* ptr = nullptr;
    // Graph tensors have positive dimensions and empty sequence-length lists are not
    // bound, so a zero-byte buffer is a caller bug.
    explicit DeviceBuffer(size_t bytes)
    {
        if(bytes == 0)
        {
            throw std::invalid_argument("ragged SDPA densify: empty device buffer");
        }
        hipdnn_gpu_ref::detail::throwOnHipError(hipMalloc(&ptr, bytes),
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

void copyBytes(void* to, const void* from, size_t bytes, hipMemcpyKind kind)
{
    if(bytes > 0)
    {
        hipdnn_gpu_ref::detail::throwOnHipError(hipMemcpy(to, from, bytes, kind),
                                                "ragged SDPA densify: hipMemcpy");
    }
}

detail::GpuPlanRegistrySignatureKey buildSignatureKey(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using NodeAttrs = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NodeAttrs::BatchnormBackwardAttributes:
        return detail::GpuBatchnormBwdSignatureKey(node, tensorMap);
    case NodeAttrs::BatchnormAttributes:
        return detail::GpuBatchnormFwdTrainSignatureKey(node, tensorMap, node.compute_data_type());
    case NodeAttrs::BlockScaleQuantizeAttributes:
        return detail::GpuBlockScaleQuantizeSignatureKey(node, tensorMap);
    case NodeAttrs::BlockScaleDequantizeAttributes:
        return detail::GpuBlockScaleDequantizeSignatureKey(node, tensorMap);
    case NodeAttrs::ConvolutionFwdAttributes:
        return detail::GpuConvolutionFwdSignatureKey(node, tensorMap, node.compute_data_type());
    case NodeAttrs::ConvolutionBwdAttributes:
        return detail::GpuConvolutionBwdSignatureKey(node, tensorMap, node.compute_data_type());
    case NodeAttrs::ConvolutionWrwAttributes:
        return detail::GpuConvolutionWrwSignatureKey(node, tensorMap, node.compute_data_type());
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
        return detail::GpuBatchnormFwdInfSignatureKey(node, tensorMap, node.compute_data_type());
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

    case NodeAttrs::CustomOpAttributes:
        throw ReferenceCapabilityError("GPU reference executor does not support custom operations");

    default:
        throw ReferenceCapabilityError("Unsupported node type for GPU signature key generation");
    }
}

} // namespace

struct GpuReferenceGraphExecutor::Impl
{
    bool isNodeApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        try
        {
            auto key = buildSignatureKey(node, tensorMap);
            const auto& builder = planRegistry.getPlanBuilder(key);
            return builder.isApplicable(node, tensorMap);
        }
        catch(...)
        {
            return false;
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

    // A ragged SDPA graph no plan accepts as is runs as its dense padded equivalent
    // (RaggedSdpaDensify.hpp), as in the CPU executor.
    bool isApplicable(void* graphBuffer, size_t size)
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
                 const std::unordered_map<int64_t, void*>& variantPack)
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

    // The CPU executor's ragged-as-dense run with device buffers: offsets, the graph's own
    // seq_len tensors and packed inputs are read back, unpacked on the host and uploaded; dense
    // outputs are read back, packed into a host copy of the packed output (so bytes outside live
    // tokens are kept) and uploaded. Live lengths come from seq_len where the graph has it, else
    // from the offset blocks.
    void executeRaggedAsDense(hipdnn_test_sdk::detail::DenseSdpaFromRagged& dense,
                              const std::unordered_map<int64_t, void*>& variantPack)
    {
        namespace rd = hipdnn_test_sdk::detail;
        const auto& primaries = dense.primaries;
        // Host copy of the graph's own seq_len tensors (INT32 [B, 1, 1, 1], element b at b * stride); empty
        // where the rewrite adds them.
        const auto readOwnSeqLen = [&](const std::optional<int64_t>& added,
                                       int64_t uid,
                                       int64_t stride) {
            std::vector<int32_t> raw;
            if(!added.has_value())
            {
                raw.resize(rd::stridedAuxSpan(dense.batch, stride, "ragged SDPA densify: seq_len"));
                copyBytes(raw.data(),
                          variantPack.at(uid),
                          raw.size() * sizeof(int32_t),
                          hipMemcpyDeviceToHost);
            }
            return raw;
        };
        const auto ownSeqLenQ
            = readOwnSeqLen(dense.addedSeqLenQUid, dense.seqLenQUid, dense.seqLenQStride);
        const auto ownSeqLenKv
            = readOwnSeqLen(dense.addedSeqLenKvUid, dense.seqLenKvUid, dense.seqLenKvStride);
        std::vector<std::vector<int64_t>> offsets;
        std::vector<std::vector<int64_t>> lengths;
        for(const auto& primary : primaries)
        {
            std::vector<uint8_t> table(rd::raggedOffsetBytes(primary, dense.batch));
            copyBytes(table.data(),
                      variantPack.at(primary.offsetUid),
                      table.size(),
                      hipMemcpyDeviceToHost);
            offsets.push_back(rd::raggedElementOffsets(primary, table.data(), dense.batch));
            const auto& seqLen = primary.queryPacking ? ownSeqLenQ : ownSeqLenKv;
            lengths.push_back(rd::raggedLengths(primary,
                                                offsets.back(),
                                                seqLen.empty() ? nullptr : seqLen.data(),
                                                primary.queryPacking ? dense.seqLenQStride
                                                                     : dense.seqLenKvStride));
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
                copyBytes(packed.data(),
                          variantPack.at(primaries[i].uid),
                          packed.size(),
                          hipMemcpyDeviceToHost);
                rd::copyRaggedTokens(
                    primaries[i], offsets[i], lengths[i], packed.data(), hostDense[i].data(), true);
            }
            deviceBuffers.push_back(std::make_unique<DeviceBuffer>(hostDense[i].size()));
            copyBytes(deviceBuffers.back()->ptr,
                      hostDense[i].data(),
                      hostDense[i].size(),
                      hipMemcpyHostToDevice);
            densePack[primaries[i].uid] = deviceBuffers.back()->ptr;
        }
        // A seq_len tensor the rewrite added carries the offset-derived lengths.
        const auto bindAddedSeqLens
            = [&](const std::optional<int64_t>& added, const std::vector<int64_t>& packed) {
                  if(!added.has_value() || packed.empty())
                  {
                      return;
                  }
                  const std::vector<int32_t> values(packed.begin(), packed.end());
                  deviceBuffers.push_back(
                      std::make_unique<DeviceBuffer>(values.size() * sizeof(int32_t)));
                  copyBytes(deviceBuffers.back()->ptr,
                            values.data(),
                            values.size() * sizeof(int32_t),
                            hipMemcpyHostToDevice);
                  densePack[*added] = deviceBuffers.back()->ptr;
              };
        bindAddedSeqLens(dense.addedSeqLenQUid, qLengths);
        bindAddedSeqLens(dense.addedSeqLenKvUid, kvLengths);

        executeAsIs(dense.graph.data(), dense.graph.size(), densePack);

        for(size_t i = 0; i < primaries.size(); ++i)
        {
            if(!primaries[i].output)
            {
                continue;
            }
            copyBytes(hostDense[i].data(),
                      densePack.at(primaries[i].uid),
                      hostDense[i].size(),
                      hipMemcpyDeviceToHost);
            std::vector<uint8_t> packed(
                rd::raggedPackedBytes(primaries[i], offsets[i], lengths[i]));
            copyBytes(packed.data(),
                      variantPack.at(primaries[i].uid),
                      packed.size(),
                      hipMemcpyDeviceToHost);
            rd::copyRaggedTokens(
                primaries[i], offsets[i], lengths[i], packed.data(), hostDense[i].data(), false);
            copyBytes(variantPack.at(primaries[i].uid),
                      packed.data(),
                      packed.size(),
                      hipMemcpyHostToDevice);
        }
    }

    std::unique_ptr<detail::IGpuGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                         const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        auto key = buildSignatureKey(node, graph.getTensorMap());

        const auto& planBuilder = planRegistry.getPlanBuilder(key);
        if(!planBuilder.isApplicable(node, graph.getTensorMap()))
        {
            const std::string nodeName
                = node.name() == nullptr ? " unknown" : " " + node.name()->str();
            throw ReferenceCapabilityError("GPU plan builder is not applicable for the given node:"
                                           + nodeName);
        }

        return planBuilder.buildNodePlan(graph, node);
    }

    detail::GpuPlanBuilderRegistry planRegistry;
};

GpuReferenceGraphExecutor::GpuReferenceGraphExecutor()
    : _impl(std::make_unique<Impl>())
{
}

GpuReferenceGraphExecutor::~GpuReferenceGraphExecutor() = default;

bool GpuReferenceGraphExecutor::isApplicable(void* graphBuffer, size_t size)
{
    return _impl->isApplicable(graphBuffer, size);
}

void GpuReferenceGraphExecutor::execute(void* graphBuffer,
                                        size_t size,
                                        const std::unordered_map<int64_t, void*>& variantPack)
{
    _impl->execute(graphBuffer, size, variantPack);
}

} // namespace hipdnn_integration_tests::gpu_graph_executor
