// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/ReferenceOpCoverage.hpp"

#include <algorithm>
#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipdnn_integration_tests::bundle
{

namespace
{

// Ops the CPU reference is required to handle.
//
// Keep this list honest: an entry here means bundles using that op are registered
// for CPU validation and will turn red if the reference cannot run them. Do not add
// an op speculatively.
const std::set<NodeAttributes>& cpuSupportedOps()
{
    static const std::set<NodeAttributes> s_ops = {
        NodeAttributes::BatchnormInferenceAttributes,
        NodeAttributes::BatchnormInferenceAttributesVarianceExt,
        NodeAttributes::BatchnormAttributes,
        NodeAttributes::BatchnormBackwardAttributes,
        NodeAttributes::LayernormAttributes,
        NodeAttributes::LayernormBackwardAttributes,
        NodeAttributes::RMSNormAttributes,
        NodeAttributes::RMSNormBackwardAttributes,
        NodeAttributes::PointwiseAttributes,
        NodeAttributes::SdpaAttributes,
    };
    return s_ops;
}

// Ops the GPU reference is required to handle. Narrower than the CPU set: the GPU
// reference dispatches through a signature-keyed plan registry, so coverage is
// per-op-shape and grows only as plan builders are written.
const std::set<NodeAttributes>& gpuSupportedOps()
{
    static const std::set<NodeAttributes> s_ops = {
        NodeAttributes::ConvolutionFwdAttributes,
        NodeAttributes::SdpaAttributes,
    };
    return s_ops;
}

} // namespace

const std::set<NodeAttributes>& referenceSupportedOps(ReferenceExecutorType type)
{
    switch(type)
    {
    case ReferenceExecutorType::CPU:
        return cpuSupportedOps();
    case ReferenceExecutorType::GPU:
        return gpuSupportedOps();
    default:
        throw std::runtime_error("Unknown reference executor type");
    }
}

std::optional<std::set<NodeAttributes>> graphNodeTypes(const void* graphBuffer, size_t size)
{
    std::set<NodeAttributes> types;
    try
    {
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
            graphBuffer, size);
        for(uint32_t i = 0; i < graph.nodeCount(); ++i)
        {
            types.insert(graph.getNode(i).attributes_type());
        }
    }
    catch(const std::exception&)
    {
        // An unreadable graph is not "covered by every reference". Reporting an
        // empty set would make referenceCoversGraph() vacuously true and register
        // a test for a bundle nobody can run.
        return std::nullopt;
    }
    return types;
}

bool referenceCoversGraph(ReferenceExecutorType type, const void* graphBuffer, size_t size)
{
    const auto types = graphNodeTypes(graphBuffer, size);
    if(!types.has_value() || types->empty())
    {
        return false;
    }

    const auto& supported = referenceSupportedOps(type);
    return std::all_of(types->begin(), types->end(), [&supported](const auto nodeType) {
        return supported.count(nodeType) != 0;
    });
}

// Two independent caps on what the scalar CPU reference is asked to validate.
//
// The tier cap keeps CPU cross-checking to the `quick` bundles. The standard tier
// adds 512x512 shapes at 3.6-5.7s each and GQA variants of shapes quick already
// covers, which is more CPU time than the extra signal is worth.
//
// The working-set cap exists because tier is a taxonomy, not a cost measure, and
// here the two disagree badly: the three most expensive bundles in the tree (seq
// 2048 and 4096, upwards of nine minutes each on CPU) live in `quick`. Tier alone
// would therefore make this lane slower, not faster. Measured, everything at seq
// 256/512 stays under 800K elements and finishes in seconds while those three
// start at 16.7M, so the threshold sits in the 21x gap between the clusters.
constexpr std::string_view K_CPU_TIER_PREFIX = "quick_";
constexpr int64_t K_CPU_MAX_WORKING_SET_ELEMENTS = int64_t{4} * 1024 * 1024;

namespace
{

int64_t elementCount(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr || tensor->dims() == nullptr)
    {
        return 0;
    }

    int64_t elements = 1;
    for(const auto dim : *tensor->dims())
    {
        elements *= dim;
    }
    return elements;
}

} // namespace

bool referenceShapeIsAffordable(ReferenceExecutorType type,
                                std::string_view bundleId,
                                const void* graphBuffer,
                                size_t size)
{
    // Only the CPU reference is gated. The GPU reference runs the same shapes in
    // milliseconds, and it is the one that keeps the excluded bundles covered --
    // which is what makes this a cost decision rather than a coverage loss.
    if(type != ReferenceExecutorType::CPU)
    {
        return true;
    }

    try
    {
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
            graphBuffer, size);
        const auto& tensorMap = graph.getTensorMap();

        for(uint32_t i = 0; i < graph.nodeCount(); ++i)
        {
            const auto& node = graph.getNode(i);
            if(node.attributes_type() != NodeAttributes::SdpaAttributes)
            {
                continue;
            }

            // Both caps apply to Sdpa only. Batchnorm and the other CPU-covered ops
            // are cheap at every checked-in shape, and gating them would trade real
            // coverage for no measurable time.
            if(bundleId.substr(0, K_CPU_TIER_PREFIX.size()) != K_CPU_TIER_PREFIX)
            {
                return false;
            }

            const auto* attributes = node.attributes_as_SdpaAttributes();
            if(attributes == nullptr)
            {
                continue;
            }

            const auto lookup = [&tensorMap](int64_t uid) {
                const auto it = tensorMap.find(uid);
                return it == tensorMap.end() ? nullptr : it->second;
            };

            const int64_t working = std::max(elementCount(lookup(attributes->q_tensor_uid())),
                                             elementCount(lookup(attributes->k_tensor_uid())));
            if(working > K_CPU_MAX_WORKING_SET_ELEMENTS)
            {
                return false;
            }
        }
    }
    catch(const std::exception&)
    {
        // Unreadable graphs are already rejected by referenceCoversGraph(); saying
        // "affordable" here keeps this function from becoming a second, quieter
        // place a bundle can disappear.
        return true;
    }

    return true;
}

std::vector<std::string>
    uncoveredNodeTypes(ReferenceExecutorType type, const void* graphBuffer, size_t size)
{
    const auto types = graphNodeTypes(graphBuffer, size);
    if(!types.has_value())
    {
        return {std::string(K_UNREADABLE_GRAPH)};
    }

    std::vector<std::string> uncovered;
    const auto& supported = referenceSupportedOps(type);
    for(const auto nodeType : *types)
    {
        if(supported.count(nodeType) == 0)
        {
            uncovered.emplace_back(
                hipdnn_flatbuffers_sdk::data_objects::EnumNameNodeAttributes(nodeType));
        }
    }
    return uncovered;
}

std::string formatUncoveredOps(const std::set<std::string>& uncoveredOps)
{
    if(uncoveredOps.empty())
    {
        return {};
    }

    std::string formatted = " (";
    const char* separator = "";
    for(const auto& op : uncoveredOps)
    {
        formatted += separator;
        formatted += op;
        separator = ", ";
    }
    formatted += ")";
    return formatted;
}

const std::vector<KnownReferenceGap>& knownReferenceGaps()
{
    // Every entry here is a promise to delete it. See ALMIOPEN-2563 item 4: the
    // GPU Sdpa reference dispatches through a dtype-keyed plan registry
    // (GpuSdpaFwdSignatureKey.hpp) with no FP8 tuple, and its plan builder rejects
    // variable sequence lengths outright (GpuSdpaFwdPlan.hpp). Both are being
    // implemented; until they are, these bundles must fail applicability rather
    // than vanish, and the run must not be red for a gap we have already written
    // down.
    //
    // Note the CPU reference already handles fp8 (CPU SdpaFwdPlan registers
    // FP8_E4M3 -> BFLOAT16), so none of the fp8 entries below apply to it. It
    // shares only the variable-sequence-length gap.
    static const std::vector<KnownReferenceGap> s_gaps = {
        {ReferenceExecutorType::GPU,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small.Small",
         "variable sequence lengths (seq_len_q/kv) are not implemented"},
        {ReferenceExecutorType::GPU,
         "quick_SdpaFwd_bhsd_bf16_hd128_nomask_group_Small.Small",
         "variable sequence lengths (seq_len_q/kv) are not implemented"},
        {ReferenceExecutorType::GPU,
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_group_Small.Small",
         "variable sequence lengths (seq_len_q/kv) are not implemented, and no FP8 plan exists"},
        {ReferenceExecutorType::GPU,
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small.Small",
         "no FP8 plan: the registry has no FP8_E4M3 tuple and descales are rejected"},
        {ReferenceExecutorType::GPU,
         "quick_SdpaFwd_bhsd_fp8_hd128_nomask_batch_Small.Small",
         "no FP8 plan: the registry has no FP8_E4M3 tuple and descales are rejected"},
        // CPU declines only variable sequence lengths (CPU SdpaFwdPlan.hpp). It
        // handles fp8 fine, which is why the fp8 batch bundles above are GPU-only
        // entries and the fp8 *group* bundle appears for both references.
        {ReferenceExecutorType::CPU,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small.Small",
         "variable sequence lengths (seq_len_q/kv) are not implemented"},
        {ReferenceExecutorType::CPU,
         "quick_SdpaFwd_bhsd_bf16_hd128_nomask_group_Small.Small",
         "variable sequence lengths (seq_len_q/kv) are not implemented"},
        {ReferenceExecutorType::CPU,
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_group_Small.Small",
         "variable sequence lengths (seq_len_q/kv) are not implemented"},
    };
    return s_gaps;
}

const KnownReferenceGap* findKnownReferenceGap(ReferenceExecutorType type,
                                               std::string_view bundleId)
{
    const auto& gaps = knownReferenceGaps();
    const auto it = std::find_if(gaps.begin(), gaps.end(), [&](const KnownReferenceGap& gap) {
        return gap.reference == type && gap.bundleId == bundleId;
    });
    return it == gaps.end() ? nullptr : &*it;
}

} // namespace hipdnn_integration_tests::bundle
