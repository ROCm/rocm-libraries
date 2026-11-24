// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/reference-validation/ReferenceOpCoverage.hpp"

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
//
// ConvolutionFwdAttributes and MatmulAttributes are absent although the CPU reference runs them:
// no golden-bearing bundle uses either, and referenceShapeIsAffordable() only knows Sdpa.
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
        NodeAttributes::SdpaBackwardAttributes,
    };
    return s_ops;
}

// Ops the GPU reference is required to handle. Narrower than the CPU set: the GPU
// reference dispatches through a signature-keyed plan registry, so coverage is
// per-op-shape and grows only as plan builders are written.
//
// PointwiseAttributes is absent while the GPU reference lacks pointwise graphs the CPU one takes.
// TestPointwiseReferenceParity requires adding it, which also stops Convolution Forward + Pointwise
// bundles reporting OUTSIDE_OP_SET, once the last gap closes.
const std::set<NodeAttributes>& gpuSupportedOps()
{
    static const std::set<NodeAttributes> s_ops = {
        NodeAttributes::ConvolutionFwdAttributes,
        NodeAttributes::SdpaAttributes,
        NodeAttributes::SdpaBackwardAttributes,
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

// Backward is capped on its own axis because its cost does not follow the forward cap: the
// scalar backward needs seconds to tens of seconds at a 256x128 head (8.4M MACs per Q*K^T
// pass) while the 64-row shapes below 3.2M finish in about a second or less, so the 4M
// element cap above would admit every one of them. The threshold sits between those two
// clusters. Measured on a loaded CI runner, so only the gap, not the absolute times, is
// meaningful.
constexpr int64_t K_CPU_MAX_SDPA_BWD_MACS = int64_t{4} * 1024 * 1024;

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

// Both caps apply to Sdpa forward and backward only, so a graph with no Sdpa node is
// affordable by construction. Batchnorm and the other CPU-covered ops are cheap at every
// checked-in shape, and gating them would trade real coverage for no measurable time.
bool isSdpaShapeAffordableOnCpuRunner(std::string_view bundleId,
                                      const void* graphBuffer,
                                      size_t size)
{
    try
    {
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
            graphBuffer, size);
        const auto& tensorMap = graph.getTensorMap();

        const auto lookup = [&tensorMap](int64_t uid) {
            const auto it = tensorMap.find(uid);
            return it == tensorMap.end() ? nullptr : it->second;
        };

        for(uint32_t i = 0; i < graph.nodeCount(); ++i)
        {
            const auto& node = graph.getNode(i);
            const bool isForward = node.attributes_type() == NodeAttributes::SdpaAttributes;
            const bool isBackward
                = node.attributes_type() == NodeAttributes::SdpaBackwardAttributes;
            if(!isForward && !isBackward)
            {
                continue;
            }

            if(bundleId.substr(0, K_CPU_TIER_PREFIX.size()) != K_CPU_TIER_PREFIX)
            {
                return false;
            }

            if(isForward)
            {
                const auto* attributes = node.attributes_as_SdpaAttributes();
                if(attributes == nullptr)
                {
                    continue;
                }

                const int64_t working = std::max(elementCount(lookup(attributes->q_tensor_uid())),
                                                 elementCount(lookup(attributes->k_tensor_uid())));
                if(working > K_CPU_MAX_WORKING_SET_ELEMENTS)
                {
                    return false;
                }
                continue;
            }

            const auto* attributes = node.attributes_as_SdpaBackwardAttributes();
            if(attributes == nullptr)
            {
                continue;
            }

            // Q is [B, H, Sq, D] and K is [B, Hkv, Skv, D], so Q's element count times Skv is
            // the multiply-accumulate count of one Q*K^T pass (B * H * Sq * Skv * D).
            const auto* k = lookup(attributes->k_tensor_uid());
            if(k == nullptr || k->dims() == nullptr || k->dims()->size() < 2)
            {
                continue;
            }
            const int64_t seqKv = (*k->dims())[k->dims()->size() - 2];
            if(elementCount(lookup(attributes->q_tensor_uid())) * seqKv > K_CPU_MAX_SDPA_BWD_MACS)
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

    return isSdpaShapeAffordableOnCpuRunner(bundleId, graphBuffer, size);
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
    // Every entry here is a promise to delete it: a listed bundle must fail
    // applicability rather than vanish, and the run must not be red for a gap we
    // have already written down. Both references run every bundle today (the GPU fp8
    // forward dequantizes first, as the CPU one does), so the list is empty.
    static const std::vector<KnownReferenceGap> s_gaps = {};
    return s_gaps;
}

const KnownReferenceGap* findKnownReferenceGap(ReferenceExecutorType type,
                                               std::string_view bundleId)
{
    return findKnownReferenceGap(knownReferenceGaps(), type, bundleId);
}

const KnownReferenceGap* findKnownReferenceGap(const std::vector<KnownReferenceGap>& gaps,
                                               ReferenceExecutorType type,
                                               std::string_view bundleId)
{
    const auto it = std::find_if(gaps.begin(), gaps.end(), [&](const KnownReferenceGap& gap) {
        return gap.reference == type && gap.bundleId == bundleId;
    });
    return it == gaps.end() ? nullptr : &*it;
}

} // namespace hipdnn_integration_tests::bundle
