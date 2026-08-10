// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ingestor_poc/NativeMatchers.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <numeric>
#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "ingestor_poc/NativeSymbolNames.hpp"

namespace hip_kernel_provider::ingestor_poc
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The tensor ranks this pack accepts. Not a property of the kernel, which indexes one
/// element and is indifferent to rank, but of the dispatch path: the provider's compile
/// options derive layout from the tensor and reject anything outside this range.
///
/// Matching has to respect that. Accepting a graph commits the engine to producing a
/// launchable kernel for it, so a matcher that admits a rank dispatch cannot serve turns
/// a free decline into a failed plan build the caller pays for.
constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;

/// True when the tensor is a supported rank holding exactly one element -- the whole of
/// this POC's supported problem space.
bool isSingleElement(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    if(dims == nullptr || dims->size() < MIN_SUPPORTED_RANK || dims->size() > MAX_SUPPORTED_RANK)
    {
        return false;
    }

    int64_t elements = 1;
    for(const auto dim : *dims)
    {
        elements *= dim;
    }
    return elements == 1;
}

/// The graph's element type, taken from the first input. The matcher below requires
/// every operand to agree, so any of them would answer the same.
std::optional<data_objects::DataType> graphDataType(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return std::nullopt;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return std::nullopt;
    }

    const auto& attributes = node.attributesAs<data_objects::PointwiseAttributes>();
    const auto* input = findTensor(context, attributes.in_0_tensor_uid());
    if(input == nullptr)
    {
        return std::nullopt;
    }
    return input->data_type();
}

std::string dataTypeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

} // namespace

bool pointwiseAddGraphMatches(const MatchContext& context)
{
    // Exactly one node: a prebuilt kernel serves one complete graph, so anything larger
    // is a different problem even if it contains this one.
    if(context.graph.nodeCount() != 1)
    {
        return false;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return false;
    }

    const auto& attributes = node.attributesAs<data_objects::PointwiseAttributes>();
    if(attributes.operation() != data_objects::PointwiseMode::ADD)
    {
        return false;
    }

    // Binary add: a second operand is required, a third would be a different operation.
    if(!attributes.in_1_tensor_uid().has_value() || attributes.in_2_tensor_uid().has_value())
    {
        return false;
    }

    const auto* inputA = findTensor(context, attributes.in_0_tensor_uid());
    const auto* inputB = findTensor(context, attributes.in_1_tensor_uid().value());
    const auto* output = findTensor(context, attributes.out_0_tensor_uid());
    if(inputA == nullptr || inputB == nullptr || output == nullptr)
    {
        return false;
    }

    // One element each: this POC's kernel indexes element 0 and nothing else.
    if(!isSingleElement(*inputA) || !isSingleElement(*inputB) || !isSingleElement(*output))
    {
        return false;
    }

    // Uniform dtype across operands. Mixed-precision add is a different kernel, and
    // accepting it here would hand one binary operands it cannot read.
    return inputA->data_type() == inputB->data_type() && inputA->data_type() == output->data_type();
}

bool pointwiseAddKernelMatches(const MatchContext& context, const KernelDefinition& kernel)
{
    const auto dataType = graphDataType(context);
    if(!dataType.has_value())
    {
        return false;
    }

    // Pins the kernel's baked element type against the graph's. Without this, the
    // graph-level gate would accept an f32 graph and selection could hand it to the
    // f16 kernel, which returns wrong numbers rather than failing.
    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == dataTypeName(*dataType);
}

double pointwiseAddScore(const KernelDefinition& kernel, const MatchContext& /*context*/)
{
    // A stand-in for a trained model: prefer the larger block size. It gives ranking a
    // defined, inspectable winner without pretending to be a performance judgement —
    // for a one-element add, no block size is actually better.
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

PointwiseAddBinding pointwiseAddBinding(const MatchContext& context)
{
    if(!pointwiseAddGraphMatches(context))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "graph is not a single-node 1-element pointwise add");
    }

    const auto& attributes
        = context.graph.getNodeWrapper(0).attributesAs<data_objects::PointwiseAttributes>();

    return {attributes.in_0_tensor_uid(),
            attributes.in_1_tensor_uid().value(),
            attributes.out_0_tensor_uid()};
}

void registerPointwiseAddMatchers()
{
    GraphMatcherRegistry::registerSymbol(std::string(GRAPH_MATCHER_SYMBOL),
                                         &pointwiseAddGraphMatches);
    KernelMatcherRegistry::registerSymbol(std::string(KERNEL_MATCHER_SYMBOL),
                                          &pointwiseAddKernelMatches);
    ScoreRegistry::registerSymbol(std::string(SCORE_SYMBOL), &pointwiseAddScore);
}

} // namespace hip_kernel_provider::ingestor_poc

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
