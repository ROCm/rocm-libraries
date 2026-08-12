// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <numeric>
#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Utils.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
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

/// The tensor ranks this pack accepts. Not a kernel property (it indexes one element
/// regardless of rank) but a dispatch-path one: compile options derive layout from the
/// tensor and reject anything outside this range, so a matcher that admits a rank
/// dispatch cannot serve turns a free decline into a failed plan build.
constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;

/// True when the tensor's stride order is one the dispatch path can classify.
///
/// Compile options derive a layout from the strides and throw on any order that is
/// neither channel-first nor channel-last; a one-element tensor viewing into a larger
/// buffer can still carry an unclassifiable order.
bool hasSupportedLayout(const data_objects::TensorAttributes& tensor)
{
    try
    {
        static_cast<void>(core::utils::isChannelLastLayout(&tensor));
        return true;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        // Asking the classifier directly keeps this gate and the dispatch path on one
        // definition of "supported layout" rather than two that can drift.
        return false;
    }
}

/// True when the tensor is a supported rank and layout holding exactly one element --
/// the whole of this pack's supported problem space.
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
    if(elements != 1)
    {
        return false;
    }

    return hasSupportedLayout(tensor);
}

/// The graph's element type, from the first input; the matcher below requires every
/// operand to agree, so any of them would answer the same.
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

bool pointwiseAddGraphMatches(const MatchContext& context, BoundTokens& bound)
{
    // No device, no launch. The resolver reports NO_DEVICE when it cannot name the
    // device this call is for, and every fact below that a kernel would be selected on
    // -- including the device properties the compile is configured from -- is
    // meaningless without one. Declining here is what keeps that failure a clean "this
    // engine does not apply" rather than a property lookup for a device that does not
    // exist.
    if(context.deviceId == hipdnn_plugin_sdk::ingestor::NO_DEVICE)
    {
        return false;
    }

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

    // One element each: this pack's kernel indexes element 0 and nothing else.
    if(!isSingleElement(*inputA) || !isSingleElement(*inputB) || !isSingleElement(*output))
    {
        return false;
    }

    // A virtual tensor never materializes into a device buffer, so it is absent from
    // the variant pack findDeviceBuffer resolves at launch.
    if(inputA->virtual_() || inputB->virtual_() || output->virtual_())
    {
        return false;
    }

    // A 1-element rank-4 tensor is also the shape of a pass-by-value scalar, whose
    // variant-pack slot holds a host pointer, not a device one.
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputA)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(inputB)
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(output))
    {
        return false;
    }

    // Uniform dtype across operands. Mixed-precision add is a different kernel, and
    // accepting it here would hand one binary operands it cannot read.
    if(inputA->data_type() != inputB->data_type() || inputA->data_type() != output->data_type())
    {
        return false;
    }

    // Bind what the launch needs, now that the walk that found it has succeeded. The
    // dispatch handler reads these back instead of re-walking the graph, so there is one
    // notion of which tensor is which operand rather than two that can drift apart.
    // The uid type is already int64_t; what matters is that these land in the
    // MetadataValue's integer alternative, which is the one the dispatch side reads.
    bound[std::string(INPUT_A_TOKEN)] = attributes.in_0_tensor_uid();
    bound[std::string(INPUT_B_TOKEN)] = attributes.in_1_tensor_uid().value();
    bound[std::string(OUTPUT_TOKEN)] = attributes.out_0_tensor_uid();
    return true;
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

PointwiseAddBinding pointwiseAddBinding(const BoundTokens& bound)
{
    // Every token was written by the graph matcher that admitted this graph, so a missing
    // one -- or one holding something other than the uid this expects -- means the
    // catalog was built by a matcher other than ours: a wiring error, not a graph this
    // pack should decline.
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "pointwise add dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(INPUT_A_TOKEN), read(INPUT_B_TOKEN), read(OUTPUT_TOKEN)};
}

void registerPointwiseAddMatchers()
{
    GraphMatcherRegistry::registerSymbol(std::string(GRAPH_MATCHER_SYMBOL),
                                         &pointwiseAddGraphMatches);
    try
    {
        KernelMatcherRegistry::registerSymbol(std::string(KERNEL_MATCHER_SYMBOL),
                                              &pointwiseAddKernelMatches);
    }
    catch(...)
    {
        GraphMatcherRegistry::unregisterSymbol(std::string(GRAPH_MATCHER_SYMBOL));
        throw;
    }
    try
    {
        ScoreRegistry::registerSymbol(std::string(SCORE_SYMBOL), &pointwiseAddScore);
    }
    catch(...)
    {
        GraphMatcherRegistry::unregisterSymbol(std::string(GRAPH_MATCHER_SYMBOL));
        KernelMatcherRegistry::unregisterSymbol(std::string(KERNEL_MATCHER_SYMBOL));
        throw;
    }
}

void unregisterPointwiseAddMatchers()
{
    GraphMatcherRegistry::unregisterSymbol(std::string(GRAPH_MATCHER_SYMBOL));
    KernelMatcherRegistry::unregisterSymbol(std::string(KERNEL_MATCHER_SYMBOL));
    ScoreRegistry::unregisterSymbol(std::string(SCORE_SYMBOL));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
