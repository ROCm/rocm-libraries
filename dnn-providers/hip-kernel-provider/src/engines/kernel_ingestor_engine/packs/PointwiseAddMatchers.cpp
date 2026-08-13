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
#include "engines/kernel_ingestor_engine/packs/PointwiseAddDispatchHandler.hpp"
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

/// The tensor ranks this pack accepts; compile options derive layout from the tensor
/// and reject anything outside this range.
constexpr uint32_t MIN_SUPPORTED_RANK = 4;
constexpr uint32_t MAX_SUPPORTED_RANK = 5;

/// True when the tensor's stride order is channel-first or channel-last, the only
/// orders the dispatch path's compile options can classify.
bool hasSupportedLayout(const data_objects::TensorAttributes& tensor)
{
    try
    {
        static_cast<void>(core::utils::isChannelLastLayout(&tensor));
        return true;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
    {
        return false;
    }
}

/// True when the tensor is a supported rank and layout holding exactly one element.
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
    // No device, no launch: every fact this matcher would select on -- including the
    // device properties the compile is configured from -- is meaningless without one.
    if(context.deviceId == hipdnn_plugin_sdk::ingestor::NO_DEVICE)
    {
        return false;
    }

    // Exactly one node: this pack's kernel serves one complete graph.
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

    // A virtual tensor has no device buffer for findDeviceBuffer to resolve at launch.
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

    // Uniform dtype across operands; mixed precision is a different kernel.
    if(inputA->data_type() != inputB->data_type() || inputA->data_type() != output->data_type())
    {
        return false;
    }

    // Binds operand uids for the dispatch handler to read back rather than re-deriving
    // them from the graph.
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

    // Pins the kernel's baked dtype against the graph's, so an f32 graph cannot reach
    // an f16 kernel and get wrong numbers back.
    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == dataTypeName(*dataType);
}

double pointwiseAddScore(const KernelDefinition& kernel, const MatchContext& /*context*/)
{
    // A stand-in for a trained model: prefers the larger block size.
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

PointwiseAddBinding pointwiseAddBinding(const BoundTokens& bound)
{
    // Every token was written by the graph matcher that admitted this graph; a missing
    // one means the catalog was built by a matcher other than ours.
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

void registerPointwiseAddSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &pointwiseAddGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &pointwiseAddKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &pointwiseAddScore);
    scope.add(std::string(DISPATCH_SYMBOL), &pointwiseAddDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
