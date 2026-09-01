// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <string>

#include "LayernormPlanBuilder.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "device/IDevicePropertyProvider.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormBwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormFwdPlan.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormUtilities.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hip_kernel_provider::layernorm
{

namespace
{

size_t getMaxBwdWorkspaceSize(const Handle& handle,
                              const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    // Workspace only needed for the parallel backward pass or for fallback mean/rstd calculations
    const auto& node = opGraph.getNode(0);
    auto deviceProperties = hip_kernel_provider_common::getDeviceProperties(handle.getStream());
    auto attr = node.attributes_as_LayernormBackwardAttributes();
    auto xTensorAttr = opGraph.getTensorMap().at(attr->x_tensor_uid());
    auto scaleTensorAttr = std::make_optional(opGraph.getTensorMap().at(attr->scale_tensor_uid()));
    auto meanTensorAttr
        = attr->mean_tensor_uid().has_value()
              ? std::make_optional(opGraph.getTensorMap().at(attr->mean_tensor_uid().value()))
              : std::nullopt;
    const auto* xDims = xTensorAttr->dims();
    const auto* xStrides = xTensorAttr->strides();
    const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(
        std::vector<int64_t>(xStrides->begin(), xStrides->end()));

    const size_t normalizedDim
        = layernorm::guessNormalizedDim(xTensorAttr, scaleTensorAttr, meanTensorAttr);
    int64_t outerSize = 1;
    int64_t innerSize = 1;
    int64_t stride = 1;
    const auto layoutNHWC = hipdnn_data_sdk::utilities::TensorLayout::NHWC;
    const auto layoutNDHWC = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

    if(normalizedDim > 1
       && (strideOrder == layoutNHWC.strideOrder || strideOrder == layoutNDHWC.strideOrder))
    {
        stride = static_cast<int64_t>(xDims->Get(1));
    }

    for(unsigned int i = 0; i < xDims->size(); ++i)
    {
        if(i < normalizedDim)
        {
            if(stride == 1 || i != 1) // Don't add C to outerSize if there is a stride
            {
                outerSize *= static_cast<int64_t>(xDims->Get(i));
            }
        }
        else
        {
            innerSize *= static_cast<int64_t>(xDims->Get(i));
        }
    }
    const bool parallel
        = LayernormBwdPlan::isParallel(deviceProperties,
                                       static_cast<size_t>(LayernormBwdParams::MAX_LOCAL_SIZE),
                                       static_cast<size_t>(innerSize),
                                       static_cast<size_t>(outerSize));
    const size_t parallelSize
        = LayernormBwdPlan::getParallelSize(deviceProperties,
                                            static_cast<size_t>(LayernormBwdParams::MAX_LOCAL_SIZE),
                                            static_cast<size_t>(innerSize),
                                            static_cast<size_t>(outerSize));
    const size_t sizeFromParallel
        = parallel ? 2 * static_cast<size_t>(innerSize) * parallelSize : 0;
    const size_t sizeFromFallbackMeanRstd
        = !attr->mean_tensor_uid().has_value() || !attr->inv_variance_tensor_uid().has_value()
              ? 2 * static_cast<size_t>(outerSize) * static_cast<size_t>(stride)
              : 0;
    return 4 * (sizeFromParallel + sizeFromFallbackMeanRstd);
}

} // namespace

LayernormPlanBuilder::LayernormPlanBuilder(const IKernelCompiler& kernelCompiler,
                                           const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool LayernormPlanBuilder::isApplicable(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the plan bakes those dims into the compiled kernel
    // launch, so decline rather than risk a mismatch (RFC 0008 §4.6).
    if(opGraph.getGraph().is_override_shape_enabled())
    {
        HIPDNN_PLUGIN_LOG_INFO("Layernorm plan builder does not support override shapes");
        return false;
    }

    auto anyNodeIsNotF32Compute = [&]() {
        return !std::all_of(
            opGraph.nodeWrappers().begin(), opGraph.nodeWrappers().end(), [](const auto& node) {
                return node->computeDataType()
                       == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
            });
    };

    switch(opGraph.nodeCount())
    {
    case 1:
    {
        if(anyNodeIsNotF32Compute())
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "Layernorm plan builder only supports nodes with an fp32 compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_flatbuffers_sdk::data_objects::NodeAttributes>{
                   hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes,
                   hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
                       LayernormBackwardAttributes}))
        {
            HIPDNN_PLUGIN_LOG_INFO("Layernorm plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        try
        {
            LayernormValidator validator(opGraph.getTensorMap());
            switch(node.attributes_type())
            {
            case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes:
                validator.checkTensorConfigSupported(*node.attributes_as_LayernormAttributes());
                break;
            case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes:
                validator.checkTensorConfigSupported(
                    *node.attributes_as_LayernormBackwardAttributes());
                break;
            default:
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               "Unexpected node attribute type");
            }
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO(e.what());
            return false;
        }

        return true;
    }
    default:
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Layernorm plan builder is only applicable to 1 node graphs. Graph has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }
    }
}

size_t LayernormPlanBuilder::getMaxWorkspaceSize(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const Settings& executionSettings) const
{
    const auto& node = opGraph.getNode(0);
    switch(node.attributes_type())
    {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes:
        // Layernorm fwd plan builder does not require workspace size
        return 0;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes:
        return getMaxBwdWorkspaceSize(handle, opGraph);
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Unexpected node attribute type");
    }
}

namespace
{

void buildPlanFwd([[maybe_unused]] const Handle& handle,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
                  const IKernelCompiler& kernelCompiler,
                  const IDevicePropertyProvider& devicePropertyProvider,
                  Context& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes>();

    LayernormFwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<LayernormFwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());

    executionContext.setPlan(std::move(plan));
}

void buildPlanBwd([[maybe_unused]] const Handle& handle,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::INodeWrapper& nodeWrapper,
                  const IKernelCompiler& kernelCompiler,
                  const IDevicePropertyProvider& devicePropertyProvider,
                  Context& executionContext)
{
    const auto& attr
        = nodeWrapper
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes>();

    LayernormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<LayernormBwdPlan>(std::move(params));
    plan->compile(kernelCompiler, devicePropertyProvider.getDeviceProperties());

    executionContext.setPlan(std::move(plan));
}

} // namespace

void LayernormPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings) const
{
}

void LayernormPlanBuilder::buildPlan(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    Context& executionContext) const
{
    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building layernorm fwd plan for node: " << nodeName);
        buildPlanFwd(handle,
                     opGraph,
                     nodeWrapper,
                     _kernelCompiler,
                     _devicePropertyProvider,
                     executionContext);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building layernorm bwd plan for node: " << nodeName);
        buildPlanBwd(handle,
                     opGraph,
                     nodeWrapper,
                     _kernelCompiler,
                     _devicePropertyProvider,
                     executionContext);
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for layernorm plan builder: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(nodeWrapper.attributesType())));
    }
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> LayernormPlanBuilder::getCustomKnobs(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider::layernorm
