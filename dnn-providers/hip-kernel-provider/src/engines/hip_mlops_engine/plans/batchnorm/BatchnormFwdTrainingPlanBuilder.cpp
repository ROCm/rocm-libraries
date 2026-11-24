// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "BatchnormFwdTrainingPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdTrainingPlan.hpp"

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::batchnorm
{

BatchnormFwdTrainingPlanBuilder::BatchnormFwdTrainingPlanBuilder(
    const IKernelCompiler& kernelCompiler, const IDevicePropertyProvider& devicePropertyProvider)
    : _kernelCompiler(kernelCompiler)
    , _devicePropertyProvider(devicePropertyProvider)
{
}

bool BatchnormFwdTrainingPlanBuilder::isApplicable(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // Execute-time override shapes can diverge from the compile-time dims this
    // builder matched exactly; the plan bakes those dims into the compiled kernel
    // launch, so decline rather than risk a mismatch (RFC 0008 §4.6).
    if(opGraph.getGraph().is_override_shape_enabled())
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "BatchnormFwdTraining plan builder does not support override shapes");
        return false;
    }

    if(opGraph.nodeCount() != 1 && opGraph.nodeCount() != 2)
    {
        return false;
    }

    try
    {
        // Common batchnorm validation
        const auto& bnAttr = checkTrainingNode(opGraph.getNodeWrapper(0));

        auto hasFloatComputeDataType = [](const auto& node) {
            return node->computeDataType() == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        };

        if(!std::all_of(opGraph.nodeWrappers().begin(),
                        opGraph.nodeWrappers().end(),
                        hasFloatComputeDataType))
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "BatchnormFwdTraining plan builder only supports nodes with an fp32 "
                "compute_data_type");
            return false;
        }

        BatchnormValidator validator(opGraph.getTensorMap());
        if(opGraph.nodeCount() == 1)
        {
            // Solo batchnorm training
            checkTrainingTensorVirtuality1Node(bnAttr, opGraph.getTensorMap());

            // Validate applicability before kernel dispatch by checking tensor configurations
            // and operation parameters manually.
            validator.checkFwdTrainingTensorConfigSupported(bnAttr);

            HIPDNN_PLUGIN_LOG_INFO("BatchnormFwdTraining plan builder applicable for single node "
                                   "batchnorm training");
            return true;
        }

        // nodeCount == 2: Batchnorm training + activation fusion
        const auto& activAttr = checkTrainingActivationNode(opGraph.getNodeWrapper(1), bnAttr);
        checkTrainingTensorVirtuality2Node(bnAttr, activAttr, opGraph.getTensorMap());

        validator.checkFwdTrainingActivationTensorConfigSupported(bnAttr, activAttr);

        HIPDNN_PLUGIN_LOG_INFO(
            "BatchnormFwdTraining plan builder applicable for training + activation fusion");
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

size_t BatchnormFwdTrainingPlanBuilder::getMaxWorkspaceSize(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const Settings& executionSettings) const
{
    // No workspace needed for batchnorm forward training
    return 0;
}

void BatchnormFwdTrainingPlanBuilder::initializeExecutionSettings(
    [[maybe_unused]] const Handle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    [[maybe_unused]] Settings& executionSettings) const
{
}

void BatchnormFwdTrainingPlanBuilder::buildPlan(
    [[maybe_unused]] const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    Context& executionContext) const
{
    if(opGraph.nodeCount() == 1)
    {
        // Solo batchnorm training
        const auto& bnAttr
            = opGraph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes>();

        BatchnormFwdTrainingParams params(bnAttr, opGraph.getTensorMap());
        auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
        plan->compile(_kernelCompiler, _devicePropertyProvider.getDeviceProperties());
        executionContext.setPlan(std::move(plan));
    }
    else if(opGraph.nodeCount() == 2)
    {
        // Batchnorm training + activation fusion
        const auto& bnAttr = checkTrainingNode(opGraph.getNodeWrapper(0));
        const auto& activAttr = checkTrainingActivationNode(opGraph.getNodeWrapper(1), bnAttr);
        BatchnormValidator validator(opGraph.getTensorMap());
        validator.checkFwdTrainingActivationTensorConfigSupported(bnAttr, activAttr);

        BatchnormFwdTrainingParams params(bnAttr, activAttr, opGraph.getTensorMap());
        auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
        plan->compile(_kernelCompiler, _devicePropertyProvider.getDeviceProperties());
        executionContext.setPlan(std::move(plan));
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fwd training plan builder supports only 1 or 2 node graphs");
    }
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
    BatchnormFwdTrainingPlanBuilder::getCustomKnobs(
        [[maybe_unused]] const Handle& handle,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace hip_kernel_provider::batchnorm
