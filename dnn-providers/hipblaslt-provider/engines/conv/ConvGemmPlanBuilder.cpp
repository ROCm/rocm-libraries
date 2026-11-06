// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/ConvGemmPlanBuilder.hpp"

#include <exception>
#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace hipblaslt_plugin
{

ConvGemmPlanBuilder::ConvGemmPlanBuilder(PlanFactory factory)
    : _factory(std::move(factory))
{
}

std::optional<ConvGemmPlanCalls> ConvGemmPlanBuilder::callsFor(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    for(const auto direction :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        const auto problem = parseConvProblem(opGraph, direction);
        if(problem.has_value())
        {
            return directGemmCalls(*problem);
        }
    }
    return std::nullopt;
}

bool ConvGemmPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto calls = callsFor(opGraph);
    if(!calls.has_value())
    {
        return false;
    }
    // The backend has the last word: hipBLASLt, say, may find no algorithm for a call.
    try
    {
        return _factory(handle, std::move(calls->calls)) != nullptr;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO("Explicit-GEMM convolution not applicable: " << e.what());
        return false;
    }
}

size_t ConvGemmPlanBuilder::getWorkspaceSize(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto calls = callsFor(opGraph);
    if(!calls.has_value())
    {
        return 0;
    }
    return _factory(handle, std::move(calls->calls))->getWorkspaceSize(handle);
}

void ConvGemmPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    auto calls = callsFor(opGraph);
    if(!calls.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "explicit-GEMM convolution does not apply to this graph");
    }
    HIPDNN_PLUGIN_LOG_INFO("Building explicit-GEMM convolution plan, row " << toString(
                               calls->row) << ", " << calls->calls.size() << " GEMM call(s)");
    executionContext.setPlan(_factory(handle, std::move(calls->calls)));
}

} // namespace hipblaslt_plugin
