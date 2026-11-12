// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/ConvGemmPlanBuilder.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/BenchmarkedPlan.hpp"
#include "engines/conv/ConvReducedPrecision.hpp"
#include "engines/conv/ConvRepackPlan.hpp"

namespace hipblaslt_plugin
{

ConvGemmPlanBuilder::ConvGemmPlanBuilder(PlanFactory factory,
                                         size_t maxRepackWorkspaceBytes,
                                         bool tf32,
                                         bool reducedPrecisionReduction)
    : _factory(std::move(factory))
    , _maxRepackWorkspaceBytes(maxRepackWorkspaceBytes)
    , _tf32(tf32)
    , _reducedPrecisionReduction(reducedPrecisionReduction)
{
}

std::optional<ConvGemmPlanCalls> ConvGemmPlanBuilder::callsFor(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph, ColumnLayout layout) const
{
    for(const auto direction :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        const auto problem = parseConvProblem(opGraph, direction);
        if(!problem.has_value())
        {
            continue;
        }
        if(_reducedPrecisionReduction)
        {
            // Read per call rather than at registration, so no plan of these engines is built
            // without the switch.
            if(!convReducedPrecisionEnginesEnabled())
            {
                return std::nullopt;
            }
            return reducedPrecisionGemmCalls(*problem);
        }
        auto calls = convGemmCalls(*problem, _maxRepackWorkspaceBytes, layout);
        if(calls.has_value() && _tf32)
        {
            // A TF32 engine serves only convolutions with an fp32 GEMM; for the rest it would
            // duplicate the fp32 engine.
            bool fp32Gemm = false;
            for(auto& stage : calls->stages)
            {
                for(auto& call : stage.calls)
                {
                    call.tf32 = true;
                    fp32Gemm = fp32Gemm
                               || call.a.dataType
                                      == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
                }
            }
            if(!fp32Gemm)
            {
                return std::nullopt;
            }
        }
        return calls;
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
        return planFor(handle, std::move(*calls)) != nullptr;
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
    return planFor(handle, std::move(*calls))->getWorkspaceSize(handle);
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
    HIPDNN_PLUGIN_LOG_INFO("Building explicit-GEMM convolution plan, row "
                           << toString(calls->row) << ", " << calls->stages.size() << " stage(s), "
                           << calls->workspaceBytes << " workspace bytes");

    // Benchmarking: when the plan has a choice of layout (an IM2COL column matrix; a 1x1
    // channels-first backward weights' batched partials or channels-last staging), time every
    // distinct one on the first execute and keep the fastest: none wins every shape.
    if(executionContext.benchmarking())
    {
        std::vector<BenchmarkedPlan::Candidate> candidates;
        std::vector<std::tuple<ConvGemmRow, bool, size_t, size_t>> seen;
        const auto signature = [](const ConvGemmPlanCalls& plan) {
            return std::make_tuple(
                plan.row, plan.columnsFirst, plan.workspaceBytes, plan.stages.size());
        };
        seen.push_back(signature(*calls));
        candidates.push_back({"default", planFor(handle, *calls)});
        for(const auto& [layout, name] : {std::pair{ColumnLayout::CHANNELS_FIRST, "channels_first"},
                                          std::pair{ColumnLayout::CHANNELS_LAST, "channels_last"}})
        {
            auto other = callsFor(opGraph, layout);
            if(other.has_value()
               && std::find(seen.begin(), seen.end(), signature(*other)) == seen.end())
            {
                seen.push_back(signature(*other));
                candidates.push_back({name, planFor(handle, std::move(*other))});
            }
        }
        if(candidates.size() > 1)
        {
            executionContext.setPlan(std::make_unique<BenchmarkedPlan>(std::move(candidates)));
            return;
        }
        executionContext.setPlan(std::move(candidates.front().plan));
        return;
    }
    executionContext.setPlan(planFor(handle, std::move(*calls)));
}

std::unique_ptr<IPlan> ConvGemmPlanBuilder::planFor(const HipdnnEnginePluginHandle& handle,
                                                    ConvGemmPlanCalls calls) const
{
    if(!calls.hasRepack())
    {
        // A direct row: one stage of GEMM calls, run by the backend's plan alone.
        return _factory(handle, std::move(calls.stages.front().calls));
    }
    std::vector<std::unique_ptr<IPlan>> gemmPlans;
    for(auto& stage : calls.stages)
    {
        gemmPlans.push_back(stage.calls.empty() ? nullptr : _factory(handle, stage.calls));
    }
    return std::make_unique<ConvRepackPlan>(std::move(calls), std::move(gemmPlans));
}

} // namespace hipblaslt_plugin
