// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/conv/ConvGemmPlanBuilder.hpp"

#include <algorithm>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
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

namespace
{

/// Whether the current device is gfx11 (gcnArchName read once per device).
bool currentDeviceIsGfx11()
{
    static std::mutex s_mutex;
    static std::map<int, bool> s_gfx11;
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
    {
        return false;
    }
    const std::lock_guard<std::mutex> lock(s_mutex);
    auto found = s_gfx11.find(device);
    if(found == s_gfx11.end())
    {
        hipDeviceProp_t properties{};
        const bool gfx11 = hipGetDeviceProperties(&properties, device) == hipSuccess
                           && std::string(properties.gcnArchName).rfind("gfx11", 0) == 0;
        found = s_gfx11.emplace(device, gfx11).first;
    }
    return found->second;
}

bool narrowPartials(ConvGemmPlanBuilder::NarrowPartials policy)
{
    return policy == ConvGemmPlanBuilder::NarrowPartials::ALWAYS
           || (policy == ConvGemmPlanBuilder::NarrowPartials::ON_GFX11 && currentDeviceIsGfx11());
}

} // namespace

ConvGemmPlanBuilder::ConvGemmPlanBuilder(PlanFactory factory,
                                         size_t maxRepackWorkspaceBytes,
                                         bool tf32,
                                         bool reducedPrecisionReduction,
                                         bool splitReduction,
                                         ColumnLayout layout,
                                         NarrowPartials narrowPartials)
    : _factory(std::move(factory))
    , _maxRepackWorkspaceBytes(maxRepackWorkspaceBytes)
    , _tf32(tf32)
    , _reducedPrecisionReduction(reducedPrecisionReduction)
    , _splitReduction(splitReduction)
    , _layout(layout)
    , _narrowPartials(narrowPartials)
{
}

std::optional<ConvGemmPlanCalls> ConvGemmPlanBuilder::callsFor(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    ColumnLayout layout,
    bool splitReduction) const
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
        auto calls = convGemmCalls(*problem,
                                   _maxRepackWorkspaceBytes,
                                   layout,
                                   splitReduction,
                                   narrowPartials(_narrowPartials));
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
    auto calls = callsFor(opGraph, _layout, _splitReduction);
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
    auto calls = callsFor(opGraph, _layout, _splitReduction);
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
    auto calls = callsFor(opGraph, _layout, _splitReduction);
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
    // channels-first backward weights' batched partials or channels-last staging) or of a
    // backward weights' reduction split, time every distinct one on the first execute and keep
    // the fastest: none wins every shape.
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
        // Staged channels-last: one GEMM over every image of a channels-first y, where the
        // other layouts batch a small per-image GEMM (1x1 stride 2: 25 x 25 outputs an image).
        for(const auto& [layout, name] :
            {std::pair{ColumnLayout::CHANNELS_FIRST, "channels_first"},
             std::pair{ColumnLayout::CHANNELS_LAST, "channels_last"},
             std::pair{ColumnLayout::STAGED_CHANNELS_LAST, "staged_channels_last"}})
        {
            auto other = callsFor(opGraph, layout);
            if(other.has_value()
               && std::find(seen.begin(), seen.end(), signature(*other)) == seen.end())
            {
                seen.push_back(signature(*other));
                candidates.push_back({name, planFor(handle, std::move(*other))});
            }
        }
        // A reduction split into enough fp32 partials to fill the GPU: the libraries' own
        // long-k kernels leave a small output's few tiles on a few CUs (backward weights only).
        auto split = callsFor(opGraph, ColumnLayout::FOLLOW_Y, true);
        if(split.has_value()
           && std::find(seen.begin(), seen.end(), signature(*split)) == seen.end())
        {
            seen.push_back(signature(*split));
            candidates.push_back({"split_reduction", planFor(handle, std::move(*split))});
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
