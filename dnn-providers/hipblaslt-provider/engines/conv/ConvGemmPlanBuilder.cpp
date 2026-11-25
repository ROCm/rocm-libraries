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
#include "common/ConvEpilogueFormulation.hpp"
#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/BenchmarkedPlan.hpp"
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

bool usesAddend(const ConvGemmPlanCalls& calls)
{
    for(const auto& stage : calls.stages)
    {
        for(const auto& call : stage.calls)
        {
            if(call.c.has_value())
            {
                return true;
            }
        }
    }
    return false;
}

const char* epilogueFormName(EpilogueForm form)
{
    switch(form)
    {
    case EpilogueForm::NATIVE:
        return "native";
    case EpilogueForm::FUSED_PASS:
        return "fused pass";
    case EpilogueForm::EXTRA_PASS:
        return "extra pass";
    default:
        break;
    }
    return "none";
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
                                         NarrowPartials narrowPartials,
                                         std::optional<size_t> maxBackwardWeightsRepackBytes,
                                         GemmEpilogueSupport gemmEpilogue)
    : _factory(std::move(factory))
    , _maxRepackWorkspaceBytes(maxRepackWorkspaceBytes)
    , _tf32(tf32)
    , _reducedPrecisionReduction(reducedPrecisionReduction)
    , _splitReduction(splitReduction)
    , _layout(layout)
    , _narrowPartials(narrowPartials)
    , _maxBackwardWeightsRepackBytes(maxBackwardWeightsRepackBytes)
    , _gemmEpilogue(gemmEpilogue)
{
}

std::optional<ConvGemmPlanCalls> ConvGemmPlanBuilder::callsFor(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    ColumnLayout layout,
    bool splitReduction,
    bool wideBackwardWeights,
    bool gemmEpilogue) const
{
    for(const auto direction :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        const auto problem = parseConvProblem(opGraph, direction);
        if(!problem.has_value())
        {
            continue;
        }
        if(wideBackwardWeights
           && (direction != ConvDirection::BACKWARD_WEIGHTS
               || !_maxBackwardWeightsRepackBytes.has_value() || _reducedPrecisionReduction))
        {
            return std::nullopt;
        }
        if(_reducedPrecisionReduction)
        {
            return reducedPrecisionGemmCalls(*problem);
        }
        // The library-activation sibling: only the graphs, and the plans, it differs on.
        if(_gemmEpilogue.libraryActivations && (!gemmEpilogue || !usesLibraryActivation(*problem)))
        {
            return std::nullopt;
        }
        auto calls = convGemmCalls(
            *problem,
            wideBackwardWeights ? _maxBackwardWeightsRepackBytes.value_or(_maxRepackWorkspaceBytes)
                                : _maxRepackWorkspaceBytes,
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
        // A fused graph: the epilogue in the GEMM, the plan's last pass or one more pass.
        if(calls.has_value() && !problem->epilogue.empty())
        {
            auto fused = fuseEpilogue(
                *problem,
                std::move(*calls),
                gemmEpilogue ? std::optional<GemmEpilogueSupport>(_gemmEpilogue) : std::nullopt);
            if(fused.has_value() && _gemmEpilogue.libraryActivations
               && fused->epilogueForm != EpilogueForm::NATIVE)
            {
                return std::nullopt;
            }
            return fused;
        }
        return calls;
    }
    return std::nullopt;
}

std::optional<ConvGemmPlanCalls> ConvGemmPlanBuilder::buildableCalls(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto calls = callsFor(opGraph, _layout, _splitReduction);
    if(!calls.has_value() || calls->epilogueForm == EpilogueForm::NONE)
    {
        return calls;
    }
    // A fused plan writes y's result in fp32. A backend may have no algorithm for one layout's
    // fp32-output GEMMs (hipBLASLt without a workspace, the deterministic engines) and one for
    // another's: the first layout it can run serves the graph, as its unfused plan does.
    const auto runs = [&](const ConvGemmPlanCalls& candidate) {
        try
        {
            return planFor(handle, candidate) != nullptr;
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_INFO("Fused explicit-GEMM layout not runnable: " << e.what());
            return false;
        }
    };
    if(runs(*calls))
    {
        return calls;
    }
    // A GEMM epilogue the backend has no algorithm for: the epilogue in a pass.
    if(calls->epilogueForm == EpilogueForm::NATIVE)
    {
        auto pass = callsFor(opGraph, _layout, _splitReduction, false, false);
        if(!pass.has_value())
        {
            return calls;
        }
        calls = std::move(pass);
        if(runs(*calls))
        {
            return calls;
        }
    }
    for(const auto layout : {ColumnLayout::STAGED_CHANNELS_LAST,
                             ColumnLayout::CHANNELS_LAST,
                             ColumnLayout::CHANNELS_FIRST,
                             ColumnLayout::FOLLOW_Y})
    {
        if(layout == _layout)
        {
            continue;
        }
        auto other = callsFor(opGraph, layout, _splitReduction);
        if(other.has_value() && runs(*other))
        {
            return other;
        }
    }
    // No layout's long fp32-output GEMM runs: the same sum in shorter reductions, each adding
    // to the fp32 result in order (hipBLASLt without a workspace has algorithms for those).
    for(const int64_t chunks : {2, 4, 8, 16})
    {
        auto split = splitFp32OutputReductions(*calls, chunks);
        if(runs(split))
        {
            return split;
        }
    }
    return calls;
}

bool ConvGemmPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto calls = buildableCalls(handle, opGraph);
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
    auto calls = buildableCalls(handle, opGraph);
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
    auto calls = buildableCalls(handle, opGraph);
    if(!calls.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "explicit-GEMM convolution does not apply to this graph");
    }
    HIPDNN_PLUGIN_LOG_INFO("Building explicit-GEMM convolution plan, row "
                           << toString(calls->row) << ", " << calls->stages.size() << " stage(s), "
                           << calls->workspaceBytes << " workspace bytes, epilogue "
                           << epilogueFormName(calls->epilogueForm));

    // Benchmarking: when the plan has a choice of layout (an IM2COL column matrix; a 1x1
    // channels-first backward weights' batched partials or channels-last staging), of a
    // backward weights' reduction split, or of where a fused epilogue runs, time every distinct
    // one on the first execute and keep the fastest: none wins every shape.
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
        // A GEMM epilogue against the same epilogue in a pass, where the GEMM's own may cost
        // more than the pass's traffic: a broadcast-bias addend (one more y-sized read a
        // GEMM, and its fill) on a bandwidth-bound layer. A bias along the rows costs the GEMM
        // nothing, and the clamp tail moves fewer bytes than the pass: never a candidate.
        if(calls->epilogueForm == EpilogueForm::NATIVE && usesAddend(*calls))
        {
            auto pass = callsFor(opGraph, _layout, _splitReduction, false, false);
            if(pass.has_value())
            {
                try
                {
                    candidates.push_back({"epilogue_pass", planFor(handle, std::move(*pass))});
                }
                catch(const std::exception& e)
                {
                    HIPDNN_PLUGIN_LOG_INFO(
                        "Explicit-GEMM candidate epilogue_pass skipped: " << e.what());
                }
            }
        }
        // Staged channels-last: one GEMM over every image of a channels-first y, where the
        // other layouts batch a small per-image GEMM (1x1 stride 2: 25 x 25 outputs an image).
        // Image slices: a 16-bit backward weights over the cap summed through an fp32
        // accumulator instead of blocks of w's columns (the default), where the backend runs
        // its fp32-output GEMMs well.
        for(const auto& [layout, name] :
            {std::pair{ColumnLayout::CHANNELS_FIRST, "channels_first"},
             std::pair{ColumnLayout::CHANNELS_LAST, "channels_last"},
             std::pair{ColumnLayout::STAGED_CHANNELS_LAST, "staged_channels_last"},
             std::pair{ColumnLayout::IMAGE_SLICES, "image_slices"}})
        {
            auto other = callsFor(opGraph, layout);
            if(other.has_value()
               && std::find(seen.begin(), seen.end(), signature(*other)) == seen.end())
            {
                seen.push_back(signature(*other));
                // A layout the backend cannot run (a fused fp32-output GEMM it has no algorithm
                // for, a GEMM it declines) is not a candidate.
                try
                {
                    candidates.push_back({name, planFor(handle, std::move(*other))});
                }
                catch(const std::exception& e)
                {
                    HIPDNN_PLUGIN_LOG_INFO("Explicit-GEMM candidate " << name
                                                                      << " skipped: " << e.what());
                }
            }
        }
        // A reduction split into enough fp32 partials to fill the GPU: the libraries' own
        // long-k kernels leave a small output's few tiles on a few CUs (backward weights only).
        auto split = callsFor(opGraph, ColumnLayout::FOLLOW_Y, true);
        if(split.has_value()
           && std::find(seen.begin(), seen.end(), signature(*split)) == seen.end())
        {
            seen.push_back(signature(*split));
            try
            {
                candidates.push_back({"split_reduction", planFor(handle, std::move(*split))});
            }
            catch(const std::exception& e)
            {
                HIPDNN_PLUGIN_LOG_INFO(
                    "Explicit-GEMM candidate split_reduction skipped: " << e.what());
            }
        }
        // Backward weights under the larger cap: fewer image slices, each an accumulating GEMM
        // (and, for 16-bit w, an fp32 accumulator), against the default cap's smaller column
        // matrices; for a 16-bit w over the cap, fewer blocks of w's columns, and fewer image
        // slices too. Neither wins everywhere: the same 7x7 fp32 stem runs 4x faster wide on
        // gfx906 and 12% slower on MI325X; wide image slices run DeepLab's channels-last ASPP
        // layers 7-9% faster than column blocks on MI325X, and 4-6x slower on gfx11.
        for(const auto& [layout, name] :
            {std::pair{ColumnLayout::FOLLOW_Y, "wide_backward_weights"},
             std::pair{ColumnLayout::IMAGE_SLICES, "wide_image_slices"}})
        {
            auto wide = callsFor(opGraph, layout, false, true);
            if(!wide.has_value()
               || std::find(seen.begin(), seen.end(), signature(*wide)) != seen.end())
            {
                continue;
            }
            seen.push_back(signature(*wide));
            try
            {
                candidates.push_back({name, planFor(handle, std::move(*wide))});
            }
            catch(const std::exception& e)
            {
                HIPDNN_PLUGIN_LOG_INFO("Explicit-GEMM candidate " << name
                                                                  << " skipped: " << e.what());
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
    gemmPlans.reserve(calls.stages.size());
    for(auto& stage : calls.stages)
    {
        gemmPlans.push_back(stage.calls.empty() ? nullptr : _factory(handle, stage.calls));
    }
    return std::make_unique<ConvRepackPlan>(std::move(calls), std::move(gemmPlans));
}

} // namespace hipblaslt_plugin
