// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend/Graph.hpp>

#include "harness/Capture.hpp"
#include "harness/PlanWorkspace.hpp"
#include "harness/VariantPack.hpp"

namespace hipdnn_integration_tests
{

namespace detail
{

class HipGraph
{
public:
    HipGraph() = default;
    ~HipGraph()
    {
        if(_graph != nullptr)
        {
            static_cast<void>(hipGraphDestroy(_graph));
        }
    }

    HipGraph(const HipGraph&) = delete;
    HipGraph& operator=(const HipGraph&) = delete;
    HipGraph(HipGraph&&) = delete;
    HipGraph& operator=(HipGraph&&) = delete;

    hipGraph_t* out()
    {
        return &_graph;
    }

    hipGraph_t get() const
    {
        return _graph;
    }

private:
    hipGraph_t _graph = nullptr;
};

class HipGraphExec
{
public:
    HipGraphExec() = default;
    ~HipGraphExec()
    {
        if(_exec != nullptr)
        {
            static_cast<void>(hipGraphExecDestroy(_exec));
        }
    }

    HipGraphExec(const HipGraphExec&) = delete;
    HipGraphExec& operator=(const HipGraphExec&) = delete;
    HipGraphExec(HipGraphExec&&) = delete;
    HipGraphExec& operator=(HipGraphExec&&) = delete;

    hipGraphExec_t* out()
    {
        return &_exec;
    }

    hipGraphExec_t get() const
    {
        return _exec;
    }

private:
    hipGraphExec_t _exec = nullptr;
};

inline std::string hipFailure(const char* step, hipError_t error)
{
    return std::string(step) + " failed: " + hipGetErrorName(error) + " ("
           + hipGetErrorString(error) + ")";
}

/// Leaves `stream` out of capture, whatever state a failed step left it in. A stream
/// that stays capturing would fail every later test that shares the handle.
///
/// Any state but None is ended, not only Active: a capture an engine invalidated
/// (an unsafe call, a sync) is still open until hipStreamEndCapture() closes it, and
/// that call then reports the invalidation. The error is expected, and is cleared so
/// it does not surface as the failure of the next HIP call.
inline void abandonCapture(hipStream_t stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if(hipStreamIsCapturing(stream, &status) == hipSuccess && status != hipStreamCaptureStatusNone)
    {
        hipGraph_t discarded = nullptr;
        static_cast<void>(hipStreamEndCapture(stream, &discarded));
        if(discarded != nullptr)
        {
            static_cast<void>(hipGraphDestroy(discarded));
        }
    }
    static_cast<void>(hipGetLastError());
}

} // namespace detail

/// Captures one execution of the plan `graph` already built into a HIP graph on the
/// handle's stream, then replays it `settings.replays` times.
///
/// The engine must report SUPPORTS_GRAPH_CAPTURE, otherwise nothing is captured and
/// the result is a decline. Everything that allocates (the plan workspace) happens
/// before hipStreamBeginCapture, and the outputs are reset in `hooks` outside capture
/// before every replay, so a replay that writes nothing cannot pass on the eager
/// run's leftovers.
inline CaptureResult captureGraph(hipdnn_frontend::graph::Graph& graph,
                                  hipdnnHandle_t handle,
                                  VariantPack& variantPack,
                                  const CaptureSettings& settings,
                                  const ReplayHooks& hooks)
{
    int64_t engineId = -1;
    if(auto status = graph.get_execution_plan_engine_id(engineId); !status.is_good())
    {
        return CaptureResult::failed("get_execution_plan_engine_id failed: "
                                     + status.get_message());
    }

    std::vector<hipdnn_frontend::BehaviorNote> notes;
    if(auto status = graph.get_behavior_notes_for_engine(engineId, notes); !status.is_good())
    {
        return CaptureResult::failed("get_behavior_notes_for_engine failed: "
                                     + status.get_message());
    }
    if(std::find(notes.begin(), notes.end(), hipdnn_frontend::BehaviorNote::SUPPORTS_GRAPH_CAPTURE)
       == notes.end())
    {
        return declinedForMissingNote(engineId);
    }

    std::string error;
    auto workspace = allocatePlanWorkspace(graph, error);
    if(!workspace.has_value())
    {
        return CaptureResult::failed("workspace allocation failed: " + error);
    }

    hipStream_t stream = nullptr;
    if(hipdnnGetStream(handle, &stream) != HIPDNN_STATUS_SUCCESS)
    {
        return CaptureResult::failed("hipdnnGetStream failed");
    }

    if(const auto status = hipStreamSynchronize(stream); status != hipSuccess)
    {
        return CaptureResult::failed(detail::hipFailure("hipStreamSynchronize", status));
    }

    detail::HipGraph captured;
    if(const auto status = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
       status != hipSuccess)
    {
        return CaptureResult::failed(detail::hipFailure("hipStreamBeginCapture", status));
    }

    std::string executeError;
    try
    {
        if(auto status = graph.execute(handle, variantPack, workspace->get()); !status.is_good())
        {
            executeError = status.get_message();
        }
    }
    catch(const std::exception& e)
    {
        executeError = e.what();
    }
    if(!executeError.empty())
    {
        detail::abandonCapture(stream);
        return CaptureResult::failed("graph.execute during stream capture failed: " + executeError);
    }

    if(const auto status = hipStreamEndCapture(stream, captured.out()); status != hipSuccess)
    {
        detail::abandonCapture(stream);
        return CaptureResult::failed(detail::hipFailure("hipStreamEndCapture", status));
    }

    detail::HipGraphExec exec;
    if(const auto status = hipGraphInstantiateWithFlags(exec.out(), captured.get(), 0);
       status != hipSuccess)
    {
        return CaptureResult::failed(detail::hipFailure("hipGraphInstantiateWithFlags", status));
    }

    for(int replay = 0; replay < settings.replays; ++replay)
    {
        try
        {
            hooks.resetOutputs();
        }
        catch(const std::exception& e)
        {
            return CaptureResult::failed(std::string("resetting outputs before replay ")
                                         + std::to_string(replay) + " failed: " + e.what());
        }
        if(const auto status = hipDeviceSynchronize(); status != hipSuccess)
        {
            return CaptureResult::failed(detail::hipFailure("hipDeviceSynchronize", status));
        }
        if(const auto status = hipGraphLaunch(exec.get(), stream); status != hipSuccess)
        {
            return CaptureResult::failed(detail::hipFailure("hipGraphLaunch", status));
        }
        if(const auto status = hipStreamSynchronize(stream); status != hipSuccess)
        {
            return CaptureResult::failed(detail::hipFailure("hipStreamSynchronize", status));
        }

        if(!hooks.checkOutputs())
        {
            auto result = CaptureResult::replayed();
            result.outputsMismatched = true;
            return result;
        }
    }
    return CaptureResult::replayed();
}

} // namespace hipdnn_integration_tests
