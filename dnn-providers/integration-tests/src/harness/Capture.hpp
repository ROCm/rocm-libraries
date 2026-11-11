// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace hipdnn_integration_tests
{

/// How --graph-capture replays one verified graph.
struct CaptureSettings
{
    static constexpr int DEFAULT_REPLAYS = 3;

    int replays = DEFAULT_REPLAYS;
};

/// The --graph-capture flag as typed, plus the flags that conflict with it.
struct CaptureRequest
{
    bool enabled = false;
    bool benchmark = false;
    bool skipGraphValidation = false;
    bool captureBundles = false;
    bool writeSupportClaims = false;
    bool bundlesDisabled = false;
};

struct CaptureResolution
{
    std::optional<CaptureSettings> settings;
    std::optional<std::string> error;
};

/// Settings for --graph-capture, or the reason the flags cannot be used together.
///
/// --benchmark is refused because Graph::autotune() synchronizes and times with
/// events, which cannot run in a capturing stream; the two lanes are separate runs.
/// The remaining modes stop a test before it executes, which would leave the capture
/// lane checking nothing and the run green.
inline CaptureResolution resolveCapture(const CaptureRequest& request)
{
    if(!request.enabled)
    {
        return {};
    }

    if(request.benchmark)
    {
        return {std::nullopt,
                "--graph-capture cannot be combined with --benchmark: autotune "
                "synchronizes and times with events, which a capturing stream forbids. "
                "Run the two lanes separately"};
    }
    if(request.skipGraphValidation || request.captureBundles || request.writeSupportClaims)
    {
        return {std::nullopt,
                "--graph-capture cannot be combined with --skip-graph-validation, "
                "--capture-bundles or --write-support-claims: those modes do not "
                "execute the graph"};
    }
    if(request.bundlesDisabled)
    {
        return {std::nullopt,
                "--graph-capture cannot be combined with --no-bundles or "
                "HIPDNN_TEST_ALLOW_BUNDLES=0: only bundle tests are captured, so no "
                "test would run the lane"};
    }
    return {CaptureSettings{}, std::nullopt};
}

/// What captureGraph() asks of the harness around each replay. Both run outside
/// capture, on the host thread.
struct ReplayHooks
{
    /// Puts the output buffers back to the sentinel, in place: a captured graph has
    /// the buffer addresses baked in.
    std::function<void()> resetOutputs;
    /// Verifies the outputs the replay just wrote. False stops the replays.
    std::function<bool()> checkOutputs;
};

/// What one capture-and-replay of a verified graph came to.
struct CaptureResult
{
    enum class Status
    {
        REPLAYED,
        /// The engine does not report SUPPORTS_GRAPH_CAPTURE, so nothing was captured.
        DECLINED,
        /// A HIP step failed or the engine could not be captured; `message` names it.
        FAILED,
    };

    Status status = Status::FAILED;
    std::string message;
    /// Set when the replayed outputs differed from the oracle. The mismatches are
    /// already on the record, so this carries no message.
    bool outputsMismatched = false;

    static CaptureResult replayed()
    {
        return {Status::REPLAYED, {}, false};
    }

    static CaptureResult declined(std::string message)
    {
        return {Status::DECLINED, std::move(message), false};
    }

    static CaptureResult failed(std::string message)
    {
        return {Status::FAILED, std::move(message), false};
    }
};

/// The decline for an engine that lacks SUPPORTS_GRAPH_CAPTURE: its name where one
/// is registered (its hex ID otherwise), then the ID the plan reported. The same
/// string is the [ CAPTURE ] line and the skip message, so the two cannot disagree.
inline CaptureResult declinedForMissingNote(int64_t engineId)
{
    return CaptureResult::declined("engine " + hipdnn_data_sdk::utilities::engineNameOrHex(engineId)
                                   + " (id " + std::to_string(engineId)
                                   + ") does not report the SUPPORTS_GRAPH_CAPTURE note");
}

} // namespace hipdnn_integration_tests
