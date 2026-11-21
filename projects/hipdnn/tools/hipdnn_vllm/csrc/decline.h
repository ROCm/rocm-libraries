// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Decline taxonomy. Mirrors hipdnn_vllm/declines.py: the reason
// string is STAGE or STAGE:detail. Backend error text goes in `message` and is
// recorded, never parsed.

#pragma once

#include <string>
#include <utility>

namespace hipdnn_vllm
{

enum class Stage
{
    NO_BUILD,
    NO_ENGINE,
    CHECK_SUPPORT,
    BUILD_PLANS,
    UNSUPPORTED_FEATURE,
    DISABLED,
    CAPTURE_INVARIANT,
    EXECUTE,
};

inline const char* stageName(Stage stage)
{
    switch(stage)
    {
    case Stage::NO_BUILD:
        return "NO_BUILD";
    case Stage::NO_ENGINE:
        return "NO_ENGINE";
    case Stage::CHECK_SUPPORT:
        return "CHECK_SUPPORT";
    case Stage::BUILD_PLANS:
        return "BUILD_PLANS";
    case Stage::UNSUPPORTED_FEATURE:
        return "UNSUPPORTED_FEATURE";
    case Stage::DISABLED:
        return "DISABLED";
    case Stage::CAPTURE_INVARIANT:
        return "CAPTURE_INVARIANT";
    case Stage::EXECUTE:
        return "EXECUTE";
    }
    return "UNKNOWN";
}

struct Decline
{
    Stage stage;
    std::string detail;
    std::string message;

    std::string reason() const
    {
        std::string out = stageName(stage);
        if(!detail.empty())
        {
            out += ':';
            out += detail;
        }
        return out;
    }

    static Decline feature(std::string name, std::string message = {})
    {
        return Decline{
            .stage = Stage::UNSUPPORTED_FEATURE, .detail = std::move(name), .message = std::move(message)};
    }
};

} // namespace hipdnn_vllm
