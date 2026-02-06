// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

namespace miopen_plugin
{

struct MiopenExecutionSettings
{
    enum class DebugMode
    {
        NONE,                          // Debug mode disabled (default)
        LOG_ALL_FOUND_PLAN_ALGORITHMS  // Log all found plan algorithms
    };

    void setBenchmarkingEnabled(bool enabled)
    {
        _benchmarkingEnabled = enabled;
    }

    bool benchmarkingEnabled() const
    {
        return _benchmarkingEnabled;
    }

    void setWorkspaceSizeLimit(size_t limit)
    {
        _workspaceSizeLimit = limit;
    }

    std::optional<size_t> workspaceSizeLimit() const
    {
        return _workspaceSizeLimit;
    }

    void setDebugMode(DebugMode mode)
    {
        _debugMode = mode;
    }

    DebugMode debugMode() const
    {
        return _debugMode;
    }

private:
    bool _benchmarkingEnabled = false;
    std::optional<size_t> _workspaceSizeLimit;
    DebugMode _debugMode = DebugMode::NONE;
};

} // namespace miopen_plugin
