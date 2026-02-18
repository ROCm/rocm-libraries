// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

/**
 * @file HipdnnEngineSpecificSettings.hpp
 * @brief MIOpen plugin's implementation of HipdnnEngineSpecificSettings.
 *
 * This type is forward-declared in the plugin SDK's IPlanBuilder interface.
 * Each plugin must define this structure to hold plugin-specific execution settings.
 */

struct HipdnnEngineSpecificSettings
{
    void setBenchmarkingEnabled(bool enabled)
    {
        _benchmarkingEnabled = enabled;
    }

    bool benchmarkingEnabled() const
    {
        return _benchmarkingEnabled;
    }

    /**
     * @brief Sets the workspace size limit for MIOpen operations.
     *
     * Constrains GPU workspace memory (in bytes) used by MIOpen convolution algorithms.
     *
     * @param limit Maximum workspace size in bytes. Must be within the range returned by
     *              MiopenConvPlanBuilder::getWorkspaceSizeRange() for the specific operation.
     *
     * @note If not set (std::nullopt), uses MIOpen's default workspace size.
     *       Smaller limits may reduce performance but save GPU memory.
     */
    void setWorkspaceSizeLimit(size_t limit)
    {
        _workspaceSizeLimit = limit;
    }

    /**
     * @brief Gets the current workspace size limit.
     *
     * @return Workspace size limit in bytes if set, std::nullopt otherwise.
     */
    std::optional<size_t> workspaceSizeLimit() const
    {
        return _workspaceSizeLimit;
    }

private:
    bool _benchmarkingEnabled = false;
    std::optional<size_t> _workspaceSizeLimit;
};
