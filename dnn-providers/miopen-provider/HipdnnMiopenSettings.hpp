// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

/**
 * @brief MIOpen plugin-specific execution settings.
 *
 * This structure holds settings that control MIOpen execution behavior,
 * such as benchmarking mode and workspace size limits.
 */
struct HipdnnMiopenSettings
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
     * @param limit Maximum workspace size in bytes.
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
