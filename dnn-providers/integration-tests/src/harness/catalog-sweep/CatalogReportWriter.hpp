// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

namespace hipdnn_integration_tests::catalog_sweep
{

/// The JSON Lines report: one compact JSON object per line, appended.
///
/// Opened in append mode and never truncated, so a report from an earlier shard or an
/// interrupted run keeps its lines. Every record is flushed as it is written: a sweep
/// can run for hours and be killed, and what it had already finished must survive.
class CatalogReportWriter
{
public:
    /// @throws std::runtime_error naming the path when it cannot be opened for append.
    /// Opened (and created) up front so a bad path stops the run before any work.
    explicit CatalogReportWriter(const std::filesystem::path& path);

    /// Appends one line and flushes it.
    /// @throws std::runtime_error when the line could not be written.
    void append(const nlohmann::json& record);

    const std::filesystem::path& path() const
    {
        return _path;
    }

private:
    std::filesystem::path _path;
    std::ofstream _stream;
};

} // namespace hipdnn_integration_tests::catalog_sweep
