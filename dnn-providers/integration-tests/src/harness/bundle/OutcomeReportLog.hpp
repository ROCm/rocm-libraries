// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string_view>

namespace hipdnn_integration_tests::bundle
{

/// The process-wide JSON Lines sink behind --outcome-report.
///
/// One line per call, flushed before it returns, so a crash keeps every earlier line:
/// the driver reads a `started` with no `finished` as the case that killed the process.
/// A no-op until open() has been called, which is every run that did not ask for a
/// report.
class OutcomeReportLog
{
public:
    static OutcomeReportLog& get();

    OutcomeReportLog(const OutcomeReportLog&) = delete;
    OutcomeReportLog& operator=(const OutcomeReportLog&) = delete;
    OutcomeReportLog(OutcomeReportLog&&) = delete;
    OutcomeReportLog& operator=(OutcomeReportLog&&) = delete;

    /// Opens `path` for appending, creating it if absent. Returns false, and leaves the
    /// sink closed, when the file cannot be opened. Reopening closes the previous file.
    bool open(const std::filesystem::path& path);

    /// Closes the file; later appends are no-ops again.
    void close();

    bool isOpen() const;

    /// Writes `line` and a newline, then flushes. Never throws: a report that cannot be
    /// written must not turn a passing test into a failing one.
    void append(std::string_view line) noexcept;

private:
    OutcomeReportLog() = default;
    ~OutcomeReportLog() = default;

    mutable std::mutex _mutex;
    std::ofstream _file;
    bool _writeFailureReported = false;
};

} // namespace hipdnn_integration_tests::bundle
