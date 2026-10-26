// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/OutcomeReportLog.hpp"

#include <ios>
#include <iostream>

namespace hipdnn_integration_tests::bundle
{

OutcomeReportLog& OutcomeReportLog::get()
{
    static OutcomeReportLog s_instance;
    return s_instance;
}

bool OutcomeReportLog::open(const std::filesystem::path& path)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    if(_file.is_open())
    {
        _file.close();
    }
    _file.clear();
    _writeFailureReported = false;

    _file.open(path, std::ios::out | std::ios::app);
    return _file.is_open();
}

void OutcomeReportLog::close()
{
    const std::lock_guard<std::mutex> lock(_mutex);
    if(_file.is_open())
    {
        _file.close();
    }
    _file.clear();
}

bool OutcomeReportLog::isOpen() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _file.is_open();
}

void OutcomeReportLog::append(std::string_view line) noexcept
{
    // Nothing below throws: the stream was not asked to (no exceptions mask), so a failed
    // write sets a flag, reported once, instead of an exception that would fail a test.
    const std::lock_guard<std::mutex> lock(_mutex);
    if(!_file.is_open())
    {
        return;
    }

    _file.write(line.data(), static_cast<std::streamsize>(line.size()));
    _file.put('\n');
    _file.flush();

    if(!_file.good() && !_writeFailureReported)
    {
        _writeFailureReported = true;
        std::cerr << "WARNING: writing the outcome report failed; later records may be "
                     "missing.\n";
    }
}

} // namespace hipdnn_integration_tests::bundle
