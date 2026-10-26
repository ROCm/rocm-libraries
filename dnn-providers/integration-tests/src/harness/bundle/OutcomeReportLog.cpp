// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/OutcomeReportLog.hpp"

#include <exception>
#include <ios>
#include <iostream>

namespace hipdnn_integration_tests::bundle
{

OutcomeReportLog& OutcomeReportLog::get()
{
    static OutcomeReportLog instance;
    return instance;
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
    try
    {
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
    catch(const std::exception&)
    {
        // Deliberately silent: see the header.
    }
}

} // namespace hipdnn_integration_tests::bundle
