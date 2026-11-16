// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ProgramFetchRecording.hpp"

#include "CompileCounters.hpp"

#include <array>
#include <cstdio>
#include <utility>

namespace hip_kernel_provider::compilation
{

namespace
{

/// The innermost recording open on this thread, or null.
ProgramFetchRecording*& innermost()
{
    thread_local ProgramFetchRecording* s_innermost = nullptr;
    return s_innermost;
}

/// Where the innermost recording on this thread collects, or null.
std::vector<ProgramFetch>*& sink()
{
    thread_local std::vector<ProgramFetch>* s_sink = nullptr;
    return s_sink;
}

std::uint64_t& recorded()
{
    thread_local std::uint64_t s_recorded = 0;
    return s_recorded;
}

std::string milliseconds(double value)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.1f ms", value);
    return text.data();
}

} // namespace

std::string_view codeSourceName(CodeSource source)
{
    switch(source)
    {
    case CodeSource::IN_PROCESS:
        return "in-process";
    case CodeSource::DISK_CACHE:
        return "disk-cache";
    case CodeSource::COMPILED:
        return "compiled";
    case CodeSource::PRECOMPILED_STORE:
        return "store";
    default:
        break;
    }
    return "unknown";
}

ProgramFetchRecording::ProgramFetchRecording()
    : _outer(innermost())
{
    innermost() = this;
    sink() = &_fetches;
}

ProgramFetchRecording::~ProgramFetchRecording()
{
    innermost() = _outer;
    sink() = _outer != nullptr ? &_outer->_fetches : nullptr;
}

void ProgramFetchRecording::record(ProgramFetch fetch)
{
    ++recorded();
    countFetch(static_cast<int>(fetch.source), fetch.milliseconds);
    if(sink() != nullptr)
    {
        sink()->push_back(std::move(fetch));
    }
}

std::uint64_t ProgramFetchRecording::recordedOnThisThread()
{
    return recorded();
}

std::string formatPlanBuildSummary(std::string_view engine,
                                   std::string_view config,
                                   bool autotune,
                                   const std::vector<ProgramFetch>& fetches)
{
    std::string line = "plan build: engine '";
    line.append(engine);
    line += "', config ";
    if(autotune)
    {
        line += "chosen by autotune at first execution";
    }
    else if(config.empty())
    {
        line += "-";
    }
    else
    {
        line += "'";
        line.append(config);
        line += "'";
    }
    line += ", " + std::to_string(fetches.size()) + " program(s)";

    double compileMilliseconds = 0.0;
    const char* separator = ": ";
    for(const auto& fetch : fetches)
    {
        line += separator;
        line += fetch.program;
        line += " ";
        line.append(codeSourceName(fetch.source));
        line += " ";
        line += milliseconds(fetch.milliseconds);
        separator = ", ";
        if(fetch.source == CodeSource::COMPILED)
        {
            compileMilliseconds += fetch.milliseconds;
        }
    }
    line += "; compile " + milliseconds(compileMilliseconds);
    return line;
}

} // namespace hip_kernel_provider::compilation
