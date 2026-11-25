// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CompileCounters.hpp"

#include "ProgramFetchRecording.hpp"

#include <array>
#include <atomic>
#include <cstdio>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::compilation
{

namespace
{

struct Totals
{
    std::atomic<std::uint64_t> compiles{0};
    std::atomic<std::uint64_t> compileMicroseconds{0};
    std::atomic<std::uint64_t> compileFailures{0};
    std::atomic<std::uint64_t> diskHits{0};
    std::atomic<std::uint64_t> corruptEntries{0};
    std::atomic<std::uint64_t> inProcessHits{0};
    std::atomic<std::uint64_t> storeHits{0};
};

/// Never destroyed, so a count taken during static destruction stays valid.
Totals& totals()
{
    static auto* s_totals = new Totals();
    return *s_totals;
}

} // namespace

CompileCounters compileCounters()
{
    auto& all = totals();
    CompileCounters counters;
    counters.compiles = all.compiles.load();
    counters.compileSeconds = static_cast<double>(all.compileMicroseconds.load()) / 1e6;
    counters.compileFailures = all.compileFailures.load();
    counters.diskHits = all.diskHits.load();
    counters.corruptEntries = all.corruptEntries.load();
    counters.inProcessHits = all.inProcessHits.load();
    counters.storeHits = all.storeHits.load();
    return counters;
}

void countFetch(int source, double milliseconds)
{
    auto& all = totals();
    switch(static_cast<CodeSource>(source))
    {
    case CodeSource::COMPILED:
        ++all.compiles;
        all.compileMicroseconds += static_cast<std::uint64_t>(milliseconds * 1000.0);
        break;
    case CodeSource::DISK_CACHE:
        ++all.diskHits;
        break;
    case CodeSource::IN_PROCESS:
        ++all.inProcessHits;
        break;
    case CodeSource::PRECOMPILED_STORE:
        ++all.storeHits;
        break;
    default:
        break;
    }
}

void countCompileFailure()
{
    ++totals().compileFailures;
}

void countCorruptEntry()
{
    ++totals().corruptEntries;
}

std::string formatCompileCounters(const CompileCounters& counters)
{
    std::array<char, 32> seconds{};
    std::snprintf(seconds.data(), seconds.size(), "%.2f s", counters.compileSeconds);
    return "hipRTC: " + std::to_string(counters.compiles) + " compiled (" + seconds.data() + "), "
           + std::to_string(counters.compileFailures) + " failed, "
           + std::to_string(counters.diskHits) + " disk-cache hits, "
           + std::to_string(counters.corruptEntries) + " corrupt entries replaced, "
           + std::to_string(counters.inProcessHits) + " in-process hits, "
           + std::to_string(counters.storeHits) + " store hits";
}

void logCompileCounters()
{
    const auto counters = compileCounters();
    if(counters.compiles + counters.compileFailures + counters.diskHits + counters.inProcessHits
           + counters.storeHits
       == 0)
    {
        return;
    }
    HIPDNN_PLUGIN_LOG_INFO(formatCompileCounters(counters));
}

} // namespace hip_kernel_provider::compilation
