// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hip_kernel_provider::compilation
{

/// Where a plan build got the code object of one program.
enum class CodeSource
{
    IN_PROCESS, ///< Already loaded by this process.
    DISK_CACHE, ///< Loaded from the HIPRTC disk cache.
    COMPILED, ///< Compiled with HIPRTC.
    PRECOMPILED_STORE, ///< Loaded from a precompiled code-object store shipped with the provider.
};

/// The name a log line uses for @p source: "in-process", "disk-cache", "compiled" or "store".
std::string_view codeSourceName(CodeSource source);

/// One program a plan build obtained.
struct ProgramFetch
{
    std::string program;
    CodeSource source = CodeSource::COMPILED;
    double milliseconds = 0.0;
};

/// Collects, on the calling thread, every program obtained while it is alive: a plan build
/// opens one, and the compiler records into the innermost one open on its thread. Recordings
/// nest; one opened inside another hides it until it closes.
class ProgramFetchRecording
{
public:
    ProgramFetchRecording();
    ~ProgramFetchRecording();

    ProgramFetchRecording(const ProgramFetchRecording&) = delete;
    ProgramFetchRecording& operator=(const ProgramFetchRecording&) = delete;
    ProgramFetchRecording(ProgramFetchRecording&&) = delete;
    ProgramFetchRecording& operator=(ProgramFetchRecording&&) = delete;

    const std::vector<ProgramFetch>& fetches() const
    {
        return _fetches;
    }

    /// Adds @p fetch to the innermost recording open on this thread, if any, and to the
    /// process counters (CompileCounters.hpp).
    static void record(ProgramFetch fetch);

    /// How many fetches this thread has recorded since it started, open recording or not. A
    /// caller compares it before and after a lookup to tell whether the lookup recorded one.
    static std::uint64_t recordedOnThisThread();

private:
    std::vector<ProgramFetch> _fetches;
    ProgramFetchRecording* _outer;
};

/// The one INFO line a plan build logs: the engine, the config it chose, each program with
/// where its code object came from and how long that took, and the total compile time.
/// @p config is empty when the engine has no config name, and @p autotune says the config
/// is chosen by benchmarking at the first execution.
std::string formatPlanBuildSummary(std::string_view engine,
                                   std::string_view config,
                                   bool autotune,
                                   const std::vector<ProgramFetch>& fetches);

} // namespace hip_kernel_provider::compilation
