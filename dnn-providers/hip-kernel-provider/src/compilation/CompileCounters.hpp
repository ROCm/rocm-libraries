// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

namespace hip_kernel_provider::compilation
{

/// Process-wide totals of how the provider obtained HIPRTC code objects.
struct CompileCounters
{
    std::uint64_t compiles = 0; ///< Programs compiled with HIPRTC.
    double compileSeconds = 0.0; ///< Wall time of those compiles, load and cache write included.
    std::uint64_t compileFailures = 0; ///< Compiles HIPRTC rejected or that could not run.
    std::uint64_t diskHits = 0; ///< Programs loaded from the HIPRTC disk cache.
    std::uint64_t corruptEntries = 0; ///< Damaged disk-cache entries recompiled and replaced.
    std::uint64_t inProcessHits = 0; ///< Lookups served by a module this process already held.
    std::uint64_t storeHits = 0; ///< Programs loaded from a precompiled store.
};

/// The totals so far.
CompileCounters compileCounters();

/// Counts one program obtained: by @p source (see CodeSource), in @p milliseconds.
void countFetch(int source, double milliseconds);
void countCompileFailure();
void countCorruptEntry();

/// One line for the log: "hipRTC: 3 compiled (2.41 s), 0 failed, 12 disk-cache hits,
/// 0 corrupt entries replaced, 40 in-process hits, 0 store hits".
std::string formatCompileCounters(const CompileCounters& counters);

/// Logs formatCompileCounters() at INFO when the process has obtained any program.
void logCompileCounters();

} // namespace hip_kernel_provider::compilation
