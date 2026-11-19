// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

// getpid() below stamps the temp path per process. MSVC ships no <unistd.h>;
// it spells the same call _getpid() in <process.h>.
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

inline int currentProcessId()
{
#ifdef _WIN32
    return _getpid();
#else
    return ::getpid();
#endif
}

/// Claims a uniquely-named scratch directory under @p base.
///
/// Suites sharing temp_directory_path() run concurrently -- ctest -j, or CI running a unit
/// target for two configurations -- so the name has to be unique per run.
///
/// ScopedDirectory itself creates the directory and throws when the name is already taken:
/// a lost race is retried rather than adopted, and create_directory is atomic, so testing a
/// name and taking it are one step. Retry rather than clear: the name may belong to a live
/// process whose fixture would go with it.
///
/// `label` names the calling suite, so a directory left behind by a crash says which binary
/// made it.
///
/// Callers want claimScratchDirectory() below. This form exists so a test can name an
/// unusable base directly: the env vars temp_directory_path() consults are advisory, and
/// Windows ignores them outright for a process running under a service account.
[[nodiscard]] inline ScopedDirectory claimScratchDirectoryUnder(const std::filesystem::path& base,
                                                                const std::string& label)
{
    // Drawn once for the process, so concurrent runs start from different names rather than
    // both walking up from zero. The pid is not redundant: where random_device is
    // deterministic, two sibling processes would otherwise draw the same session value.
    static const uint64_t s_session = (static_cast<uint64_t>(std::random_device{}()) << 32U)
                                      ^ static_cast<uint64_t>(currentProcessId());
    static std::atomic<uint64_t> s_counter{0};

    std::ostringstream prefix;
    prefix << "hipdnn_test_" << label << '_' << std::hex << s_session << '_';

    for(int attempt = 0; attempt < 64; ++attempt)
    {
        const std::filesystem::path candidate
            = base / (prefix.str() + std::to_string(s_counter.fetch_add(1)));
        try
        {
            return {candidate};
        }
        // filesystem_error derives from runtime_error, so it has to be caught first: an
        // unwritable temp directory fails identically 64 times and must not be reported as
        // name exhaustion.
        catch(const std::filesystem::filesystem_error&)
        {
            throw;
        }
        catch(const std::runtime_error&)
        {
            // Name taken. The counter has advanced, so retrying cannot spin on this name.
            continue;
        }
    }
    throw std::runtime_error("claimScratchDirectory: no free scratch name under the temp dir");
}

/// Claims a scratch directory under the system temp path. See claimScratchDirectoryUnder().
[[nodiscard]] inline ScopedDirectory claimScratchDirectory(const std::string& label)
{
    return claimScratchDirectoryUnder(std::filesystem::temp_directory_path(), label);
}

} // namespace hipdnn_test_sdk::utilities
