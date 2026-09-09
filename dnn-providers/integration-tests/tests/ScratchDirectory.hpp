// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

// getpid() below stamps the temp path per process. MSVC ships no <unistd.h>;
// it spells the same call _getpid() in <process.h>.
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

// Deliberately not `testing_support`: HarnessTestSupport.hpp already owns that name
// under `bundle::`, and two namespaces spelled the same would make every call site
// ambiguous to read even where it compiles.
namespace hipdnn_integration_tests::scratch
{

/// This process's id. MSVC has no <unistd.h> and spells the call _getpid().
inline int currentProcessId()
{
#ifdef _WIN32
    return _getpid();
#else
    return ::getpid();
#endif
}

/// Claims a uniquely-named scratch directory under the system temp path.
///
/// Seeded from the clock, the pid and a per-call counter, so neither a sibling
/// process nor a second call in this one draws the same name. That matters because
/// this binary's suites are run concurrently -- ctest -j, or CI running the unit
/// target for two configurations -- and every one of them shares
/// temp_directory_path().
///
/// A name keyed only by source line collides across those runs, and the
/// remove_all() such a scheme needs in order to reuse the name deletes the other
/// run's fixture mid-test. Nothing here removes a path it did not create.
///
/// ScopedDirectory itself creates the directory and throws when the name is already
/// taken, which is the property that makes this safe: a lost race is retried rather
/// than adopted, and the returned object still owns exactly what it created.
inline hipdnn_test_sdk::utilities::ScopedDirectory makeDir(std::string_view prefix)
{
    static std::atomic<uint64_t> s_counter{0};
    const auto base = std::filesystem::temp_directory_path();
    const auto seed
        = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())
          ^ (static_cast<uint64_t>(currentProcessId()) << 32U);

    for(int attempt = 0; attempt < 64; ++attempt)
    {
        const auto candidate
            = base / (std::string(prefix) + std::to_string(seed + s_counter.fetch_add(1)));
        try
        {
            return {candidate};
        }
        catch(const std::runtime_error&)
        {
            // Name taken by a concurrent run. Draw the next one -- the counter has
            // already advanced, so this cannot spin on the same candidate.
            continue;
        }
    }
    throw std::runtime_error("scratch::makeDir: no free temp directory name after 64 attempts");
}

} // namespace hipdnn_integration_tests::scratch
