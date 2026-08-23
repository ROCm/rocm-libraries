// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>

namespace hipdnn_test_sdk::utilities
{

/// Redirects `HIPDNN_CACHE_DIR` at a process-private scratch directory for this
/// object's lifetime, and removes it on destruction.
///
/// Why a test binary needs this: hipDNN's on-disk caches (`ingestor-winners/`, the
/// kernel ingestor's benchmarked ranking, and `autotune-rankings/`, the exact-match
/// engine ranking) default to the *developer's* real cache root -- `~/.cache/hipdnn`
/// on Linux. A suite that benchmarks anything therefore both reads whatever a previous
/// run left behind and writes into it. Two concrete failure modes, one observed:
///
///  - **Flakiness.** A shard written by an earlier run of the same build makes the
///    ingestor serve a persisted ranking instead of benchmarking, so assertions on
///    benchmarking behaviour ("will benchmark", a knob default, a workspace size) fail
///    for a reason that has nothing to do with the change under test. The shard is
///    version-stamped with the git short hash, so this bites hardest when re-running
///    one commit -- exactly what a developer iterating does.
///  - **Pollution.** A test run silently mutates state the developer's own runs then
///    inherit.
///
/// Construct once in `main()`, before `RUN_ALL_TESTS()`. Doing it there rather than in
/// a fixture keeps the guarantee whether the binary is launched by ctest or by hand;
/// a ctest-only `ENVIRONMENT` property does not cover the direct invocation.
///
/// Uses `HIPDNN_CACHE_DIR` rather than `HIPDNN_DISABLE_CACHE`: disabling the cache
/// outright would make the persistence paths untestable, whereas redirecting keeps
/// them exercised and merely private to this process.
///
/// An explicit `HIPDNN_CACHE_DIR` already in the environment is left untouched, so CI
/// or a developer debugging a specific shard can still point the suite at one.
class ScopedTestCacheDir
{
public:
    /// @param binaryTag Short name of the owning test binary; only used to make the
    ///     scratch directory identifiable when someone goes looking at it.
    explicit ScopedTestCacheDir(const std::string& binaryTag)
    {
        if(!hipdnn_data_sdk::utilities::getEnv("HIPDNN_CACHE_DIR").empty())
        {
            // Caller pinned a cache root deliberately; respect it and own nothing.
            return;
        }

        // A temp-dir name unique to this process: create_directory() reports whether it
        // did the creating, so a collision is retried rather than silently shared.
        std::error_code ignored;
        const auto base = std::filesystem::temp_directory_path();
        auto seed = static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        bool created = false;
        for(int attempt = 0; attempt < 64 && !created; ++attempt, ++seed)
        {
            _path = base / ("hipdnn-test-cache-" + binaryTag + "-" + std::to_string(seed));
            created = std::filesystem::create_directory(_path, ignored) && !ignored;
        }
        if(!created)
        {
            // Fail soft: an unwritable temp dir is not worth aborting a suite over, and
            // cacheRoot() itself degrades to in-memory behaviour on a bad path.
            _path.clear();
            return;
        }

        hipdnn_data_sdk::utilities::setEnv("HIPDNN_CACHE_DIR", _path.string().c_str());
        _owned = true;
    }

    ~ScopedTestCacheDir()
    {
        if(!_owned)
        {
            return;
        }
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_CACHE_DIR");
        std::error_code ignored;
        std::filesystem::remove_all(_path, ignored);
    }

    /// The scratch root, or an empty path when the caller's own value was kept.
    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return _path;
    }

    ScopedTestCacheDir(const ScopedTestCacheDir&) = delete;
    ScopedTestCacheDir& operator=(const ScopedTestCacheDir&) = delete;
    ScopedTestCacheDir(ScopedTestCacheDir&&) = delete;
    ScopedTestCacheDir& operator=(ScopedTestCacheDir&&) = delete;

private:
    std::filesystem::path _path;
    bool _owned = false;
};

} // namespace hipdnn_test_sdk::utilities
