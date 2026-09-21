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
/// Two scopes, chosen by the `Scope` argument:
///
///  - `Scope::BINARY` (the default) is constructed once in `main()`, before
///    `RUN_ALL_TESTS()`. Doing it there rather than in a fixture keeps the guarantee
///    whether the binary is launched by ctest or by hand; a ctest-only `ENVIRONMENT`
///    property does not cover the direct invocation. It defers to an explicit
///    `HIPDNN_CACHE_DIR` already in the environment, so CI or a developer debugging a
///    specific shard can still point the suite at one.
///
///  - `Scope::TEST` is constructed per test case, and *always* takes ownership --
///    including over the binary-wide instance `main()` already installed, which is why
///    it cannot defer the way `BINARY` does. A binary-wide root is private to the
///    process but shared by every case in it, so one case that benchmarks writes a
///    shard the next case reads back. That is not hypothetical: the ingestor's
///    `ExecutesCorrectlyWithBenchmarkingEnabled` records a measured ranking for a
///    single-node FLOAT add, and `ReportsAKnobWhoseValuesComeFromTheCatalog` and
///    `ReportsTheMaximumWorkspaceAcrossSurvivingKernels` then observe that ranking
///    instead of the heuristic order -- the knob default and the workspace both follow
///    `sortedDefinitions().front()`. Worse, the two candidates differ by ~5% of a
///    microsecond, so which one the sweep records is a coin flip and the failure is
///    nondeterministic rather than merely order-dependent.
///
///    Nesting is what makes this subtle: a `TEST`-scoped instance that deferred to the
///    already-set `HIPDNN_CACHE_DIR` would be a silent no-op, looking like isolation
///    while changing nothing. Hence the explicit ownership, and the restore-to-previous
///    (rather than unset) in the destructor.
///
/// Uses `HIPDNN_CACHE_DIR` rather than `HIPDNN_DISABLE_CACHE`: disabling the cache
/// outright would make the persistence paths untestable, whereas redirecting keeps
/// them exercised and merely private to this scope. `cacheRoot()` re-reads the variable
/// on every access with no memoization, which is what makes a mid-process rescope take
/// effect at all.
class ScopedTestCacheDir
{
public:
    /// Whether this instance defers to an already-set HIPDNN_CACHE_DIR (BINARY) or
    /// always takes it over (TEST). See the class comment: a deferring per-test
    /// instance nested inside the binary-wide one would isolate nothing.
    enum class Scope
    {
        BINARY,
        TEST
    };

    /// @param tag Short name of the owning test binary or case; only used to make the
    ///     scratch directory identifiable when someone goes looking at it.
    /// @param scope See Scope.
    explicit ScopedTestCacheDir(const std::string& tag, Scope scope = Scope::BINARY)
    {
        auto existing = hipdnn_data_sdk::utilities::getEnv("HIPDNN_CACHE_DIR");
        if(!existing.empty() && scope == Scope::BINARY)
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
            _path = base / ("hipdnn-test-cache-" + tag + "-" + std::to_string(seed));
            created = std::filesystem::create_directory(_path, ignored) && !ignored;
        }
        if(!created)
        {
            // Fail soft: an unwritable temp dir is not worth aborting a suite over, and
            // cacheRoot() itself degrades to in-memory behaviour on a bad path.
            _path.clear();
            return;
        }

        // Saved rather than assumed empty: a TEST-scoped instance is normally nested
        // inside the binary-wide one, and unsetting on the way out would strip the
        // outer isolation from every later case in the process.
        _previous = std::move(existing);
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_CACHE_DIR", _path.string().c_str());
        _owned = true;
    }

    /// Restores the previous HIPDNN_CACHE_DIR (or unsets it when there was none) and
    /// removes the scratch tree. Runs on both the passing and failing path, since it is
    /// a destructor -- an assertion failure that unwinds out of a test body still
    /// leaves no residue and no redirected environment behind.
    ~ScopedTestCacheDir()
    {
        if(!_owned)
        {
            return;
        }
        if(_previous.empty())
        {
            hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_CACHE_DIR");
        }
        else
        {
            hipdnn_data_sdk::utilities::setEnv("HIPDNN_CACHE_DIR", _previous.c_str());
        }
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
    /// The HIPDNN_CACHE_DIR this instance displaced, restored on destruction. Non-empty
    /// only for a Scope::TEST instance nested inside an outer one; unsetting instead
    /// would strip that outer isolation from every later case in the process.
    std::string _previous;
    bool _owned = false;
};

} // namespace hipdnn_test_sdk::utilities
