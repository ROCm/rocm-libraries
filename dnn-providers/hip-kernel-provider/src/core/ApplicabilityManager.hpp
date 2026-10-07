// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/EngineManager.hpp>

#include "ApplicabilityPool.hpp"
#include "Handle.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/ingestor/StagedApplicability.hpp>
#endif

namespace hip_kernel_provider::core
{

/// Total threads, caller included, for one applicability call. 1 keeps the sequential loop;
/// 0 or `auto` sizes from the pending work and the CPUs.
static constexpr const char* APPLICABILITY_MAX_THREADS_ENV_NAME
    = "HIPDNN_APPLICABILITY_MAX_THREADS";

/// Offloaded work, in catalog variants, below which a call stays on the caller thread.
static constexpr const char* APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME
    = "HIPDNN_APPLICABILITY_MIN_OFFLOAD_VARIANTS";

static constexpr std::size_t DEFAULT_APPLICABILITY_MAX_THREADS = 1;

/// About 50 µs at the measured 0.26 µs per variant.
static constexpr std::size_t DEFAULT_MIN_OFFLOAD_VARIANTS = 200;

/// Upper bound on total threads when HIPDNN_APPLICABILITY_MAX_THREADS is auto.
static constexpr std::size_t AUTO_APPLICABILITY_MAX_THREADS = 9;

/// Parses a HIPDNN_APPLICABILITY_MAX_THREADS value. Empty returns the default, 1. `0` and
/// `auto` return 0, meaning auto. Anything but a non-negative integer logs a WARN and returns
/// the default.
std::size_t parseMaxThreads(const std::string& raw);

/// Parses a HIPDNN_APPLICABILITY_MIN_OFFLOAD_VARIANTS value. Empty returns the default.
/// Anything but a non-negative integer logs a WARN and returns the default.
std::size_t parseMinOffloadVariants(const std::string& raw);

/// Read on every call, never cached.
std::size_t maxThreadsFromEnv();
std::size_t minOffloadVariantsFromEnv();

/// Threads, caller included, for `pending` engines: min(pending, cpus,
/// AUTO_APPLICABILITY_MAX_THREADS) when maxThreads is 0, else min(maxThreads, pending).
/// Never less than 1.
std::size_t
    resolveTotalThreads(std::size_t maxThreads, std::size_t pending, std::size_t cpus) noexcept;

/// The provider's engine manager. Adds threaded applicability for staged (ingestor) engines:
/// with at least two of them registered and HIPDNN_APPLICABILITY_MAX_THREADS other than 1,
/// their per-variant catalog loops may finish on pool threads. Otherwise every call is the
/// base class's sequential loop.
class ApplicabilityManager : public hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>
{
public:
    using Base = hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>;

    /// pool must outlive every getApplicableEngineIds call.
    explicit ApplicabilityManager(ApplicabilityPool& pool);

    /// Hides the base: the manager tracks which engines are staged. Register through this type.
    void addEngine(std::unique_ptr<Engine> engine, std::string_view name = {});

    /// Same ids, in the same order, and the same exception as the base class's loop.
    std::vector<int64_t> getApplicableEngineIds(Handle& handle, const IGraph& opGraph);

    std::size_t stagedEngineCount() const noexcept;

private:
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    using StagedEngine = hipdnn_plugin_sdk::ingestor::IStagedApplicability<Handle>;

    struct Entry
    {
        int64_t id;
        Engine* engine;
        StagedEngine* staged;
    };

    std::vector<int64_t>
        getApplicableEngineIdsStaged(Handle& handle, const IGraph& opGraph, std::size_t maxThreads);

    // Inside the guard: the pool is only used by the staged path, and an unused private
    // reference member fails -Werror (-Wunused-private-field) in a build without the ingestor.
    ApplicabilityPool& _pool;
    // The base class's iteration order, rebuilt on every addEngine.
    std::vector<Entry> _entries;
#endif

    std::size_t _stagedCount = 0;
};

} // namespace hip_kernel_provider::core
