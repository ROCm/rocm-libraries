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

/// Offloaded work, in catalog variants, below which a call stays on the caller thread. An
/// override of DEFAULT_MIN_OFFLOAD_VARIANTS, for calibration.
static constexpr const char* APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME
    = "HIPDNN_APPLICABILITY_MIN_OFFLOAD_VARIANTS";

/// The smallest offload measured to beat staying inline on gfx950 dense-attention packs.
static constexpr std::size_t DEFAULT_MIN_OFFLOAD_VARIANTS = 50;

/// Upper bound on the total threads, caller included, one applicability call uses.
static constexpr std::size_t MAX_APPLICABILITY_THREADS = 9;

/// Parses a HIPDNN_APPLICABILITY_MIN_OFFLOAD_VARIANTS value. Empty returns the default.
/// Anything but a non-negative integer logs a WARN and returns the default.
std::size_t parseMinOffloadVariants(const std::string& raw);

/// Read on every call that has two or more pending engines, never cached.
std::size_t minOffloadVariantsFromEnv();

/// Threads, caller included, for `pending` engines on `cpus` CPUs: min(pending, cpus,
/// MAX_APPLICABILITY_THREADS). Never less than 1.
std::size_t resolveTotalThreads(std::size_t pending, std::size_t cpus) noexcept;

/// The provider's engine manager. Adds threaded applicability for staged (ingestor) engines:
/// with at least two of them registered, a call whose pending engines have enough variants left
/// to walk finishes all but the largest on pool threads. Below that, every call is the base
/// class's sequential loop or its equivalent on the caller thread.
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

    std::vector<int64_t> getApplicableEngineIdsStaged(Handle& handle, const IGraph& opGraph);

    // Inside the guard: the pool is only used by the staged path, and an unused private
    // reference member fails -Werror (-Wunused-private-field) in a build without the ingestor.
    ApplicabilityPool& _pool;
    // The base class's iteration order, rebuilt on every addEngine.
    std::vector<Entry> _entries;
#endif

    std::size_t _stagedCount = 0;
};

} // namespace hip_kernel_provider::core
