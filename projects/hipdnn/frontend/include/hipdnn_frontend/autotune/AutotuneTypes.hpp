// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file AutotuneTypes.hpp
 * @brief Core type definitions for the hipDNN autotuning system
 *
 * Defines the tuning mode and strategy enumerations, the AutotuneConfig
 * struct for controlling autotuning behavior, the AutotuneResult struct
 * for per-engine benchmarking results, and the AutotuneStorageConfig
 * struct for config file output parameters.
 */

#pragma once

#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace hipdnn_frontend
{

/**
 * @enum TuneMode
 * @brief Controls whether autotune() performs engine-internal cache priming
 *
 * STANDARD mode benchmarks engines as-is. EXHAUSTIVE mode first builds and
 * executes temporary priming plans with the global.benchmarking knob to
 * prime engine caches, then benchmarks with real plans.
 */
enum class TuneMode
{
    STANDARD, ///< Simple wall-time comparison (no engine-internal cache priming)
    EXHAUSTIVE ///< Build temporary priming plans, prime engine caches, then benchmark
};

/**
 * @enum AutotuneStrategy
 * @brief Controls how many timed iterations each engine runs
 *
 * Selects when the timed loop stops, and so how many samples each engine's timing
 * statistics are drawn from. It does not select which statistic ranks the engines.
 */
enum class AutotuneStrategy
{
    /// Run exactly timedIterations iterations.
    FIXED_AVERAGE,

    /// Run until the trailing-window variation falls below stabilityThreshold, or
    /// maxIterations is reached (default).
    RUN_UNTIL_STABLE
};

/**
 * @enum PrimingFailurePolicy
 * @brief Controls how autotune() reacts when an engine's exhaustive priming fails
 *
 * During EXHAUSTIVE priming, an engine's temporary priming plan may fail to
 * build, finalize, or execute. This policy selects whether that failure aborts
 * the whole autotune() call or whether the engine is benchmarked unprimed. When
 * priming fails and the policy is BENCHMARK_UNPRIMED, the autotuneResult will set
 * ranExhaustive=false and exhaustiveNotRunReason describes the reason.
 */
enum class PrimingFailurePolicy
{
    ABORT_ON_PRIMING_FAILURE, ///< Abort autotune() and return an error
    BENCHMARK_UNPRIMED ///< Continue and benchmark the engine unprimed
};

/**
 * @enum AutotuneCacheWriteOutcome
 * @brief Write-provenance of autotuneExhaustiveSweep()'s exact-match cache write-back
 *
 * Reports whether a given autotuneExhaustiveSweep() call's measured ranking reached the
 * exact-match autotune cache, and if not, the specific reason.
 */
enum class AutotuneCacheWriteOutcome
{
    /// No engine succeeded benchmarking, so there was no ranking to write and the
    /// write-back API was never called.
    NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE,
    /// The ranking was written to the exact-match cache.
    WRITTEN,
    /// The exact-match cache is disabled via HIPDNN_DISABLE_EXACT_ENGINE_CACHE; the
    /// write-back API was called but declined without writing.
    DECLINED_DISABLED,
    /// The write-back API was called but declined: the graph descriptor was not
    /// finalized, or the graph/device pair could not be reduced to a cache key.
    DECLINED_UNKEYABLE,
    /// The cache already held this exact ranking, so nothing was written. Not a failure:
    /// a re-tune that measured the same order has nothing to add.
    UNCHANGED,
    /// A caller choice -- config.engineIdFilter, deselect_engines(),
    /// deselect_workspace_greater_than(), or a workspaceSize too small for a compiled
    /// plan -- kept at least one applicable candidate out of the timing loop, so the
    /// sweep did not cover the engine set a later lookup will see.
    ///
    /// Declined because neither alternative is safe: omitting the engine from the sampled
    /// set makes every later lookup reject the entry, and including it claims a
    /// measurement that never happened. autotuneExhaustiveSweep() persists only a full
    /// sweep; see its documentation.
    NOT_ATTEMPTED_PARTIAL_SWEEP
};

/// Get the string representation of an AutotuneCacheWriteOutcome
inline const char* autotuneCacheWriteOutcomeToString(AutotuneCacheWriteOutcome outcome)
{
    switch(outcome)
    {
    case AutotuneCacheWriteOutcome::NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE:
        return "NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE";
    case AutotuneCacheWriteOutcome::WRITTEN:
        return "WRITTEN";
    case AutotuneCacheWriteOutcome::DECLINED_DISABLED:
        return "DECLINED_DISABLED";
    case AutotuneCacheWriteOutcome::DECLINED_UNKEYABLE:
        return "DECLINED_UNKEYABLE";
    case AutotuneCacheWriteOutcome::UNCHANGED:
        return "UNCHANGED";
    case AutotuneCacheWriteOutcome::NOT_ATTEMPTED_PARTIAL_SWEEP:
        return "NOT_ATTEMPTED_PARTIAL_SWEEP";
    default:
        return "UNKNOWN";
    }
}

/// Get the string representation of a TuneMode
inline const char* tuneModeToString(TuneMode mode)
{
    switch(mode)
    {
    case TuneMode::STANDARD:
        return "STANDARD";
    case TuneMode::EXHAUSTIVE:
        return "EXHAUSTIVE";
    default:
        return "UNKNOWN";
    }
}

/// Get the lowercase string representation of a TuneMode (for config file output)
inline std::string tuneModeToLowerString(TuneMode mode)
{
    switch(mode)
    {
    case TuneMode::STANDARD:
        return "standard";
    case TuneMode::EXHAUSTIVE:
        return "exhaustive";
    default:
        return "unknown";
    }
}

/// Get the string representation of an AutotuneStrategy
inline const char* strategyToString(AutotuneStrategy strategy)
{
    switch(strategy)
    {
    case AutotuneStrategy::FIXED_AVERAGE:
        return "FIXED_AVERAGE";
    case AutotuneStrategy::RUN_UNTIL_STABLE:
        return "RUN_UNTIL_STABLE";
    default:
        return "UNKNOWN";
    }
}

/// Get the lowercase string representation of an AutotuneStrategy (for config file output)
inline std::string strategyToLowerString(AutotuneStrategy strategy)
{
    switch(strategy)
    {
    case AutotuneStrategy::FIXED_AVERAGE:
        return "fixed_average";
    case AutotuneStrategy::RUN_UNTIL_STABLE:
        return "run_until_stable";
    default:
        return "unknown";
    }
}

/// Get the string representation of a PrimingFailurePolicy
inline const char* primingFailurePolicyToString(PrimingFailurePolicy policy)
{
    switch(policy)
    {
    case PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE:
        return "ABORT_ON_PRIMING_FAILURE";
    case PrimingFailurePolicy::BENCHMARK_UNPRIMED:
        return "BENCHMARK_UNPRIMED";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Per-engine benchmarking result from autotune()
 *
 * Contains timing data, ranking information, and status for each
 * engine configuration that was benchmarked (or attempted).
 */
struct AutotuneResult
{
    // --- Identity ---
    int64_t engineId = -1; ///< Engine that was benchmarked
    std::string engineName; ///< Human-readable engine name

    /// Informational, records knobs explicitly set on the engine.
    std::vector<KnobSetting> knobSettings;

    // --- Timing ---
    float minTimeMs = 0.0f; ///< Fastest single iteration (used for default ranking)
    float avgTimeMs = 0.0f; ///< Mean across all iterations
    float stddevMs = 0.0f; ///< Standard deviation of timing measurements

    /// Mean of this engine's iterations after discarding the slow tail: what the engine
    /// usually costs, rather than the best it ever managed. Always populated, but only
    /// autotuneExhaustiveSweep() ranks on it; autotune() ranks on @c minTimeMs. Compare
    /// @c stddevMs to see how much an engine's timings actually varied.
    float robustTimeMs = 0.0f;

    int iterationsRun = 0; ///< Actual number of timed iterations executed

    /// For RUN_UNTIL_STABLE, true when the trailing-window variation fell below
    /// stabilityThreshold before maxIterations was reached. For FIXED_AVERAGE, true when
    /// every iteration completed. False on benchmark failure under either strategy.
    ///
    /// Note this does not affect ranking: a candidate that never converged is ranked on
    /// its measured timings like any other.
    bool converged = false;

    // --- Status ---
    int rank = -1; ///< 0-based ranking (0 = fastest); -1 for failed engines
    bool succeeded = false; ///< Whether this engine succeeded benchmarking

    /// true if this engine reached the timing loop, whether or not it produced a usable
    /// measurement. Distinct from @ref succeeded, which is false both for an engine that
    /// was measured and failed and for one never measured at all.
    bool benchmarked = false;

    /// true if a caller choice kept this candidate out of the timing loop:
    /// config.engineIdFilter, deselect_engines(), deselect_workspace_greater_than(), or a
    /// compiled workspace over the workspaceSize passed to this call.
    ///
    /// Decides whether a sweep may be persisted. A miss intrinsic to the engine and graph
    /// (failed to compile or finalize) recurs on every run, so the record can name it.
    /// A caller-chosen miss may not recur: the next run can lift the filter or pass a
    /// larger workspace, making the engine a live candidate the stored ranking never
    /// measured. autotuneExhaustiveSweep() therefore declines the write; see
    /// AutotuneCacheWriteOutcome::NOT_ATTEMPTED_PARTIAL_SWEEP.
    bool excludedByCaller = false;

    std::string errorMessage; ///< Empty if no error; describes the benchmarking failure otherwise

    int64_t workspaceSize = 0; ///< Workspace bytes used by this engine
    int64_t estimatedWorkspaceSize = 0; ///< Pre-compile workspace estimate from engine config
    int compiledPlanIndex = -1; ///< Index into compiled plans vector; used for winner selection

    TuneMode modeUsed = TuneMode::STANDARD; ///< Which mode was used for this engine

    /// true if the engine exposes the benchmarking knob and therefore supports
    /// exhaustive priming, independent of whether priming actually ran.
    bool supportsExhaustive = false;

    /// true if this engine was primed via a temporary benchmarking plan before
    /// timing. false if the engine does not support exhaustive priming or
    /// STANDARD mode was used.
    bool ranExhaustive = false;

    /// Why exhaustive priming did not run for this engine (ranExhaustive = false) when
    /// TuneMode::EXHAUSTIVE was configured (e.g. priming failed or was skipped because
    /// its workspace exceeded the budget).
    std::string exhaustiveNotRunReason;

    AutotuneStrategy strategyUsed
        = AutotuneStrategy::RUN_UNTIL_STABLE; ///< Which strategy was used for this engine
};

/**
 * @brief User-provided ranking function for autotune results
 *
 * When provided in AutotuneConfig, this function sorts the results vector
 * in-place to determine the final ranking. When nullptr (default),
 * results are ranked by minTimeMs ascending (fastest first).
 */
using AutotuneRankingFn = std::function<void(std::vector<AutotuneResult>&)>;

/**
 * @brief Configuration parameters for autotuning
 *
 * Controls the tuning mode, benchmarking strategy, iteration counts,
 * convergence parameters, workspace limits, engine filtering, and
 * custom ranking behavior.
 *
 * @code{.cpp}
 * AutotuneConfig config;
 * config.mode = TuneMode::EXHAUSTIVE;
 * config.strategy = AutotuneStrategy::FIXED_AVERAGE;
 * config.timedIterations = 20;
 * graph.autotune(handle, variantPack, workspace, config);
 * @endcode
 */
struct AutotuneConfig
{
    TuneMode mode = TuneMode::STANDARD; ///< Tuning mode (STANDARD or EXHAUSTIVE)
    AutotuneStrategy strategy = AutotuneStrategy::RUN_UNTIL_STABLE; ///< When the timed loop stops

    int warmupIterations = 1; ///< Untimed iterations run before timing starts
    int timedIterations = 10; ///< Timed iterations per engine; FIXED_AVERAGE only

    /// Maximum iterations for RUN_UNTIL_STABLE (must be >= windowSize)
    int maxIterations = 100;

    /// Window size for convergence check in RUN_UNTIL_STABLE (must be >= 2)
    int windowSize = 3;

    /// Coefficient of variation threshold for RUN_UNTIL_STABLE convergence (e.g. 0.05 = 5%)
    float stabilityThreshold = 0.05f;

    /// Engine filter: only benchmark candidates with these engine IDs (empty = all engines).
    /// Does not discover or add new candidates. Unmatched engine IDs are silently ignored.
    std::vector<int64_t> engineIdFilter;

    /// Custom ranking function (nullptr = rank by minTimeMs ascending)
    AutotuneRankingFn rankingFn = nullptr;

    /// Policy for engine priming failures during EXHAUSTIVE priming. With
    /// ABORT_ON_PRIMING_FAILURE (default), autotune() returns an error if any
    /// engine's priming plan fails. With BENCHMARK_UNPRIMED, that engine is
    /// benchmarked unprimed (AutotuneResult::ranExhaustive set to false) and
    /// tuning continues. Has no effect in STANDARD mode.
    PrimingFailurePolicy primingFailurePolicy = PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE;
};

/**
 * @brief Config file output parameters for autotune results
 *
 * When filePath is non-empty, autotune() writes the ranked results
 * to a JSON file in heuristic config format. The file can be
 * loaded on subsequent runs via HIPDNN_HEUR_CONFIG_PATH.
 */
struct AutotuneStorageConfig
{
    /// Output file path (empty = no file output)
    std::filesystem::path filePath;

    /// When true, delete all existing file content before writing new results.
    /// When false, replace only matching (operation, tensors, criteria) entries.
    bool deleteAllExistingFileContent = false;
};

} // namespace hipdnn_frontend
