/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#pragma once

#include "TuningCacheStore.hpp"
#include "auxiliary.hpp"
#include "tensile_host.hpp"
#include <Tensile/DataTypes.hpp>

#include <atomic>
#include <cstdint>
#include <string>

class OverrideSingleton
{
public:
    std::string file_path;
    bool        env_mode = false;

    static OverrideSingleton& getInstance()
    {
        static OverrideSingleton gInstance;
        return gInstance;
    }

    // copy contructor
    OverrideSingleton(const OverrideSingleton&) = delete;
    // assignment operator
    OverrideSingleton& operator=(const OverrideSingleton&) = delete;

    /**
     * Re-read HIPBLASLT_TUNING_OVERRIDE_FILE. Tests only, for the same reason as
     * TuningModeSingleton::reloadForTest: the variable is latched the first time
     * anything asks for it, which in a test binary is whichever case ran first,
     * so a later case setting it would otherwise be testing nothing.
     */
    void reloadForTest()
    {
        readEnv();
    }

private:
    OverrideSingleton()
    {
        readEnv();
    }

    void readEnv()
    {
        file_path.clear();
        env_mode = false;

        if(const char* env = getenv("HIPBLASLT_TUNING_OVERRIDE_FILE"))
        {
            file_path = env;
            env_mode  = true;
        }
    }

    ~OverrideSingleton() {}
};

namespace TensileLite
{
    /**
     * The tuning mode switch, latched on first use.
     *
     * Latched rather than read per call so the hot path costs nothing and so a
     * process cannot change mode halfway through a run. Setting the variable
     * after the first matmul therefore has no effect, which is the documented
     * behaviour.
     */
    class TuningModeSingleton
    {
    public:
        static TuningModeSingleton& getInstance()
        {
            static TuningModeSingleton gInstance;
            return gInstance;
        }

        TuningModeSingleton(const TuningModeSingleton&)            = delete;
        TuningModeSingleton& operator=(const TuningModeSingleton&) = delete;

        TuningMode         mode() const { return m_config.mode; }
        const std::string& cachePath() const { return m_config.cachePath; }

        /**
         * Re-read the environment after the singleton already exists.
         *
         * Only for tests, which run every mode in one process and so cannot
         * rely on the latch. Mirrors Debug::reloadDebugBitsForTest().
         * asPrivileged reads it the way a set-user-ID process would, since a
         * test cannot become one.
         */
        void reloadForTest(bool asPrivileged = false);

        bool reads() const { return m_config.reads(); }
        bool writes() const { return m_config.writes(); }

    private:
        TuningModeSingleton();

        void load(bool isPrivileged);

        TuningModeConfig m_config;
    };

    /**
     * Running tallies for the tuning cache.
     *
     * Cheap atomics rather than a reporting API. "How many of my entries
     * survived this upgrade" is the question per-entry validation makes
     * answerable, and without a count the answer is only visible by grepping
     * individual log lines.
     */
    struct TuningCounters
    {
        static TuningCounters& instance()
        {
            static TuningCounters gInstance;
            return gInstance;
        }

        std::atomic<uint64_t> entriesLoaded{0};
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
        std::atomic<uint64_t> invalidated{0};
        std::atomic<uint64_t> tuned{0};
        std::atomic<uint64_t> skipped{0};

        // Searches that got as far as tuning-start, whatever their outcome.
        std::atomic<uint64_t> attempts{0};

        std::string summary() const
        {
            return "loaded=" + std::to_string(entriesLoaded.load())
                   + " hits=" + std::to_string(hits.load())
                   + " misses=" + std::to_string(misses.load())
                   + " invalidated=" + std::to_string(invalidated.load())
                   + " tuned=" + std::to_string(tuned.load())
                   + " skipped=" + std::to_string(skipped.load());
        }
    };

    /**
     * Which tuning file this process should consult, if any.
     *
     * HIPBLASLT_TUNING_CACHE_PATH and HIPBLASLT_TUNING_OVERRIDE_FILE are
     * mutually exclusive rather than merged. There is a single global map and a
     * per-path load latch, so letting both load would mean whichever arrived
     * first silently suppressed the other. In off mode the legacy override
     * behaves exactly as it always has; in cache or tune mode only the managed
     * cache is consulted and the legacy override is ignored with a log line.
     */
    struct TuningFileSelection
    {
        bool        active   = false;
        bool        writable = false;
        std::string path;
    };

    TuningFileSelection selectTuningFile();

    /** The build stamp rows are written with and legacy rows are trusted against. */
    const std::string& currentBuildStamp();

    /**
     * Append one tuned winner to the tuning file.
     *
     * Takes the problem rather than the key so the type columns can be written
     * in the spelling the parser reads back, and so a lossy key-to-string
     * inverse is never needed. See appendTuningRow for what concurrent writers
     * can rely on.
     */
    bool appendTunedEntry(const std::string&                 path,
                          const RocblasltContractionProblem& problem,
                          const TunedEntry&                  entry);

    /**
     * Load a tuning file into the map. Safe to call repeatedly; each distinct
     * path is parsed at most once.
     */
    void getContractionProblemsFromFile(const std::string& path);

    /**
     * What one tuning attempt did.
     *
     * The benchmarker used to answer with an index or a bare -1, and that -1
     * covered both deliberate declines and half a dozen silent failures. The
     * call site could not tell which of them had already logged, so it could
     * not promise exactly one terminal event per attempt without either
     * printing twice or saying nothing at all.
     *
     * Skips are policy: the tuner understood the problem and chose not to
     * measure it. Fallbacks are everything else, and they are worth
     * distinguishing because a skip is expected on some shapes forever while a
     * fallback usually means something is wrong. Scratch splits across that
     * line: a request over the configured cap is the cap doing its job, while a
     * device that refuses the allocation is a failure the user should see as
     * one.
     */
    enum class TuningAttempt : uint32_t
    {
        Tuned = 0,
        /** Budget stopped the search, but a winner was measured and recorded. */
        TunedPartial,
        SkippedInPlaceBeta,
        SkippedExtentUnknown,
        SkippedScratchCap,
        SkippedBudget,
        FallbackScratchAlloc,
        FallbackSetup,
        FallbackEnumeration,
        FallbackNoWinner,
        FallbackException,
    };

    /** True for a policy decline, false for a failure. */
    bool tuningAttemptIsSkip(TuningAttempt result);

    /** Human-readable cause, without the tuning-cache prefix or event token. */
    const char* tuningAttemptReason(TuningAttempt result);

    /** How the managed cache file was read, for the startup announcement. */
    enum class TuningLoadStatus : uint32_t
    {
        Ok = 0,
        NotFound,
        ReadError,
        NoPath,
    };

    /**
     * Announce mode, path and load result once per process, and arrange for the
     * closing summary.
     *
     * Repeat calls are free. Callers do not need to know whether they are the
     * first; the load path and the no-path branch both call this.
     */
    void announceTuningModeOnce(TuningLoadStatus status);

    /**
     * Note that a lookup for this problem did or did not find a usable entry.
     *
     * Counted by distinct key rather than by call, because the summary is read
     * against `loaded=N`: a hot loop over one uncached shape would otherwise
     * report tens of thousands of fallbacks for a single missing row. Matching
     * is sticky, so a shape that is tuned mid-run and served afterwards ends up
     * counted as served. Does nothing in off mode.
     */
    void recordTuningLookup(const ProblemOverride& key, bool matched);

    /**
     * Note that this process has already benchmarked this problem, and ask
     * whether it has.
     *
     * Set for the outcomes that spent the search and left the shape wanting
     * another one. Without it the next matmul starts the same search again: a
     * 4096-cubed shape under a 30 s ceiling re-ran it on every call, spending
     * the whole ceiling each time and never recording a row. A truncated search
     * that did record a partial winner needs the latch just as much, because the
     * retry would measure the same prefix under the same ceiling and reach the
     * same answer.
     *
     * A search that finished needs no latch, since its entry closes the gate on
     * its own. Neither do the declines made before the measurement loop: they
     * cost microseconds, so retrying them is free and lets a shape recover if
     * the condition was transient.
     *
     * Per process, not per file. Raising or clearing
     * HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE in a later run is what lets the shape
     * finish, so the latch must not outlive the process that set it.
     */
    void recordTuningAttempt(const ProblemOverride& key);
    bool tuningAlreadyAttempted(const ProblemOverride& key);

    /**
     * Note that this problem was successfully tuned in this process.
     *
     * Distinct-key, and deliberately independent of whether the winner reached
     * the file. The winner is published to the in-memory cache before the append
     * is attempted, so it serves every later call either way, and counting the
     * append instead made a run that reported `tuning-done ... persisted=no`
     * close with `tuned=0`. Counting attempts rather than keys could also report
     * more tuned shapes than shapes seen, since one key can be retuned.
     */
    void recordTuningWinner(const ProblemOverride& key);

    /**
     * Claim the first sighting of a cache entry that failed identity validation.
     *
     * True only the first time this key/index pair is rejected. A stale row is
     * rediscovered by the heuristic lookup, the execution probe and the recheck
     * that follows the tuning lock, so counting every sighting reported three
     * invalidations for one entry.
     */
    bool recordTuningInvalidation(const ProblemOverride& key, int solutionIndex);

    /**
     * Whether a tuning lifecycle line should be written.
     *
     * These exist because a skipped or failed attempt records no winner, so the
     * shape stays uncached and the whole attempt runs again on the next matmul.
     * Unbounded, that is one line per call on the default channel.
     *
     * Both consult the info bit first and never suppress anything once it is
     * on: a user who asked for diagnostics gets every attempt, matching what
     * the info logging did before.
     *
     * Success and failure are bounded separately, so a key whose first attempt
     * failed can still report the tune that eventually succeeds. A failure is
     * bounded per key once that key has announced a start, because a start with
     * no ending is the hang this feature exists to rule out, and per reason
     * before then, because the declines made before a start are the ones that
     * repeat across thousands of shapes.
     */
    bool shouldLogTuningStart(const ProblemOverride& key);
    bool shouldLogTuningTerminal(const ProblemOverride& key, TuningAttempt result);

    /** Cache events that are logged at most once per key when info is on. */
    enum class TuningKeyEvent : uint32_t
    {
        Hit = 0,
        Miss,
        Invalid,
    };

    /**
     * Whether this key-scoped cache event is worth logging.
     *
     * Always false when the info bit is off, so the replay hot path never pays
     * for the hash or the set probe.
     */
    bool shouldLogTuningKeyEvent(TuningKeyEvent kind, const ProblemOverride& key);

    /**
     * Drop every process-lifetime diagnostic latch. Tests only: one test binary
     * runs many modes and cache files, and would otherwise inherit the first
     * test's announcement and bounds.
     */
    void resetTuningDiagnosticsForTest();

    /** Distinct-key tally behind the summary line, for tests. */
    void tuningLookupTallyForTest(uint64_t* shapes,
                                  uint64_t* matched,
                                  uint64_t* fellback,
                                  uint64_t* tuned);
} // namespace TensileLite
