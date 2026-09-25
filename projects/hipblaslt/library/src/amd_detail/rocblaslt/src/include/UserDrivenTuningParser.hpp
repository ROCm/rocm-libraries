/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
     * Re-read HIPBLASLT_TUNING_OVERRIDE_FILE after the singleton exists.
     *
     * Tests only: they set and clear the variable within one process, and the
     * singleton otherwise reads it once, at its first use.
     */
    void reloadForTest()
    {
        file_path.clear();
        env_mode = false;
        load();
    }

private:
    OverrideSingleton()
    {
        load();
    }

    void load()
    {
        char* Env = getenv("HIPBLASLT_TUNING_OVERRIDE_FILE");
        if(Env)
        {
            file_path = Env;
            env_mode  = true;
        }
    }

    ~OverrideSingleton() {}
};

namespace TensileLite
{
    /**
     * HIPBLASLT_TUNING_MODE and HIPBLASLT_TUNING_CACHE_PATH, read on first use.
     *
     * Read once rather than per call, so the hot path costs nothing and a
     * process cannot change mode halfway through a run. Setting either variable
     * after the first hipBLASLt call has no effect.
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

        TuningMode mode() const
        {
            return m_config.mode;
        }
        const std::string& cachePath() const
        {
            return m_config.cachePath;
        }
        bool reads() const
        {
            return m_config.reads();
        }
        bool writes() const
        {
            return m_config.writes();
        }

        /** Re-read the environment. Tests only, like OverrideSingleton::reloadForTest. */
        void reloadForTest()
        {
            load();
        }

    private:
        TuningModeSingleton()
        {
            load();
        }

        void load();

        TuningModeConfig m_config;
    };

    /** Running tallies behind the closing summary and the test hooks. */
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
            return "loaded=" + std::to_string(entriesLoaded.load()) + " hits="
                   + std::to_string(hits.load()) + " misses=" + std::to_string(misses.load())
                   + " invalidated=" + std::to_string(invalidated.load()) + " tuned="
                   + std::to_string(tuned.load()) + " skipped=" + std::to_string(skipped.load());
        }
    };

    /**
     * Which tuning file this process consults, if any.
     *
     * HIPBLASLT_TUNING_CACHE_PATH and HIPBLASLT_TUNING_OVERRIDE_FILE are
     * mutually exclusive rather than merged: with no tuning mode set the
     * override file behaves as it always has, and with one set only the cache
     * is consulted and the override is ignored, which the startup line says.
     */
    struct TuningFileSelection
    {
        bool        active = false;
        std::string path;
    };

    TuningFileSelection selectTuningFile();

    /** The build rows are trusted against: the running library's own. */
    const std::string& currentBuildStamp();

    /**
     * Append one tuned winner to the tuning file.
     *
     * Takes the problem rather than the key so the type columns can be written
     * in the spelling the parser reads back. See appendTuningRow for what
     * concurrent writers can rely on.
     */
    bool appendTunedEntry(const std::string&                 path,
                          const RocblasltContractionProblem& problem,
                          const TunedEntry&                  entry);

    /**
     * What one tuning attempt did.
     *
     * Skips are policy: the tuner understood the problem and chose not to
     * measure it, which is expected on some shapes forever. Fallbacks are
     * everything else and usually mean something is wrong. Scratch splits
     * across that line: a request over the configured cap is the cap doing its
     * job, while a device that refuses the allocation is a failure.
     */
    enum class TuningAttempt : uint32_t
    {
        Tuned = 0,
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

    /**
     * Load a tuning file into OverrideMap::getMap(). Each path is read at most
     * once per process.
     */
    void getContractionProblemsFromFile(const std::string& path);

    /** How the cache file was read, for the startup line. */
    enum class TuningLoadStatus : uint32_t
    {
        Ok = 0,
        NotFound,
        ReadError,
        NoPath,
    };

    /**
     * Announce the mode, path and load result once per process, and arrange for
     * the closing summary. Does nothing in off mode.
     */
    void announceTuningModeOnce(TuningLoadStatus status);

    /**
     * Note that a lookup for this problem did or did not find a usable entry.
     *
     * Counted by distinct key rather than by call, because the summary is read
     * against loaded=N and a hot loop over one uncached shape would otherwise
     * report thousands of fallbacks for one missing row. A key that matches
     * once counts as matched. Does nothing in off mode.
     */
    void recordTuningLookup(const ProblemOverride& key, bool matched);

    /**
     * True only the first time this key's entry at this index is rejected. The
     * heuristic lookup, the execution path and the recheck under the tuning lock
     * can all meet one stale row, and it is still one rejected entry.
     */
    bool recordTuningInvalidation(const ProblemOverride& key, int solutionIndex);

    /**
     * Note that this process spent a search on this problem, and ask whether it
     * has.
     *
     * Set for the outcomes that spent the search and left the shape wanting
     * another, so the next matmul does not start the same search again. Per
     * process, not per file: a later run with a higher
     * HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE is what lets such a shape finish.
     */
    void recordTuningAttempt(const ProblemOverride& key);
    bool tuningAlreadyAttempted(const ProblemOverride& key);

    /**
     * Note that this problem was tuned in this process, whether or not the
     * winner reached the file: it is in the in-memory cache either way.
     */
    void recordTuningWinner(const ProblemOverride& key);

    /**
     * Whether a tuning-start or terminal line should be written.
     *
     * With the info bit set, always. Otherwise success and failure are bounded
     * separately, so a key whose first attempt failed can still report the tune
     * that succeeds later. A failure is bounded per key once that key announced
     * a start, since a start with no ending would look like a hang, and per
     * reason before that, since those declines repeat across thousands of
     * shapes.
     */
    bool shouldLogTuningStart(const ProblemOverride& key);
    bool shouldLogTuningTerminal(const ProblemOverride& key, TuningAttempt result);

    /** Cache events logged at most once per key. */
    enum class TuningKeyEvent : uint32_t
    {
        Hit = 0,
        Miss,
        Invalid,
    };

    /**
     * Whether this key's event is worth logging: only with the info bit set,
     * and once per key, since replay meets the same key on every call.
     */
    bool shouldLogTuningKeyEvent(TuningKeyEvent kind, const ProblemOverride& key);

    /** Drop the announcement latch and every per-key set. Tests only. */
    void resetTuningDiagnosticsForTest();

    /** The distinct-key tally behind the summary line, for tests. */
    void tuningLookupTallyForTest(uint64_t* shapes,
                                  uint64_t* matched,
                                  uint64_t* fellback,
                                  uint64_t* tuned);
} // namespace TensileLite
