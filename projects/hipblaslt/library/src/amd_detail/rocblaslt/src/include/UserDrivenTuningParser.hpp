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

#include "auxiliary.hpp"
#include "tensile_host.hpp"
#include <Tensile/DataTypes.hpp>
#include <shared_mutex>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

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
#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
    enum class TuningMode : uint32_t
    {
        Off   = 0,
        Cache = 1,
        Tune  = 2,
    };

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

        TuningMode         mode() const { return m_mode; }
        const std::string& cachePath() const { return m_cachePath; }

        /**
         * Re-read the environment after the singleton already exists.
         *
         * Only for tests, which run every mode in one process and so cannot
         * rely on the latch. Mirrors Debug::reloadDebugBitsForTest().
         */
        void reloadForTest();

        // Tuning does nothing without somewhere to keep the results. There is
        // no default cache location in this milestone, on purpose: a managed
        // default directory brings a lifetime and growth policy with it.
        bool reads() const { return m_mode != TuningMode::Off && !m_cachePath.empty(); }
        bool writes() const { return m_mode == TuningMode::Tune && !m_cachePath.empty(); }

    private:
        TuningModeSingleton();

        TuningMode  m_mode = TuningMode::Off;
        std::string m_cachePath;
    };
#endif // HIPBLASLT_ENABLE_TUNING_CACHE

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
     * Row schema version written into and read from the tuning file.
     *
     * v0 is the historical format: ten problem columns plus solution_index, and
     * no per-row build stamp. Rows without an explicit schema_version column are
     * v0 and are trusted only when the file build stamp matches, because the
     * fields they omit (leading dimensions, strides) have no safe default.
     */
    enum class TuningSchemaVersion : uint32_t
    {
        Legacy  = 0,
        Current = 1,
    };

    /**
     * Where an entry came from. Used for diagnostics and for deciding how much
     * to trust an entry that carries no solution name.
     */
    enum class TuningEntrySource : uint32_t
    {
        LegacyOverrideFile = 0,
        ManagedCacheFile   = 1,
        OnlineTuning       = 2,
    };

    /**
     * The semantic problem key.
     *
     * Deliberately built from plain scalars rather than rocblaslt/hip types so
     * that it can be constructed and compared in a unit test without a solution
     * library or a device. Conversion from RocblasltContractionProblem lives in
     * tensile_host.cpp.
     *
     * FIELD CLASSIFICATION. Every field of RocblasltContractionProblem must be
     * accounted for as key, metadata, or deliberately ignored. Everything listed
     * as a member below is key. The deliberate exclusions, with reasons:
     *
     *   A, B, C, D, E, batch_A..batch_D, bias, scaleA..scaleE, scaleAlphaVec,
     *   amaxD, workspace, Synchronizer
     *       Device addresses. Only their presence is keyed, never the value.
     *
     *   C == D (aliasing)
     *       construct_rocblaslt_problem passes null for both C and D, so at
     *       heuristic time this is nullptr == nullptr and setCEqualsD(true) is
     *       called for every problem, in-place or not. The value is a constant
     *       at lookup and carries no information.
     *
     *   alpha, beta, alpha_owned
     *       assignAlphaBeta1 fabricates both as one at heuristic time, so the
     *       key cannot represent them. The tuner does measure with the caller's
     *       real values, which means a beta=0-tuned entry can serve a beta=2
     *       caller. Accepted imprecision for now; see the plan.
     *
     *   pointer alignment
     *       No solution predicate consumes it today.
     *
     *   stream, workspaceSize
     *       Execution context, not problem identity. The winner's required
     *       workspace is stored in the entry instead.
     *
     *   row_stride_a..row_stride_e
     *       Always 1 on every construction path.
     */
    class ProblemOverride
    {
    public:
        ProblemOverride() = default;

        // Orientation and shape
        bool   transA     = false;
        bool   transB     = false;
        size_t m          = 0;
        size_t n          = 0;
        size_t k          = 0;
        size_t batchSize  = 0;

        // Types. c and d are kept separate: the historical key used only c,
        // which silently merged mixed-precision problems where c != d.
        rocisa::DataType inputTypeA  = rocisa::DataType::None;
        rocisa::DataType inputTypeB  = rocisa::DataType::None;
        rocisa::DataType outputTypeC = rocisa::DataType::None;
        rocisa::DataType outputTypeD = rocisa::DataType::None;
        rocisa::DataType computeType = rocisa::DataType::None;
        int32_t          computeInputTypeA = 0;
        int32_t          computeInputTypeB = 0;

        // Layout
        size_t colStrideA   = 0;
        size_t colStrideB   = 0;
        size_t colStrideC   = 0;
        size_t colStrideD   = 0;
        size_t batchStrideA = 0;
        size_t batchStrideB = 0;
        size_t batchStrideC = 0;
        size_t batchStrideD = 0;
        int32_t batchMode   = 0;

        // Epilogue. The enum is kept whole rather than decomposed into an
        // activation type, because bias source, aux direction and gradient all
        // derive from it and a partial decomposition merges distinct problems.
        int32_t epilogue   = 0;
        bool    gradient   = false;
        int32_t biasType   = 0;
        int32_t biasStride = 0;
        bool    hasBias    = false;
        int32_t auxType    = 0;

        // Scaling. Formats matter as well as presence: a block-scaled problem
        // selects different kernels from a scalar-scaled one.
        int32_t scaleAFormat     = 0;
        int32_t scaleBFormat     = 0;
        bool    hasScaleA        = false;
        bool    hasScaleB        = false;
        bool    hasScaleC        = false;
        bool    hasScaleD        = false;
        bool    hasScaleE        = false;
        bool    hasScaleAlphaVec = false;
        bool    hasAmaxD         = false;

        // Kernel-shaping hints that change which solution is applicable
        bool    swizzleA               = false;
        bool    swizzleB               = false;
        int32_t streamkTileScheduling  = 0;
        int32_t smCountTarget          = 0;

        // Device identity. Entries are scoped to the architecture they were
        // measured on; replaying a gfx942 winner on gfx950 is meaningless.
        std::string archName;
        int32_t     cuCount = 0;

        /**
         * The single authoritative field list. Ordering and hashing both derive
         * from this, so a field added above is only actually part of the key
         * once it appears here.
         */
        auto key_tuple() const
        {
            return std::tie(transA,
                            transB,
                            m,
                            n,
                            k,
                            batchSize,
                            inputTypeA,
                            inputTypeB,
                            outputTypeC,
                            outputTypeD,
                            computeType,
                            computeInputTypeA,
                            computeInputTypeB,
                            colStrideA,
                            colStrideB,
                            colStrideC,
                            colStrideD,
                            batchStrideA,
                            batchStrideB,
                            batchStrideC,
                            batchStrideD,
                            batchMode,
                            epilogue,
                            gradient,
                            biasType,
                            biasStride,
                            hasBias,
                            auxType,
                            scaleAFormat,
                            scaleBFormat,
                            hasScaleA,
                            hasScaleB,
                            hasScaleC,
                            hasScaleD,
                            hasScaleE,
                            hasScaleAlphaVec,
                            hasAmaxD,
                            swizzleA,
                            swizzleB,
                            streamkTileScheduling,
                            smCountTarget,
                            archName,
                            cuCount);
        }

        /**
         * The key reduced to the fields the historical format actually keyed.
         *
         * A v0 row cannot populate the widened key: the columns simply are not
         * in the file, so they parse as defaults and the row then matches
         * nothing. Reducing both the stored row and the lookup to this same
         * subset reproduces the pre-existing matching behaviour exactly for old
         * files, without weakening the widened key for new ones.
         */
        ProblemOverride legacyKey() const
        {
            ProblemOverride k;
            k.transA      = transA;
            k.transB      = transB;
            k.m           = m;
            k.n           = n;
            k.k           = k_dim();
            k.batchSize   = batchSize;
            k.inputTypeA  = inputTypeA;
            k.inputTypeB  = inputTypeB;
            k.outputTypeC = outputTypeC;
            k.computeType = computeType;
            return k;
        }

    private:
        // `k` is shadowed by the local in legacyKey(); this keeps the accessor
        // unambiguous without renaming the public field.
        size_t k_dim() const { return k; }
    };

    /**
     * What a tuning file row resolves to.
     *
     * The historical map stored a bare solution index, which left nowhere to put
     * the solution name that per-entry validation needs.
     */
    struct TunedEntry
    {
        int32_t solutionIndex = -1;

        // Kernel identity. When either is present, replay resolves the index in
        // the current library and requires the name to still match before using
        // the entry; both absent means a legacy row with nothing to check but
        // the build stamp.
        //
        // kernelName is what new rows carry. It names the compiled kernel and
        // leaves out the solution-level defaults (GSU, staggerU, WGM) that
        // solutionName also encodes, so it is both shorter and stable across a
        // build that only retunes those defaults. solutionName is still read so
        // files written before the switch keep validating, against the field
        // they were actually written from.
        std::optional<std::string> kernelName;
        std::optional<std::string> solutionName;

        TuningSchemaVersion schemaVersion = TuningSchemaVersion::Legacy;
        TuningEntrySource   source        = TuningEntrySource::LegacyOverrideFile;

        // False when the per-shape budget stopped the search, so this winner is
        // the best of a prefix rather than of the whole candidate list.
        //
        // Such an entry is still worth keeping. Default selection's own pick is
        // forced to the front of the candidate list and is therefore always
        // measured first, so the best of any prefix is no slower than what the
        // shape runs untuned. What the flag buys is that the partial answer does
        // not become permanent: a later process whose budget can finish the
        // search treats the shape as still needing tuning and replaces the row.
        //
        // Absent from a row means complete, which is what every row written
        // before this column existed was.
        bool complete = true;

        // The per-shape ceiling in force when this row was written, in
        // milliseconds, with zero meaning unlimited and a negative value meaning
        // the row predates the column.
        //
        // Read only for an incomplete row, to answer whether this run could get
        // any further than the one that produced it. Without it a shape that
        // keeps truncating is re-benchmarked by every process that opens the
        // cache, each spending the whole ceiling to stop in exactly the same
        // place and append another identical row.
        int64_t budgetMs = -1;

        // Build that produced this row. Carried per row rather than only in the
        // file header because appending after an upgrade makes mixed-build files
        // the normal case.
        std::string buildStamp;

        // What the winner needed, so replay can reason about a caller whose
        // budget differs rather than only accept or reject.
        size_t requiredWorkspaceBytes = 0;

        // Measurement of the winner, which is persisted, and of what default
        // selection would have chosen at tuning time, which is not: the baseline
        // pair only carries the numbers from the benchmarker out to the level-4
        // diagnostic that compares them. Persisting a column nothing reads would
        // commit the file format to data with no consumer.
        double  winnerTimeUs   = 0.0;
        int32_t baselineIndex  = -1;
        double  baselineTimeUs = 0.0;
    };

    /**
     * Whether a ceiling of nowMs can get further than one of thenMs did.
     *
     * Zero is unlimited on either side, so an unlimited run beats any finite one
     * and nothing beats a previous unlimited run. A negative thenMs is a row
     * written before the ceiling was recorded: the answer is unknowable, so it
     * is worth one more attempt, and the row that attempt writes carries the
     * value from then on.
     */
    inline bool tuningBudgetIsMoreGenerous(int64_t nowMs, int64_t thenMs)
    {
        if(thenMs < 0)
            return true;
        if(thenMs == 0)
            return false;
        return nowMs == 0 || nowMs > thenMs;
    }

    std::optional<std::pair<ProblemOverride, TunedEntry>>
        problemFromEntries(const std::map<std::string, std::string>& row);

    /**
     * Append one tuned winner to the tuning file.
     *
     * Takes the problem rather than the key so the type columns can be written
     * in the spelling the parser reads back, and so a lossy key-to-string
     * inverse is never needed.
     *
     * Serialised behind a file-global mutex rather than any per-key guard: two
     * different shapes can finish tuning at the same moment and would otherwise
     * interleave their rows. That mutex only orders writers inside one process.
     * Concurrent writers in separate processes, such as the ranks of an MPI job
     * sharing a cache path, are not supported: the header and row go out as a
     * single write so a local filesystem will usually append them whole, but
     * nothing serialises the processes themselves.
     */
#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
    bool appendTunedEntry(const std::string&                 path,
                          const RocblasltContractionProblem& problem,
                          const TunedEntry&                  entry);
#endif

    /**
     * Load a tuning file into the map. Safe to call repeatedly; each distinct
     * path is parsed at most once.
     */
    void getContractionProblemsFromFile(const std::string& path);

#ifdef HIPBLASLT_ENABLE_TUNING_CACHE
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

#else // HIPBLASLT_ENABLE_TUNING_CACHE

    /**
     * Stubs for the shared lookup and validation paths, which stay compiled so
     * the offline override file keeps working, and which call into the tuning
     * bookkeeping above.
     *
     * recordTuningLookup feeds the closing summary, which does not exist here,
     * so it drops the call. recordTuningInvalidation deduplicates a rejected
     * entry across the several probes a tune-mode call makes; without tuning
     * there is only the heuristic lookup, so every sighting is the first and
     * the counter behaves exactly as it did before this feature existed.
     */
    inline void recordTuningLookup(const ProblemOverride&, bool) { }
    inline bool recordTuningInvalidation(const ProblemOverride&, int)
    {
        return true;
    }

#endif // HIPBLASLT_ENABLE_TUNING_CACHE

    template <>
    struct Comparison<ProblemOverride>
    {
        enum
        {
            implemented = true
        };

        static int compare(ProblemOverride const& lhs, ProblemOverride const& rhs)
        {
            auto l = lhs.key_tuple();
            auto r = rhs.key_tuple();
            if(l < r)
                return -1;
            if(r < l)
                return 1;
            return 0;
        }
    };

    class OverrideMap
    {
    public:
        static OverrideMap& getMap()
        {
            static OverrideMap gInstance;
            return gInstance;
        }

        OverrideMap() {}
        ~OverrideMap() {}
        // copy contructor
        OverrideMap(const OverrideMap&) = delete;
        // assignment operator
        OverrideMap& operator=(const OverrideMap&) = delete;

        /**
         * Entries in both maps.
         *
         * Callers use this as a cheap "is anything loaded at all" gate before
         * looking a key up. Counting only the widened map made that gate fire
         * for files consisting entirely of v0 rows, so their entries were never
         * consulted.
         */
        size_t size() const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            return m_override.size() + m_legacy.size();
        }

        /**
         * Copy out every entry matching a key.
         *
         * Returns values rather than iterators on purpose. The previous
         * signature returned an equal_range pair after its shared_lock had gone
         * out of scope, so every caller walked the multimap unlocked. That was
         * survivable while the map was written once at load; online tuning
         * inserts entries throughout the run, which turns it into a live race.
         */
        std::vector<TunedEntry> find(const ProblemOverride& prob_key) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            std::vector<TunedEntry> found;
            auto                    range = m_override.equal_range(prob_key);
            for(auto it = range.first; it != range.second; ++it)
                found.push_back(it->second);

            return found;
        }

        /**
         * Entries from v0 rows, keyed by the historical field subset.
         *
         * Kept in a separate map rather than mixed into the widened one so a
         * legacy row can never accidentally satisfy a lookup that differs in a
         * field the old format could not express.
         */
        std::vector<TunedEntry> findLegacy(const ProblemOverride& prob_key) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            std::vector<TunedEntry> found;
            auto                    range = m_legacy.equal_range(prob_key.legacyKey());
            for(auto it = range.first; it != range.second; ++it)
                found.push_back(it->second);

            return found;
        }

        /**
         * Two rows are the same entry only when the index and *both* identity
         * fields agree.
         *
         * Comparing the index and solutionName alone silently treated every
         * new-format row as identical to every other row with that index, since
         * new rows leave solutionName empty and carry kernelName instead. A
         * stale row then suppressed the fresh replacement written beside it,
         * which is the same append-only retune failure the index-only compare
         * caused before.
         */
        static bool sameEntry(const TunedEntry& a, const TunedEntry& b)
        {
            return a.solutionIndex == b.solutionIndex && a.kernelName == b.kernelName
                   && a.solutionName == b.solutionName;
        }

        bool addLegacyIfAbsent(const ProblemOverride& key, const TunedEntry& entry)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            const ProblemOverride narrow = key.legacyKey();
            auto                  range  = m_legacy.equal_range(narrow);
            for(auto it = range.first; it != range.second; ++it)
                if(sameEntry(it->second, entry))
                {
                    // The file is append-only, so the later row is the newer
                    // observation. Refresh metadata even when its kernel
                    // identity did not change: workspace requirements and build
                    // stamps can change, and keeping the older values makes the
                    // validity gate reject the row on every process start and
                    // append the same replacement forever.
                    it->second = entry;
                    return false;
                }

            m_legacy.emplace(narrow, entry);
            return true;
        }

        /**
         * Whether this key is worth benchmarking again to finish its search.
         *
         * Three things must hold. The key has entries at all, so this is not
         * simply an untuned shape. None of them finished, which is deliberately
         * "none is complete" rather than "any is partial": the file is
         * append-only, so the run that finishes a search appends its winner
         * beside the partial row instead of rewriting it, and a key holding both
         * has been tuned properly at least once. And this run's ceiling beats
         * every ceiling those rows were written under, since a run that cannot
         * get further would measure the same prefix, stop in the same place, and
         * append another identical row.
         *
         * Only the widened map is consulted: a legacy row has no completeness to
         * record and is never partial.
         */
        bool needsFinishing(const ProblemOverride& key, int64_t currentBudgetMs)
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            auto                                      range = m_override.equal_range(key);
            bool                                      any   = false;
            for(auto it = range.first; it != range.second; ++it)
            {
                any = true;
                if(it->second.complete)
                    return false;
                if(!tuningBudgetIsMoreGenerous(currentBudgetMs, it->second.budgetMs))
                    return false;
            }
            return any;
        }

        /**
         * Drop every entry for a key and install one.
         *
         * Needed for re-tuning: a shape whose entries all failed validation must
         * be replaceable, and addIfAbsent would refuse when the fresh winner
         * happens to land on the same solution index as the dead row.
         */
        void replaceAll(const ProblemOverride& key, const TunedEntry& entry)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.erase(key);
            m_override.emplace(key, entry);
        }

        void add(const ProblemOverride& key, const TunedEntry& entry)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.emplace(key, entry);
        }

        /**
         * Insert unless an identical entry already exists for this key, so
         * re-reading a file does not stack duplicates. An identical later row
         * refreshes the earlier row's metadata. Returns true only if a distinct
         * entry was inserted.
         *
         * Identity is the index *and* the recorded name, not the index alone.
         * The file is append-only, so a re-tuned shape leaves the superseded row
         * in place ahead of its replacement, and the two commonly share an index
         * because re-tuning often reaches the same kernel. Deduping on index
         * alone kept only the stale row, which then failed validation and made
         * the shape look untuned forever.
         */
        bool addIfAbsent(const ProblemOverride& key, const TunedEntry& entry)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            auto range = m_override.equal_range(key);
            for(auto it = range.first; it != range.second; ++it)
                if(sameEntry(it->second, entry))
                {
                    // Same append-order rule as the legacy map: identical
                    // identity means replace its metadata, not discard the
                    // newer row.
                    it->second = entry;
                    return false;
                }

            m_override.emplace(key, entry);
            return true;
        }

        /**
         * Claim a path for parsing.
         *
         * A concurrent caller for the same path waits for the active parser
         * rather than observing a partially populated map and retuning an entry
         * that is still later in the file.
         */
        bool claimLoad(const std::string& path)
        {
            std::unique_lock<std::shared_timed_mutex> lock(m_mutex);
            m_loadCv.wait(lock, [&] { return m_loadingPaths.count(path) == 0; });
            if(m_loadedPaths.count(path) != 0)
                return false;
            m_loadingPaths.insert(path);
            return true;
        }

        /**
         * Publish a completed load, or release a failed claim so a later call
         * can retry (for example after a transient open failure).
         */
        void finishLoad(const std::string& path, bool success)
        {
            {
                std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
                m_loadingPaths.erase(path);
                if(success)
                    m_loadedPaths.insert(path);
            }
            m_loadCv.notify_all();
        }

        /**
         * Drop everything loaded so far. Tests only: a single test process
         * exercises several cache files, and the per-path load latch would
         * otherwise make every file after the first invisible.
         */
        void resetForTest()
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.clear();
            m_legacy.clear();
            m_loadedPaths.clear();
            m_loadingPaths.clear();
            m_loadCv.notify_all();
        }

        bool isLoaded(const std::string& path) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            return m_loadedPaths.count(path) != 0;
        }

    private:
        std::multimap<ProblemOverride, TunedEntry> m_override;
        std::multimap<ProblemOverride, TunedEntry> m_legacy;
        std::set<std::string>                      m_loadedPaths;
        std::set<std::string>                      m_loadingPaths;
        std::condition_variable_any                m_loadCv;
        mutable std::shared_timed_mutex            m_mutex;
    };
} // namespace TensileLite

namespace std
{
    template <>
    struct hash<TensileLite::ProblemOverride>
    {
        inline size_t operator()(TensileLite::ProblemOverride const& po) const
        {
            return std::apply([](auto const&... field) { return TensileLite::hash_combine(field...); },
                              po.key_tuple());
        }
    };
} // namespace std
