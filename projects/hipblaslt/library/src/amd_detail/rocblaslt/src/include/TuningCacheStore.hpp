// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The tuning file's row format and in-memory store: the problem key, what a row
// resolves to, reading and writing rows, and the map replay consults. Nothing
// here touches a device, the solution library or the logger, so the same code
// builds into hipBLASLt and into a host-only test.

// Standard headers first: Tensile/Comparison.hpp uses size_t, std::hash and
// std::tuple without including them.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Tensile/Comparison.hpp>
#include <rocisa/include/enum.hpp>

namespace TensileLite
{
    /**
     * Row schema version.
     *
     * Legacy rows are what hipblaslt-bench writes: the ten problem columns the
     * historical key used, solution_index, and no schema_version column. They
     * have no leading dimensions, strides or epilogue, so they can only be
     * matched on the historical fields; see ProblemOverride::legacyKey.
     */
    enum class TuningSchemaVersion : uint32_t
    {
        Legacy  = 0,
        Current = 1,
    };

    /**
     * The problem key.
     *
     * Built from plain scalars rather than rocblaslt or HIP types so that it can
     * be constructed and compared in a unit test without a solution library or
     * a device. The builder from RocblasltContractionProblem lives in
     * tensile_host.cpp.
     *
     * Every field of RocblasltContractionProblem is either keyed below or
     * deliberately left out:
     *
     *   A, B, C, D, E, batch_A..batch_D, bias, scaleA..scaleE, scaleAlphaVec,
     *   amaxD, workspace, Synchronizer
     *       Device addresses. Only their presence is keyed, never the value.
     *
     *   C == D (aliasing)
     *       construct_rocblaslt_problem passes null for both C and D, so at
     *       heuristic time every problem looks in-place. The value carries no
     *       information at lookup.
     *
     *   alpha, beta, alpha_owned
     *       assignAlphaBeta1 sets both to one at heuristic time, so the key
     *       cannot represent them, and an entry serves callers whatever their
     *       alpha and beta.
     *
     *   pointer alignment
     *       No solution predicate consumes it.
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

        // Orientation and shape. transA and transB mean "not N", which is all
        // the historical key recorded; conjugateA and conjugateB then separate
        // a conjugate transpose from a plain one, which select different
        // kernels for complex types.
        bool   transA     = false;
        bool   transB     = false;
        bool   conjugateA = false;
        bool   conjugateB = false;
        size_t m          = 0;
        size_t n          = 0;
        size_t k          = 0;
        size_t batchSize  = 0;

        // Types. c and d are kept separate: the historical key used only c,
        // which merged mixed-precision problems where c != d.
        rocisa::DataType inputTypeA        = rocisa::DataType::None;
        rocisa::DataType inputTypeB        = rocisa::DataType::None;
        rocisa::DataType outputTypeC       = rocisa::DataType::None;
        rocisa::DataType outputTypeD       = rocisa::DataType::None;
        rocisa::DataType computeType       = rocisa::DataType::None;
        int32_t          computeInputTypeA = 0;
        int32_t          computeInputTypeB = 0;

        // Layout
        size_t  colStrideA   = 0;
        size_t  colStrideB   = 0;
        size_t  colStrideC   = 0;
        size_t  colStrideD   = 0;
        size_t  batchStrideA = 0;
        size_t  batchStrideB = 0;
        size_t  batchStrideC = 0;
        size_t  batchStrideD = 0;
        size_t  colStrideE   = 0;
        size_t  batchStrideE = 0;
        int32_t batchMode    = 0;

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
        bool    swizzleA              = false;
        bool    swizzleB              = false;
        int32_t streamkTileScheduling = 0;
        int32_t smCountTarget         = 0;
        bool    uniformSummationOrder = false;

        // Device identity. Entries are scoped to the device they were measured
        // on; replaying a gfx942 winner on gfx950 is meaningless.
        std::string archName;
        int32_t     cuCount = 0;

        /**
         * The single authoritative field list. Ordering and hashing both derive
         * from this, so a field added above is only part of the key once it
         * appears here.
         */
        auto key_tuple() const
        {
            return std::tie(transA,
                            transB,
                            conjugateA,
                            conjugateB,
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
                            colStrideE,
                            batchStrideE,
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
                            uniformSummationOrder,
                            archName,
                            cuCount);
        }

        /**
         * The key reduced to the fields legacy rows record.
         *
         * A legacy row cannot fill in the rest: those columns are not in the
         * file, so they would parse as defaults and the row would match
         * nothing. Reducing both the stored row and the lookup to this subset
         * matches legacy rows exactly as before, without weakening the key for
         * current rows.
         */
        ProblemOverride legacyKey() const
        {
            ProblemOverride narrow;
            narrow.transA      = transA;
            narrow.transB      = transB;
            narrow.m           = m;
            narrow.n           = n;
            narrow.k           = k;
            narrow.batchSize   = batchSize;
            narrow.inputTypeA  = inputTypeA;
            narrow.inputTypeB  = inputTypeB;
            narrow.outputTypeC = outputTypeC;
            narrow.computeType = computeType;
            return narrow;
        }
    };

    /**
     * What a tuning file row resolves to.
     *
     * A solution index is only a position in one build's kernel library, so on
     * its own it cannot tell whether it still names the kernel that was tuned.
     * The name recorded beside it is what lets replay check that: kernel_name
     * in files written now, solution_name in some older files, and neither in
     * the oldest, which are trusted only when they were written by the running
     * build.
     */
    struct TunedEntry
    {
        int32_t                    solutionIndex = -1;
        std::optional<std::string> kernelName;
        std::optional<std::string> solutionName;

        TuningSchemaVersion schemaVersion = TuningSchemaVersion::Legacy;

        // The build that wrote the row: its git_version column, or the file's
        // "Git Version:" line for rows without one. Kept per row because a file
        // appended to across an upgrade holds rows from more than one build.
        std::string buildStamp;

        // What the winner needed and how fast it ran, as recorded when it was
        // tuned.
        size_t requiredWorkspaceBytes = 0;
        double winnerTimeUs           = 0.0;

        /**
         * Two rows are the same entry only when the index and both names
         * agree. Two rows can share an index while naming different kernels,
         * and only one of them can still be valid.
         */
        bool sameIdentity(const TunedEntry& other) const
        {
            return solutionIndex == other.solutionIndex && kernelName == other.kernelName
                   && solutionName == other.solutionName;
        }
    };

    /** One row, zipped from its header and value lines, as a key and an entry. */
    std::optional<std::pair<ProblemOverride, TunedEntry>>
        problemFromEntries(const std::map<std::string, std::string>& row);

    /**
     * The type columns in the spelling the parser reads back. The key cannot
     * supply them: it keeps Tensile types, and several compute types share one.
     */
    struct TuningRowTypes
    {
        std::string a;
        std::string b;
        std::string c;
        std::string d;
        std::string compute;
    };

    /**
     * One entry as a header line and a value line, ready to append.
     *
     * Every column the key holds is written from the key, so what is written is
     * exactly what a later lookup rebuilds. A key field missing from the row
     * would parse back as its default and never match.
     */
    std::string formatTuningRow(const ProblemOverride& key,
                                const TuningRowTypes&  types,
                                const TunedEntry&      entry,
                                const std::string&     buildStamp);

    /**
     * Append a row from formatTuningRow, starting the file with its build stamp
     * if it is new or empty.
     *
     * Writers inside one process are serialised. Separate processes appending
     * to the same file, such as the ranks of an MPI job sharing a path, are not
     * supported: each row goes out as a single write, which a local filesystem
     * will usually append whole, but nothing orders the processes themselves.
     *
     * Returns false when the row did not reach the file.
     */
    bool appendTuningRow(const std::string& path,
                         const std::string& row,
                         const std::string& buildStamp);

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

        OverrideMap() = default;

        OverrideMap(const OverrideMap&)            = delete;
        OverrideMap& operator=(const OverrideMap&) = delete;

        /** Entries in both maps. */
        size_t size() const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            return m_override.size() + m_legacy.size();
        }

        /**
         * Copy out every current-schema entry for a key, in file order. Copies,
         * so no caller walks the multimap outside the lock.
         */
        std::vector<TunedEntry> find(const ProblemOverride& key) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            std::vector<TunedEntry> found;
            auto                    range = m_override.equal_range(key);
            for(auto it = range.first; it != range.second; ++it)
                found.push_back(it->second);
            return found;
        }

        /**
         * Entries from legacy rows, matched on the key's legacy subset, in file
         * order.
         *
         * Kept in a separate map rather than mixed into the full one, so a
         * legacy row can never satisfy a lookup that differs in a field the old
         * format could not express.
         */
        std::vector<TunedEntry> findLegacy(const ProblemOverride& key) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            std::vector<TunedEntry> found;
            auto                    range = m_legacy.equal_range(key.legacyKey());
            for(auto it = range.first; it != range.second; ++it)
                found.push_back(it->second);
            return found;
        }

        /**
         * Insert unless this key already holds the same entry, so a file that
         * repeats a row does not stack duplicates. The repeat is the later row
         * of an append-only file, so it refreshes the stored metadata. Returns
         * true only when a distinct entry was inserted.
         */
        bool addIfAbsent(const ProblemOverride& key, const TunedEntry& entry)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            return addTo(m_override, key, entry);
        }

        /** addIfAbsent for a legacy row, filed under the key's legacy subset. */
        bool addLegacyIfAbsent(const ProblemOverride& key, const TunedEntry& entry)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            return addTo(m_legacy, key.legacyKey(), entry);
        }

        /**
         * Whether a path has already been read. Tracked per path rather than
         * inferred from the map, so a file that yields no usable rows is still
         * read only once.
         */
        bool isLoaded(const std::string& path) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
            return m_loaded.count(path) != 0;
        }

        void markLoaded(const std::string& path)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_loaded.insert(path);
        }

        /** Held for the whole of a load, so a path is parsed by one caller at a time. */
        std::mutex& getLock()
        {
            return m_guard;
        }

        void resetForTest()
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            m_override.clear();
            m_legacy.clear();
            m_loaded.clear();
        }

    private:
        using Entries = std::multimap<ProblemOverride, TunedEntry>;

        static bool addTo(Entries& entries, const ProblemOverride& key, const TunedEntry& entry)
        {
            auto range = entries.equal_range(key);
            for(auto it = range.first; it != range.second; ++it)
                if(it->second.sameIdentity(entry))
                {
                    it->second = entry;
                    return false;
                }

            entries.emplace(key, entry);
            return true;
        }

        Entries                         m_override;
        Entries                         m_legacy;
        std::set<std::string>           m_loaded;
        std::mutex                      m_guard;
        mutable std::shared_timed_mutex m_mutex;
    };

    struct TuningRowsLoaded
    {
        // Distinct entries added to the map; a repeated row refreshes rather
        // than adds.
        uint64_t accepted = 0;

        // Rows without a kernel or solution name that were left out because
        // they were not written by the running build.
        uint64_t skippedUnnamed = 0;

        // The file's "Git Version:" line, or empty when it has none.
        std::string fileBuildStamp;

        // The stream went bad partway, so the file may hold rows that were
        // never reached.
        bool readError = false;
    };

    /**
     * Read every row of a tuning file into the map.
     *
     * A row without a kernel or solution name has nothing to validate it at
     * replay, so it is kept only when its build stamp, or the file's, matches
     * currentBuildStamp. An empty currentBuildStamp matches nothing.
     */
    TuningRowsLoaded
        loadTuningRows(std::istream& in, OverrideMap& map, const std::string& currentBuildStamp);
} // namespace TensileLite

namespace std
{
    template <>
    struct hash<TensileLite::ProblemOverride>
    {
        inline size_t operator()(TensileLite::ProblemOverride const& po) const
        {
            return std::apply(
                [](auto const&... field) { return TensileLite::hash_combine(field...); },
                po.key_tuple());
        }
    };
} // namespace std
