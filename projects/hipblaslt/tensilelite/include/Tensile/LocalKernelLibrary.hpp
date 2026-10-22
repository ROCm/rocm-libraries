// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/LocalKernelIndex.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Task.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /// [M, N, batch, K], the same key as the shipped Equality tables.
    struct LocalExactShape
    {
        size_t m     = 0;
        size_t n     = 0;
        size_t batch = 0;
        size_t k     = 0;

        bool operator==(LocalExactShape const& rhs) const
        {
            return m == rhs.m && n == rhs.n && batch == rhs.batch && k == rhs.k;
        }
    };

    /**
     * Exact match key: the shape, the transposes and the data types. Mapping a
     * kernel takes the transposes and types from the kernel, so mappings for
     * different data types or transposes at one shape never displace each
     * other.
     *
     * The key only narrows the candidates. Whether a kernel serves a problem
     * is decided by its own problem predicate, which TensileLite generates
     * from its full problem type, so fields left out here (grouped GEMM, MX
     * block sizes, epilogue, and any added later) are still checked.
     */
    struct LocalExactKey
    {
        LocalExactShape  shape;
        bool             transA      = false;
        bool             transB      = false;
        rocisa::DataType aType       = rocisa::DataType::Float;
        rocisa::DataType bType       = rocisa::DataType::Float;
        rocisa::DataType cType       = rocisa::DataType::Float;
        rocisa::DataType dType       = rocisa::DataType::Float;
        rocisa::DataType computeType = rocisa::DataType::Float;

        auto tie() const
        {
            return std::forward_as_tuple(shape.m,
                                         shape.n,
                                         shape.batch,
                                         shape.k,
                                         transA,
                                         transB,
                                         aType,
                                         bType,
                                         cType,
                                         dType,
                                         computeType);
        }
        bool operator==(LocalExactKey const& rhs) const
        {
            return tie() == rhs.tie();
        }
        bool operator!=(LocalExactKey const& rhs) const
        {
            return !(*this == rhs);
        }
    };

    struct LocalExactKeyHash
    {
        size_t operator()(LocalExactKey const& key) const noexcept
        {
            size_t h = 0;
            for(size_t v : {key.shape.m,
                            key.shape.n,
                            key.shape.batch,
                            key.shape.k,
                            size_t(key.transA),
                            size_t(key.transB),
                            size_t(key.aType),
                            size_t(key.bType),
                            size_t(key.cType),
                            size_t(key.dType),
                            size_t(key.computeType)})
                h ^= std::hash<size_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    template <typename MyProblem>
    inline LocalExactShape localExactShapeFor(MyProblem const& problem)
    {
        return LocalExactShape{
            problem.freeSizeA(0), problem.freeSizeB(0), problem.batchSize(0), problem.boundSize(0)};
    }

    /// The key a problem is looked up by.
    template <typename MyProblem>
    inline LocalExactKey localExactKeyFor(MyProblem const& problem)
    {
        return LocalExactKey{localExactShapeFor(problem),
                             problem.transA(),
                             problem.transB(),
                             problem.a().dataType(),
                             problem.b().dataType(),
                             problem.c().dataType(),
                             problem.d().dataType(),
                             problem.computeType()};
    }

    /// The key a kernel of this problem type is mapped under at `shape`.
    template <typename ProblemType>
    inline LocalExactKey localExactKeyFor(LocalExactShape const& shape, ProblemType const& pt)
    {
        return LocalExactKey{
            shape, pt.transA, pt.transB, pt.aType, pt.bType, pt.cType, pt.dType, pt.computeType};
    }

    /**
     * Immutable state of the local kernel library. Each change publishes a new
     * snapshot, so readers never see a partial update.
     *
     * `solutions` holds every registered kernel and serves enumeration and
     * dispatch by index. `exactTable` holds only kernels mapped to a shape and
     * serves selection, so registering a kernel does not change selection.
     *
     * Kernels that differ only in fields outside the key share it. Each key
     * keeps a list, newest mapping first, and selection takes the first kernel
     * that can serve the problem.
     */
    template <typename MySolution>
    struct LocalKernelSnapshot
    {
        using Candidates = std::vector<std::shared_ptr<MySolution>>;

        std::map<int, std::shared_ptr<MySolution>>                       solutions;
        std::unordered_map<LocalExactKey, Candidates, LocalExactKeyHash> exactTable;
    };

    /**
     * Kernels registered at runtime. The top-level MasterSolutionLibrary
     * consults this once per GEMM query (findBestSolution, findTopSolutions
     * and findAllSolutions), ahead of the shipped library, so a mapped kernel
     * is chosen for its shape regardless of how the shipped library is split
     * into shards.
     *
     * Grouped GEMM is not covered yet: findAllSolutionsGroupedGemm and
     * findTopSolutionsGroupedGemm query the shipped library alone, so a
     * grouped GEMM kernel registered and mapped here is never listed or
     * chosen.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class LocalKernelLibrary
    {
    public:
        using Snapshot = LocalKernelSnapshot<MySolution>;

        LocalKernelLibrary()
            : m_snapshot(std::make_shared<const Snapshot>())
        {
        }

        LocalKernelLibrary(LocalKernelLibrary const&)            = delete;
        LocalKernelLibrary& operator=(LocalKernelLibrary const&) = delete;

        /**
         * The newest kernel mapped under this problem's key that can serve
         * it, or null. It is tagged Equal, like a shipped exact match, so the
         * caller reports it as tuned.
         */
        std::shared_ptr<MySolution> findExact(MyProblem const& problem,
                                              Hardware const&  hardware) const
        {
            if(!m_hasExact.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->exactTable.find(localExactKeyFor(problem));
            if(it == snap->exactTable.end())
                return std::shared_ptr<MySolution>();

            for(auto const& solution : it->second)
            {
                if(canServe(*solution, problem, hardware))
                    return solution;
            }
            return std::shared_ptr<MySolution>();
        }

        /**
         * Registered kernels, mapped or not, that match the problem as far as
         * `searchType` asks, with the filters the shipped libraries apply: the
         * problem type, hardware and software predicates for DEFAULT, the GEMM
         * type for GEMM_TYPE_ONLY, the hardware alone for HARDWARE_ONLY.
         */
        SolutionSet<MySolution> findAll(MyProblem const&          problem,
                                        Hardware const&           hardware,
                                        SolutionLibrarySearchType searchType
                                        = SolutionLibrarySearchType::DEFAULT) const
        {
            SolutionSet<MySolution> rv;
            if(!m_hasAny.load(std::memory_order_acquire))
                return rv;

            auto snap = current();
            for(auto const& entry : snap->solutions)
            {
                auto const& solution = *entry.second;
                bool        match    = false;
                switch(searchType)
                {
                case SolutionLibrarySearchType::DEFAULT:
                    if(matchesProblemType(solution, problem))
                    {
                        Task task(hardware, problem, solution);
                        match = (*solution.hardwarePredicate)(hardware)
                                && softwarePredicate(searchType, task, hardware, solution, problem);
                    }
                    break;
                case SolutionLibrarySearchType::GEMM_TYPE_ONLY:
                    match = (*solution.hardwarePredicate)(hardware)
                            && isGemmTypeSame(solution, problem);
                    break;
                case SolutionLibrarySearchType::HARDWARE_ONLY:
                    match = (*solution.hardwarePredicate)(hardware);
                    break;
                default:
                    break;
                }
                if(match)
                    rv.insert(entry.second);
            }
            return rv;
        }

        /// Any registered kernel by index, mapped or not.
        std::shared_ptr<MySolution> lookupIndex(int index) const
        {
            if(!isLocalKernelIndex(index) || !m_hasAny.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->solutions.find(index);
            return it == snap->solutions.end() ? std::shared_ptr<MySolution>() : it->second;
        }

        /**
         * Registers solutions and returns their indices in order. Each solution
         * gets a new index in the local range, written back to
         * `solution->index`. Indices are valid only in this process.
         *
         * The library keeps and modifies the objects passed in, so it takes
         * only solutions with no index yet (a negative one). A solution the
         * shipped library, a JIT library or this one has already indexed is
         * shared with that library and is rejected, not renumbered.
         *
         * Callers must finish initializing metadata and predicates and have
         * exclusive access to the incoming objects during registration.
         * After registration, they must not change the index,
         * tag, problem type, launch metadata or predicates, including through
         * pointers returned by lookups. Snapshots share these objects; they
         * do not copy them or synchronize changes to their fields.
         *
         * Each is also tagged Equal here, before it is published, because a
         * local kernel is selected only through an exact mapping; selection then
         * never writes to a solution that concurrent lookups share.
         *
         * Registers all of them, or none and returns an empty vector if any is
         * null, repeated, or already has an index, or if the local range cannot
         * hold them all.
         */
        std::vector<int> addSolutions(std::vector<std::shared_ptr<MySolution>> const& incoming)
        {
            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);

            constexpr int64_t indexEnd = int64_t(std::numeric_limits<int>::max()) + 1;
            if(incoming.empty() || m_nextIndex + int64_t(incoming.size()) > indexEnd)
                return {};

            std::unordered_set<MySolution const*> seen;
            for(auto const& solution : incoming)
            {
                if(!solution || solution->index >= 0 || !seen.insert(solution.get()).second)
                    return {};
            }

            auto             next = std::make_shared<Snapshot>(*m_snapshot);
            std::vector<int> assigned;
            assigned.reserve(incoming.size());
            for(auto const& solution : incoming)
            {
                const int index        = static_cast<int>(m_nextIndex + int64_t(assigned.size()));
                next->solutions[index] = solution;
                assigned.push_back(index);
            }

            // Map insertion can throw. Finish allocating before modifying the
            // caller's solutions or consuming indices, so a failed call can retry.
            for(size_t i = 0; i < incoming.size(); ++i)
            {
                incoming[i]->index = assigned[i];
                incoming[i]->tag   = MySolution::MatchingTag::Equal;
            }
            m_nextIndex += int64_t(assigned.size());

            publishLocked(std::move(next));
            return assigned;
        }

        /**
         * Maps a registered kernel to one exact shape, under its own transposes
         * and data types, making it selectable for the problems it can serve
         * there. It takes priority over kernels mapped under the same key
         * before; mapping it again moves it back to the front. Returns false if
         * the index is not registered.
         */
        bool setExactMatch(LocalExactShape const& shape, int index)
        {
            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);

            auto it = m_snapshot->solutions.find(index);
            if(it == m_snapshot->solutions.end())
                return false;

            auto  next       = std::make_shared<Snapshot>(*m_snapshot);
            auto& candidates = next->exactTable[localExactKeyFor(shape, it->second->problemType)];
            candidates.erase(std::remove(candidates.begin(), candidates.end(), it->second),
                             candidates.end());
            candidates.insert(candidates.begin(), it->second);

            publishLocked(std::move(next));
            return true;
        }

        size_t registeredCount() const
        {
            return current()->solutions.size();
        }

        /// Number of keys (shape, transposes and data types) with a kernel
        /// mapped.
        size_t selectableCount() const
        {
            return current()->exactTable.size();
        }

    private:
        std::shared_ptr<const Snapshot> current() const
        {
            std::shared_lock<std::shared_mutex> lock(m_snapshotMutex);
            return m_snapshot;
        }

        /// Caller must hold m_snapshotMutex exclusively.
        void publishLocked(std::shared_ptr<const Snapshot> next)
        {
            const bool hasAny   = !next->solutions.empty();
            const bool hasExact = !next->exactTable.empty();

            m_snapshot = std::move(next);
            m_hasAny.store(hasAny, std::memory_order_release);
            m_hasExact.store(hasExact, std::memory_order_release);
        }

        /**
         * The common problem type mismatches, rejected before any predicate
         * runs. The shipped library reaches a solution only through rows that
         * already matched the problem type; nothing does that here, so the
         * solution's problem predicate, which includes its problem type, is
         * what rejects the rest.
         */
        static bool matchesProblemType(MySolution const& solution, MyProblem const& problem)
        {
            auto const& pt = solution.problemType;

            if(pt.transA != problem.transA() || pt.transB != problem.transB())
                return false;

            if(pt.aType != problem.a().dataType() || pt.bType != problem.b().dataType()
               || pt.cType != problem.c().dataType() || pt.dType != problem.d().dataType())
                return false;

            if(pt.computeType != problem.computeType())
                return false;

            return pt.highPrecisionAccumulate == problem.highPrecisionAccumulate();
        }

        /// Whether a registered kernel can serve this problem, by the same
        /// test the shipped exact-match rows apply.
        static bool
            canServe(MySolution const& solution, MyProblem const& problem, Hardware const& hardware)
        {
            if(!matchesProblemType(solution, problem))
                return false;

            Task task(hardware, problem, solution);
            return selectionPredicate(task, hardware, solution, problem);
        }

        // Guards the pointer only; readers copy it and release the lock.
        // std::atomic<std::shared_ptr> is not usable with every libstdc++ we
        // build with.
        mutable std::shared_mutex       m_snapshotMutex;
        std::shared_ptr<const Snapshot> m_snapshot;

        // Checked before taking the lock, so lookups cost no lock traffic while
        // nothing is registered.
        std::atomic<bool> m_hasAny{false};
        std::atomic<bool> m_hasExact{false};

        int64_t m_nextIndex = LocalKernelIndexBase;
    };

    /**
     * The process-wide local kernel library. Defined only for the GEMM problem,
     * and out of line in tensilelite-host, so every shared library that links
     * it, hipblaslt included, sees the same instance. A function-local static
     * in a header would give each shared library its own copy, because they
     * build with hidden visibility.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    std::shared_ptr<LocalKernelLibrary<MyProblem, MySolution>> globalLocalKernelLibrary();

    template <>
    TENSILELITEHOST_EXPORT
        std::shared_ptr<LocalKernelLibrary<ContractionProblemGemm, ContractionSolution>>
        globalLocalKernelLibrary<ContractionProblemGemm, ContractionSolution>();
} // namespace TensileLite
