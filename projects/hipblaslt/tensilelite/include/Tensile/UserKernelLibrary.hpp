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
#include <vector>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/UserKernelIndex.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /// [M, N, batch, K], the same key as the shipped Equality tables.
    struct UserExactShape
    {
        size_t m     = 0;
        size_t n     = 0;
        size_t batch = 0;
        size_t k     = 0;

        bool operator==(UserExactShape const& rhs) const
        {
            return m == rhs.m && n == rhs.n && batch == rhs.batch && k == rhs.k;
        }
    };

    /**
     * Exact match key: the shape, the transposes and the data types. Mapping a
     * kernel takes the transposes and types from the kernel, so mappings for
     * different data types or transposes at one shape never displace each
     * other.
     */
    struct UserExactKey
    {
        UserExactShape   shape;
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
        bool operator==(UserExactKey const& rhs) const
        {
            return tie() == rhs.tie();
        }
        bool operator!=(UserExactKey const& rhs) const
        {
            return !(*this == rhs);
        }
    };

    struct UserExactKeyHash
    {
        size_t operator()(UserExactKey const& key) const noexcept
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
    inline UserExactShape userExactShapeFor(MyProblem const& problem)
    {
        return UserExactShape{
            problem.freeSizeA(0), problem.freeSizeB(0), problem.batchSize(0), problem.boundSize(0)};
    }

    /// The key a problem is looked up by.
    template <typename MyProblem>
    inline UserExactKey userExactKeyFor(MyProblem const& problem)
    {
        return UserExactKey{userExactShapeFor(problem),
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
    inline UserExactKey userExactKeyFor(UserExactShape const& shape, ProblemType const& pt)
    {
        return UserExactKey{
            shape, pt.transA, pt.transB, pt.aType, pt.bType, pt.cType, pt.dType, pt.computeType};
    }

    /**
     * Immutable state of the user kernel library. Each change publishes a new
     * snapshot, so readers never see a partial update.
     *
     * `solutions` holds every registered kernel and serves enumeration and
     * dispatch by index. `exactTable` holds only kernels mapped to a shape and
     * serves selection, so registering a kernel does not change selection.
     *
     * The key leaves out the rest of the problem type, such as the epilogue,
     * so kernels that differ only there share a key. Each key keeps a list,
     * newest mapping first, and selection takes the first kernel that can
     * serve the problem.
     */
    template <typename MySolution>
    struct UserKernelSnapshot
    {
        using Candidates = std::vector<std::shared_ptr<MySolution>>;

        std::map<int, std::shared_ptr<MySolution>>                     solutions;
        std::unordered_map<UserExactKey, Candidates, UserExactKeyHash> exactTable;
    };

    /**
     * Kernels registered at runtime. The top-level MasterSolutionLibrary
     * consults this once per query, ahead of the shipped library, so a mapped
     * kernel is chosen for its shape regardless of how the shipped library is
     * split into shards.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class UserKernelLibrary
    {
    public:
        using Snapshot = UserKernelSnapshot<MySolution>;

        UserKernelLibrary()
            : m_snapshot(std::make_shared<const Snapshot>())
        {
        }

        UserKernelLibrary(UserKernelLibrary const&)            = delete;
        UserKernelLibrary& operator=(UserKernelLibrary const&) = delete;

        /**
         * The newest kernel mapped under this problem's key that can serve
         * it, or null. Tagged Equal, like a shipped exact match, so the caller
         * reports it as tuned.
         */
        std::shared_ptr<MySolution> findExact(MyProblem const& problem,
                                              Hardware const&  hardware) const
        {
            if(!m_hasExact.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->exactTable.find(userExactKeyFor(problem));
            if(it == snap->exactTable.end())
                return std::shared_ptr<MySolution>();

            for(auto const& solution : it->second)
            {
                if(canServe(*solution, problem, hardware))
                {
                    solution->tag = MySolution::MatchingTag::Equal;
                    return solution;
                }
            }
            return std::shared_ptr<MySolution>();
        }

        /**
         * Registered kernels, mapped or not, that match the problem as far as
         * `searchType` asks: every check for DEFAULT, the GEMM type for
         * GEMM_TYPE_ONLY, the hardware alone for HARDWARE_ONLY.
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
                    match = canServe(solution, problem, hardware);
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
            if(!isUserKernelIndex(index) || !m_hasAny.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->solutions.find(index);
            return it == snap->solutions.end() ? std::shared_ptr<MySolution>() : it->second;
        }

        /**
         * Registers solutions and returns their indices in order. Each solution
         * gets a new index in the user range, written back to
         * `solution->index`. Indices are valid only in this process.
         *
         * Registers all of them or, if the user range cannot hold them all,
         * none and returns an empty vector.
         */
        std::vector<int> addSolutions(std::vector<std::shared_ptr<MySolution>> const& incoming)
        {
            std::vector<std::shared_ptr<MySolution>> valid;
            for(auto const& solution : incoming)
            {
                if(solution)
                    valid.push_back(solution);
            }

            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);

            constexpr int64_t indexEnd = int64_t(std::numeric_limits<int>::max()) + 1;
            if(m_nextIndex + int64_t(valid.size()) > indexEnd)
                return {};

            auto             next = std::make_shared<Snapshot>(*m_snapshot);
            std::vector<int> assigned;
            assigned.reserve(valid.size());
            for(auto const& solution : valid)
            {
                const int index        = static_cast<int>(m_nextIndex++);
                solution->index        = index;
                next->solutions[index] = solution;
                assigned.push_back(index);
            }

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
        bool setExactMatch(UserExactShape const& shape, int index)
        {
            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);

            auto it = m_snapshot->solutions.find(index);
            if(it == m_snapshot->solutions.end())
                return false;

            auto  next       = std::make_shared<Snapshot>(*m_snapshot);
            auto& candidates = next->exactTable[userExactKeyFor(shape, it->second->problemType)];
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
         * Whether a registered kernel can serve this problem, by the same test
         * the shipped exact-match rows apply. The shipped library reaches a
         * solution only through rows that already matched the problem type;
         * nothing does that here, so the solution's problem predicate, which
         * includes its problem type, is what rejects a kernel of the wrong
         * type. The explicit checks reject the common mismatches first.
         */
        static bool
            canServe(MySolution const& solution, MyProblem const& problem, Hardware const& hardware)
        {
            auto const& pt = solution.problemType;

            if(pt.transA != problem.transA() || pt.transB != problem.transB())
                return false;

            if(pt.aType != problem.a().dataType() || pt.bType != problem.b().dataType()
               || pt.cType != problem.c().dataType() || pt.dType != problem.d().dataType())
                return false;

            if(pt.computeType != problem.computeType())
                return false;

            if(pt.highPrecisionAccumulate != problem.highPrecisionAccumulate())
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

        int64_t m_nextIndex = UserKernelIndexBase;
    };

    /**
     * The process-wide user kernel library. Defined only for the GEMM problem,
     * and out of line in tensilelite-host, so every shared library that links
     * it, hipblaslt included, sees the same instance. A function-local static
     * in a header would give each shared library its own copy, because they
     * build with hidden visibility.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    std::shared_ptr<UserKernelLibrary<MyProblem, MySolution>> globalUserKernelLibrary();

    template <>
    TENSILELITEHOST_EXPORT
        std::shared_ptr<UserKernelLibrary<ContractionProblemGemm, ContractionSolution>>
        globalUserKernelLibrary<ContractionProblemGemm, ContractionSolution>();
} // namespace TensileLite
