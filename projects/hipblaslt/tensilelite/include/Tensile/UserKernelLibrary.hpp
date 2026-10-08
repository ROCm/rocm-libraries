// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <Tensile/LibraryGeneration.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/UserKernelIndex.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
     * Exact match key: [M, N, batch, K], the same key as the shipped Equality
     * tables.
     */
    struct UserExactKey
    {
        size_t m     = 0;
        size_t n     = 0;
        size_t batch = 0;
        size_t k     = 0;

        bool operator==(UserExactKey const& rhs) const
        {
            return m == rhs.m && n == rhs.n && batch == rhs.batch && k == rhs.k;
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
            for(size_t v : {key.m, key.n, key.batch, key.k})
                h ^= std::hash<size_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    template <typename MyProblem>
    inline UserExactKey userExactKeyFor(MyProblem const& problem)
    {
        return UserExactKey{
            problem.freeSizeA(0), problem.freeSizeB(0), problem.batchSize(0), problem.boundSize(0)};
    }

    /**
     * Immutable state of the user kernel library. Each change publishes a new
     * snapshot, so readers never see a partial update.
     *
     * `solutions` holds every registered kernel and serves enumeration and
     * dispatch by index. `exactTable` holds only kernels mapped to a shape and
     * serves selection, so registering a kernel does not change selection.
     */
    template <typename MySolution>
    struct UserTierSnapshot
    {
        std::map<int, std::shared_ptr<MySolution>>                                      solutions;
        std::unordered_map<UserExactKey, std::shared_ptr<MySolution>, UserExactKeyHash> exactTable;
    };

    /**
     * Runtime-registered kernels, at the head of every selection row ladder.
     * The row is fixed at load time; all changes happen here by replacing the
     * snapshot.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct UserKernelLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Snapshot = UserTierSnapshot<MySolution>;

        UserKernelLibrary(std::shared_ptr<LibraryGeneration> generation = nullptr)
            : m_generation(generation ? generation : std::make_shared<LibraryGeneration>())
            , m_snapshot(std::make_shared<const Snapshot>())
        {
        }

        static std::string Type()
        {
            return "UserKernel";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            auto snap = current();
            return concatenate(type(),
                               " (",
                               snap->solutions.size(),
                               " registered, ",
                               snap->exactTable.size(),
                               " selectable)");
        }

        // Selection considers only kernels with an exact mapping.

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            auto solution = findExact(problem, hardware);
            if(solution && fitness)
                *fitness = 0.0;
            return solution;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            SolutionVector<MySolution> rv;
            if(numSolutions <= 0)
            {
                m_lastFindTopRetAll = true;
                return rv;
            }

            if(auto solution = findExact(problem, hardware))
                rv.push_back(solution);

            // At most one kernel is mapped per shape; later rows supply the rest.
            m_lastFindTopRetAll = (static_cast<int>(rv.size()) < numSolutions);
            return rv;
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return m_lastFindTopRetAll;
        }

        // Enumeration reports every applicable registered kernel, mapped or not.

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;
            if(!m_hasAny.load(std::memory_order_acquire))
                return rv;

            auto snap = current();
            for(auto const& entry : snap->solutions)
            {
                if(canServe(*entry.second, problem, hardware))
                    rv.insert(entry.second);
            }
            return rv;
        }

        // Grouped GEMM is not supported.
        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const&,
                                        Hardware const&,
                                        SolutionLibrarySearchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return SolutionSet<MySolution>();
        }

        // Dispatch by index works for every registered kernel, mapped or not.

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            if(!isUserKernelIndex(index))
                return std::shared_ptr<MySolution>();

            return lookupIndex(index);
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(const int index) const override
        {
            if(!isUserKernelIndex(index))
                return std::shared_ptr<MySolution>();
            return lookupIndex(index);
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(Hardware const&,
                                                               const int index) const override
        {
            if(!isUserKernelIndex(index))
                return std::shared_ptr<MySolution>();
            return lookupIndex(index);
        }

        /**
         * Registers solutions and returns their indices in order. Each solution
         * gets a new index in the user range, written back to
         * `solution->index`. Indices are valid only in this process.
         */
        std::vector<int> addSolutions(std::vector<std::shared_ptr<MySolution>> const& incoming)
        {
            std::vector<int> assigned;
            assigned.reserve(incoming.size());

            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);
            auto                                next = std::make_shared<Snapshot>(*m_snapshot);

            for(auto const& solution : incoming)
            {
                if(!solution)
                    continue;
                const int index        = m_nextIndex++;
                solution->index        = index;
                next->solutions[index] = solution;
                assigned.push_back(index);
            }

            publishLocked(std::move(next));
            return assigned;
        }

        /**
         * Maps a registered kernel to one exact shape, making it selectable.
         * Returns false if the index is not registered.
         */
        bool setExactMatch(UserExactKey const& key, int index)
        {
            std::unique_lock<std::shared_mutex> lock(m_snapshotMutex);

            auto it = m_snapshot->solutions.find(index);
            if(it == m_snapshot->solutions.end())
                return false;

            auto next             = std::make_shared<Snapshot>(*m_snapshot);
            next->exactTable[key] = it->second;

            publishLocked(std::move(next));
            return true;
        }

        size_t registeredCount() const
        {
            return current()->solutions.size();
        }

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

        /**
         * Caller must hold m_snapshotMutex exclusively. The generation must
         * advance after the snapshot is stored, or a cache could stamp an old
         * result as current.
         */
        void publishLocked(std::shared_ptr<const Snapshot> next)
        {
            const bool hasAny   = !next->solutions.empty();
            const bool hasExact = !next->exactTable.empty();

            m_snapshot = std::move(next);
            m_hasAny.store(hasAny, std::memory_order_release);
            m_hasExact.store(hasExact, std::memory_order_release);

            m_generation->value.fetch_add(1, std::memory_order_release);
        }

        std::shared_ptr<MySolution> lookupIndex(int index) const
        {
            if(!m_hasAny.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->solutions.find(index);
            return it == snap->solutions.end() ? std::shared_ptr<MySolution>() : it->second;
        }

        std::shared_ptr<MySolution> findExact(MyProblem const& problem,
                                              Hardware const&  hardware) const
        {
            if(!m_hasExact.load(std::memory_order_acquire))
                return std::shared_ptr<MySolution>();

            auto snap = current();
            auto it   = snap->exactTable.find(userExactKeyFor(problem));
            if(it == snap->exactTable.end())
                return std::shared_ptr<MySolution>();

            if(!canServe(*it->second, problem, hardware))
                return std::shared_ptr<MySolution>();

            return it->second;
        }

        /**
         * Whether a registered kernel can serve this problem. One instance
         * serves every problem type, so the problem type is checked here before
         * the solution's own predicates.
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

            if(solution.problemPredicate && !(*solution.problemPredicate)(problem))
                return false;

            Task task(hardware, problem, solution);
            if(solution.taskPredicate && !(*solution.taskPredicate)(task))
                return false;

            if(solution.hardwarePredicate && !(*solution.hardwarePredicate)(hardware))
                return false;

            return true;
        }

        std::shared_ptr<LibraryGeneration> m_generation;

        // Guards the pointer only; readers copy it and release the lock.
        // std::atomic<std::shared_ptr> is not usable with every libstdc++ we
        // build with.
        mutable std::shared_mutex       m_snapshotMutex;
        std::shared_ptr<const Snapshot> m_snapshot;

        // Checked before taking the lock, so lookups cost no lock traffic while
        // nothing is registered.
        std::atomic<bool> m_hasAny{false};
        std::atomic<bool> m_hasExact{false};

        int m_nextIndex = UserKernelIndexBase;

        mutable std::atomic<bool> m_lastFindTopRetAll{false};
    };

    /**
     * The process-wide user kernel library. Lazily loaded shards are
     * deserialized after registrations happen, so every ladder must share one
     * instance.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    inline std::shared_ptr<UserKernelLibrary<MyProblem, MySolution>> globalUserKernelLibrary()
    {
        static std::shared_ptr<UserKernelLibrary<MyProblem, MySolution>> instance
            = std::make_shared<UserKernelLibrary<MyProblem, MySolution>>(globalLibraryGeneration());
        return instance;
    }
} // namespace TensileLite
