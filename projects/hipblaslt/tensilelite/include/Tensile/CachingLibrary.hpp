/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <atomic>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ReadMostlyMap.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
     * Thread-safe multi-valued cache. find() takes no lock and writes no shared
     * memory (see ReadMostlyMap); add() serializes writers on a mutex.
     *
     * Keys are passed to find() and add() in the order of the type.
     */
    template <typename Value, typename... Keys>
    class CacheMap
    {
        using KeyTuple = std::tuple<Keys...>;

        // Hashes each key as its stored type, so lookups by reference match.
        struct KeyHash
        {
            template <typename Tuple>
            size_t operator()(Tuple const& keys) const
            {
                return std::apply(
                    [](auto const&... k) {
                        size_t rv = 0;
                        ((rv = rv * 31 + std::hash<Keys>{}(k)), ...);
                        return rv;
                    },
                    keys);
            }
        };

    public:
        CacheMap(Value const& nullValue)
            : m_nullValue(nullValue)
            , m_lookupEfficiency(Debug::Instance().printLookupEfficiency())
            , m_lookups(0)
            , m_hits(0)

        {
        }

        ~CacheMap()
        {
            if(m_lookupEfficiency)
                std::cout << "CacheMap: " << m_hits << "/" << m_lookups << " cache hits"
                          << std::endl;
        }

        template <typename... Ks>
        Value find(Ks const&... keys)
        {
            Value const* found = m_map.find(std::forward_as_tuple(keys...));
            Value        rv    = found ? *found : m_nullValue;

            if(m_lookupEfficiency)
            {
                m_lookups++;
                if(rv != m_nullValue)
                    m_hits++;
            }

            return rv;
        }

        template <typename... Ks>
        void add(Value const& value, Ks const&... ks)
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            // Replaced entries stay allocated, so skip re-adding an unchanged value.
            Value const* found = m_map.find(std::forward_as_tuple(ks...));
            if(found && *found == value)
                return;

            m_map.insertOrAssign(KeyTuple(ks...), value);
        }

    private:
        ReadMostlyMap<KeyTuple, Value, KeyHash, std::equal_to<>> m_map;
        std::mutex                                               m_mutex;
        Value                                                    m_nullValue;

        bool                 m_lookupEfficiency;
        std::atomic<int64_t> m_lookups;
        std::atomic<int64_t> m_hits;
    };

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class CachingLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
    public:
        using Library = SolutionLibrary<MyProblem, MySolution>;
        using Cache  = CacheMap<std::tuple<std::shared_ptr<MySolution>, double>, MyProblem, AMDGPU>;
        using Caches = CacheMap<SolutionVector<MySolution>, MyProblem, AMDGPU>;
        using CachesAllSolsFlag
            = CacheMap<bool, MyProblem, AMDGPU>;
        using CachesGroupedGemm
            = CacheMap<SolutionVector<MySolution>, std::vector<MyProblem>, AMDGPU>;

        CachingLibrary(std::shared_ptr<Library> subLibrary)
            : m_subLibrary(subLibrary)
            , m_cache(std::make_tuple(nullptr, std::numeric_limits<double>::max()))
            , m_caches(SolutionVector<MySolution>{})
            , m_cachesAllSolutions(false)
            , m_cachesGroupedGemm(SolutionVector<MySolution>{})
        {
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            return m_subLibrary->getSolutionByIndex(problem, hardware, index);
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            try
            {
                double cachedFitness = std::numeric_limits<double>::max();
                fitness              = (fitness) ? fitness : &cachedFitness;

                auto const&                 amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                std::shared_ptr<MySolution> solution;
                std::tie(solution, *fitness) = m_cache.find(problem, amdgpu);

                if(solution)
                    return solution;

                solution = m_subLibrary->findBestSolution(problem, hardware, fitness);
                if(solution)
                    m_cache.add(std::make_tuple(solution, *fitness), problem, amdgpu);

                return solution;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findBestSolution(problem, hardware, fitness);
            }
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            return m_subLibrary->findAllSolutions(problem, hardware, searchType);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return m_subLibrary->findAllSolutionsGroupedGemm(problems, hardware, searchType);
        }

        std::shared_ptr<MySolution> findSolutionInCache(MyProblem const& problem,
                                                        Hardware const&  hardware) const
        {
            auto const& amdgpu = dynamic_cast<AMDGPU const&>(hardware);

            return std::get<std::shared_ptr<MySolution>>(m_cache.find(problem, amdgpu));
        }

        virtual std::string type() const override
        {
            return "Caching Library";
        }
        virtual std::string description() const override
        {
            return "Caching Library";
        }

        std::shared_ptr<Library> library() const
        {
            return m_subLibrary;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            try
            {
                auto const&                amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                SolutionVector<MySolution> solutions;
                bool                       cacheAlreadyContainAll;
                solutions = m_caches.find(problem, amdgpu);
                cacheAlreadyContainAll = m_cachesAllSolutions.find(problem, amdgpu);
                // set flag in case of early return
                lastFindTopRetAll = cacheAlreadyContainAll;

                if(solutions.size() >= numSolutions || cacheAlreadyContainAll)
                    return solutions;

                solutions = m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
                if(solutions.size() != 0)
                {
                    bool alreadyRetAll = m_subLibrary->lastFindTopAlreadyRetAll();
                    m_caches.add(solutions, problem, amdgpu);
                    m_cachesAllSolutions.add(alreadyRetAll, problem, amdgpu);
                    // debug
                    // std::cout << "m_cachesAllSolutions.add() with solution.size() = " << solutions.size()
                    //           << " and alreadyRetAll: " << (alreadyRetAll? "True" : "False") << std::endl;
                }

                // can't reach the requested number, means findTop already done its best
                lastFindTopRetAll = (solutions.size() < numSolutions);
                return solutions;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
            }
            // TODO- redundant ??
            return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            try
            {
                auto const&                amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                SolutionVector<MySolution> solutions;
                solutions = m_cachesGroupedGemm.find(problems, amdgpu);

                if(solutions.size() != 0)
                    return solutions;

                solutions
                    = m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
                if(solutions.size() != 0)
                    m_cachesGroupedGemm.add(solutions, problems, amdgpu);

                return solutions;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
            }
            return m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
        }

    private:
        std::shared_ptr<Library>  m_subLibrary;
        mutable Cache             m_cache;
        mutable Caches            m_caches;
        mutable CachesAllSolsFlag m_cachesAllSolutions;
        mutable CachesGroupedGemm m_cachesGroupedGemm;
        mutable std::atomic<bool> lastFindTopRetAll = false;
    };

} // namespace TensileLite
