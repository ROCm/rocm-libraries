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
#include <shared_mutex>
#include <unordered_map>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    template <typename Value, typename Key, typename... Keys>
    struct MultiLevelMap
    {
        using type = typename MultiLevelMap<std::unordered_map<Key, Value>, Keys...>::type;
    };

    template <typename Value, typename Key>
    struct MultiLevelMap<Value, Key>
    {
        using type = std::unordered_map<Key, Value>;
    };

    /**
     * Thread-safe multi-valued cache.
     *
     * Note that due to a quirk with templates, the order of the keys in find() and add() is *opposite* of that in the type.
     *
     * e.g.
     *
     *     CacheMap<int, float, std::string> myCache
     *     myCache.find("foo", 1.4); // great
     *     myCache.find(1.4, "foo"); // error!
     */
    template <typename Value, typename... Keys>
    class CacheMap
    {
        using Map = typename MultiLevelMap<Value, Keys...>::type;

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
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto rv = find_impl(m_map, keys...);

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
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            add_impl(m_map, value, ks...);
        }

        // Keeps the stored value when preferIncoming is false. The check and
        // the write share this lock.
        template <typename PreferIncoming, typename... Ks>
        void addIfPreferred(Value const& incoming, PreferIncoming preferIncoming, Ks const&... ks)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            addIfPreferred_impl(m_map, incoming, preferIncoming, ks...);
        }

    private:
        template <typename SubMap, typename K>
        Value find_impl(SubMap const& map, K const& key)
        {
            auto iter = map.find(key);

            if(iter == map.end())
                return m_nullValue;

            return iter->second;
        }

        template <typename SubMap, typename K, typename... Ks>
        Value find_impl(SubMap const& map, K const& key, Ks const&... ks)
        {
            auto iter = map.find(key);

            if(iter == map.end())
                return m_nullValue;

            return find_impl(iter->second, ks...);
        }

        template <typename SubMap, typename K>
        void add_impl(SubMap& map, Value const& value, K const& key)
        {
            map.insert_or_assign(key, value);
        }

        template <typename SubMap, typename K, typename... Ks>
        void add_impl(SubMap& map, Value const& value, K const& key, Ks const&... ks)
        {
            add_impl(map[key], value, ks...);
        }

        template <typename SubMap, typename PreferIncoming, typename K>
        void addIfPreferred_impl(SubMap&          map,
                                 Value const&     incoming,
                                 PreferIncoming   preferIncoming,
                                 K const&         key)
        {
            auto iter = map.find(key);
            if(iter == map.end() || preferIncoming(iter->second, incoming))
                map.insert_or_assign(key, incoming);
        }

        template <typename SubMap, typename PreferIncoming, typename K, typename... Ks>
        void addIfPreferred_impl(SubMap&          map,
                                 Value const&     incoming,
                                 PreferIncoming   preferIncoming,
                                 K const&         key,
                                 Ks const&...     ks)
        {
            addIfPreferred_impl(map[key], incoming, preferIncoming, ks...);
        }

        Map                     m_map;
        std::shared_timed_mutex m_mutex;
        Value                   m_nullValue;

        bool                 m_lookupEfficiency;
        std::atomic<int64_t> m_lookups;
        std::atomic<int64_t> m_hits;
    };

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class CachingLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
    public:
        using Library = SolutionLibrary<MyProblem, MySolution>;
        using Cache  = CacheMap<std::tuple<std::shared_ptr<MySolution>, double>, AMDGPU, MyProblem>;

        // The solutions found for a problem and whether they are all of the
        // solutions the sub-library has. They are published and read as one
        // CacheMap value, so no thread can see one request's solutions paired
        // with another request's already-returned-all bit.
        struct TopSolutionsEntry
        {
            SolutionVector<MySolution> solutions;
            bool                       alreadyRetAll = false;

            bool operator==(TopSolutionsEntry const& other) const
            {
                return alreadyRetAll == other.alreadyRetAll && solutions == other.solutions;
            }
            bool operator!=(TopSolutionsEntry const& other) const
            {
                return !(*this == other);
            }
        };

        using Caches = CacheMap<TopSolutionsEntry, AMDGPU, MyProblem>;
        using CachesGroupedGemm
            = CacheMap<SolutionVector<MySolution>, AMDGPU, std::vector<MyProblem>>;

        CachingLibrary(std::shared_ptr<Library> subLibrary)
            : m_subLibrary(subLibrary)
            , m_cache(std::make_tuple(nullptr, std::numeric_limits<double>::max()))
            , m_caches(TopSolutionsEntry{})
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

        virtual TopSolutionsResult<MySolution> findTopSolutions(MyProblem const& problem,
                                                                Hardware const&  hardware,
                                                                int numSolutions) const override
        {
            try
            {
                auto const&                amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                SolutionVector<MySolution> solutions;

                // One lookup returns the solutions and their already-returned-all bit together.
                auto cached = m_caches.find(problem, amdgpu);

                // flag for this call in case of early return
                if(cached.solutions.size() >= numSolutions || cached.alreadyRetAll)
                    return {std::move(cached.solutions), cached.alreadyRetAll};

                auto sub = m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
                solutions = std::move(sub.solutions);
                if(solutions.size() != 0)
                {
                    // A shorter list must not replace a longer one for this problem.
                    // For equal-length lists, keep the incoming one only if it upgrades
                    // alreadyRetAll from false to true to avoid repeated top-up rescans.
                    m_caches.addIfPreferred(
                        TopSolutionsEntry{solutions, sub.alreadyRetAll},
                        [](TopSolutionsEntry const& existing, TopSolutionsEntry const& incoming) {
                            return incoming.solutions.size() > existing.solutions.size()
                                   || (incoming.solutions.size() == existing.solutions.size()
                                       && incoming.alreadyRetAll && !existing.alreadyRetAll);
                        },
                        problem,
                        amdgpu);
                }

                // can't reach the requested number, means findTop already done its best
                const bool alreadyRetAll = (solutions.size() < numSolutions);
                return {std::move(solutions), alreadyRetAll};
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
            }
            // TODO- redundant ??
            return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
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
        mutable CachesGroupedGemm m_cachesGroupedGemm;
    };

} // namespace TensileLite
