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
#include <iterator>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
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
        using Caches = CacheMap<SolutionVector<MySolution>, AMDGPU, MyProblem>;
        using CachesAllSolsFlag
            = CacheMap<bool, AMDGPU, MyProblem>;
        using CachesGroupedGemm
            = CacheMap<SolutionVector<MySolution>, AMDGPU, std::vector<MyProblem>>;

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
            if constexpr(CacheKey<MyProblem>::implemented)
            {
                auto const* amdgpu = dynamic_cast<AMDGPU const*>(&hardware);
                if(amdgpu == nullptr)
                    return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
                return findTopSolutionsByKey(problem, *amdgpu, numSolutions);
            }
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

        /// Upper bound on the problems whose findTopSolutions results are kept;
        /// at the bound, a new problem replaces the least recently used one.
        static constexpr size_t TopSolutionsCapacity = size_t(1) << 14;

    private:
        struct TopSolutions
        {
            SolutionVector<MySolution> solutions;
            bool                       returnedAll = false;
        };

        struct TopEntry
        {
            std::string  key;
            size_t       hash; // std::hash<std::string_view> of key
            TopSolutions value;
        };
        using TopList = std::list<TopEntry>;

        // A key of m_top: a view of a cache key with its std::hash<std::string_view>
        // value, so that a lookup hashes the key once however many map operations
        // it makes. Equal views have equal hashes, so comparing the hashes first
        // changes no equality.
        struct TopMapKey
        {
            std::string_view text;
            size_t           hash;
        };
        struct TopMapKeyHash
        {
            size_t operator()(TopMapKey const& k) const noexcept
            {
                return k.hash;
            }
        };
        struct TopMapKeyEqual
        {
            bool operator()(TopMapKey const& a, TopMapKey const& b) const noexcept
            {
                return a.hash == b.hash && a.text == b.text;
            }
        };
        using TopMap = std::
            unordered_map<TopMapKey, typename TopList::iterator, TopMapKeyHash, TopMapKeyEqual>;

        SolutionVector<MySolution> findTopSolutionsByKey(MyProblem const& problem,
                                                         AMDGPU const&    amdgpu,
                                                         int              numSolutions) const
        {
            thread_local std::string key;
            key.clear();
            AppendCacheKeys(key, amdgpu, problem);
            // The key's only hash: the lookup, the store and the insert reuse it.
            size_t const keyHash = std::hash<std::string_view>{}(key);

            SolutionVector<MySolution> solutions;
            bool                       returnedAll = false;
            {
                std::lock_guard<std::mutex> lock(m_topMutex);
                auto                        it = m_top.find(TopMapKey{key, keyHash});
                if(it != m_top.end())
                {
                    m_topOrder.splice(m_topOrder.begin(), m_topOrder, it->second);
                    solutions   = it->second->value.solutions;
                    returnedAll = it->second->value.returnedAll;
                }
            }
            lastFindTopRetAll = returnedAll;
            if(solutions.size() >= numSolutions || returnedAll)
                return solutions;

            // The sub-library can reach another CachingLibrary on this thread,
            // which reuses `key`.
            std::string ownKey = key;
            solutions          = m_subLibrary->findTopSolutions(problem, amdgpu, numSolutions);
            if(solutions.size() != 0)
            {
                TopSolutions entry{solutions, m_subLibrary->lastFindTopAlreadyRetAll()};
                std::lock_guard<std::mutex> lock(m_topMutex);
                storeTopSolutions(std::move(ownKey), keyHash, std::move(entry));
            }

            // can't reach the requested number, means findTop already done its best
            lastFindTopRetAll = (solutions.size() < numSolutions);
            return solutions;
        }

        // Called with m_topMutex held; keyHash is std::hash<std::string_view> of key.
        void storeTopSolutions(std::string&& key, size_t keyHash, TopSolutions&& entry) const
        {
            auto it = m_top.find(TopMapKey{key, keyHash});
            if(it != m_top.end())
            {
                it->second->value = std::move(entry);
                m_topOrder.splice(m_topOrder.begin(), m_topOrder, it->second);
                return;
            }
            // Reserving the full capacity up front means the map never rehashes.
            if(m_top.empty())
                m_top.reserve(TopSolutionsCapacity);
            if(m_top.size() < TopSolutionsCapacity)
            {
                m_topOrder.push_front(TopEntry{std::move(key), keyHash, std::move(entry)});
                m_top.emplace(TopMapKey{m_topOrder.front().key, keyHash}, m_topOrder.begin());
                return;
            }
            // The map keys view the list's strings: drop the evicted key from the
            // map before its string is overwritten, then reuse both nodes.
            auto oldest   = std::prev(m_topOrder.end());
            auto node     = m_top.extract(TopMapKey{oldest->key, oldest->hash});
            oldest->key   = std::move(key);
            oldest->hash  = keyHash;
            oldest->value = std::move(entry);
            m_topOrder.splice(m_topOrder.begin(), m_topOrder, oldest);
            node.key() = TopMapKey{oldest->key, keyHash};
            m_top.insert(std::move(node));
        }

        std::shared_ptr<Library>  m_subLibrary;
        mutable Cache             m_cache;
        mutable Caches            m_caches;
        mutable CachesAllSolsFlag m_cachesAllSolutions;
        mutable CachesGroupedGemm m_cachesGroupedGemm;
        mutable std::atomic<bool> lastFindTopRetAll = false;

        mutable std::mutex m_topMutex;
        // Most recently used first.
        mutable TopList m_topOrder;
        mutable TopMap  m_top;
    };

    /// `library` without the CachingLibrary that wraps it, if any.
    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        withoutCache(std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> const& library)
    {
        auto cache = std::dynamic_pointer_cast<CachingLibrary<MyProblem, MySolution>>(library);
        return cache ? cache->library() : library;
    }

} // namespace TensileLite
