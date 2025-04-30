/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/UserDrivenTuningParser.hpp>

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

namespace Tensile
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

        template <typename... Ks>
        void add_or_replace(Value const& value, Ks const&... ks)
        {
            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            add_or_replace_impl(m_map, value, ks...);
        }

    private:
        template <typename SubMap, typename K>
        Value find_impl(SubMap& map, K const& key)
        {
            Value* val = find_impl_ptr(map, key);
            return val ? *val : m_nullValue;
        }

        template <typename SubMap, typename K, typename... Ks>
        Value find_impl(SubMap& map, K const& key, Ks const&... ks)
        {
            Value* val = find_impl_ptr(map, key, ks...);
            return val ? *val : m_nullValue;
        }

        template <typename SubMap, typename K>
        Value* find_impl_ptr(SubMap& map, K const& key)
        {
            auto iter = map.find(key);

            if(iter == map.end())
                return nullptr;

            return &(iter->second);
        }

        template <typename SubMap, typename K, typename... Ks>
        Value* find_impl_ptr(SubMap& map, K const& key, Ks const&... ks)
        {
            auto iter = map.find(key);

            if(iter == map.end())
                return nullptr;

            return find_impl_ptr(iter->second, ks...);
        }

        template <typename SubMap, typename K>
        void add_impl(SubMap& map, Value const& value, K const& key)
        {
            map.emplace(key, value);
        }

        template <typename SubMap, typename K, typename... Ks>
        void add_impl(SubMap& map, Value const& value, K const& key, Ks const&... ks)
        {
            add_impl(map[key], value, ks...);
        }

        template <typename SubMap, typename K>
        void add_or_replace_impl(SubMap& map, Value const& value, K const& key)
        {
            Value* current_value = find_impl_ptr(map, key);
            if(current_value)
            {
                *current_value = value;
            }
            else
            {
                add_impl(map, value, key);
            }
        }

        template <typename SubMap, typename K, typename... Ks>
        void add_or_replace_impl(SubMap& map, Value const& value, K const& key, Ks const&... ks)
        {
            Value* current_value = find_impl_ptr(map, key, ks...);
            if(current_value)
            {
                *current_value = value;
            }
            else
            {
                add_impl(map, value, key, ks...);
            }
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
        using Cache = CacheMap<std::tuple<std::shared_ptr<MySolution>, double>, AMDGPU, MyProblem>;
        using Override = CacheMap<std::tuple<std::shared_ptr<MySolution>, double>,
                                  AMDGPU,
                                  ProblemOverride<ContractionProblem>>;

        CachingLibrary(std::shared_ptr<Library> subLibrary)
            : m_subLibrary(subLibrary)
            , m_cache(std::make_tuple(nullptr, std::numeric_limits<double>::max()))
            , m_override(std::make_tuple(nullptr, std::numeric_limits<double>::max()))
        {
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

                bool override_debug = Debug::Instance().printOverrideLogs();

                // Check override
                ProblemOverride<MyProblem> po(problem);
                std::tie(solution, *fitness) = m_override.find(po, amdgpu);

                if(override_debug && solution)
                {
                    std::cout << "Override found for problem:\n" << problem << "\n";

                    if(solution->canSolve(problem, hardware))
                        std::cout << "Using solution:\n" << solution->name() << "\n";
                    else
                        std::cout
                            << "WARNING: solution: " << solution->name()
                            << "\nis not valid for this problem. Possible library mismatch.\n";
                }

                if(solution && solution->canSolve(problem, hardware))
                    return solution;

                // Check cache
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

        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) const override
        {
            return m_subLibrary->findAllSolutions(problem, hardware);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsMatchingType(MyProblem const& problem,
                                         Hardware const&  hardware) const override
        {
            return m_subLibrary->findAllSolutionsMatchingType(problem, hardware);
        }

        std::shared_ptr<MySolution> findSolutionInCache(MyProblem const& problem,
                                                        Hardware const&  hardware) const
        {
            auto const& amdgpu = dynamic_cast<AMDGPU const&>(hardware);

            return std::get<std::shared_ptr<MySolution>>(m_cache.find(problem, amdgpu));
        }

        bool addToOverride(ProblemOverride<MyProblem> const& po,
                           Hardware const&                   hardware,
                           std::shared_ptr<MySolution>       solution,
                           double*                           fitness = nullptr)
        {
            try
            {
                auto const& amdgpu        = dynamic_cast<AMDGPU const&>(hardware);
                double      cachedFitness = std::numeric_limits<double>::max();
                fitness                   = (fitness) ? fitness : &cachedFitness;

                if(solution)
                {
                    m_override.add_or_replace(std::make_tuple(solution, *fitness), po, amdgpu);
                    return true;
                }
                else
                {
                    return false;
                }
            }
            catch(std::bad_cast const& exc)
            {
                return false;
            }
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

    private:
        std::shared_ptr<Library> m_subLibrary;
        mutable Cache            m_cache;
        mutable Override         m_override;
    };

#if 0
    struct ContractionCachingLibrary: public CachingLibrary<ContractionProblem>
    {
        using Super = CachingLibrary<ContractionProblem>;
        using Library = typename Super::Library;
        using Key = typename Super::Key;

        ContractionCachingLibrary(std::shared_ptr<Library> subLibrary)
            : CachingLibrary<ContractionProblem>(subLibrary)
        {}

    };
#endif

} // namespace Tensile
