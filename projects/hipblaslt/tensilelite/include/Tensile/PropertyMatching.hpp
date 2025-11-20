/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <array>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/Properties.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup PropertyMatching Property Matching
     *
     * @brief Distance-based matching of Property values to a table.
     *
     * Generic algorithm for comparing an object to a table of predefined
     * values based on Property objects and a Distance function. Used for
     * MatchingLibrary.
     */

    /**
     * \ingroup PropertyMatching
     */
    namespace Matching
    {
        template <typename Key, typename Value>
        struct MatchingTableEntry
        {
            Key    key;
            Value  value;
            double speed;
        };

        template <typename Value>
        struct KBEntry
        {
            int   k;
            int   b;
            Value value;
        };

        template <typename Object, typename Value, typename ReturnValue>
        struct MatchingTable
        {
            using Properties = std::vector<std::shared_ptr<Property<Object>>>;
            using Transform  = std::function<ReturnValue(Value)>;

            MatchingTable() = default;
            MatchingTable(Properties const& properties)
                : properties(properties)
            {
            }

            virtual ~MatchingTable() = default;

            virtual std::tuple<ReturnValue, double> findBestMatch(Object const& object,
                                                                  Transform     transform) const
                = 0;

            virtual std::vector<ReturnValue>
                findTopMatch(Object const& object, Transform transform, int numSolutions) const = 0;

            virtual std::vector<Value> matchesInOrder(Object const& object) const = 0;

            virtual std::vector<Value> GetAll() const = 0;

            virtual std::string description() const = 0;

            virtual std::string distanceType() const = 0;

            Properties properties;
        };

        template <typename T, std::size_t N>
        struct PointND
        {
            std::array<T, N> coord;
        };

        template <typename T, std::size_t N>
        struct KDTree
        {
            struct Node
            {
                explicit Node(const PointND<T, N>& p, std::size_t ax)
                    : pt(p)
                    , left(nullptr)
                    , right(nullptr)
                    , axis(ax)
                {
                }
                PointND<T, N>         pt;
                std::unique_ptr<Node> left;
                std::unique_ptr<Node> right;
                std::size_t           axis{};
            };

            struct SearchResult
            {
                const Node* node{};
                float       distance{};
                bool        operator<(const SearchResult& rhs) const
                {
                    if(distance != rhs.distance)
                    {
                        return distance < rhs.distance;
                    }

                    for(size_t i = 0; i < N; ++i)
                    {
                        if(node->pt.coord[i] != rhs.node->pt.coord[i])
                        {
                            return node->pt.coord[i] < rhs.node->pt.coord[i];
                        }
                    }

                    return false;
                }

                bool operator>(const SearchResult& rhs) const
                {
                    return rhs < *this;
                }
            };

            //Hacked pq to make it iterable
            struct IterableSearchResult : std::priority_queue<SearchResult,
                                                              std::deque<SearchResult>,
                                                              std::less<SearchResult>>
            {
                auto begin() const
                {
                    return this->c.begin();
                }

                auto end() const
                {
                    return this->c.end();
                }
            };

            using SortedSearchResults = IterableSearchResult;
            using InsertCriteria = std::function<bool(const PointND<T, N>&, const PointND<T, N>&)>;

            template <typename Iter>
            void build(std::unique_ptr<Node>& root, Iter first, Iter last, std::size_t depth)
            {
                const auto dim         = depth % N;
                const auto numElements = std::distance(first, last);

                if(!numElements)
                {
                    return;
                }

                if(!depth)
                {
                    numPoints = numElements;
                }

                std::sort(first, last, [dim](const auto& lhs, const auto& rhs) {
                    return lhs.coord[dim] < rhs.coord[dim];
                });

                maximums[dim]      = std::max(maximums[dim], std::next(last, -1)->coord[dim]);
                const auto midIdx  = numElements / 2;
                const auto midIter = std::next(first, midIdx);
                root               = std::make_unique<Node>(*midIter, dim);
                build(root->left, first, std::next(first, midIdx), depth + 1);
                build(root->right, std::next(first, midIdx + 1), last, depth + 1);
            }

            std::vector<SearchResult>
                query(const PointND<T, N>& pt, std::size_t n, InsertCriteria criteria)
            {
                SortedSearchResults res;
                queryImpl(pt, root.get(), n, criteria, res);
                std::vector<SearchResult> ret;

                while(res.size())
                {
                    ret.push_back(res.top());
                    res.pop();
                }
                return {rbegin(ret), rend(ret)};
            }

            std::size_t size() const
            {
                return numPoints;
            }

            std::unique_ptr<Node> root;
            std::array<T, N>      maximums{};

        private:
            void queryImpl(const PointND<T, N>& pt,
                           Node*                root,
                           std::size_t          n,
                           InsertCriteria       criteria,
                           SortedSearchResults& result)
            {
                if(!root)
                {
                    return;
                }

                if(criteria(pt, root->pt))
                {
                    result.push(SearchResult{root, distance(pt, root->pt)});
                    popResultUntil(n, result);
                }

                const auto dim = root->axis;

                if(pt.coord[dim] < root->pt.coord[dim])
                {
                    queryImpl(pt, root->left.get(), n, criteria, result);

                    if(result.size() < n || shouldCheckOtherHyperplane(result, pt, root))
                    {
                        queryImpl(pt, root->right.get(), n, criteria, result);
                    }
                }
                else
                {
                    queryImpl(pt, root->right.get(), n, criteria, result);
                    if(result.size() < n || shouldCheckOtherHyperplane(result, pt, root))
                    {
                        queryImpl(pt, root->left.get(), n, criteria, result);
                    }
                }
            }

            bool shouldCheckOtherHyperplane(SortedSearchResults& result,
                                            const PointND<T, N>& pt,
                                            const Node*          hyperplane) const
            {
                const auto dim = hyperplane->axis;
                const auto d   = std::abs(pt.coord[dim] - hyperplane->pt.coord[dim]);
                return (d * d) < result.top().distance;
            }

            float distance(const PointND<T, N>& p, const PointND<T, N>& q)
            {
                float d{};
                for(size_t i = 0; i < N; ++i)
                {
                    d += static_cast<float>(p.coord[i] - q.coord[i]) * (p.coord[i] - q.coord[i]);
                }
                return d;
            }

            void popResultUntil(std::size_t n, SortedSearchResults& result)
            {
                while(result.size() > n)
                {
                    result.pop();
                }
            }

            std::size_t numPoints{};
        };

        /**
         * Shared code between the generic DistanceMatchingTable and the specialization
         * for the special Equality distance
         */
        template <typename Key,
                  typename Object,
                  typename Value,
                  typename ReturnValue,
                  typename Distance>
        struct DistanceMatchingCommon : public MatchingTable<Object, Value, ReturnValue>
        {
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using KBEntry    = KBEntry<Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;

            DistanceMatchingCommon(ReturnValue nullValue = ReturnValue())
                : nullValue(nullValue)
            {
            }

            DistanceMatchingCommon(Properties const& properties,
                                   ReturnValue       nullValue = ReturnValue())
                : Base(properties)
                , nullValue(nullValue)
            {
            }

            DistanceMatchingCommon(Distance const&   distance,
                                   Properties const& properties,
                                   ReturnValue       nullValue = ReturnValue())
                : Base(properties)
                , nullValue(nullValue)
                , distance(distance)
            {
            }

            virtual std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                                     Transform  transform) const
                = 0;

            virtual std::tuple<ReturnValue, double>
                findBestMatch(Object const& object, Transform transform) const override
            {
                return findBestKeyMatch(
                    ProblemKey::keyForProblem<Key, Object>(object, this->properties), transform);
            }

            virtual std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const = 0;

            virtual std::vector<ReturnValue> findTopMatch(Object const& object,
                                                          Transform     transform,
                                                          int           numSolutions) const override
            {
                return findTopKeyMatch(
                    ProblemKey::keyForProblem<Key, Object>(object, this->properties),
                    transform,
                    numSolutions);
            }

            virtual std::vector<Value> matchesInOrder(Object const& object) const override
            {
                return keyMatchesInOrder(
                    ProblemKey::keyForProblem<Key, Object>(object, this->properties));
            }

            std::vector<Value> keyMatchesInOrder(Key const& key) const
            {
                std::vector<std::pair<double, size_t>> indices(this->table.size());

                for(size_t i = 0; i < this->table.size(); i++)
                    indices[i] = std::make_pair(distance(key, this->table[i].key), i);

                std::sort(indices.begin(), indices.end());

                std::vector<Value> result;
                result.reserve(this->table.size());

                for(auto const& entry : indices)
                    result.push_back(this->table[entry.second].value);

                return result;
            }

            std::vector<Value> GetAll() const override
            {
                std::vector<Value> result;
                result.reserve(this->table.size());

                for(auto& data : this->table)
                    result.push_back(data.value);

                return result;
            }

            virtual std::string description() const override
            {
                std::string rv = concatenate(
                    "Table: Properties: ", this->properties, ", ", table.size(), " row(s), ");

                rv += concatenate("Distance: ", Distance::Type());

                return rv;
            }

            virtual std::string distanceType() const override
            {
                return Distance::Type();
            }

            std::vector<Entry> table;
            Distance           distance;

            ReturnValue nullValue;

            mutable KDTree<int32_t, 2>                                  kdTree;
            std::map<std::tuple<int32_t, int32_t>, std::vector<KBEntry>> kSolutionMap;
        };

        /**
         * Generic version of DistanceMatchingTable
         */
        template <typename Key,
                  typename Object,
                  typename Value,
                  typename ReturnValue,
                  typename Distance>
        struct DistanceMatchingTable
            : public DistanceMatchingCommon<Key, Object, Value, ReturnValue, Distance>
        {
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;
            using Common     = DistanceMatchingCommon<Key, Object, Value, ReturnValue, Distance>;
            using Common::distance;
            using Common::nullValue;
            using Common::table;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Common(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(properties, nullValue)
            {
            }

            DistanceMatchingTable(Distance const&   distance,
                                  Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(distance, properties, nullValue)
            {
            }

            std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                             Transform  transform) const
            {
                const bool debug = Debug::Instance().printPropertyEvaluation();
                const bool naive = Debug::Instance().naivePropertySearch();

                if(naive)
                {
                    if(debug)
                        return findBestKeyMatch_NaiveSearch<true>(key, transform);
                    else
                        return findBestKeyMatch_NaiveSearch<false>(key, transform);
                }
                else
                {
                    if(debug)
                        return findBestKeyMatch_BinSearch<true>(key, transform);
                    else
                        return findBestKeyMatch_BinSearch<false>(key, transform);
                }
            }

            std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const
            {
                ReturnValue              solution;
                std::vector<ReturnValue> solutions;
                double                   fitness;
            std:
                tie(solution, fitness) = findBestKeyMatch(key, transform);
                if(solution)
                    solutions.push_back(solution);
                return solutions;
            }

            template <bool T_Debug>
            std::tuple<ReturnValue, double> findBestKeyMatch_BinSearch(Key const& key,
                                                                       Transform  transform) const
            {
                if(this->table.empty())
                    return std::make_tuple(this->nullValue, std::numeric_limits<double>::max());

                auto comp = [](Entry const& e, Key const& key) { return e.key < key; };

                auto origIter = std::lower_bound(table.begin(), table.end(), key, comp);

                if(T_Debug)
                {
                    std::cout << "Key: ";
                    streamJoin(std::cout, key, ", ");
                    std::cout << std::endl;

                    std::cout << "Starting point: ";
                    streamJoin(std::cout, origIter->key, ", ");
                    std::cout << std::endl;

                    std::cout << "Rightward search..." << std::endl;
                }

                double bestDistance = std::numeric_limits<double>::max();
                auto   bestMatch    = this->nullValue;
                double bestSpeed    = 0.0;

                ptrdiff_t count = 0;

                for(auto iter = origIter; iter != table.end(); iter++)
                {
                    if(bestMatch && !distance.improvementPossible(key, iter->key, 0, bestDistance))
                    {
                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": Stopping rightward search early." << std::endl;
                        }

                        break;
                    }

                    count++;

                    auto myDistance = distance(key, iter->key);
                    bool thisMatch  = false;

                    if(myDistance < bestDistance
                       || (myDistance == bestDistance && iter->speed > bestSpeed))
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            bestSpeed    = iter->speed;
                            thisMatch    = true;
                        }
                    }

                    if(T_Debug)
                    {
                        if(myDistance <= bestDistance)
                            std::cout << std::endl;

                        std::cout << "speed: " << iter->speed << " | ";
                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;

                        if(myDistance < bestDistance)
                            std::cout << " < ";
                        else if(myDistance > bestDistance)
                            std::cout << " > ";
                        else
                            std::cout << " == ";

                        std::cout << bestDistance;

                        if(myDistance < bestDistance)
                        {
                            if(thisMatch)
                                std::cout << " <-- Best so far";
                            else
                                std::cout << " <-- Best distance, but no matching solution";
                        }

                        std::cout << std::endl;
                    }
                }

                auto iter = table.rbegin();

                if(origIter != table.end())
                    iter = std::make_reverse_iterator(origIter);

                if(T_Debug)
                    std::cout << "Leftward search..." << std::endl;

                for(; iter != table.rend(); iter++)
                {
                    if(bestMatch && !distance.improvementPossible(key, iter->key, 0, bestDistance))
                    {
                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": Stopping leftward search early." << std::endl;
                        }

                        break;
                    }

                    count++;

                    auto myDistance = distance(key, iter->key);
                    bool thisMatch  = false;

                    if(myDistance < bestDistance
                       || (myDistance == bestDistance && iter->speed > bestSpeed))
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            bestSpeed    = iter->speed;
                            thisMatch    = true;
                        }
                    }

                    if(T_Debug)
                    {
                        if(myDistance <= bestDistance)
                            std::cout << std::endl;

                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;

                        if(myDistance < bestDistance)
                            std::cout << " < ";
                        else if(myDistance > bestDistance)
                            std::cout << " > ";
                        else
                            std::cout << " == ";

                        std::cout << bestDistance;

                        if(myDistance < bestDistance)
                        {
                            if(thisMatch)
                                std::cout << " <-- Best so far";
                            else
                                std::cout << " <-- Best distance, but no matching solution";
                        }

                        std::cout << std::endl;
                    }
                }

                if((T_Debug || Debug::Instance().printLookupEfficiency()) && table.size() > 0)
                {
                    double considered = count;
                    considered /= table.size();
                    considered *= 100;
                    std::cout << "Considered " << considered << "% of entries." << std::endl;
                }

                if(T_Debug && bestMatch)
                    std::cout << "Solution index selected: " << bestMatch->index << std::endl;

                return std::make_tuple(bestMatch, bestDistance);
            }

            template <bool T_Debug>
            std::tuple<ReturnValue, double> findBestKeyMatch_NaiveSearch(Key const& key,
                                                                         Transform  transform) const
            {
                double bestDistance = std::numeric_limits<double>::max();

                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return std::make_tuple(this->nullValue, bestDistance);

                ReturnValue bestMatch = transform(iter->value);

                if(bestMatch)
                    bestDistance = distance(key, iter->key);

                if(T_Debug)
                {
                    std::cout << "Key: ";
                    streamJoin(std::cout, key, ", ");
                    std::cout << std::endl;

                    streamJoin(std::cout, iter->key, ", ");

                    std::cout << ": " << bestDistance << " <-- First" << std::endl;
                }

                iter++;

                while(iter != this->table.end())
                {
                    auto myDistance = distance(key, iter->key);
                    bool thisMatch  = false;

                    if(myDistance < bestDistance)
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            thisMatch    = true;
                        }
                    }

                    if(T_Debug)
                    {
                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;
                        if(myDistance < bestDistance)
                        {
                            std::cout << " <-- Best so far";

                            if(thisMatch)
                                std::cout << " (has a matching solution)";
                            else
                                std::cout << " (no match)";
                        }

                        std::cout << std::endl;
                    }

                    iter++;
                }

                if(T_Debug && bestMatch)
                    std::cout << "Solution index selected: " << bestMatch->index << std::endl;

                return std::make_tuple(bestMatch, bestDistance);
            }
        };

        /**
         * Specialization of DistanceMatchingTable for Equality Distance. This special case will
         * only select key in the table if it exactly matches the provided key
         */
        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct DistanceMatchingTable<Key, Object, Value, ReturnValue, Matching::Equality<Key>>
            : public DistanceMatchingCommon<Key,
                                            Object,
                                            Value,
                                            ReturnValue,
                                            Matching::Equality<Key>>
        {
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;
            using Equality   = Matching::Equality<Key>;
            using Common
                = DistanceMatchingCommon<Key, Object, Value, ReturnValue, Matching::Equality<Key>>;
            using Common::distance;
            using Common::nullValue;
            using Common::table;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Common(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(properties, nullValue)
            {
            }

            DistanceMatchingTable(Equality const&   distance,
                                  Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(distance, properties, nullValue)
            {
            }

            std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                             Transform  transform) const
            {
                auto comp = [](Entry const& e, Key const& key) { return e.key < key; };
                auto iter = std::lower_bound(table.begin(), table.end(), key, comp);

                return (iter->key == key)
                           ? std::make_tuple(transform(iter->value), 0.0)
                           : std::make_tuple(this->nullValue, std::numeric_limits<double>::max());
            }

            std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const
            {
                ReturnValue              solution;
                std::vector<ReturnValue> solutions;
                double                   fitness;
            std:
                tie(solution, fitness) = findBestKeyMatch(key, transform);
                if(solution)
                    solutions.push_back(solution);
                return solutions;
            }
        };

        /**
         * Specialization of DistanceMatchingTable for Range Distance. This special case will
         * only select key in the table if it is within the range described by the provided key
         */
        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct DistanceMatchingTable<Key, Object, Value, ReturnValue, Matching::Range<Key>>
            : public DistanceMatchingCommon<Key,
                                            Object,
                                            Value,
                                            ReturnValue,
                                            Matching::Range<Key>>
        {
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;
            using Range      = Matching::Range<Key>;
            using Common
                = DistanceMatchingCommon<Key, Object, Value, ReturnValue, Matching::Range<Key>>;
            using Common::distance;
            using Common::nullValue;
            using Common::table;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Common(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                               ReturnValue       nullValue = ReturnValue())
                : Common(properties, nullValue)
            {
            }

            DistanceMatchingTable(Range const&   distance,
                               Properties const& properties,
                               ReturnValue       nullValue = ReturnValue())
                : Common(distance, properties, nullValue)
            {
            }

            std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                             Transform  transform) const
            {
                for(const auto& row : table)
                {
                    if((row.key[0] == -1 || key[0] >= row.key[0]) &&  // M >= M_min
                       (row.key[1] == -1 || key[1] <= row.key[1]) &&  // M <= M_max
                       (row.key[2] == -1 || key[2] >= row.key[2]) &&  // N >= N_min
                       (row.key[3] == -1 || key[3] <= row.key[3]) &&  // N <= N_max
                       (row.key[4] == -1 || key[4] >= row.key[4]) &&  // batch >= batch_min
                       (row.key[5] == -1 || key[5] <= row.key[5]) &&  // batch <= batch_max
                       (row.key[6] == -1 || key[6] >= row.key[6]) &&  // K >= K_min
                       (row.key[7] == -1 || key[7] <= row.key[7]))    // K <= K_max
                    {
                        return std::make_tuple(transform(row.value), 0.0);
                    }
                }
                return std::make_tuple(this->nullValue, std::numeric_limits<double>::max());
            }

            std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const
            {
                ReturnValue              solution;
                std::vector<ReturnValue> solutions;
                double                   fitness;
            std:
                tie(solution, fitness) = findBestKeyMatch(key, transform);
                if(solution)
                    solutions.push_back(solution);
                return solutions;
            }
        };

        /**
         * Specialization of DistanceMatchingTable for GridBased Distance. This special case will
         * use a more efficient search algorithm that depends on the grid structure
         */
        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct DistanceMatchingTable<Key,
                                     Object,
                                     Value,
                                     ReturnValue,
                                     Matching::GridBasedDistance<Key>>
            : public DistanceMatchingCommon<Key,
                                            Object,
                                            Value,
                                            ReturnValue,
                                            Matching::GridBasedDistance<Key>>
        {
            using Base              = MatchingTable<Object, Value, ReturnValue>;
            using Entry             = MatchingTableEntry<Key, Value>;
            using Transform         = typename Base::Transform;
            using Properties        = typename Base::Properties;
            using GridBasedDistance = Matching::GridBasedDistance<Key>;
            using Common            = DistanceMatchingCommon<Key,
                                                             Object,
                                                             Value,
                                                             ReturnValue,
                                                             Matching::GridBasedDistance<Key>>;
            using Common::distance;
            using Common::nullValue;
            using Common::table;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Common(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(properties, nullValue)
            {
            }

            DistanceMatchingTable(GridBasedDistance const& distance,
                                  Properties const&        properties,
                                  ReturnValue              nullValue = ReturnValue())
                : Common(distance, properties, nullValue)
            {
            }

            std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                             Transform  transform) const
            {
                std::vector<ReturnValue> solutions = findTopKeyMatch(key, transform, 1);
                ReturnValue              solution  = this->nullValue;
                if(solutions.size() > 0)
                    solution = solutions[0];
                return std::make_tuple(solution, std::numeric_limits<double>::max());
            }

            std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const
            {
                if(Debug::Instance().printPropertyEvaluation())
                    return findBestKeyMatch_GridBased<true>(key, transform, numSolutions);
                else
                    return findBestKeyMatch_GridBased<false>(key, transform, numSolutions);
            }

            template <bool T_Debug>
            std::vector<ReturnValue> findBestKeyMatch_GridBased(Key const& key_orig,
                                                                Transform  transform,
                                                                int        numSolutions) const
            {
                std::vector<ReturnValue> bestmatches;
                if(this->table.empty())
                {
                    return bestmatches;
                }

                ptrdiff_t count = 0;
                bool      Debug = T_Debug;
                std::cout << std::setprecision(2) << std::fixed;

                Key key = key_orig;

                if(Debug::Instance().gridBasedKDTree())
                {
                    // roctxRangePush("KDTree");
                    auto compK = [](KBEntry<Value> const& e, int const N) { return e.k < N; };
                    auto lowerB = [](KBEntry<Value> const& e, int const N) { return e.b < N; };
                    auto upperB = [](int const N, KBEntry<Value> const& e) { return N < e.b; };
kd_tree_batch_1_again:
                    auto k     = key.size() > 3 ? key[3] : key[2];
                    auto                b     = key.size() > 3 ? key[2] : 1;
                    PointND<int32_t, 2> target;
                    target.coord[0] = key[0];
                    target.coord[1] = key[1];

                    if(Debug)
                    {
                        std::cout << "Searching space: " << this->kdTree.size() << '\n';
                    }

                    auto results = this->kdTree.query(
                        target, numSolutions, [this](auto pt, auto best) {
                            if(pt.coord[0] <= this->kdTree.maximums[0]
                               && pt.coord[1] <= this->kdTree.maximums[1])
                            {
                                return best.coord[0] >= pt.coord[0] && best.coord[1] >= pt.coord[1];
                            }
                            else if(pt.coord[0] > this->kdTree.maximums[0]
                                    && pt.coord[1] <= this->kdTree.maximums[1])
                            {
                                return best.coord[1] >= pt.coord[1];
                            }
                            else if(pt.coord[0] <= this->kdTree.maximums[0]
                                    && pt.coord[1] > this->kdTree.maximums[1])
                            {
                                return best.coord[0] >= pt.coord[0];
                            }
                            return true;
                        });

                    for(auto result : results)
                    {
                        if(Debug)
                            std::cout << "2D point: " << result.node->pt.coord[0] << ", "
                                      << result.node->pt.coord[1] << std::endl;

                        auto iter = this->kSolutionMap.find(
                            std::make_tuple(result.node->pt.coord[0], result.node->pt.coord[1]));
                        if(iter != this->kSolutionMap.end())
                        {
                            if(b > 1 && iter->second.size() == 1)
                            {
                                if(key[0] > key[1])
                                    key[0] = key[0] * key[2];
                                else
                                    key[1] = key[1] * key[2];
                                key[2] = 1;
                                goto kd_tree_batch_1_again;
                            }
                            auto bEnd = std::lower_bound(
                                iter->second.begin(), iter->second.end(), b, lowerB);
                            auto bStart = iter->second.begin();
                            if(bEnd == iter->second.end()) // b is too large
                            {
                                auto bval = std::prev(bEnd)->b;
                                bStart    = std::lower_bound(
                                    iter->second.begin(), iter->second.end(), bval, lowerB);
                            }
                            else if(bEnd == iter->second.begin()) // b is 1
                            {
                                bEnd = std::upper_bound(
                                    iter->second.begin(), iter->second.end(), b, upperB);
                            }
                            else
                            {
                                auto bval = bEnd->b - 1;
                                bStart    = std::lower_bound(
                                    iter->second.begin(), iter->second.end(), bval, lowerB);
                                bEnd++;
                            }
                            if(Debug)
                                std::cout << "bStart " << bStart->b << ", bEnd " << bEnd->b
                                          << " k at bStart " << bStart->k << ", k at bEnd "
                                          << bEnd->k << std::endl;

                            auto lower = std::lower_bound(bStart, bEnd, k, compK);
                            if(lower == bEnd)
                                lower--;
                            auto prev = lower == bStart ? lower : lower--;
                            if(prev != lower)
                            {
                                if(std::abs(prev->k - k) < std::abs(lower->k - k))
                                {
                                    lower = prev;
                                }
                            }
                            auto thisMatch = transform(lower->value);
                            if(thisMatch)
                            {
                                if(bestmatches.size())
                                {
                                    if(std::find(bestmatches.begin(), bestmatches.end(), thisMatch)
                                       != bestmatches.end())
                                    {
                                        if(Debug)
                                            std::cout << "Duplicated solution" << std::endl;
                                        continue;
                                    }
                                }
                                if(Debug)
                                    std::cout
                                        << "Final selected points: M: " << result.node->pt.coord[0]
                                        << ", N: " << result.node->pt.coord[1]
                                        << ", B: " << lower->b << ", K: " << lower->k << std::endl;

                                bestmatches.push_back(thisMatch);
                            }
                        }
                    }
                    // roctxRangePop();
                    return bestmatches;
                }

                // roctxRangePush("Binary");

                auto compM = [&count, Debug](Entry const& e, long const M) {
                    if(Debug)
                        printf("[ %ld ,%ld,%ld,%ld]\n", e.key[0], e.key[1], e.key[2], e.key[3]);
                    count++;
                    return e.key[0] < M;
                };

                auto compN = [&count, Debug](Entry const& e, long const N) {
                    if(Debug)
                        printf("[%ld, %ld ,%ld,%ld]\n", e.key[0], e.key[1], e.key[2], e.key[3]);
                    count++;
                    return e.key[1] < N;
                };

                auto compB = [&count, Debug](Entry const& e, long const B) {
                    if(Debug)
                        printf("[%ld,%ld, %ld ,%ld]\n", e.key[0], e.key[1], e.key[2], e.key[3]);
                    count++;
                    return e.key[2] < B;
                };

                auto origIter_M_lower = table.begin();
                auto origIter_M_upper = table.begin();
                auto origIter_N_lower = table.begin();
                auto origIter_N_upper = table.begin();
                auto origIter_B_lower = table.begin();
                auto origIter_B_upper = table.begin();
                auto start            = table.begin();
                auto batch            = table.begin();

                double bestDistance = std::numeric_limits<double>::max();
                auto   bestMatch    = this->nullValue;
                bool   thisMatch    = false;
                bool   uniqueSol    = false;
                bool   firstFind    = true;
                int    baseN        = 0;
                int    stepN        = 0;
                for(int i = 1;; i++)
                {
                    if(i * (i + 1) / 2 >= numSolutions)
                    {
                        baseN = i;
                        break;
                    }
                }

                while(origIter_B_upper != table.end() and bestmatches.size() < numSolutions)
                {
                    bestDistance = std::numeric_limits<double>::max();
                    thisMatch    = false;
                    uniqueSol    = false;
                    if(T_Debug)
                    {
                        std::cout << "Searching next MN... ";
                        std::cout << std::endl << std::endl;
                    }

                    if(stepN == baseN && baseN != 1)
                    {
                        stepN = 1;
                        baseN--;
                        start = origIter_M_upper; // find next M
                    }
                    else
                    {
                        stepN++;
                        start = origIter_N_upper; // find next N
                    }

                    origIter_M_lower = std::lower_bound(
                        start, table.end(), std::min(key[0], (table.end() - 1)->key[0]), compM);

                    if(T_Debug)
                    {
                        std::cout << "M lower: ";
                        streamJoin(std::cout, origIter_M_lower->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_M_upper = std::lower_bound(
                        origIter_M_lower,
                        table.end(),
                        std::min(origIter_M_lower->key[0] + 1, (table.end() - 1)->key[0] + 1),
                        compM);

                    if(T_Debug)
                    {
                        std::cout << "M upper: ";
                        streamJoin(std::cout, (origIter_M_upper - 1)->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_N_lower
                        = std::lower_bound(origIter_M_lower,
                                           origIter_M_upper,
                                           std::min(key[1], (origIter_M_upper - 1)->key[1]),
                                           compN);

                    if(T_Debug)
                    {
                        std::cout << "N lower: ";
                        streamJoin(std::cout, origIter_N_lower->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_N_upper = std::lower_bound(
                        origIter_N_lower,
                        origIter_M_upper,
                        std::min(origIter_N_lower->key[1] + 1, (origIter_M_upper - 1)->key[1] + 1),
                        compN);

                    if(T_Debug)
                    {
                        std::cout << "N upper: ";
                        streamJoin(std::cout, (origIter_N_upper - 1)->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_B_lower
                        = std::lower_bound(origIter_N_lower,
                                           origIter_N_upper,
                                           std::min(key[2], (origIter_N_upper - 1)->key[2]),
                                           compB);

                    if(T_Debug)
                    {
                        std::cout << "B lower: ";
                        streamJoin(std::cout, origIter_B_lower->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_B_upper = std::lower_bound(
                        origIter_B_lower,
                        origIter_N_upper,
                        std::min(origIter_B_lower->key[2] + 1, (origIter_N_upper - 1)->key[2] + 1),
                        compB);

                    if(T_Debug)
                    {
                        std::cout << "B upper: ";
                        streamJoin(std::cout, (origIter_B_upper - 1)->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    if(T_Debug)
                    {
                        std::cout << "K start point: ";
                        streamJoin(std::cout, origIter_B_lower->key, ", ");
                        std::cout << std::endl << std::endl;

                        std::cout << "K End point: ";
                        streamJoin(std::cout, (origIter_B_upper - 1)->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    for(auto iter = origIter_B_lower; iter != origIter_B_upper; iter++)
                    {

                        count++;

                        auto myDistance = distance(key, iter->key);
                        auto preBestDistance = bestDistance;

                        if(myDistance <= bestDistance)
                        {
                            auto myMatch = transform(iter->value);

                            if(myMatch)
                            {
                                batch        = iter;
                                bestDistance = myDistance;
                                bestMatch    = myMatch;
                                thisMatch    = true;
                            }
                        }

                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": " << myDistance;

                            if(myDistance < preBestDistance)
                                std::cout << " < ";
                            else if(myDistance > preBestDistance)
                                std::cout << " > ";
                            else
                                std::cout << " == ";

                            std::cout << preBestDistance;

                            if(myDistance <= preBestDistance)
                            {
                                if(thisMatch)
                                    std::cout << " <-- Best so far";
                                else
                                    std::cout << " <-- Best distance, but no matching solution";
                            }

                            std::cout << std::endl << std::endl;
                        }


                    }

                    if(!Debug::Instance().gridBasedBatchExp() && firstFind)
                    {
                        firstFind = false;
                        if(key.size() > 3)
                        {
                            if(key[2] != 1)
                            {
                                if(key[0] > key[1])
                                    key[0] = key[0] * key[2];
                                else
                                    key[1] = key[1] * key[2];
                                key[2] = 1;

                                if(T_Debug)
                                {
                                    std::cout << "New Size for batch grid: ["
                                              << key[0] << "," << key[1] << "," << key[2] << "," << key[3] << "]";
                                    std::cout << std::endl << std::endl;
                                }

                                if(batch->key[2] == 1)
                                {
                                    origIter_M_lower = table.begin();
                                    origIter_M_upper = table.begin();
                                    origIter_N_lower = table.begin();
                                    origIter_N_upper = table.begin();
                                    origIter_B_lower = table.begin();
                                    origIter_B_upper = table.begin();
                                    start            = table.begin();
                                    baseN            = 0;
                                    stepN            = 0;
                                    if(T_Debug)
                                    {
                                        std::cout << "Not found batch in grid, restart search with new size" << std::endl;
                                    }
                                    continue;
                                }
                            }
                        }
                    }

                    if(thisMatch or bestmatches.size())
                    {
                        if(std::find(bestmatches.begin(), bestmatches.end(), bestMatch)
                           == bestmatches.end())
                        {
                            bestmatches.push_back(bestMatch);
                            uniqueSol = true;
                        }
                        if(T_Debug)
                        {
                            if(uniqueSol)
                                std::cout << "add Best to Top Solutions" << std::endl << std::endl;
                            else
                                std::cout << "Best solution is duplicated" << std::endl
                                          << std::endl;
                        }
                    }
                }

                if(bestmatches.size() < numSolutions)
                {
                    if(T_Debug)
                    {
                        std::cout << std::endl
                                  << "Foward Search end but number of solutions not enough" << std::endl;
                        std::cout << "Start to backward search..." << std::endl;
                    }

                    for(auto iter = std::make_reverse_iterator(origIter_B_lower);
                        iter != table.rend();
                        iter++)
                    {
                        thisMatch       = false;
                        uniqueSol       = false;
                        auto myDistance = distance(key, iter->key);
                        auto myMatch    = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            thisMatch    = true;
                            if(std::find(bestmatches.begin(), bestmatches.end(), bestMatch)
                               == bestmatches.end())
                            {
                                bestmatches.push_back(bestMatch);
                                uniqueSol = true;
                            }
                        }

                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": " << myDistance;
                            if(thisMatch)
                                if(uniqueSol)
                                    std::cout << " (has a matching solution)";
                                else
                                    std::cout << " (duplicated solution)";
                            else
                                std::cout << " (no match)";
                            std::cout << std::endl;
                        }

                        if(bestmatches.size() == numSolutions)
                            break;
                    }
                }

                if((T_Debug || Debug::Instance().printLookupEfficiency()) && table.size() > 0)
                {
                    double considered = count;
                    considered /= table.size();
                    considered *= 100;
                    std::cout << "Considered " << count << "(" << considered << "%) of entries."
                              << std::endl;
                }

                if(T_Debug && bestmatches.size())
                    std::cout << "Solution index selected: " << bestmatches[0]->index << std::endl;

                // roctxRangePop();
                return bestmatches;
            }
        };
    } // namespace Matching
} // namespace TensileLite
