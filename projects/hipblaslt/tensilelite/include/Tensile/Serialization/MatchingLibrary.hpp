/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Debug.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/MatchingLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>

#include <cstddef>
#include <tuple>
#include <unordered_set>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename Key,
                  typename MyProblem,
                  typename Element,
                  typename Return,
                  typename Distance,
                  typename IO>
        struct MappingTraits<
            Matching::DistanceMatchingTable<Key, MyProblem, Element, Return, Distance>,
            IO>
        {
            using Table
                = Matching::DistanceMatchingTable<Key, MyProblem, Element, Return, Distance>;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Table& table)
            {
                iot::mapRequired(io, "table", table.table);

                if(!iot::outputting(io))
                {
                    using Entry  = typename Table::Entry;
                    using KBEntry = typename Table::KBEntry;
                    using std::get;
                    auto comp    = [](Entry const& e1, Entry const& e2) {
                        return e1.key < e2.key || (e1.key == e2.key && e1.speed > e2.speed);
                    };
                    std::sort(table.table.begin(), table.table.end(), comp);

                    if constexpr(std::is_same<Distance, Matching::GridBasedDistance<Key>>{})
                    {
                        if(Debug::Instance().gridBasedKDTree())
                        {
                            // Creating K map
                            for(auto it = table.table.begin(); it != table.table.end(); ++it)
                            {
                                auto k   = it->key.size() > 3 ? it->key[3] : it->key[2];
                                auto b   = it->key.size() > 3 ? it->key[2] : 1;
                                auto key = std::tuple(it->key[0], it->key[1]);
                                table.kSolutionMap[key].emplace_back(KBEntry{static_cast<int32_t>(k), static_cast<int32_t>(b), it->value});
                            }

                            // Creating kd-tree
                            std::vector<Matching::PointND<int32_t, 2>> pts;
                            for(auto it = table.kSolutionMap.begin();
                                it != table.kSolutionMap.end();
                                ++it)
                            {
                                pts.push_back({get<0>(it->first), get<1>(it->first)});
                            }
                            table.kdTree.build(table.kdTree.root, begin(pts), end(pts), 0);
                        }
                    }
                }
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<ProblemMatchingLibrary<MyProblem, MySolution>, IO>
        {
            using Library    = ProblemMatchingLibrary<MyProblem, MySolution>;
            using Properties = typename Library::Table::Properties;
            using Element    = typename Library::Element;

            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                Properties properties;
                if(iot::outputting(io))
                {
                    properties = lib.table->properties;
                }

                iot::mapRequired(io, "properties", properties);

                bool success = false;
                if(properties.size() == 0)
                    iot::setError(io, "Matching table must have at least one property.");
                else if(properties.size() == 1)
                    success = mappingKey<std::array<int64_t, 1>>(io, lib, properties);
                else if(properties.size() == 2)
                    success = mappingKey<std::array<int64_t, 2>>(io, lib, properties);
                else if(properties.size() == 3)
                    success = mappingKey<std::array<int64_t, 3>>(io, lib, properties);
                else if(properties.size() == 4)
                    success = mappingKey<std::array<int64_t, 4>>(io, lib, properties);
                else if(properties.size() == 5)
                    success = mappingKey<std::array<int64_t, 5>>(io, lib, properties);
                else if(properties.size() == 6)
                    success = mappingKey<std::array<int64_t, 6>>(io, lib, properties);
                else if(properties.size() == 7)
                    success = mappingKey<std::array<int64_t, 7>>(io, lib, properties);
                else if(properties.size() == 8)
                    success = mappingKey<std::array<int64_t, 8>>(io, lib, properties);
                else if(properties.size() == 9)
                    success = mappingKey<std::array<int64_t, 9>>(io, lib, properties);
                else if(properties.size() == 10)
                    success = mappingKey<std::array<int64_t, 10>>(io, lib, properties);

                if(!success)
                    success = mappingKey<std::vector<int64_t>>(io, lib, properties);

                if(!success)
                    iot::setError(io, "Can't write out key: wrong type.");
            }

            template <typename Key>
            static bool mappingKey(IO& io, Library& lib, Properties const& properties)
            {
                std::string distanceType;

                if(iot::outputting(io))
                    distanceType = lib.table->distanceType();

                iot::mapRequired(io, "distance", distanceType);

                bool success = false;

                std::string tensile_metric = Debug::Instance().getMetric();

                if(!tensile_metric.empty())
                    distanceType = tensile_metric;

                if(distanceType == "Euclidean")
                {
                    success = mappingDistance<Key, Matching::EuclideanDistance<Key>>(
                        io, lib, properties);
                }
                else if(distanceType == "Equality")
                {
                    success = mappingDistance<Key, Matching::Equality<Key>>(io, lib, properties);
                }
                else if(distanceType == "Range")
                {
                    success = mappingDistance<Key, Matching::Range<Key>>(io, lib, properties);
                }
                else if(distanceType == "JSD")
                {
                    success
                        = mappingDistance<Key, Matching::JSDivergence<Key>>(io, lib, properties);
                }
                else if(distanceType == "Manhattan")
                {
                    success = mappingDistance<Key, Matching::ManhattanDistance<Key>>(
                        io, lib, properties);
                }
                else if(distanceType == "Ratio")
                {
                    success
                        = mappingDistance<Key, Matching::RatioDistance<Key>>(io, lib, properties);
                }
                else if(distanceType == "Random")
                {
                    success
                        = mappingDistance<Key, Matching::RandomDistance<Key>>(io, lib, properties);
                }
                else if(distanceType == "GridBased")
                {
                    success = mappingDistance<Key, Matching::GridBasedDistance<Key>>(
                        io, lib, properties);
                }
                else
                {
                    iot::setError(io, concatenate("Unknown distance function ", distanceType));
                }

                return success;
            }

            template <typename Key, typename Distance>
            static bool mappingDistance(IO& io, Library& lib, Properties const& properties)
            {
                using Table = Matching::DistanceMatchingTable<Key,
                                                              MyProblem,
                                                              Element,
                                                              std::shared_ptr<MySolution>,
                                                              Distance>;

                std::shared_ptr<Table> table;

                if(iot::outputting(io))
                {
                    table = std::dynamic_pointer_cast<Table>(lib.table);
                    if(!table)
                        return false;
                }
                else
                {
                    table             = std::make_shared<Table>();
                    table->properties = properties;
                    lib.table         = table;
                }

                MappingTraits<Table, IO>::mapping(io, *table);

                return true;
            }

            const static bool flow = false;
        };

        template <typename Key, typename Value, typename IO>
        struct MappingTraits<Matching::MatchingTableEntry<Key, Value>, IO>
        {
            using Entry = Matching::MatchingTableEntry<Key, Value>;
            using iot   = IOTraits<IO>;

            static void mapping(IO& io, Entry& entry)
            {
                int32_t index = -1;
                iot::mapRequired(io, "key", entry.key);
                iot::mapOptional(io, "speed", entry.speed);
                iot::mapOptional(io, "index", index);

                // TODO there is probably a way to make this properly general
                if(index != -1)
                {
                    using SSLibrary
                        = SingleSolutionLibrary<ContractionProblemGemm, ContractionSolution>;

                    auto ctx
                        = static_cast<LibraryIOContext<ContractionSolution>*>(iot::getContext(io));
                    if(ctx == nullptr || ctx->solutions == nullptr)
                    {
                        iot::setError(io,
                                      "SingleSolutionLibrary requires that context be set to "
                                      "a SolutionMap.");
                    }

                    auto iter = ctx->solutions->find(index);
                    if(iter == ctx->solutions->end())
                    {
                        std::ostringstream msg;
                        msg << "[MatchingLibrary] Invalid solution index: " << index;
                        iot::setError(io, msg.str());
                    }
                    else
                    {
                        std::shared_ptr<ContractionSolution> solution = iter->second;
                        entry.value = std::make_shared<SSLibrary>(solution);
                    }
                }
                else
                {
                    iot::mapRequired(io, "value", entry.value);
                }
            }

            const static bool flow = true;
        };

        template <typename Key, typename IO>
        struct MappingTraits<std::shared_ptr<Matching::Distance<Key>>, IO>
            : public BaseClassMappingTraits<Matching::Distance<Key>, IO, true>
        {
        };

        template <typename Key, typename IO>
        struct SubclassMappingTraits<Matching::Distance<Key>, IO>
            : public DefaultSubclassMappingTraits<
                  SubclassMappingTraits<Matching::Distance<Key>, IO>,
                  Matching::Distance<Key>,
                  IO>
        {
            using Self = SubclassMappingTraits<Matching::Distance<Key>, IO>;
            using Base
                = DefaultSubclassMappingTraits<SubclassMappingTraits<Matching::Distance<Key>, IO>,
                                               Matching::Distance<Key>,
                                               IO>;

            using SubclassMap = typename Base::SubclassMap;
            const static SubclassMap subclasses;

            static SubclassMap GetSubclasses()
            {
                return SubclassMap({Base::template Pair<Matching::RatioDistance<Key>>(),
                                    Base::template Pair<Matching::ManhattanDistance<Key>>(),
                                    Base::template Pair<Matching::EuclideanDistance<Key>>(),
                                    Base::template Pair<Matching::RandomDistance<Key>>(),
                                    Base::template Pair<Matching::GridBasedDistance<Key>>()});
            }
        };

        template <typename Key, typename IO>
        const typename SubclassMappingTraits<Matching::Distance<Key>, IO>::SubclassMap
            SubclassMappingTraits<Matching::Distance<Key>, IO>::subclasses
            = SubclassMappingTraits<Matching::Distance<Key>, IO>::GetSubclasses();

        template <typename Key, typename IO>
        struct MappingTraits<Matching::RatioDistance<Key>, IO>
            : public AutoMappingTraits<Matching::RatioDistance<Key>, IO>
        {
        };

        template <typename Key, typename IO>
        struct MappingTraits<Matching::ManhattanDistance<Key>, IO>
            : public AutoMappingTraits<Matching::ManhattanDistance<Key>, IO>
        {
        };

        template <typename Key, typename IO>
        struct MappingTraits<Matching::EuclideanDistance<Key>, IO>
            : public AutoMappingTraits<Matching::EuclideanDistance<Key>, IO>
        {
        };

        template <typename Key, typename IO>
        struct MappingTraits<Matching::RandomDistance<Key>, IO>
            : public AutoMappingTraits<Matching::RandomDistance<Key>, IO>
        {
        };

        template <typename Key, typename IO>
        struct MappingTraits<Matching::GridBasedDistance<Key>, IO>
            : public AutoMappingTraits<Matching::GridBasedDistance<Key>, IO>
        {
        };
    } // namespace Serialization
} // namespace TensileLite
