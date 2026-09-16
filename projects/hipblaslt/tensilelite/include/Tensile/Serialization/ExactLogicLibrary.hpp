/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Serialization/Base.hpp>
#include <Tensile/Serialization/Predicates.hpp>
#include <Tensile/Serialization/PredictionLibrary.hpp>

#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/MatchingLibrary.hpp>
#include <Tensile/PredictionLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>

#include <set>
#include <type_traits>
#include <unordered_set>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    namespace Serialization
    {
        /**
         * @brief Put the Equality row's kernels into the Prediction row's pool.
         *
         * The two rows sit side by side in one ProblemSelectionLibrary but only
         * one of them is ever consulted: findTopSolutions skips Equality and
         * Range outright whenever the prediction library is in use
         * (ExactLogicLibrary.hpp, the predictionLib guard). So the Equality
         * kernels are unreachable through ranking, and on this workload they
         * are unreachable through their own row too -- it keys on an exact
         * [M, N, batch, K] and matches none of the benchmark shapes. Moving
         * them into the pool is what puts them in front of rank_configs.
         *
         * Deliberately not filtered. A large share of these kernels cannot be
         * ranked at any shape, and others will tie with each other to the last
         * bit of predicted latency; both are the behaviour under study.
         *
         * Runs at deserialization, before the library is reachable by any
         * other thread, so nothing here needs to be synchronised.
         */
        template <typename MyProblem, typename MySolution, typename MyPredicate>
        void mergeEqualityIntoPredictionPool(
            std::vector<LibraryRow<MyProblem, MySolution, MyPredicate>>& rows)
        {
            using Matching   = ProblemMatchingLibrary<MyProblem, MySolution>;
            using Prediction = ProblemPredictionLibrary<MyProblem, MySolution>;
            using Single     = SingleSolutionLibrary<MyProblem, MySolution>;

            std::shared_ptr<Prediction>              prediction;
            std::vector<std::shared_ptr<MySolution>> equality;

            for(auto const& row : rows)
            {
                if(auto found = std::dynamic_pointer_cast<Prediction>(row.second))
                {
                    prediction = found;
                    continue;
                }

                // The same test the rest of this library uses to recognise the
                // Equality row, rather than trusting row order.
                if(!dynamic_cast<Predicates::Contraction::EqualityMatching*>(row.first.value.get()))
                    continue;

                auto matching = std::dynamic_pointer_cast<Matching>(row.second);
                if(!matching || !matching->table)
                    continue;

                // GetAll() is the whole table rather than what some problem
                // matches, which is the point: there is no problem yet.
                for(auto const& entry : matching->table->GetAll())
                {
                    if(auto single = std::dynamic_pointer_cast<Single>(entry))
                    {
                        if(single->solution)
                            equality.push_back(single->solution);
                    }
                }
            }

            if(!prediction || equality.empty() || prediction->mergedEntryCount)
                return;

            // The table lists a pinned winner per benchmarked size, so one
            // kernel can appear under many sizes; the pool wants it once.
            std::unordered_set<int> seen;
            for(auto const& entry : prediction->solution_list)
                seen.insert(entry.first);

            for(auto const& solution : equality)
            {
                if(!seen.insert(solution->index).second)
                    continue;

                solution->fromEqualityPool = true;
                prediction->addMergedEntry(
                    solution->index, solution, makeOrigamiConfig(*solution));
            }

            if(Debug::Instance().printLibraryVersion())
            {
                std::cerr << "TensileLite: merged " << prediction->mergedEntryCount
                          << " Equality kernels into a Prediction pool of "
                          << prediction->mergedBegin() << ", giving "
                          << prediction->origami_config_list.size() << " configs to rank\n";
            }
        }

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<HardwareSelectionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = HardwareSelectionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                iot::mapRequired(io, "rows", lib.rows);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<ProblemSelectionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = ProblemSelectionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                iot::mapRequired(io, "rows", lib.rows);

                // Both rows the merge needs are in this one library, and they
                // are only both present now that the whole list has been read.
                if(!iot::outputting(io) && Debug::Instance().mergeEqualityIntoPredictionPool())
                    mergeEqualityIntoPredictionPool(lib.rows);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename MyPredicate, typename IO>
        struct MappingTraits<LibraryRow<MyProblem, MySolution, MyPredicate>, IO>
        {
            using Row = typename ExactLogicLibrary<MyProblem, MySolution, MyPredicate>::Row;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Row& row)
            {
                iot::mapRequired(io, "predicate", row.first.value);
                iot::mapRequired(io, "library", row.second);

                // After deserialization, extract target PCI chip IDs from
                // the predicate tree so the runtime path is cast-free.
                if constexpr(std::is_same_v<MyPredicate, HardwarePredicate>)
                {
                    if(!iot::outputting(io))
                        row.first.targetPciChipIds
                            = extractPciChipIds(row.first.value.get());
                }
            }

            const static bool flow = false;

        private:
            // Walk the predicate tree once at deserialization to find all
            // PciChipIdEqual nodes and extract their target chip IDs.
            static std::set<int> extractPciChipIds(Predicates::Predicate<Hardware> const* root)
            {
                if(!root)
                    return {};

                auto const* isc = dynamic_cast<Predicates::IsSubclass<Hardware, AMDGPU> const*>(root);
                if(!isc || !isc->value)
                    return {};

                return findPciChipIds(isc->value.get());
            }

            static std::set<int> findPciChipIds(Predicates::Predicate<AMDGPU> const* pred)
            {
                if(!pred)
                    return {};

                // Leaf
                if(auto const* pci = dynamic_cast<Predicates::GPU::PciChipIdEqual const*>(pred))
                    return {pci->value};

                // Search children of composite predicates
                auto searchChildren = [](auto const& children) -> std::set<int> {
                    std::set<int> ids;
                    for(auto const& child : children)
                    {
                        auto childIds = findPciChipIds(child.get());
                        ids.insert(childIds.begin(), childIds.end());
                    }
                    return ids;
                };

                if(auto const* a = dynamic_cast<Predicates::And<AMDGPU> const*>(pred))
                    return searchChildren(a->value);
                if(auto const* o = dynamic_cast<Predicates::Or<AMDGPU> const*>(pred))
                    return searchChildren(o->value);
                if(auto const* n = dynamic_cast<Predicates::Not<AMDGPU> const*>(pred))
                    return findPciChipIds(n->value.get());

                return {};
            }
        };
    } // namespace Serialization
} // namespace TensileLite

