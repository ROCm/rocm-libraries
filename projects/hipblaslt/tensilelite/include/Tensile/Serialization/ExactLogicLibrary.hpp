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

#include <Tensile/ExactLogicLibrary.hpp>

#include <set>
#include <type_traits>

namespace TensileLite
{
    namespace Serialization
    {
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
