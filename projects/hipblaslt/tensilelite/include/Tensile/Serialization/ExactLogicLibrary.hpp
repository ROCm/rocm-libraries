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

                // After deserialization, extract the target PCI chip ID from
                // the predicate tree so the runtime path is cast-free.
                if constexpr(std::is_same_v<MyPredicate, HardwarePredicate>)
                {
                    if(!iot::outputting(io))
                        row.first.targetPciChipId
                            = extractPciChipId(row.first.value.get());
                }
            }

            const static bool flow = false;

        private:
            // Walk the predicate tree once at deserialization to find
            // PciChipIdEqual and extract its target chip ID.
            static std::optional<int> extractPciChipId(Predicates::Predicate<Hardware> const* root)
            {
                if(!root)
                    return std::nullopt;

                auto const* isc = dynamic_cast<Predicates::IsSubclass<Hardware, AMDGPU> const*>(root);
                if(!isc || !isc->value)
                    return std::nullopt;

                return findPciChipId(isc->value.get());
            }

            static std::optional<int> findPciChipId(Predicates::Predicate<AMDGPU> const* pred)
            {
                if(!pred)
                    return std::nullopt;

                // Leaf
                if(auto const* pci = dynamic_cast<Predicates::GPU::PciChipIdEqual const*>(pred))
                    return pci->value;

                // Search children of composite predicates
                auto searchChildren = [](auto const& children) -> std::optional<int> {
                    for(auto const& child : children)
                    {
                        auto id = findPciChipId(child.get());
                        if(id.has_value())
                            return id;
                    }
                    return std::nullopt;
                };

                if(auto const* a = dynamic_cast<Predicates::And<AMDGPU> const*>(pred))
                    return searchChildren(a->value);
                if(auto const* o = dynamic_cast<Predicates::Or<AMDGPU> const*>(pred))
                    return searchChildren(o->value);
                if(auto const* n = dynamic_cast<Predicates::Not<AMDGPU> const*>(pred))
                    return findPciChipId(n->value.get());

                return std::nullopt;
            }
        };
    } // namespace Serialization
} // namespace TensileLite
