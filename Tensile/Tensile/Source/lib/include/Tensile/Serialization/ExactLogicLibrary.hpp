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

#include <Tensile/Serialization/Base.hpp>
#include <Tensile/Serialization/Predicates.hpp>

#include <Tensile/ExactLogicLibrary.hpp>

namespace Tensile
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
            }

            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace Tensile
