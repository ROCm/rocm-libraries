/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <string>

#include <rocRoller/KernelGraph/ControlGraph/ControlEdge_fwd.hpp>
#include <rocRoller/KernelGraph/StructUtils.hpp>
#include <rocRoller/Utilities/Concepts.hpp>

namespace rocRoller
{

    namespace KernelGraph::ControlGraph
    {
        /**
         * Control graph edge types
         */

        /**
         * Sequence edges indicate sequential dependencies.  A node is considered schedulable
         * if all of its incoming Sequence edges have been scheduled.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Sequence);

        /**
         * Body edges indicate code nesting.  A Body node could indicate the body of a kernel,
         * a for loop, an unrolled section, an if statement (for the true branch), or any other
         * control block.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Body);

        /**
         * Else edge indicates the code that should be executed given a false condition.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Else);

        /**
         * Indicates code that should come before a Body edge.  Currently only applicable to
         * for loops.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Initialize);

        /**
         * Indicates the increment node(s) of a for loop.
         */
        RR_EMPTY_STRUCT_WITH_NAME(ForLoopIncrement);

        inline std::string toString(ControlEdge const& e)
        {
            return std::visit([](const auto& a) { return a.toString(); }, e);
        }

        template <CConcreteControlEdge Edge>
        inline std::string toString(Edge const& e)
        {
            return e.toString();
        }

        inline std::string name(ControlEdge const& e)
        {
            return toString(e);
        }

        template <typename T>
        concept CIsContainingEdge = CIsAnyOf<T, Body, Else, Initialize, ForLoopIncrement>;
    }
}
