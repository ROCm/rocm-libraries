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

#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Return value for any function that applies constraints to the KernelGraph.
         *
         */
        struct ConstraintStatus
        {
            bool        satisfied   = true;
            std::string explanation = "";

            void combine(bool inputSat, std::string inputExpl)
            {
                satisfied &= inputSat;
                if(!explanation.empty() && !inputExpl.empty())
                {
                    explanation += "\n";
                }
                explanation += inputExpl;
            }

            void combine(ConstraintStatus input)
            {
                combine(input.satisfied, input.explanation);
            }
        };

        ConstraintStatus NoDanglingMappings(const KernelGraph& k);
        ConstraintStatus SingleControlRoot(const KernelGraph& k);
        ConstraintStatus NoRedundantSetCoordinates(const KernelGraph& k);

        using GraphConstraint = ConstraintStatus (*)(const KernelGraph& k);
    }
}
