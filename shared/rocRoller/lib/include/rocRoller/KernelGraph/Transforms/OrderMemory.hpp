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
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief After OrderMemory, there should be no ambiguously ordered memory nodes of the same type.
         */
        ConstraintStatus NoAmbiguousNodes(const KernelGraph& k);
        ConstraintStatus NoBadBodyEdges(const KernelGraph& k);

        /**
         * @brief Ensure there are no ambiguous memory operations in the control graph.
         */
        class OrderMemory : public GraphTransform
        {
        public:
            OrderMemory(bool checkOrder = true)
                : m_checkOrder(checkOrder)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "OrderMemory";
            }

            std::vector<GraphConstraint> postConstraints() const override
            {
                if(m_checkOrder)
                {
                    return {&NoBadBodyEdges, &NoAmbiguousNodes};
                }
                return {&NoBadBodyEdges};
            }

        private:
            bool m_checkOrder = true;
        };
    }
}
