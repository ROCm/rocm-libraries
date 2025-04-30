/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Identify MacroTile dimensions that must be allocated
         * simultaneously but are used in a way that means they can share an
         * allocation, and add Alias edges to allow this to happen.
         *
         * Technically this could also apply to VGPR or other dimensions but
         * MacroTile is the primary way that this can save a lot of registers.
         *
         * E.g.:
         * >--> Program order (due to control graph sequence edges) >-->
         *              |   0   |   1   |  2   |   3   |  4   |   5   |  6  |
         * -------------|-------|-------|------|-------|------|-------|-----|
         * Loop:        |       | Begin |      |       |      |       | End |
         * MacroTile A: | Write |       | Read |       |      | Write |     |
         * MacroTile B: |       |       |      | Write | Read |       |     |
         *
         * In this case, MacroTile B could borrow MacroTile A's register since:
         *  - It will use and be done with the register (3-4) after A has read
         *    it (2).
         *  - The next use of A is to write to it (5), so the value in the
         *    register doesn't matter.
         *
         * This transformation will create an Alias edge from B to A,
         * indicating that B will borrow A's register allocation.
         *
         * @ingroup Transformations
         */
        class AliasDataFlowTags : public GraphTransform
        {
        public:
            AliasDataFlowTags() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AliasDataFlowTags";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {};
            }
        };
    }
}
