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

#include <vector>

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Connect unbound (leaf) MacroTileNumber coordinates
         * to a linearised ForLoop+Workgroup coordinate pair.
         *
         * This transform searches for MacroTileNumber coordinates
         * that are leafs (don't have outgoing/incoming edges), and
         * transforms them into a Workgroup+ForLoop pair.  That is,
         * tiles are iterated over using a combination of an implicit
         * workgroup and an explicit for-loop.
         *
         * For example, suppose we have a graph with dangling
         * MacroTileNumbers with sub-dimensions 0 and 1; and we want
         * to iterate over those using a combined Workgroup+ForLoop
         * coordinate pair.
         *
         * We create:
         *
         *    WG = Workgroup()
         *    FL = ForLoop()
         *    LinearTileIterationCoord = Flatten(WG, FL)
         *
         *    WG0 = MacroTileNumber(SubDimension=0)
         *    WG1 = MacroTileNumber(SubDimension=1)
         *
         *    WG0, WG1 = Tile(LinearTileIterationCoord)
         *
         * Then we connect all pre-exisitng leaf-MacroTileNumbers to
         * either WG0 or WG1 (matching on the SubDimension).
         *
         * The transformation is parameterised by:
         *
         * @param dims The sub-dimensions that should be iterated over
         * using the new Workgroup+ForLoop pair.
         *
         * @param tileNumberCoordSizes How many tiles are in each
         * sud-dimension.
         *
         * @param numIteratedTiles How many tiles are iterated over
         * using a ForLoop.
         *
         * @param topLoop Where to insert the new ForLoopOp in the
         * control graph.
         *
         * Note that the kernel should be launched so that the number
         * of Workgroups matches:
         *
         *    Launched workgroups = product(tileNumberCoordSizes) * numIteratedTiles
         */
        class LoopOverTileNumbers : public GraphTransform
        {
        public:
            LoopOverTileNumbers() = delete;
            LoopOverTileNumbers(std::vector<int> const&  dims,
                                std::vector<uint> const& tileNumberCoordSizes,
                                uint                     numIteratedTiles,
                                std::string const&       topLoop,
                                ContextPtr               context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;

        private:
            void stage(KernelGraph const& graph);
            void commit(KernelGraph& graph);

            ContextPtr m_context;

            // Location
            std::vector<int> m_dimensions;
            std::string      m_topLoop;

            int m_topLoopOp;

            // Parameters
            std::vector<Expression::ExpressionPtr> m_tileNumberCoordSizes;
            Expression::ExpressionPtr              m_numIteratedTileNumbers;

            // Staged MacroTileNumber coodinates
            //
            // Mapping: dimension -> set of MacroTileNumber coordinates
            std::map<int, std::unordered_set<int>> m_tileNumberCoords;
        };
    }
}
