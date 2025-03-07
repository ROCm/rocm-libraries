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
#include <rocRoller/CommandSolution_fwd.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Flatten tile space and stream accumulation tiles.
         *
         * See `StreamKCoordinatetransformDesign`.
         *
         * The AddStreamK transformation is typically applied in
         * matrix-matrix multiply problems of the form D = A B where A
         * and B have been tiled with A: M x K tiles, and B: K x N
         * tiles.  Here the K tiles are the accumulation tiles.
         *
         * The `dims` parameter selects the free (M and N) dimensions.
         * The `topLoop` parameter selects the accumulation loop
         * (which was most likely created during the
         * LowerTensorContraction transformation).
         *
         * The AddStreamK transform creates a flattened "global tile
         * space" from all of the M/N/K tiles.  The flattened M/N/K
         * global tile-space is distributed evenly among the WGs.
         * Each WG iterates over its portion of the flattened global
         * tile-space; with the K tiles iterated over in the
         * inner-most "streaming" loop.
         *
         * The transformation is parameterised by:
         *
         * @param dims The sub-dimensions of dangling
         * `MacroTileNumber`s that should be included in the streaming
         * construct.
         *
         * @param tileNumberCoordSizes Sizes of `MacroTileNumber`s
         * matched by `dims`.
         *
         * @param topLoop Which loop to insert the local tile-loop
         * above.
         *
         * @param accmulatorLoop Which accumulation loop to stream.
         *
         * @param numWGs How many workgroups will be launched.
         */
        class AddStreamK : public GraphTransform
        {
        public:
            AddStreamK()                  = delete;
            AddStreamK(AddStreamK const&) = delete;

            AddStreamK(std::vector<int> const&   dims,
                       std::string const&        topLoop,
                       std::string const&        accumulatorLoop,
                       bool                      twoTile,
                       Expression::ExpressionPtr numWGs,
                       CommandParametersPtr      params,
                       ContextPtr                context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;

        private:
            CommandParametersPtr m_params;
            ContextPtr           m_context;

            /**
             * The sub-dimensions of dangling `MacroTileNumber`s that
             * should be included in the streaming construct.
             */
            std::vector<int> m_dimensionIndices;

            /**
             * Name of the loop to insert the local tile-loop above.
             */
            std::string m_topLoop;

            /**
             * Name of the accumulator (K) loop.
             */
            std::string m_accumulatorLoop;

            /**
             * Use two-tile SK + DP variant?
             */
            bool m_twoTile;

            /**
             * Number of Workgroups.
             *
             * An Expression that either:
             * 1. Pulls a value from a CommandArgument
             * 2. Is a literal (for testing)
             */
            Expression::ExpressionPtr m_numWGs;
        };
    }
}
