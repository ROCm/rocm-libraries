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
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace ConnectWorkgroupsDetail
        {
            /**
             * @brief Workgroup count/size information.
             *
             *
             */
            struct TileSizeInfo
            {
                /**
                 * @brief MacroTileNumber size expressions for each dimension.
                 *
                 * The size of dangling MacroTileNumber coordinates.
                 */
                std::array<Expression::ExpressionPtr, 3> sizes;

                /**
                 * @brief Map from {dim, direction} pairs to set of
                 * disconnected (dangling) MacroTileNumber
                 * coordinates.
                 */
                std::map<std::pair<int, rocRoller::Graph::Direction>, std::unordered_set<int>>
                    danglers;

                /**
                 * @brief Record the size of a tile.
                 */
                void recordSize(int dim, int tileNumTag, auto direction, auto expr);
            };

            /**
             * @brief Query the graph and return TileSizeInfo.
             */
            TileSizeInfo getTileSizeInfo(KernelGraph const& kgraph);

            /**
             * @brief Return number of active dimensions (one, two, or three).
             */
            int workgroupDimensions(TileSizeInfo const& info);

            /**
             * @brief Return total number of workgroups (product of sizes).
             *
             * This matches the number of workgroups required for launch.
             */
            Expression::ExpressionPtr totalNumberOfWorkgroups(TileSizeInfo const& info);

            /**
             * @brief Connect dangling MacroTileNumber coordinate to
             * matching Workgroup coordinates.
             *
             * No mapping is done.  That is; MacroTileNumber(dim=X)
             * gets PassThrough'd to Workgroup(dim=X).
             */
            std::map<std::pair<int, rocRoller::Graph::Direction>, int>
                connectWorkgroupsNoMapping(TileSizeInfo const& info, KernelGraph& kgraph);

            /**
             * @brief Connect dangling MacroTileNumber coordinate to
             * matching Workgroup coordinates.
             *
             * Performs Workgroup Mapping (via workgroupMapping).
             */
            void connectWorkgroupsWithMapping(TileSizeInfo const&                  info,
                                              rocRoller::KernelGraph::KernelGraph& graph,
                                              int                                  dimension,
                                              Expression::ExpressionPtr            size);

            /**
             * @brief Apply Workgroup Mapping.
             *
             * Map workgroups to tiles in a Z-order-inspired blockwise manner where
             * the blocks are divided/bounded by `size` along `dimension` (M=0 or N=1).
             *
             * TODO add a more descriptive comment.
             */
            std::tuple<int, int, int> workgroupMapping(TileSizeInfo const&                  info,
                                                       rocRoller::KernelGraph::KernelGraph& graph,
                                                       rocRoller::Graph::Direction direction,
                                                       uint                        dimension,
                                                       Expression::ExpressionPtr   size);

            /**
             * @brief Remap Workgroup to be more cache friendly
             * (consecutive workgroups land within the same XCC).
             *
             * Modifies the coordinate graph.
             *
             * Returns the newly added Workgroup dimension.
             */
            int remapWorkgroupXCC(rocRoller::KernelGraph::KernelGraph& graph,
                                  int                                  workgroupTag,
                                  uint                                 numXCC);
        }
    }
}
