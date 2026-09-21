// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/RemapOutputTiles.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace RemapOutputTilesDetail
        {
            /**
             * @brief Workgroup count/size information.
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
             * @brief New dimensions created by workgroup mapping
             */
            struct RemappedDimensions
            {
                RemappedDimensions(int total, int parallel, int perpendicular)
                    : totalTiles(total)
                    , parallelDim(parallel)
                    , perpendicularDim(perpendicular)
                {
                }

                /**
                 * @brief The total number of output tiles
                 */
                int totalTiles;

                /**
                 * @brief Remapped dimension parallel to the workgroup mapping dimension
                 */
                int parallelDim;

                /**
                 * @brief Remapped dimension perpendicular to the workgroup mapping dimension
                 */
                int perpendicularDim;
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
             * The returned values are a dimension representing total number of tiles,
             * and the M/N dimensions after mapping.
             *
             * See shared/rocroller/docs/src/WorkgroupMapping.rst for more information.
             */
            RemappedDimensions workgroupMapping(TileSizeInfo const&                  info,
                                                rocRoller::KernelGraph::KernelGraph& graph,
                                                rocRoller::Graph::Direction          direction,
                                                uint                                 dimension,
                                                Expression::ExpressionPtr            size);

        }
    }
}
