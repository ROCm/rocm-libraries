// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace SwizzleScaleDetail
    {
        /**
         * @brief Collect information about scale loads in a loop.
         *
         * Returns a map of load tags to pairs of multiply tags and macro tile tags.
         */
        std::map<int, std::pair<int, int>>
            collectScaleLoadInfo(KernelGraph& graph, NaryArgument arg, int loopTag);

        /**
         * @brief Order exchange operations before multiply operations in loop body.
         */
        void orderExchangesBeforeMultipliesInLoopBody(KernelGraph&       graph,
                                                      ContextPtr         context,
                                                      NaryArgument       arg,
                                                      std::map<int, int> tileExchangeMap,
                                                      std::map<int, std::pair<int, int>> scaleLoads,
                                                      int                                loopTag);

        /**
         * @brief Filter load unroll colouring for scale loads.
         */
        std::map<int, std::map<int, int>>
            filterLoadUnrollColouring(UnrollColouring const&                    colouring,
                                      std::map<int, std::pair<int, int>> const& scaleLoads);

        /**
         * @brief Add exchange coordinate transform.
         */
        std::vector<DeferredConnection> addExchangeCT(KernelGraph& graph,
                                                      ContextPtr   context,
                                                      int          macTileTag,
                                                      int          waveTileTag,
                                                      NaryArgument arg);

        /**
         * @brief Add swizzle load coordinate transform.
         */
        std::tuple<std::vector<DeferredConnection>,
                   std::vector<DeferredConnection>,
                   std::map<int, int>>
            addSwizzleLoadCT(KernelGraph& graph, ContextPtr context, int tag, NaryArgument arg);

        /**
         * @brief Get outer merge factors from macro tile.
         */
        std::pair<int, int> getOuterMergeFactors(KernelGraph const& graph, int macTileTag);

        /**
         * @brief Get inner merge factors from macro tile.
         */
        std::pair<int, int> getInnerMergeFactors(KernelGraph const& graph, int macTileTag);

        /**
         * @brief Find loads that can be merged.
         */
        std::map<int, std::vector<std::pair<int, int>>>
            findMergeableLoads(KernelGraph const&                        graph,
                               std::map<int, std::pair<int, int>> const& scaleLoads,
                               std::map<int, std::map<int, int>>&        loadUnrollMap,
                               NaryArgument                              arg);
    }
}
