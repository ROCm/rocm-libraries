// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <unordered_set>
#include <vector>

#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace RemoveSetCoordinateDetails
        {
            /**
             * Find the leave nodes (in terms of program order) inside the subgraph rooted
             * at a given node.
            */
            void findLeaves(int                      tag,
                            KernelGraph const&       kg,
                            std::unordered_set<int>& visited,
                            std::vector<int>&        leaves);

            /**
             * Find the leave nodes (in terms of program order) inside the subgraphs rooted
             * at a given set of nodes.
            */
            std::vector<int> findLeaves(std::vector<int> nodes, KernelGraph const& kg);

            /**
             * Remove all SetCoordinate nodes from a control graph of a given kernel graph
            */
            void removeSetCoordinates(KernelGraph& kg);
        }
    }
}
