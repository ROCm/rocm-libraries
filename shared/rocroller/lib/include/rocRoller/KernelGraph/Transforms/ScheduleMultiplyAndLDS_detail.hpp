// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ScheduleMultiplyAndLDS.hpp"

namespace rocRoller::KernelGraph
{
    namespace ScheduleMultiplyAndLDSDetail
    {
        /**
         * Glossary: Within this namespace, a 'chain' refers to
         *
         *  - a series of nodes of the same type that are all directly connected to each other
         *    with Sequence edges, or
         *  - a vector containing the node IDs of those nodes, or
         *  - a vector containing the node IDs ot those nodes, but filtered according to
         *    some criteria.
         */

        using Chain  = std::vector<int>;
        using Chains = std::vector<Chain>;
        using Groups = std::vector<Chains>;

        /**
         * Identifies groups within `nodes` that are directly connected in a linear chain.
         *
         * If the connections between `nodes` is:
         *
         * 1 -> 2 -> 3 -> 4 -> NOP -> 5 -> 6 -> 7 -> 8
         *
         * This would return 2 vectors, {1, 2, 3, 4} and {5, 6, 7, 8}.
         */
        Chains makeChains(KernelGraph const& graph, std::vector<int> nodes);

        /**
         * Given a SetCoordinate node, find the actual memory node associated with it.
         * Given a memory node, just return that node.
         *
         * SetCoordinate(3) -> Body -> SetCoordinate(4) -> Body -> LoadLDSTile(5)
         *
         * getLDSNode(graph, 3) should return 5.
         * getLDSNode(graph, 5) should also return 5.
         */
        int getLDSNode(KernelGraph const& graph, int node);

        /**
         * Get the data type associated with the memory node associated with the
         * SetCoordinate node.
         */
        DataType getLDSType(KernelGraph const& graph, int node);

        using ChainTypes = std::vector<std::map<DataType, int>>;

        std::string toString(std::map<DataType, int> const& typeCounts);

        std::string toString(ChainTypes const& chainTypes);

        /**
         * Given a chain of (multiply) nodes, modify it to only contain those nodes that are the
         * last nodes that read a DataFlowTag. Returns a parallel vector containing the data
         * types read by each node.
         */
        ChainTypes filterLastCoordinateReads(KernelGraph const& graph, Chain& chain);

        /**
         * Given a graph,
         *
         * - Finds chains of multiply nodes
         * - Filters those chains to only the ones that are the last reads of some tag.
         * - Returns those chains as well as info about the data types of those tags.
         */
        std::tuple<Chains, std::vector<ChainTypes>>
            findMultiplyChainsAndCoords(KernelGraph const& graph);

        void getImmediateBodyParents(KernelGraph const& graph, std::vector<int>& nodes);

        Chains findLoadLDSChains(KernelGraph const& graph);
        Chains findMultiplyChains(KernelGraph const& graph);

        /**
         * Logs a nice table that will show which nodes in `chain` use which DataFlowTags, and
         * what type they are. Very useful for determining what the schedule should be.
         */
        void logChainTagTable(KernelGraph const& graph, Chain chain);

        /**
         * Generates a nice table that will show which nodes in `chain` use which DataFlowTags, and
         * what type they are. Very useful for determining what the schedule should be.
         */
        std::string chainTagTable(KernelGraph const& graph, Chain chain);

        std::string showChain(Chain const& chain);

        std::string showChains(Chains const& chains);

        std::string showGroups(Groups const& groups);

        /**
         * Given groups of chains (grouped by node type), identifies which chains of different
         * types are parallel to each other. Returns sets of chains that each contain at least 2
         * chains and are parallel to each other.
         *
         * E.g. Input:
         *
         * {
         *     {
         *         {chain of multiply A}, {chain of multiply B}, {chain of multiply C}
         *     },
         *     {
         *         {chain of LoadLDSTile E}, {chain of LoadLDSTile F},
         *         {chain of LoadLDSTile G}, {chain of LoadLDSTile H},
         *     }
         * }
         *
         * Let's say that:
         *     * Chain E is not parallel to anything because it's before the K loop
         *       (it's for prefetching)
         *     * Chain A is parallel to chain F
         *     * Chain B is parallel to chain G
         *     * Chain C is parallel to chain H
         *
         * The return value will be:
         * {
         *      { {chain A}, {chain F} },
         *      { {chain B}, {chain G} },
         *      { {chain C}, {chain H} }
         * }
         *
         * If instead, Chain A is not parallel to anything, then the return value will be:
         * {
         *      {},
         *      { {chain B}, {chain G} },
         *      { {chain C}, {chain H} }
         * }
         */
        Groups identifyParallelChains(KernelGraph const& graph, Groups groups);

        struct ParallelChainSet
        {
            Chain multiplyChain;
            Chain ldsChain;

            ChainTypes            multiplyTagTypes;
            std::vector<DataType> ldsChainTypes;
        };

        /**
         * Allows the multiply nodes to get a head start so that we can hide
         * more of the arithmetic and global loads at the beginning of each
         * subiter.
         */
        std::map<DataType, int> getMultiplyHeadStarts(KernelGraph&            graph,
                                                      ParallelChainSet const& chainSet);

        std::vector<ParallelChainSet>
            identifyParallelMultiplyAndLDSChainsWithTypes(KernelGraph const& graph);

        void distributeChains(KernelGraph& graph, std::vector<ParallelChainSet> const& chainSets);
    }

}
