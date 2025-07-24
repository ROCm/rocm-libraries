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

#include <functional>
#include <optional>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        class TopologicalCompare
        {
        public:
            TopologicalCompare() = delete;
            TopologicalCompare(KernelGraphPtr graph)
                : m_graph(graph)
            {
                AssertFatal(graph);
            };

            bool operator()(int a, int b) const
            {
                return m_graph->control.compareNodes(rocRoller::UpdateCache, a, b)
                       == ControlGraph::NodeOrdering::LeftFirst;
            }

        private:
            KernelGraphPtr m_graph;
        };

        // Return value of colourByUnrollValue.  A colour-mapping is...
        struct UnrollColouring
        {
            std::map<int, std::map<int, int>>
                operationColour; //< Mapping: operation tag to colour-mapping.
            std::map<int, std::map<int, int>>
                          coordinateColour; //< Mapping: coordinate tag to colour-mapping.
            std::set<int> separators; //< Separator edges in the control graph
        };

        std::string toString(UnrollColouring const&);

        /**
         * @brief
         */
        UnrollColouring colourByUnrollValue(KernelGraph const&             kgraph,
                                            int                            topOp   = -1,
                                            std::unordered_set<int> const& exclude = {});

        /**
        * @brief Return DataFlowTag of LHS of binary expression in Assign node.
        */
        template <Expression::CBinary T>
        std::tuple<int, Expression::ExpressionPtr> getBinaryLHS(KernelGraph const& kgraph,
                                                                int                assign);

        /**
        * @brief Return DataFlowTag of RHS of binary expression in Assign node.
        */
        template <Expression::CBinary T>
        std::tuple<int, Expression::ExpressionPtr> getBinaryRHS(KernelGraph const& kgraph,
                                                                int                assign);

        /**
         * @brief Create a range-based for loop.
         *
         * returns {dimension, operation}
         */
        std::pair<int, int> rangeFor(KernelGraph&              graph,
                                     Expression::ExpressionPtr size,
                                     const std::string&        name,
                                     VariableType              vtype        = DataType::None,
                                     int                       forLoopCoord = -1);

        /**
         * @brief Remove a range-based for loop created by rangeFor.
         */
        void purgeFor(KernelGraph& graph, int tag);

        /**
         * @brief Create a clone of a ForLoopOp. This new ForLoopOp
         * will use the same ForLoop Dimension as the original
         * ForLoopOp.
        */
        int cloneForLoop(KernelGraph& graph, int tag);

        /**
         * @brief Remove a node and all of its children from the control graph
         *
         * Also purges the mapper of references to the deleted nodes.
         *
         * @param kgraph
         * @param node
         */
        void purgeNodeAndChildren(KernelGraph& kgraph, int node);

        template <std::ranges::forward_range Range = std::initializer_list<int>>
        void purgeNodes(KernelGraph& kgraph, Range nodes);

        bool isHardwareCoordinate(int tag, KernelGraph const& kgraph);
        bool isLoopishCoordinate(int tag, KernelGraph const& kgraph);
        bool isStorageCoordinate(int tag, KernelGraph const& kgraph);

        /**
         * @brief Filter coordinates by type.
         */
        template <typename T>
        std::unordered_set<int> filterCoordinates(auto const&        candidates,
                                                  KernelGraph const& kgraph);

        /**
         * @brief Find storage neighbour in either direction.
         *
         * Looks upstream and downstream for a neighbour that
         * satisfies isStorageCoordinate.
         *
         * If found, returns the neighbour tag, and the direction to
         * search for required coordinates.
         *
         * Tries upstream first.
         */
        std::optional<std::pair<int, Graph::Direction>>
            findStorageNeighbour(int tag, KernelGraph const& kgraph);

        /**
         * @brief Return Unroll coordinate beside (as part of a Split
         * edge) the ForLoop coordinate.
         */
        std::optional<int> findUnrollNeighbour(KernelGraph const& kgraph, int forLoopCoord);

        /**
        * @brief Return DataFlowTag of DEST of Assign node.
        */
        int getDEST(KernelGraph const& kgraph, int assign);

        /**
         * @brief Return target coordinate for load/store operation.
         *
         * For loads, the target is the source (User or LDS) of the
         * load.
         *
         * For stores, the target is the destination (User or LDS) of
         * the store.
         *
         * For load direct-to-lds, the target is the source (User) and destination (LDS) of
         * the operation.
         */
        std::pair<int, Graph::Direction>
            getOperationTarget(int tag, KernelGraph const& kgraph, bool isDirect2LDS = false);

        /**
         * Returns the true coordinate that should be the target of a
         * coordinate traversal, given a coordinate node used for storage.
         *
         * For now this will just follow any Duplicate edge leaving
         * `storageTarget`.
         */
        int getTransformTarget(int storageTarget, KernelGraph const& kgraph);

        /**
         * @brief Find all required coordintes needed to compute
         * indexes for the target dimension.
         *
         * @return Pair of: vector required coordinates; set of
         * coordinates in the connecting path.
         */
        std::pair<std::vector<int>, std::unordered_set<int>> findRequiredCoordinates(
            int target, Graph::Direction direction, KernelGraph const& kgraph);

        std::pair<std::vector<int>, std::unordered_set<int>>
            findRequiredCoordinates(int                      target,
                                    Graph::Direction         direction,
                                    std::function<bool(int)> fullStop,
                                    KernelGraph const&       kgraph);

        std::pair<std::unordered_set<int>, std::unordered_set<int>>
            findAllRequiredCoordinates(int op, KernelGraph const& graph);

        /**
         * @brief Return an augmented path that includes all
         * neighbours (in direction `direction`) of edges in the
         * original path.
         */
        std::unordered_set<int> includeEdgeNeighbours(
            rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph const& coordinates,
            Graph::Direction                                                direction,
            std::unordered_set<int> const&                                  path);

        /**
         * @brief Find the operation of type T that contains the
         * candidate load/store operation.
         */
        template <typename T>
        std::optional<int> findContainingOperation(int candidate, KernelGraph const& kgraph);

        /**
         * @brief Reconnect incoming/outgoing edges from op to newop.
         */
        template <Graph::Direction direction>
        void reconnect(KernelGraph& graph, int newop, int op);

        /**
         * @brief Find the operation of type T that contains the
         * candidate load/store operation. Then return the top element of the
         * body of that operation.
         */
        template <typename T>
        std::optional<int> findTopOfContainingOperation(int candidate, KernelGraph const& kgraph);

        /**
         * @brief Create a new coordinate representing data within the scratch space. This will return a
         * coordinate that can be added to a coordinate graph. It also allocates the required scratch space
         * within the context.
         *
         * @param size
         * @param varType
         * @param context
         * @return User
         */
        rocRoller::KernelGraph::CoordinateGraph::User newScratchCoordinate(
            Expression::ExpressionPtr size, VariableType varType, ContextPtr context);

        /**
         * @brief Replace operation with a new operation.
         *
         * @param op Operation to replace.
         * @param newOp Replacement.
         *
         * Does not delete the original operation.
         */
        int replaceWith(KernelGraph& graph, int op, int newOp, bool includeBody = true);

        /**
         * @brief Insert chain (from top to bottom) above operation.
         *
         * Bottom is attached to op via a Sequence edge.
         */
        void insertBefore(KernelGraph& graph, int op, int top, int bottom);

        /**
         * @brief Insert chain (from top to bottom) above operation.
         *
         * Top is attached to op via a Sequence edge.
         */
        void insertAfter(KernelGraph& graph, int op, int top, int bottom);

        /**
         * @brief Replace operation with a new operation.
         */
        void insertWithBody(KernelGraph& graph, int op, int newOp);

        /**
         * @brief Find load/store operations that need their indexes
         * precomputed by ComputeIndex.
         */
        std::vector<int> findComputeIndexCandidates(KernelGraph const& kgraph, int start);

        /**
         * Removes all CommandArgruments found within an expression
         * with the appropriate AssemblyKernel Argument.
         */
        Expression::ExpressionPtr cleanArguments(Expression::ExpressionPtr, AssemblyKernelPtr);

        /**
         * @brief Get ForLoop and increment (Linear) dimensions
         * assciated with ForLoopOp.
         */
        std::pair<int, int> getForLoopCoords(int forLoopOp, KernelGraph const& kgraph);

        template <CForwardRangeOf<int> Range>
        std::optional<int>
            getForLoopCoord(std::optional<int> forLoopOp, KernelGraph const& kgraph, Range within);

        /**
         * @brief Get a pair of expressions representing a for loop increment
         *
         * This assumes that there is only a single for loop increment for a given loop.
         *
         * This also assumes that the increment is of the form: Add(DataFlowTag(N), Val),
         * where N is the data tag associated with the for loop.
         *
         * The first item in the pair is the data flow tag associated with the for loop.
         *
         * The second item is the amount that it is being incremented by.
         *
         * @param graph
         * @param forLoop
         * @return std::pair<ExpressionPtr, ExpressionPtr>
         */
        std::pair<Expression::ExpressionPtr, Expression::ExpressionPtr>
            getForLoopIncrement(KernelGraph const& graph, int forLoop);

        void duplicateMacroTile(KernelGraph& graph, int tag);

        int duplicateControlNode(KernelGraph& graph, int tag);

        /**
         * @brief Delete a control node from the graph.
         */
        void deleteControlNode(KernelGraph& graph, int);

        /**
         * Updates the threadtile size for enabling the use of long dword instructions
         */
        void updateThreadTileForLongDwords(int& t_m,
                                           int& t_n,
                                           int  maxWidth,
                                           uint macTileFastMovingDimSize,
                                           int  numDwordsPerElement);

        /**
         * @brief Get the tag of the highest SetCoordinate directly upstream from load.
         *
         * @param graph
         * @param load
         * @return int
         */
        int getTopSetCoordinate(KernelGraph const& graph, int load);

        /**
         * @brief Get the unique tags of the highest SetCoordinate nodes directly upstream from each load.
         *
         * @param graph
         * @param loads
         * @return std::set<int>
         */
        std::set<int> getTopSetCoordinates(KernelGraph& graph, std::vector<int> loads);

        /**
         * @brief Get the tags of all of the SetCoordinate nodes directly upstream from node.
         *
         * @param graph
         * @param node
         * @return std::set<int>
         */
        std::set<int> getContainingSetCoordinates(KernelGraph const& graph, int node);

        /**
         * @brief Get the SetCoordinate object upstream from load that sets the
         * coordinate for the dimension dim.
         *
         * @param graph
         * @param dim
         * @param load
         * @return int
         */
        int getSetCoordinateForDim(KernelGraph const& graph, int dim, int load);

        /**
         * @brief Determine whether a matching SetCoordinate object exists upstream
         * from op for a given coordValue and coordTag.
         *
         * @param graph
         * @param op
         * @param coordValue
         * @param coordTag
         * @return bool
         */
        bool hasExistingSetCoordinate(KernelGraph const& graph,
                                      int                op,
                                      int                coordValue,
                                      int                coordTag);

        /**
         * Gets the unroll coordinate value that is set by a SetCoordinate node upstream
         * from the operation op, for the dimension unrollDim.
         */
        unsigned int getUnrollValueForOp(KernelGraph const& graph, int unrollDim, int op);

        /**
         * @brief Create duplicates of all of the nodes downstream of the provided
         *        start nodes.
         *        Add the duplicates to the provided graph.
         *        Return the location of the new start nodes.
         *
         * @param graph KernelGraph that nodes are duplicated from and into.
         * @param startNodes Starting nodes of sub-graph to duplicate.
         * @param reindexer Graph reindexer.
         * @param dontDuplicate Predicate to determine if a coordinate node is duplicated.
         *
         * @return New start nodes for the duplicated sub-graph.
         */
        template <std::predicate<int> Predicate>
        std::vector<int> duplicateControlNodes(KernelGraph&                    graph,
                                               std::shared_ptr<GraphReindexer> reindexer,
                                               std::vector<int> const&         startNodes,
                                               Predicate                       dontDuplicate);

        /**
         * @brief Return VariableType of load/store operation.
         */
        VariableType getVariableType(KernelGraph const& graph, int opTag);

        /**
         * @brief Add coordinate-transforms for storing a MacroTile
         * from a ThreadTile into global.
         *
         * Implemented in LowerTile.cpp.
         */
        void storeMacroTile_VGPR(KernelGraph&                     graph,
                                 std::vector<DeferredConnection>& connections,
                                 int                              userTag,
                                 int                              macTileTag,
                                 std::vector<int> const&          sdim,
                                 std::vector<unsigned int> const& jammedTiles,
                                 CommandParametersPtr             params,
                                 ContextPtr                       context);

        /**
         * @brief Add coordinate-transforms for loading a MacroTile
         * from global into a ThreadTile.
         */
        void loadMacroTile_VGPR(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              userTag,
                                int                              macTileTag,
                                std::vector<int> const&          sdim,
                                std::vector<unsigned int> const& jammedTiles,
                                CommandParametersPtr             params,
                                ContextPtr                       context,
                                bool                             isDirect2LDS = false);

        /**
         * @brief Store version of addLoadThreadTileCT.
         */
        void addStoreThreadTileCT(KernelGraph&                       graph,
                                  std::vector<DeferredConnection>&   connections,
                                  int                                macTileTag,
                                  int                                iMacX,
                                  int                                iMacY,
                                  std::array<unsigned int, 3> const& workgroupSizes,
                                  std::vector<unsigned int> const&   jammedTiles,
                                  bool                               useSwappedAccess,
                                  bool                               isDirect2LDS = false);

        /**
         * @brief Store version of addLoadMacroTileCT.
         */
        std::tuple<int, int, int, int>
            addStoreMacroTileCT(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              macTileTag,
                                std::vector<int> const&          sdim,
                                std::vector<unsigned int> const& jammedTiles = {1, 1});

        /**
         * @brief Store version of addLoad1DMacroTileCT.
         */
        std::tuple<int, int, int>
            addStore1DMacroTileCT(KernelGraph&                     graph,
                                  std::vector<DeferredConnection>& connections,
                                  int                              macTileTag,
                                  std::vector<int> const&          sdim,
                                  std::vector<unsigned int> const& jammedTiles = {1, 1});

        /**
         * @brief Add coordinate-transforms for tiling two
         * SubDimension coordinates into macro number/index
         * coordinates.
         *
         * The geometry of the tiling is taken from the MacroTile
         * associated with `macTileTag`.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         *
         * @return Tuple of: row MacroTileNumber, row MacroTileIndex,
         * column MacroTileNumber, column MacroTileIndex.
         */
        std::tuple<int, int, int, int>
            addLoadMacroTileCT(KernelGraph&                     graph,
                               std::vector<DeferredConnection>& connections,
                               int                              macTileTag,
                               std::vector<int> const&          sdim);

        /**
         * @brief Add coordinate-transforms for tiling the X
         * SubDimension coordinate into macro number/index
         * coordinates. No tiling is done on the Y SubDimension
         * coordinate, instead it is passed through to a macro
         * index coordinate only.
         *
         * The geometry of the tiling is taken from the MacroTile
         * associated with `macTileTag`.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         *
         * @return Tuple of: row MacroTileNumber, row MacroTileIndex,
         * column MacroTileIndex.
         */
        std::tuple<int, int, int> addLoad1DMacroTileCT(KernelGraph&                     graph,
                                                       std::vector<DeferredConnection>& connections,
                                                       int                              macTileTag,
                                                       std::vector<int> const&          sdim);

        /**
         * @brief Add coordinate-transforms for loading a ThreadTile
         * from row/column coordinates iMacX and iMacY.
         *
         * The geometry of the ThreadTile is taken from the MacroTile
         * associated with `macTileTag`.
         *
         * By default:
         *
         *   - For A/B matrix layouts, the Y thread tile number is
         *     fast wrt the workitem/lane index and the X thread tile
         *     number is slow.  For other layous, the X/Y thread tile
         *     numbers are taken from the X/Y workitem index.
         *
         *   - The row index of a thread tile is fast wrt the VGPR
         *     index.
         *
         * When `useSwappedAccess` is true, both of these orders are
         * reversed.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         */
        void addLoadThreadTileCT(KernelGraph&                       graph,
                                 std::vector<DeferredConnection>&   connections,
                                 int                                macTileTag,
                                 int                                iMacX,
                                 int                                iMacY,
                                 std::array<unsigned int, 3> const& workgroupSizes,
                                 std::vector<unsigned int> const&   jammedTiles,
                                 bool                               useSwappedAccess,
                                 bool                               isDirect2LDS = false);

        /**
         * @brief Create an internal tile backed by a ThreadTile.
         *
         * Implemented in LowerTile.cpp.
         */
        int createInternalTile(KernelGraph&         graph,
                               VariableType         varType,
                               int                  macTileTag,
                               CommandParametersPtr params,
                               ContextPtr           context);

        /**
         * @brief Create an internal tile backed by a ThreadTile.  The
         * internal tile is reduced in size according to numWaveTiles.
         *
         * Implemented in LowerTile.cpp.
         */
        int createInternalTile(KernelGraph&                     graph,
                               VariableType                     varType,
                               int                              macTileTag,
                               std::vector<unsigned int> const& numWaveTiles,
                               bool                             splitStore,
                               CommandParametersPtr             params,
                               ContextPtr                       context);

        /**
         * @brief Order all input pairs of memory nodes in graph.
         *
         * @param graph
         * @param pairs Pairs of memory nodes to be ordered.
         * @param ordered If true, the pairs are passed in order.
         */
        void orderMemoryNodes(KernelGraph&                         graph,
                              std::set<std::pair<int, int>> const& pairs,
                              bool                                 ordered);

        /**
         * @brief Order all memory nodes in srcs with respect to all memory nodes in dests.
         *
         * @param graph
         * @param srcs
         * @param dests
         * @param ordered If true, all orderings will be src -> dest.
         */
        void orderMemoryNodes(KernelGraph&         graph,
                              std::set<int> const& srcs,
                              std::set<int> const& dests,
                              bool                 ordered);

        /**
         * @brief Order all input nodes with respect to each other.
         *
         * @param graph
         * @param nodes
         * @param ordered If true, all orderings will be nodes[i-1] -> nodes[i].
         */
        void orderMemoryNodes(KernelGraph& graph, std::vector<int> const& nodes, bool ordered);

        /**
         * Replace the use of an old macrotile in the given control
         * nodes with a new macrotile.
         */
        void replaceMacroTile(KernelGraph&                   graph,
                              std::unordered_set<int> const& ops,
                              int                            oldMacTileTag,
                              int                            newMacTileTag);

        /**
         * @brief Move connections of LoadTiled and StoreLDSTile to new LoadTileDirect2LDS
         *
         *
         * @param kgraph
         * @param op LoadTiled or StoreLDSTile operation
         * @param newOp LoadTileDirect2LDS operation
         * @param subdimStride newSubdim = subdim + subdimStride
         *
         */
        void moveConnections(rocRoller::KernelGraph::KernelGraph& kgraph,
                             int                                  op,
                             int                                  newOp,
                             int                                  subdimStride);

        /**
        * @brief ceil(a/b) = (a+b-1)/b
        *
        * @param sdSize SubDimension size
        * @param tileSize MacroTile size
        *
        */
        Expression::ExpressionPtr tileCeilDivide(Expression::ExpressionPtr sdSize, int tileSize);

        /**
        * @brief Identifies whether a registerTag has an associated deallocate node.
        *
        * @param graph
        * @param registerTag
        *
        */
        bool hasDeallocate(const KernelGraph& graph, int tag);

        /**
         * @brief For LDS load/stores, follow DataFlow edges from the
         * LDS node to find the associated User node.
         */
        int getLDSOperationTarget(KernelGraph const& k, int opTag);

        /**
         * @brief Duplicate a chain of nodes
         */
        int duplicateChain(KernelGraph& graph, std::vector<int> const& startNodes);

        /**
        * @brief Get the unroll coordinate size, given the unroll coordinate tag.
        */
        unsigned int getUnrollSize(KernelGraph const& graph, int unroll);

        /**
        * @brief Get coordinates required by the code-generator.
        */
        std::vector<int> getCodeGeneratorCoordinates(KernelGraph const& graph,
                                                     int                tag,
                                                     bool               isDirect2LDS = false);

        /**
        * @brief Get the first and last nodes from a set of nodes that are totally ordered
        */
        template <typename T>
        std::pair<int, int> getFirstAndLastNodes(KernelGraph const& graph, T const& nodes);

        /**
        * @brief Remove redundant body edges in control graph. This is a baseline method for
        *        verifying correctness.
        */
        void removeRedundantBodyEdgesBaselineMethod(KernelGraph& graph);

        std::deque<int> controlStack(int control, KernelGraph const& graph);
        std::deque<int> controlStack(int control, ControlGraph::ControlGraph const& graph);
    }
}

#include <rocRoller/KernelGraph/Utils_impl.hpp>
