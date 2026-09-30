// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

        /**
         * A functor that compares two nodes in a control graph in topological
         * order. This is provided as a convenience to use with std::sort and
         * other STL algorithms. This object does not own the graph, so the graph
         * must be valid for the lifetime of the functor.
         */
        class TopologicalCompare
        {
        public:
            TopologicalCompare() = delete;

            explicit TopologicalCompare(KernelGraph const* graph)
                : m_graph(graph)
            {
                AssertFatal(graph);
            };

            explicit TopologicalCompare(KernelGraph const& graph)
                : TopologicalCompare(&graph)
            {
            }

            bool operator()(int a, int b) const
            {
                if(a == b)
                    return false;

                return m_graph->control.compareNodes(rocRoller::UpdateCache, a, b)
                       == ControlGraph::NodeOrdering::LeftFirst;
            }

        private:
            KernelGraph const* m_graph;
        };

        /**
         * @brief Mapping from unroll-coordinate to unroll-value.
         *
         * For example, consider a control-subgraph similar to
         *
         *     SetCoordinate(coordinate=4, value=0)
         *     SetCoordinate(coordinate=5, value=1)
         *
         * where coordinate tags 4 and 5 are both correspond to Unroll
         * coordinates.
         *
         * The ColourMapping would be {4:0, 5:1}.
         */
        using ColourMapping = std::map<int, int>;

        /**
         * @brief Colouring of operations and coordinates by unroll value.
         *
         * Return value of `colourByUnrollValue`.
         */
        struct UnrollColouring
        {
            std::map<int, ColourMapping> operationColour; //< Operation colouring.
            std::map<int, ColourMapping> coordinateColour; //< Coordinate colouring.
            std::set<int>                separators; //< Separator edges in the control graph
        };

        std::string toString(UnrollColouring const&);

        /**
         * @brief Colouring of operations and coordinates by NaryArgument.
         *
         * Return value of `colourByNaryArgument`.
         */
        struct NaryArgumentColouring
        {
            std::map<int, NaryArgument> coordinateColour; //< Coordinate colouring.
            std::map<int, NaryArgument> operationColour; //< Control operation colouring.
        };

        /**
         * @brief Colour coordinates and operations by NaryArgument
         * based on Multiply operations.
         *
         * Traverses the graph starting from `start` (or the entire
         * graph if `start` is -1) and finds all Multiply
         * operations. For each multiply, determines which coordinates
         * and control operations contribute to each argument (LHS,
         * LHS_SCALE, RHS, RHS_SCALE) by tracing backward
         * dependencies.
         *
         * @param graph The kernel graph to analyze
         * @param start Starting control operation (-1 for entire graph)
         * @return NaryArgumentColouring with coordinate and control mappings
         */
        NaryArgumentColouring colourByNaryArgument(KernelGraph const& graph, int start = -1);

        /**
         * @brief Colour operations and coordinates by unroll value.
         *
         * Starts at `topOp` (or the top of the graph if `topOp` is
         * -1) and traverses its body.
         *
         * Unroll tags in `exclude` are excluded from the colouring
         * (ignored).
         *
         * For example, consider a control-subgraph similar to
         *
         *     SetCoordinate(coordinate=4, value=0)
         *         SetCoordinate(coordinate=5, value=1)
         *             A = LoadTiled()
         *         SetCoordinate(coordinate=7, value=3)
         *             C = LoadTiled()
         *     B = Assign(A + C)
         *
         * Then:
         *
         * - The LoadTiled operation for A will be coloured {4:0, 5:1}.
         * - The LoadTiled operation for C will be coloured {4:0, 7:3}.
         * - The A coordinate will be coloured {4:0, 5:1}.
         * - The C coordinate will be coloured {4:0, 7:3}.
         * - The Assign operation for B will be coloured {4:0, 5:1, 7:3}.
         * - The B coordinate will be coloured {4:0, 5:1, 7:3}.
         */
        UnrollColouring colourByUnrollValue(KernelGraph const&             kgraph,
                                            int                            topOp   = -1,
                                            std::unordered_set<int> const& exclude = {});

        /**
         * @brief Get the name of a ForLoop operation.
         */
        std::string getForLoopName(KernelGraph const& graph, int tag);

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
         * @brief Return the edge tag of type EdgeType connected to tag in direction Direction.
         */
        template <Graph::Direction Direction, typename EdgeType>
        std::optional<int> GetEdgeTag(KernelGraph const& graph, int tag);

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
        int cloneForLoop(KernelGraph&               graph,
                         int                        tag,
                         std::optional<std::string> name = std::nullopt);

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
         * For Global to LDS ops (e.g buffer to lds), target is User for the
         * load from global part and LDS for the store into LDS part.
         */
        std::pair<int, Graph::Direction> getOperationTarget(int                tag,
                                                            KernelGraph const& kgraph,
                                                            bool isStorePartOfGlobalToLDSOp
                                                            = false);

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
         * @brief Follow Identify edges.
         *
         * Starting from the coordinate tag, follow outgoing Identify
         * edges until there aren't any more.  Returns the coordinate
         * at the end of the Identify chain.
         *
         * If there aren't any outgoing Identify edges, this returns
         * the original coordinate tag.
         */
        int followIdentify(int coordinateTag, KernelGraph const& graph);

        /**
         * @brief Create a new coordinate representing data within the
         * scratch space. This will return a coordinate that can be
         * added to a coordinate graph. It also allocates the required
         * scratch space within the context.
         *
         * @param size
         * @param varType
         * @param policy The scratch policy to use for allocation
         * @param context
         * @return User
         */
        rocRoller::KernelGraph::CoordinateGraph::User
            newScratchCoordinate(Expression::ExpressionPtr size,
                                 VariableType              varType,
                                 Operations::ScratchPolicy policy,
                                 ContextPtr                context);

        /**
         * @brief Replace operation with a new operation.
         *
         * @param op Operation to replace.
         * @param newOp Replacement.
         * @param includeBody If true, transfer Body edges.
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
         * @brief Find load/store operations that need their indexes precomputed.
         */
        std::vector<int> findIndexAssignmentCandidates(KernelGraph const& kgraph, int start);

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
                                           int  numDwordsPerElement,
                                           bool avoidDWordX2);

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
                                bool                             isGlobalToLDS = false,
                                bool                             ldsSwizzle    = false);

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
                                  bool                               rightmostFastest,
                                  bool                               isGlobalToLDS = false);

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
         *
         * @param updatePreTiledSubDimStrides When `sdim` has four elements (pre-tiled
         * load path), whether to rewrite outer SubDimension strides/sizes before adding
         * CTs. Default true; callers that duplicate an existing pre-tiled graph (e.g.
         * swizzle scale) should pass false.
         *
         * Note that when pretiling is enabled:
         *
         * - The client/user creates a 2D tensor (with the usual strides etc).
         * - LowerFromCommand creates 4 SubDimension nodes; but does
         *   not update the strides of the first two dimensions.
         * - When `updatePreTiledSubDimStrides` is true, the first two
         *   SubDimensions are update and made consistent with a 4D
         *   tensor.
         */
        std::tuple<int, int, int, int>
            addLoadMacroTileCT(KernelGraph&                     graph,
                               std::vector<DeferredConnection>& connections,
                               int                              macTileTag,
                               std::vector<int> const&          sdim,
                               bool                             updatePreTiledSubDimStrides = true);

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
         * When `rightmostFastest` is true, both of these orders are
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
                                 bool                               rightmostFastest,
                                 bool                               isGlobalToLDS     = false,
                                 bool                               ldsSwizzle        = false,
                                 unsigned int                       columnsPerBankRow = 0u);

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
                                                     bool isStorePartOfGlobalToLDSOp = false);

        /**
         * @brief Get the number of LDS elements for a given LDS tag.
         */
        int GetNumLDSElements(KernelGraph const& graph, int ldsTag);

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

        /**
         * Yields all nodes that contain `control` via a non-Sequence edge, in order from
         * the immediate parent up to the root of the graph, paired with the containing edge
         * connecting the parent to the child (e.g. Body, Else, Initialize, ForLoopIncrement).
         */
        Generator<std::pair<int, ControlGraph::ControlEdge>>
            containingAncestors(int control, KernelGraph const& graph);

        /**
         * Yields all nodes that contain `control` via a non-Sequence edge, in order from
         * the immediate parent up to the root of the graph, paired with the containing edge
         * connecting the parent to the child (e.g. Body, Else, Initialize, ForLoopIncrement).
         */
        Generator<std::pair<int, ControlGraph::ControlEdge>>
            containingAncestors(int control, ControlGraph::ControlGraph const& graph);

        /**
         * Returns all nodes that contain `control` via a non-Sequence edge, in order
         * from the root of the graph down to `control` itself.
         */
        std::deque<int> controlStack(int control, KernelGraph const& graph);

        /**
         * Returns all nodes that contain `control` via a non-Sequence edge, in order
         * from the root of the graph down to `control` itself.
         */
        std::deque<int> controlStack(int control, ControlGraph::ControlGraph const& graph);

        /**
         * @brief Connect all nodes in A with all nodes in B using edge with EdgeType
         */
        template <typename EdgeType>
        void connectAllPairs(std::vector<int> const& A, std::vector<int> const& B, KernelGraph& kg);

        /**
         * @brief Given a tag in coordinate graph, this function returns which direction it is
         * dangling. If the node is not dangling, it returns std::nullopt.
         */
        std::optional<Graph::Direction> danglingDirection(KernelGraph const& graph, int tag);

        /**
         * @brief Returns a vector of all tags of operations of type DstOpType
         * that are connected to the same MacroTile as the given srcOpTag of
         * an operation of type SrcOpType.
         *
         * Currently SrcOpType and DstOpType can only exclusively either be
         * LoadTiled or StoreLDSTile
         */
        template <ControlGraph::COperation SrcOpType, ControlGraph::COperation DstOpType>
        requires(
            (std::is_same_v<
                 SrcOpType,
                 ControlGraph::LoadTiled> && std::is_same_v<DstOpType, ControlGraph::StoreLDSTile>)
            || (std::is_same_v<
                    SrcOpType,
                    ControlGraph::
                        StoreLDSTile> && std::is_same_v<DstOpType, ControlGraph::LoadTiled>))
            std::vector<int> getAssociatedOps(KernelGraph const& kgraph, int srcOpTag);

        /**
         * @brief Return true for operations that read from global to store into LDS and false otherwise.
         */
        bool isGlobalToLDSOp(KernelGraph const& graph, int op);

        /**
         * @brief Find the Exchange node ...
         */
        std::optional<int>
            getExchangeForMultiply(KernelGraph const& graph, int multiplyTag, NaryArgument arg);

        /**
         * @brief Return true if the layout of the given element number is swapped.
         *
         * In LowerTile, swapped layout means when X & Y dimensions for a particular tile are swapped
         * to improve the use of long dword instructions. This function determines whether the layout
         * was swapped for a given element number, which affects how tile elements should be accessed.
         */
        bool isSwappedLayout(KernelGraph const&                    graph,
                             int                                   elementNumberTag,
                             CoordinateGraph::ElementNumber const& elementNumber);

        /**
         * @brief Get the size of the VGPRBlockSet dimension for a given tag.
         *
         * @return The VGPRBlockSet size, or std::nullopt if no VGPRBlockSet dimension exists.
         */
        std::optional<uint> GetVGPRBlockSetDimSize(KernelGraph const& graph, int tag);
    }
}

#include <rocRoller/KernelGraph/Utils_impl.hpp>
