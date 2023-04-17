
#pragma once

#include <optional>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        namespace CF = rocRoller::KernelGraph::ControlGraph;

        /**
         * @brief Create a range-based for loop.
         */
        std::pair<int, int>
            rangeFor(KernelGraph& graph, Expression::ExpressionPtr size, const std::string& name);

        /**
         * @brief Remove a range-based for loop created by rangeFor.
         */
        void purgeFor(KernelGraph& graph, int tag);

        /**
         * @brief Remove a node and all of its children from the control graph
         *
         * Also purges the mapper of references to the deleted nodes.
         *
         * @param kgraph
         * @param node
         */
        void purgeNodeAndChildren(KernelGraph& kgraph, int node);

        bool isHardwareCoordinate(int tag, KernelGraph const& kgraph);
        bool isLoopishCoordinate(int tag, KernelGraph const& kgraph);
        bool isStorageCoordinate(int tag, KernelGraph const& kgraph);

        /**
         * @brief Filter coordinates by type.
         */
        template <typename T>
        std::unordered_set<int> filterCoordinates(std::vector<int>   candidates,
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
         * @brief Return target coordinate for load/store operation.
         *
         * For loads, the target is the source (User or LDS) of the
         * load.
         *
         * For stores, the target is the destination (User or LDS) of
         * the store.
         */
        std::pair<int, Graph::Direction> getOperationTarget(int tag, KernelGraph const& kgraph);

        /**
         * @brief Find all required coordintes needed to compute
         * indexes for the target dimension.
         *
         * @return Pair of: vector required coordinates; set of
         * coordinates in the connecting path.
         */
        std::pair<std::vector<int>, std::unordered_set<int>> findRequiredCoordinates(
            int target, Graph::Direction direction, KernelGraph const& kgraph);

        /**
         * @brief Find the operation of type T that contains the
         * candidate load/store operation.
         */
        template <typename T>
        std::optional<int> findContainingOperation(int candidate, KernelGraph const& kgraph);

        /**
         * Replace operation with a new operation.  Does not delete the original operation.
         */
        int replaceWith(KernelGraph& graph, int op, int newOp, bool includeBody = true);

        /**
         * @brief Insert chain (from top to bottom) above operation.
         *
         * Bottom is attached to op via a Sequence edge.
         */
        void insertBefore(KernelGraph& graph, int op, int top, int bottom);

        /**
         * @brief Find load/store operations that need their indexes
         * precomputed by ComputeIndex.
         */
        std::vector<int> findComputeIndexCandidates(KernelGraph const& kgraph, int start);

        void loadMacroTile(KernelGraph&                       graph,
                           int                                load_tag,
                           int                                user_tag,
                           int                                mac_tile_tag,
                           std::vector<int>&                  sdim,
                           std::array<unsigned int, 3> const& workgroupSizes,
                           int                                wavefrontSize,
                           std::vector<unsigned int> const&   wavetilesPerWorkgroup);

        std::vector<DeferredConnection>
            loadMacroTileForLDS(KernelGraph&                       graph,
                                int                                user_tag,
                                int                                mac_tile_tag,
                                std::vector<int>&                  sdim,
                                int                                K,
                                std::array<unsigned int, 3> const& workgroupSizes,
                                int                                unroll,
                                bool                               useSwappedAccess);

        void updateLoadLDSMacroTile(KernelGraph&         graph,
                                    CT::MacroTile const& mac_tile,
                                    int                  load_tag,
                                    std::vector<int>&    sdims,
                                    int                  K,
                                    int                  lds,
                                    bool                 useSwappedAccess);

        void loadWaveMacroTile(KernelGraph&                     graph,
                               CT::MacroTile const&             mac_tile,
                               int                              load_tag,
                               int                              i_mac_x,
                               int                              i_mac_y,
                               int                              user_tag,
                               int                              wavefrontSize,
                               std::vector<unsigned int> const& wavetilesPerWorkgroup);

        void storeMacroTile(KernelGraph&                       graph,
                            int                                store_tag,
                            int                                user_tag,
                            int                                mac_tile_tag,
                            std::vector<int>&                  sdims,
                            std::array<unsigned int, 3> const& workgroupSizes,
                            int                                wavefrontSize,
                            std::vector<unsigned int> const&   wavetilesPerWorkgroup);

        void storeWaveMacroTile(KernelGraph&                     graph,
                                CT::MacroTile const&             mac_tile,
                                int                              store_tag,
                                int                              i_mac_x,
                                int                              i_mac_y,
                                int                              workitem,
                                int                              user_tag,
                                int                              wavefrontSize,
                                std::vector<unsigned int> const& wavetilesPerWorkgroup);

        std::vector<DeferredConnection>
            storeMacroTileForLDS(KernelGraph&                       graph,
                                 int                                user_tag,
                                 int                                mac_tile_tag,
                                 std::vector<int>&                  sdims,
                                 std::array<unsigned int, 3> const& workgroupSizes);

        void updateStoreLDSMacroTile(KernelGraph&         graph,
                                     CT::MacroTile const& mac_tile,
                                     int                  store_tag,
                                     std::vector<int>&    sdims,
                                     int                  lds);

        std::vector<DeferredConnection>
            storeMacroTileIntoLDS(KernelGraph&                       graph,
                                  int                                lds_tag,
                                  int                                mac_tile_tag,
                                  std::array<unsigned int, 3> const& workgroupSizes,
                                  bool                               useSwappedAccess);

        std::vector<DeferredConnection>
            loadMacroTileFromLDS(KernelGraph&                       graph,
                                 int                                lds_tag,
                                 int                                mac_tile_tag,
                                 std::array<unsigned int, 3> const& workgroupSizes);

        void addConnectionsMultiply(KernelGraph& graph, int waveMult, int loadA, int loadB);

        /**
         * @brief Get ForLoop dimension assciated with ForLoopOp.
         */
        int getForLoop(int forLoopOp, KernelGraph const& kgraph);

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

        /**
         * @brief Return first entry of vector.
         *
         * If vector does not contain a single result, return empty.
         */
        template <typename T>
        inline std::optional<T> only(std::vector<T> v);

        /**
         * @brief Return first result of generator.
         *
         * If generator does not return a single result, return empty.
         */
        inline std::optional<int> only(Generator<int> g);

        int duplicateControlNode(KernelGraph& graph, int tag);

        /**
         * Updates the threadtile size for enabling the use of long dword instructions
         */
        void updateThreadTileForLongDwords(int& t_m,
                                           int& t_n,
                                           int  maxWidth,
                                           int  numDwordsPerElement);

        /**
         * @brief Get the tag of the highest SetCoordinate directly upstream from load.
         *
         * @param graph
         * @param load
         * @return int
         */
        int getTopSetCoordinate(KernelGraph& graph, int load);

        /**
         * @brief Get the unique tags of the highest SetCoordinate nodes directly upstream from each load.
         *
         * @param graph
         * @param loads
         * @return std::set<int>
         */
        std::set<int> getTopSetCoordinates(KernelGraph& graph, std::vector<int> loads);

        /**
         * @brief Get the SetCoordinate object upstream from load that sets the coordinate for the dimension dim.
         *
         * @param graph
         * @param dim
         * @param load
         * @return int
         */
        int getSetCoordinateForDim(KernelGraph const& graph, int dim, int load);

        /**
         * @brief Retrieve all loads from the input vector that have a SetCoordinate which sets the input unrollCoord dimension to unroll.
         *
         * @param graph
         * @param unrollCoord
         * @param loads
         * @param unroll
         * @return std::vector<int>
         */
        std::vector<int> getLoadsForUnroll(KernelGraph&     graph,
                                           int              unrollCoord,
                                           std::vector<int> loads,
                                           int              unroll);

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
        std::vector<int> duplicateControlNodes(KernelGraph&            graph,
                                               GraphReindexer&         reindexer,
                                               std::vector<int> const& startNodes,
                                               Predicate               dontDuplicate);

    }
}

#include "Utils_impl.hpp"
