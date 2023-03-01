
#pragma once

#include <optional>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Create a range-based for loop.
         */
        std::pair<int, int>
            rangeFor(KernelGraph& graph, Expression::ExpressionPtr size, const std::string& name);

        /**
         * @brief Remove a range-based for loop created by rangeFor.
         */
        void purgeFor(KernelGraph& graph, int tag);

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
         */
        std::vector<int> findRequiredCoordinates(int                target,
                                                 Graph::Direction   direction,
                                                 KernelGraph const& kgraph);

        /**
         * @brief Find the operation of type T that contains the
         * candidate load/store operation.
         */
        template <typename T>
        std::optional<int> findContainingOperation(int candidate, KernelGraph const& kgraph);

        /**
         * Replace operation with a scope.  Does not delete the original operation.
         */
        int replaceWithScope(KernelGraph& graph, int op, bool includeBody = true);

        void loadMacroTile(KernelGraph&                       graph,
                           int                                load_tag,
                           int                                user_tag,
                           int                                mac_tile_tag,
                           std::vector<int>&                  sdim,
                           std::array<unsigned int, 3> const& workgroupSizes,
                           int                                wavefrontSize,
                           std::vector<unsigned int> const&   wavetilesPerWorkgroup);

        void loadMacroTileForLDS(KernelGraph&                       graph,
                                 int                                load_tag,
                                 int                                user_tag,
                                 int                                mac_tile_tag,
                                 std::vector<int>&                  sdim,
                                 int                                K,
                                 std::array<unsigned int, 3> const& workgroupSizes,
                                 std::shared_ptr<Context>           context);

        void updateLoadLDSMacroTile(KernelGraph&                      graph,
                                    CoordinateGraph::MacroTile const& mac_tile,
                                    int                               load_tag,
                                    std::vector<int>&                 sdims,
                                    int                               K,
                                    int                               lds);

        void loadWaveMacroTile(KernelGraph&                      graph,
                               CoordinateGraph::MacroTile const& mac_tile,
                               int                               load_tag,
                               int                               i_mac_x,
                               int                               i_mac_y,
                               int                               user_tag,
                               int                               wavefrontSize,
                               std::vector<unsigned int> const&  wavetilesPerWorkgroup);

        void storeMacroTile(KernelGraph&                       graph,
                            int                                store_tag,
                            int                                user_tag,
                            int                                mac_tile_tag,
                            std::vector<int>&                  sdims,
                            std::array<unsigned int, 3> const& workgroupSizes,
                            int                                wavefrontSize,
                            std::vector<unsigned int> const&   wavetilesPerWorkgroup);

        void storeWaveMacroTile(KernelGraph&                      graph,
                                CoordinateGraph::MacroTile const& mac_tile,
                                int                               store_tag,
                                int                               i_mac_x,
                                int                               i_mac_y,
                                int                               workitem,
                                int                               user_tag,
                                int                               wavefrontSize,
                                std::vector<unsigned int> const&  wavetilesPerWorkgroup);

        void storeMacroTileForLDS(KernelGraph&                       graph,
                                  int                                store_tag,
                                  int                                user_tag,
                                  int                                mac_tile_tag,
                                  std::vector<int>&                  sdims,
                                  std::array<unsigned int, 3> const& workgroupSizes);

        void updateStoreLDSMacroTile(KernelGraph&                      graph,
                                     CoordinateGraph::MacroTile const& mac_tile,
                                     int                               store_tag,
                                     std::vector<int>&                 sdims,
                                     int                               lds);

        void storeMacroTileIntoLDS(KernelGraph&                       graph,
                                   int                                store_tag,
                                   int                                lds_tag,
                                   int                                mac_tile_tag,
                                   std::array<unsigned int, 3> const& workgroupSizes,
                                   std::shared_ptr<Context>           context);

        void loadMacroTileFromLDS(KernelGraph&                       graph,
                                  int                                load_tag,
                                  int                                lds_tag,
                                  int                                mac_tile_tag,
                                  std::array<unsigned int, 3> const& workgroupSizes);

        void addConnectionsMultiply(KernelGraph& graph, int waveMult);

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

    }
}

#include "Utils_impl.hpp"
