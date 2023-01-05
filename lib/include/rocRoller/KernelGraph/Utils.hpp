
#include <rocRoller/Expression.hpp>

#include "KernelGraph/KernelGraph.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * Create a range-based for loop.
         */
        std::pair<int, int> rangeFor(KernelGraph& graph, Expression::ExpressionPtr size);

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
                                 std::array<unsigned int, 3> const& workgroupSizes);

        void loadWaveMacroTileFromLDS(KernelGraph&                      graph,
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

        void storeMacroTileIntoLDS(KernelGraph&                       graph,
                                   int                                store_tag,
                                   int                                lds_tag,
                                   int                                mac_tile_tag,
                                   std::array<unsigned int, 3> const& workgroupSizes);

        void loadMacroTileFromLDS(KernelGraph&                       graph,
                                  int                                load_tag,
                                  int                                lds_tag,
                                  int                                mac_tile_tag,
                                  std::array<unsigned int, 3> const& workgroupSizes);

        void addConnectionsMultiply(KernelGraph& graph, int waveMult);
    }
}
