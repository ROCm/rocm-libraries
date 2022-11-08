#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::CoordGraph
    {
        /*
         * Nodes (Dimensions)
         */

        struct ForLoop;
        struct Adhoc;
        struct Lane;
        struct Linear;
        struct LDS;
        struct MacroTile;
        struct MacroTileIndex;
        struct MacroTileNumber;
        struct SubDimension;
        struct ThreadTile;
        struct ThreadTileIndex;
        struct ThreadTileNumber;
        struct Unroll;
        struct User;
        struct VGPR;
        struct WaveTile;
        struct WaveTileIndex;
        struct WaveTileNumber;
        struct JammedWaveTileNumber;
        struct Wavefront;
        struct Workgroup;
        struct Workitem;

        using Dimension = std::variant<ForLoop,
                                       Adhoc,
                                       Lane,
                                       LDS,
                                       Linear,
                                       MacroTile,
                                       MacroTileIndex,
                                       MacroTileNumber,
                                       SubDimension,
                                       ThreadTile,
                                       ThreadTileIndex,
                                       ThreadTileNumber,
                                       Unroll,
                                       User,
                                       VGPR,
                                       WaveTile,
                                       WaveTileIndex,
                                       WaveTileNumber,
                                       JammedWaveTileNumber,
                                       Wavefront,
                                       Workgroup,
                                       Workitem>;

        template <typename T>
        concept CDimension = std::constructible_from<Dimension, T>;
    }
}
