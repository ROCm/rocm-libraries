#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /*
         * Nodes (Dimensions)
         */

        struct ForLoop;
        struct Adhoc;
        struct ElementNumber;
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
        struct VGPRBlockNumber;
        struct VGPRBlockIndex;
        struct WaveTile;
        struct WaveTileIndex;
        struct WaveTileNumber;
        struct JammedWaveTileNumber;
        struct Wavefront;
        struct Workgroup;
        struct Workitem;

        using Dimension = std::variant<ForLoop,
                                       Adhoc,
                                       ElementNumber,
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
                                       VGPRBlockNumber,
                                       VGPRBlockIndex,
                                       WaveTile,
                                       WaveTileIndex,
                                       WaveTileNumber,
                                       JammedWaveTileNumber,
                                       Wavefront,
                                       Workgroup,
                                       Workitem>;

        template <typename T>
        concept CDimension = std::constructible_from<Dimension, T>;

        template <typename T>
        concept CConcreteDimension = (CDimension<T> && !std::same_as<Dimension, T>);
    }
}
