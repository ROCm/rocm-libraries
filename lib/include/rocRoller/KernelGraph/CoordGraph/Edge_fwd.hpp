#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::CoordGraph
    {
        struct ConstructMacroTile;
        struct DestructMacroTile;
        struct Flatten;
        struct Forget;
        struct Inherit;
        struct Join;
        struct MakeOutput;
        struct PassThrough;
        struct Split;
        struct Tile;

        using CoordinateTransformEdge = std::variant<ConstructMacroTile,
                                                     DestructMacroTile,
                                                     Flatten,
                                                     Forget,
                                                     Inherit,
                                                     Join,
                                                     MakeOutput,
                                                     PassThrough,
                                                     Split,
                                                     Tile>;

        struct DataFlow;

        using DataFlowEdge = std::variant<DataFlow>;

        using Edge = std::variant<CoordinateTransformEdge, DataFlowEdge>;
    }
}
