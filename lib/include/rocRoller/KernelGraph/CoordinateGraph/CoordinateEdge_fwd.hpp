#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
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
        struct Buffer;
        struct Offset;
        struct Stride;

        using DataFlowEdge = std::variant<DataFlow, Buffer, Offset, Stride>;

        using Edge = std::variant<CoordinateTransformEdge, DataFlowEdge>;
    }
}
