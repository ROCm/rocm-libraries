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
        struct Sunder;
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
                                                     Sunder,
                                                     Tile>;

        template <typename T>
        concept CCoordinateTransformEdge = std::constructible_from<CoordinateTransformEdge, T>;

        template <typename T>
        concept CConcreteCoordinateTransformEdge
            = (CCoordinateTransformEdge<T> && !std::same_as<CoordinateTransformEdge, T>);

        struct DataFlow;
        struct Buffer;
        struct Offset;
        struct Stride;
        struct View;

        using DataFlowEdge = std::variant<DataFlow, Buffer, Offset, Stride, View>;

        template <typename T>
        concept CDataFlowEdge = std::constructible_from<DataFlowEdge, T>;

        template <typename T>
        concept CConcreteDataFlowEdge = (CDataFlowEdge<T> && !std::same_as<DataFlowEdge, T>);

        using Edge = std::variant<CoordinateTransformEdge, DataFlowEdge>;

        template <typename T>
        concept CEdge = std::constructible_from<Edge, T>;

        template <typename T>
        concept CConcreteEdge = (CEdge<T> && !std::same_as<Edge, T>);
    }
}
