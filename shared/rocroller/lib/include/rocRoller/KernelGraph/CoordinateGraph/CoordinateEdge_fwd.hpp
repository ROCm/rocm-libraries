// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
        struct PairSwap;
        struct Rotate;
        struct MakeOutput;
        struct PassThrough;
        struct PiecewiseAffineJoin;
        struct Split;
        struct Sunder;
        struct Tile;

        using CoordinateTransformEdge = std::variant<ConstructMacroTile,
                                                     DestructMacroTile,
                                                     Flatten,
                                                     Forget,
                                                     Inherit,
                                                     Join,
                                                     PairSwap,
                                                     Rotate,
                                                     MakeOutput,
                                                     PassThrough,
                                                     PiecewiseAffineJoin,
                                                     Split,
                                                     Sunder,
                                                     Tile>;

        template <typename T>
        concept CCoordinateTransformEdge = std::constructible_from<CoordinateTransformEdge, T>;

        template <typename T>
        concept CConcreteCoordinateTransformEdge
            = (CCoordinateTransformEdge<T> && !std::same_as<CoordinateTransformEdge, T>);

        struct DataFlow;

        struct Alias;
        struct Buffer;
        struct BaseAddress;
        struct Duplicate;
        struct Identify;
        struct Index;
        struct Offset;
        struct Segment;
        struct Stride;
        struct View;
        struct TDM;

        using DataFlowEdge = std::variant<DataFlow,
                                          Alias,
                                          Buffer,
                                          BaseAddress,
                                          Duplicate,
                                          Identify,
                                          Index,
                                          Offset,
                                          Segment,
                                          Stride,
                                          View,
                                          TDM>;

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
