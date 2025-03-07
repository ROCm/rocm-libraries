/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
        struct Duplicate;
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

        using DataFlowEdge = std::variant<DataFlow, Duplicate, Buffer, Offset, Stride, View>;

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
