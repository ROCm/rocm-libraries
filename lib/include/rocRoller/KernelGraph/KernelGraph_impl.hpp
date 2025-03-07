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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/RTTI.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        template <typename T>
        std::pair<int, T> KernelGraph::getDimension(int                         controlIndex,
                                                    Connections::ConnectionSpec conn) const
        {
            int  tag     = mapper.get(controlIndex, conn);
            auto element = coordinates.getElement(tag);
            AssertFatal(std::holds_alternative<CoordinateGraph::Dimension>(element),
                        "Invalid connection: element isn't a Dimension.",
                        ShowValue(controlIndex));
            auto dim = std::get<CoordinateGraph::Dimension>(element);
            AssertFatal(std::holds_alternative<T>(dim),
                        "Invalid connection: Dimension type mismatch.",
                        ShowValue(controlIndex),
                        ShowValue(typeName<T>()),
                        ShowValue(name<T>()),
                        ShowValue(dim));
            return {tag, std::get<T>(dim)};
        }

        template <typename T>
        std::pair<int, T> KernelGraph::getDimension(int controlIndex, NaryArgument arg) const
        {
            return getDimension<T>(controlIndex, Connections::JustNaryArgument{arg});
        }

        template <typename T>
        std::pair<int, T> KernelGraph::getDimension(int controlIndex, int subDimension) const
        {
            return getDimension<T>(controlIndex,
                                   Connections::TypeAndSubDimension{name<T>(), subDimension});
        }

    }
}
