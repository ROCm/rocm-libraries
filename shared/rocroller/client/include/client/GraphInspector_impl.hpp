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

#include "client/GraphInspector.hpp"

namespace rocRoller
{
    namespace Client
    {
        template <KernelGraph::CoordinateGraph::CDimension DimensionType>
        void GraphInspector::setCoordinate(CommandArgumentValue val)
        {
            auto predicate = isNode<DimensionType>(m_coords);
            setCoordinate(predicate, val);
        }

        template <std::predicate<int> Predicate>
        inline void GraphInspector::setCoordinate(Predicate predicate, CommandArgumentValue val)
        {
            auto exp = Expression::literal(val);
            for(auto const& idx : m_coords->findElements(predicate))
            {
                m_tx.setCoordinate(idx, exp);
            }
        }

        template <rocRoller::CForwardRangeOf<int> Range>
        inline void GraphInspector::setCoordinate(Range const& indices, CommandArgumentValue val)
        {
            auto exp = Expression::literal(val);
            for(auto const& idx : indices)
            {
                m_tx.setCoordinate(idx, exp);
            }
        }

        inline void GraphInspector::setCoordinate(int idx, CommandArgumentValue val)
        {
            auto exp = Expression::literal(val);
            m_tx.setCoordinate(idx, exp);
        }

        inline KernelInvocation const& GraphInspector::kernelInvocation() const
        {
            return m_invocation;
        }

        inline std::map<std::string, CommandArgumentValue> const& GraphInspector::argValues()
        {
            return m_argValues;
        }

        inline std::shared_ptr<KernelGraph::CoordinateGraph::CoordinateGraph>
            GraphInspector::coords()
        {
            return m_coords;
        }

        inline KernelGraph::CoordinateGraph::Transformer& GraphInspector::tx()
        {
            return m_tx;
        }
    }
}
