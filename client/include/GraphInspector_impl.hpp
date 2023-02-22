#pragma once

#include "GraphInspector.hpp"

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
