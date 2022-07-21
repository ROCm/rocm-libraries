
#pragma once

#include <vector>

#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/Edge.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateTransform
    {
        template <typename T>
        concept CTEdgePassthrough = std::is_same<Edge, T>::value || std::is_same<DataFlowEdge, T>::
            value || std::is_same<CoordinateTransformEdge, T>::value;

        template <typename T>
        concept CTUndefinedEdge = std::is_same<ConstructMacroTile, T>::value || std::
            is_same<DestructMacroTile, T>::value || std::is_same<Forget, T>::value;

        struct BaseEdgeVisitor
        {
            std::vector<Expression::ExpressionPtr> indexes;
            std::vector<Dimension>                 srcs;
            std::vector<Dimension>                 dsts;

            inline void setLocation(std::vector<Expression::ExpressionPtr> _indexes,
                                    std::vector<Dimension>                 _srcs,
                                    std::vector<Dimension>                 _dsts)
            {
                indexes = _indexes;
                srcs    = _srcs;
                dsts    = _dsts;
            }
        };
    }
}
