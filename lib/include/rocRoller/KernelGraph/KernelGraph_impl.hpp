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
