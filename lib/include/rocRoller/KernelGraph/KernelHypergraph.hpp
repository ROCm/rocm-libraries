#pragma once

#include "ControlHypergraph/ControlHypergraph.hpp"
#include "CoordGraph/CoordinateHypergraph.hpp"

#include "ControlToCoordinateMapper.hpp"
#include "Utilities/Error.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /*
         * Kernel graph containers
         */

        class KernelHypergraph
        {
        public:
            ControlHypergraph::ControlHypergraph control;
            CoordGraph::CoordinateHypergraph     coordinates;
            ControlToCoordinateMapper            mapper;

            std::string toDOT(bool drawMappings = false) const;

            template <typename T>
            std::pair<int, T> getDimension(int controlIndex, int subDimension = 0) const
            {
                int  tag     = mapper.get<T>(controlIndex, subDimension);
                auto element = coordinates.getElement(tag);
                AssertFatal(std::holds_alternative<CoordGraph::Dimension>(element),
                            "Invalid connection: element isn't a Dimension.",
                            ShowValue(controlIndex));
                auto dim = std::get<CoordGraph::Dimension>(element);
                AssertFatal(std::holds_alternative<T>(dim),
                            "Invalid connection: Dimension type mismatch.",
                            ShowValue(controlIndex),
                            ShowValue(typeid(T).name()),
                            ShowValue(dim));
                return {tag, std::get<T>(dim)};
            }
        };
    }
}
