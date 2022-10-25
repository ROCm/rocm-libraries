#pragma once

#include "ControlHypergraph/ControlHypergraph.hpp"
#include "CoordGraph/CoordinateHypergraph.hpp"

#include "ControlToCoordinateMapper.hpp"

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
                AssertFatal(std::holds_alternative<CoordGraph::Dimension>(element));
                auto dim = std::get<CoordGraph::Dimension>(element);
                return {tag, std::get<T>(dim)};
            }
        };
    }
}
