#pragma once

#include "ControlHypergraph/ControlHypergraph.hpp"
#include "CoordGraph/CoordinateHypergraph.hpp"

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

            std::string toDOT() const;
        };
    }
}
