#include <rocRoller/KernelGraph/KernelHypergraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::string KernelHypergraph::toDOT(bool drawMappings) const
        {
            std::stringstream ss;
            ss << "digraph {\n";
            ss << coordinates.toDOT("coord", false);
            ss << "subgraph clusterCF {";
            ss << control.toDOT("cntrl", false);
            ss << "}" << std::endl;
            if(drawMappings)
            {
                ss << mapper.toDOT("coord", "cntrl");
            }
            ss << "}" << std::endl;
            return ss.str();
        }
    }
}
