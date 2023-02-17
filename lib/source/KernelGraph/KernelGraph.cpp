#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::string KernelGraph::toDOT(bool drawMappings, std::string title) const
        {
            std::stringstream ss;
            ss << "digraph {\n";
            if(!title.empty())
            {
                ss << "labelloc=\"t\";" << std::endl;
                ss << "label=\"" << title << "\";" << std::endl;
            }
            ss << coordinates.toDOT("coord", false);
            ss << "subgraph clusterCF {";
            ss << "label = \"Control Graph\";" << std::endl;
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
