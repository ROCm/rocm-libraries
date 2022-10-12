#include <rocRoller/KernelGraph/KernelHypergraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::string KernelHypergraph::toDOT() const
        {
            std::stringstream ss;
            ss << "digraph {\n";
            ss << coordinates.toDOT("coord", false);
            ss << "subgraph clusterCF {";
            ss << control.toDOT("cntrl", false);
            ss << "} }" << std::endl;
            return ss.str();
        }
    }
}
