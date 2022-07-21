#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::string KernelGraph::toDOT() const
        {
            std::stringstream ss;
            ss << "digraph {\n";
            ss << coordinates.toDOT() << std::endl;
            ss << "subgraph clusterCF {";
            control.toDOT(ss, "krn");
            ss << "} }" << std::endl;
            return ss.str();
        }
    }
}
