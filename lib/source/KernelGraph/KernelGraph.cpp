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

        ConstraintStatus
            KernelGraph::checkConstraints(const std::vector<GraphConstraint>& constraints) const
        {
            ConstraintStatus retval;
            for(int i = 0; i < constraints.size(); i++)
            {
                retval.combine(constraints[i](*this));
            }
            return retval;
        }

        ConstraintStatus KernelGraph::checkConstraints() const
        {
            return checkConstraints(m_constraints);
        }

        void KernelGraph::addConstraints(const std::vector<GraphConstraint>& constraints)
        {
            m_constraints.insert(m_constraints.end(), constraints.begin(), constraints.end());
        }
    }
}
