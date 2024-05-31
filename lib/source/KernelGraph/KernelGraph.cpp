#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

#include <rocRoller/Utilities/Settings.hpp>

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
                auto check = constraints[i](*this);
                if(!check.satisfied)
                {
                    Log::warn("Constraint failed: {}", check.explanation);
                }
                retval.combine(check);
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

        std::vector<GraphConstraint> KernelGraph::getConstraints() const
        {
            return m_constraints;
        }

        KernelGraph KernelGraph::transform(std::shared_ptr<GraphTransform> const& transformation)
        {
            auto transformString  = concatenate("KernelGraph::transform ", transformation->name());
            auto checkConstraints = Settings::getInstance()->get(Settings::EnforceGraphConstraints);

            auto logger = rocRoller::Log::getLogger();
            logger->debug(transformString);

            if(checkConstraints)
            {
                auto check = (*this).checkConstraints(transformation->preConstraints());
                AssertFatal(check.satisfied,
                            concatenate(transformString, " PreCheck: \n", check.explanation));
            }

            KernelGraph newGraph = transformation->apply(*this);

            if(Settings::getInstance()->get(Settings::LogGraphs))
                logger->debug("KernelGraph::transform: {}, post: {}",
                              transformation->name(),
                              newGraph.toDOT(false, transformString));

            if(checkConstraints)
            {
                newGraph.addConstraints(transformation->postConstraints());
                auto check = newGraph.checkConstraints();
                AssertFatal(check.satisfied,
                            concatenate(transformString, " PostCheck: \n", check.explanation));
            }

            return newGraph;
        }
    }
}
