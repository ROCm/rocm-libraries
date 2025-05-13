/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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

            if(checkConstraints)
            {
                auto check = (*this).checkConstraints(transformation->preConstraints());
                AssertFatal(check.satisfied,
                            concatenate(transformString, " PreCheck: \n", check.explanation));
            }

            KernelGraph newGraph = transformation->apply(*this);

            bool drawMappings = Settings::getInstance()->get(Settings::LogGraphMapperConnections);

            if(Settings::getInstance()->get(Settings::LogGraphs))
                Log::debug("KernelGraph::transform: {}, post: {}",
                           transformation->name(),
                           newGraph.toDOT(drawMappings, transformString));

            if(checkConstraints)
            {
                newGraph.addConstraints(transformation->postConstraints());
                auto check = newGraph.checkConstraints();
                AssertFatal(check.satisfied,
                            concatenate(transformString, " PostCheck: \n", check.explanation));
            }

            newGraph.m_transforms.push_back(transformation->name());

            return newGraph;
        }

        std::vector<std::string> const& KernelGraph::appliedTransforms() const
        {
            return m_transforms;
        }

        void KernelGraph::addAppliedTransforms(std::vector<std::string> const& transforms)
        {
            m_transforms.insert(m_transforms.end(), transforms.begin(), transforms.end());
        }
    }
}
