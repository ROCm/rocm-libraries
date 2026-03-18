// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/TopoVisitor.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        ConstraintStatus NoDanglingMappings(const KernelGraph& k)
        {
            TIMER(t, "Constraint::NoDanglingMappings");
            ConstraintStatus retval;
            for(auto control : k.mapper.getControls())
            {
                if(!k.control.exists(control))
                {
                    retval.combine(false,
                                   concatenate("Dangling Mapping: Control node ",
                                               control,
                                               " does not exist."));
                }
                for(auto& connection : k.mapper.getConnections(control))
                {
                    if(!k.coordinates.exists(connection.coordinate))
                    {
                        retval.combine(false,
                                       concatenate("Dangling Mapping: Control node ",
                                                   control,
                                                   " maps to coordinate node ",
                                                   connection.coordinate,
                                                   ", which doesn't exist."));
                    }
                }
            }
            return retval;
        }

        ConstraintStatus SingleControlRoot(const KernelGraph& k)
        {
            TIMER(t, "Constraint::SingleControlRoot");
            ConstraintStatus retval;

            auto controlRoots = k.control.roots().to<std::vector>();

            if(controlRoots.size() != 1)
            {
                std::ostringstream msg;
                msg << "Single Control Root: Control graph must have exactly one root node, not "
                    << controlRoots.size() << ". Nodes: (";
                streamJoin(msg, controlRoots, ", ");
                msg << ")";

                retval.combine(false, msg.str());
            }

            return retval;
        }

        ConstraintStatus NoRedundantSetCoordinates(const KernelGraph& k)
        {
            TIMER(t, "Constraint::NoRedundantSetCoordinates");
            using namespace ControlGraph;
            using GD = rocRoller::Graph::Direction;
            ConstraintStatus retval;

            for(const auto& op : k.control.leaves())
            {
                std::set<std::pair<int, int>> existingSetCoordData;
                int                           tag = op;

                while(true)
                {
                    auto parent = only(k.control.getInputNodeIndices<Body>(tag));
                    if(!parent)
                        break;

                    tag           = parent.value();
                    auto setCoord = k.control.get<SetCoordinate>(tag);
                    if(!setCoord)
                        break;

                    auto valueExpr = setCoord.value().value;
                    AssertFatal(evaluationTimes(valueExpr)[Expression::EvaluationTime::Translate],
                                "SetCoordinate::value should be a literal.");

                    auto value = getUnsignedInt(evaluate(valueExpr));
                    for(auto const& dst : k.mapper.getConnections(tag))
                    {
                        auto insertResult = existingSetCoordData.insert({dst.coordinate, value});
                        if(!insertResult.second)
                        {
                            auto setCoordData = insertResult.first;
                            retval.combine(false,
                                           concatenate("Redundant SetCoordinate for node ",
                                                       op,
                                                       ": SetCoordinate ",
                                                       tag,
                                                       " with target coordinate ",
                                                       setCoordData->first,
                                                       " and value ",
                                                       setCoordData->second));
                        }
                    }
                }
            }

            return retval;
        }

        struct WalkableControlGraphVisitor
            : public TopoControlGraphVisitor<WalkableControlGraphVisitor>
        {
            using TopoControlGraphVisitor<WalkableControlGraphVisitor>::TopoControlGraphVisitor;

            ConstraintStatus status;
            std::set<int>    visitedNodes;

            void operator()(int nodeIdx, auto const& node)
            {
                visitedNodes.insert(nodeIdx);
            }

            virtual void errorCondition(std::string const& message) override
            {
                status.combine(false, message);
            }
        };

        ConstraintStatus WalkableControlGraph(KernelGraph const& k)
        {
            TIMER(t, "Constraint::WalkableControlGraph");
            WalkableControlGraphVisitor visitor(k);
            visitor.walk();

            auto allNodes = k.control.getNodes().to<std::set>();

            if(visitor.visitedNodes != allNodes)
            {
                std::set<int> nonVisitedNodes;
                std::set_difference(allNodes.begin(),
                                    allNodes.end(),
                                    visitor.visitedNodes.begin(),
                                    visitor.visitedNodes.end(),
                                    std::inserter(nonVisitedNodes, nonVisitedNodes.end()));

                std::ostringstream msg;
                msg << "Not all nodes were visited! Missing: ";
                streamJoin(msg, nonVisitedNodes, ", ");
                msg << "\n All nodes: ";
                streamJoin(msg, allNodes, ", ");
                msg << "\n Visited nodes: ";
                streamJoin(msg, visitor.visitedNodes, ", ");

                visitor.status.combine(false, msg.str());
            }

            return visitor.status;
        }
    }
}
