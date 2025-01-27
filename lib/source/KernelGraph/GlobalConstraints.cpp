
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        ConstraintStatus NoDanglingMappings(const KernelGraph& k)
        {
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
    }
}
