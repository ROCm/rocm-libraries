
#include <rocRoller/KernelGraph/KernelGraph.hpp>

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
    }
}
