#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMemory.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        ConstraintStatus NoAmbiguousNodes(const KernelGraph& k)
        {
            ConstraintStatus retval;
            std::set<int>    searchNodes;

            std::set<std::pair<int, int>> ambiguousNodes;
            for(auto pair : k.control.ambiguousNodes<ControlGraph::LoadTiled>())
                ambiguousNodes.insert(pair);
            for(auto pair : k.control.ambiguousNodes<ControlGraph::StoreTiled>())
                ambiguousNodes.insert(pair);
            for(auto pair : k.control.ambiguousNodes<ControlGraph::LoadLDSTile>())
                ambiguousNodes.insert(pair);
            for(auto pair : k.control.ambiguousNodes<ControlGraph::StoreLDSTile>())
                ambiguousNodes.insert(pair);
            for(auto pair : k.control.ambiguousNodes<ControlGraph::LoadLinear>())
                ambiguousNodes.insert(pair);
            for(auto pair : k.control.ambiguousNodes<ControlGraph::StoreLinear>())
                ambiguousNodes.insert(pair);
            for(auto pair : k.control.ambiguousNodes<ControlGraph::LoadVGPR>())
                ambiguousNodes.insert(pair);
            for(auto pair : k.control.ambiguousNodes<ControlGraph::StoreVGPR>())
                ambiguousNodes.insert(pair);

            for(auto ambiguousPair : ambiguousNodes)
            {
                retval.combine(false,
                               concatenate("Ambiguous Memory Nodes Found: (",
                                           ambiguousPair.first,
                                           ",",
                                           ambiguousPair.second,
                                           ")"));
                searchNodes.insert(ambiguousPair.first);
                searchNodes.insert(ambiguousPair.second);
            }
            if(!searchNodes.empty())
            {
                std::string searchPattern = "";
                for(auto searchNode : searchNodes)
                {
                    searchPattern += concatenate("|(\\(", searchNode, "\\))");
                }
                searchPattern.erase(0, 1);
                retval.combine(true, concatenate("Regex Search String:\n", searchPattern));
            }
            return retval;
        }

        /**
         * Rewrite HyperGraph to make sure ambiguous memory operations
         * in the control graph are ordered.
         */
        KernelGraph OrderMemory::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::OrderMemory");
            rocRoller::Log::getLogger()->debug("KernelGraph::OrderMemory(TODO)");
            auto newGraph = k;
            //TODO: Implement
            return newGraph;
        }

    }
}
