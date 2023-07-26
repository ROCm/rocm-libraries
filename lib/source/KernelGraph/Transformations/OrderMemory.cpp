#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMemory.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        std::set<std::pair<int, int>> allAmbiguousNodes(KernelGraph const& graph)
        {
            std::set<std::pair<int, int>> ambiguousNodes;
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::LoadTiled>())
                ambiguousNodes.insert(pair);
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::StoreTiled>())
                ambiguousNodes.insert(pair);
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::LoadLDSTile>())
                ambiguousNodes.insert(pair);
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::StoreLDSTile>())
                ambiguousNodes.insert(pair);
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::LoadLinear>())
                ambiguousNodes.insert(pair);
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::StoreLinear>())
                ambiguousNodes.insert(pair);
            /* TODO: Handle these node types.
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::LoadVGPR>())
                ambiguousNodes.insert(pair);
            for(auto pair :
                graph.control.ambiguousNodes<rocRoller::KernelGraph::ControlGraph::StoreVGPR>())
                ambiguousNodes.insert(pair);
            */
            return ambiguousNodes;
        }

        ConstraintStatus NoAmbiguousNodes(const KernelGraph& k)
        {
            ConstraintStatus retval;
            std::set<int>    searchNodes;
            auto             ambiguousNodes = allAmbiguousNodes(k);
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
         * This transformation is used to provide the initial ordering of memory nodes in the first stage of the kernel graph.
         */
        KernelGraph OrderMemory::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::OrderMemory");
            rocRoller::Log::getLogger()->debug("KernelGraph::OrderMemory");
            auto newGraph = k;
            orderMemoryNodes(newGraph, allAmbiguousNodes(k), false);
            return newGraph;
        }

    }
}
