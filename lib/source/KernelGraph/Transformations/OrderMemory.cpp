#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMemory.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CG = rocRoller::KernelGraph::ControlGraph;

        std::set<std::pair<int, int>> allAmbiguousNodes(KernelGraph const& graph)
        {
            std::set<std::pair<int, int>> ambiguousNodes;
            for(auto pair : graph.control.ambiguousNodes<CG::LoadTiled>())
                ambiguousNodes.insert(pair);
            for(auto pair : graph.control.ambiguousNodes<CG::StoreTiled>())
                ambiguousNodes.insert(pair);
            for(auto pair : graph.control.ambiguousNodes<CG::LoadLDSTile>())
                ambiguousNodes.insert(pair);
            for(auto pair : graph.control.ambiguousNodes<CG::StoreLDSTile>())
                ambiguousNodes.insert(pair);
            for(auto pair : graph.control.ambiguousNodes<CG::LoadLinear>())
                ambiguousNodes.insert(pair);
            for(auto pair : graph.control.ambiguousNodes<CG::StoreLinear>())
                ambiguousNodes.insert(pair);
            /* TODO: Handle these node types.
            for(auto pair :
                graph.control.ambiguousNodes<CG::LoadVGPR>())
                ambiguousNodes.insert(pair);
            for(auto pair :
                graph.control.ambiguousNodes<CG::StoreVGPR>())
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
                               concatenate("Ambiguous memory nodes found: (",
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
                retval.combine(true, concatenate("Handy regex search string:\n", searchPattern));
            }
            return retval;
        }

        ConstraintStatus NoBadBodyEdges(const KernelGraph& graph)
        {
            ConstraintStatus retval;
            try
            {
                ControlFlowRWTracer tracer(graph);
            }
            catch(RecoverableError&)
            {
                retval.combine(false, "Invalid control graph!");
            }
            return retval;
        }

        /**
         * This transformation is used to provide the initial ordering
         * of memory nodes in the first stage of the kernel graph.
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
