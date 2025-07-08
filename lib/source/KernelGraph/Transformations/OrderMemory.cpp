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
            for(auto pair : graph.control.ambiguousNodes<CG::LoadTileDirect2LDS>())
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
                retval.combine(false, "OrderMemory:Invalid control graph!");
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
