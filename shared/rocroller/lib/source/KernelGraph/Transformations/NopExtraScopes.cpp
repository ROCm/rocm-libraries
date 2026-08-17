// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iterator>

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/NopExtraScopes.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        using GD = rocRoller::Graph::Direction;

        KernelGraph NopExtraScopes::apply(KernelGraph const& original)
        {
            auto scopes = original.control.getNodes<Scope>().to<std::set>();

            std::unordered_set<int> extraScopes;

            for(int scopeIdx : scopes)
            {
                if(!original.control.getOutputNodeIndices<Sequence>(scopeIdx).empty())
                {
                    Log::debug("Scope {} due to outgoing sequence", scopeIdx);
                    continue;
                }

                auto containingScope = [&]() -> std::optional<int> {
                    auto stack = controlStack(scopeIdx, original);

                    for(auto iter = stack.rbegin(); iter != stack.rend(); ++iter)
                    {
                        if(*iter != scopeIdx && scopes.contains(*iter))
                            return *iter;
                    }

                    return std::nullopt;
                }();

                if(!containingScope)
                {
                    Log::debug("Scope {} due to no containing scope", scopeIdx);
                    continue;
                }

                // siblings: all nodes within the containing scope other than scopeIdx;
                auto siblings
                    = original.control.getOutputNodeIndices<Body>(*containingScope).to<std::set>();
                siblings = original.control.followEdges<Sequence>(siblings);
                siblings.erase(scopeIdx);

                auto allSiblingsAreBeforeScope = [&]() {
                    for(auto const& sibling : siblings)
                    {
                        auto order = original.control.compareNodes(UpdateCache, sibling, scopeIdx);
                        if(order != NodeOrdering::LeftFirst)
                        {
                            Log::debug("Scope {} due to sibling {} ordered {}",
                                       scopeIdx,
                                       sibling,
                                       toString(order));
                            return false;
                        }
                    }

                    return true;
                }();

                if(!allSiblingsAreBeforeScope)
                {
                    Log::debug("Scope {} due to siblings", scopeIdx);
                    continue;
                }

                // if we get this far, this scope is extra.
                extraScopes.insert(scopeIdx);
            }

            auto graph = original;

            for(auto scopeIdx : extraScopes)
            {
                Log::debug("Replacing {} with NOP", scopeIdx);

                graph.control.setElement(scopeIdx, NOP());

                auto nodesByBody
                    = graph.control.getOutputNodeIndices<Body>(scopeIdx).to<std::set>();

                for(auto nodeIdx : nodesByBody)
                {
                    auto edgeIdx = graph.control.findEdge(scopeIdx, nodeIdx);
                    AssertFatal(edgeIdx);

                    graph.control.setElement(*edgeIdx, Sequence());
                }
            }

            return graph;
        }
    }
}
