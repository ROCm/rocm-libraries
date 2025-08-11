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
                    if(scopeIdx == 7107)
                    {
                        std::ostringstream msg;
                        msg << "Scope " << scopeIdx << ": ";
                        streamJoin(msg, stack, ", ");
                        Log::debug(msg.str());
                    }

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
