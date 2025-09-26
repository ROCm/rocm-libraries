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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using GD = rocRoller::Graph::Direction;

        namespace ConnectWorkgroupsDetail
        {
            std::map<std::pair<int, rocRoller::Graph::Direction>, int>
                connectWorkgroups(KernelGraph& kgraph)
            {
                auto tileNumTags = kgraph.coordinates.getNodes<MacroTileNumber>().to<std::vector>();

                std::map<std::pair<int, rocRoller::Graph::Direction>, int> rv;
                for(auto const& tileNumTag : tileNumTags)
                {
                    if(std::empty(kgraph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                    {
                        // MacroTileNumber is dangling, connect it to a Workgroup
                        auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                        auto workgroupTag
                            = kgraph.coordinates.addElement(Workgroup(tileNum.dim, tileNum.size));
                        Log::debug(
                            "KernelGraph::ConnectWorkgroups: Adding PassThrough from tile {} "
                            "({}) to workgroup {}",
                            tileNumTag,
                            toString(tileNum.size),
                            workgroupTag);
                        kgraph.coordinates.addElement(PassThrough(), {tileNumTag}, {workgroupTag});

                        rv[{tileNum.dim, GD::Upstream}] = workgroupTag;
                    }
                    if(std::empty(kgraph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
                    {
                        // MacroTileNumber is dangling, connect it to a Workgroup
                        auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                        auto workgroupTag
                            = kgraph.coordinates.addElement(Workgroup(tileNum.dim, tileNum.size));
                        Log::debug("KernelGraph::ConnectWorkgroups: Adding PassThrough from "
                                   "workgroup {} to tile {} ({})",
                                   workgroupTag,
                                   tileNumTag,
                                   toString(tileNum.size));
                        kgraph.coordinates.addElement(PassThrough(), {workgroupTag}, {tileNumTag});

                        rv[{tileNum.dim, GD::Downstream}] = workgroupTag;
                    }
                }

                return rv;
            }
        }

        KernelGraph ConnectWorkgroups::apply(KernelGraph const& original)
        {
            using namespace ConnectWorkgroupsDetail;

            auto kgraph = original;
            connectWorkgroups(kgraph);

            return kgraph;
        }
    }
}
