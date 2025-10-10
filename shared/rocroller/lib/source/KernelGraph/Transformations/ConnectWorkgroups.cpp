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
                // Some MacroTileNumber do not have `size`. So, gather sizes of
                // dimensions first that can be used for later workgroup creation.
                std::array<Expression::ExpressionPtr, 3> sizes = {nullptr};
                auto tileNumTags = kgraph.coordinates.getNodes<MacroTileNumber>().to<std::vector>();
                for(auto const& tag : tileNumTags)
                {
                    if(not danglingDirection(kgraph, tag).has_value())
                        continue;

                    auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tag);
                    if(sizes[tileNum.dim] == nullptr && tileNum.size != nullptr)
                    {
                        //sizes[tileNum.dim] = tileNum.size;
                        sizes[tileNum.dim] = convert(DataType::Int32, tileNum.size);
                    }
                }

                std::map<std::pair<int, rocRoller::Graph::Direction>, int> rv;

                for(auto const& tileNumTag : tileNumTags)
                {
                    auto const dir = danglingDirection(kgraph, tileNumTag);
                    if(not dir.has_value())
                        continue;

                    if(dir.value() == GD::Downstream)
                    {
                        // MacroTileNumber is dangling, connect it to a Workgroup
                        auto tileNum      = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                        auto workgroupTag = kgraph.coordinates.addElement(
                            Workgroup(tileNum.dim, sizes[tileNum.dim]));
                        Log::info("KernelGraph::ConnectWorkgroups: Adding PassThrough from tile {} "
                                  "({}) to workgroup {},  dim = {}",
                                  tileNumTag,
                                  toString(sizes[tileNum.dim]),
                                  workgroupTag,
                                  tileNum.dim);
                        kgraph.coordinates.addElement(PassThrough(), {tileNumTag}, {workgroupTag});

                        rv[{tileNum.dim, GD::Upstream}] = workgroupTag;
                    }

                    if(dir.value() == GD::Upstream)
                    {
                        // MacroTileNumber is dangling, connect it to a Workgroup
                        auto tileNum      = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                        auto workgroupTag = kgraph.coordinates.addElement(
                            Workgroup(tileNum.dim, sizes[tileNum.dim]));
                        Log::info("KernelGraph::ConnectWorkgroups: Adding PassThrough from "
                                  "workgroup {} to tile {} ({}),  dim = {}",
                                  workgroupTag,
                                  tileNumTag,
                                  toString(sizes[tileNum.dim]),
                                  tileNum.dim);
                        kgraph.coordinates.addElement(PassThrough(), {workgroupTag}, {tileNumTag});

                        rv[{tileNum.dim, GD::Downstream}] = workgroupTag;
                    }
                }

                return rv;
            }

            int remapWorkgroupXCC(rocRoller::KernelGraph::KernelGraph& graph,
                                  int                                  workgroupTag,
                                  uint                                 numXCC)
            {
                auto const direction = danglingDirection(graph, workgroupTag);
                if(not direction.has_value())
                    return -1;

                auto workgroup = graph.coordinates.get<Workgroup>(workgroupTag).value();
                auto size      = workgroup.size;
                // Skip workgroups that do not have size (e.g., these could come from
                // AddStreamK)
                if(size == nullptr)
                {
                    Log::debug("remapWorkgroupXCC: workgroup {} has no size", workgroupTag);
                    return -1;
                }

                Log::info("RemapXCC gets a workgroup = {}, dim = {}, size = {}",
                          workgroupTag,
                          workgroup.dim,
                          toString(workgroup.size));

                using ExpressionPtr     = Expression::ExpressionPtr;
                using ExpressionPtrPair = std::pair<ExpressionPtr, ExpressionPtr>;
                using ExpressionPtrVectorPair
                    = std::pair<std::vector<ExpressionPtr>, std::vector<ExpressionPtr>>;

                auto newWorkgroupTag = graph.coordinates.addElement(Workgroup(0, size));

                auto one           = Expression::literal(1u);
                auto numXCCLiteral = Expression::literal(numXCC);

                auto ceilDiv = [&](ExpressionPtr a, ExpressionPtr b) { return (a + b - one) / b; };

                auto xcc = graph.coordinates.addElement(Linear(numXCCLiteral, nullptr));
                auto cu
                    = graph.coordinates.addElement(Linear(ceilDiv(size, numXCCLiteral), nullptr));

                // 0 argument is XCC, 1 argument is CU
                auto condition
                    = Expression::positionalArgument(0, Register::Type::Scalar, DataType::UInt32)
                      <= (size % numXCCLiteral);

                ExpressionPtrVectorPair strides{{ceilDiv(size, numXCCLiteral), one},
                                                {size / numXCCLiteral, one}};
                ExpressionPtrPair       initialValues{nullptr, size % numXCCLiteral};

                if(direction == GD::Upstream)
                {
                    // cu = CD(size / xcc)
                    // strides = { CD(size / xcc),         1}
                    // initial = {              0,    size % xcc}
                    //
                    // Condition: size % xcc >= PA(0)
                    //    True  => xcc * CD(size / xcc) +  cu
                    //    False => xcc * (size / xcc)   +  cu          + size % xcc
                    graph.coordinates.addElement(Tile(), {newWorkgroupTag}, {cu, xcc});
                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, strides, initialValues),
                        {xcc, cu},
                        {workgroupTag});

                    Log::info("RemapXCC Tile connects new workgroup {} ->  workgroup {}",
                              newWorkgroupTag,
                              workgroupTag);
                }
                else
                {
                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, strides, initialValues),
                        {workgroupTag},
                        {xcc, cu});
                    graph.coordinates.addElement(Flatten(), {cu, xcc}, {newWorkgroupTag});

                    Log::info("RemapXCC Flatten connects workgroup {} ->  new workgroup {}",
                              workgroupTag,
                              newWorkgroupTag);
                }

                return newWorkgroupTag;
            }

            void remapWorkgroupXCC(rocRoller::KernelGraph::KernelGraph& kgraph, uint numXCC)
            {
                connectWorkgroups(kgraph);

                auto const workgroups = kgraph.coordinates.getNodes<Workgroup>().to<std::vector>();
                for(auto tag : workgroups)
                    remapWorkgroupXCC(kgraph, tag, numXCC);
            }
        }

        KernelGraph ConnectWorkgroups::apply(KernelGraph const& original)
        {
            using namespace ConnectWorkgroupsDetail;

            auto kgraph = original;

            if(m_workgroupRemapXCC.has_value())
            {
                Log::info("================RemapXCC=================");
                auto const& arch = m_context->targetArchitecture();
                AssertFatal(arch.HasCapability(GPUCapability::HasXCC),
                            "XCC-aware workgroup remapping not available on: ",
                            arch.target().toString());

                Log::info("XCC value  = {}", m_workgroupRemapXCC.value());
                remapWorkgroupXCC(kgraph, m_workgroupRemapXCC.value());
                Log::info("================RemapXCC Done=================");
            }
            else
            {
                Log::info("================ConnectWorkgroups=================");
                connectWorkgroups(kgraph);
                Log::info("================ConnectWorkgroups Done=================");
            }

            return kgraph;
        }
    }
}
