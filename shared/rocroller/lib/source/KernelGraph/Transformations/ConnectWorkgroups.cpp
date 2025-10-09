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
            rocRoller::Graph::Direction danglingDirection(KernelGraph const& kgraph, int const tag)
            {
                bool const upstreamDangling
                    = std::empty(kgraph.coordinates.getNeighbours<GD::Upstream>(tag));
                bool const downstreamDangling
                    = std::empty(kgraph.coordinates.getNeighbours<GD::Downstream>(tag));

                if(upstreamDangling == downstreamDangling)
                    return rocRoller::Graph::Direction::Count;

                return upstreamDangling ? rocRoller::Graph::Direction::Upstream
                                        : rocRoller::Graph::Direction::Downstream;
            }

            std::map<std::pair<int, rocRoller::Graph::Direction>, int>
                connectWorkgroups(KernelGraph& kgraph)
            {
                std::array<Expression::ExpressionPtr, 3> sizes = {nullptr};
                auto tileNumTags = kgraph.coordinates.getNodes<MacroTileNumber>().to<std::vector>();
                for(auto const& tag : tileNumTags)
                {
                    auto const dir = danglingDirection(kgraph, tag);

                    if(dir != GD::Count)
                    {
                        auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tag);
                        if(sizes[tileNum.dim] == nullptr && tileNum.size != nullptr)
                        {
                            Log::info("Dangling dim = {}, size = {} , tag = {}",
                                      tileNum.dim,
                                      toString(tileNum.size),
                                      tag);
                            sizes[tileNum.dim] = convert(DataType::Int32, tileNum.size);
                        }
                    }
                }

                std::map<std::pair<int, rocRoller::Graph::Direction>, int> rv;

                for(auto const& tileNumTag : tileNumTags)
                {
                    auto const dir = danglingDirection(kgraph, tileNumTag);

                    //if(dir == GD::Count)
                    //{
                    //    Log::info("Count !!!!!!!!!!!!!! = {}", tileNumTag)
                    //}

                    //if(std::empty(kgraph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                    if(dir == GD::Downstream)
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

                    //if(std::empty(kgraph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
                    if(dir == GD::Upstream)
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

                using ExpressionPtr     = Expression::ExpressionPtr;
                using ExpressionPtrPair = std::pair<ExpressionPtr, ExpressionPtr>;
                using ExpressionPtrVectorPair
                    = std::pair<std::vector<ExpressionPtr>, std::vector<ExpressionPtr>>;

                auto workgroup = graph.coordinates.get<Workgroup>(workgroupTag).value();
                auto size      = workgroup.size;

                // Need to check if size is nullptr (AddStreamK adds some workgroups).
                // Otherwise, test A123 will fail
                if(workgroup.size == nullptr)
                {
                    Log::info("remapWorkgroupXCC: workgroup {} size is nullptr", workgroupTag);
                    return -1;
                }

                if(workgroup.dim != 0)
                {
                    // Throw an exception. Run tests to see which fail
                    AssertFatal(false, "workgroup dim is not 0 = ", workgroup.dim);
                }

                // Skip the workgroup if it is not dangling
                bool const emptyUpstream
                    = std::empty(graph.coordinates.getNeighbours(workgroupTag, GD::Upstream));
                bool const emptyDownstream
                    = std::empty(graph.coordinates.getNeighbours(workgroupTag, GD::Downstream));
                bool const isDangling = (emptyUpstream != emptyDownstream);
                if(not isDangling)
                {
                    Log::info("remapWorkgroupXCC: workgroup {} is NOT dangling", workgroupTag);
                    return -1;
                }

                Log::info("RemapXCC gets a workgroup = {}", workgroupTag);

                // Upstream: newWorkgroupTag is added above workgroupTag
                auto direction = emptyUpstream ? GD::Upstream : GD::Downstream;

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
                    graph.coordinates.addElement(Tile(), {newWorkgroupTag}, {cu, xcc});
                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, strides, initialValues),
                        {xcc, cu},
                        {workgroupTag});

                    Log::info("RemapXCC connects new workgroup {} ->  workgroup {}",
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

                    Log::info("RemapXCC connects workgroup {} ->  new workgroup {}",
                              workgroupTag,
                              newWorkgroupTag);
                }

                return newWorkgroupTag;
            }

            int remapWorkgroupXCC_3(rocRoller::KernelGraph::KernelGraph& graph,
                                    int                                  workgroupTag,
                                    uint                                 numXCC,
                                    GD const                             direction)
            {
                using ExpressionPtr     = Expression::ExpressionPtr;
                using ExpressionPtrPair = std::pair<ExpressionPtr, ExpressionPtr>;
                using ExpressionPtrVectorPair
                    = std::pair<std::vector<ExpressionPtr>, std::vector<ExpressionPtr>>;

                auto workgroup = graph.coordinates.get<Workgroup>(workgroupTag).value();
                auto size      = workgroup.size;

                Log::info("RemapXCC gets a workgroup = {}, dim = {}, size = {}",
                          workgroupTag,
                          workgroup.dim,
                          toString(workgroup.size));

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
                    AssertFatal(danglingDirection(graph, workgroupTag) == GD::Upstream);

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
                    AssertFatal(danglingDirection(graph, workgroupTag) == GD::Downstream);

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

            void remapWorkgroupXCC_2(rocRoller::KernelGraph::KernelGraph& kgraph, uint numXCC)
            {
                std::vector<int> upstream;
                std::vector<int> downstream;

                std::array<Expression::ExpressionPtr, 3> sizes = {nullptr};
                auto tileNumTags = kgraph.coordinates.getNodes<MacroTileNumber>().to<std::vector>();
                for(auto const& tag : tileNumTags)
                {
                    auto const dir = danglingDirection(kgraph, tag);

                    if(dir != GD::Count)
                    {
                        auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tag);
                        if(sizes[tileNum.dim] == nullptr && tileNum.size != nullptr)
                        {
                            Log::info("Dangling dim = {}, size = {} , tag = {}",
                                      tileNum.dim,
                                      toString(tileNum.size),
                                      tag);
                            sizes[tileNum.dim] = convert(DataType::Int32, tileNum.size);
                        }

                        if(tileNum.size == nullptr)
                            Log::info("MTN tag = {} , dim = {}, size is null", tag, tileNum.dim);
                        else
                            Log::info("MTN tag = {} , dim = {}, size is {}",
                                      tag,
                                      tileNum.dim,
                                      toString(tileNum.size));

                        if(dir == GD::Upstream)
                            upstream.push_back(tag);
                        else
                            downstream.push_back(tag);
                    }
                }

                auto attachWorkgroupToDanglingMTN = [&](std::vector<int>& tags, GD const dir) {
                    for(auto& tag : tags)
                    {
                        auto tileNum      = *kgraph.coordinates.get<MacroTileNumber>(tag);
                        auto workgroupTag = kgraph.coordinates.addElement(
                            Workgroup(tileNum.dim, sizes[tileNum.dim]));
                        if(dir == GD::Upstream)
                        {
                            kgraph.coordinates.addElement(PassThrough(), {workgroupTag}, {tag});
                            Log::info("PassThrough {} -> {}", workgroupTag, tag);
                        }
                        else
                        {
                            kgraph.coordinates.addElement(PassThrough(), {tag}, {workgroupTag});
                            Log::info("PassThrough {} -> {}", tag, workgroupTag);
                        }

                        // Replace the MTN with the workgroup
                        tag = workgroupTag;
                    }
                };

                attachWorkgroupToDanglingMTN(upstream, GD::Upstream);
                attachWorkgroupToDanglingMTN(downstream, GD::Downstream);

                // Upstream dangling MacroTileNumbers do not set size
                //for(auto& tag: upstream)
                //{
                //    auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tag);
                //    auto workgroupTag = kgraph.coordinates.addElement(Workgroup(tileNum.dim, sizes[tileNum.dim]));
                //    kgraph.coordinates.addElement(PassThrough(), {workgroupTag}, {tag});
                //    Log::info("PassThrough {} -> {}", workgroupTag, tag);
                //    tag = workgroupTag;
                //}

                //for(auto& tag: downstream)
                //{
                //    auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tag);
                //    auto workgroupTag = kgraph.coordinates.addElement(Workgroup(tileNum.dim, sizes[tileNum.dim]));
                //    kgraph.coordinates.addElement(PassThrough(), {tag}, {workgroupTag});
                //    Log::info("PassThrough {} -> {}", tag, workgroupTag);
                //    tag = workgroupTag;
                //}
                //exit(0);

                Log::info("Dangling Upstream = {}", upstream);
                Log::info("Dangling Downstream = {}", downstream);

                auto flattenIntoWorkgroup = [&](std::vector<int>& tags, GD const dir) -> int {
                    if(tags.empty())
                        return -1;

                    Expression::ExpressionPtr total = nullptr;

                    for(auto& size : sizes)
                    {
                        if(size != nullptr)
                            total = (total == nullptr) ? size : total * size;
                    }

                    auto workgroupTag = kgraph.coordinates.addElement(Workgroup(0, total));
                    Log::info("RemapXCC -> {} -> flattened workgroup = {}, size = {}",
                              tags,
                              workgroupTag,
                              toString(total));

                    // Flatten dangling workgroups into a workgroup
                    if(dir == GD::Upstream)
                    {
                        Log::info("RemapXCC, upstream workgroup {} to MTNs {}", workgroupTag, tags);
                        if(tags.size() > 1)
                            kgraph.coordinates.addElement(
                                Tile(), std::vector<int>{workgroupTag}, tags);
                        else
                            kgraph.coordinates.addElement(
                                PassThrough(), std::vector<int>{workgroupTag}, tags);
                    }
                    else
                    {
                        Log::info(
                            "RemapXCC, downstream MTNs {} to workgroup {}", tags, workgroupTag);
                        if(tags.size() > 1)
                            kgraph.coordinates.addElement(
                                Flatten(), tags, std::vector<int>{workgroupTag});
                        else
                            kgraph.coordinates.addElement(
                                PassThrough(), tags, std::vector<int>{workgroupTag});
                    }

                    return workgroupTag;
                };

                auto workgroups = kgraph.coordinates.getNodes<Workgroup>().to<std::vector>();
                for(auto wg : workgroups)
                {
                    auto dir = danglingDirection(kgraph, wg);
                    if(dir != GD::Count)
                    {
                        remapWorkgroupXCC_3(kgraph, wg, numXCC, dir);
                    }
                }

                //for(auto tag: upstream)
                //{
                //    remapWorkgroupXCC_3(kgraph, tag, numXCC, GD::Upstream);
                //}
                //for(auto tag: downstream)
                //{
                //    remapWorkgroupXCC_3(kgraph, tag, numXCC, GD::Downstream);
                //}
                return;

                Log::info("Flatten into workgroup upstream => {}", upstream);
                auto workgroupTag = flattenIntoWorkgroup(upstream, GD::Upstream);
                //AssertFatal(workgroupTag > 0);
                if(workgroupTag != -1)
                {
                    remapWorkgroupXCC_3(kgraph, workgroupTag, numXCC, GD::Upstream);
                }

                Log::info("Flatten into workgroup downstream => {}", downstream);
                //std::vector<int> v0 = {50, 111};
                //std::vector<int> v1 = {170,171};
                //std::vector<int> v0 = {435, 437};
                //std::vector<int> v1 = {439, 441};
                //workgroupTag = flattenIntoWorkgroup(v0, GD::Downstream);
                workgroupTag = flattenIntoWorkgroup(downstream, GD::Downstream);
                //AssertFatal(workgroupTag > 0);
                if(workgroupTag != -1)
                {
                    remapWorkgroupXCC_3(kgraph, workgroupTag, numXCC, GD::Downstream);
                }

                //workgroupTag = flattenIntoWorkgroup(v1, GD::Downstream);
                //if(workgroupTag != -1)
                //{
                //    remapWorkgroupXCC_3(kgraph, workgroupTag, numXCC, GD::Downstream);
                //}
            }
        }

        KernelGraph ConnectWorkgroups::apply(KernelGraph const& original)
        {
            using namespace ConnectWorkgroupsDetail;

            auto kgraph = original;

            if(m_workgroupRemapXCC.has_value())
            {
                {
                    std::ofstream ofs("xcc.dot");
                    ofs << kgraph.coordinates.toDOT();
                }

                Log::info("XCC value  = {}", m_workgroupRemapXCC.value());

                Log::info("================RemapXCC=================");
                auto const& arch = m_context->targetArchitecture();
                AssertFatal(arch.HasCapability(GPUCapability::HasXCC),
                            "XCC-aware workgroup remapping not available on: ",
                            arch.target().toString());

                remapWorkgroupXCC_2(kgraph, m_workgroupRemapXCC.value());

                //auto workgroupTags = kgraph.coordinates.getNodes<Workgroup>().to<std::vector>();
                //for(auto workgroupTag : workgroupTags)
                //{
                //    remapWorkgroupXCC(kgraph, workgroupTag, m_workgroupRemapXCC.value());
                //}
                Log::info("================RemapXCC Done=================");
            }

            Log::info("================ConnectWorkgroups=================");
            // If a MacroTileNumber is dangling, attach a workgroup to it.
            connectWorkgroups(kgraph);
            Log::info("================ConnectWorkgroups Done=================");

            //if(m_workgroupRemapXCC.has_value())
            //{
            //    Log::info("================RemapXCC=================");
            //    auto const& arch = m_context->targetArchitecture();
            //    AssertFatal(arch.HasCapability(GPUCapability::HasXCC),
            //                "XCC-aware workgroup remapping not available on: ",
            //                arch.target().toString());

            //    auto workgroupTags = kgraph.coordinates.getNodes<Workgroup>().to<std::vector>();
            //    for(auto workgroupTag : workgroupTags)
            //    {
            //        remapWorkgroupXCC(kgraph, workgroupTag, m_workgroupRemapXCC.value());
            //    }
            //    Log::info("================RemapXCC Done=================");
            //}

            return kgraph;
        }
    }
}
