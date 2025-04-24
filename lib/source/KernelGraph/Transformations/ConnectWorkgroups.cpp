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
            void TileSizeInfo::recordSize(int dim, int tileNumTag, auto direction, auto expr)
            {
                AssertFatal(0 <= dim && dim < 3);

                if(expr != nullptr)
                {
                    if(sizes[dim] == nullptr)
                    {
                        sizes[dim] = expr;
                    }
                    else
                    {
                        // They aren't alway identical, but they
                        // should be equivalent after resolving
                        // command arguments.
                        //
                        // For example: A_size_1 is K; and so is B_size_0
                        //
                        // TODO: Emit a command predicate to enforce this
                    }
                }

                danglers[{dim, direction}].insert(tileNumTag);
            }

            TileSizeInfo getTileSizeInfo(KernelGraph const& kgraph)
            {
                TileSizeInfo info;
                auto tileNumTags = kgraph.coordinates.getNodes<MacroTileNumber>().to<std::vector>();
                for(auto const& tileNumTag : tileNumTags)
                {
                    if(empty(kgraph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                    {
                        // If we have no downstream neighbours, we
                        // will create a new workgroup below this, and
                        // look upstream
                        auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                        info.recordSize(tileNum.dim, tileNumTag, GD::Upstream, tileNum.size);
                    }
                    if(empty(kgraph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
                    {
                        auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                        info.recordSize(tileNum.dim, tileNumTag, GD::Downstream, tileNum.size);
                    }
                }
                return info;
            }

            int workgroupDimensions(TileSizeInfo const& info)
            {
                if(info.sizes[0] != nullptr && info.sizes[1] != nullptr && info.sizes[2] != nullptr)
                    return 3;
                if(info.sizes[0] != nullptr && info.sizes[1] != nullptr)
                    return 2;
                if(info.sizes[0] != nullptr)
                    return 1;
                Throw<FatalError>("Invalid number of dimensions.");
            }

            Expression::ExpressionPtr totalNumberOfWorkgroups(TileSizeInfo const& info)
            {
                AssertFatal(info.sizes[0] != nullptr);
                auto rv = info.sizes[0];
                for(int i = 1; i < 3; ++i)
                    if(info.sizes[i] != nullptr)
                        rv = rv * info.sizes[i];
                return rv;
            }

            std::map<std::pair<int, rocRoller::Graph::Direction>, int>
                connectWorkgroupsNoMapping(TileSizeInfo const& info, KernelGraph& kgraph)
            {
                std::map<std::pair<int, rocRoller::Graph::Direction>, int> rv;

                for(auto [key, tileNumTags] : info.danglers)
                {
                    auto [dim, direction] = key;
                    for(auto tileNumTag : tileNumTags)
                    {
                        auto workgroupTag
                            = kgraph.coordinates.addElement(Workgroup(dim, info.sizes[dim]));
                        rv[{dim, direction}] = workgroupTag;
                        if(direction == GD::Upstream)
                        {
                            Log::debug("KernelGraph::ConnectWorkgroups: Adding PassThrough from "
                                       "tile {} (size {}) to workgroup {}",
                                       tileNumTag,
                                       toString(info.sizes[dim]),
                                       workgroupTag);
                            kgraph.coordinates.addElement(
                                PassThrough(), {tileNumTag}, {workgroupTag});
                        }
                        else
                        {
                            Log::debug("KernelGraph::ConnectWorkgroups: Adding PassThrough from "
                                       "workgroup {} to tile {} (size {})",
                                       workgroupTag,
                                       tileNumTag,
                                       toString(info.sizes[dim]));
                            kgraph.coordinates.addElement(
                                PassThrough(), {workgroupTag}, {tileNumTag});
                        }
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

                auto newWorkgroupTag = graph.coordinates.addElement(Workgroup(0, size));

                auto direction = empty(graph.coordinates.getNeighbours(workgroupTag, GD::Upstream))
                                     ? GD::Upstream
                                     : GD::Downstream;

                auto one           = Expression::literal(1u);
                auto numXCCLiteral = Expression::literal(numXCC);

                auto ceilDiv = [&](ExpressionPtr a, ExpressionPtr b) { return (a + b - one) / b; };

                auto xcc = graph.coordinates.addElement(Linear(numXCCLiteral, nullptr));
                auto cu
                    = graph.coordinates.addElement(Linear(ceilDiv(size, numXCCLiteral), nullptr));

                auto DF = [](int tag) {
                    return std::make_shared<Expression::Expression>(
                        Expression::DataFlowTag{tag, Register::Type::Scalar, DataType::UInt32});
                };

                // 0 argument is XCC, 1 argument is CU
                auto condition = DF(0) <= (size % numXCCLiteral);

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
                }
                else
                {
                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, strides, initialValues),
                        {workgroupTag},
                        {xcc, cu});
                    graph.coordinates.addElement(Flatten(), {cu, xcc}, {newWorkgroupTag});
                }

                return newWorkgroupTag;
            }
        }

        ConnectWorkgroups::ConnectWorkgroups(CommandParametersPtr params, ContextPtr context)
            : m_params(params)
            , m_context(context)
        {
        }

        KernelGraph ConnectWorkgroups::apply(KernelGraph const& original)
        {
            using namespace ConnectWorkgroupsDetail;

            auto kgraph = original;
            auto info   = getTileSizeInfo(original);

            connectWorkgroupsNoMapping(info, kgraph);

            return kgraph;
        }
    }
}
