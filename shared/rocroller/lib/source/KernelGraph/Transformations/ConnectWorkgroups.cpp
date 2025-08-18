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

            Expression::ExpressionPtr toInt32(Expression::ExpressionPtr expr)
            {
                return std::make_shared<Expression::Expression>(
                    Expression::Convert{{.arg{expr}}, DataType::Int32});
            }

            void TileSizeInfo::recordSize(int dim, int tileNumTag, auto direction, auto expr)
            {
                // TODO: Is there a way to make this safer?
                //
                // The result of, for example:
                //
                //     matrixSize(M) / tileSize(M)
                //
                // should fit within a Int32

                AssertFatal(0 <= dim && dim < 3);

                if(expr != nullptr)
                {
                    if(sizes[dim] == nullptr)
                    {
                        sizes[dim]
                            = resultType(expr).varType != DataType::Int32 ? toInt32(expr) : expr;
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

            void connectWorkgroupsWithMapping(TileSizeInfo const&                  info,
                                              rocRoller::KernelGraph::KernelGraph& graph,
                                              int                                  dimension,
                                              Expression::ExpressionPtr            size)
            {
                auto totalSize = totalNumberOfWorkgroups(info);
                auto numDims   = workgroupDimensions(info);

                AssertFatal(numDims == 2);
                AssertFatal(dimension == 0 || dimension == 1);

                for(auto direction : {GD::Downstream, GD::Upstream})
                {
                    // Downstream: Starting at wgTop looking down
                    // Upstream:   Starting at wgBot looking up

                    auto [_, parallel, perpendicular]
                        = workgroupMapping(info, graph, direction, dimension, size);

                    std::array<int, 2> tileNumTags = {parallel, perpendicular};

                    for(auto dim = 0; dim < numDims; ++dim)
                    {
                        for(auto tileNumTag : info.danglers.at({dim, direction}))
                        {
                            if(direction == GD::Upstream)
                            {
                                Log::debug("KernelGraph::ConnectWorkgroups: Adding PassThrough "
                                           "from tile {} to mapped-tile {} (size {})",
                                           tileNumTag,
                                           tileNumTags[dim],
                                           toString(info.sizes[dim]));
                                graph.coordinates.addElement(
                                    PassThrough(), {tileNumTag}, {tileNumTags[dim]});
                            }
                            else
                            {
                                Log::debug("KernelGraph::ConnectWorkgroups: Adding PassThrough "
                                           "from mapped-tile {} (size {}) to tile {}",
                                           tileNumTags[dim],
                                           toString(info.sizes[dim]),
                                           tileNumTag);
                                graph.coordinates.addElement(
                                    PassThrough(), {tileNumTags[dim]}, {tileNumTag});
                            }
                        }
                    }
                }
            }

            std::tuple<int, int, int> workgroupMapping(TileSizeInfo const&                  info,
                                                       rocRoller::KernelGraph::KernelGraph& graph,
                                                       GD                        direction,
                                                       uint                      dimension,
                                                       Expression::ExpressionPtr size)
            {
                AssertFatal(dimension == 0 || dimension == 1);
                AssertFatal(workgroupDimensions(info) == 2);

                auto totalSize = totalNumberOfWorkgroups(info);
                auto workgroup = graph.coordinates.addElement(Workgroup(0, totalSize));

                // Downstream: Starting at workgroup looking down (forward transform)
                // Upstream:   Starting at workgroup looking up (reverse transform)

                using ExpressionPtr     = Expression::ExpressionPtr;
                using ExpressionPtrPair = std::pair<ExpressionPtr, ExpressionPtr>;
                using ExpressionPtrVectorPair
                    = std::pair<std::vector<ExpressionPtr>, std::vector<ExpressionPtr>>;

                auto one  = Expression::literal(1);
                auto zero = Expression::literal(0);

                auto PA = [](int slot) {
                    return std::make_shared<Expression::Expression>(
                        Expression::PositionalArgument{slot});
                };

                auto parallelSize      = info.sizes[dimension];
                auto perpendicularSize = info.sizes[1 - dimension];

                auto blockSize = toInt32(size * perpendicularSize);
                setComment(blockSize, "WGM block size");
                auto mainBlockSize = toInt32(totalSize / blockSize) * blockSize;
                setComment(mainBlockSize, "WGM main block size");
                auto tailBlockSize = toInt32(parallelSize % size);
                setComment(tailBlockSize, "WGM tail block size");

                auto groupNumber = graph.coordinates.addElement(Linear());
                auto groupIndex  = graph.coordinates.addElement(Linear(size, nullptr));

                auto blockNumber = graph.coordinates.addElement(Linear());
                auto blockIndex  = graph.coordinates.addElement(Linear(perpendicularSize, nullptr));

                auto mainBlockNumber = graph.coordinates.addElement(Linear());
                auto mainBlockIndex  = graph.coordinates.addElement(Linear(mainBlockSize, nullptr));

                auto tailBlockNumber = graph.coordinates.addElement(Linear());
                auto tailBlockIndex  = graph.coordinates.addElement(Linear(tailBlockSize, nullptr));

                auto parallel = graph.coordinates.addElement(Linear(parallelSize, nullptr));
                auto perpendicular
                    = graph.coordinates.addElement(Linear(perpendicularSize, nullptr));

                // 0 argument is mainBlockNumber
                auto condition = PA(0) == Expression::literal(0);

                ExpressionPtrVectorPair stridesParallel{{zero, size, one, zero},
                                                        {zero, zero, zero, one}};
                ExpressionPtrPair       initialValuesParallel{nullptr,
                                                        toInt32(parallelSize / size) * size};

                ExpressionPtrVectorPair stridesPerpendicular{{zero, one, zero}, {zero, zero, one}};
                ExpressionPtrPair       initialValuesPerpendicular{nullptr, nullptr};

                if(direction == GD::Upstream)
                {
                    graph.coordinates.addElement(PiecewiseAffineJoin(condition,
                                                                     stridesPerpendicular,
                                                                     initialValuesPerpendicular),
                                                 {perpendicular},
                                                 {mainBlockNumber, blockIndex, tailBlockNumber});

                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, stridesParallel, initialValuesParallel),
                        {parallel},
                        {mainBlockNumber, blockNumber, groupIndex, tailBlockIndex});

                    graph.coordinates.addElement(
                        Flatten(), {tailBlockNumber, tailBlockIndex}, {mainBlockIndex});

                    graph.coordinates.addElement(
                        Flatten(), {blockNumber, blockIndex}, {groupNumber});
                    graph.coordinates.addElement(Flatten(), {groupNumber, groupIndex}, {workgroup});
                    graph.coordinates.addElement(
                        Flatten(), {mainBlockNumber, mainBlockIndex}, {workgroup});
                }
                else
                {
                    graph.coordinates.addElement(
                        Tile(), {workgroup}, {mainBlockNumber, mainBlockIndex});
                    graph.coordinates.addElement(Tile(), {workgroup}, {groupNumber, groupIndex});
                    graph.coordinates.addElement(Tile(), {groupNumber}, {blockNumber, blockIndex});
                    graph.coordinates.addElement(
                        Tile(), {mainBlockIndex}, {tailBlockNumber, tailBlockIndex});

                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, stridesParallel, initialValuesParallel),
                        {mainBlockNumber, blockNumber, groupIndex, tailBlockIndex},
                        {parallel});

                    graph.coordinates.addElement(PiecewiseAffineJoin(condition,
                                                                     stridesPerpendicular,
                                                                     initialValuesPerpendicular),
                                                 {mainBlockNumber, blockIndex, tailBlockNumber},
                                                 {perpendicular});
                }

                if(dimension == 0)
                    return {workgroup, parallel, perpendicular};

                return {workgroup, perpendicular, parallel};
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

                // Upstream: newWorkgroupTag is added above workgroupTag
                auto direction = empty(graph.coordinates.getNeighbours(workgroupTag, GD::Upstream))
                                     ? GD::Upstream
                                     : GD::Downstream;

                auto one           = Expression::literal(1u);
                auto numXCCLiteral = Expression::literal(numXCC);

                auto ceilDiv = [&](ExpressionPtr a, ExpressionPtr b) { return (a + b - one) / b; };

                auto xcc = graph.coordinates.addElement(Linear(numXCCLiteral, nullptr));
                auto cu
                    = graph.coordinates.addElement(Linear(ceilDiv(size, numXCCLiteral), nullptr));

                auto PA = [](int slot) {
                    return std::make_shared<Expression::Expression>(
                        Expression::PositionalArgument{slot});
                };

                // 0 argument is XCC, 1 argument is CU
                auto condition = PA(0) <= (size % numXCCLiteral);

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

        ConnectWorkgroups::ConnectWorkgroups(ContextPtr                context,
                                             std::optional<int>        workgroupMappingDim,
                                             std::optional<int>        workgroupRemapXCC,
                                             Expression::ExpressionPtr workgroupMappingValue)
            : m_context(context)
            , m_workgroupMappingDim(workgroupMappingDim)
            , m_workgroupRemapXCC(workgroupRemapXCC)
            , m_workgroupMappingValue(workgroupMappingValue)
        {
        }

        KernelGraph ConnectWorkgroups::apply(KernelGraph const& original)
        {
            using namespace ConnectWorkgroupsDetail;

            auto kgraph = original;
            auto info   = getTileSizeInfo(original);

            if(m_workgroupMappingDim.has_value())
            {
                connectWorkgroupsWithMapping(
                    info, kgraph, m_workgroupMappingDim.value(), m_workgroupMappingValue);
            }
            else
            {
                connectWorkgroupsNoMapping(info, kgraph);
            }

            if(m_workgroupRemapXCC.has_value())
            {
                auto const& arch = m_context->targetArchitecture();
                AssertFatal(arch.HasCapability(GPUCapability::HasXCC),
                            "XCC-aware workgroup remapping not available on: ",
                            arch.target().toString());
                auto workgroupTags = kgraph.coordinates.getNodes<Workgroup>().to<std::vector>();
                for(auto workgroupTag : workgroupTags)
                {
                    remapWorkgroupXCC(kgraph, workgroupTag, m_workgroupRemapXCC.value());
                }
            }

            return kgraph;
        }
    }
}
