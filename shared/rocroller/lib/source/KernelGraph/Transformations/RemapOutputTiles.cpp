// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/RemapOutputTiles.hpp>
#include <rocRoller/KernelGraph/Transforms/RemapOutputTiles_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using GD = rocRoller::Graph::Direction;

        namespace RemapOutputTilesDetail
        {
            void TileSizeInfo::recordSize(int dim, int tileNumTag, auto direction, auto expr)
            {
                // TODO: Is there a way to make this safer?
                //
                // The result of, for example:
                //
                //     matrixSize(M) / tileSize(M)
                //
                // should fit within a Int32

                AssertFatal(0 <= dim && dim < 3, ShowValue(dim));

                if(expr != nullptr)
                {
                    if(sizes[dim] == nullptr)
                    {
                        sizes[dim] = convert(DataType::Int32, expr);
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
                    if(std::empty(kgraph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                    {
                        // If we have no downstream neighbours, we
                        // will create a new workgroup below this, and
                        // look upstream
                        auto tileNum = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                        info.recordSize(tileNum.dim, tileNumTag, GD::Upstream, tileNum.size);
                    }
                    if(std::empty(kgraph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
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

            void connectWorkgroupsWithMapping(TileSizeInfo const&                  info,
                                              rocRoller::KernelGraph::KernelGraph& graph,
                                              int                                  dimension,
                                              Expression::ExpressionPtr            size)
            {
                auto totalSize = totalNumberOfWorkgroups(info);
                auto numDims   = workgroupDimensions(info);

                AssertFatal(numDims == 2, ShowValue(numDims));
                AssertFatal(dimension == 0 || dimension == 1, ShowValue(dimension));

                for(auto direction : {GD::Downstream, GD::Upstream})
                {
                    // Downstream: Starting at wgTop looking down
                    // Upstream:   Starting at wgBot looking up

                    auto remappedDims = workgroupMapping(info, graph, direction, dimension, size);
                    std::array<int, 2> tileNumTags
                        = {remappedDims.parallelDim, remappedDims.perpendicularDim};

                    for(auto dim = 0; dim < numDims; ++dim)
                    {
                        for(auto tileNumTag : info.danglers.at({dim, direction}))
                        {
                            if(direction == GD::Upstream)
                            {
                                Log::debug("KernelGraph::RemapOutputTiles: Adding PassThrough "
                                           "from tile {} to mapped-tile {} (size {})",
                                           tileNumTag,
                                           tileNumTags[dim],
                                           toString(info.sizes[dim]));
                                graph.coordinates.addElement(
                                    PassThrough(), {tileNumTag}, {tileNumTags[dim]});
                            }
                            else
                            {
                                Log::debug("KernelGraph::RemapOutputTiles: Adding PassThrough "
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

            RemappedDimensions workgroupMapping(TileSizeInfo const&                  info,
                                                rocRoller::KernelGraph::KernelGraph& graph,
                                                GD                                   direction,
                                                uint                                 dimension,
                                                Expression::ExpressionPtr            size)
            {
                AssertFatal(dimension == 0 || dimension == 1);
                AssertFatal(workgroupDimensions(info) == 2);

                auto totalSize = totalNumberOfWorkgroups(info);

                // This MacroTileNumber will be connected to a workgroup in
                // later transformation or be used in AddStreamK with dimension
                // K to create a flattened tile space.
                auto workgroup
                    = graph.coordinates.addElement(MacroTileNumber(0, totalSize, nullptr));

                // Downstream: Starting at workgroup looking down (forward transform)
                // Upstream:   Starting at workgroup looking up (reverse transform)

                using ExpressionPtr     = Expression::ExpressionPtr;
                using ExpressionPtrPair = std::pair<ExpressionPtr, ExpressionPtr>;
                using ExpressionPtrVectorPair
                    = std::pair<std::vector<ExpressionPtr>, std::vector<ExpressionPtr>>;

                auto one  = Expression::literal(1);
                auto zero = Expression::literal(0);

                auto parallelSize      = info.sizes[dimension];
                auto perpendicularSize = info.sizes[1 - dimension];

                auto blockSize = convert(DataType::Int32, size * perpendicularSize);
                setComment(blockSize, "WGM block size");
                auto mainBlockSize = convert(DataType::Int32, totalSize / blockSize) * blockSize;
                setComment(mainBlockSize, "WGM main block size");
                auto tailBlockSize = convert(DataType::Int32, parallelSize % size);
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
                auto condition
                    = Expression::positionalArgument(0, Register::Type::Scalar, DataType::UInt32)
                      == Expression::literal(0);

                ExpressionPtrVectorPair stridesParallel{{zero, size, one, zero},
                                                        {zero, zero, zero, one}};
                ExpressionPtrPair       initialValuesParallel{
                    nullptr, convert(DataType::Int32, parallelSize / size) * size};

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
        }

        RemapOutputTiles::RemapOutputTiles(std::optional<int>        workgroupMappingDim,
                                           Expression::ExpressionPtr workgroupMappingValue)
            : m_workgroupMappingDim(workgroupMappingDim)
            , m_workgroupMappingValue(workgroupMappingValue)
        {
        }

        KernelGraph RemapOutputTiles::apply(KernelGraph const& original)
        {
            using namespace RemapOutputTilesDetail;

            auto kgraph = original;

            if(m_workgroupMappingDim.has_value())
            {
                auto info = getTileSizeInfo(original);
                connectWorkgroupsWithMapping(
                    info, kgraph, m_workgroupMappingDim.value(), m_workgroupMappingValue);
            }

            return kgraph;
        }
    }
}
