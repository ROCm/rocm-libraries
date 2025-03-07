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

#include <map>
#include <unordered_set>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/LoopOverTileNumbers.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using GD = rocRoller::Graph::Direction;

        std::string LoopOverTileNumbers::name() const
        {
            return concatenate("LoopOverTileNumbers");
        }

        LoopOverTileNumbers::LoopOverTileNumbers(std::vector<int> const&  dims,
                                                 std::vector<uint> const& tileNumberCoordSizes,
                                                 uint                     numIteratedTiles,
                                                 std::string const&       topLoop,
                                                 ContextPtr               context)
            : m_dimensions(dims)
            , m_topLoop(topLoop)
            , m_context(context)
        {
            for(auto s : tileNumberCoordSizes)
                m_tileNumberCoordSizes.push_back(Expression::literal(s));
            m_numIteratedTileNumbers = Expression::literal(numIteratedTiles);
        }

        //
        // Stage
        //
        // Look for all leaf MacroTileNumbers with matching
        // sub-dimension.
        //
        // Matches are: tile->dim in m_dimensions.
        //
        void LoopOverTileNumbers::stage(KernelGraph const& graph)
        {
            // Find all dangling MacroTileNumber dimensions associated
            // with the requested dimensions
            for(auto dimension : m_dimensions)
            {
                auto danglingTileNumberPredicate = [&](int tag) {
                    auto maybeTileNumber = graph.coordinates.get<MacroTileNumber>(tag);
                    if(!maybeTileNumber)
                        return false;
                    if(maybeTileNumber->dim != dimension)
                        return false;
                    if(empty(graph.coordinates.getNeighbours<GD::Upstream>(tag))
                       || empty(graph.coordinates.getNeighbours<GD::Downstream>(tag)))
                        return true;
                    return false;
                };

                m_tileNumberCoords[dimension]
                    = graph.coordinates.findElements(danglingTileNumberPredicate)
                          .to<std::unordered_set>();
            }
        }

        //
        // Commit
        //
        void LoopOverTileNumbers::commit(KernelGraph& graph)
        {
            auto numTotalTiles = Expression::literal(1u);
            for(auto d : m_dimensions)
                numTotalTiles = numTotalTiles * m_tileNumberCoordSizes[d];
            numTotalTiles = simplify(numTotalTiles);

            auto [forLoopCoord, forLoopOp]
                = rangeFor(graph, m_numIteratedTileNumbers, "TileLoop", DataType::UInt32);

            // Create forward/reverse tile-numbers for each dimension
            // and attach to all staged tile-number coordinates
            std::vector<int> tileNumbersUp, tileNumbersDown;
            for(auto d : m_dimensions)
            {
                auto up = graph.coordinates.addElement(
                    MacroTileNumber(d, m_tileNumberCoordSizes[d], nullptr));
                auto down = graph.coordinates.addElement(
                    MacroTileNumber(d, m_tileNumberCoordSizes[d], nullptr));

                for(auto tileNumTag : m_tileNumberCoords.at(d))
                {
                    if(empty(graph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
                        graph.coordinates.addElement(PassThrough(), {down}, {tileNumTag});
                    if(empty(graph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                        graph.coordinates.addElement(PassThrough(), {tileNumTag}, {up});
                }

                tileNumbersUp.push_back(up);
                tileNumbersDown.push_back(down);
            }

            // Create foward/reverse transforms from (Workgroup,
            // ForLoop) to new tile-numbers
            auto workgroups = simplify(numTotalTiles / m_numIteratedTileNumbers);

            auto initUp = graph.coordinates.addElement(MacroTileNumber(0, workgroups, nullptr));
            auto flatUp = graph.coordinates.addElement(Linear(numTotalTiles, nullptr));
            graph.coordinates.addElement(Flatten(), {forLoopCoord, initUp}, {flatUp});
            graph.coordinates.addElement(Tile(), std::vector<int>{flatUp}, tileNumbersDown);

            auto initDown = graph.coordinates.addElement(MacroTileNumber(0, workgroups, nullptr));
            auto flatDown = graph.coordinates.addElement(Linear(numTotalTiles, nullptr));
            graph.coordinates.addElement(Tile(), {flatDown}, {forLoopCoord, initDown});
            graph.coordinates.addElement(Flatten(), tileNumbersUp, std::vector<int>{flatDown});

            // Insert ForLoop into graph
            insertWithBody(graph, m_topLoopOp, forLoopOp);
        }

        //
        // Apply
        //
        KernelGraph LoopOverTileNumbers::apply(KernelGraph const& original)
        {
            // Make sure we can find the top-for-loop location
            auto findTopLoopPredicate = [&](int tag) -> bool {
                auto maybeForLoop = original.control.get<ForLoopOp>(tag);
                if(!maybeForLoop)
                    return false;
                if(maybeForLoop->loopName == m_topLoop)
                    return true;
                return false;
            };
            auto maybeTopLoopOp = original.control.findNodes(
                *original.control.roots().begin(), findTopLoopPredicate, GD::Downstream);
            if(maybeTopLoopOp.empty())
            {
                rocRoller::Log::getLogger()->warn(
                    "Unable to find ForLoop '{}' during LoopOverTileNumbers pass.  "
                    "LoopOverTileNumbers transform skipped.",
                    m_topLoop);
                return original;
            }
            m_topLoopOp = *maybeTopLoopOp.take(1).only();

            auto graph = original;
            stage(graph);
            commit(graph);
            return graph;
        }
    }
}
