/**
 * AddStreamK -- stream accumulation tiles.
 *
 * The usual example is matrix-matrix multiply:
 *
 *   D = A B
 *
 * where A and B are tiled so that A has M x K tiles, and B has K x N
 * tiles.  Tiles in the accumulation (K) dimension will be streamed.
 *
 * The flattened M * N * K global tile-space will be distributed
 * evenly among the CUs.
 *
 * Each CU needs to iterate over its portion of the flattened
 * global tile-space.
 *
 * To facilitate accumulation initialization and prefetching etc, we
 * use two loops to accomplish this: an outer forTileIdx loop and an
 * inner forAccumIdx loop.
 *
 * The inner forAccumIdx loop iterates over the accumulation tiles
 * one-by-one.  The outer forTileIdx loop iterates over the local
 * tile-space.  Its increment is: however many tiles the inner
 * forAccumIdx loop processed.
 *
 * When the inner forAccumIdx loop advances, it will be advancing to a
 * new K tile, but will remain within the same M/N tile.  When the
 * outer forTileIdx loop advances, it will be advancing to a new M/N
 * tile.
 *
 * The local combined index
 *
 *    cuTile = forTileIdx + forAccumIdx
 *
 * is the current CUs local tile in its portion of the global
 * tile-space.  Then
 *
 *    tile = tilesPerCU * cu + cuTile
 *
 * is the global tile that the CU is processing.  Given the global
 * tile, the M/N/K tile coordinates are
 *
 *    m = (tile / numTilesK) / numTilesN;
 *    n = (tile / numTilesK) % numTilesN;
 *    k = tile % numTilesK;
 */

#include "Expression_fwd.hpp"
#include <map>
#include <unordered_set>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddStreamK.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CG = rocRoller::KernelGraph::CoordinateGraph;
        using GD     = rocRoller::Graph::Direction;

        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Register;

        std::string AddStreamK::name() const
        {
            return concatenate("AddStreamK");
        }

        AddStreamK::AddStreamK(std::vector<int> const&                       dims,
                               std::vector<Expression::ExpressionPtr> const& tileNumberCoordSizes,
                               std::string const&                            topLoop,
                               Expression::ExpressionPtr                     numCUs,
                               ContextPtr                                    context)
            : m_dimensions(dims)
            , m_topLoop(topLoop)
            , m_tileNumberCoordSizes(tileNumberCoordSizes)
            , m_numCUs(numCUs)
            , m_context(context)
        {
        }

        //
        // Helpers
        //
        int conditionalFor(KernelGraph&              graph,
                           int                       incrementCoord,
                           Expression::ExpressionPtr conditionExpr,
                           Expression::ExpressionPtr incrementExpr,
                           const std::string&        loopName,
                           VariableType              varType)
        {
            auto forLoop = graph.control.addElement(ForLoopOp{conditionExpr, loopName});
            graph.mapper.connect<Dimension>(forLoop, incrementCoord);

            auto initialize = graph.control.addElement(
                Assign{Register::Type::Scalar, Expression::literal(0, varType)});
            graph.mapper.connect(initialize, incrementCoord, NaryArgument::DEST);

            auto increment
                = graph.control.addElement(Assign{Register::Type::Scalar, incrementExpr});
            graph.mapper.connect(increment, incrementCoord, NaryArgument::DEST);

            graph.control.addElement(Initialize(), {forLoop}, {initialize});
            graph.control.addElement(ForLoopIncrement(), {forLoop}, {increment});

            return forLoop;
        }

        int AddStreamK::addTileSpaceCT(KernelGraph&              graph,
                                       bool                      forward,
                                       Expression::ExpressionPtr numTotalTiles,
                                       Expression::ExpressionPtr numTilesPerCU)
        {
            // Create forward/reverse tile-numbers for each dimension
            // and attach to all staged tile-number coordinates
            std::vector<int> tileNumbers;
            for(auto d : m_dimensions)
            {
                auto tileNumber = graph.coordinates.addElement(
                    MacroTileNumber(d, m_tileNumberCoordSizes[d], nullptr));

                for(auto tileNumTag : m_tileNumberCoords.at(d))
                {
                    if(forward
                       && empty(graph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                        graph.coordinates.addElement(PassThrough(), {tileNumTag}, {tileNumber});
                    if(!forward && empty(graph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
                        graph.coordinates.addElement(PassThrough(), {tileNumber}, {tileNumTag});
                }

                tileNumbers.push_back(tileNumber);
            }

            // Create forward/reverse accumulator tile-numbers
            //
            // Appending means: accumulating dimension is fastest moving
            tileNumbers.push_back(m_accumulatorCoord);

            // Create foward/reverse flattend tile-space
            auto tileSpace = graph.coordinates.addElement(Linear(numTotalTiles, nullptr));
            if(forward)
            {
                graph.coordinates.addElement(Flatten(), tileNumbers, std::vector<int>{tileSpace});
            }
            else
            {
                graph.coordinates.addElement(Tile(), std::vector<int>{tileSpace}, tileNumbers);
            }

            auto CUs        = graph.coordinates.addElement(Linear(m_numCUs, nullptr));
            auto workgroup  = graph.coordinates.addElement(Workgroup());
            auto tilesByCUs = graph.coordinates.addElement(Linear(numTilesPerCU, nullptr));
            if(forward)
            {
                graph.coordinates.addElement(PassThrough(), {CUs}, {workgroup});
                graph.coordinates.addElement(Tile(), {tileSpace}, {CUs, tilesByCUs});
            }
            else
            {
                graph.coordinates.addElement(PassThrough(), {workgroup}, {CUs});
                graph.coordinates.addElement(Flatten(), {CUs, tilesByCUs}, {tileSpace});
            }

            return tilesByCUs;
        }

        //
        // Stage
        //
        // Look for all leaf MacroTileNumbers with matching
        // sub-dimension.
        //
        // Matches are: tile->dim in m_dimensions.
        //
        void AddStreamK::stage(KernelGraph const& graph)
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

                for(auto tileTag : m_tileNumberCoords[dimension])
                {
                    Log::debug("  dimension {} coord: {}", dimension, tileTag);
                }
            }

            m_accumulatorCoord = getForLoop(m_topLoopOp, graph);

            Log::debug("  accumulator loop coord: {}", m_accumulatorCoord);
        }

        //
        // Commit
        //
        void AddStreamK::commit(KernelGraph& graph)
        {
            //
            // Compute size of global and local tile-spaces
            //
            auto accumulatorCoordSize = graph.coordinates.get<ForLoop>(m_accumulatorCoord)->size;
            auto numTotalTiles        = accumulatorCoordSize;
            for(auto d : m_dimensions)
                numTotalTiles = numTotalTiles * m_tileNumberCoordSizes[d];
            numTotalTiles = simplify(numTotalTiles);

            auto varType = resultType(numTotalTiles).varType;
            auto one     = Expression::literal(1, varType);
            auto zero    = Expression::literal(0, varType);

            auto numTilesPerCU = simplify((numTotalTiles + m_numCUs - one) / m_numCUs);

            //
            // Helper
            //
            auto DF = [=](int tag) {
                return std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{tag, Register::Type::Scalar, varType});
            };

            auto cu = m_context->kernel()->workgroupIndex().at(0)->expression();

            //
            // Add forward/reverse tile-space coordinate transforms
            //
            auto forwardTilesByCUs = addTileSpaceCT(graph, true, numTilesPerCU, numTilesPerCU);
            auto reverseTilesByCUs = addTileSpaceCT(graph, false, numTilesPerCU, numTilesPerCU);

            //
            // Add tile and accumulator for loops dimensions and iterators
            //
            auto forTileIncr       = graph.coordinates.addElement(Linear(numTilesPerCU, one));
            auto forwardForTileIdx = graph.coordinates.addElement(ForLoop(numTilesPerCU, one));
            auto reverseForTileIdx = graph.coordinates.addElement(ForLoop(numTilesPerCU, one));
            auto forwardForAccumIdx
                = graph.coordinates.addElement(ForLoop(accumulatorCoordSize, one));
            auto reverseForAccumIdx
                = graph.coordinates.addElement(ForLoop(accumulatorCoordSize, one));
            auto forAccumIncr = *only(graph.coordinates.getInputNodeIndices(
                m_accumulatorCoord, CG::isEdge<CG::DataFlow>));

            graph.coordinates.addElement(
                Split(), {forwardTilesByCUs}, {forwardForTileIdx, forwardForAccumIdx});
            graph.coordinates.addElement(
                Join(), {reverseForTileIdx, reverseForAccumIdx}, {reverseTilesByCUs});

            //
            // Create tile loop
            //
            auto cuTilesOuterExpr    = DF(forTileIncr) < numTilesPerCU;
            auto totalTilesOuterExpr = (numTilesPerCU * cu + DF(forTileIncr)) < numTotalTiles;
            auto incrementOuterExpr  = DF(forTileIncr) + DF(forAccumIncr);

            auto forTileOp = conditionalFor(graph,
                                            forTileIncr,
                                            cuTilesOuterExpr && totalTilesOuterExpr,
                                            incrementOuterExpr,
                                            "StreamTileLoop",
                                            varType);

            graph.coordinates.addElement(DataFlow(), {forTileIncr}, {forwardForTileIdx});
            graph.coordinates.addElement(DataFlow(), {forTileIncr}, {reverseForTileIdx});

            //
            // Move old dataflow edge from forAccumIncr to forAccumIdx
            //
            {
                auto iterator = *only(graph.coordinates.getInputNodeIndices(
                    m_accumulatorCoord, CG::isEdge<CG::DataFlow>));
                auto dataflow = *only(graph.coordinates.getNeighbours<GD::Downstream>(iterator));
                graph.coordinates.deleteElement(dataflow);
                graph.coordinates.addElement(DataFlow(), {forAccumIncr}, {forwardForAccumIdx});
                graph.coordinates.addElement(DataFlow(), {forAccumIncr}, {reverseForAccumIdx});
            }

            //
            // Hi-jack accumulator loop
            //
            auto currentNonAccumTileTag = graph.coordinates.addElement(Dimension());
            auto currentNonAccumTileExpr
                = (numTilesPerCU * cu + DF(forTileIncr)) / accumulatorCoordSize;
            auto assignNonAccumTile
                = graph.control.addElement(Assign{Register::Type::Scalar, currentNonAccumTileExpr});
            graph.mapper.connect(assignNonAccumTile, currentNonAccumTileTag, NaryArgument::DEST);

            auto differentAccumTileInnerExpr
                = ((numTilesPerCU * cu + DF(forTileIncr) + DF(forAccumIncr)) / accumulatorCoordSize)
                  == DF(currentNonAccumTileTag);
            auto cuTilesInnerExpr = (DF(forTileIncr) + DF(forAccumIncr)) < numTilesPerCU;
            auto totalTilesInnerExpr
                = (numTilesPerCU * cu + DF(forTileIncr) + DF(forAccumIncr)) < numTotalTiles;
            auto incrementInnerExpr = DF(forAccumIncr) + one;

            auto forAccumOp = *graph.control.get<ForLoopOp>(m_topLoopOp);
            forAccumOp.condition
                = differentAccumTileInnerExpr && cuTilesInnerExpr && totalTilesInnerExpr;
            graph.control.setElement(m_topLoopOp, forAccumOp);

            //
            // Insert for-loops into graph
            //
            std::vector<int> defineChain;
            for(auto tag : {forTileIncr, forAccumIncr, currentNonAccumTileTag})
            {
                auto def = graph.control.addElement(Assign{Register::Type::Scalar, zero});
                graph.mapper.connect(def, tag, NaryArgument::DEST);
                defineChain.push_back(def);
            }

            auto scope = graph.control.addElement(Scope());
            graph.control.addElement(Body(), {scope}, {defineChain.front()});
            for(int i = 1; i < defineChain.size(); ++i)
            {
                graph.control.addElement(Sequence(), {defineChain[i - 1]}, {defineChain[i]});
            }
            graph.control.addElement(Sequence(), {defineChain.back()}, {forTileOp});
            graph.control.addElement(Body(), {forTileOp}, {assignNonAccumTile});

            //
            // Insert the new Scope inplace of the original topLoopOp
            // (which was an accumulation loop).
            //
            auto location = graph.control.getLocation(m_topLoopOp);
            for(auto const& input : location.incoming)
            {
                auto edge = graph.control.getElement(input);
                auto node = *only(graph.control.getNeighbours<GD::Upstream>(input));
                graph.control.deleteElement(input);
                graph.control.addElement(edge, {node}, {scope});
            }
            for(auto const& output : location.outgoing)
            {
                auto edge = graph.control.getElement(output);
                auto node = *only(graph.control.getNeighbours<GD::Downstream>(output));

                auto maybeSequence = graph.control.get<Sequence>(output);
                if(maybeSequence)
                {
                    graph.control.deleteElement(output);
                    graph.control.addElement(edge, {scope}, {node});
                }
            }

            graph.control.addElement(Sequence(), {assignNonAccumTile}, {m_topLoopOp});
        }

        KernelGraph AddStreamK::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddStreamK");

            // Make sure we can find the top-for-loop location
            auto findTopLoopPredicate = [&](int tag) -> bool {
                auto maybeForLoop = original.control.get<ForLoopOp>(tag);
                if(!maybeForLoop)
                    return false;
                if(maybeForLoop->loopName == m_topLoop)
                    return true;
                return false;
            };
            auto maybeTopLoopOp = only(original.control.findElements(findTopLoopPredicate));
            if(!maybeTopLoopOp)
            {
                rocRoller::Log::warn("Unable to find ForLoop '{}' during AddStreamK pass.  "
                                     "AddStreamK transform skipped.",
                                     m_topLoop);
                return original;
            }
            m_topLoopOp = *maybeTopLoopOp;

            auto graph = original;
            stage(graph);
            commit(graph);

            return graph;
        }
    }
}
