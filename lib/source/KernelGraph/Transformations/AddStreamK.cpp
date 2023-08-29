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
 * evenly among the WGs.
 *
 * Each WG needs to iterate over its portion of the flattened
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
 *    wgTile = forTileIdx + forAccumIdx
 *
 * is the current WGs local tile in its portion of the global
 * tile-space.  Then
 *
 *    tile = tilesPerWG * wg + wgTile
 *
 * is the global tile that the WG is processing.  Given the global
 * tile, the M/N/K tile coordinates are
 *
 *    m = (tile / numTilesK) / numTilesN;
 *    n = (tile / numTilesK) % numTilesN;
 *    k = tile % numTilesK;
 */

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

        AddStreamK::AddStreamK(std::vector<int> const&   dims,
                               std::string const&        topLoop,
                               std::string const&        accumulatorLoop,
                               Expression::ExpressionPtr numWGs,
                               ContextPtr                context)
            : m_dimensionIndices(dims)
            , m_topLoop(topLoop)
            , m_accumulatorLoop(accumulatorLoop)
            , m_numWGs(numWGs)
            , m_context(context)
        {
            m_numTileArgExprs.resize(m_dimensionIndices.size() + 1);
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
                                       Expression::ExpressionPtr numTilesPerWG)
        {
            // Create forward/reverse tile-numbers for each dimension
            // and attach to all staged tile-number coordinates
            std::vector<int> tileNumbers;
            for(auto d : m_dimensionIndices)
            {
                auto tileNumber
                    = graph.coordinates.addElement(MacroTileNumber(d, m_numTiles[d], nullptr));

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

            // Create foward/reverse flattened tile-space
            auto tileSpace = graph.coordinates.addElement(Linear(numTotalTiles, nullptr));
            if(forward)
            {
                graph.coordinates.addElement(Flatten(), tileNumbers, std::vector<int>{tileSpace});
            }
            else
            {
                graph.coordinates.addElement(Tile(), std::vector<int>{tileSpace}, tileNumbers);
            }

            auto WGs        = graph.coordinates.addElement(Linear(m_numWGs, nullptr));
            auto workgroup  = graph.coordinates.addElement(Workgroup());
            auto tilesByWGs = graph.coordinates.addElement(Linear(numTilesPerWG, nullptr));
            if(forward)
            {
                graph.coordinates.addElement(PassThrough(), {WGs}, {workgroup});
                graph.coordinates.addElement(Tile(), {tileSpace}, {WGs, tilesByWGs});
            }
            else
            {
                graph.coordinates.addElement(PassThrough(), {workgroup}, {WGs});
                graph.coordinates.addElement(Flatten(), {WGs, tilesByWGs}, {tileSpace});
            }

            return tilesByWGs;
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
            m_accumulatorCoord = getForLoop(m_accumulatorLoopOp, graph);
            Log::debug("  accumulator loop coord: {}", m_accumulatorCoord);

            // Find all dangling MacroTileNumber dimensions associated
            // with the requested dimensions
            for(auto dimension : m_dimensionIndices)
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

                // Fill number-of-tiles using MacroTileNumber sizes
                // from load operations (store operations are missing
                // that info).
                for(auto tileNumberTag : m_tileNumberCoords[dimension])
                {
                    if(!m_numTileArgExprs[dimension])
                    {
                        auto macTileNumber = graph.coordinates.get<MacroTileNumber>(tileNumberTag);
                        if(macTileNumber->size)
                            m_numTileArgExprs[dimension] = macTileNumber->size;
                    }
                }
                m_numTileArgExprs.back() = graph.coordinates.get<ForLoop>(m_accumulatorCoord)->size;

                for(auto tileNumberTag : m_tileNumberCoords[dimension])
                {
                    AssertFatal(m_numTileArgExprs[dimension]);
                    Log::debug("  dimension: {} coord: {} size: {}",
                               dimension,
                               tileNumberTag,
                               toString(m_numTileArgExprs[dimension]));
                }
            }
        }

        //
        // Commit
        //
        void AddStreamK::commit(KernelGraph& graph)
        {
            //
            // Compute size of global and local tile-spaces
            //
            auto accumulatorCoordSize = m_numTileArgExprs.back();
            auto numTotalTiles        = accumulatorCoordSize;
            for(auto d : m_dimensionIndices)
                numTotalTiles = numTotalTiles * m_numTiles.at(d);
            numTotalTiles = simplify(numTotalTiles);

            auto varType = resultType(numTotalTiles).varType;
            auto one     = Expression::literal(1, varType);
            auto zero    = Expression::literal(0, varType);

            Log::debug("  accumulatorCoordSize: {}", toString(accumulatorCoordSize));
            Log::debug("  numTotalTiles: {}", toString(numTotalTiles));
            Log::debug("  numTilesPerWG: {}", toString(m_numTilesPerWG));

            //
            // Helper
            //
            auto DF = [varType](int tag) {
                return std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{tag, Register::Type::Scalar, varType});
            };

            auto wg     = graph.coordinates.addElement(Workgroup(0));
            auto wgExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{wg, Register::Type::Scalar, DataType::UInt32});

            //
            // Add forward/reverse tile-space coordinate transforms
            //
            auto forwardTilesByWGs = addTileSpaceCT(graph, true, numTotalTiles, m_numTilesPerWG);
            auto reverseTilesByWGs = addTileSpaceCT(graph, false, numTotalTiles, m_numTilesPerWG);

            //
            // Add tile and accumulator for loops dimensions and iterators
            //
            auto forTileIncr       = graph.coordinates.addElement(Linear(m_numTilesPerWG, one));
            auto forwardForTileIdx = graph.coordinates.addElement(ForLoop(m_numTilesPerWG, one));
            auto reverseForTileIdx = graph.coordinates.addElement(ForLoop(m_numTilesPerWG, one));
            auto forwardForAccumIdx
                = graph.coordinates.addElement(ForLoop(accumulatorCoordSize, one));
            auto reverseForAccumIdx
                = graph.coordinates.addElement(ForLoop(accumulatorCoordSize, one));

            Log::debug(
                "  forward ForLoops: tile {} accum {}", forwardForTileIdx, forwardForAccumIdx);
            Log::debug(
                "  reverse ForLoops: tile {} accum {}", reverseForTileIdx, reverseForAccumIdx);

            auto forAccumIncr = *only(graph.coordinates.getInputNodeIndices(
                m_accumulatorCoord, CG::isEdge<CG::DataFlow>));

            graph.coordinates.addElement(
                Split(), {forwardTilesByWGs}, {forwardForTileIdx, forwardForAccumIdx});
            graph.coordinates.addElement(
                Join(), {reverseForTileIdx, reverseForAccumIdx}, {reverseTilesByWGs});

            //
            // Create tile loop
            //
            auto wgTilesOuterExpr    = DF(forTileIncr) < m_numTilesPerWG;
            auto totalTilesOuterExpr = (m_numTilesPerWG * wgExpr + DF(forTileIncr)) < numTotalTiles;
            auto incrementOuterExpr  = DF(forTileIncr) + DF(forAccumIncr);

            auto forTileOp = conditionalFor(graph,
                                            forTileIncr,
                                            wgTilesOuterExpr && totalTilesOuterExpr,
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
                = (m_numTilesPerWG * wgExpr + DF(forTileIncr)) / accumulatorCoordSize;
            currentNonAccumTileExpr = Expression::fastDivision(currentNonAccumTileExpr, m_context);

            auto assignNonAccumTile
                = graph.control.addElement(Assign{Register::Type::Scalar, currentNonAccumTileExpr});
            graph.mapper.connect(assignNonAccumTile, currentNonAccumTileTag, NaryArgument::DEST);

            // TODO: Just before the assignNonAccumTile, need to do:
            //
            // bool sendPartialTile = k > 0;
            //

            auto differentAccumTileInnerExpr
                = ((m_numTilesPerWG * wgExpr + DF(forTileIncr) + DF(forAccumIncr))
                   / accumulatorCoordSize)
                  == DF(currentNonAccumTileTag);
            differentAccumTileInnerExpr
                = Expression::fastDivision(differentAccumTileInnerExpr, m_context);
            auto wgTilesInnerExpr = (DF(forTileIncr) + DF(forAccumIncr)) < m_numTilesPerWG;
            auto totalTilesInnerExpr
                = (m_numTilesPerWG * wgExpr + DF(forTileIncr) + DF(forAccumIncr)) < numTotalTiles;
            auto incrementInnerExpr = DF(forAccumIncr) + one;

            auto forAccumOp = *graph.control.get<ForLoopOp>(m_accumulatorLoopOp);
            forAccumOp.condition
                = differentAccumTileInnerExpr && wgTilesInnerExpr && totalTilesInnerExpr;
            graph.control.setElement(m_accumulatorLoopOp, forAccumOp);

            // TODO After the forAccumOp, need to do:
            //
            // bool receivePartialTile = k < numTilesK - 1;
            // if(sendPartialTile)
            // {
            //     // store partial tile to buffer
            //     // store flag
            // }
            // if(receivePartialTile)
            // {
            //    while(flag == 0);
            //    // load partial tile from buffer
            //    // accumulate into my tile from for-K loop
            // }
            //
            // Then remaining stuff operations (load C, multiply by alpha etc)
            //

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

            auto scope                    = graph.control.addElement(Scope());
            auto forwardSetCoordAccumZero = graph.control.addElement(SetCoordinate(zero));
            graph.mapper.connect(forwardSetCoordAccumZero, forwardForAccumIdx, NaryArgument::DEST);
            auto reverseSetCoordAccumZero = graph.control.addElement(SetCoordinate(zero));
            graph.mapper.connect(reverseSetCoordAccumZero, reverseForAccumIdx, NaryArgument::DEST);

            graph.control.addElement(Body(), {scope}, {forwardSetCoordAccumZero});
            graph.control.addElement(
                Body(), {forwardSetCoordAccumZero}, {reverseSetCoordAccumZero});
            graph.control.addElement(Body(), {reverseSetCoordAccumZero}, {defineChain.front()});
            for(int i = 1; i < defineChain.size(); ++i)
            {
                graph.control.addElement(Sequence(), {defineChain[i - 1]}, {defineChain[i]});
            }
            graph.control.addElement(Sequence(), {defineChain.back()}, {forTileOp});
            graph.control.addElement(Body(), {forTileOp}, {assignNonAccumTile});

            //
            // Insert the new Scope inplace of the original topLoopOp.
            //
            auto location = graph.control.getLocation(m_topLoopOp);
            for(auto const& input : location.incoming)
            {
                auto edge   = graph.control.getElement(input);
                auto parent = *only(graph.control.getNeighbours<GD::Upstream>(input));
                graph.control.deleteElement(input);
                graph.control.addElement(edge, {parent}, {scope});
            }
            for(auto const& output : location.outgoing)
            {
                auto edge  = graph.control.getElement(output);
                auto child = *only(graph.control.getNeighbours<GD::Downstream>(output));

                auto maybeSequence = graph.control.get<Sequence>(output);
                if(maybeSequence)
                {
                    graph.control.deleteElement(output);
                    graph.control.addElement(edge, {scope}, {child});
                }
            }

            graph.control.addElement(Sequence(), {assignNonAccumTile}, {m_topLoopOp});
        }

        void AddStreamK::setupArguments()
        {
            // On entry, numWGs is an Expression that either:
            //   1. Pulls a value from a CommandArgument
            //   2. Is a literal (for testing)

            auto numWGsDT   = DataType::UInt32;
            auto numTilesDT = DataType::Int64;

            // Make kernel arguments:
            //
            // numWGs:        Value
            // numTiles0:     Computed on host: M / macM
            // numTiles1:     Computed on host: N / macN
            // numTilesAcc:   Computed on host: K / macK
            // numTilesPerWG: Computed on host: (numTiles0 * numTiles1 * numTilesAcc + numWGs - 1) / numWGs

            // Note that m_numTileArgExprs was filled during staging

            auto numWGsArg
                = AssemblyKernelArgument{"numWGs", numWGsDT, DataDirection::ReadOnly, m_numWGs};

            std::vector<AssemblyKernelArgument> numTileArgs;
            for(auto d : m_dimensionIndices)
            {
                numTileArgs.push_back(AssemblyKernelArgument{concatenate("numTiles", d),
                                                             numTilesDT,
                                                             DataDirection::ReadOnly,
                                                             m_numTileArgExprs[d]});
            }
            numTileArgs.push_back(AssemblyKernelArgument{
                "numTilesAcc", numTilesDT, DataDirection::ReadOnly, m_numTileArgExprs.back()});

            auto numTilesPerWGArgExpr = m_numTileArgExprs.back();
            for(auto d : m_dimensionIndices)
            {
                numTilesPerWGArgExpr = numTilesPerWGArgExpr * m_numTileArgExprs[d];
            }
            numTilesPerWGArgExpr
                = (numTilesPerWGArgExpr + m_numWGs - Expression::literal(1u)) / m_numWGs;

            auto numTilesPerWGArg = AssemblyKernelArgument{
                "numTilesPerWG", numTilesDT, DataDirection::ReadOnly, numTilesPerWGArgExpr};

            // Make expressions that reference the KernelArguments.
            // These expression are used throughout the transform.
            auto makeArgExpr = [](AssemblyKernelArgument arg) {
                return std::make_shared<Expression::Expression>(
                    std::make_shared<AssemblyKernelArgument>(arg));
            };
            for(auto arg : numTileArgs)
            {
                m_numTiles.push_back(makeArgExpr(arg));
            }
            m_numTilesPerWG = makeArgExpr(numTilesPerWGArg);

            // Add arguments to the kernel
            auto k = m_context->kernel();
            k->addArgument(numWGsArg);
            for(auto arg : numTileArgs)
            {
                k->addArgument(arg);
            }
            k->addArgument(numTilesPerWGArg);

            // On exit, numWGs references the KernelArgument.
            m_numWGs = makeArgExpr(numWGsArg);
        }

        KernelGraph AddStreamK::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddStreamK");

            auto makeFindLoopPredicate = [&](std::string loopName) -> std::function<bool(int)> {
                auto findLoopPredicate = [&](int tag) -> bool {
                    auto maybeForLoop = original.control.get<ForLoopOp>(tag);
                    if(!maybeForLoop)
                        return false;
                    if(maybeForLoop->loopName == loopName)
                        return true;
                    return false;
                };
                return findLoopPredicate;
            };

            // Make sure we can find the loop locations
            auto maybeTopLoopOp
                = only(original.control.findElements(makeFindLoopPredicate(m_topLoop)));
            if(!maybeTopLoopOp)
            {
                rocRoller::Log::warn("Unable to find ForLoop '{}' during AddStreamK pass.  "
                                     "AddStreamK transform skipped.",
                                     m_topLoop);
                return original;
            }
            m_topLoopOp = *maybeTopLoopOp;

            // Make sure we can find the top-for-loop location
            auto maybeAccumLoopOp
                = only(original.control.findElements(makeFindLoopPredicate(m_accumulatorLoop)));
            if(!maybeAccumLoopOp)
            {
                rocRoller::Log::warn("Unable to find ForLoop '{}' during AddStreamK pass.  "
                                     "AddStreamK transform skipped.",
                                     m_accumulatorLoop);
                return original;
            }
            m_accumulatorLoopOp = *maybeAccumLoopOp;

            auto graph = original;
            stage(graph);
            setupArguments();
            commit(graph);

            return graph;
        }
    }
}
