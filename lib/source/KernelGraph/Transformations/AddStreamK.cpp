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
 * Each WG needs to iterate over its portion of the flattened global
 * tile-space.
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
 *
 * In the "two-tile" version, we unevenly split (sunder) the global
 * tile space into two partitions: the "streaming" partition and the
 * "data-parallel" partition.
 *
 * In the data-parallel partition, we want each WG to process whole
 * output tiles so that: synchronization between WGs is not necessary,
 * and cache locality may be improved.  Processing whole output tiles
 * implies that the number of global tiles processed in the
 * data-parallel partition must be a multiple of WG * K.
 *
 * To minimize quantization inefficiency over the WGs in the streaming
 * partition, we want each WG to process at least one output tile, but
 * no more than two.
 *
 * In the data-parallel section, each WG will process
 *
 *   N_DP = (((M * N * K) // (WG * K) - 1) * WG * K
 *
 * tiles; where // is "floor division".  Note:
 *
 *   1. This a multiple of WG * K (no synchronization)
 *   2. The "- 1" pushes enough tiles over to the streaming partition
 *      to minimize quantization inefficiency over the WGs.
 *
 * Simplifying, we obtain
 *
 *   N_DP = ((M * N * K) // (WG * K) - 1) * WG * K
 *        = ((M * N) // WG - 1) * WG * K
 *
 * This means that
 *
 *   N_SK = M * N * K - N_DP
 *        = K * (M * N - ((M * N) // WG - 1) * WG)
 *        = K * (M * N - ((M * N) // WG * WG - WG))
 *        = K * (M * N - (M * N) // WG * WG + WG)
 *        = K * ((M * N) % WG + WG)
 *
 * tiles will be processed in the streaming partition.
 */

#include <map>
#include <unordered_set>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddStreamK.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        namespace CG         = rocRoller::KernelGraph::CoordinateGraph;

        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;
        using namespace Register;

        using GD = rocRoller::Graph::Direction;

        /*
         * Transform info.
         */

        /// Basic info about the incoming loops that need to be transformed.
        struct LoopInfo
        {
            /// Sub-dimensions of dangling `MacroTileNumber`s that should be included in the streaming construct.
            std::vector<int> dimensionIndices;
            /// Control tag of incoming top-loop
            int topLoopOp;
            /// Control tag of incoming accumulator-loop
            int accumulatorLoopOp;
            /// Control tag of XLoop
            int xLoop = -1;
            /// Control tag of YLoop
            int yLoop = -1;
            /// Length of XLoop
            unsigned int xLoopSize = 1;
            /// Length of YLoop
            unsigned int yLoopSize = 1;
        };

        /// Info about accumulator.
        struct AccumulatorInfo
        {
            /// Coordinate dimension of the accumulator-loop
            int accumulatorCoord;
            /// Coordinate dimension of tile into which the accumulator-loop accumulates
            int accumulatorTile;
            /// DataType of accumulator tile
            VariableType accumulatorVarType;
            /// Set of control tags, after the accumulator-loop, that use the accumulator-tile.
            std::unordered_set<int> usesAccumulatorTile;

            /// Staged MacroTileNumber coordinates
            //
            // Mapping: dimension -> set of MacroTileNumber coordinates
            std::map<int, std::unordered_set<int>> tileNumberCoords;
        };

        /// Info about the SK and DP tile spaces.
        struct StreamKInfo
        {
            int localTileSpaceSK; /// Streaming tile space
            int localTileSpaceDP; /// Data-parallel tile space
            int selector; /// SK vs DP selector (for Sunder)
        };

        /// Info about the "number of tiles" kernel arguments.
        struct ArgumentInfo
        {
            // Kernel arguments
            std::vector<Expression::ExpressionPtr> numTiles, numTileArgExprs;
            Expression::ExpressionPtr              numSKTiles, numDPTiles;
            Expression::ExpressionPtr              numWGs, numSKTilesPerWG, numDPTilesPerWG;
        };

        //
        // Helpers
        //

        auto makeFindLoopPredicate(KernelGraph const& graph, std::string const& loopName)
        {
            auto findLoopPredicate = [&](int tag) -> bool {
                auto maybeForLoop = graph.control.get<ForLoopOp>(tag);
                if(!maybeForLoop)
                    return false;
                if(maybeForLoop->loopName == loopName)
                    return true;
                return false;
            };
            return findLoopPredicate;
        }

        /**
         * Create Assign node to add partially accumulated tiles.
         *
         * Returns tag of new Assign node.
         */
        int addFixup(KernelGraph& graph,
                     int          accumMacTileTag,
                     int          partialMacTileTag,
                     int          destMacTileTag,
                     DataType     dataType,
                     uint         numVGPRs)
        {
            auto lhsExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{accumMacTileTag, Register::Type::Vector, DataType::None});
            auto rhsExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{partialMacTileTag, Register::Type::Vector, DataType::None});

            auto addExpr = lhsExpr + rhsExpr;
            auto fixupTag
                = graph.control.addElement(Assign{Register::Type::Vector, addExpr, numVGPRs});

            graph.coordinates.addElement(DataFlow(), {partialMacTileTag}, {destMacTileTag});

            graph.mapper.connect(fixupTag, destMacTileTag, NaryArgument::DEST);

            return fixupTag;
        }

        /*
         * Scratch tile info.
         */

        struct ScratchInfo
        {
            int store; //< Source internal tile for store operation
            int load; //< Destination internal tile for load operation
            int tile; //< Scratch tile
            int setPlusOne; //< SetCoordinate operation; must be inserted above the Load
        };

        struct SendInfo
        {
            int preWaitZero; //< WaitZero before conditional
            int sendCond; //< Conditional send-block
            int assignSendBoolSGPR; //< Assign operator to hold condition boolean
            int sendBoolSGPR; //< SGPR into which condition boolean is stored
        };

        struct RecvInfo
        {
            int preWaitZero;
            int receiveCond;
            int setPlusOne;
        };

        /**
         * Create coordinate transforms for exchanging partially
         * accumulated tiles through scratch space.
         */
        ScratchInfo loadStoreMacroTileSCRATCH(KernelGraph&                     graph,
                                              std::vector<DeferredConnection>& storeConnections,
                                              std::vector<DeferredConnection>& loadConnections,
                                              int                              macTileTag,
                                              VariableType                     varType,
                                              LoopInfo const&                  loopInfo,
                                              ArgumentInfo const&              argInfo,
                                              CommandParametersPtr             params,
                                              ContextPtr                       context)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto numWorkgroupsX = argInfo.numWGs;
            auto numWorkgroupsY = literal(1u);

            auto sizeX = simplify(numWorkgroupsX * literal(static_cast<uint>(macTile.sizes[0])));
            auto sizeY = simplify(numWorkgroupsY * literal(static_cast<uint>(macTile.sizes[1])));

            auto strideX = sizeY;
            auto strideY = literal(1u);

            auto globalScratch    = newScratchCoordinate(simplify(sizeX * sizeY), varType, context);
            auto globalScratchTag = graph.coordinates.addElement(globalScratch);

            std::vector<unsigned int> jammedSizes = {loopInfo.xLoopSize, loopInfo.yLoopSize};

            // Store
            auto jammedStoreScratchTileTag = createInternalTile(
                graph, varType, macTileTag, jammedSizes, false, params, context);
            graph.coordinates.addElement(View(), {jammedStoreScratchTileTag}, {macTileTag});

            auto jammedStoreScratchTile
                = *graph.coordinates.get<MacroTile>(jammedStoreScratchTileTag);
            jammedStoreScratchTile.layoutType = LayoutType::SCRATCH;
            graph.coordinates.setElement(jammedStoreScratchTileTag, jammedStoreScratchTile);

            std::vector<int> storeSubDimensions
                = {graph.coordinates.addElement(SubDimension(0, sizeX, strideX)),
                   graph.coordinates.addElement(SubDimension(1, sizeY, strideY))};
            graph.coordinates.addElement(
                Join(), storeSubDimensions, std::vector<int>{globalScratchTag});

            {
                auto [nMacX, iMacX, iMacY] = addStore1DMacroTileCT(
                    graph, storeConnections, macTileTag, storeSubDimensions);

                addStoreThreadTileCT(graph,
                                     storeConnections,
                                     jammedStoreScratchTileTag,
                                     iMacX,
                                     iMacY,
                                     context->kernel()->workgroupSize(),
                                     jammedSizes,
                                     true);

                graph.coordinates.addElement(
                    DataFlow(), {jammedStoreScratchTileTag}, {globalScratchTag});
            }

            storeConnections.push_back(DC<MacroTile>(jammedStoreScratchTileTag));
            storeConnections.push_back(DC<User>(globalScratchTag));

            // Load
            auto jammedLoadScratchTileTag = createInternalTile(
                graph, varType, macTileTag, jammedSizes, false, params, context);
            auto jammedLoadScratchTile
                = *graph.coordinates.get<MacroTile>(jammedLoadScratchTileTag);
            jammedLoadScratchTile.layoutType = LayoutType::SCRATCH;
            graph.coordinates.setElement(jammedLoadScratchTileTag, jammedLoadScratchTile);

            auto loadScratchTileTag
                = createInternalTile(graph, varType, macTileTag, params, context);
            auto loadScratchTile       = *graph.coordinates.get<MacroTile>(loadScratchTileTag);
            loadScratchTile.layoutType = LayoutType::SCRATCH;
            graph.coordinates.setElement(loadScratchTileTag, loadScratchTile);

            graph.coordinates.addElement(View(), {jammedLoadScratchTileTag}, {loadScratchTileTag});

            std::vector<int> loadSubDimensions
                = {graph.coordinates.addElement(SubDimension(0, sizeX, strideX)),
                   graph.coordinates.addElement(SubDimension(1, sizeY, strideY))};
            graph.coordinates.addElement(
                Split(), std::vector<int>{globalScratchTag}, loadSubDimensions);

            {
                auto [nMacX, iMacX, iMacY]
                    = addLoad1DMacroTileCT(graph, loadConnections, macTileTag, loadSubDimensions);

                addLoadThreadTileCT(graph,
                                    loadConnections,
                                    jammedLoadScratchTileTag,
                                    iMacX,
                                    iMacY,
                                    context->kernel()->workgroupSize(),
                                    jammedSizes,
                                    true);

                graph.coordinates.addElement(DataFlow(), {loadScratchTileTag}, {globalScratchTag});
            }

            // Find the dangling MacroTileNumber and add one to it...
            auto danglingMacroTileNumber = [&graph](int tag) -> bool {
                auto maybeTileNumber = graph.coordinates.get<MacroTileNumber>(tag);
                if(!maybeTileNumber)
                    return false;
                if(!empty(graph.coordinates.getNeighbours<GD::Downstream>(tag)))
                    return false;
                if(maybeTileNumber->dim != 0)
                    return false;
                return true;
            };

            auto nextTileNumTag
                = *graph.coordinates.findNodes(globalScratchTag, danglingMacroTileNumber)
                       .take(1)
                       .only();

            auto one           = Expression::literal(1u);
            auto plusOneTag    = graph.coordinates.addElement(Linear(one, one));
            auto setPlusOneTag = graph.control.addElement(SetCoordinate(one));
            graph.mapper.connect<Linear>(setPlusOneTag, plusOneTag);

            auto tileNumTag = graph.coordinates.addElement(
                *graph.coordinates.get<MacroTileNumber>(nextTileNumTag)); // Copy existing
            graph.coordinates.addElement(Split(), {nextTileNumTag}, {tileNumTag, plusOneTag});

            loadConnections.push_back(DC<MacroTile>(loadScratchTileTag));
            loadConnections.push_back(DC<User>(globalScratchTag));

            return {jammedStoreScratchTileTag, loadScratchTileTag, globalScratchTag, setPlusOneTag};
        }

        /**
         * Create send-tile block, which is roughly:
         *
         *     WaitZero()
         *     if sendTileExpr:
         *       StoreTile()
         *       WaitZero()
         *       Barrier()
         *       flag = Assign(SGPR, 1u);
         *       StoreSGPR(flag)
         *       WaitZero()
         */
        SendInfo sendTile(KernelGraph&                           graph,
                          ExpressionPtr                          tileExpr,
                          std::vector<DeferredConnection> const& storeConnections,
                          int                                    flagsScratchTag,
                          DataType                               scratchDataType,
                          VariableType                           numTilesVarType,
                          LoopInfo const&                        loopInfo,
                          ContextPtr                             context)
        {
            auto one  = Expression::literal(1u);
            auto zero = Expression::literal(0, numTilesVarType);
            auto DF   = [numTilesVarType](int tag) {
                return std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{tag, Register::Type::Scalar, numTilesVarType});
            };

            auto workgroup = graph.coordinates.addElement(Workgroup(0));
            graph.coordinates.addElement(PassThrough(), {workgroup}, {flagsScratchTag});

            // Store tile
            int forX = cloneForLoop(graph, loopInfo.xLoop);
            int forY = cloneForLoop(graph, loopInfo.yLoop);

            auto store = StoreTiled(scratchDataType);

            // TODO: Improve setting of arch-specific buffer options
            if(!(context->targetArchitecture().target().isCDNA1GPU()
                 || context->targetArchitecture().target().isCDNA2GPU()))
            {
                store.bufOpts = {.sc1 = true};
            }

            auto storeTileTag = graph.control.addElement(store);
            for(auto const& c : storeConnections)
                graph.mapper.connect(storeTileTag, c.coordinate, c.connectionSpec);

            graph.control.addElement(Body(), {forX}, {forY});
            graph.control.addElement(Body(), {forY}, {storeTileTag});

            auto sendTileRegister = graph.coordinates.addElement(VGPR());
            auto sendTileAssign
                = graph.control.addElement(Assign{Register::Type::Scalar, tileExpr});
            graph.mapper.connect(sendTileAssign, sendTileRegister, NaryArgument::DEST);

            auto jammedX = graph.mapper.get<JammedWaveTileNumber>(storeTileTag, 0);
            if(jammedX != -1)
                graph.coordinates.addElement(
                    PassThrough(), {graph.mapper.get<ForLoop>(forX)}, {jammedX});
            auto jammedY = graph.mapper.get<JammedWaveTileNumber>(storeTileTag, 1);
            if(jammedY != -1)
                graph.coordinates.addElement(
                    PassThrough(), {graph.mapper.get<ForLoop>(forY)}, {jammedY});

            // Yoda-expression is a workaround for an issue in
            // GreaterThan.  A more natural condtion would be:
            //   DF(sendTileRegister) > zero.
            auto sendTileTag
                = graph.control.addElement(ConditionalOp{zero < DF(sendTileRegister), "Send Tile"});
            auto barrierTag  = graph.control.addElement(Barrier());
            auto waitZeroTag = graph.control.addElement(WaitZero());

            // Store flag
            auto flagRegister = graph.coordinates.addElement(VGPR());

            auto assignFlagTag = graph.control.addElement(Assign{Register::Type::Scalar, one});
            graph.mapper.connect(assignFlagTag, flagRegister, NaryArgument::DEST);

            // TODO: Improve setting of arch-specific buffer options
            BufferInstructionOptions bufOpts;
            if(context->targetArchitecture().target().isCDNA1GPU()
               || context->targetArchitecture().target().isCDNA2GPU())
            {
                bufOpts = {.glc = true};
            }
            else
            {
                bufOpts = {.sc1 = true};
            }
            auto storeFlagTag = graph.control.addElement(StoreSGPR(DataType::UInt32, bufOpts));
            graph.mapper.connect<User>(storeFlagTag, flagsScratchTag);
            graph.mapper.connect<VGPR>(storeFlagTag, flagRegister);

            // Add to control
            auto preWaitZeroTag  = graph.control.addElement(WaitZero());
            auto postWaitZeroTag = graph.control.addElement(WaitZero());

            graph.control.addElement(Sequence(), {preWaitZeroTag}, {sendTileTag});
            graph.control.addElement(Body(), {sendTileTag}, {forX});
            graph.control.chain<Sequence>(
                forX, waitZeroTag, barrierTag, assignFlagTag, storeFlagTag, postWaitZeroTag);

            return {preWaitZeroTag, sendTileTag, sendTileAssign, sendTileRegister};
        }

        /**
         * Create send-tile block, which is roughly:
         *
         *     WaitZero()
         *     fullyAccumulatedTile = Assign(localPartiallyAccumulatedTile)
         *     if receiveTileExpr:
         *       if WG + 1 < numScratch:
         *         do:
         *           LoadSGPR(flag)
         *         while flag == 0
         *         partiallyAccumulatedTile = LoadTiled()
         *         fullyAccumulatedTile = Assign(fullyAccumulatedTile + partiallyAccumulatedTile)
         *         WaitZero()
         *
         * Note this also update all subsequent references to
         * localPartiallyAccumulatedTile to fullyAccumulatedTile.
         */
        RecvInfo receiveTile(KernelGraph&                           graph,
                             ExpressionPtr                          receiveTileExpr,
                             int                                    scratchTileTag,
                             std::vector<DeferredConnection> const& loadConnections,
                             int                                    flagsScratchTag,
                             int                                    accumulatorTileTag,
                             std::unordered_set<int> const&         usesAccumulatorTile,
                             DataType                               dataType,
                             std::vector<int> const&                epilogueOperations,
                             LoopInfo const&                        loopInfo,
                             ArgumentInfo const&                    argInfo,
                             CommandParametersPtr                   params,
                             ContextPtr                             context)
        {
            auto DF = [](int tag) {
                return std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{tag, Register::Type::Scalar, DataType::UInt32});
            };

            auto one  = Expression::literal(1u);
            auto zero = Expression::literal(0u);

            auto workgroup = graph.coordinates.addElement(Workgroup(0));

            // Read tile
            auto receiveTileTag
                = graph.control.addElement(ConditionalOp{receiveTileExpr, "Receive Tile"});

            // Read flag
            auto plusOneTag    = graph.coordinates.addElement(Linear(one, one));
            auto setPlusOneTag = graph.control.addElement(SetCoordinate(one));
            graph.mapper.connect<Linear>(setPlusOneTag, plusOneTag);

            auto nextWorkgroupTag = graph.coordinates.addElement(Linear(nullptr, one));
            graph.coordinates.addElement(Split(), {nextWorkgroupTag}, {workgroup, plusOneTag});

            // TODO: Improve setting of arch-specific buffer options
            BufferInstructionOptions bufOpts;
            if(context->targetArchitecture().target().isCDNA1GPU()
               || context->targetArchitecture().target().isCDNA2GPU())
            {
                bufOpts = {.glc = true};
            }
            else
            {
                bufOpts = {.sc1 = true};
            }
            auto flagRegister = graph.coordinates.addElement(VGPR());
            auto loadFlagTag  = graph.control.addElement(LoadSGPR(DataType::UInt32, bufOpts));

            auto numScratch     = argInfo.numWGs;
            auto boundsCheckTag = graph.control.addElement(
                ConditionalOp{(DF(workgroup) + one < numScratch), "Bounds Check"});

            graph.mapper.connect<User>(loadFlagTag, flagsScratchTag);
            graph.mapper.connect<VGPR>(loadFlagTag, flagRegister);
            graph.coordinates.addElement(PassThrough(), {flagsScratchTag}, {nextWorkgroupTag});

            auto doWhileTag = graph.control.addElement(
                DoWhileOp{(DF(flagRegister) == zero), "Global sync spin loop"});

            auto accumulatorTile = graph.coordinates.get<MacroTile>(accumulatorTileTag);
            uint numRegisters    = accumulatorTile->elements()
                                / (product(context->kernel()->workgroupSize()) * loopInfo.xLoopSize
                                   * loopInfo.yLoopSize);
            auto fullyAccumulatedTileTag = graph.coordinates.addElement(MacroTile());

            // Read tile
            int loadAddForX = cloneForLoop(graph, loopInfo.xLoop);
            int loadAddForY = cloneForLoop(graph, loopInfo.yLoop);

            auto loadTileTag = graph.control.addElement(LoadTiled(dataType));
            for(auto const& c : loadConnections)
                graph.mapper.connect(loadTileTag, c.coordinate, c.connectionSpec);

            graph.control.addElement(Body(), {loadAddForX}, {loadAddForY});
            graph.control.addElement(Body(), {loadAddForY}, {loadTileTag});

            auto jammedX = graph.mapper.get<JammedWaveTileNumber>(loadTileTag, 0);
            if(jammedX != -1)
                graph.coordinates.addElement(
                    PassThrough(), {jammedX}, {graph.mapper.get<ForLoop>(loadAddForX)});
            auto jammedY = graph.mapper.get<JammedWaveTileNumber>(loadTileTag, 1);
            if(jammedY != -1)
                graph.coordinates.addElement(
                    PassThrough(), {jammedY}, {graph.mapper.get<ForLoop>(loadAddForY)});

            auto fixupTag = addFixup(graph,
                                     accumulatorTileTag,
                                     scratchTileTag,
                                     fullyAccumulatedTileTag,
                                     dataType,
                                     numRegisters);

            graph.control.addElement(Sequence(), {loadTileTag}, {fixupTag});

            // Attach epilogue operations after the fixup
            auto epilogueYLoop = only(graph.control.findNodes(
                epilogueOperations, makeFindLoopPredicate(graph, rocRoller::YLOOP)));
            AssertFatal(epilogueYLoop, "Must have exactly one Y loop in the epilogue");
            auto reindexer       = std::make_shared<GraphReindexer>();
            auto newEpilogueBody = duplicateControlNodes(
                graph,
                reindexer,
                graph.control.getOutputNodeIndices<Body>(*epilogueYLoop).to<std::vector>(),
                [](int x) { return false; });

            // Replace accumulatorTileTag with fullyAccumulatedTileTag in the epilogue
            {
                GraphReindexer expressionReindexer;
                expressionReindexer.coordinates.emplace(accumulatorTileTag,
                                                        fullyAccumulatedTileTag);
                for(auto const& node : graph.control.depthFirstVisit(newEpilogueBody))
                {
                    reindexExpressions(graph, node, expressionReindexer);
                }
            }

            for(auto const& epilogueBody : newEpilogueBody)
            {
                graph.control.addElement(Sequence(), {fixupTag}, {epilogueBody});
            }

            // Add to control
            auto preWaitZeroTag  = graph.control.addElement(WaitZero());
            auto postWaitZeroTag = graph.control.addElement(WaitZero());

            graph.control.chain<Sequence>(preWaitZeroTag, receiveTileTag);

            graph.control.addElement(Body(), {receiveTileTag}, {boundsCheckTag});
            graph.control.addElement(Body(), {boundsCheckTag}, {doWhileTag});
            graph.control.addElement(Body(), {doWhileTag}, {loadFlagTag});

            graph.control.chain<Sequence>(doWhileTag, loadAddForX, postWaitZeroTag);

            return {preWaitZeroTag, receiveTileTag, setPlusOneTag};
        }

        /**
         * Creates a ForLoop with a non-constant increment.
         */
        int conditionalFor(KernelGraph&       graph,
                           int                incrementCoord,
                           ExpressionPtr      conditionExpr,
                           ExpressionPtr      incrementExpr,
                           std::string const& loopName,
                           VariableType       varType)
        {
            auto forLoop = graph.control.addElement(ForLoopOp{conditionExpr, loopName});
            graph.mapper.connect(forLoop, incrementCoord, NaryArgument::DEST);

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

        /**
         * Add chain of SetCoordinate and Assign nodes to initialize coordinates.
         */
        int initializeCoordinates(KernelGraph&               graph,
                                  int                        top,
                                  std::initializer_list<int> setCoordinates,
                                  ExpressionPtr              setValue,
                                  std::initializer_list<int> assigns,
                                  ExpressionPtr              assignValue)

        {
            std::vector<int> setChain;
            for(auto coordinate : setCoordinates)
            {
                auto setCoordinate = graph.control.addElement(SetCoordinate(setValue));
                graph.mapper.connect(setCoordinate, coordinate, NaryArgument::DEST);
                setChain.push_back(setCoordinate);
            }

            std::vector<int> assignChain;
            for(auto tag : assigns)
            {
                auto def = graph.control.addElement(Assign{Register::Type::Scalar, assignValue});
                graph.mapper.connect(def, tag, NaryArgument::DEST);
                assignChain.push_back(def);
            }

            graph.control.addElement(Body(), {top}, {setChain.front()});
            for(int i = 1; i < setChain.size(); ++i)
            {
                graph.control.addElement(Body(), {setChain[i - 1]}, {setChain[i]});
            }

            graph.control.addElement(Body(), {setChain.back()}, {assignChain.front()});
            for(int i = 1; i < assignChain.size(); ++i)
            {
                graph.control.addElement(Sequence(), {assignChain[i - 1]}, {assignChain[i]});
            }

            auto nop = graph.control.addElement(NOP());
            graph.control.addElement(Sequence(), {assignChain.back()}, {nop});

            return nop;
        }

        //
        // AddStreamK methods
        //

        std::string AddStreamK::name() const
        {
            return concatenate("AddStreamK");
        }

        AddStreamK::AddStreamK(std::vector<int> const& dims,
                               std::string const&      topLoop,
                               std::string const&      accumulatorLoop,
                               bool                    twoTile,
                               ExpressionPtr           numWGs,
                               CommandParametersPtr    params,
                               ContextPtr              context)
            : m_dimensionIndices(dims)
            , m_topLoop(topLoop)
            , m_accumulatorLoop(accumulatorLoop)
            , m_twoTile(twoTile)
            , m_numWGs(numWGs)
            , m_params(params)
            , m_context(context)
        {
        }

        /**
         * Add coordinate transforms that go between:
         * 1. the global tile space and the SK tile space,
         * 2. the global tile space and the DP tile space, and
         * 3. the SK/DP tile spaces and the various MacroTileNumber coordinates.
         */
        StreamKInfo addTileSpaceCT(KernelGraph&           graph,
                                   bool                   forward,
                                   LoopInfo const&        loopInfo,
                                   AccumulatorInfo const& accumInfo,
                                   ArgumentInfo const&    argInfo,
                                   ExpressionPtr          numTotalTiles)
        {
            // Create forward/reverse tile-numbers for each dimension
            // and attach to all staged tile-number coordinates
            std::vector<int> tileNumbers;
            for(auto d : loopInfo.dimensionIndices)
            {
                auto tileNumber = graph.coordinates.addElement(
                    MacroTileNumber(d, argInfo.numTiles[d], nullptr));

                for(auto tileNumTag : accumInfo.tileNumberCoords.at(d))
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
            tileNumbers.push_back(accumInfo.accumulatorCoord);

            // Create foward/reverse flattened tile-space
            auto tileSpace   = graph.coordinates.addElement(Linear(numTotalTiles, nullptr));
            auto tileSpaceSK = graph.coordinates.addElement(Linear(argInfo.numSKTiles, nullptr));
            auto tileSpaceDP = graph.coordinates.addElement(Linear(argInfo.numDPTiles, nullptr));
            auto selector    = graph.coordinates.addElement(Linear(literal(2u), nullptr));

            if(forward)
            {
                graph.coordinates.addElement(Flatten(), tileNumbers, std::vector<int>{tileSpace});
                graph.coordinates.addElement(
                    Sunder(), {tileSpace}, {tileSpaceSK, tileSpaceDP, selector});
            }
            else
            {
                graph.coordinates.addElement(
                    Sunder(), {tileSpaceSK, tileSpaceDP, selector}, {tileSpace});
                graph.coordinates.addElement(Tile(), std::vector<int>{tileSpace}, tileNumbers);
            }

            auto WGs       = graph.coordinates.addElement(Linear(argInfo.numWGs, nullptr));
            auto workgroup = graph.coordinates.addElement(Workgroup());

            auto localTileSpaceSK
                = graph.coordinates.addElement(Linear(argInfo.numSKTilesPerWG, nullptr));
            auto localTileSpaceDP
                = graph.coordinates.addElement(Linear(argInfo.numDPTilesPerWG, nullptr));
            if(forward)
            {
                graph.coordinates.addElement(PassThrough(), {WGs}, {workgroup});
                graph.coordinates.addElement(Tile(), {tileSpaceSK}, {WGs, localTileSpaceSK});
                graph.coordinates.addElement(Tile(), {tileSpaceDP}, {WGs, localTileSpaceDP});
            }
            else
            {
                graph.coordinates.addElement(PassThrough(), {workgroup}, {WGs});
                graph.coordinates.addElement(Flatten(), {WGs, localTileSpaceSK}, {tileSpaceSK});
                graph.coordinates.addElement(Flatten(), {WGs, localTileSpaceDP}, {tileSpaceDP});
            }

            return {localTileSpaceSK, localTileSpaceDP, selector};
        }

        //
        // Stage
        //
        // Look for all leaf MacroTileNumbers with matching
        // sub-dimension.
        //
        // Matches are: tile->dim in m_dimensions.
        //
        AccumulatorInfo stage(KernelGraph const& graph, LoopInfo const& loopInfo)
        {
            AccumulatorInfo accumInfo;

            accumInfo.accumulatorCoord = getForLoopCoords(loopInfo.accumulatorLoopOp, graph).first;
            Log::debug("  accumulator loop coord: {}", accumInfo.accumulatorCoord);

            // Find accumulator tile: look above accumulator-loop for an assign statement
            accumInfo.accumulatorTile = -1;
            auto maybeAccumulatorInit = only(graph.control.findElements([&](int tag) -> bool {
                auto maybeAssign = graph.control.get<Assign>(tag);
                if(!maybeAssign)
                    return false;
                auto nextTag = only(graph.control.getOutputNodeIndices<Sequence>(tag));
                return nextTag.value_or(-1) == loopInfo.accumulatorLoopOp;
            }));
            if(maybeAccumulatorInit)
            {
                auto init                    = graph.control.get<Assign>(*maybeAccumulatorInit);
                accumInfo.accumulatorVarType = resultVariableType(init->expression);

                auto dst            = graph.mapper.get(*maybeAccumulatorInit, NaryArgument::DEST);
                auto maybeAccumTile = graph.coordinates.get<MacroTile>(dst);
                if(maybeAccumTile)
                    accumInfo.accumulatorTile = dst;
            }

            ControlFlowRWTracer tracer(graph);
            for(auto m : tracer.coordinatesReadWrite(accumInfo.accumulatorTile))
            {
                if(graph.control.compareNodes(loopInfo.accumulatorLoopOp, m.control)
                   == NodeOrdering::LeftFirst)
                {
                    accumInfo.usesAccumulatorTile.insert(m.control);
                }
            }

            // Find all dangling MacroTileNumber dimensions associated
            // with the requested dimensions
            for(auto dimension : loopInfo.dimensionIndices)
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

                accumInfo.tileNumberCoords[dimension]
                    = graph.coordinates.findElements(danglingTileNumberPredicate)
                          .to<std::unordered_set>();
            }

            return accumInfo;
        }

        //
        // Commit
        //
        void commit(KernelGraph&           graph,
                    bool                   twoTile,
                    LoopInfo const&        loopInfo,
                    AccumulatorInfo const& accumInfo,
                    ArgumentInfo const&    argInfo,
                    CommandParametersPtr   params,
                    ContextPtr             context)
        {
            //
            // Find the epilogue operations; detach them.
            //
            // They will be re-attached later.
            //
            std::vector<int> epilogueOperations;
            for(auto tag : graph.control.getNeighbours<GD::Downstream>(loopInfo.topLoopOp))
            {
                auto maybeSequence = graph.control.get<Sequence>(tag);
                if(maybeSequence)
                {
                    epilogueOperations.push_back(
                        *only(graph.control.getNeighbours<GD::Downstream>(tag)));
                    graph.control.deleteElement(tag);
                }
            }

            //
            // Create new Scope and insert it above the top-loop
            //
            auto scope = graph.control.addElement(Scope());
            replaceWith(graph, loopInfo.topLoopOp, scope, false);

            //
            // Compute size of global and local tile-spaces
            //
            auto numAccumTiles = argInfo.numTileArgExprs.back();
            auto numTotalTiles = numAccumTiles;
            for(auto d : loopInfo.dimensionIndices)
                numTotalTiles = numTotalTiles * argInfo.numTiles.at(d);
            numTotalTiles = simplify(numTotalTiles);
            enableDivideBy(numTotalTiles, context);

            auto numTilesVarType = resultType(numTotalTiles).varType;
            auto one             = Expression::literal(1, numTilesVarType);
            auto zero            = Expression::literal(0, numTilesVarType);

            Log::debug("    numAccumTiles: {}", toString(numAccumTiles));
            Log::debug("    numTotalTiles: {}", toString(numTotalTiles));
            Log::debug("          varType: {}", toString(numTilesVarType));

            //
            // Helper
            //
            auto DF = [numTilesVarType](int tag) {
                return std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{tag, Register::Type::Scalar, numTilesVarType});
            };

            auto wg     = graph.coordinates.addElement(Workgroup(0));
            auto wgExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{wg, Register::Type::Scalar, DataType::UInt32});

            //
            // Duplicate operations from the top-loop down for DP section of two-tile StreamK.
            //
            // TODO This seems overly aggressive.  There should be a
            // way of looking at the DataFlow graph to figure out
            // where to stop.
            int dpTopLoop, dpAccumLoop;
            if(twoTile)
            {
                // We disable duplication here so that the fix-up pass
                // uses the correct registers.
                auto reindexer = std::make_shared<GraphReindexer>();
                dpTopLoop      = *only(duplicateControlNodes(
                    graph, reindexer, {loopInfo.topLoopOp}, [](int x) { return true; }));
                dpAccumLoop    = reindexer->control[loopInfo.accumulatorLoopOp];

                // Duplicate the epilogue.  We enable duplication here
                // to help the deallocation pass reduce register
                // pressure (shortens lifetimes)
                auto epilogueDuplicates = duplicateControlNodes(
                    graph, nullptr, epilogueOperations, [](int x) { return false; });
                for(auto const& epilogueDuplicate : epilogueDuplicates)
                    graph.control.addElement(Sequence(), {dpTopLoop}, {epilogueDuplicate});
            }

            //
            // Add forward/reverse tile-space coordinate transforms
            //
            auto forwardInfo
                = addTileSpaceCT(graph, true, loopInfo, accumInfo, argInfo, numTotalTiles);
            auto reverseInfo
                = addTileSpaceCT(graph, false, loopInfo, accumInfo, argInfo, numTotalTiles);

            //
            // Add local-tile and accumulator for-loop dimensions and iterators
            //
            auto forTileIncr = graph.coordinates.addElement(Linear(argInfo.numSKTilesPerWG, one));
            auto forwardForTileIdx
                = graph.coordinates.addElement(ForLoop(argInfo.numSKTilesPerWG, one));
            auto reverseForTileIdx
                = graph.coordinates.addElement(ForLoop(argInfo.numSKTilesPerWG, one));
            auto forwardForAccumIdx = graph.coordinates.addElement(ForLoop(numAccumTiles, one));
            auto reverseForAccumIdx = graph.coordinates.addElement(ForLoop(numAccumTiles, one));

            Log::debug(
                "  forward ForLoops: tile {} accum {}", forwardForTileIdx, forwardForAccumIdx);
            Log::debug(
                "  reverse ForLoops: tile {} accum {}", reverseForTileIdx, reverseForAccumIdx);

            auto forAccumIncr = *only(graph.coordinates.getInputNodeIndices(
                accumInfo.accumulatorCoord, CG::isEdge<CG::DataFlow>));

            graph.coordinates.addElement(
                Split(), {forwardInfo.localTileSpaceSK}, {forwardForTileIdx, forwardForAccumIdx});
            graph.coordinates.addElement(
                Join(), {reverseForTileIdx, reverseForAccumIdx}, {reverseInfo.localTileSpaceSK});

            graph.coordinates.addElement(
                Split(), {forwardInfo.localTileSpaceDP}, {forwardForTileIdx, forwardForAccumIdx});
            graph.coordinates.addElement(
                Join(), {reverseForTileIdx, reverseForAccumIdx}, {reverseInfo.localTileSpaceDP});

            //
            // Create local-tile loop
            //
            auto wgTilesOuterExpr = DF(forTileIncr) < argInfo.numSKTilesPerWG;
            auto skTilesOuterExpr
                = (argInfo.numSKTilesPerWG * wgExpr + DF(forTileIncr)) < argInfo.numSKTiles;
            auto incrementOuterExpr = DF(forTileIncr) + DF(forAccumIncr);

            auto forTileSKOp = conditionalFor(graph,
                                              forTileIncr,
                                              wgTilesOuterExpr && skTilesOuterExpr,
                                              incrementOuterExpr,
                                              "SKStreamTileLoop",
                                              numTilesVarType);

            graph.coordinates.addElement(DataFlow(), {forTileIncr}, {forwardForTileIdx});
            graph.coordinates.addElement(DataFlow(), {forTileIncr}, {reverseForTileIdx});

            //
            // Hijack old accumulator loop.
            //
            // The old accumulator loop was a simple range-based loop.
            // The new accumulator loop streams the accumulator tile
            // index.
            //
            // This is done in two steps:
            //
            // 1. Remove the previous DataFlow edge between the loop
            //    increment and the ForLoop coordinate.
            // 2. Replace the conditions.
            //

            //
            // Remove DataFlow edge from accumulator increment to
            // accumulator ForLoop dimension.  Add new DataFlow edges
            // from accumulator increment to forward and reverse
            // accumulator ForLoop dimensions.
            //
            {
                auto iterator = *only(graph.coordinates.getInputNodeIndices(
                    accumInfo.accumulatorCoord, CG::isEdge<CG::DataFlow>));
                auto dataflow = *only(graph.coordinates.getNeighbours<GD::Downstream>(iterator));
                graph.coordinates.deleteElement(dataflow);
                graph.coordinates.addElement(DataFlow(), {forAccumIncr}, {forwardForAccumIdx});
                graph.coordinates.addElement(DataFlow(), {forAccumIncr}, {reverseForAccumIdx});
            }

            auto nextNonAccumTileTag = graph.coordinates.addElement(Dimension());
            auto nextNonAccumTileExpr
                = ((argInfo.numSKTilesPerWG * wgExpr + DF(forTileIncr)) / numAccumTiles + one)
                  * numAccumTiles;

            auto assignNextNonAccumTile
                = graph.control.addElement(Assign{Register::Type::Scalar, nextNonAccumTileExpr});
            graph.mapper.connect(assignNextNonAccumTile, nextNonAccumTileTag, NaryArgument::DEST);

            auto nextNonAccumTileInnerExpr
                = (argInfo.numSKTilesPerWG * wgExpr + DF(forTileIncr) + DF(forAccumIncr))
                  < DF(nextNonAccumTileTag);
            auto wgTilesInnerExpr = (DF(forTileIncr) + DF(forAccumIncr)) < argInfo.numSKTilesPerWG;
            auto skTilesInnerExpr
                = (argInfo.numSKTilesPerWG * wgExpr + DF(forTileIncr) + DF(forAccumIncr))
                  < argInfo.numSKTiles;
            auto incrementInnerExpr = DF(forAccumIncr) + one;

            auto forAccumOp = *graph.control.get<ForLoopOp>(loopInfo.accumulatorLoopOp);
            forAccumOp.condition
                = nextNonAccumTileInnerExpr && wgTilesInnerExpr && skTilesInnerExpr;
            graph.control.setElement(loopInfo.accumulatorLoopOp, forAccumOp);

            //
            // After the accumulator loop, we might need to send and/or
            // receive partially accumulated tiles.
            //
            ScratchInfo scratchTileInfo;
            SendInfo    sendInfo;
            RecvInfo    receiveInfo;
            int         postAccumulationCond;
            if(accumInfo.accumulatorTile != -1)
            {
                // Create scratch space for flags
                auto flagsScratch = newScratchCoordinate(argInfo.numWGs, DataType::UInt32, context);
                auto flagsScratchTag = graph.coordinates.addElement(flagsScratch);

                // Create scratch space for partially accumulated tiles
                std::vector<DeferredConnection> storeConnections, loadConnections;
                scratchTileInfo = loadStoreMacroTileSCRATCH(graph,
                                                            storeConnections,
                                                            loadConnections,
                                                            accumInfo.accumulatorTile,
                                                            accumInfo.accumulatorVarType,
                                                            loopInfo,
                                                            argInfo,
                                                            params,
                                                            context);

                // Add send
                auto sendTileExpr
                    = (argInfo.numSKTilesPerWG * wgExpr + DF(forTileIncr)) % numAccumTiles;
                sendInfo = sendTile(graph,
                                    sendTileExpr,
                                    storeConnections,
                                    flagsScratchTag,
                                    accumInfo.accumulatorVarType.dataType,
                                    numTilesVarType,
                                    loopInfo,
                                    context);

                // Add receive
                auto receiveTileExpr
                    = (argInfo.numSKTilesPerWG * wgExpr + DF(forTileIncr) + DF(forAccumIncr) - one)
                      % numAccumTiles;
                receiveInfo = receiveTile(graph,
                                          receiveTileExpr < (numAccumTiles - one),
                                          scratchTileInfo.load,
                                          loadConnections,
                                          flagsScratchTag,
                                          accumInfo.accumulatorTile,
                                          accumInfo.usesAccumulatorTile,
                                          accumInfo.accumulatorVarType.dataType,
                                          epilogueOperations,
                                          loopInfo,
                                          argInfo,
                                          params,
                                          context);

                postAccumulationCond = graph.control.addElement(ConditionalOp{
                    zero >= DF(sendInfo.sendBoolSGPR), "Post-accumulation Condition"});

                graph.control.addElement(Else(), {receiveInfo.receiveCond}, {postAccumulationCond});
            }
            else
            {
                scratchTileInfo.setPlusOne  = graph.control.addElement(Scope());
                sendInfo.assignSendBoolSGPR = graph.control.addElement(NOP());
                sendInfo.preWaitZero        = graph.control.addElement(NOP());
                sendInfo.sendCond           = graph.control.addElement(NOP());
                receiveInfo.preWaitZero     = graph.control.addElement(NOP());
                receiveInfo.receiveCond     = graph.control.addElement(NOP());
                receiveInfo.setPlusOne      = graph.control.addElement(Scope());
                postAccumulationCond        = graph.control.addElement(Scope());

                graph.control.addElement(Sequence(), {sendInfo.preWaitZero}, {sendInfo.sendCond});
                graph.control.addElement(
                    Sequence(), {receiveInfo.preWaitZero}, {receiveInfo.receiveCond});
                graph.control.addElement(
                    Sequence(), {receiveInfo.receiveCond}, {postAccumulationCond});
            }

            //
            // Add definitions to the Scope
            //
            auto lastInit = initializeCoordinates(graph,
                                                  scope,
                                                  {forwardForAccumIdx,
                                                   reverseForAccumIdx,
                                                   forwardInfo.selector,
                                                   reverseInfo.selector},
                                                  zero,
                                                  {forTileIncr, forAccumIncr, nextNonAccumTileTag},
                                                  zero);

            //
            // Add local-tile loop
            //
            graph.control.addElement(Sequence(), {lastInit}, {receiveInfo.setPlusOne});

            //
            // Add accumulator loop; send and receive after
            //
            graph.control.addElement(
                Body(), {receiveInfo.setPlusOne}, {scratchTileInfo.setPlusOne});
            graph.control.addElement(Body(), {scratchTileInfo.setPlusOne}, {forTileSKOp});
            graph.control.addElement(Body(), {forTileSKOp}, {assignNextNonAccumTile});
            graph.control.addElement(Sequence(), {assignNextNonAccumTile}, {loopInfo.topLoopOp});

            graph.control.addElement(
                Sequence(), {loopInfo.topLoopOp}, {sendInfo.assignSendBoolSGPR});
            graph.control.addElement(
                Sequence(), {sendInfo.assignSendBoolSGPR}, {sendInfo.preWaitZero});
            graph.control.addElement(Sequence(), {sendInfo.sendCond}, {receiveInfo.preWaitZero});

            if(twoTile)
            {
                auto scopeSK
                    = replaceWith(graph, forTileSKOp, graph.control.addElement(Scope()), false);
                auto scopeDP = graph.control.addElement(Scope());
                graph.control.addElement(Body(), {scopeSK}, {forTileSKOp});
                graph.control.addElement(Sequence(), {scopeSK}, {scopeDP});

                //
                // Set SK/DP selectors to select DP
                //
                auto lastInit = initializeCoordinates(graph,
                                                      scopeDP,
                                                      {forwardInfo.selector, reverseInfo.selector},
                                                      one,
                                                      {forTileIncr, forAccumIncr},
                                                      zero);

                //
                // Create DP tile loop
                //
                auto wgTilesOuterExpr = DF(forTileIncr) < argInfo.numDPTilesPerWG;
                auto dpTilesOuterExpr
                    = (argInfo.numDPTilesPerWG * wgExpr + DF(forTileIncr)) < argInfo.numDPTiles;
                auto incrementOuterExpr = DF(forTileIncr) + DF(forAccumIncr);

                auto forTileDPOp = conditionalFor(graph,
                                                  forTileIncr,
                                                  wgTilesOuterExpr && dpTilesOuterExpr,
                                                  incrementOuterExpr,
                                                  "DPStreamTileLoop",
                                                  numTilesVarType);

                graph.control.addElement(Sequence(), {lastInit}, {forTileDPOp});
                graph.control.addElement(Body(), {forTileDPOp}, {dpTopLoop});

                auto dpTilesInnerExpr   = DF(forAccumIncr) < numAccumTiles;
                auto incrementInnerExpr = DF(forAccumIncr) + one;

                auto forAccumOp      = *graph.control.get<ForLoopOp>(dpAccumLoop);
                forAccumOp.condition = dpTilesInnerExpr;
                graph.control.setElement(dpAccumLoop, forAccumOp);
            }

            //
            // Now can re-attach epilogueOperations.  We defer
            // this to now so that code above can duplicate the
            // for-loops.
            //
            // Make sure receive happens before other operations after
            // the accumulator loop.
            //
            for(auto tag : epilogueOperations)
            {
                graph.control.addElement(Body(), {postAccumulationCond}, {tag});
            }
        }

        ArgumentInfo setupArguments(ExpressionPtr          numWGs,
                                    bool                   twoTile,
                                    KernelGraph const&     graph,
                                    LoopInfo const&        loopInfo,
                                    AccumulatorInfo const& accumInfo,
                                    ContextPtr             context)
        {
            ArgumentInfo argInfo;

            auto k = context->kernel();

            argInfo.numTileArgExprs.resize(loopInfo.dimensionIndices.size() + 1);

            auto numWGsDT   = DataType::UInt32;
            auto numTilesDT = DataType::UInt32;
            auto one        = Expression::literal(1, numTilesDT);
            auto zero       = Expression::literal(0, numTilesDT);

            // On entry, numWGs is an Expression that either:
            //   1. Pulls a value from a CommandArgument
            //   2. Is a literal (for testing)

            argInfo.numWGs = k->addArgument(
                {rocRoller::NUMWGS, numWGsDT, DataDirection::ReadOnly, convert(numWGsDT, numWGs)});

            // Fill number-of-tiles using MacroTileNumber sizes
            // from load operations (store operations are missing
            // that info).
            for(auto dimension : loopInfo.dimensionIndices)
            {
                for(auto tileNumberTag : accumInfo.tileNumberCoords.at(dimension))
                {
                    if(!argInfo.numTileArgExprs[dimension])
                    {
                        auto macTileNumber = graph.coordinates.get<MacroTileNumber>(tileNumberTag);
                        if(macTileNumber->size)
                            argInfo.numTileArgExprs[dimension]
                                = convert(numTilesDT, macTileNumber->size);
                    }
                }
                argInfo.numTileArgExprs.back() = convert(
                    numTilesDT, graph.coordinates.get<ForLoop>(accumInfo.accumulatorCoord)->size);

                for(auto tileNumberTag : accumInfo.tileNumberCoords.at(dimension))
                {
                    AssertFatal(argInfo.numTileArgExprs[dimension]);
                    Log::debug("  dimension: {} coord: {} size: {}",
                               dimension,
                               tileNumberTag,
                               toString(argInfo.numTileArgExprs[dimension]));
                }
            }

            // Make kernel arguments:
            //
            // numWGs:          Value
            // numTiles0:       Computed on host: M / macM
            // numTiles1:       Computed on host: N / macN
            // numTilesAcc:     Computed on host: K / macK
            // numSKTilesPerWG: Computed on host:
            //
            //   for basic StreamK:   (numTiles0 * numTiles1 * numTilesAcc + numWGs - 1) / numWGs
            //   fro 2-tile StreamK:  ((numTilesAcc * ((numTiles0 * numTiles1) % numWGs + numWGs) + numWGs - 1) / numWGs
            //
            // numDPTilesPerWG: Computed on host:
            //
            //   for basic StreamK:   0
            //   fro 2-tile StreamK:  ((numTiles0 * numTiles1) / numWGs - 1) * numTilesAcc
            //

            for(auto d : loopInfo.dimensionIndices)
            {
                argInfo.numTiles.push_back(k->addArgument({concatenate("numTiles", d),
                                                           numTilesDT,
                                                           DataDirection::ReadOnly,
                                                           argInfo.numTileArgExprs[d]}));
                if(d > 0)
                    enableDivideBy(argInfo.numTiles.back(), context);
            }
            argInfo.numTiles.push_back(k->addArgument({"numTilesAcc",
                                                       numTilesDT,
                                                       DataDirection::ReadOnly,
                                                       argInfo.numTileArgExprs.back()}));

            ExpressionPtr numSKTilesArgExpr, numDPTilesArgExpr;
            ExpressionPtr numSKTilesPerWGArgExpr, numDPTilesPerWGArgExpr;
            {
                ExpressionPtr numAccTiles    = argInfo.numTileArgExprs.back();
                ExpressionPtr numNonAccTiles = nullptr;
                for(auto d : loopInfo.dimensionIndices)
                {
                    numNonAccTiles = numNonAccTiles ? numNonAccTiles * argInfo.numTileArgExprs[d]
                                                    : argInfo.numTileArgExprs[d];
                }

                if(twoTile)
                {
                    numSKTilesArgExpr      = (numNonAccTiles % numWGs + numWGs) * numAccTiles;
                    numSKTilesPerWGArgExpr = (numSKTilesArgExpr + numWGs - one) / numWGs;
                    numDPTilesArgExpr      = (numNonAccTiles / numWGs - one) * numWGs * numAccTiles;
                    numDPTilesPerWGArgExpr = (numNonAccTiles / numWGs - one) * numAccTiles;
                }
                else
                {
                    numSKTilesArgExpr      = numNonAccTiles * numAccTiles;
                    numSKTilesPerWGArgExpr = (numSKTilesArgExpr + numWGs - one) / numWGs;
                    numDPTilesArgExpr      = zero;
                    numDPTilesPerWGArgExpr = zero;
                }
            }

            argInfo.numSKTiles = k->addArgument(
                {"numSKTiles", numTilesDT, DataDirection::ReadOnly, numSKTilesArgExpr});

            argInfo.numSKTilesPerWG = k->addArgument(
                {"numSKTilesPerWG", numTilesDT, DataDirection::ReadOnly, numSKTilesPerWGArgExpr});

            if(twoTile)
            {

                argInfo.numDPTiles = k->addArgument(
                    {"numDPTiles", numTilesDT, DataDirection::ReadOnly, numDPTilesArgExpr});

                argInfo.numDPTilesPerWG = k->addArgument({"numDPTilesPerWG",
                                                          numTilesDT,
                                                          DataDirection::ReadOnly,
                                                          numDPTilesPerWGArgExpr});
            }
            else
            {
                argInfo.numDPTiles      = zero;
                argInfo.numDPTilesPerWG = zero;
            }

            return argInfo;
        }

        KernelGraph AddStreamK::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddStreamK");

            LoopInfo loopInfo;

            loopInfo.dimensionIndices = m_dimensionIndices;

            auto kernel = *original.control.roots().begin();

            // Find the loop control nodes
            auto maybeTopLoopOp
                = original.control.findNodes(kernel, makeFindLoopPredicate(original, m_topLoop))
                      .take(1)
                      .only();
            if(!maybeTopLoopOp)
            {
                rocRoller::Log::warn("Unable to find ForLoop '{}' during AddStreamK pass.  "
                                     "AddStreamK transform skipped.",
                                     m_topLoop);
                return original;
            }
            loopInfo.topLoopOp = *maybeTopLoopOp;

            auto maybeAccumLoopOp = only(
                original.control.findElements(makeFindLoopPredicate(original, m_accumulatorLoop)));
            if(!maybeAccumLoopOp)
            {
                rocRoller::Log::warn("Unable to find ForLoop '{}' during AddStreamK pass.  "
                                     "AddStreamK transform skipped.",
                                     m_accumulatorLoop);
                return original;
            }
            loopInfo.accumulatorLoopOp = *maybeAccumLoopOp;

            auto loopLength = [&](int tag) {
                auto dimTag   = original.mapper.get(tag, NaryArgument::DEST);
                auto sizeExpr = getSize(original.coordinates.getNode(dimTag));

                AssertFatal(
                    Expression::evaluationTimes(sizeExpr)[Expression::EvaluationTime::Translate],
                    "Invalid length for x or y loop");
                auto length = Expression::evaluate(sizeExpr);
                AssertFatal(isInteger(length), "For loop length should be an integer");

                return getUnsignedInt(length);
            };

            auto maybeXLoopOp
                = original.control
                      .findNodes(kernel, makeFindLoopPredicate(original, rocRoller::XLOOP))
                      .take(1)
                      .only();
            if(maybeXLoopOp)
            {
                loopInfo.xLoop     = *maybeXLoopOp;
                loopInfo.xLoopSize = loopLength(loopInfo.xLoop);
            }

            auto maybeYLoopOp
                = original.control
                      .findNodes(kernel, makeFindLoopPredicate(original, rocRoller::YLOOP))
                      .take(1)
                      .only();
            if(maybeYLoopOp)
            {
                loopInfo.yLoop     = *maybeYLoopOp;
                loopInfo.yLoopSize = loopLength(loopInfo.yLoop);
            }

            auto graph     = original;
            auto accumInfo = stage(graph, loopInfo);
            auto argInfo
                = setupArguments(m_numWGs, m_twoTile, graph, loopInfo, accumInfo, m_context);
            commit(graph, m_twoTile, loopInfo, accumInfo, argInfo, m_params, m_context);

            return graph;
        }
    }
}
