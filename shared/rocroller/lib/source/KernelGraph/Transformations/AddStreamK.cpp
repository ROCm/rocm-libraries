// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
#include <optional>
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
            /// Coordinate dimension of the accumulator-loop (the ForLoop)
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
            int                localTileSpaceSK; /// Streaming tile space
            std::optional<int> localTileSpaceDP; /// Data-parallel tile space (TwoTile only)
            std::optional<int> selector; /// SK vs DP selector for Sunder (TwoTile only)
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

        template <typename... Args>
        ExpressionPtr minimum(ExpressionPtr x, ExpressionPtr y, Args... remaining)
        {
            if constexpr(sizeof...(remaining) > 0)
                return minimum(minimum(x, y), remaining...);

            return std::make_shared<Expression::Expression>(Expression::Conditional{x < y, x, y});
        }

        // x - y, but if x < y, return 0.
        ExpressionPtr conditionalSubtract(ExpressionPtr x, ExpressionPtr y, DataType dataType)
        {
            return std::make_shared<Expression::Expression>(
                Expression::Conditional{x > y, x - y, literal(0, dataType)});
        }

        /**
         * Create Assign node to add partially accumulated tiles.
         *
         * Returns tag of new Assign node.
         */
        int addFixup(KernelGraph& graph,
                     int          partialMacTileTag,
                     int          destMacTileTag,
                     DataType     dataType,
                     uint         numVGPRs)
        {
            auto lhsExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{destMacTileTag, Register::Type::Vector, dataType});
            auto rhsExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{partialMacTileTag, Register::Type::Vector, dataType});

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
                                              int                  forReceiveTileLoopCoord,
                                              CommandParametersPtr params,
                                              ContextPtr           context)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto numWorkgroupsX = argInfo.numWGs;
            auto numWorkgroupsY = literal(1u);

            auto sizeX = simplify(numWorkgroupsX * literal(static_cast<uint>(macTile.sizes[0])));
            auto sizeY = simplify(numWorkgroupsY * literal(static_cast<uint>(macTile.sizes[1])));

            auto strideX = sizeY;
            auto strideY = literal(1u);

            auto globalScratch = newScratchCoordinate(
                simplify(sizeX * sizeY), varType, Operations::ScratchPolicy::None, context);
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
                if(!std::empty(graph.coordinates.getNeighbours<GD::Downstream>(tag)))
                    return false;
                if(maybeTileNumber->dim != 0)
                    return false;
                return true;
            };

            auto nextTileNumTag
                = graph.coordinates.findNodes(globalScratchTag, danglingMacroTileNumber)
                      .take(1)
                      .only()
                      .value();

            auto one           = Expression::literal(1u);
            auto plusOneTag    = graph.coordinates.addElement(Linear(one, one));
            auto setPlusOneTag = graph.control.addElement(SetCoordinate(one));
            graph.mapper.connect<Linear>(setPlusOneTag, plusOneTag);

            auto tileNumTag = graph.coordinates.addElement(
                *graph.coordinates.get<MacroTileNumber>(nextTileNumTag)); // Copy existing
            graph.coordinates.addElement(
                Split(), {nextTileNumTag}, {tileNumTag, plusOneTag, forReceiveTileLoopCoord});

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
         *       if wave0:
         *          flag = Assign(SGPR, 1u);
         *          StoreSGPR(flag)
         *          WaitZero()
         */
        SendInfo sendTile(KernelGraph&                           graph,
                          ExpressionPtr                          sendTileExpr,
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

            auto jammedX = graph.mapper.get<JammedWaveTileNumber>(storeTileTag, 0);
            if(jammedX != -1)
                graph.coordinates.addElement(
                    PassThrough(), {graph.mapper.get<ForLoop>(forX)}, {jammedX});
            auto jammedY = graph.mapper.get<JammedWaveTileNumber>(storeTileTag, 1);
            if(jammedY != -1)
                graph.coordinates.addElement(
                    PassThrough(), {graph.mapper.get<ForLoop>(forY)}, {jammedY});

            auto sendTileTag = graph.control.addElement(
                ConditionalOp{sendTileExpr, ConditionalMode::Branch, "Send Tile"});
            auto barrierTag  = graph.control.addElement(Barrier());
            auto waitZeroTag = graph.control.addElement(WaitZero());

            // Store flag
            auto flagRegister = graph.coordinates.addElement(VGPR());

            auto assignFlagTag = graph.control.addElement(Assign{Register::Type::Scalar, one});
            graph.mapper.connect(assignFlagTag, flagRegister, NaryArgument::DEST);

            // TODO: Improve setting of arch-specific buffer options
            BufferInstructionOptions bufOpts{.glc = true};

            auto storeFlagTag = graph.control.addElement(StoreSGPR(DataType::UInt32, bufOpts));
            graph.mapper.connect<User>(storeFlagTag, flagsScratchTag);
            graph.mapper.connect<VGPR>(storeFlagTag, flagRegister);

            // Create workitem coordinate and expression for wave 0 check
            // Only workitem 0 (wave 0) should write to the flag
            auto workitemTag = graph.coordinates.addElement(Workitem(0));
            auto workitemDF  = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{workitemTag, Register::Type::Vector, DataType::UInt32});
            auto isWave0Expr       = (workitemDF == Expression::literal(0u));
            auto wave0FlagStoreTag = graph.control.addElement(
                ConditionalOp{isWave0Expr, ConditionalMode::Branch, "Wave0 Store Flag"});

            // Add to control
            auto preWaitZeroTag = graph.control.addElement(WaitZero());

            graph.control.addElement(Sequence(), {preWaitZeroTag}, {sendTileTag});
            graph.control.addElement(Body(), {sendTileTag}, {forX});
            graph.control.chain<Sequence>(forX, waitZeroTag, barrierTag, wave0FlagStoreTag);
            graph.control.addElement(Body(), {wave0FlagStoreTag}, {assignFlagTag});
            auto waitAfterStoreFlagTag = graph.control.addElement(WaitZero());
            graph.control.chain<Sequence>(assignFlagTag, storeFlagTag, waitAfterStoreFlagTag);

            return {preWaitZeroTag, sendTileTag};
        }

        /**
         * Create receive-tile block, which is roughly:
         *
         *     WaitZero()
         *     if receiveTileExpr:
         *       for each partial result (loop created externally)
         *          nextWG = WG + 1 + i
         *          Assert nextWG < numScratch
         *          do:
         *          LoadSGPR(flag[nextWG])
         *          while flag[nextWG] == 0
         *          Barrier()
         *          if wave0:
         *              Assign(flag[nextWG] = 0)
         *              StoreSGPR(flag[nextWG])
         *              WaitZero()
         *          partiallyAccumulatedTile = LoadTiled()
         *          fullyAccumulatedTile = Assign(localPartiallyAccumulatedTile)
         *          fullyAccumulatedTile = Assign(fullyAccumulatedTile + partiallyAccumulatedTile)
         *          localPartiallyAccumulatedTile = Assign(fullyAccumulatedTile)
         *          WaitZero()
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
                             LoopInfo const&                        loopInfo,
                             ArgumentInfo const&                    argInfo,
                             int                                    forReceiveTileLoopOp,
                             int                                    forReceiveTileLoopCoord,
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
            auto receiveTileTag = graph.control.addElement(
                ConditionalOp{receiveTileExpr, ConditionalMode::Branch, "Receive Tile"});

            // Read flag
            auto plusOneTag    = graph.coordinates.addElement(Linear(one, one));
            auto setPlusOneTag = graph.control.addElement(SetCoordinate(one));
            graph.mapper.connect<Linear>(setPlusOneTag, plusOneTag);

            auto nextWorkgroupTag = graph.coordinates.addElement(Linear(nullptr, one));
            graph.coordinates.addElement(
                Split(), {nextWorkgroupTag}, {workgroup, plusOneTag, forReceiveTileLoopCoord});

            // TODO: Improve setting of arch-specific buffer options
            BufferInstructionOptions bufOpts{.glc = true};

            auto flagRegister = graph.coordinates.addElement(VGPR());
            auto loadFlagTag  = graph.control.addElement(LoadSGPR(DataType::UInt32, bufOpts));

            auto numScratch = argInfo.numWGs;

            auto boundsCheckTag = graph.control.addElement(
                AssertOp{"Bounds Check", (DF(nextWorkgroupTag) < numScratch)});

            graph.mapper.connect<User>(loadFlagTag, flagsScratchTag);
            graph.mapper.connect<VGPR>(loadFlagTag, flagRegister);

            // TODO: See if the ControlFlowRWTracer can discover this dependency without this explicit connection.
            // This is currently needed for InlineIncrements to work correctly.
            graph.mapper.connect<ForLoop>(loadFlagTag, forReceiveTileLoopCoord);

            graph.coordinates.addElement(PassThrough(), {flagsScratchTag}, {nextWorkgroupTag});

            auto doWhileTag = graph.control.addElement(
                DoWhileOp{(DF(flagRegister) == zero), "Global sync spin loop"});

            // The coordinate graph for load, store, and reset flags:
            //
            //     Workgroup
            //           |
            //      PassThrough
            //          |
            //          v
            //   flagsScratchTag <-Duplicate-- resetFlagsScratchTag
            //          |                             ^
            //     PassThrough                   PassThrough
            //          |                             |
            //          v                      resetNextWorkgroupTag
            //   nextWorkgroupTag                     ^
            //          |                            Join
            //        Split                        /  |  \
            //       /  |  \                      /   |   \
            //      v   v   v                    /    |    \
            //   Workgroup  plusOne  forReceiveTileLoop
            //
            // Note: nextWorkgroupTag and resetNextWorkgroupTag both connect to the same
            // neighbors (Workgroup, plusOne, forReceiveTileLoop) but in opposite directions
            // (Split vs Join). This allows loading flags from WG+1+i and resetting flags
            // at the same index.

            // Create coordinate to indicate flag index to reset
            auto resetNextWorkgroupTag = graph.coordinates.addElement(Linear(nullptr, one));
            graph.coordinates.addElement(
                Join(), {workgroup, plusOneTag, forReceiveTileLoopCoord}, {resetNextWorkgroupTag});

            // Duplicate flags scratch coordinate for reset operation
            auto resetFlagsScratchTag
                = graph.coordinates.addElement(*graph.coordinates.get<User>(flagsScratchTag));
            graph.coordinates.addElement(Duplicate(), {resetFlagsScratchTag}, {flagsScratchTag});
            graph.coordinates.addElement(
                PassThrough(), {resetNextWorkgroupTag}, {resetFlagsScratchTag});

            // Reset flag operations
            auto assignResetFlagTag
                = graph.control.addElement(Assign{Register::Type::Scalar, zero});
            graph.mapper.connect(assignResetFlagTag, flagRegister, NaryArgument::DEST);

            auto resetFlagTag = graph.control.addElement(StoreSGPR(DataType::UInt32, bufOpts));
            graph.mapper.connect<User>(resetFlagTag, resetFlagsScratchTag);
            graph.mapper.connect<VGPR>(resetFlagTag, flagRegister);

            // TODO: See if the ControlFlowRWTracer can discover this dependency without this explicit connection.
            // This is currently needed for InlineIncrements to work correctly.
            graph.mapper.connect<ForLoop>(resetFlagTag, forReceiveTileLoopCoord);

            // Create workitem coordinate and expression for wave 0 check
            // Only workitem 0 (wave 0) should write to the flag
            auto workitemTag = graph.coordinates.addElement(Workitem(0));
            auto workitemDF  = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{workitemTag, Register::Type::Vector, DataType::UInt32});
            auto isWave0Expr       = (workitemDF == Expression::literal(0u));
            auto wave0ResetFlagTag = graph.control.addElement(
                ConditionalOp{isWave0Expr, ConditionalMode::Branch, "Wave0 Reset Flag"});

            auto barrierBeforeResetTag = graph.control.addElement(Barrier());

            auto accumulatorTile = graph.coordinates.get<MacroTile>(accumulatorTileTag);
            uint numRegisters    = accumulatorTile->elements()
                                / (product(context->kernel()->workgroupSize()) * loopInfo.xLoopSize
                                   * loopInfo.yLoopSize);

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

            // Assign accumulator tile to temporal vgpr tile
            auto fullyAccumulatedTileTag  = graph.coordinates.addElement(MacroTile());
            auto localAccumulatorTileExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{accumulatorTileTag, Register::Type::Vector, dataType});
            auto assignAccTileOp = graph.control.addElement(
                Assign{Register::Type::Vector, localAccumulatorTileExpr, numRegisters});
            graph.mapper.connect(assignAccTileOp, fullyAccumulatedTileTag, NaryArgument::DEST);
            graph.control.addElement(Sequence(), {loadTileTag}, {assignAccTileOp});

            // Fixup
            auto fixupTag
                = addFixup(graph, scratchTileTag, fullyAccumulatedTileTag, dataType, numRegisters);
            graph.control.addElement(Sequence(), {assignAccTileOp}, {fixupTag});

            // Assign fixup result to accumulator tile
            auto fullyAccumulatedTileExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{fullyAccumulatedTileTag, Register::Type::Vector, dataType});
            auto assignFixupResult = graph.control.addElement(
                Assign{Register::Type::Accumulator, fullyAccumulatedTileExpr, numRegisters});
            graph.mapper.connect(assignFixupResult, accumulatorTileTag, NaryArgument::DEST);
            graph.control.addElement(Sequence(), {fixupTag}, {assignFixupResult});

            // Add to control
            auto preWaitZeroTag  = graph.control.addElement(WaitZero());
            auto postWaitZeroTag = graph.control.addElement(WaitZero());

            graph.control.chain<Sequence>(preWaitZeroTag, receiveTileTag);

            graph.control.addElement(Body(), {receiveTileTag}, {forReceiveTileLoopOp});
            graph.control.addElement(Body(), {forReceiveTileLoopOp}, {boundsCheckTag});
            graph.control.addElement(Sequence(), {boundsCheckTag}, {doWhileTag});
            graph.control.addElement(Body(), {doWhileTag}, {loadFlagTag});

            graph.control.chain<Sequence>(doWhileTag, barrierBeforeResetTag, wave0ResetFlagTag);
            graph.control.addElement(Body(), {wave0ResetFlagTag}, {assignResetFlagTag});
            auto waitAfterRestFlagStoreTag = graph.control.addElement(WaitZero());
            graph.control.chain<Sequence>(
                assignResetFlagTag, resetFlagTag, waitAfterRestFlagStoreTag);
            graph.control.chain<Sequence>(wave0ResetFlagTag, loadAddForX, postWaitZeroTag);

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
         *
         * @p setCoordinateGroups is a list of {coordinates, value} pairs.
         * Each group's coordinates get SetCoordinate nodes with the group's
         * value. All SetCoordinate nodes are chained via Body edges from @p top.
         *
         * @p assigns get Assign nodes with @p assignValue, chained via
         * Sequence edges after the SetCoordinate chain.
         */
        int initializeCoordinates(
            KernelGraph& graph,
            int          top,
            std::initializer_list<std::pair<std::initializer_list<int>, ExpressionPtr>>
                                       setCoordinateGroups,
            std::initializer_list<int> assigns,
            ExpressionPtr              assignValue)
        {
            std::vector<int> setChain;
            for(auto const& [coordinates, value] : setCoordinateGroups)
            {
                for(auto coordinate : coordinates)
                {
                    auto setCoordinate = graph.control.addElement(SetCoordinate(value));
                    graph.mapper.connect(setCoordinate, coordinate, NaryArgument::DEST);
                    setChain.push_back(setCoordinate);
                }
            }

            std::vector<int> assignChain;
            for(auto tag : assigns)
            {
                auto def = graph.control.addElement(Assign{Register::Type::Scalar, assignValue});
                graph.mapper.connect(def, tag, NaryArgument::DEST);
                assignChain.push_back(def);
            }

            graph.control.addElement(Body(), {top}, {setChain.front()});
            for(auto i = 1; i < setChain.size(); ++i)
            {
                graph.control.addElement(Body(), {setChain[i - 1]}, {setChain[i]});
            }

            graph.control.addElement(Body(), {setChain.back()}, {assignChain.front()});
            for(auto i = 1; i < assignChain.size(); ++i)
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

        AddStreamK::AddStreamK(ContextPtr           context,
                               CommandParametersPtr params,
                               std::string const&   topLoop,
                               std::string const&   accumulatorLoop,
                               ExpressionPtr        numWGs)
            : m_context(context)
            , m_params(params)
            , m_dimensionIndices(params->loopOverOutputTilesDimensions)
            , m_topLoop(topLoop)
            , m_accumulatorLoop(accumulatorLoop)
            , m_numWGs(numWGs)
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
                                   ExpressionPtr          numTotalTiles,
                                   StreamKConfig          streamKMode)
        {
            // Create forward/reverse tile-numbers for each dimension
            // and attach to all staged tile-number coordinates
            std::vector<int> tileNumbers;
            for(auto d : loopInfo.dimensionIndices)
            {
                // Skip if no dangling MacroTileNumber at this dimension
                if(accumInfo.tileNumberCoords.at(d).empty())
                    continue;

                auto tileNumber = graph.coordinates.addElement(
                    MacroTileNumber(d, argInfo.numTiles[d], nullptr));

                for(auto tileNumTag : accumInfo.tileNumberCoords.at(d))
                {
                    if(forward
                       && std::empty(graph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                        graph.coordinates.addElement(PassThrough(), {tileNumTag}, {tileNumber});
                    if(!forward
                       && std::empty(graph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
                        graph.coordinates.addElement(PassThrough(), {tileNumber}, {tileNumTag});
                }

                tileNumbers.push_back(tileNumber);
            }

            // Create forward/reverse accumulator tile-numbers
            //
            // Appending means: accumulating dimension is fastest moving
            //
            tileNumbers.push_back(accumInfo.accumulatorCoord);

            // Create forward/reverse flattened tile-space
            auto tileSpace   = graph.coordinates.addElement(Linear(numTotalTiles, nullptr));
            auto tileSpaceSK = graph.coordinates.addElement(Linear(argInfo.numSKTiles, nullptr));

            auto WGs       = graph.coordinates.addElement(Linear(argInfo.numWGs, nullptr));
            auto workgroup = graph.coordinates.addElement(Workgroup());

            auto localTileSpaceSK
                = graph.coordinates.addElement(Linear(argInfo.numSKTilesPerWG, nullptr));

            std::optional<int> localTileSpaceDP;
            std::optional<int> selector;

            if(streamKMode.isTwoTileMode())
            {
                bool dpFirst = streamKMode == StreamKMode::TwoTileDPFirst;

                auto tileSpaceDP
                    = graph.coordinates.addElement(Linear(argInfo.numDPTiles, nullptr));
                auto selectorCoord = graph.coordinates.addElement(Linear(literal(2u), nullptr));
                auto localDP
                    = graph.coordinates.addElement(Linear(argInfo.numDPTilesPerWG, nullptr));

                if(forward)
                {
                    graph.coordinates.addElement(
                        Flatten(), tileNumbers, std::vector<int>{tileSpace});
                    if(dpFirst)
                    {
                        graph.coordinates.addElement(
                            Sunder(), {tileSpace}, {tileSpaceDP, tileSpaceSK, selectorCoord});
                    }
                    else
                    {
                        graph.coordinates.addElement(
                            Sunder(), {tileSpace}, {tileSpaceSK, tileSpaceDP, selectorCoord});
                    }
                    graph.coordinates.addElement(PassThrough(), {WGs}, {workgroup});
                    graph.coordinates.addElement(Tile(), {tileSpaceSK}, {WGs, localTileSpaceSK});
                    graph.coordinates.addElement(Tile(), {tileSpaceDP}, {WGs, localDP});
                }
                else
                {
                    if(dpFirst)
                    {
                        graph.coordinates.addElement(
                            Sunder(), {tileSpaceDP, tileSpaceSK, selectorCoord}, {tileSpace});
                    }
                    else
                    {
                        graph.coordinates.addElement(
                            Sunder(), {tileSpaceSK, tileSpaceDP, selectorCoord}, {tileSpace});
                    }
                    graph.coordinates.addElement(Tile(), std::vector<int>{tileSpace}, tileNumbers);
                    graph.coordinates.addElement(PassThrough(), {workgroup}, {WGs});
                    graph.coordinates.addElement(Flatten(), {WGs, localTileSpaceSK}, {tileSpaceSK});
                    graph.coordinates.addElement(Flatten(), {WGs, localDP}, {tileSpaceDP});
                }

                localTileSpaceDP = localDP;
                selector         = selectorCoord;
            }
            else
            {
                // Standard mode: entire tile space is SK, no Sunder needed.
                if(forward)
                {
                    graph.coordinates.addElement(
                        Flatten(), tileNumbers, std::vector<int>{tileSpace});
                    graph.coordinates.addElement(PassThrough(), {tileSpace}, {tileSpaceSK});
                    graph.coordinates.addElement(PassThrough(), {WGs}, {workgroup});
                    graph.coordinates.addElement(Tile(), {tileSpaceSK}, {WGs, localTileSpaceSK});
                }
                else
                {
                    graph.coordinates.addElement(PassThrough(), {tileSpaceSK}, {tileSpace});
                    graph.coordinates.addElement(Tile(), std::vector<int>{tileSpace}, tileNumbers);
                    graph.coordinates.addElement(PassThrough(), {workgroup}, {WGs});
                    graph.coordinates.addElement(Flatten(), {WGs, localTileSpaceSK}, {tileSpaceSK});
                }
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
                if(graph.control.compareNodes(
                       rocRoller::UseCacheIfAvailable, loopInfo.accumulatorLoopOp, m.control)
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
                    if(std::empty(graph.coordinates.getNeighbours<GD::Upstream>(tag))
                       || std::empty(graph.coordinates.getNeighbours<GD::Downstream>(tag)))
                        return true;
                    return false;
                };

                // NOTE: a dimension might not have any dangling MacroTileNumber.
                //       This happens when workgroupMapping (RemapOutputTiles)
                //       applied as M & N dimensions get flattened int a new MacroTileNumber.
                accumInfo.tileNumberCoords[dimension]
                    = graph.coordinates.findElements(danglingTileNumberPredicate)
                          .to<std::unordered_set>();
            }

            return accumInfo;
        }

        //
        // Commit
        //
        void commit(ContextPtr             context,
                    CommandParametersPtr   params,
                    KernelGraph&           graph,
                    LoopInfo const&        loopInfo,
                    AccumulatorInfo const& accumInfo,
                    ArgumentInfo const&    argInfo)
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
                        only(graph.control.getNeighbours<GD::Downstream>(tag)).value());
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
            if(params->streamK.isTwoTileMode())
            {
                // Keep the accumulator tile shared between SK and DP sections
                // so that the fix-up pass uses the correct registers.
                auto accumTile = accumInfo.accumulatorTile;
                auto reindexer = std::make_shared<GraphReindexer>();
                dpTopLoop
                    = only(duplicateControlNodes(graph,
                                                 reindexer,
                                                 {loopInfo.topLoopOp},
                                                 [accumTile](int x) { return x == accumTile; }))
                          .value();
                dpAccumLoop = reindexer->control[loopInfo.accumulatorLoopOp];

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

            auto forwardInfo = addTileSpaceCT(
                graph, true, loopInfo, accumInfo, argInfo, numTotalTiles, params->streamK);
            auto reverseInfo = addTileSpaceCT(
                graph, false, loopInfo, accumInfo, argInfo, numTotalTiles, params->streamK);

            // Hijack the old accumulator loop. Remove the DataFlow edge
            // from the accumulator increment counter to the original
            // accumulator ForLoop coordinate.
            auto forAccumIncr = only(graph.coordinates.getInputNodeIndices(
                                         accumInfo.accumulatorCoord, CG::isEdge<CG::DataFlow>))
                                    .value();
            {
                auto dataflow
                    = only(graph.coordinates.getNeighbours<GD::Downstream>(forAccumIncr)).value();
                graph.coordinates.deleteElement(dataflow);
            }

            // Create new forward/reverse accum ForLoop dims, shared by both
            // SK and DP (both iterate over the same number of accum tiles).
            auto forwardForAccumIdx = graph.coordinates.addElement(ForLoop(numAccumTiles, one));
            auto reverseForAccumIdx = graph.coordinates.addElement(ForLoop(numAccumTiles, one));

            // Wire the accumulator increment counter to the new dims.
            graph.coordinates.addElement(DataFlow(), {forAccumIncr}, {forwardForAccumIdx});
            graph.coordinates.addElement(DataFlow(), {forAccumIncr}, {reverseForAccumIdx});

            // Identify edges let index computation traverse from the new
            // ForLoop dims back to the original coordinate for offsets/strides.
            graph.coordinates.addElement(
                Identify(), {forwardForAccumIdx}, {accumInfo.accumulatorCoord});
            graph.coordinates.addElement(
                Identify(), {reverseForAccumIdx}, {accumInfo.accumulatorCoord});

            graph.mapper.connect<ForLoop>(loopInfo.accumulatorLoopOp, forwardForAccumIdx);

            // Create tile ForLoop dims and connect them to the local tile
            // spaces via Split/Join. SK and DP get separate tile dims
            // but share the accum dims created above.

            auto skForTileIncr = graph.coordinates.addElement(Linear(argInfo.numSKTilesPerWG, one));
            auto skForwardForTileIdx
                = graph.coordinates.addElement(ForLoop(argInfo.numSKTilesPerWG, one));
            auto skReverseForTileIdx
                = graph.coordinates.addElement(ForLoop(argInfo.numSKTilesPerWG, one));

            Log::debug(
                "  SK tile ForLoops: fwd {} rev {}", skForwardForTileIdx, skReverseForTileIdx);
            Log::debug(
                "  Shared accum ForLoops: fwd {} rev {}", forwardForAccumIdx, reverseForAccumIdx);

            graph.coordinates.addElement(
                Split(), {forwardInfo.localTileSpaceSK}, {skForwardForTileIdx, forwardForAccumIdx});
            graph.coordinates.addElement(
                Join(), {skReverseForTileIdx, reverseForAccumIdx}, {reverseInfo.localTileSpaceSK});

            graph.coordinates.addElement(DataFlow(), {skForTileIncr}, {skForwardForTileIdx});
            graph.coordinates.addElement(DataFlow(), {skForTileIncr}, {skReverseForTileIdx});

            // DP tile ForLoop dims (TwoTile only)
            std::optional<int> dpForTileIncr;
            std::optional<int> dpForwardForTileIdx;
            std::optional<int> dpReverseForTileIdx;

            if(params->streamK.isTwoTileMode())
            {
                dpForTileIncr = graph.coordinates.addElement(Linear(argInfo.numDPTilesPerWG, one));

                dpForwardForTileIdx
                    = graph.coordinates.addElement(ForLoop(argInfo.numDPTilesPerWG, one));
                dpReverseForTileIdx
                    = graph.coordinates.addElement(ForLoop(argInfo.numDPTilesPerWG, one));

                Log::debug("  DP tile ForLoops: fwd {} rev {}",
                           dpForwardForTileIdx.value(),
                           dpReverseForTileIdx.value());

                graph.coordinates.addElement(Split(),
                                             {forwardInfo.localTileSpaceDP.value()},
                                             {dpForwardForTileIdx.value(), forwardForAccumIdx});
                graph.coordinates.addElement(Join(),
                                             {dpReverseForTileIdx.value(), reverseForAccumIdx},
                                             {reverseInfo.localTileSpaceDP.value()});

                graph.coordinates.addElement(
                    DataFlow(), {dpForTileIncr.value()}, {dpForwardForTileIdx.value()});
                graph.coordinates.addElement(
                    DataFlow(), {dpForTileIncr.value()}, {dpReverseForTileIdx.value()});
            }

            // The original accumulator coordinate no longer has a DataFlow
            // input. Replace it with a Linear so that loop-related passes
            // don't treat it as an active ForLoop iterator.
            {
                auto accumulatorCoordSize
                    = getSize(graph.coordinates.getNode(accumInfo.accumulatorCoord));
                auto linear = Linear(accumulatorCoordSize, nullptr);
                graph.coordinates.setElement(accumInfo.accumulatorCoord, linear);
            }

            // Pre-compute and save (into SGPRs):
            //
            // 1. Number of accumulation tiles processed in the accumulator loop
            // 2. The first accumulator tile index processed
            // 3. The last accumulator tile index processed
            //
            // The matching Assign operations are created below
            auto currentTile            = graph.coordinates.addElement(VGPR());
            auto numAccumTilesProcessed = graph.coordinates.addElement(VGPR());
            auto firstAccumTile         = graph.coordinates.addElement(VGPR());
            auto lastAccumTile          = graph.coordinates.addElement(VGPR());

            //
            // Create local-tile loop
            //
            int forTileSKOp;
            {
                auto numSKTilesPerWGBound = argInfo.numSKTilesPerWG;
                auto numSKTilesBound      = conditionalSubtract(
                    argInfo.numSKTiles, argInfo.numSKTilesPerWG * wgExpr, numTilesVarType.dataType);
                auto forTileSKConditionExpr
                    = DF(skForTileIncr) < minimum(numSKTilesPerWGBound, numSKTilesBound);

                auto forTileSKIncrementExpr = DF(skForTileIncr) + DF(numAccumTilesProcessed);

                forTileSKOp = conditionalFor(graph,
                                             skForTileIncr,
                                             forTileSKConditionExpr,
                                             forTileSKIncrementExpr,
                                             "SKStreamTileLoop",
                                             numTilesVarType);
            }

            int assignCurrentTile;
            {
                assignCurrentTile = graph.control.addElement(Assign{
                    Register::Type::Scalar, argInfo.numSKTilesPerWG * wgExpr + DF(skForTileIncr)});
                graph.mapper.connect(assignCurrentTile, currentTile, NaryArgument::DEST);
            }

            // Compute the starting K tile index within the current output tile.
            // firstAccumTile = currentTile % numAccumTiles gives which K tile (0 to numAccumTiles-1)
            // we begin processing within this (M, N) output tile.
            int assignFirstAccumTile;
            {
                auto firstAccumTileExpr = DF(currentTile) % numAccumTiles;
                assignFirstAccumTile
                    = graph.control.addElement(Assign{Register::Type::Scalar, firstAccumTileExpr});
                graph.mapper.connect(assignFirstAccumTile, firstAccumTile, NaryArgument::DEST);
            }

            int assignNumAccumTilesProcessed;
            {
                // Compute how many K tiles to process in this iteration.
                // sameNonAccumTileBound = numAccumTiles - firstAccumTile is the number of K tiles
                // remaining in the current output tile. We take the minimum of this and other bounds.
                auto sameNonAccumTileBound = conditionalSubtract(
                    numAccumTiles, DF(firstAccumTile), numTilesVarType.dataType);
                auto numSKTilesPerWGBound = conditionalSubtract(
                    argInfo.numSKTilesPerWG, DF(skForTileIncr), numTilesVarType.dataType);
                auto numSKTilesExprBound = conditionalSubtract(
                    argInfo.numSKTiles, DF(currentTile), numTilesVarType.dataType);

                assignNumAccumTilesProcessed = graph.control.addElement(Assign{
                    Register::Type::Scalar,
                    minimum(sameNonAccumTileBound, numSKTilesPerWGBound, numSKTilesExprBound)});
                graph.mapper.connect(
                    assignNumAccumTilesProcessed, numAccumTilesProcessed, NaryArgument::DEST);
            }

            int assignLastAccumTile;
            {
                // Compute the last K tile index processed within this output tile.
                // Since numAccumTilesProcessed <= numAccumTiles - firstAccumTile (bounded by
                // sameNonAccumTileBound to stay within the same output tile), the result is
                // guaranteed to be in range [0, numAccumTiles - 1].
                auto lastAccumTileExpr = DF(firstAccumTile) + DF(numAccumTilesProcessed) - one;
                assignLastAccumTile
                    = graph.control.addElement(Assign{Register::Type::Scalar, lastAccumTileExpr});
                graph.mapper.connect(assignLastAccumTile, lastAccumTile, NaryArgument::DEST);
            }

            auto forAccumOp      = *graph.control.get<ForLoopOp>(loopInfo.accumulatorLoopOp);
            forAccumOp.condition = DF(forAccumIncr) < DF(numAccumTilesProcessed);
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
                auto remainAccumTiles = numAccumTiles - DF(lastAccumTile) - one;

                // For loop that receives and accumulates partial results from other workgroups.
                // The loop iterates ceil(remainAccumTiles / numSKTilesPerWG) times.
                // Using the equivalence: i < ceil(a/b) iff i*b < a (for a >= 0, b > 0),
                // we express the loop condition as a multiplication to avoid runtime division.
                auto iteratorCoord = graph.coordinates.addElement(Linear(numAccumTiles, one));
                auto forReceiveTileLoopCoord
                    = graph.coordinates.addElement(ForLoop(numAccumTiles, one));
                graph.coordinates.addElement(
                    DataFlow(), {iteratorCoord}, {forReceiveTileLoopCoord});

                auto iterator = std::make_shared<Expression::Expression>(Expression::DataFlowTag{
                    iteratorCoord, Register::Type::Scalar, numTilesVarType});

                // Loop condition: iterates while iterator * numSKTilesPerWG < remainAccumTiles
                // This is equivalent to iterator < ceil(remainAccumTiles / numSKTilesPerWG)
                auto loopCondition = iterator * argInfo.numSKTilesPerWG < remainAccumTiles;

                auto forReceiveTileLoopOp
                    = graph.control.addElement(ForLoopOp{loopCondition, rocRoller::RECEIVE});
                graph.mapper.connect(forReceiveTileLoopOp, iteratorCoord, NaryArgument::DEST);
                graph.mapper.connect<ForLoop>(forReceiveTileLoopOp, forReceiveTileLoopCoord);

                auto initialAssign = graph.control.addElement(
                    Assign{Register::Type::Scalar, Expression::literal(0, numTilesVarType)});
                graph.mapper.connect(initialAssign, iteratorCoord, NaryArgument::DEST);

                auto incrementAssign
                    = graph.control.addElement(Assign{Register::Type::Scalar, iterator + one});
                graph.mapper.connect(incrementAssign, iteratorCoord, NaryArgument::DEST);

                graph.control.addElement(Initialize(), {forReceiveTileLoopOp}, {initialAssign});
                graph.control.addElement(
                    ForLoopIncrement(), {forReceiveTileLoopOp}, {incrementAssign});

                // Create scratch space for flags
                auto flagsScratch
                    = newScratchCoordinate(argInfo.numWGs,
                                           DataType::UInt32,
                                           Operations::ScratchPolicy::ZeroedBeforeAndAfter,
                                           context);
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
                                                            forReceiveTileLoopCoord,
                                                            params,
                                                            context);

                // Add send and receive
                auto hasFirstAccumTile        = DF(firstAccumTile) == zero;
                auto doesntHaveFirstAccumTile = DF(firstAccumTile) != zero;
                auto doesntHaveLastAccumTile  = DF(lastAccumTile) < (numAccumTiles - one);

                sendInfo = sendTile(graph,
                                    doesntHaveFirstAccumTile,
                                    storeConnections,
                                    flagsScratchTag,
                                    accumInfo.accumulatorVarType.dataType,
                                    numTilesVarType,
                                    loopInfo,
                                    context);

                receiveInfo = receiveTile(graph,
                                          hasFirstAccumTile && doesntHaveLastAccumTile,
                                          scratchTileInfo.load,
                                          loadConnections,
                                          flagsScratchTag,
                                          accumInfo.accumulatorTile,
                                          accumInfo.usesAccumulatorTile,
                                          accumInfo.accumulatorVarType.dataType,
                                          loopInfo,
                                          argInfo,
                                          forReceiveTileLoopOp,
                                          forReceiveTileLoopCoord,
                                          params,
                                          context);

                postAccumulationCond = graph.control.addElement(ConditionalOp{
                    hasFirstAccumTile, ConditionalMode::Branch, "Post-accumulation Condition"});
            }
            else
            {
                scratchTileInfo.setPlusOne = graph.control.addElement(Scope());
                sendInfo.preWaitZero       = graph.control.addElement(NOP());
                sendInfo.sendCond          = graph.control.addElement(NOP());
                receiveInfo.preWaitZero    = graph.control.addElement(NOP());
                receiveInfo.receiveCond    = graph.control.addElement(NOP());
                receiveInfo.setPlusOne     = graph.control.addElement(Scope());
                postAccumulationCond       = graph.control.addElement(Scope());

                graph.control.addElement(Sequence(), {sendInfo.preWaitZero}, {sendInfo.sendCond});
                graph.control.addElement(
                    Sequence(), {receiveInfo.preWaitZero}, {receiveInfo.receiveCond});
            }
            graph.control.addElement(Sequence(), {receiveInfo.receiveCond}, {postAccumulationCond});

            // Add local-tile loop and accumulator loop
            if(params->streamK.isTwoTileMode())
            {
                bool dpFirst = params->streamK == StreamKMode::TwoTileDPFirst;
                int  scopeSK = graph.control.addElement(Scope());
                int  scopeDP = graph.control.addElement(Scope());

                // Connect scopes in the appropriate order.
                if(dpFirst)
                {
                    graph.control.addElement(Sequence(), {scope}, {scopeDP});
                    graph.control.addElement(Sequence(), {scopeDP}, {scopeSK});
                }
                else
                {
                    graph.control.addElement(Sequence(), {scope}, {scopeSK});
                    graph.control.addElement(Sequence(), {scopeSK}, {scopeDP});
                }

                auto skSelectorValue = dpFirst ? one : zero;
                auto dpSelectorValue = dpFirst ? zero : one;

                // Initialize SK section.
                // The DP tile dims are also initialized to zero here so
                // that the reverse Sunder can be fully traversed.  The
                // selector ensures only the SK branch is evaluated at
                // runtime.
                auto skInit = initializeCoordinates(
                    graph,
                    scopeSK,
                    {{{skForwardForTileIdx,
                       skReverseForTileIdx,
                       dpForwardForTileIdx.value(),
                       dpReverseForTileIdx.value(),
                       forwardForAccumIdx,
                       reverseForAccumIdx},
                      zero},
                     {{forwardInfo.selector.value(), reverseInfo.selector.value()},
                      skSelectorValue}},
                    {skForTileIncr,
                     forAccumIncr,
                     currentTile,
                     numAccumTilesProcessed,
                     firstAccumTile,
                     lastAccumTile},
                    zero);

                // Chain SK-specific setup inside scopeSK
                graph.control.chain<Sequence>(skInit, receiveInfo.setPlusOne);
                graph.control.chain<Body>(
                    receiveInfo.setPlusOne, scratchTileInfo.setPlusOne, forTileSKOp);
                graph.control.chain<Body>(forTileSKOp, assignCurrentTile);
                graph.control.chain<Sequence>(assignCurrentTile,
                                              assignFirstAccumTile,
                                              assignNumAccumTilesProcessed,
                                              assignLastAccumTile,
                                              loopInfo.topLoopOp,
                                              sendInfo.preWaitZero);

                graph.control.chain<Sequence>(sendInfo.sendCond, receiveInfo.preWaitZero);

                // Initialize DP section.
                // The SK tile dims are also initialized to zero here so
                // that the reverse Sunder can be fully traversed.  The
                // selector ensures only the DP branch is evaluated at
                // runtime.
                auto dpInit = initializeCoordinates(
                    graph,
                    scopeDP,
                    {{{skForwardForTileIdx,
                       skReverseForTileIdx,
                       dpForwardForTileIdx.value(),
                       dpReverseForTileIdx.value(),
                       forwardForAccumIdx,
                       reverseForAccumIdx},
                      zero},
                     {{forwardInfo.selector.value(), reverseInfo.selector.value()},
                      dpSelectorValue}},
                    {dpForTileIncr.value(), forAccumIncr},
                    zero);

                // Create DP tile loop (uses its own dpForTileIncr counter).
                int forTileDPOp;
                {
                    auto numDPTilesPerWGBound   = argInfo.numDPTilesPerWG;
                    auto numDPTilesBound        = conditionalSubtract(argInfo.numDPTiles,
                                                               argInfo.numDPTilesPerWG * wgExpr,
                                                               numTilesVarType.dataType);
                    auto forTileDPConditionExpr = DF(dpForTileIncr.value())
                                                  < minimum(numDPTilesPerWGBound, numDPTilesBound);

                    auto forTileDPIncrementExpr = DF(dpForTileIncr.value()) + DF(forAccumIncr);

                    forTileDPOp = conditionalFor(graph,
                                                 dpForTileIncr.value(),
                                                 forTileDPConditionExpr,
                                                 forTileDPIncrementExpr,
                                                 "DPStreamTileLoop",
                                                 numTilesVarType);
                }

                graph.control.addElement(Sequence(), {dpInit}, {forTileDPOp});
                graph.control.addElement(Body(), {forTileDPOp}, {dpTopLoop});

                auto forAccumOp      = *graph.control.get<ForLoopOp>(dpAccumLoop);
                forAccumOp.condition = DF(forAccumIncr) < numAccumTiles;
                graph.control.setElement(dpAccumLoop, forAccumOp);

                graph.mapper.connect<ForLoop>(dpAccumLoop, forwardForAccumIdx);
            }
            else
            {
                // Standard mode -- no selector, no DP dims
                auto standardInit = initializeCoordinates(graph,
                                                          scope,
                                                          {{{skForwardForTileIdx,
                                                             skReverseForTileIdx,
                                                             forwardForAccumIdx,
                                                             reverseForAccumIdx},
                                                            zero}},
                                                          {skForTileIncr,
                                                           forAccumIncr,
                                                           currentTile,
                                                           numAccumTilesProcessed,
                                                           firstAccumTile,
                                                           lastAccumTile},
                                                          zero);

                graph.control.chain<Sequence>(standardInit, receiveInfo.setPlusOne);
                graph.control.chain<Body>(
                    receiveInfo.setPlusOne, scratchTileInfo.setPlusOne, forTileSKOp);
                graph.control.chain<Body>(forTileSKOp, assignCurrentTile);
                graph.control.chain<Sequence>(assignCurrentTile,
                                              assignFirstAccumTile,
                                              assignNumAccumTilesProcessed,
                                              assignLastAccumTile,
                                              loopInfo.topLoopOp,
                                              sendInfo.preWaitZero);

                graph.control.chain<Sequence>(sendInfo.sendCond, receiveInfo.preWaitZero);
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

        ArgumentInfo setupArguments(ContextPtr             context,
                                    CommandParametersPtr   params,
                                    ExpressionPtr          numWGs,
                                    KernelGraph const&     graph,
                                    LoopInfo const&        loopInfo,
                                    AccumulatorInfo const& accumInfo)
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
            //   for 2-tile StreamK:  ((numTilesAcc * ((numTiles0 * numTiles1) % numWGs + numWGs) + numWGs - 1) / numWGs
            //
            // numDPTilesPerWG: Computed on host:
            //
            //   for basic StreamK:   0
            //   for 2-tile StreamK:  ((numTiles0 * numTiles1) / numWGs - 1) * numTilesAcc
            //
            for(auto d : loopInfo.dimensionIndices)
            {
                if(argInfo.numTileArgExprs[d] == nullptr)
                    continue;

                argInfo.numTiles.push_back(k->addArgument({concatenate("numTiles", d),
                                                           numTilesDT,
                                                           DataDirection::ReadOnly,
                                                           argInfo.numTileArgExprs[d]}));
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
                    if(argInfo.numTileArgExprs[d] == nullptr)
                        continue;

                    numNonAccTiles = numNonAccTiles ? numNonAccTiles * argInfo.numTileArgExprs[d]
                                                    : argInfo.numTileArgExprs[d];
                }

                if(params->streamK.isTwoTileMode())
                {
                    auto enoughNonAccTilesExpr = numNonAccTiles > numWGs - one;

                    numSKTilesArgExpr
                        = conditional(enoughNonAccTilesExpr,
                                      (numNonAccTiles % numWGs + numWGs) * numAccTiles,
                                      numNonAccTiles * numAccTiles);

                    numSKTilesPerWGArgExpr = (numSKTilesArgExpr + numWGs - one) / numWGs;
                    numDPTilesArgExpr
                        = conditional(enoughNonAccTilesExpr,
                                      (numNonAccTiles / numWGs - one) * numWGs * numAccTiles,
                                      zero);
                    numDPTilesPerWGArgExpr = conditional(
                        enoughNonAccTilesExpr, (numNonAccTiles / numWGs - one) * numAccTiles, zero);
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

            if(params->streamK.isTwoTileMode())
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
                = setupArguments(m_context, m_params, m_numWGs, graph, loopInfo, accumInfo);

            commit(m_context, m_params, graph, loopInfo, accumInfo, argInfo);

            return graph;
        }
    }
}
