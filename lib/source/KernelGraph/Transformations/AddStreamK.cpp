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
 */

#include <map>
#include <unordered_set>
#include <vector>

#include <rocRoller/Expression.hpp>
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

        ConstraintStatus NoLoadLDS(const KernelGraph& k)
        {
            ConstraintStatus retval;

            // Look for any loads of Matrix A or Matrix B that aren't using LDS
            for(auto const& loadTag : k.control.getNodes<LoadTiled>())
            {
                auto [tileTag, tile] = k.getDimension<MacroTile>(loadTag);

                if((tile.layoutType == LayoutType::MATRIX_A
                    || tile.layoutType == LayoutType::MATRIX_B)
                   && tile.memoryType == MemoryType::WAVE)
                {
                    return {false,
                            "The StreamK transformation does not work when data is not loaded "
                            "through LDS"};
                }
            }

            return retval;
        }

        //
        // Helpers
        //

        /**
         * Replace the use of an old macrotile in the given control
         * nodes with a new macrotile.
         */
        void replaceMacroTile(KernelGraph&                   graph,
                              std::unordered_set<int> const& ops,
                              int                            oldMacTileTag,
                              int                            newMacTileTag)
        {
            for(auto const& opTag : ops)
            {
                auto element = graph.control.getElement(opTag);
                visit(
                    rocRoller::overloaded{
                        [&](StoreTiled store) {
                            graph.mapper.connect<MacroTile>(opTag, oldMacTileTag);

                            // update the data flow in the coordinate graph
                            auto dstTag = graph.mapper.get<User>(opTag);
                            auto df
                                = *only(graph.coordinates.getNeighbours<Graph::Direction::Upstream>(
                                    dstTag));
                            graph.coordinates.deleteElement(df);
                            graph.coordinates.addElement(DataFlow(),
                                                         std::vector<int>{newMacTileTag},
                                                         std::vector<int>{dstTag});
                        },
                        [&](Assign assign) {
                            GraphReindexer contractionReindexer;
                            contractionReindexer.coordinates.emplace(oldMacTileTag, newMacTileTag);
                            reindexExpressions(graph, opTag, contractionReindexer);

                            // update the data flow in the coordinate graph
                            auto dstTag = only(graph.mapper.getConnections(opTag))->coordinate;
                            std::vector<int> srcTags;
                            for(auto const& edgeTag :
                                graph.coordinates.getNeighbours<Graph::Direction::Upstream>(dstTag))
                            {
                                auto df = graph.coordinates.get<DataFlow>(edgeTag);
                                if(!df)
                                    continue;
                                auto srcs
                                    = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(
                                        edgeTag);
                                for(auto const src : srcs)
                                {
                                    if(src == oldMacTileTag)
                                        srcTags.push_back(newMacTileTag);
                                    else
                                        srcTags.push_back(src);
                                }
                                graph.coordinates.deleteElement(edgeTag);
                            }
                            graph.coordinates.addElement(
                                DataFlow(), srcTags, std::vector<int>{dstTag});
                        },
                        [&](auto op) { Throw<FatalError>("Not handled yet."); }},
                    std::get<Operation>(element));
            }
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

            graph.coordinates.addElement(
                DataFlow(), {accumMacTileTag, partialMacTileTag}, {destMacTileTag});
            graph.mapper.connect(fixupTag, destMacTileTag, NaryArgument::DEST);

            return fixupTag;
        }

        /**
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
                                              ContextPtr                       context)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto numWorkgroupsX = literal(context->kernelOptions().numScratchTiles);
            auto numWorkgroupsY = literal(1u);

            auto sizeX = simplify(numWorkgroupsX * literal(static_cast<uint>(macTile.sizes[0])));
            auto sizeY = simplify(numWorkgroupsY * literal(static_cast<uint>(macTile.sizes[1])));

            auto strideX = sizeY;
            auto strideY = literal(1u);

            auto globalScratch    = newScratchCoordinate(simplify(sizeX * sizeY), varType, context);
            auto globalScratchTag = graph.coordinates.addElement(globalScratch);

            // Store
            auto storeScratchTileTag = createInternalTile(graph, varType, macTileTag, context);
            graph.coordinates.addElement(View(), {storeScratchTileTag}, {macTileTag});

            auto storeScratchTile       = *graph.coordinates.get<MacroTile>(storeScratchTileTag);
            storeScratchTile.layoutType = LayoutType::SCRATCH;
            graph.coordinates.setElement(storeScratchTileTag, storeScratchTile);

            std::vector<int> storeSubDimensions
                = {graph.coordinates.addElement(SubDimension(0, sizeX, strideX)),
                   graph.coordinates.addElement(SubDimension(1, sizeY, strideY))};
            graph.coordinates.addElement(
                Join(), storeSubDimensions, std::vector<int>{globalScratchTag});

            storeMacroTile_VGPR(graph,
                                storeConnections,
                                globalScratchTag,
                                storeScratchTileTag,
                                storeSubDimensions,
                                context);

            storeConnections.push_back(DC<MacroTile>(storeScratchTileTag));
            storeConnections.push_back(DC<User>(globalScratchTag));

            // Load
            auto loadScratchTileTag    = createInternalTile(graph, varType, macTileTag, context);
            auto loadScratchTile       = *graph.coordinates.get<MacroTile>(loadScratchTileTag);
            loadScratchTile.layoutType = LayoutType::SCRATCH;
            graph.coordinates.setElement(loadScratchTileTag, loadScratchTile);

            std::vector<int> loadSubDimensions
                = {graph.coordinates.addElement(SubDimension(0, sizeX, strideX)),
                   graph.coordinates.addElement(SubDimension(1, sizeY, strideY))};
            graph.coordinates.addElement(
                Split(), std::vector<int>{globalScratchTag}, loadSubDimensions);

            loadMacroTile_VGPR(graph,
                               loadConnections,
                               globalScratchTag,
                               loadScratchTileTag,
                               loadSubDimensions,
                               context);

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

            return {storeScratchTileTag, loadScratchTileTag, globalScratchTag, setPlusOneTag};
        }

        /**
         * Create StoreTiled operation that stores partially accumulated tile.
         */
        int storeScratchTile(KernelGraph& graph, int afterOpTag, DataType dataType)
        {
            auto storePartialTag = graph.control.addElement(StoreTiled(dataType));
            graph.control.addElement(Sequence(), {afterOpTag}, {storePartialTag});
            return storePartialTag;
        }

        /**
         * Create LoadTiled operation that loads partially accumulated tile.
         */
        int loadScratchTile(KernelGraph& graph, int afterOpTag, DataType dataType)
        {
            auto loadPartialTag = graph.control.addElement(LoadTiled(dataType));
            graph.control.addElement(Sequence(), {afterOpTag}, {loadPartialTag});
            return loadPartialTag;
        }

        /**
         * Create Assign operation to copy tiles.
         */
        int copyAssign(KernelGraph& graph,
                       int          srcMacTileTag,
                       int          destMacTileTag,
                       DataType     dataType,
                       uint         numRegisters)
        {
            auto srcExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{srcMacTileTag, Register::Type::Accumulator, dataType});

            auto copyAssignTag
                = graph.control.addElement(Assign{Register::Type::Vector, srcExpr, numRegisters});

            graph.coordinates.addElement(DataFlow(), {srcMacTileTag}, {destMacTileTag});
            graph.mapper.connect(copyAssignTag, destMacTileTag, NaryArgument::DEST);

            return copyAssignTag;
        }

        /**
         * Create send-tile block, which is roughly:
         *
         *     WaitZero()
         *     if sendTileExpr:
         *       StoreTile()
         *       WaitZero()
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
                          ContextPtr                             context)
        {
            auto one  = Expression::literal(1u);
            auto zero = Expression::literal(0, numTilesVarType);
            auto DF   = [numTilesVarType](int tag) {
                return std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{tag, Register::Type::Scalar, numTilesVarType});
            };

            auto workgroup = graph.coordinates.addElement(Workgroup(0, one));
            graph.coordinates.addElement(PassThrough(), {workgroup}, {flagsScratchTag});

            // Store tile
            auto storeTileTag = graph.control.addElement(StoreTiled(scratchDataType));
            for(auto const& c : storeConnections)
                graph.mapper.connect(storeTileTag, c.coordinate, c.connectionSpec);

            auto sendTileRegister = graph.coordinates.addElement(VGPR());
            auto sendTileAssign
                = graph.control.addElement(Assign{Register::Type::Scalar, tileExpr});
            graph.mapper.connect(sendTileAssign, sendTileRegister, NaryArgument::DEST);
            // Yoda-expression is a workaround for an issue in
            // GreaterThan.  A more natural condtion would be:
            //   DF(sendTileRegister) > zero.
            auto sendTileTag = graph.control.addElement(ConditionalOp{zero < DF(sendTileRegister)});
            auto waitZeroTag = graph.control.addElement(WaitZero());

            // Store flag
            auto flagRegister = graph.coordinates.addElement(VGPR());

            auto assignFlagTag = graph.control.addElement(Assign{Register::Type::Scalar, one});
            graph.mapper.connect(assignFlagTag, flagRegister, NaryArgument::DEST);

            auto storeFlagTag = graph.control.addElement(StoreSGPR(DataType::UInt32, true));
            graph.mapper.connect<User>(storeFlagTag, flagsScratchTag);
            graph.mapper.connect<VGPR>(storeFlagTag, flagRegister);

            // Add to control
            auto preWaitZeroTag  = graph.control.addElement(WaitZero());
            auto postWaitZeroTag = graph.control.addElement(WaitZero());

            graph.control.addElement(Sequence(), {preWaitZeroTag}, {sendTileTag});
            graph.control.addElement(Body(), {sendTileTag}, {storeTileTag});
            graph.control.chain<Sequence>(
                storeTileTag, waitZeroTag, assignFlagTag, storeFlagTag, postWaitZeroTag);

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
                             ContextPtr                             context)
        {
            auto DF = [](int tag) {
                return std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{tag, Register::Type::Scalar, DataType::UInt32});
            };

            auto one  = Expression::literal(1u);
            auto zero = Expression::literal(0u);

            auto workgroup = graph.coordinates.addElement(Workgroup(0, one));

            // Read tile
            auto receiveTileTag = graph.control.addElement(ConditionalOp{receiveTileExpr});

            auto loadTileTag = graph.control.addElement(LoadTiled(dataType));
            for(auto const& c : loadConnections)
                graph.mapper.connect(loadTileTag, c.coordinate, c.connectionSpec);

            // Read flag
            auto plusOneTag    = graph.coordinates.addElement(Linear(one, one));
            auto setPlusOneTag = graph.control.addElement(SetCoordinate(one));
            graph.mapper.connect<Linear>(setPlusOneTag, plusOneTag);

            auto nextWorkgroupTag = graph.coordinates.addElement(Linear(nullptr, one));
            graph.coordinates.addElement(Split(), {nextWorkgroupTag}, {workgroup, plusOneTag});

            auto flagRegister = graph.coordinates.addElement(VGPR());
            auto loadFlagTag  = graph.control.addElement(LoadSGPR(DataType::UInt32, true));

            auto numScratch = Expression::literal(context->kernelOptions().numScratchTiles);
            auto boundsCheckTag
                = graph.control.addElement(ConditionalOp{(DF(workgroup) + one < numScratch)});

            graph.mapper.connect<User>(loadFlagTag, flagsScratchTag);
            graph.mapper.connect<VGPR>(loadFlagTag, flagRegister);
            graph.coordinates.addElement(PassThrough(), {flagsScratchTag}, {nextWorkgroupTag});

            auto doWhileTag = graph.control.addElement(
                DoWhileOp{(DF(flagRegister) == zero), "Global sync spin loop"});

            // Copy AGPRs to VGPRs before adding fixup
            auto accumulatorTile = graph.coordinates.get<MacroTile>(accumulatorTileTag);
            uint numRegisters
                = accumulatorTile->elements() / product(context->kernel()->workgroupSize());
            auto fullyAccumulatedTileTag = graph.coordinates.addElement(MacroTile());
            auto copyAssignTag           = copyAssign(
                graph, accumulatorTileTag, fullyAccumulatedTileTag, dataType, numRegisters);

            replaceMacroTile(
                graph, usesAccumulatorTile, accumulatorTileTag, fullyAccumulatedTileTag);

            auto fixupTag = addFixup(graph,
                                     fullyAccumulatedTileTag,
                                     scratchTileTag,
                                     fullyAccumulatedTileTag,
                                     dataType,
                                     numRegisters);

            // Add to control
            auto preWaitZeroTag  = graph.control.addElement(WaitZero());
            auto postWaitZeroTag = graph.control.addElement(WaitZero());

            graph.control.chain<Sequence>(preWaitZeroTag, copyAssignTag, receiveTileTag);

            graph.control.addElement(Body(), {receiveTileTag}, {boundsCheckTag});
            graph.control.addElement(Body(), {boundsCheckTag}, {doWhileTag});
            graph.control.addElement(Body(), {doWhileTag}, {loadFlagTag});

            graph.control.chain<Sequence>(doWhileTag, loadTileTag, fixupTag, postWaitZeroTag);

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
                               ExpressionPtr           numWGs,
                               ContextPtr              context)
            : m_dimensionIndices(dims)
            , m_topLoop(topLoop)
            , m_accumulatorLoop(accumulatorLoop)
            , m_numWGs(numWGs)
            , m_context(context)
        {
            m_numTileArgExprs.resize(m_dimensionIndices.size() + 1);
        }

        int AddStreamK::addTileSpaceCT(KernelGraph&  graph,
                                       bool          forward,
                                       ExpressionPtr numTotalTiles,
                                       ExpressionPtr numTilesPerWG)
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
            auto toUInt32 = [](ExpressionPtr expr) -> ExpressionPtr {
                return std::make_shared<Expression::Expression>(
                    Expression::Convert<DataType::UInt32>{expr});
            };

            m_accumulatorCoord = getForLoopCoords(m_accumulatorLoopOp, graph).first;
            Log::debug("  accumulator loop coord: {}", m_accumulatorCoord);

            // Find accumulator tile: look above accumulator-loop for an assign statement
            m_accumulatorTile         = -1;
            auto maybeAccumulatorInit = only(graph.control.findElements([&](int tag) -> bool {
                auto maybeAssign = graph.control.get<Assign>(tag);
                if(!maybeAssign)
                    return false;
                auto nextTag = only(graph.control.getOutputNodeIndices<Sequence>(tag));
                return nextTag.value_or(-1) == m_accumulatorLoopOp;
            }));
            if(maybeAccumulatorInit)
            {
                auto init            = graph.control.get<Assign>(*maybeAccumulatorInit);
                m_accumulatorVarType = resultVariableType(init->expression);

                auto dst            = graph.mapper.get(*maybeAccumulatorInit, NaryArgument::DEST);
                auto maybeAccumTile = graph.coordinates.get<MacroTile>(dst);
                if(maybeAccumTile)
                    m_accumulatorTile = dst;
            }

            ControlFlowRWTracer tracer(graph);
            for(auto m : tracer.coordinatesReadWrite(m_accumulatorTile))
            {
                if(graph.control.compareNodes(m_accumulatorLoopOp, m.control)
                   == NodeOrdering::LeftFirst)
                {
                    m_usesAccumulatorTile.insert(m.control);
                }
            }

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
                            m_numTileArgExprs[dimension] = toUInt32(macTileNumber->size);
                    }
                }
                m_numTileArgExprs.back()
                    = toUInt32(graph.coordinates.get<ForLoop>(m_accumulatorCoord)->size);

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
            // Create new Scope and insert it above the top-loop
            //
            auto scope = graph.control.addElement(Scope());
            replaceWith(graph, m_topLoopOp, scope, false);

            //
            // Compute size of global and local tile-spaces
            //
            auto accumulatorCoordSize = m_numTileArgExprs.back();
            auto numTotalTiles        = accumulatorCoordSize;
            for(auto d : m_dimensionIndices)
                numTotalTiles = numTotalTiles * m_numTiles.at(d);
            numTotalTiles = simplify(numTotalTiles);

            auto numTilesVarType = resultType(numTotalTiles).varType;
            auto one             = Expression::literal(1, numTilesVarType);
            auto zero            = Expression::literal(0, numTilesVarType);

            Log::debug("  accumulatorCoordSize: {}", toString(accumulatorCoordSize));
            Log::debug("  numTotalTiles: {}", toString(numTotalTiles));
            Log::debug("        varType: {}", toString(numTilesVarType));
            Log::debug("  numTilesPerWG: {}", toString(m_numTilesPerWG));

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
            // Add forward/reverse tile-space coordinate transforms
            //
            auto forwardTilesByWGs = addTileSpaceCT(graph, true, numTotalTiles, m_numTilesPerWG);
            auto reverseTilesByWGs = addTileSpaceCT(graph, false, numTotalTiles, m_numTilesPerWG);

            //
            // Add local-tile and accumulator for-loop dimensions and iterators
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
            // Create local-tile loop
            //
            auto wgTilesOuterExpr    = DF(forTileIncr) < m_numTilesPerWG;
            auto totalTilesOuterExpr = (m_numTilesPerWG * wgExpr + DF(forTileIncr)) < numTotalTiles;
            auto incrementOuterExpr  = DF(forTileIncr) + DF(forAccumIncr);

            auto forTileOp = conditionalFor(graph,
                                            forTileIncr,
                                            wgTilesOuterExpr && totalTilesOuterExpr,
                                            incrementOuterExpr,
                                            "StreamTileLoop",
                                            numTilesVarType);

            graph.coordinates.addElement(DataFlow(), {forTileIncr}, {forwardForTileIdx});
            graph.coordinates.addElement(DataFlow(), {forTileIncr}, {reverseForTileIdx});

            //
            // Remove DataFlow edge from accumulator increment to
            // accumulator ForLoop dimension.  Add new DataFlow edges
            // from accumulator increment to forward and reverse
            // accumulator ForLoop dimensions.
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
            // Hi-jack accumulator loop.
            //
            // The old accumulator loop was a simple range-based loop.
            // The new accumulator loop streams the accumulator tile
            // index.
            //
            auto currentNonAccumTileTag = graph.coordinates.addElement(Dimension());
            auto currentNonAccumTileExpr
                = (m_numTilesPerWG * wgExpr + DF(forTileIncr)) / accumulatorCoordSize;
            currentNonAccumTileExpr = Expression::fastDivision(currentNonAccumTileExpr, m_context);

            auto assignNonAccumTile
                = graph.control.addElement(Assign{Register::Type::Scalar, currentNonAccumTileExpr});
            graph.mapper.connect(assignNonAccumTile, currentNonAccumTileTag, NaryArgument::DEST);

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

            //
            // After the accumulator loop, we might need to send and/or
            // receive partially accumulated tiles.
            //
            ScratchInfo scratchTileInfo;
            SendInfo    sendInfo;
            RecvInfo    receiveInfo;
            int         postAccumulationCond;
            if(m_accumulatorTile != -1)
            {
                // Create scratch space for flags
                auto flagsScratch
                    = newScratchCoordinate(literal(m_context->kernelOptions().numScratchTiles),
                                           DataType::UInt32,
                                           m_context);
                auto flagsScratchTag = graph.coordinates.addElement(flagsScratch);

                // Create scratch space for partially accumulated tiles
                std::vector<DeferredConnection> storeConnections, loadConnections;
                scratchTileInfo = loadStoreMacroTileSCRATCH(graph,
                                                            storeConnections,
                                                            loadConnections,
                                                            m_accumulatorTile,
                                                            m_accumulatorVarType,
                                                            m_context);

                // Add send
                auto sendTileExpr
                    = (m_numTilesPerWG * wgExpr + DF(forTileIncr)) % accumulatorCoordSize;
                sendTileExpr = Expression::fastDivision(sendTileExpr, m_context);
                sendInfo     = sendTile(graph,
                                    sendTileExpr,
                                    storeConnections,
                                    flagsScratchTag,
                                    m_accumulatorVarType.dataType,
                                    numTilesVarType,
                                    m_context);

                // Add receive
                auto receiveTileExpr
                    = (m_numTilesPerWG * wgExpr + DF(forTileIncr) + DF(forAccumIncr) - one)
                      % accumulatorCoordSize;
                receiveTileExpr = Expression::fastDivision(receiveTileExpr, m_context);
                receiveInfo     = receiveTile(graph,
                                          receiveTileExpr < (accumulatorCoordSize - one),
                                          scratchTileInfo.load,
                                          loadConnections,
                                          flagsScratchTag,
                                          m_accumulatorTile,
                                          m_usesAccumulatorTile,
                                          m_accumulatorVarType.dataType,
                                          m_context);

                postAccumulationCond
                    = graph.control.addElement(ConditionalOp{zero >= DF(sendInfo.sendBoolSGPR)});
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
            }

            //
            // Add definitions to the Scope
            //
            std::vector<int> defineChain;
            for(auto tag : {forTileIncr, forAccumIncr, currentNonAccumTileTag})
            {
                auto def = graph.control.addElement(Assign{Register::Type::Scalar, zero});
                graph.mapper.connect(def, tag, NaryArgument::DEST);
                defineChain.push_back(def);
            }

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

            //
            // Add local-tile loop
            //
            graph.control.addElement(Sequence(), {defineChain.back()}, {forTileOp});

            //
            // Add accumulator loop; send and receive after
            //
            std::vector<int> postAccumulationOperations;
            for(auto tag : graph.control.getNeighbours<GD::Downstream>(m_accumulatorLoopOp))
            {
                auto maybeSequence = graph.control.get<Sequence>(tag);
                if(maybeSequence)
                {
                    postAccumulationOperations.push_back(
                        *only(graph.control.getNeighbours<GD::Downstream>(tag)));
                    graph.control.deleteElement(tag);
                }
            }

            graph.control.addElement(Body(), {forTileOp}, {receiveInfo.setPlusOne});
            graph.control.addElement(
                Body(), {receiveInfo.setPlusOne}, {scratchTileInfo.setPlusOne});
            graph.control.addElement(Body(), {scratchTileInfo.setPlusOne}, {assignNonAccumTile});
            graph.control.addElement(Sequence(), {assignNonAccumTile}, {m_topLoopOp});

            graph.control.addElement(
                Sequence(), {m_accumulatorLoopOp}, {sendInfo.assignSendBoolSGPR});
            graph.control.addElement(
                Sequence(), {sendInfo.assignSendBoolSGPR}, {sendInfo.preWaitZero});
            graph.control.addElement(Sequence(), {sendInfo.sendCond}, {receiveInfo.preWaitZero});

            graph.control.addElement(Sequence(), {receiveInfo.receiveCond}, {postAccumulationCond});

            // Make sure receive happens before other operations after the accumulator loop
            for(auto tag : postAccumulationOperations)
            {
                graph.control.addElement(Body(), {postAccumulationCond}, {tag});
            }
        }

        void AddStreamK::setupArguments()
        {
            // On entry, numWGs is an Expression that either:
            //   1. Pulls a value from a CommandArgument
            //   2. Is a literal (for testing)

            auto numWGsDT   = DataType::UInt32;
            auto numTilesDT = DataType::UInt32;

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

            // Find the loop control nodes
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
