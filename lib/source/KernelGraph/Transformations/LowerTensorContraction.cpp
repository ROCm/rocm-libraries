
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTensorContraction.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /**
         * Replace the use of an old macrotile in the given control nodes with a new macrotile
         */
        void replaceMacroTile(KernelGraph&             graph,
                              std::unordered_set<int>& ops,
                              int                      oldMacTileTag,
                              int                      newMacTileTag)
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

        int copyAssign(KernelGraph& graph,
                       int          srcMacTileTag,
                       int          destMacTileTag,
                       DataType     dataType,
                       uint         numAGPRs)
        {
            auto srcExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{srcMacTileTag, Register::Type::Accumulator, dataType});

            auto copyAssignTag
                = graph.control.addElement(Assign{Register::Type::Vector, srcExpr, numAGPRs});

            graph.coordinates.addElement(DataFlow(), {srcMacTileTag}, {destMacTileTag});
            graph.mapper.connect(copyAssignTag, destMacTileTag, NaryArgument::DEST);

            return copyAssignTag;
        }

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

        std::tuple<int, int, int>
            loadStoreMacroTileSCRATCH(KernelGraph&                     graph,
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

            loadConnections.push_back(DC<MacroTile>(loadScratchTileTag));
            loadConnections.push_back(DC<User>(globalScratchTag));

            return {storeScratchTileTag, loadScratchTileTag, globalScratchTag};
        }

        int storeScratchTile(KernelGraph& graph, int afterOpTag, DataType dataType)
        {
            auto storePartialTag = graph.control.addElement(StoreTiled(dataType));
            graph.control.addElement(Sequence(), {afterOpTag}, {storePartialTag});
            return storePartialTag;
        }

        int loadScratchTile(KernelGraph& graph, int afterOpTag, DataType dataType)
        {
            auto loadPartialTag = graph.control.addElement(LoadTiled(dataType));
            graph.control.addElement(Sequence(), {afterOpTag}, {loadPartialTag});
            return loadPartialTag;
        }

        int enableScratch(KernelGraph&             graph,
                          int                      contractionTag,
                          std::unordered_set<int>& uses,
                          int                      forK,
                          int                      macTileTag,
                          VariableType             varType,
                          uint                     numAGPRs,
                          ContextPtr               context)
        {
            std::vector<DeferredConnection> storeConnections, loadConnections;

            auto [storeScratchTileTag, loadScratchTileTag, globalScratchTag]
                = loadStoreMacroTileSCRATCH(
                    graph, storeConnections, loadConnections, macTileTag, varType, context);

            auto storePartialTag = storeScratchTile(graph, forK, varType.dataType);
            auto waitZeroTag     = graph.control.addElement(WaitZero());
            auto one             = Expression::literal(1u);
            auto zero            = Expression::literal(0u);

            for(auto const& c : storeConnections)
                graph.mapper.connect(storePartialTag, c.coordinate, c.connectionSpec);

            // Store Flag
            auto flagsScratch = newScratchCoordinate(
                literal(context->kernelOptions().numScratchTiles), DataType::UInt32, context);
            auto flagsScratchTag = graph.coordinates.addElement(flagsScratch);
            auto wg              = graph.coordinates.addElement(Workgroup(0, one));
            auto wgExpr          = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{wg, Register::Type::Scalar, DataType::UInt32});
            graph.coordinates.addElement(PassThrough(), {wg}, {flagsScratchTag});

            auto flagVGPR = graph.coordinates.addElement(VGPR());

            auto assignFlag = graph.control.addElement(Assign{Register::Type::Scalar, one});
            graph.mapper.connect(assignFlag, flagVGPR, NaryArgument::DEST);

            auto storeFlag = graph.control.addElement(StoreSGPR(rocRoller::DataType::Int32, true));
            graph.mapper.connect<User>(storeFlag, flagsScratchTag);
            graph.mapper.connect<VGPR>(storeFlag, flagVGPR);

            //ReadFlag for workgroup + 1
            auto elemNum      = graph.coordinates.addElement(Linear(0, one));
            auto setCoordElem = graph.control.addElement(SetCoordinate(one));
            graph.mapper.connect<ElementNumber>(setCoordElem, elemNum);

            // clear waitcnt queues before ConditionalOp
            // TODO : remove this WaitZero after waitcount observer handles if/else control flow.
            auto waitZeroTag2 = graph.control.addElement(WaitZero());

            graph.control.chain<Sequence>(storePartialTag, waitZeroTag, assignFlag, setCoordElem);

            graph.control.addElement(Body{}, {setCoordElem}, {storeFlag});

            // create destMacTileTag before the if/else part
            auto destMacTileTag = graph.coordinates.addElement(MacroTile());
            replaceMacroTile(graph, uses, macTileTag, destMacTileTag);

            auto flagExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{flagVGPR, Register::Type::Scalar, DataType::UInt32});

            auto numScratch = Expression::literal(context->kernelOptions().numScratchTiles);

            auto conditionalTag = graph.control.addElement(
                ConditionalOp{(flagExpr == one) && (wgExpr + one < numScratch)});

            graph.control.chain<Sequence>(storeFlag, waitZeroTag2, conditionalTag);

            auto loadPartialTag = graph.control.addElement(LoadTiled(varType.dataType));
            for(auto const& c : loadConnections)
                graph.mapper.connect(loadPartialTag, c.coordinate, c.connectionSpec);

            auto newWG = graph.coordinates.addElement(Linear());
            graph.coordinates.addElement(Split(), {newWG}, {wg, elemNum});
            auto loadFlag = graph.control.addElement(LoadSGPR(DataType::UInt32, true));

            graph.mapper.connect<User>(loadFlag, flagsScratchTag);
            graph.mapper.connect<VGPR>(loadFlag, flagVGPR);
            graph.coordinates.addElement(PassThrough(), {flagsScratchTag}, {newWG});

            // WaitCycle --> True Body.
            // If the condition is true to do fixup, we check to see if wg + 1 is out of bounds for
            // flagsArray in scratch. If the wg + 1 is in bounds we perform the wait cycle.
            // Else continue.
            auto doWhile    = graph.control.addElement(DoWhileOp{(flagExpr == zero), "WaitCycle"});
            auto trueBranch = graph.control.addElement(Body(), {conditionalTag}, {doWhile});
            graph.control.addElement(Body(), {doWhile}, {loadFlag});
            graph.control.addElement(Sequence(), {doWhile}, {loadPartialTag});

            // add fixup
            // destMacTileTag = macTileTag + loadScratchTileTag
            auto fixupTag = addFixup(
                graph, macTileTag, loadScratchTileTag, destMacTileTag, varType.dataType, numAGPRs);
            graph.control.addElement(Sequence(), {loadPartialTag}, {fixupTag});

            // copy AGPRs to VGPRs
            auto copyTag
                = copyAssign(graph, macTileTag, destMacTileTag, varType.dataType, numAGPRs);
            graph.control.addElement(Else(), {conditionalTag}, {copyTag});

            return conditionalTag;
        }

        void duplicateMacroTile(KernelGraph& graph, int load)
        {
            auto original = graph.mapper.get<MacroTile>(load);
            auto newMacroTile
                = graph.coordinates.addElement(graph.coordinates.getElement(original));
            graph.coordinates.addElement(PassThrough(), {newMacroTile}, {original});
            graph.mapper.disconnect<MacroTile>(load, original);
            graph.mapper.connect<MacroTile>(load, newMacroTile);
        }

        void addConnectionsMultiply(KernelGraph& graph, int waveMult, int loadATag, int loadBTag)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::addConnectionsMultiply(): Multiply({})", waveMult);

            auto loadA = graph.control.getElement(loadATag);
            auto loadB = graph.control.getElement(loadBTag);
            AssertFatal(isOperation<LoadTiled>(loadA) && isOperation<LoadTiled>(loadB),
                        "Both operands should be LoadTiled");

            // LoadTiled A
            auto userATag = graph.mapper.get<User>(loadATag);
            AssertFatal(userATag > 0, "User dimension not found");
            graph.mapper.connect<User>(waveMult, userATag, 0);

            // LoadTiled B
            auto userBTag = graph.mapper.get<User>(loadBTag);
            AssertFatal(userBTag > 0, "User dimension not found");
            graph.mapper.connect<User>(waveMult, userBTag, 1);

            AssertFatal(userATag > 0 && userBTag > 0, "User dimensions not found");

            auto [waveATag, waveA] = graph.getDimension<WaveTile>(loadATag);
            auto [waveBTag, waveB] = graph.getDimension<WaveTile>(loadBTag);

            auto macroTileA = graph.mapper.get<MacroTile>(loadATag);
            auto macroTileB = graph.mapper.get<MacroTile>(loadBTag);

            graph.mapper.connect(
                waveMult, macroTileA, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
            graph.mapper.connect(
                waveMult, macroTileB, Connections::typeArgument<MacroTile>(NaryArgument::RHS));
            graph.mapper.connect(
                waveMult, waveATag, Connections::typeArgument<WaveTile>(NaryArgument::LHS));
            graph.mapper.connect(
                waveMult, waveBTag, Connections::typeArgument<WaveTile>(NaryArgument::RHS));
        }

        /**
         * Lower rank-2 TensorContraction into a matrix multiply.
         */
        void lowerMatrixMultiply(KernelGraph&                       graph,
                                 int                                tag,
                                 int                                a,
                                 int                                b,
                                 int                                d,
                                 std::shared_ptr<CommandParameters> params,
                                 std::shared_ptr<Context>           context)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerMatrixMultiply({})", tag);

            // find the control nodes that follow the contraction and use its output macrotile
            ControlFlowRWTracer tracer(graph);

            std::unordered_set<int> uses;
            for(auto m : tracer.coordinatesReadWrite(d))
            {
                if(graph.control.compareNodes(tag, m.control) == NodeOrdering::LeftFirst)
                    uses.insert(m.control);
            }

            auto macrotile_a = graph.coordinates.getNode<MacroTile>(a);
            auto macrotile_b = graph.coordinates.getNode<MacroTile>(b);

            // get tensor contraction operands
            auto parents = graph.control.parentNodes(tag).to<std::vector>();
            AssertFatal(parents.size() == 2);
            auto operandA = parents[0];
            auto operandB = parents[1];
            AssertFatal(a == graph.mapper.get<MacroTile>(operandA));
            AssertFatal(b == graph.mapper.get<MacroTile>(operandB));

            auto reachable_from_tensor
                = graph.control.depthFirstVisit(tag).to<std::unordered_set>();

            // find loadtiled ops that lead to the tensor contraction op
            // find storetiled ops that follow the tensor contraction op
            // find elementOps that follow the tensor contraction op
            std::vector<int> loadA;
            std::vector<int> loadB;
            std::vector<int> stores;
            std::vector<int> assigns;
            for(auto const index : graph.control.getNodes())
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) {},
                          [&](LoadTiled const& load) {
                              auto reachable_from_load
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              if(reachable_from_load.find(operandA) != reachable_from_load.end())
                                  loadA.push_back(index);
                              else if(reachable_from_load.find(operandB)
                                      != reachable_from_load.end())
                                  loadB.push_back(index);
                          },
                          [&](StoreTiled const& store) {
                              if(reachable_from_tensor.find(index) != reachable_from_tensor.end())
                                  stores.push_back(index);
                          },
                          [&](Assign const& op) {
                              if(reachable_from_tensor.find(index) != reachable_from_tensor.end())
                                  assigns.push_back(index);
                          }},
                      std::get<Operation>(elem));
            }
            AssertFatal(loadA.size() == 1 && loadB.size() == 1);
            AssertFatal(stores.size() <= 1);
            auto storeD = !stores.empty() ? stores[0] : -1;

            // find kernel
            auto root = graph.control.roots().to<std::vector>();
            AssertFatal(root.size() == 1, "More than one Kernel node not supported");
            auto kernel = root[0];

            graph.control.deleteElement<Body>(std::vector<int>{kernel}, std::vector<int>{loadA[0]});
            graph.control.deleteElement<Body>(std::vector<int>{kernel}, std::vector<int>{loadB[0]});

            auto matK = (graph.coordinates.getNode<SubDimension>(
                             graph.mapper.get<SubDimension>(loadA[0], 1)))
                            .size;

            auto macK      = literal(static_cast<uint>(macrotile_a.sizes[1])); // M x K
            auto [K, forK] = rangeFor(
                graph, matK / macK, rocRoller::KLOOP); // num of loop iterations : matK / macK

            auto a_tilenum_y = graph.mapper.get<MacroTileNumber>(loadA[0], 1);
            auto b_tilenum_x = graph.mapper.get<MacroTileNumber>(loadB[0], 0);

            // A row block is x-workgroup, column block is for loop index
            graph.coordinates.addElement(PassThrough(), {a_tilenum_y}, {K});

            // B row block is for loop index, column block is y-workgroup
            graph.coordinates.addElement(PassThrough(), {b_tilenum_x}, {K});

            auto [waveA_tag, waveA] = graph.getDimension<WaveTile>(loadA[0]);
            auto [waveB_tag, waveB] = graph.getDimension<WaveTile>(loadB[0]);
            uint num_elements       = waveA.sizes[0] * waveB.sizes[1];
            uint wfs                = context->kernel()->wavefront_size();
            uint numAGPRs           = num_elements / wfs; // number of output registers per thread

            auto initD = graph.control.addElement(
                Assign{Register::Type::Accumulator, literal(0.f), numAGPRs});

            graph.mapper.connect(initD, d, NaryArgument::DEST);

            graph.control.addElement(Sequence(), {initD}, {forK});

            auto waveTileNumYA = graph.mapper.get<WaveTileNumber>(loadA[0], 1);
            auto waveTileNumXB = graph.mapper.get<WaveTileNumber>(loadB[0], 0);

            // Add an unroll dimension that connects to both A's WaveTileNumber[1] and B's
            // WaveTileNumber[0]. This is because we are unrolling the "small k" loop.
            uint const num_wave_tiles = macrotile_a.sizes[1] / waveA.sizes[1];
            auto       smallKUnroll   = graph.coordinates.addElement(Unroll(num_wave_tiles));
            graph.coordinates.addElement(PassThrough(), {waveTileNumYA}, {smallKUnroll});
            graph.coordinates.addElement(PassThrough(), {waveTileNumXB}, {smallKUnroll});

            int lastWaveMult  = -1;
            int lastSetCoordA = -1;
            int lastSetCoordB = -1;
            for(uint k = 0; k < num_wave_tiles; k++)
            {
                auto setCoordA = graph.control.addElement(SetCoordinate(literal(k)));
                graph.mapper.connect<Unroll>(setCoordA, smallKUnroll);
                graph.control.addElement(Body(), {forK}, {setCoordA});

                auto newLoadA = duplicateControlNode(graph, loadA[0]);
                if(k != 0)
                    duplicateMacroTile(graph, newLoadA);
                graph.control.addElement(Body(), {setCoordA}, {newLoadA});

                auto setCoordB = graph.control.addElement(SetCoordinate(literal(k)));
                graph.mapper.connect<Unroll>(setCoordB, smallKUnroll);
                graph.control.addElement(Body(), {forK}, {setCoordB});

                auto newLoadB = duplicateControlNode(graph, loadB[0]);
                if(k != 0)
                    duplicateMacroTile(graph, newLoadB);
                graph.control.addElement(Body(), {setCoordB}, {newLoadB});

                auto waveMult = graph.control.addElement(Multiply());
                graph.mapper.connect(
                    waveMult, d, Connections::typeArgument<MacroTile>(NaryArgument::DEST));

                graph.control.addElement(Sequence(), {setCoordA}, {waveMult});
                graph.control.addElement(Sequence(), {setCoordB}, {waveMult});
                graph.control.addElement(Sequence(), {setCoordA}, {setCoordB});

                addConnectionsMultiply(graph, waveMult, newLoadA, newLoadB);
                if(lastWaveMult >= 0)
                {
                    graph.control.addElement(Sequence(), {lastWaveMult}, {waveMult});
                    graph.control.addElement(Sequence(), {lastSetCoordA}, {setCoordA});
                    graph.control.addElement(Sequence(), {lastSetCoordA}, {setCoordB});
                    graph.control.addElement(Sequence(), {lastSetCoordB}, {setCoordA});
                    graph.control.addElement(Sequence(), {lastSetCoordB}, {setCoordB});
                }

                lastWaveMult  = waveMult;
                lastSetCoordA = setCoordA;
                lastSetCoordB = setCoordB;
            }

            int finalContractionOp = forK;
            if(context->kernelOptions().enableScratch)
            {
                auto varType = getVariableType(graph, storeD);
                finalContractionOp
                    = enableScratch(graph, tag, uses, forK, d, varType, numAGPRs, context);
            }

            // connect ops after contraction to finalContractionOp, remove contraction and its incoming edges
            auto tensor_outgoing_edges
                = graph.control.getNeighbours<Graph::Direction::Downstream>(tag).to<std::vector>();
            for(auto const e : tensor_outgoing_edges)
            {
                auto elem = graph.control.getElement(e);
                auto dst  = graph.control.getNeighbours<Graph::Direction::Downstream>(e)
                               .to<std::vector>();
                graph.control.deleteElement(e);
                graph.control.addElement(e, elem, std::vector<int>{finalContractionOp}, dst);
            }
            auto tensor_incoming_edges
                = graph.control.getNeighbours<Graph::Direction::Upstream>(tag).to<std::vector>();
            for(auto const e : tensor_incoming_edges)
                graph.control.deleteElement(e);
            graph.control.deleteElement(tag);
            graph.mapper.purge(tag);

            // Add loops to iterate over wavetiles within a wavefront
            auto wavetilesPerWorkgroup = params->getWaveTilesPerWorkgroup();
            AssertFatal(wavetilesPerWorkgroup.size() > 1);

            auto [WaveTilesX, forWaveTilesX]
                = rangeFor(graph, literal(wavetilesPerWorkgroup[0]), rocRoller::XLOOP);
            auto [WaveTilesY, forWaveTilesY]
                = rangeFor(graph, literal(wavetilesPerWorkgroup[1]), rocRoller::YLOOP);

            // find other loadtiled ops from kernel that lead to assigns
            auto             kernel_outputs = graph.control.childNodes(kernel).to<std::vector>();
            std::vector<int> otherLoads;
            std::vector<int> otherOps;
            for(auto const index : kernel_outputs)
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) { otherOps.push_back(index); },
                          [&](LoadTiled const& load) {
                              auto reachable_from_load
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              for(auto const& assign : assigns)
                              {
                                  if(reachable_from_load.find(assign) != reachable_from_load.end())
                                  {
                                      otherLoads.push_back(index);
                                      break;
                                  }
                              }
                          }},
                      std::get<Operation>(elem));
            }
            AssertFatal(otherLoads.size() <= 1);

            // Add edges from inner loop to some kernel outputs : forK and otherLoads
            // need to leave other nodes attached with kernel
            // ex: loadtiled ops that don't lead to assigns
            // ex : loadVGPRs for alpha and beta in GEMM

            graph.control.addElement(Body(), {forWaveTilesY}, {initD});

            for(auto const index : otherLoads)
            {
                auto e = graph.control.getNeighbours<Graph::Direction::Upstream>(index)
                             .to<std::vector>()[0];
                auto elem = graph.control.getElement(e);
                graph.control.deleteElement(e);
                // TODO: This explicitly puts the + beta * C portion of a GEMM after the
                //       forK loop. We might want to remove this after the dynamic
                //       scheduling has been implemented.
                //graph.control.addElement(
                //    e, elem, std::vector<int>{forWaveTilesY}, std::vector<int>{index});
                graph.control.addElement(Sequence(), {finalContractionOp}, {index});
            }

            for(auto const index : otherOps)
            {
                auto e = graph.control.getNeighbours<Graph::Direction::Downstream>(index).only();
                AssertFatal(e.has_value());

                auto elem = graph.control.getElement(*e);
                graph.control.deleteElement(*e);
                graph.control.addElement(
                    *e, elem, std::vector<int>{index}, std::vector<int>{forWaveTilesX});
            }

            // make nested inner loops (forWaveTilesY inside the forWaveTilesX loop)
            graph.control.addElement(Body(), {forWaveTilesX}, {forWaveTilesY});

            // Add edges from Kernel to outer loop
            graph.control.addElement(Body(), {kernel}, {forWaveTilesX});

            // Add edges from all JammedWaveTileNumber dimensions to the for loop
            auto a_jammed_x = graph.mapper.get<JammedWaveTileNumber>(loadA[0], 0);
            graph.coordinates.addElement(PassThrough(), {a_jammed_x}, {WaveTilesX});
            auto b_jammed_x = graph.mapper.get<JammedWaveTileNumber>(loadB[0], 0);
            graph.coordinates.addElement(PassThrough(), {b_jammed_x}, {WaveTilesX});

            auto a_jammed_y = graph.mapper.get<JammedWaveTileNumber>(loadA[0], 1);
            graph.coordinates.addElement(PassThrough(), {a_jammed_y}, {WaveTilesY});
            auto b_jammed_y = graph.mapper.get<JammedWaveTileNumber>(loadB[0], 1);
            graph.coordinates.addElement(PassThrough(), {b_jammed_y}, {WaveTilesY});

            if(otherLoads.size() > 0)
            {
                auto c_jammed_x = graph.mapper.get<JammedWaveTileNumber>(otherLoads[0], 0);
                graph.coordinates.addElement(PassThrough(), {c_jammed_x}, {WaveTilesX});
                auto c_jammed_y = graph.mapper.get<JammedWaveTileNumber>(otherLoads[0], 1);
                graph.coordinates.addElement(PassThrough(), {c_jammed_y}, {WaveTilesY});
            }

            if(storeD > 0)
            {
                auto d_jammed_x = graph.mapper.get<JammedWaveTileNumber>(storeD, 0);
                graph.coordinates.addElement(PassThrough(), {WaveTilesX}, {d_jammed_x});
                auto d_jammed_y = graph.mapper.get<JammedWaveTileNumber>(storeD, 1);
                graph.coordinates.addElement(PassThrough(), {WaveTilesY}, {d_jammed_y});
            }

            // Delete original loadA and loadB.
            for(auto edge :
                graph.control.getNeighbours<Graph::Direction::Downstream>(loadA[0]).to<std::set>())
            {
                graph.control.deleteElement(edge);
            }
            for(auto edge :
                graph.control.getNeighbours<Graph::Direction::Upstream>(loadA[0]).to<std::set>())
            {
                graph.control.deleteElement(edge);
            }
            for(auto edge :
                graph.control.getNeighbours<Graph::Direction::Downstream>(loadB[0]).to<std::set>())
            {
                graph.control.deleteElement(edge);
            }
            for(auto edge :
                graph.control.getNeighbours<Graph::Direction::Upstream>(loadB[0]).to<std::set>())
            {
                graph.control.deleteElement(edge);
            }
            graph.control.deleteElement(loadA[0]);
            graph.control.deleteElement(loadB[0]);
            graph.mapper.purge(loadA[0]);
            graph.mapper.purge(loadB[0]);
        }

        KernelGraph LowerTensorContraction::apply(KernelGraph const& graph)
        {
            TIMER(t, "KernelGraph::lowerTensorContraction");

            auto contractions = graph.control.getNodes<TensorContraction>().to<std::vector>();
            AssertFatal(contractions.size() <= 1,
                        "More than one TensorContraction not supported yet.");

            if(contractions.size() < 1)
                return graph;

            auto kgraph       = graph;
            auto tag          = contractions[0];
            auto op           = kgraph.control.getNode<TensorContraction>(tag);
            auto [aTag, aMac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::LHS);
            auto [bTag, bMac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::RHS);
            auto [dTag, dMac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::DEST);
            if(aMac.rank == 2 && bMac.rank == 2 && op.aDims == std::vector<int>{1}
               && op.bDims == std::vector<int>{0})
            {
                lowerMatrixMultiply(kgraph, tag, aTag, bTag, dTag, m_params, m_context);
            }
            else
            {
                Throw<FatalError>("General contraction not implemented yet.");
            }

            return kgraph;
        }
    }
}
