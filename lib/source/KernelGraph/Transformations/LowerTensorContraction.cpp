
#include <algorithm>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTensorContraction.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        void duplicateMacroTile(KernelGraph& graph, int load)
        {
            auto original = graph.mapper.get<MacroTile>(load);
            auto newMacroTile
                = graph.coordinates.addElement(graph.coordinates.getElement(original));
            graph.coordinates.addElement(Duplicate(), {newMacroTile}, {original});
            graph.mapper.disconnect<MacroTile>(load, original);
            graph.mapper.connect<MacroTile>(load, newMacroTile);
        }

        void addConnectionsMultiply(KernelGraph& graph, int waveMult, int loadATag, int loadBTag)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTensorContraction::addConnectionsMultiply(): Multiply({})",
                waveMult);

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

        int getUserFromDataFlow(KernelGraph const& graph, int start, Graph::Direction direction)
        {
            auto predicate
                = [&](int x) -> bool { return graph.coordinates.get<DataFlow>(x).has_value(); };
            for(auto elem : graph.coordinates.depthFirstVisit(start, predicate, direction))
            {
                auto maybeUser = graph.coordinates.get<User>(elem);
                if(maybeUser)
                    return elem;
            }
            return -1;
        }

        struct LoadStoreInfo
        {
            int global   = -1; //< Global memory operation: LoadTiled or StoreTiled.
            int storeLDS = -1; //< Store to LDS operation: StoreLDSTile
            int loadLDS  = -1; //< Load from LDS operation: LoadLDSTile

            /// If true, this is a
            ///     LoadTile+StoreLDSTile+LoadLDSTile
            /// chain.  If false, this is a
            ///     StoreLDSTile+LoadLDSTile+StoreTiled
            /// chain.
            bool isLoad;

            /// For load operations, this is the operation and fills
            /// the destination tile.  For non-LDS loads this is the
            /// global operation.  For LDS loads this is the loadLDS
            /// operation.
            int load() const
            {
                return loadLDS == -1 ? global : loadLDS;
            }
        };

        Graph::Direction getCoordinateAddDirection(KernelGraph& graph, int opTag)
        {
            auto isLoadTiled   = graph.control.get<LoadTiled>(opTag).has_value();
            auto isLoadLDSTile = graph.control.get<LoadLDSTile>(opTag).has_value();
            if(isLoadTiled || isLoadLDSTile)
                return Graph::Direction::Downstream;
            auto isStoreTiled   = graph.control.get<StoreTiled>(opTag).has_value();
            auto isStoreLDSTile = graph.control.get<StoreLDSTile>(opTag).has_value();
            if(isStoreTiled || isStoreLDSTile)
                return Graph::Direction::Upstream;
            Throw<FatalError>("Cannot determine direction: invalid operation.");
        }

        /**
         * @brief
         */
        void connectJammedOperation(KernelGraph& graph, int opTag, int waveTilesX, int waveTilesY)
        {
            if(opTag == -1)
                return;

            if(getCoordinateAddDirection(graph, opTag) == Graph::Direction::Downstream)
            {
                auto jammedX = graph.mapper.get<JammedWaveTileNumber>(opTag, 0);
                if(jammedX != -1)
                    graph.coordinates.addElement(PassThrough(), {jammedX}, {waveTilesX});
                auto jammedY = graph.mapper.get<JammedWaveTileNumber>(opTag, 1);
                if(jammedY != -1)
                    graph.coordinates.addElement(PassThrough(), {jammedY}, {waveTilesY});
            }
            else
            {
                auto jammedX = graph.mapper.get<JammedWaveTileNumber>(opTag, 0);
                if(jammedX != -1)
                    graph.coordinates.addElement(PassThrough(), {waveTilesX}, {jammedX});
                auto jammedY = graph.mapper.get<JammedWaveTileNumber>(opTag, 1);
                if(jammedY != -1)
                    graph.coordinates.addElement(PassThrough(), {waveTilesY}, {jammedY});
            }
        }

        void connectJammedOperation(KernelGraph&         graph,
                                    LoadStoreInfo const& info,
                                    int                  waveTilesX,
                                    int                  waveTilesY)
        {
            if(info.global == -1)
                return;
            connectJammedOperation(graph, info.global, waveTilesX, waveTilesY);
            connectJammedOperation(graph, info.storeLDS, waveTilesX, waveTilesY);
            connectJammedOperation(graph, info.loadLDS, waveTilesX, waveTilesY);
        }

        void connectJammedOperation(KernelGraph&            graph,
                                    std::vector<int> const& opTags,
                                    int                     waveTilesX,
                                    int                     waveTilesY)
        {
            for(auto opTag : opTags)
                connectJammedOperation(graph, opTag, waveTilesX, waveTilesY);
        }

        void connectGlobalLoadOperations(KernelGraph& graph, int forLoop, LoadStoreInfo const& info)
        {
            if(info.loadLDS == -1)
                return;

            auto edge = *only(graph.control.getNeighbours(info.global, Graph::Direction::Upstream));
            graph.control.deleteElement(edge);
            graph.control.addElement(Body(), {forLoop}, {info.global});

            // XXX: Attach MacroTileNumber of LDS buffers to K ForLoop coordinate
            // graph.coordinates.addElement(
            //     PassThrough(), {K}, {graph.mapper.get<MacroTileNumber>(chain->storeLDSTile)});
            // graph.coordinates.addElement(
            //     PassThrough(), {graph.mapper.get<MacroTileNumber>(chain->loadLDSTile)}, {K});
        }

        /**
         * Returns global-load-and-store-to-lds chain ({top, bottom} pair) above LoadLDSTile.
         */
        std::optional<LoadStoreInfo> getLoadStoreInfo(int op, KernelGraph const& graph)
        {
            auto loadLDS = graph.control.get<LoadLDSTile>(op);
            if(loadLDS)
            {
                auto storeLDSTag = *only(graph.control.getInputNodeIndices<Sequence>(op));
                auto storeLDS    = graph.control.get<StoreLDSTile>(storeLDSTag);
                if(!storeLDS)
                    return {};

                auto loadTileTag = *only(graph.control.getInputNodeIndices<Sequence>(storeLDSTag));
                auto loadTile    = graph.control.get<LoadTiled>(loadTileTag);
                if(!loadTile)
                    return {};

                return LoadStoreInfo{loadTileTag, storeLDSTag, op, true};
            }

            auto store = graph.control.get<StoreTiled>(op);
            if(store)
            {
                auto loadLDSTag = *only(graph.control.getInputNodeIndices<Sequence>(op));
                auto loadLDS    = graph.control.get<LoadLDSTile>(loadLDSTag);
                if(!loadLDS)
                    return LoadStoreInfo{op, -1, -1, false};

                auto storeLDSTag = *only(graph.control.getInputNodeIndices<Sequence>(loadLDSTag));
                auto storeLDS    = graph.control.get<StoreLDSTile>(storeLDSTag);
                if(!storeLDS)
                    return {};

                return LoadStoreInfo{op, storeLDSTag, loadLDSTag, false};
            }

            return {};
        }

        struct MatrixMultiplyInfo
        {
            int                          kernel; //< Kernel operation
            LoadStoreInfo                loadA; //< Load operation that loads the A (LHS) operand
            LoadStoreInfo                loadB; //< Load operation that loads the B (RHS) operand
            std::optional<LoadStoreInfo> storeD; //< Store operation that stores the result (D)
            int                          userA; //< Tag of global A Tensor
            int                          userB; //< Tag of global B Tensor

            std::vector<int> dependentAssigns; //< Assign operations that use the result (D)
            std::vector<int> siblingLoads; //< Load operations that flow into dependentAssigns
            std::vector<int> siblingOps; //< Other operations that flow into dependentAssigns
        };

        MatrixMultiplyInfo getMatrixMultiplyInfo(KernelGraph const& graph, int tensorContractionTag)
        {
            MatrixMultiplyInfo info;

            // Get tensor contraction operands
            int aTag = graph.mapper.get(tensorContractionTag, NaryArgument::LHS);
            int bTag = graph.mapper.get(tensorContractionTag, NaryArgument::RHS);

            info.userA = getUserFromDataFlow(graph, aTag, Graph::Direction::Upstream);
            info.userB = getUserFromDataFlow(graph, bTag, Graph::Direction::Upstream);

            // Double check operands
            {
                auto parents = graph.control.parentNodes(tensorContractionTag).to<std::set>();
                AssertFatal(parents.size() == 2);
                std::set<int> tiles;
                std::for_each(parents.cbegin(), parents.cend(), [&](int op) {
                    tiles.insert(graph.mapper.get<MacroTile>(op));
                });
                AssertFatal(tiles.contains(aTag));
                AssertFatal(tiles.contains(bTag));
            }

            // Find loads, stores, assigns etc
            auto reachableFromTC
                = graph.control.depthFirstVisit(tensorContractionTag).to<std::unordered_set>();

            std::vector<int> stores;
            for(auto const index : graph.control.getNodes())
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{[&](auto op) {},
                                            [&](LoadTiled const& load) {
                                                auto tileTag = graph.mapper.get<MacroTile>(index);
                                                if(tileTag == aTag)
                                                    info.loadA = {index, -1, -1, true};
                                                if(tileTag == bTag)
                                                    info.loadB = {index, -1, -1, true};
                                            },
                                            [&](LoadLDSTile const& load) {
                                                auto tileTag = graph.mapper.get<MacroTile>(index);
                                                if(tileTag == aTag)
                                                    info.loadA = *getLoadStoreInfo(index, graph);
                                                if(tileTag == bTag)
                                                    info.loadB = *getLoadStoreInfo(index, graph);
                                            },
                                            [&](StoreTiled const& store) {
                                                if(reachableFromTC.contains(index))
                                                    stores.push_back(index);
                                            },
                                            [&](Assign const& op) {
                                                if(reachableFromTC.contains(index))
                                                    info.dependentAssigns.push_back(index);
                                            }},
                      std::get<Operation>(elem));
            }

            AssertFatal(info.loadA.global != -1);
            AssertFatal(info.loadB.global != -1);

            AssertFatal(stores.size() <= 1);
            if(!stores.empty())
                info.storeD = getLoadStoreInfo(stores[0], graph);

            auto root = only(graph.control.roots());
            AssertFatal(root, "More than one Kernel node not supported");
            info.kernel = *root;

            // Find sibling loads and ops
            auto upstreamOfTC
                = graph.control.depthFirstVisit(tensorContractionTag, Graph::Direction::Upstream)
                      .to<std::unordered_set>();

            auto filterOutUpstreamOfTC = [&](int x) { return !upstreamOfTC.contains(x); };
            auto kernelOutputs
                = filter(filterOutUpstreamOfTC, graph.control.childNodes(info.kernel))
                      .to<std::vector>();
            for(auto const index : kernelOutputs)
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) { info.siblingOps.push_back(index); },
                          [&](LoadTiled const& load) {
                              auto reachableFromLoad
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              for(auto const& assign : info.dependentAssigns)
                              {
                                  if(reachableFromLoad.contains(assign))
                                  {
                                      info.siblingLoads.push_back(index);
                                      break;
                                  }
                              }
                          },
                      },
                      std::get<Operation>(elem));
            }
            AssertFatal(info.siblingLoads.size() <= 1);

            return info;
        }

        ExpressionPtr getAccumulationLoopSize(KernelGraph const& graph, int tileTag, int userTag)
        {
            auto sdims = graph.coordinates
                             .getOutputNodeIndices(
                                 userTag, rocRoller::KernelGraph::CoordinateGraph::isEdge<Split>)
                             .to<std::vector>();

            auto userA = graph.coordinates.getNode<User>(userTag);
            auto tileA = graph.coordinates.getNode<MacroTile>(tileTag);
            auto matK  = graph.coordinates.getNode<SubDimension>(sdims[1]).size;
            auto macK  = literal(static_cast<uint>(tileA.sizes[1])); // M x K

            auto toUInt32 = [](ExpressionPtr expr) -> ExpressionPtr {
                return std::make_shared<Expression::Expression>(
                    Expression::Convert{{.arg{expr}}, DataType::UInt32});
            };

            return toUInt32(matK / macK);
        }

        int getMacroTileNumber(KernelGraph const& graph, int userTag, int sdim)
        {
            auto [required, path]
                = findRequiredCoordinates(userTag, Graph::Direction::Downstream, graph);
            auto macroTileNumbers = filterCoordinates<MacroTileNumber>(required, graph);
            for(auto mtnTag : macroTileNumbers)
            {
                for(auto input : graph.coordinates.getInputNodeIndices(
                        mtnTag, rocRoller::KernelGraph::CoordinateGraph::isEdge<Tile>))
                {
                    auto maybeSubDimension = graph.coordinates.get<SubDimension>(input);
                    if(!maybeSubDimension)
                        continue;
                    if(maybeSubDimension->dim == sdim)
                        return mtnTag;
                }
            }
            return -1;
        }

        /**
         * Lower rank-2 TensorContraction into a matrix multiply.
         */
        void lowerMatrixMultiply(KernelGraph&             graph,
                                 int                      tag,
                                 int                      a,
                                 int                      b,
                                 int                      d,
                                 CommandParametersPtr     params,
                                 std::shared_ptr<Context> context)
        {
            rocRoller::Log::debug("KernelGraph::lowerMatrixMultiply({})", tag);

            auto info = getMatrixMultiplyInfo(graph, tag);

            auto accumulationCoordSize = getAccumulationLoopSize(graph, a, info.userA);

            auto [K, forK] = rangeFor(graph, accumulationCoordSize, rocRoller::KLOOP);

            // A row block is x-workgroup, column block is for loop index
            // B row block is for loop index, column block is y-workgroup
            //
            // TODO: For macTileNumYA: Look for Number siblings of
            // the first bound Index nodes of the WORKGROUP Tensor
            // above `b`.  Similarly for A.
            auto macTileNumYA = getMacroTileNumber(graph, info.userA, 1);
            auto macTileNumXB = getMacroTileNumber(graph, info.userB, 0);

            rocRoller::Log::debug("  Load A {} MTN {}; Load B {} MTN {}",
                                  info.loadA.load(),
                                  macTileNumYA,
                                  info.loadB.load(),
                                  macTileNumXB);

            graph.coordinates.addElement(PassThrough(), {macTileNumYA}, {K});
            graph.coordinates.addElement(PassThrough(), {macTileNumXB}, {K});

            auto [waveATag, waveA] = graph.getDimension<WaveTile>(info.loadA.load());
            auto [waveBTag, waveB] = graph.getDimension<WaveTile>(info.loadB.load());
            uint num_elements      = waveA.sizes[0] * waveB.sizes[1];
            uint wfs               = context->kernel()->wavefront_size();
            uint numAGPRs          = num_elements / wfs; // number of output registers per thread

            auto initD = graph.control.addElement(
                Assign{Register::Type::Accumulator, literal(0.f), numAGPRs});

            graph.mapper.connect(initD, d, NaryArgument::DEST);

            auto waveTileNumYA = graph.mapper.get<WaveTileNumber>(info.loadA.load(), 1);
            auto waveTileNumXB = graph.mapper.get<WaveTileNumber>(info.loadB.load(), 0);

            rocRoller::Log::debug("  Load A {} WTN {}; Load B {} WTN {}",
                                  info.loadA.load(),
                                  waveTileNumYA,
                                  info.loadB.load(),
                                  waveTileNumXB);

            // Add an unroll dimension that connects to both A's WaveTileNumber[1] and B's
            // WaveTileNumber[0]. This is because we are unrolling the "small k" loop.
            auto tileA = graph.coordinates.getNode<MacroTile>(a);

            uint const numWaveTiles = tileA.sizes[1] / waveA.sizes[1];
            auto       smallKUnroll = graph.coordinates.addElement(Unroll(numWaveTiles));
            graph.coordinates.addElement(PassThrough(), {waveTileNumYA}, {smallKUnroll});
            graph.coordinates.addElement(PassThrough(), {waveTileNumXB}, {smallKUnroll});

            int lastWaveMult   = -1;
            int lastSetCoordA  = -1;
            int lastSetCoordB  = -1;
            int firstSetCoordA = -1;
            int firstSetCoordB = -1;
            for(uint k = 0; k < numWaveTiles; k++)
            {
                auto setCoordA = graph.control.addElement(SetCoordinate(literal(k)));
                graph.mapper.connect<Unroll>(setCoordA, smallKUnroll);
                graph.control.addElement(Body(), {forK}, {setCoordA});
                if(firstSetCoordA == -1)
                    firstSetCoordA = setCoordA;

                auto newLoadA = duplicateControlNode(graph, info.loadA.load());
                if(k != 0)
                    duplicateMacroTile(graph, newLoadA);
                graph.control.addElement(Body(), {setCoordA}, {newLoadA});

                auto setCoordB = graph.control.addElement(SetCoordinate(literal(k)));
                graph.mapper.connect<Unroll>(setCoordB, smallKUnroll);
                graph.control.addElement(Body(), {forK}, {setCoordB});
                if(firstSetCoordB == -1)
                    firstSetCoordB = setCoordB;

                auto newLoadB = duplicateControlNode(graph, info.loadB.load());
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

            // Add loops to iterate over wavetiles within a wavefront
            auto wavetilesPerWavefront = params->getWaveTilesPerWavefront();
            AssertFatal(wavetilesPerWavefront.size() > 1);

            auto [WaveTilesX, forWaveTilesX]
                = rangeFor(graph, literal(wavetilesPerWavefront[0]), rocRoller::XLOOP);
            auto [WaveTilesY, forWaveTilesY]
                = rangeFor(graph, literal(wavetilesPerWavefront[1]), rocRoller::YLOOP);

            auto forWaveTilesEpilogueX = *only(
                duplicateControlNodes(graph, nullptr, {forWaveTilesX}, [](int x) { return true; }));

            auto forWaveTilesEpilogueY = *only(
                duplicateControlNodes(graph, nullptr, {forWaveTilesY}, [](int x) { return true; }));

            auto forWaveTilesEpilogueYNOP = graph.control.addElement(NOP());

            graph.control.addElement(Body(), {info.kernel}, {forWaveTilesX});
            graph.control.addElement(Body(), {forWaveTilesX}, {forWaveTilesY});
            graph.control.addElement(Body(), {forWaveTilesY}, {initD});
            graph.control.addElement(Sequence(), {initD}, {forK});
            graph.control.addElement(Sequence(), {forWaveTilesX}, {forWaveTilesEpilogueX});
            graph.control.addElement(Body(), {forWaveTilesEpilogueX}, {forWaveTilesEpilogueY});
            graph.control.addElement(Body(), {forWaveTilesEpilogueY}, {forWaveTilesEpilogueYNOP});

            // Connect ops after contraction to forK, remove contraction and its incoming edges
            auto tcOutgoingEdges
                = graph.control.getNeighbours<Graph::Direction::Downstream>(tag).to<std::vector>();
            for(auto const e : tcOutgoingEdges)
            {
                auto elem = graph.control.getElement(e);
                auto dst  = graph.control.getNeighbours<Graph::Direction::Downstream>(e)
                               .to<std::vector>();
                graph.control.deleteElement(e);
                graph.control.addElement(
                    Sequence(), std::vector<int>{forWaveTilesEpilogueYNOP}, dst);
            }
            auto tcIncomingEdges
                = graph.control.getNeighbours<Graph::Direction::Upstream>(tag).to<std::vector>();
            for(auto const e : tcIncomingEdges)
                graph.control.deleteElement(e);
            graph.control.deleteElement(tag);
            graph.mapper.purge(tag);

            // Add siblings...
            for(auto const index : info.siblingLoads)
            {
                for(auto e : graph.control.getNeighbours<Graph::Direction::Upstream>(index)
                                 .to<std::vector>())
                {
                    graph.control.deleteElement(e);
                }
                // TODO: This explicitly puts the + beta * C portion of a GEMM after the
                //       forK loop. We might want to remove this after the dynamic
                //       scheduling has been implemented.
                graph.control.addElement(Sequence(), {forWaveTilesEpilogueYNOP}, {index});
            }

            for(auto const index : info.siblingOps)
            {
                auto e = only(graph.control.getNeighbours<Graph::Direction::Downstream>(index))
                             .value();
                auto elem = graph.control.getElement(e);
                graph.control.deleteElement(e);
                graph.control.addElement(
                    e, elem, std::vector<int>{index}, std::vector<int>{forWaveTilesX});
            }

            // Add PassThrough edges from all JammedWaveTileNumbers to
            // their matching jammed ForLoop coordinate
            connectJammedOperation(graph, info.loadA, WaveTilesX, WaveTilesY);
            connectJammedOperation(graph, info.loadB, WaveTilesX, WaveTilesY);
            connectJammedOperation(graph, info.siblingLoads, WaveTilesX, WaveTilesY);
            if(info.storeD)
                connectJammedOperation(graph, *info.storeD, WaveTilesX, WaveTilesY);

            // Delete original loadA and loadB.
            purgeNodes(graph, {info.loadA.load(), info.loadB.load()});

            // If the original loads were through LDS, attach their
            // LoadTiled+StoreLDSTile operations to the ForLoop.
            // Barriers and/or prefetching will be added during the
            // AddPrefetch transform.
            connectGlobalLoadOperations(graph, forK, info.loadA);
            connectGlobalLoadOperations(graph, forK, info.loadB);

            // Memory ordering...
            {
                auto loadAChain = info.loadA.storeLDS != -1;
                auto loadBChain = info.loadB.storeLDS != -1;
                if(loadAChain)
                    graph.control.addElement(Sequence(), {info.loadA.storeLDS}, {firstSetCoordA});
                if(loadBChain)
                    graph.control.addElement(Sequence(), {info.loadB.storeLDS}, {firstSetCoordB});
                if(loadAChain && !loadBChain)
                    graph.control.addElement(Sequence(), {info.loadA.storeLDS}, {firstSetCoordB});
                if(!loadAChain && loadBChain)
                    graph.control.addElement(Sequence(), {firstSetCoordA}, {info.loadB.global});
                if(loadAChain && loadBChain)
                    graph.control.addElement(Sequence(), {info.loadA.global}, {info.loadB.global});
            }

            // Order StoreLDSTile operations
            auto toOrder = filter(graph.control.isElemType<StoreLDSTile>(),
                                  graph.control.depthFirstVisit(forK, Graph::Direction::Downstream))
                               .to<std::vector>();
            orderMemoryNodes(graph, toOrder, false);
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

        ConstraintStatus NoDanglingJammedNumbers(const KernelGraph& graph)
        {
            using GD = rocRoller::Graph::Direction;

            ConstraintStatus retval;
            for(auto tag : graph.coordinates.getNodes<JammedWaveTileNumber>())
            {
                auto noIncoming = empty(graph.coordinates.getNeighbours<GD::Upstream>(tag));
                auto noOutgoing = empty(graph.coordinates.getNeighbours<GD::Downstream>(tag));
                if(noIncoming || noOutgoing)
                {
                    retval.combine(false, concatenate("Dangling JammedWaveTileNumber: ", tag));
                }
            }
            return retval;
        }

        std::vector<GraphConstraint> LowerTensorContraction::postConstraints() const
        {
            return {NoDanglingJammedNumbers};
        }
    }
}
