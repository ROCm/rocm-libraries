
#include <rocRoller/CommandSolution.hpp>
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

        struct MatrixMultiplyInfo
        {
            int kernel; //< Kernel operation
            int loadA; //< Load operation that loads the A (LHS) operand
            int loadB; //< Load operation that loads the B (RHS) operand
            int storeD; //< Store operation that stores the result (D); can be -1

            std::vector<int> dependentAssigns; //< Assign operations that use the result (D)
            std::vector<int> siblingLoads; //< Load operations that flow into dependentAssigns
            std::vector<int> siblingOps; //< Other operations that flow into dependentAssigns
        };

        MatrixMultiplyInfo getMatrixMultiplyInfo(KernelGraph const& graph, int tensorContractionTag)
        {
            MatrixMultiplyInfo info;

            // Get tensor contraction operands
            auto [aTag, aTile]
                = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::LHS);
            auto [bTag, bTile]
                = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::RHS);

            auto parents = graph.control.parentNodes(tensorContractionTag).to<std::vector>();
            AssertFatal(parents.size() == 2);
            auto operandA = parents[0];
            auto operandB = parents[1];
            AssertFatal(aTag == graph.mapper.get<MacroTile>(operandA));
            AssertFatal(bTag == graph.mapper.get<MacroTile>(operandB));

            // Find loads, stores, assigns etc
            auto reachableFromTC
                = graph.control.depthFirstVisit(tensorContractionTag).to<std::unordered_set>();

            std::vector<int> loadA;
            std::vector<int> loadB;
            std::vector<int> stores;
            for(auto const index : graph.control.getNodes())
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) {},
                          [&](LoadTiled const& load) {
                              auto reachableFromLoad
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              if(reachableFromLoad.contains(operandA))
                                  loadA.push_back(index);
                              else if(reachableFromLoad.contains(operandB))
                                  loadB.push_back(index);
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

            AssertFatal(loadA.size() == 1 && loadB.size() == 1);
            info.loadA = loadA[0];
            info.loadB = loadB[0];

            AssertFatal(stores.size() <= 1);
            info.storeD = !stores.empty() ? stores[0] : -1;

            auto root = only(graph.control.roots());
            AssertFatal(root, "More than one Kernel node not supported");
            info.kernel = *root;

            // Find sibling loads and ops
            auto filterOutABLoads = [&](int x) { return x != info.loadA && x != info.loadB; };
            auto kernelOutputs
                = filter(filterOutABLoads, graph.control.childNodes(info.kernel)).to<std::vector>();
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
                          }},
                      std::get<Operation>(elem));
            }
            AssertFatal(info.siblingLoads.size() <= 1);

            return info;
        }

        ExpressionPtr getAccumulationLoopSize(KernelGraph const& graph, int a, int loadA)
        {
            auto tileA              = graph.coordinates.getNode<MacroTile>(a);
            auto [_sdimYTag, sdimY] = graph.getDimension<SubDimension>(loadA, 1);

            auto matK = sdimY.size;
            auto macK = literal(static_cast<uint>(tileA.sizes[1])); // M x K

            auto toUInt32 = [](ExpressionPtr expr) -> ExpressionPtr {
                return std::make_shared<Expression::Expression>(
                    Expression::Convert<DataType::UInt32>{expr});
            };

            return toUInt32(matK / macK);
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
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerMatrixMultiply({})", tag);

            auto info = getMatrixMultiplyInfo(graph, tag);

            auto accumulationCoordSize = getAccumulationLoopSize(graph, a, info.loadA);

            auto [K, forK] = rangeFor(graph, accumulationCoordSize, rocRoller::KLOOP);

            // A row block is x-workgroup, column block is for loop index
            // B row block is for loop index, column block is y-workgroup
            auto macTileNumYA = graph.mapper.get<MacroTileNumber>(info.loadA, 1);
            auto macTileNumXB = graph.mapper.get<MacroTileNumber>(info.loadB, 0);

            graph.coordinates.addElement(PassThrough(), {macTileNumYA}, {K});
            graph.coordinates.addElement(PassThrough(), {macTileNumXB}, {K});

            auto [waveATag, waveA] = graph.getDimension<WaveTile>(info.loadA);
            auto [waveBTag, waveB] = graph.getDimension<WaveTile>(info.loadB);
            uint num_elements      = waveA.sizes[0] * waveB.sizes[1];
            uint wfs               = context->kernel()->wavefront_size();
            uint numAGPRs          = num_elements / wfs; // number of output registers per thread

            auto initD = graph.control.addElement(
                Assign{Register::Type::Accumulator, literal(0.f), numAGPRs});

            graph.mapper.connect(initD, d, NaryArgument::DEST);

            auto waveTileNumYA = graph.mapper.get<WaveTileNumber>(info.loadA, 1);
            auto waveTileNumXB = graph.mapper.get<WaveTileNumber>(info.loadB, 0);

            // Add an unroll dimension that connects to both A's WaveTileNumber[1] and B's
            // WaveTileNumber[0]. This is because we are unrolling the "small k" loop.
            auto tileA = graph.coordinates.getNode<MacroTile>(a);

            uint const numWaveTiles = tileA.sizes[1] / waveA.sizes[1];
            auto       smallKUnroll = graph.coordinates.addElement(Unroll(numWaveTiles));
            graph.coordinates.addElement(PassThrough(), {waveTileNumYA}, {smallKUnroll});
            graph.coordinates.addElement(PassThrough(), {waveTileNumXB}, {smallKUnroll});

            int lastWaveMult  = -1;
            int lastSetCoordA = -1;
            int lastSetCoordB = -1;
            for(uint k = 0; k < numWaveTiles; k++)
            {
                auto setCoordA = graph.control.addElement(SetCoordinate(literal(k)));
                graph.mapper.connect<Unroll>(setCoordA, smallKUnroll);
                graph.control.addElement(Body(), {forK}, {setCoordA});

                auto newLoadA = duplicateControlNode(graph, info.loadA);
                if(k != 0)
                    duplicateMacroTile(graph, newLoadA);
                graph.control.addElement(Body(), {setCoordA}, {newLoadA});

                auto setCoordB = graph.control.addElement(SetCoordinate(literal(k)));
                graph.mapper.connect<Unroll>(setCoordB, smallKUnroll);
                graph.control.addElement(Body(), {forK}, {setCoordB});

                auto newLoadB = duplicateControlNode(graph, info.loadB);
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
            std::vector<int> jammedOperations = {info.loadA, info.loadB};
            std::copy(info.siblingLoads.cbegin(),
                      info.siblingLoads.cend(),
                      std::back_inserter(jammedOperations));

            for(auto tag : jammedOperations)
            {
                auto jammedX = graph.mapper.get<JammedWaveTileNumber>(tag, 0);
                if(jammedX != -1)
                    graph.coordinates.addElement(PassThrough(), {jammedX}, {WaveTilesX});
                auto jammedY = graph.mapper.get<JammedWaveTileNumber>(tag, 1);
                if(jammedY != -1)
                    graph.coordinates.addElement(PassThrough(), {jammedY}, {WaveTilesY});
            }

            if(info.storeD > 0)
            {
                std::vector<int> waveTiles = {WaveTilesX, WaveTilesY};

                for(auto c : graph.mapper.getConnections(info.storeD))
                {
                    std::visit(rocRoller::overloaded{
                                   [&](Connections::TypeAndSubDimension const& x) {
                                       if(x.id == JammedWaveTileNumber().name())
                                           graph.coordinates.addElement(PassThrough(),
                                                                        {waveTiles[x.subdimension]},
                                                                        {c.coordinate});
                                   },
                                   [&](Connections::LDSTypeAndSubDimension const& x) {
                                       if(x.id == JammedWaveTileNumber().name())
                                           graph.coordinates.addElement(PassThrough(),
                                                                        {waveTiles[x.subdimension]},
                                                                        {c.coordinate});
                                   },
                                   [](auto const& x) {}},
                               c.connection);
                }
            }

            // Delete original loadA and loadB.
            purgeNodes(graph, {info.loadA, info.loadB});
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
