
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

        void addConnectionsMultiply(KernelGraph& graph,
                                    int          waveMult,
                                    int          loadTag,
                                    NaryArgument argType)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTensorContraction::addConnectionsMultiply(): Multiply({})",
                waveMult);

            auto load = graph.control.getElement(loadTag);
            AssertFatal(isOperation<LoadTiled>(load), "Operand should be LoadTiled");

            // LoadTiled A
            auto userTag = graph.mapper.get<User>(loadTag);
            AssertFatal(userTag > 0, "User dimension not found");
            graph.mapper.connect<User>(waveMult, userTag, 0);

            AssertFatal(userTag > 0, "User dimension not found");

            auto [waveTag, wave] = graph.getDimension<WaveTile>(loadTag);

            auto macroTile = graph.mapper.get<MacroTile>(loadTag);

            graph.mapper.connect(
                waveMult, macroTile, Connections::typeArgument<MacroTile>(argType));
            graph.mapper.connect(waveMult, waveTag, Connections::typeArgument<WaveTile>(argType));
        }

        struct MatrixMultiplyInfo
        {
            int                kernel; //< Kernel operation
            int                loadA; //< Load operation that loads the A (LHS) operand
            int                loadB; //< Load operation that loads the B (RHS) operand
            std::optional<int> loadAScale; //< Load operation that loads the A (LHS) scale
            std::optional<int> loadBScale; //< Load operation that loads the B (RHS) scale
            int                storeD; //< Store operation that stores the result (D); can be -1

            std::vector<int> dependentAssigns; //< Assign operations that use the result (D)
            std::vector<int> siblingLoads; //< Load operations that flow into dependentAssigns
            std::vector<int> siblingOps; //< Other operations that flow into dependentAssigns
        };

        MatrixMultiplyInfo getMatrixMultiplyInfo(KernelGraph const& graph, int tensorContractionTag)
        {
            MatrixMultiplyInfo info;

            auto parents = graph.control.parentNodes(tensorContractionTag).to<std::vector>();
            AssertFatal(parents.size() == 2 || parents.size() == 4, ShowValue(parents.size()));

            // Get tensor contraction operands

            std::optional<int> operandA, operandAScale;
            std::optional<int> operandB, operandBScale;

            std::map<int, int> parentTags;
            for(auto p : parents)
            {
                auto mapped        = graph.mapper.get<MacroTile>(p);
                parentTags[mapped] = p;
            }

            {
                auto [aTag, aTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::LHS);
                operandA = parentTags.at(aTag);
                parentTags.erase(aTag);
            }

            {
                auto [bTag, bTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::RHS);
                operandB = parentTags.at(bTag);
                parentTags.erase(bTag);
            }

            if(parents.size() == 4)
            {
                // TODO: We will need to implement versions of getDimension that return an
                // optional<>.  Once that is done, we can support one scaled input and one
                // non-scaled input.

                auto [aScaledTag, aScaledTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::LHS_SCALE);
                operandAScale = parentTags.at(aScaledTag);
                parentTags.erase(aScaledTag);

                auto [bScaledTag, bScaledTile]
                    = graph.getDimension<MacroTile>(tensorContractionTag, NaryArgument::RHS_SCALE);
                operandBScale = parentTags.at(bScaledTag);
                parentTags.erase(bScaledTag);
            }

            AssertFatal(parentTags.empty());

            info.loadA      = *operandA;
            info.loadB      = *operandB;
            info.loadAScale = operandAScale;
            info.loadBScale = operandBScale;

            // Find loads, stores, assigns etc
            auto reachableFromTC
                = graph.control.depthFirstVisit(tensorContractionTag).to<std::unordered_set>();

            std::vector<int> stores;
            for(auto const index : graph.control.getNodes())
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{[&](auto op) {},
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

            if(!info.loadAScale.has_value() && info.loadBScale.has_value())
            {
                AssertFatal(info.siblingLoads.size() <= 1, ShowValue(info.siblingLoads.size()));
            }

            return info;
        }

        ExpressionPtr getAccumulationLoopSize(KernelGraph const& graph, int a, int loadA)
        {
            auto tileA              = graph.coordinates.getNode<MacroTile>(a);
            auto [_sdimYTag, sdimY] = graph.getDimension<SubDimension>(loadA, 1);

            auto matK = sdimY.size;
            auto macK = literal(static_cast<uint>(tileA.sizes[1])); // M x K

            return Expression::convert<DataType::UInt32>(matK / macK);
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

            AssertFatal(info.loadAScale.has_value() == info.loadBScale.has_value(),
                        "A and B must both be scaled or neither.",
                        ShowValue(info.loadAScale.has_value()),
                        ShowValue(info.loadBScale.has_value()));
            bool scaled    = info.loadAScale.has_value();
            auto scaleMode = scaled ? Operations::ScaleMode::Separate : Operations::ScaleMode::None;

            std::optional<int> scaleSize, scaledK;

            auto accumulationCoordSize = getAccumulationLoopSize(graph, a, info.loadA);

            auto [K, forK] = rangeFor(graph, accumulationCoordSize, rocRoller::KLOOP);

            if(scaled)
            {
                scaledK = K;
            }

            // A row block is x-workgroup, column block is for loop index
            // B row block is for loop index, column block is y-workgroup
            auto macTileNumYA = graph.mapper.get<MacroTileNumber>(info.loadA, 1);
            auto macTileNumXB = graph.mapper.get<MacroTileNumber>(info.loadB, 0);

            graph.coordinates.addElement(PassThrough(), {macTileNumYA}, {K});
            graph.coordinates.addElement(PassThrough(), {macTileNumXB}, {K});

            if(scaled)
            {
                auto macTileNumYAScale = graph.mapper.get<MacroTileNumber>(*info.loadAScale, 1);
                auto macTileNumXBScale = graph.mapper.get<MacroTileNumber>(*info.loadBScale, 0);

                graph.coordinates.addElement(PassThrough(), {macTileNumYAScale}, {*scaledK});
                graph.coordinates.addElement(PassThrough(), {macTileNumXBScale}, {*scaledK});
            }

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

            std::optional<int>      waveAScaleTag, waveBScaleTag;
            std::optional<WaveTile> waveAScale, waveBScale;
            std::optional<int>      waveTileNumYAScale;
            std::optional<int>      waveTileNumXBScale;

            if(scaled)
            {
                std::tie(waveAScaleTag, waveAScale)
                    = graph.getDimension<WaveTile>(*info.loadAScale);
                std::tie(waveBScaleTag, waveBScale)
                    = graph.getDimension<WaveTile>(*info.loadBScale);
                waveTileNumYAScale = graph.mapper.get<WaveTileNumber>(*info.loadAScale, 1);
                waveTileNumXBScale = graph.mapper.get<WaveTileNumber>(*info.loadBScale, 0);
            }

            // Add an unroll dimension that connects to both A's WaveTileNumber[1] and B's
            // WaveTileNumber[0]. This is because we are unrolling the "small k" loop.
            auto tileA = graph.coordinates.getNode<MacroTile>(a);

            uint const numWaveTiles = tileA.sizes[1] / waveA.sizes[1];
            auto       smallKUnroll = graph.coordinates.addElement(Unroll(numWaveTiles));
            graph.coordinates.addElement(PassThrough(), {waveTileNumYA}, {smallKUnroll});
            graph.coordinates.addElement(PassThrough(), {waveTileNumXB}, {smallKUnroll});

            if(scaled)
            {
                graph.coordinates.addElement(PassThrough(), {*waveTileNumYAScale}, {smallKUnroll});
                graph.coordinates.addElement(PassThrough(), {*waveTileNumXBScale}, {smallKUnroll});
            }

            int                lastWaveMult  = -1;
            int                lastSetCoordA = -1;
            int                lastSetCoordB = -1;
            std::optional<int> lastSetCoordAScale;
            std::optional<int> lastSetCoordBScale;

            std::vector<int> nodesToOrder;

            for(uint k = 0; k < numWaveTiles; k++)
            {
                auto createUnrollLoad = [&, forK = forK](int load) -> std::tuple<int, int> {
                    auto setCoord = graph.control.addElement(SetCoordinate(literal(k)));
                    graph.mapper.connect<Unroll>(setCoord, smallKUnroll);
                    graph.control.addElement(Body(), {forK}, {setCoord});

                    auto newLoad = duplicateControlNode(graph, load);
                    if(k != 0)
                        duplicateMacroTile(graph, newLoad);
                    graph.control.addElement(Body(), {setCoord}, {newLoad});

                    return {setCoord, newLoad};
                };

                auto [setCoordA, newLoadA] = createUnrollLoad(info.loadA);
                auto [setCoordB, newLoadB] = createUnrollLoad(info.loadB);

                std::optional<int> setCoordAScale, newLoadAScale;
                std::optional<int> setCoordBScale, newLoadBScale;

                if(scaled)
                {
                    std::tie(setCoordAScale, newLoadAScale) = createUnrollLoad(*info.loadAScale);
                    std::tie(setCoordBScale, newLoadBScale) = createUnrollLoad(*info.loadBScale);
                }

                auto waveMult = graph.control.addElement(Multiply(scaleMode, scaleMode));
                graph.mapper.connect(
                    waveMult, d, Connections::typeArgument<MacroTile>(NaryArgument::DEST));

                addConnectionsMultiply(graph, waveMult, newLoadA, NaryArgument::LHS);
                addConnectionsMultiply(graph, waveMult, newLoadB, NaryArgument::RHS);

                if(scaled)
                {
                    addConnectionsMultiply(
                        graph, waveMult, *newLoadAScale, NaryArgument::LHS_SCALE);
                    addConnectionsMultiply(
                        graph, waveMult, *newLoadBScale, NaryArgument::RHS_SCALE);
                }

                nodesToOrder.insert(nodesToOrder.end(), {setCoordA, setCoordB});
                if(scaled)
                    nodesToOrder.insert(nodesToOrder.end(), {*setCoordAScale, *setCoordBScale});
                nodesToOrder.push_back(waveMult);
            }

            auto prev = nodesToOrder.begin();
            for(auto cur = prev + 1; prev != nodesToOrder.end() && cur != nodesToOrder.end();
                prev++, cur++)
            {
                graph.control.addElement(Sequence(), {*prev}, {*cur});
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
            if(scaled)
                purgeNodes(graph, {*info.loadAScale, *info.loadBScale});
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
