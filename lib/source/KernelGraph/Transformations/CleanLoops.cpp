#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        KernelGraph cleanLoops(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::cleanLoops");
            auto k = original;
            for(auto const& loop : k.control.getNodes<ForLoopOp>().to<std::vector>())
            {
                auto [lhs, rhs] = getForLoopIncrement(k, loop);
                auto forLoopDim = getSize(
                    std::get<Dimension>(k.coordinates.getElement(k.mapper.get<Dimension>(loop))));

                //Ensure forLoopDim is translate time evaluatable.
                if(!(evaluationTimes(forLoopDim)[EvaluationTime::Translate]))
                    continue;

                //Only remove single iteration loops!
                if(evaluate(rhs) != evaluate(forLoopDim))
                    continue;
                //fetch parents of ForLoopOp and children
                auto parentBodies = k.control.getInputNodeIndices<Body>(loop).to<std::vector>();
                auto children     = k.control.getOutputNodeIndices<Body>(loop).to<std::vector>();
                auto parentSequences
                    = k.control.getInputNodeIndices<Sequence>(loop).to<std::vector>();
                auto childrenSequences
                    = k.control.getOutputNodeIndices<Sequence>(loop).to<std::vector>();

                // Add connections between setCoord and parents/children.
                // Remove old versions
                auto setCoord     = k.control.addElement(SetCoordinate(literal(0u)));
                auto loopIterator = k.mapper.getConnections(loop)[0].coordinate;
                auto loopDims     = k.coordinates.getOutputNodeIndices<DataFlowEdge>(loopIterator)
                                    .to<std::vector>();
                AssertFatal(loopDims.size() == 1);
                k.mapper.connect<ForLoop>(setCoord, loopDims[0]);

                for(auto const& parentBody : parentBodies)
                {
                    k.control.addElement(Body(), {parentBody}, {setCoord});
                    k.control.deleteElement<Body>(std::vector<int>{parentBody},
                                                  std::vector<int>{loop});
                }

                for(auto const& parentSeq : parentSequences)
                {
                    k.control.addElement(Sequence(), {parentSeq}, {setCoord});
                    k.control.deleteElement<Sequence>(std::vector<int>{parentSeq},
                                                      std::vector<int>{loop});
                }

                for(auto const& child : children)
                {
                    k.control.addElement(Body(), {setCoord}, {child});
                    k.control.deleteElement<Body>(std::vector<int>{loop}, std::vector<int>{child});
                }

                for(auto const& child : childrenSequences)
                {
                    k.control.addElement(Sequence(), {setCoord}, {child});
                    k.control.deleteElement<Sequence>(std::vector<int>{loop},
                                                      std::vector<int>{child});
                }

                //Delete Initialize, ForLoopIncrement, ForLoop and their immediate children
                auto forLoopChildren = k.control.depthFirstVisit(loop).to<std::vector>();
                for(auto const& toDelete : forLoopChildren)
                {
                    k.control.deleteElement(toDelete);
                }

                for(auto const& c : k.mapper.getConnections(loop))
                {
                    k.mapper.disconnect(loop, c.coordinate, c.connection);
                }
            }

            return k;
        }
    }
}
