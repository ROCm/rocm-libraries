#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Create duplicates of all of the nodes downstream of the provided
         *        start nodes.
         *        Add the duplicates to the provided graph.
         *        Return the location of the new start nodes.
         *
         * @param graph
         * @param startNodes
         * @return std::vector<int>
         */
        std::vector<int> duplicateControlNodes(KernelHypergraph&    graph,
                                               std::set<int> const& startNodes)
        {
            std::vector<int>   newStartNodes;
            std::map<int, int> reindexer;

            // Create duplicates of all of the nodes downstream of the startNodes
            for(auto const& node :
                graph.control.depthFirstVisit(startNodes, Graph::Direction::Downstream))
            {
                // Only do this step if element is a node
                if(graph.control.getElementType(node) == Graph::ElementType::Node)
                {
                    auto op         = graph.control.addElement(graph.control.getElement(node));
                    reindexer[node] = op;
                }
            }

            for(auto const& reindex : reindexer)
            {
                // Create all edges within new sub-graph
                auto location = graph.control.getLocation(reindex.first);
                for(auto const& output : location.outgoing)
                {
                    int child = *graph.control.getNeighbours<Graph::Direction::Downstream>(output)
                                     .begin();
                    graph.control.addElement(
                        graph.control.getElement(output), {reindex.second}, {reindexer[child]});
                }

                // Use the same coordinate graph mappings
                for(auto const& c : graph.mapper.getConnections(reindex.first))
                {
                    graph.mapper.connect(reindex.second, c.coordinate, c.tindex, c.subDimension);
                }
            }

            // Return the new start nodes
            for(auto const& startNode : startNodes)
            {
                newStartNodes.push_back(reindexer[startNode]);
            }

            return newStartNodes;
        }

        /**
         * @brief Determine whether to unroll a specified loop
         *
         */
        bool performUnroll(KernelHypergraph& graph, int loopTag)
        {
            // TODO: Better loop dependency checker
            // Do not unroll loops that have a dependency between iterations.
            // At the moment, we are saying that if the first node in a loop's
            // body is a Multiply, then we know it has a dependency. This should
            // be improved in the future.
            auto directBodyNodes
                = graph.control.getOutputNodeIndices<ControlHypergraph::Body>(loopTag);

            for(auto const& directBodyNode : directBodyNodes)
            {
                auto x = std::get<ControlHypergraph::Operation>(
                    graph.control.getElement(directBodyNode));
                if(std::holds_alternative<ControlHypergraph::Multiply>(x))
                    return false;
            }

            return true;
        }

        struct UnrollLoopsVisitor : public BaseGraphVisitor
        {
            UnrollLoopsVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context, Graph::Direction::Upstream, false)
            {
            }

            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            virtual void visitOperation(KernelHypergraph&                   graph,
                                        KernelHypergraph const&             original,
                                        GraphReindexer&                     reindexer,
                                        int                                 tag,
                                        ControlHypergraph::ForLoopOp const& op) override
            {
                copyOperation(graph, original, reindexer, tag);
                auto newTag = reindexer.control.at(tag);
                auto bodies = graph.control.getOutputNodeIndices<ControlHypergraph::Body>(newTag)
                                  .to<std::set>();

                if(!performUnroll(graph, newTag))
                    return;

                // ---------------------------------
                // Add Unroll dimension to the coordinates graph

                // Use the same coordinate graph mappings
                auto loopIterator = original.mapper.getConnections(tag)[0].coordinate;

                // The loop iterator should have a dataflow link to the ForLoop dimension
                auto forLoopDimension
                    = graph.coordinates.getOutputNodeIndices<CoordGraph::DataFlowEdge>(loopIterator)
                          .to<std::vector>()[0];

                // Find all incoming PassThrough edges to the ForLoop dimension and replace them with
                // a Split edge with an Unroll dimension.
                auto forLoopLocation = graph.coordinates.getLocation(forLoopDimension);
                auto unrollDimension
                    = graph.coordinates.addElement(CoordGraph::Unroll(UNROLL_AMOUNT));
                for(auto const& input : forLoopLocation.incoming)
                {
                    if(CoordGraph::isEdge<CoordGraph::PassThrough>(
                           std::get<CoordGraph::Edge>(graph.coordinates.getElement(input))))
                    {
                        int parent
                            = *graph.coordinates.getNeighbours<Graph::Direction::Upstream>(input)
                                   .begin();
                        graph.coordinates.addElement(
                            CoordGraph::Split(), {parent}, {forLoopDimension, unrollDimension});
                        graph.coordinates.deleteElement(input);
                    }
                }

                // Find all outgoing PassThrough edges from the ForLoop dimension and replace them with
                // a Join edge with an Unroll dimension.
                for(auto const& output : forLoopLocation.outgoing)
                {
                    if(CoordGraph::isEdge<CoordGraph::PassThrough>(
                           std::get<CoordGraph::Edge>(graph.coordinates.getElement(output))))
                    {
                        int child
                            = *graph.coordinates.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        graph.coordinates.addElement(
                            CoordGraph::Join(), {forLoopDimension, unrollDimension}, {child});
                        graph.coordinates.deleteElement(output);
                    }
                }

                // ------------------------------
                // Change the loop increment calculation
                // Multiply the increment amount by the unroll amount

                // Find the ForLoopIcrement calculation
                // TODO: Handle multiple ForLoopIncrement edges that might be in a different
                // format, such as ones coming from ComputeIndex.
                auto loopIncrement
                    = graph.control
                          .getOutputNodeIndices<ControlHypergraph::ForLoopIncrement>(newTag)
                          .to<std::vector>();
                AssertFatal(loopIncrement.size() == 1, "Should only have 1 loop increment edge");

                auto loopIncrementOp
                    = graph.control.getNode<ControlHypergraph::Assign>(loopIncrement[0]);

                AssertFatal(std::holds_alternative<Expression::Add>(*loopIncrementOp.expression),
                            "Loop increment expression must be an addition");
                auto addExpr = std::get<Expression::Add>(*loopIncrementOp.expression);

                auto connections = graph.mapper.getConnections(loopIncrement[0]);
                AssertFatal(connections.size() == 1,
                            "Invalid Assign operation; coordinate missing.");
                auto dim_tag = connections[0].coordinate;
                AssertFatal(std::holds_alternative<Expression::DataFlowTag>(*addExpr.lhs),
                            "First argument in loop increment expression should be dataflow tag");
                AssertFatal(std::get<Expression::DataFlowTag>(*addExpr.lhs).tag == dim_tag,
                            "First argument in loop increment expression should be loop iterator "
                            "data flow tag");
                auto newAddExpr = addExpr.lhs + (addExpr.rhs * Expression::literal(UNROLL_AMOUNT));
                loopIncrementOp.expression = newAddExpr;

                graph.control.setElement(loopIncrement[0], loopIncrementOp);

                // ------------------------------
                // Add a setCoordinate node in between the original ForLoopOp and the loop bodies

                // Delete edges between original ForLoopOp and original loop body
                for(auto const& child :
                    graph.control.getNeighbours<Graph::Direction::Downstream>(newTag))
                {
                    if(ControlHypergraph::isEdge<ControlHypergraph::Body>(
                           graph.control.getElement(child)))
                    {
                        graph.control.deleteElement(child);
                    }
                }

                // Function for adding a SetCoordinate node inbetween the ForLoop
                // and a list of nodes.
                auto connectWithSetCoord = [&](auto const& toConnect, unsigned int coordValue) {
                    auto setCoord = graph.control.addElement(
                        ControlHypergraph::SetCoordinate(Expression::literal(coordValue)));
                    graph.mapper.connect<CoordGraph::Unroll>(setCoord, unrollDimension);
                    graph.control.addElement(ControlHypergraph::Body(), {newTag}, {setCoord});
                    for(auto const& body : toConnect)
                    {
                        graph.control.addElement(ControlHypergraph::Body(), {setCoord}, {body});
                    }
                };

                // Add setCoordinate nodes to original body
                connectWithSetCoord(bodies, 0u);

                // ------------------------------
                // Create duplicates of the loop body and add a setCoordinate node in between
                // the ForLoopOp and the new bodies
                for(unsigned int i = 1; i < UNROLL_AMOUNT; i++)
                {
                    auto newBodies = duplicateControlNodes(graph, bodies);

                    connectWithSetCoord(newBodies, i);
                }
            }

        private:
            const unsigned int UNROLL_AMOUNT = 2;
        };

        KernelHypergraph unrollLoops(KernelHypergraph const& k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::unrollLoops");
            auto visitor = UnrollLoopsVisitor(context);
            return rewrite(k, visitor);
        }
    }
}
