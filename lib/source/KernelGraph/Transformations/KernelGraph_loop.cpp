#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        struct LoopDistributeVisitor : public BaseGraphVisitor
        {
            LoopDistributeVisitor(std::shared_ptr<Context> context, ExpressionPtr loopSize)
                : BaseGraphVisitor(context)
                , m_loopSize(loopSize)
                , m_loopStride(Expression::literal(1))
            {
            }

            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            int loopIndexDim(KernelHypergraph& graph)
            {
                if(m_loopDimTag < 0)
                {
                    m_loopDimTag = graph.coordinates.addElement(
                        CoordGraph::Linear(m_loopSize, m_loopStride));
                }

                return m_loopDimTag;
            }

            int getLoop(int localTag, KernelHypergraph& graph)
            {
                if(m_loopDims.count(localTag))
                    return m_loopDims.at(localTag);

                auto loopIndex = loopIndexDim(graph);
                auto loop
                    = graph.coordinates.addElement(CoordGraph::ForLoop(m_loopSize, m_loopStride));
                graph.coordinates.addElement(CoordGraph::DataFlow(), {loopIndex}, {loop});

                m_loopDims.emplace(localTag, loop);

                return loop;
            }

            void addLoopDst(KernelHypergraph&                          graph,
                            KernelHypergraph const&                    original,
                            GraphReindexer&                            reindexer,
                            int                                        tag,
                            CoordGraph::CoordinateTransformEdge const& edge)
            {
                auto new_tag = reindexer.coordinates.at(tag);

                auto outgoing_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(new_tag)
                          .to<std::vector>();
                auto dstTag = -1;
                for(auto const& out : outgoing_nodes)
                {
                    auto odim = std::get<CoordGraph::Dimension>(graph.coordinates.getElement(out));
                    if(CoordGraph::isDimension<CoordGraph::Workgroup>(odim))
                    {
                        dstTag = out;
                        break;
                    }
                }

                AssertFatal(dstTag > 0, "addLoopDst : Workgroup dimension not found");

                auto loop = getLoop(dstTag, graph);
                outgoing_nodes.insert(outgoing_nodes.begin(), loop);

                auto incoming_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(new_tag)
                          .to<std::vector>();

                graph.coordinates.deleteElement(new_tag);
                graph.coordinates.addElement(new_tag, edge, incoming_nodes, outgoing_nodes);
            }

            void visitEdge(KernelHypergraph&       graph,
                           KernelHypergraph const& original,
                           GraphReindexer&         reindexer,
                           int                     tag,
                           CoordGraph::Tile const& edge) override
            {
                copyEdge(graph, original, reindexer, tag);
                addLoopDst(graph, original, reindexer, tag, edge);
            }

            void visitEdge(KernelHypergraph&          graph,
                           KernelHypergraph const&    original,
                           GraphReindexer&            reindexer,
                           int                        tag,
                           CoordGraph::Inherit const& edge) override
            {
                copyEdge(graph, original, reindexer, tag);
                addLoopDst(graph, original, reindexer, tag, edge);
            }

            void addLoopSrc(KernelHypergraph&                          graph,
                            KernelHypergraph const&                    original,
                            GraphReindexer&                            reindexer,
                            int                                        tag,
                            CoordGraph::CoordinateTransformEdge const& edge)
            {
                auto new_tag = reindexer.coordinates.at(tag);
                auto incoming_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(new_tag)
                          .to<std::vector>();
                auto srcTag = -1;
                for(auto const& in : incoming_nodes)
                {
                    auto idim = std::get<CoordGraph::Dimension>(graph.coordinates.getElement(in));
                    if(CoordGraph::isDimension<CoordGraph::Workgroup>(idim))
                    {
                        srcTag = in;
                        break;
                    }
                }

                AssertFatal(srcTag > 0, "addLoopSrc : Workgroup dimension not found");

                auto loop = getLoop(srcTag, graph);
                incoming_nodes.insert(incoming_nodes.begin(), loop);

                auto outgoing_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(new_tag)
                          .to<std::vector>();
                graph.coordinates.deleteElement(new_tag);
                graph.coordinates.addElement(new_tag, edge, incoming_nodes, outgoing_nodes);
            }

            void visitEdge(KernelHypergraph&          graph,
                           KernelHypergraph const&    original,
                           GraphReindexer&            reindexer,
                           int                        tag,
                           CoordGraph::Flatten const& edge) override
            {
                copyEdge(graph, original, reindexer, tag);

                auto incoming_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(tag)
                          .to<std::vector>();
                if(incoming_nodes.size() > 1)
                    addLoopSrc(graph, original, reindexer, tag, edge);
            }

            void visitEdge(KernelHypergraph&         graph,
                           KernelHypergraph const&   original,
                           GraphReindexer&           reindexer,
                           int                       tag,
                           CoordGraph::Forget const& edge) override
            {
                copyEdge(graph, original, reindexer, tag);
                addLoopSrc(graph, original, reindexer, tag, edge);
            }

            virtual void visitOperation(KernelHypergraph&                   graph,
                                        KernelHypergraph const&             original,
                                        GraphReindexer&                     reindexer,
                                        int                                 tag,
                                        ControlHypergraph::ElementOp const& op) override
            {
                copyOperation(graph, original, reindexer, tag);

                auto new_tag = reindexer.control.at(tag);
                auto new_op  = graph.control.getNode<ControlHypergraph::ElementOp>(new_tag);
                new_op.a     = op.a > 0 ? reindexer.coordinates.at(op.a) : op.a;
                new_op.b     = op.b > 0 ? reindexer.coordinates.at(op.b) : op.b;
                graph.control.setElement(new_tag, new_op);
            }

            void visitOperation(KernelHypergraph&                  graph,
                                KernelHypergraph const&            original,
                                GraphReindexer&                    reindexer,
                                int                                tag,
                                ControlHypergraph::LoadVGPR const& op) override
            {
                copyOperation(graph, original, reindexer, tag);

                auto new_tag = reindexer.control.at(tag);
                auto incoming_edge_tag
                    = graph.control.getNeighbours<Graph::Direction::Upstream>(new_tag)
                          .to<std::vector>();
                AssertFatal(incoming_edge_tag.size() == 1, "one parent node is expected");
                auto incoming_edge = graph.control.getElement(incoming_edge_tag[0]);

                graph.control.deleteElement(incoming_edge_tag[0]);
                graph.control.addElement(incoming_edge_tag[0],
                                         incoming_edge,
                                         std::vector<int>{m_loopOp},
                                         std::vector<int>{new_tag});
            }

            void visitRoot(KernelHypergraph&       graph,
                           KernelHypergraph const& original,
                           GraphReindexer&         reindexer,
                           int                     tag) override
            {
                BaseGraphVisitor::visitRoot(graph, original, reindexer, tag);

                auto iterTag = loopIndexDim(graph);
                auto new_tag = reindexer.control.at(tag);

                // create loopOp and attach with Kernel
                auto loopVarExp = std::make_shared<Expression::Expression>(
                    DataFlowTag{iterTag, Register::Type::Scalar, DataType::Int32});
                m_loopOp = graph.control.addElement(
                    ControlHypergraph::ForLoopOp{loopVarExp < m_loopSize});
                graph.control.addElement(ControlHypergraph::Body(), {new_tag}, {m_loopOp});

                // create initOp and attach with the loopOp
                auto zero   = Expression::literal(0);
                auto initOp = graph.control.addElement(
                    ControlHypergraph::Assign{Register::Type::Scalar, zero});
                graph.control.addElement(ControlHypergraph::Initialize(), {m_loopOp}, {initOp});

                // create incOp and attach with the loopOp
                auto incOp = graph.control.addElement(
                    ControlHypergraph::Assign{Register::Type::Scalar, loopVarExp + m_loopStride});
                graph.control.addElement(
                    ControlHypergraph::ForLoopIncrement(), {m_loopOp}, {incOp});

                graph.mapper.connect<CoordGraph::Dimension>(m_loopOp, iterTag);
                graph.mapper.connect<CoordGraph::Dimension>(initOp, iterTag);
                graph.mapper.connect<CoordGraph::ForLoop>(incOp, iterTag);
            }

        private:
            int m_loopDimTag = -1;

            ExpressionPtr m_loopSize, m_loopStride;
            int           m_loopOp;

            std::unordered_map<int, int> m_loopDims;
        };

        KernelHypergraph lowerLinearLoop(KernelHypergraph const&  k,
                                         ExpressionPtr            loopSize,
                                         std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerLinearLoop");
            auto visitor = LoopDistributeVisitor(context, loopSize);
            return rewrite(k, visitor);
        }
    }
}
