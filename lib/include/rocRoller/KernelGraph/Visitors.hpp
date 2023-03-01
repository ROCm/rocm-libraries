#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include "ControlGraph/ControlGraph.hpp"
#include "CoordinateGraph/CoordinateEdge_fwd.hpp"
#include "CoordinateGraph/CoordinateGraph.hpp"
#include "Graph/Hypergraph.hpp"
#include "KernelGraph.hpp"

#include "Expression_fwd.hpp"

#include <rocRoller/AssemblyKernel.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        using CTEdge = rocRoller::KernelGraph::CoordinateGraph::Edge;

        /***************************
         * KernelGraphs rewrite utils
         */

        template <typename T>
        inline KernelGraph rewriteDimensions(KernelGraph const& k, T& visitor)
        {
            KernelGraph graph = k;
            for(auto const& tag : k.coordinates.getNodes())
            {
                auto x = std::get<Dimension>(k.coordinates.getElement(tag));
                auto y
                    = std::visit([&](auto&& arg) { return visitor.visitDimension(tag, arg); }, x);
                graph.coordinates.setElement(tag, y);
            }
            for(auto const& tag : k.control.getNodes())
            {
                auto x = std::get<Operation>(k.control.getElement(tag));
                auto y = std::visit([&](auto&& arg) { return visitor.visitOperation(arg); }, x);
                graph.control.setElement(tag, y);
            }

            return graph;
        }

        class GraphReindexer
        {
        public:
            std::map<int, int> coordinates;
            std::map<int, int> control;
        };

        /**
         * @brief Reindex coordinate tags in an Expression according to a reindexer.
         *
         * Expressions can contain references to tags within a coordinate graph.  For example, a
         * DataFlowTag expression represents the value of a coordinate within a coordinate graph,
         * and hence stores an explicit tag.
         *
         * After re-writing a coordinate graph, tag references need to be updated in all
         * expressions.
         */
        struct ReindexExpressionVisitor
        {
            ReindexExpressionVisitor(GraphReindexer const& reindexer)
                : reindexer(reindexer)
            {
            }

            GraphReindexer const& reindexer;

            template <Expression::CUnary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CBinary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CTernary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            Expression::ExpressionPtr operator()(std::shared_ptr<CommandArgument> const& expr) const
            {
                return std::make_shared<Expression::Expression>(expr);
            }

            Expression::ExpressionPtr operator()(Expression::DataFlowTag const& expr) const
            {
                if(reindexer.coordinates.count(expr.tag) > 0)
                {
                    return std::make_shared<Expression::Expression>(Expression::DataFlowTag{
                        reindexer.coordinates.at(expr.tag), expr.regType, expr.varType});
                }
                return std::make_shared<Expression::Expression>(expr);
            }

            template <Expression::CValue Value>
            Expression::ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression::Expression>(expr);
            }

            Expression::ExpressionPtr call(Expression::ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

#define MAKE_EDGE_VISITOR(CLS)                           \
    virtual void visitEdge(KernelGraph&       graph,     \
                           KernelGraph const& original,  \
                           GraphReindexer&    reindexer, \
                           int                tag,       \
                           CLS const&         edge)      \
    {                                                    \
        copyEdge(graph, original, reindexer, tag);       \
    }

#define MAKE_OPERATION_VISITOR(CLS)                           \
    virtual void visitOperation(KernelGraph&       graph,     \
                                KernelGraph const& original,  \
                                GraphReindexer&    reindexer, \
                                int                tag,       \
                                CLS const&         dst)       \
    {                                                         \
        copyOperation(graph, original, reindexer, tag);       \
    }

        /**
         * The BaseGraphVisitor implements a "copy" re-write visitor.
         *
         * To rewrite, for example, only LoadLinear edges, simply
         * override the `visitEdge` method.
         */
        struct BaseGraphVisitor
        {
            BaseGraphVisitor(std::shared_ptr<Context> context,
                             Graph::Direction controlGraphOrder  = Graph::Direction::Downstream,
                             bool             rewriteCoordinates = true)
                : m_context(context)
                , m_controlGraphDirection(controlGraphOrder)
                , m_rewriteCoordinates(rewriteCoordinates)
            {
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        Assign const&      assign)
            {
                copyOperation(graph, original, reindexer, tag);
                ReindexExpressionVisitor visitor(reindexer);
                auto                     new_assign
                    = Assign{assign.regType, visitor.call(assign.expression), assign.valueCount};
                graph.control.setElement(reindexer.control.at(tag), new_assign);
            }

            void copyEdge(KernelGraph&       graph,
                          KernelGraph const& original,
                          GraphReindexer&    reindexer,
                          int                edge)
            {
                auto             location = original.coordinates.getLocation(edge);
                std::vector<int> inputs;
                for(auto const& input : location.incoming)
                {
                    if(reindexer.coordinates.count(input) == 0)
                    {
                        auto newInput
                            = graph.coordinates.addElement(original.coordinates.getElement(input));
                        reindexer.coordinates.emplace(input, newInput);
                    }
                    inputs.push_back(reindexer.coordinates.at(input));
                }

                std::vector<int> outputs;
                for(auto const& output : location.outgoing)
                {
                    if(reindexer.coordinates.count(output) == 0)
                    {
                        auto newOutput
                            = graph.coordinates.addElement(original.coordinates.getElement(output));
                        reindexer.coordinates.emplace(output, newOutput);
                    }
                    outputs.push_back(reindexer.coordinates.at(output));
                }

                auto newEdge = graph.coordinates.addElement(
                    original.coordinates.getElement(edge), inputs, outputs);

                reindexer.coordinates.emplace(edge, newEdge);
            }

            /**
             * @brief Reconnect inputs from original operation to new operation.
             */
            void reconnectOperation(KernelGraph&       graph,
                                    KernelGraph const& original,
                                    GraphReindexer&    reindexer,
                                    int                new_tag,
                                    int                old_tag)
            {
                auto location = original.control.getLocation(old_tag);

                if(m_controlGraphDirection == Graph::Direction::Downstream)
                {
                    for(auto const& input : location.incoming)
                    {
                        int parent
                            = *original.control.getNeighbours<Graph::Direction::Upstream>(input)
                                   .begin();
                        AssertFatal(reindexer.control.count(parent) > 0,
                                    "Missing control input: ",
                                    ShowValue(old_tag),
                                    ShowValue(input),
                                    ShowValue(parent));
                        graph.control.addElement(original.control.getElement(input),
                                                 {reindexer.control.at(parent)},
                                                 {new_tag});
                    }
                }
                else
                {
                    for(auto const& output : location.outgoing)
                    {
                        int child
                            = *original.control.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        AssertFatal(reindexer.control.count(child) > 0,
                                    "Missing control output: ",
                                    ShowValue(old_tag),
                                    ShowValue(output),
                                    ShowValue(child));
                        graph.control.addElement(original.control.getElement(output),
                                                 {new_tag},
                                                 {reindexer.control.at(child)});
                    }
                }
            }

            /**
             * @brief Reconnect mappings from original operation to new operation.
             */
            void reconnectMappings(KernelGraph&       graph,
                                   KernelGraph const& original,
                                   GraphReindexer&    reindexer,
                                   int                new_tag,
                                   int                old_tag)
            {
                if(m_rewriteCoordinates)
                {
                    for(auto const& c : original.mapper.getConnections(old_tag))
                    {
                        AssertFatal(reindexer.coordinates.count(c.coordinate) > 0,
                                    "Missing mapped coordinate: ",
                                    ShowValue(old_tag),
                                    ShowValue(c.coordinate));
                        graph.mapper.connect(
                            new_tag, reindexer.coordinates.at(c.coordinate), c.connection);
                    }
                }
                else
                {
                    for(auto const& c : original.mapper.getConnections(old_tag))
                    {
                        graph.mapper.connect(new_tag, c.coordinate, c.connection);
                    }
                }
            }

            /**
             * @brief Copy operation from original graph to new graph.
             *
             * Inputs and mappings are preserved.
             */
            void copyOperation(KernelGraph&       graph,
                               KernelGraph const& original,
                               GraphReindexer&    reindexer,
                               int                tag)
            {
                auto op = graph.control.addElement(original.control.getElement(tag));
                reconnectOperation(graph, original, reindexer, op, tag);
                reconnectMappings(graph, original, reindexer, op, tag);

                reindexer.control.emplace(tag, op);
            }

            Expression::ExpressionPtr wavefrontSize() const
            {
                uint wfs = static_cast<uint>(m_context->kernel()->wavefront_size());
                return Expression::literal(wfs);
            }

            std::array<Expression::ExpressionPtr, 3> workgroupSize() const
            {
                auto const& wgSize = m_context->kernel()->workgroupSize();
                return {
                    Expression::literal(wgSize[0]),
                    Expression::literal(wgSize[1]),
                    Expression::literal(wgSize[2]),
                };
            }

            Expression::ExpressionPtr workgroupCountX() const
            {
                return m_context->kernel()->workgroupCount(0);
            }

            Expression::ExpressionPtr workgroupCountY() const
            {
                return m_context->kernel()->workgroupCount(1);
            }

            Expression::ExpressionPtr workgroupCountZ() const
            {
                return m_context->kernel()->workgroupCount(2);
            }

            Graph::Direction controlGraphDirection() const
            {
                return m_controlGraphDirection;
            }

            bool rewriteCoordinates() const
            {
                return m_rewriteCoordinates;
            }

            MAKE_EDGE_VISITOR(ConstructMacroTile);
            MAKE_EDGE_VISITOR(DataFlow);
            MAKE_EDGE_VISITOR(Offset);
            MAKE_EDGE_VISITOR(Stride);
            MAKE_EDGE_VISITOR(DestructMacroTile);
            MAKE_EDGE_VISITOR(Flatten);
            MAKE_EDGE_VISITOR(Forget);
            MAKE_EDGE_VISITOR(Inherit);
            MAKE_EDGE_VISITOR(Join);
            MAKE_EDGE_VISITOR(MakeOutput);
            MAKE_EDGE_VISITOR(PassThrough);
            MAKE_EDGE_VISITOR(Split);
            MAKE_EDGE_VISITOR(Tile);

            MAKE_OPERATION_VISITOR(Barrier);
            MAKE_OPERATION_VISITOR(ComputeIndex);
            MAKE_OPERATION_VISITOR(Deallocate);
            MAKE_OPERATION_VISITOR(ForLoopOp);
            MAKE_OPERATION_VISITOR(Kernel);
            MAKE_OPERATION_VISITOR(LoadLDSTile);
            MAKE_OPERATION_VISITOR(LoadLinear);
            MAKE_OPERATION_VISITOR(LoadTiled);
            MAKE_OPERATION_VISITOR(LoadVGPR);
            MAKE_OPERATION_VISITOR(Multiply);
            MAKE_OPERATION_VISITOR(Scope);
            MAKE_OPERATION_VISITOR(SetCoordinate);
            MAKE_OPERATION_VISITOR(StoreLDSTile);
            MAKE_OPERATION_VISITOR(StoreLinear);
            MAKE_OPERATION_VISITOR(StoreTiled);
            MAKE_OPERATION_VISITOR(StoreVGPR);
            MAKE_OPERATION_VISITOR(TensorContraction);
            MAKE_OPERATION_VISITOR(UnrollOp);

            virtual void visitEdge(KernelGraph&                   graph,
                                   KernelGraph const&             original,
                                   GraphReindexer&                reindexer,
                                   int                            tag,
                                   CoordinateTransformEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(graph, original, reindexer, tag, arg); },
                           edge);
            }

            virtual void visitEdge(KernelGraph&        graph,
                                   KernelGraph const&  original,
                                   GraphReindexer&     reindexer,
                                   int                 tag,
                                   DataFlowEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(graph, original, reindexer, tag, arg); },
                           edge);
            }

            virtual void visitRoot(KernelGraph&       graph,
                                   KernelGraph const& original,
                                   GraphReindexer&    reindexer,
                                   int                tag)
            {
                copyOperation(graph, original, reindexer, tag);
            }

        protected:
            std::shared_ptr<Context> m_context;
            Graph::Direction         m_controlGraphDirection;
            bool                     m_rewriteCoordinates;
        };
#undef MAKE_OPERATION_VISITOR
#undef MAKE_EDGE_VISITOR

        /**
         * Apply rewrite visitor to kernel graph.
         *
         * Edges are visited in topological order.
         *
         * Internal convenience routine.
         */
        template <typename T>
        inline KernelGraph rewrite(KernelGraph const& original, T& visitor)
        {
            KernelGraph    graph;
            GraphReindexer reindexer;

            if(visitor.rewriteCoordinates())
            {
                // add coordinate roots
                for(auto const& index : original.coordinates.roots())
                {
                    reindexer.coordinates.emplace(
                        index,
                        graph.coordinates.addElement(original.coordinates.getElement(index)));
                }

                for(auto const& index : original.coordinates.topologicalSort())
                {
                    auto element = original.coordinates.getElement(index);
                    if(std::holds_alternative<CTEdge>(element))
                    {
                        auto edge = std::get<CTEdge>(element);
                        std::visit(
                            [&](auto&& arg) {
                                visitor.visitEdge(graph, original, reindexer, index, arg);
                            },
                            edge);
                    }
                }
            }
            else
            {
                graph.coordinates = original.coordinates;
            }

            // add control flow roots
            int kernel = *original.control.roots().begin();

            if(visitor.controlGraphDirection() == Graph::Direction::Downstream)
            {
                visitor.visitRoot(graph, original, reindexer, kernel);
            }

            for(auto const& index : visitor.controlGraphDirection() == Graph::Direction::Downstream
                                        ? original.control.topologicalSort()
                                        : original.control.reverseTopologicalSort())
            {
                auto element = original.control.getElement(index);
                if(std::holds_alternative<Operation>(element))
                {
                    auto node = std::get<Operation>(element);
                    if(std::holds_alternative<Kernel>(node))
                        continue;
                    std::visit(
                        [&](auto&& arg) {
                            visitor.visitOperation(graph, original, reindexer, index, arg);
                        },
                        node);
                }
            }

            if(visitor.controlGraphDirection() == Graph::Direction::Upstream)
            {
                visitor.visitRoot(graph, original, reindexer, kernel);
            }

            return graph;
        }
    }
}
