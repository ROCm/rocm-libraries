#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include "Graph/Hypergraph.hpp"
#include "KernelGraph.hpp"
#include "KernelGraph/ControlHypergraph/ControlHypergraph.hpp"
#include "KernelGraph/CoordGraph/CoordinateHypergraph.hpp"
#include "KernelGraph/CoordGraph/Edge_fwd.hpp"

#include <rocRoller/AssemblyKernel.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /***************************
         * KernelGraphs rewrite utils
         */

        // Delete this when graph rearch complete
        /* Location: rewrite visitors receive this when visiting an edge. */
        struct Location
        {
            Location(CoordinateTransform::HyperGraph& coordGraph,
                     ControlGraph::ControlGraph&      controlGraph)
                : coordGraph(coordGraph)
                , controlGraph(controlGraph)
            {
            }

            std::vector<TagType>                        srcTags, dstTags;
            std::vector<CoordinateTransform::Dimension> srcDims, dstDims;
            std::vector<ControlGraph::Operation>        srcs;

            ControlGraph::ControlEdge controlEdge;

            CoordinateTransform::HyperGraph& coordGraph;
            ControlGraph::ControlGraph&      controlGraph;

            void update(CoordinateTransform::HyperGraph const&  graph,
                        CoordinateTransform::EdgeKeyType const& e)
            {
                srcTags = e.stags;
                dstTags = e.dtags;
                srcDims = graph.getDimensions(e.stags);
                dstDims = graph.getDimensions(e.dtags);
            }

            void update(ControlGraph::ControlGraph const& graph, ControlGraph::EdgeKey const& e)
            {
                srcs        = graph.getInputs(e.second);
                controlEdge = graph.getEdge(e);
            }
        };

#define MAKE_EDGE_VISITOR(CLS)                                            \
    virtual void visitEdge(CoordinateTransform::HyperGraph& coordGraph,   \
                           ControlGraph::ControlGraph&      controlGraph, \
                           Location const&                  loc,          \
                           CoordinateTransform::CLS const&  edge)         \
    {                                                                     \
        coordGraph.addEdge(loc.srcDims, loc.dstDims, edge);               \
    }

#define MAKE_OPERATION_VISITOR(CLS)                                            \
    virtual void visitOperation(CoordinateTransform::HyperGraph& coordGraph,   \
                                ControlGraph::ControlGraph&      controlGraph, \
                                Location const&                  loc,          \
                                ControlGraph::CLS const&         dst)          \
    {                                                                          \
        controlGraph.addEdge(loc.srcs, {dst}, loc.controlEdge);                \
    }

        /**
         * The BaseGraphVisitor implements a "copy" re-write visitor.
         *
         * To rewrite, for example, only LoadLinear edges, simply
         * override the `visitEdge` method.
         */
        // Delete this when graph rearch complete
        struct BaseGraphVisitor
        {
            BaseGraphVisitor(std::shared_ptr<Context> context)
                : m_context(context)
            {
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

            MAKE_EDGE_VISITOR(ConstructMacroTile);
            MAKE_EDGE_VISITOR(DataFlow);
            MAKE_EDGE_VISITOR(DestructMacroTile);
            MAKE_EDGE_VISITOR(Flatten);
            MAKE_EDGE_VISITOR(Forget);
            MAKE_EDGE_VISITOR(Inherit);
            MAKE_EDGE_VISITOR(Join);
            MAKE_EDGE_VISITOR(MakeOutput);
            MAKE_EDGE_VISITOR(PassThrough);
            MAKE_EDGE_VISITOR(Split);
            MAKE_EDGE_VISITOR(Tile);

            MAKE_OPERATION_VISITOR(ElementOp);
            MAKE_OPERATION_VISITOR(Kernel);
            MAKE_OPERATION_VISITOR(LoadLDSTile);
            MAKE_OPERATION_VISITOR(LoadLinear);
            MAKE_OPERATION_VISITOR(LoadTiled);
            MAKE_OPERATION_VISITOR(LoadVGPR);
            MAKE_OPERATION_VISITOR(Multiply);
            MAKE_OPERATION_VISITOR(TensorContraction);
            MAKE_OPERATION_VISITOR(StoreLDSTile);
            MAKE_OPERATION_VISITOR(StoreLinear);
            MAKE_OPERATION_VISITOR(StoreTiled);
            MAKE_OPERATION_VISITOR(StoreVGPR);
            MAKE_OPERATION_VISITOR(ForLoopOp);
            MAKE_OPERATION_VISITOR(Assign);
            MAKE_OPERATION_VISITOR(UnrollOp);
            MAKE_OPERATION_VISITOR(Barrier);

            virtual void visitEdge(CoordinateTransform::HyperGraph&                    coordGraph,
                                   ControlGraph::ControlGraph&                         controlGraph,
                                   Location const&                                     loc,
                                   CoordinateTransform::CoordinateTransformEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(coordGraph, controlGraph, loc, arg); },
                           edge);
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph&         coordGraph,
                                   ControlGraph::ControlGraph&              controlGraph,
                                   Location const&                          loc,
                                   CoordinateTransform::DataFlowEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(coordGraph, controlGraph, loc, arg); },
                           edge);
            }

            virtual void visitRoot(CoordinateTransform::HyperGraph& coordGraph,
                                   ControlGraph::ControlGraph&      controlGraph,
                                   ControlGraph::Kernel const&      edge)
            {
            }

        protected:
            std::shared_ptr<Context> m_context;
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
        // Delete this when graph rearch complete
        template <typename T>
        inline KernelGraph rewrite(KernelGraph k, T& visitor)
        {
            auto location = Location(k.coordinates, k.control);

            KernelGraph graph{k.coordinates.nextTag(), k.control.nextTag()};
            for(auto const& ekey :
                k.coordinates.topographicalSort(CoordinateTransform::EdgeType::Any))
            {
                location.update(k.coordinates, ekey);
                std::visit(
                    [&](auto&& arg) {
                        visitor.visitEdge(graph.coordinates, graph.control, location, arg);
                    },
                    k.coordinates.getEdge(ekey));
            }

            auto rootOp = k.control.getRootOperation();
            visitor.visitRoot(
                graph.coordinates, graph.control, std::get<ControlGraph::Kernel>(rootOp));

            for(auto const& edge : k.control.getOperationEdges())
            {
                location.update(k.control, edge.first);
                std::visit(
                    [&](auto&& arg) {
                        visitor.visitOperation(graph.coordinates, graph.control, location, arg);
                    },
                    k.control.getOperation(edge.first.second));
            }

            return graph;
        }

        // Delete this when graph rearch complete
        template <typename T>
        inline KernelGraph rewriteDimensions(KernelGraph k, T& visitor)
        {
            KernelGraph graph = k;
            for(auto const& x : k.coordinates.getDimensions())
            {
                auto y = std::visit([&](auto&& arg) { return visitor.visitDimension(arg); }, x);
                graph.coordinates.resetDimension(y);
            }
            for(auto const& x : k.control.getOperations())
            {
                auto y = std::visit([&](auto&& arg) { return visitor.visitOperation(arg); }, x);
                graph.control.resetOperation(y);
            }

            return graph;
        }

        template <typename T>
        inline KernelHypergraph rewriteDimensions(KernelHypergraph const& k, T& visitor)
        {
            KernelHypergraph graph = k;
            for(auto const& tag : k.coordinates.getNodes())
            {
                auto x = std::get<CoordGraph::Dimension>(k.coordinates.getElement(tag));
                auto y
                    = std::visit([&](auto&& arg) { return visitor.visitDimension(tag, arg); }, x);
                graph.coordinates.setElement(tag, y);
            }
            for(auto const& tag : k.control.getNodes())
            {
                auto x = std::get<ControlHypergraph::Operation>(k.control.getElement(tag));
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

#define MAKE_EDGE_VISITOR(CLS)                                \
    virtual void visitEdge(KernelHypergraph&       graph,     \
                           KernelHypergraph const& original,  \
                           GraphReindexer&         reindexer, \
                           int                     tag,       \
                           CoordGraph::CLS const&  edge)      \
    {                                                         \
        copyEdge(graph, original, reindexer, tag);            \
    }

#define MAKE_OPERATION_VISITOR(CLS)                                      \
    virtual void visitOperation(KernelHypergraph&             graph,     \
                                KernelHypergraph const&       original,  \
                                GraphReindexer&               reindexer, \
                                int                           tag,       \
                                ControlHypergraph::CLS const& dst)       \
    {                                                                    \
        copyOperation(graph, original, reindexer, tag);                  \
    }

        /**
         * The BaseGraphVisitor implements a "copy" re-write visitor.
         *
         * To rewrite, for example, only LoadLinear edges, simply
         * override the `visitEdge` method.
         */
        // Rename this when graph rearch complete
        struct BaseGraphVisitor2
        {
            BaseGraphVisitor2(std::shared_ptr<Context> context)
                : m_context(context)
            {
            }

            void copyEdge(KernelHypergraph&       graph,
                          KernelHypergraph const& original,
                          GraphReindexer&         reindexer,
                          int                     edge)
            {
                auto location = original.coordinates.getLocation(edge);

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
                    rocRoller::Log::getLogger()->debug(
                        "KernelGraph::copyEdge(): input {} -> {}", input, inputs.back());
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
                    rocRoller::Log::getLogger()->debug(
                        "KernelGraph::copyEdge(): output {} -> {}", output, outputs.back());
                }

                auto newEdge = graph.coordinates.addElement(
                    original.coordinates.getElement(edge), inputs, outputs);

                reindexer.coordinates.emplace(edge, newEdge);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::copyEdge(): Edge {} -> Edge {}", edge, newEdge);
            }

            void copyOperation(KernelHypergraph&       graph,
                               KernelHypergraph const& original,
                               GraphReindexer&         reindexer,
                               int                     tag)
            {
                auto location = original.control.getLocation(tag);

                auto op = graph.control.addElement(original.control.getElement(tag));
                for(auto const& input : location.incoming)
                {
                    int parent = *original.control.getNeighbours<Graph::Direction::Upstream>(input)
                                      .begin();
                    AssertFatal(reindexer.control.count(parent) > 0,
                                "Missing control input: ",
                                ShowValue(tag),
                                ShowValue(input),
                                ShowValue(parent));
                    graph.control.addElement(
                        original.control.getElement(input), {reindexer.control.at(parent)}, {op});
                }

                for(auto const& c : original.mapper.getConnections(tag))
                {
                    AssertFatal(reindexer.coordinates.count(c.coordinate) > 0,
                                "Missing mapped coordinate: ",
                                ShowValue(tag),
                                ShowValue(c.coordinate));
                    graph.mapper.connect(
                        op, reindexer.coordinates.at(c.coordinate), c.tindex, c.subDimension);
                }

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

            MAKE_EDGE_VISITOR(ConstructMacroTile);
            MAKE_EDGE_VISITOR(DataFlow);
            MAKE_EDGE_VISITOR(DestructMacroTile);
            MAKE_EDGE_VISITOR(Flatten);
            MAKE_EDGE_VISITOR(Forget);
            MAKE_EDGE_VISITOR(Inherit);
            MAKE_EDGE_VISITOR(Join);
            MAKE_EDGE_VISITOR(MakeOutput);
            MAKE_EDGE_VISITOR(PassThrough);
            MAKE_EDGE_VISITOR(Split);
            MAKE_EDGE_VISITOR(Tile);

            MAKE_OPERATION_VISITOR(ElementOp);
            MAKE_OPERATION_VISITOR(Kernel);
            MAKE_OPERATION_VISITOR(LoadLDSTile);
            MAKE_OPERATION_VISITOR(LoadLinear);
            MAKE_OPERATION_VISITOR(LoadTiled);
            MAKE_OPERATION_VISITOR(LoadVGPR);
            MAKE_OPERATION_VISITOR(Multiply);
            MAKE_OPERATION_VISITOR(TensorContraction);
            MAKE_OPERATION_VISITOR(Scope);
            MAKE_OPERATION_VISITOR(StoreLDSTile);
            MAKE_OPERATION_VISITOR(StoreLinear);
            MAKE_OPERATION_VISITOR(StoreTiled);
            MAKE_OPERATION_VISITOR(StoreVGPR);
            MAKE_OPERATION_VISITOR(ForLoopOp);
            MAKE_OPERATION_VISITOR(Assign);
            MAKE_OPERATION_VISITOR(UnrollOp);
            MAKE_OPERATION_VISITOR(Barrier);

            virtual void visitEdge(KernelHypergraph&                          graph,
                                   KernelHypergraph const&                    original,
                                   GraphReindexer&                            reindexer,
                                   int                                        tag,
                                   CoordGraph::CoordinateTransformEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(graph, original, reindexer, tag, arg); },
                           edge);
            }

            virtual void visitEdge(KernelHypergraph&               graph,
                                   KernelHypergraph const&         original,
                                   GraphReindexer&                 reindexer,
                                   int                             tag,
                                   CoordGraph::DataFlowEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(graph, original, reindexer, tag, arg); },
                           edge);
            }

            virtual void visitRoot(KernelHypergraph&       graph,
                                   KernelHypergraph const& original,
                                   GraphReindexer&         reindexer,
                                   int                     kernel)
            {
            }

        protected:
            std::shared_ptr<Context> m_context;
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
        inline KernelHypergraph rewrite(KernelHypergraph const& original, T& visitor)
        {
            KernelHypergraph graph;
            GraphReindexer   reindexer;

            // add coordinate roots
            for(auto const& index : original.coordinates.roots())
            {
                reindexer.coordinates.emplace(
                    index, graph.coordinates.addElement(original.coordinates.getElement(index)));
            }

            for(auto const& index : original.coordinates.topologicalSort())
            {
                auto element = original.coordinates.getElement(index);
                if(std::holds_alternative<CoordGraph::Edge>(element))
                {
                    auto edge = std::get<CoordGraph::Edge>(element);
                    std::visit(
                        [&](auto&& arg) {
                            visitor.visitEdge(graph, original, reindexer, index, arg);
                        },
                        edge);
                }
            }

            // add control flow roots
            int kernel = *original.control.roots().begin();
            reindexer.control.emplace(
                kernel, graph.control.addElement(original.control.getElement(kernel)));
            visitor.visitRoot(graph, original, reindexer, kernel);

            for(auto const& index : original.control.topologicalSort())
            {
                auto element = original.control.getElement(index);
                if(std::holds_alternative<ControlHypergraph::Operation>(element))
                {
                    auto node = std::get<ControlHypergraph::Operation>(element);
                    if(std::holds_alternative<ControlHypergraph::Kernel>(node))
                        continue;
                    std::visit(
                        [&](auto&& arg) {
                            visitor.visitOperation(graph, original, reindexer, index, arg);
                        },
                        node);
                }
            }

            return graph;
        }
    }
}
