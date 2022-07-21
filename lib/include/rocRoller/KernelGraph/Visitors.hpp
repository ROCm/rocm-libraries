#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include "KernelGraph.hpp"

#include <rocRoller/AssemblyKernel.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /***************************
         * KernelGraphs rewrite utils
         */

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

            std::array<Expression::ExpressionPtr, 3> workgroupCount() const
            {
                return m_context->kernel()->workgroupCount();
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
    }
}
