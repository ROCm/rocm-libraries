#include "KernelGraph/KernelHypergraph.hpp"
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

// TODO update this
#if 0
        struct KernelUnrollVisitor : public BaseGraphVisitor
        {
            KernelUnrollVisitor(std::shared_ptr<Context> context, ExpressionPtr unrollSize)
                : BaseGraphVisitor(context)
                , m_unrollSize(unrollSize)
            {
            }

            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            CoordinateTransform::Dimension unrollDim(CoordinateTransform::HyperGraph& coordGraph,
                                                     ControlGraph::ControlGraph&      controlGraph)
            {
                if(m_unrollDimTag.ctag < 0)
                {
                    //This will ensure that the tag for the unrollOp and the unrollDim are the same.
                    int newTag = std::max(coordGraph.nextTag(), controlGraph.nextTag());
                    CoordinateTransform::Dimension dim
                        = CoordinateTransform::Linear{newTag, m_unrollSize, Expression::literal(1)};
                    m_unrollDimTag = coordGraph.addDimension(dim);
                    controlGraph.recognizeTag(m_unrollDimTag);
                    return dim;
                }
                return coordGraph.getDimension(m_unrollDimTag);
            }

            void addUnrollDst(CoordinateTransform::HyperGraph&                   coordGraph,
                              ControlGraph::ControlGraph&                        controlGraph,
                              std::vector<CoordinateTransform::Dimension> const& src,
                              std::vector<CoordinateTransform::Dimension>        dst,
                              CoordinateTransform::CoordinateTransformEdge       e)
            {
                auto                           dstTag = getTag(dst[0]).ctag;
                CoordinateTransform::Dimension unroll
                    = CoordinateTransform::Unroll(dstTag, m_unrollSize);

                dst.insert(dst.begin(), unroll);
                coordGraph.addEdge(src, dst, e);
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph& coordGraph,
                                   ControlGraph::ControlGraph&      controlGraph,
                                   Location const&                  loc,
                                   CoordinateTransform::Tile const& t) override
            {
                addUnrollDst(coordGraph, controlGraph, loc.srcDims, loc.dstDims, t);
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph&    coordGraph,
                                   ControlGraph::ControlGraph&         controlGraph,
                                   Location const&                     loc,
                                   CoordinateTransform::Inherit const& t) override
            {
                addUnrollDst(coordGraph, controlGraph, loc.srcDims, loc.dstDims, t);
            }

            void addUnrollSrc(CoordinateTransform::HyperGraph&                   coordGraph,
                              ControlGraph::ControlGraph&                        controlGraph,
                              std::vector<CoordinateTransform::Dimension>        src,
                              std::vector<CoordinateTransform::Dimension> const& dst,
                              CoordinateTransform::CoordinateTransformEdge       e)
            {
                auto                           srcTag = getTag(src[0]).ctag;
                CoordinateTransform::Dimension unroll
                    = CoordinateTransform::Unroll(srcTag, m_unrollSize);

                src.insert(src.begin(), unroll);
                coordGraph.addEdge(src, dst, e);
                coordGraph.addEdge({unrollDim(coordGraph, controlGraph)},
                                   {unroll},
                                   CoordinateTransform::DataFlow{});
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph&    coordGraph,
                                   ControlGraph::ControlGraph&         controlGraph,
                                   Location const&                     loc,
                                   CoordinateTransform::Flatten const& f) override
            {
                if(loc.srcDims.size() > 1)
                {
                    addUnrollSrc(coordGraph, controlGraph, loc.srcDims, loc.dstDims, f);
                }
                else
                {
                    BaseGraphVisitor::visitEdge(coordGraph, controlGraph, loc, f);
                }
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph&   coordGraph,
                                   ControlGraph::ControlGraph&        controlGraph,
                                   Location const&                    loc,
                                   CoordinateTransform::Forget const& f) override
            {
                addUnrollSrc(coordGraph, controlGraph, loc.srcDims, loc.dstDims, f);
            }

            virtual void visitOperation(CoordinateTransform::HyperGraph& coordGraph,
                                        ControlGraph::ControlGraph&      controlGraph,
                                        Location const&                  loc,
                                        ControlGraph::LoadVGPR const&    dst) override
            {
                auto m_unrollOp = ControlGraph::UnrollOp{
                    getTag(unrollDim(coordGraph, controlGraph)).ctag, m_unrollSize};
                controlGraph.addEdge({loc.srcs}, {m_unrollOp}, loc.controlEdge);
                controlGraph.addEdge({m_unrollOp}, {dst}, ControlGraph::Body{});
            }

        private:
            TagType       m_unrollDimTag;
            ExpressionPtr m_unrollSize;
        };
#endif

        KernelHypergraph lowerLinearUnroll(KernelHypergraph const&  k,
                                           ExpressionPtr            unrollSize,
                                           std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerLinearUnroll");
            // auto visitor = KernelUnrollVisitor(context, unrollSize);
            // return rewrite(k, visitor);
            return k;
        }
    }
}
