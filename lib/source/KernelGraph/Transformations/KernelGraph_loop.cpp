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

            CoordinateTransform::Dimension loopIndexDim(CoordinateTransform::HyperGraph& coordGraph)
            {
                if(m_loopDimTag.ctag < 0)
                {
                    CoordinateTransform::Dimension dim
                        = CoordinateTransform::Linear{-1, m_loopSize, m_loopStride};

                    m_loopDimTag = coordGraph.addDimension(dim);

                    return dim;
                }

                return coordGraph.getDimension(m_loopDimTag);
            }

            CoordinateTransform::Dimension getLoop(int                              localTag,
                                                   CoordinateTransform::HyperGraph& coordGraph)
            {
                if(m_loopDims.count(localTag))
                {
                    return m_loopDims.at(localTag);
                }

                CoordinateTransform::Dimension loop
                    = CoordinateTransform::ForLoop(-1, m_loopSize, m_loopStride);

                CoordinateTransform::Dimension loopIndex = loopIndexDim(coordGraph);

                coordGraph.allocateTag(loop);
                m_loopDims.emplace(localTag, loop);

                coordGraph.addEdge({loopIndex}, {loop}, CoordinateTransform::DataFlow{});

                return loop;
            }

            void addLoopDst(CoordinateTransform::HyperGraph&             coordGraph,
                            ControlGraph::ControlGraph&                  controlGraph,
                            Location const&                              loc,
                            CoordinateTransform::CoordinateTransformEdge e)
            {
                auto newLoc = loc;

                auto dstTag = getTag(newLoc.dstDims[0]).ctag;

                CoordinateTransform::Dimension loop = getLoop(dstTag, coordGraph);

                newLoc.dstDims.insert(newLoc.dstDims.begin(), loop);
                coordGraph.addEdge(newLoc.srcDims, newLoc.dstDims, e);
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph& coordGraph,
                                   ControlGraph::ControlGraph&      controlGraph,
                                   Location const&                  loc,
                                   CoordinateTransform::Tile const& t) override
            {
                addLoopDst(coordGraph, controlGraph, loc, t);
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph&    coordGraph,
                                   ControlGraph::ControlGraph&         controlGraph,
                                   Location const&                     loc,
                                   CoordinateTransform::Inherit const& t) override
            {
                addLoopDst(coordGraph, controlGraph, loc, t);
            }

            void addLoopSrc(CoordinateTransform::HyperGraph&             coordGraph,
                            ControlGraph::ControlGraph&                  controlGraph,
                            Location const&                              loc,
                            CoordinateTransform::CoordinateTransformEdge e)
            {
                auto newLoc = loc;

                auto srcTag = getTag(newLoc.srcDims[0]).ctag;

                CoordinateTransform::Dimension loop = getLoop(srcTag, coordGraph);

                newLoc.srcDims.insert(newLoc.srcDims.begin(), loop);
                coordGraph.addEdge(newLoc.srcDims, newLoc.dstDims, e);
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph&    coordGraph,
                                   ControlGraph::ControlGraph&         controlGraph,
                                   Location const&                     loc,
                                   CoordinateTransform::Flatten const& f) override
            {
                if(loc.srcDims.size() > 1)
                    addLoopSrc(coordGraph, controlGraph, loc, f);
                else
                    BaseGraphVisitor::visitEdge(coordGraph, controlGraph, loc, f);
            }

            virtual void visitEdge(CoordinateTransform::HyperGraph&   coordGraph,
                                   ControlGraph::ControlGraph&        controlGraph,
                                   Location const&                    loc,
                                   CoordinateTransform::Forget const& f) override
            {
                addLoopSrc(coordGraph, controlGraph, loc, f);
            }

            virtual void visitOperation(CoordinateTransform::HyperGraph& coordGraph,
                                        ControlGraph::ControlGraph&      controlGraph,
                                        Location const&                  loc,
                                        ControlGraph::LoadVGPR const&    dst) override
            {
                controlGraph.addEdge({m_loopOp}, {dst}, ControlGraph::Body{});
            }

            virtual void visitRoot(CoordinateTransform::HyperGraph& coordGraph,
                                   ControlGraph::ControlGraph&      controlGraph,
                                   ControlGraph::Kernel const&      k) override
            {
                auto iterTag = getTag(loopIndexDim(coordGraph));

                auto loopVarExp = std::make_shared<Expression::Expression>(
                    DataFlowTag{iterTag.ctag, Register::Type::Scalar, DataType::Int32});

                auto condition = loopVarExp < m_loopSize;
                m_loopOp       = ControlGraph::ForLoopOp{-1, iterTag, condition};
                controlGraph.allocateTag(m_loopOp);

                controlGraph.addEdge({k}, {m_loopOp}, ControlGraph::Body());

                auto zero = Expression::literal(0);

                ControlGraph::Operation initOp
                    = ControlGraph::Assign({-1, iterTag.ctag, Register::Type::Scalar, zero});
                controlGraph.allocateTag(initOp);

                controlGraph.addEdge({m_loopOp}, {initOp}, ControlGraph::Initialize{});

                auto                    incExp = loopVarExp + m_loopStride;
                ControlGraph::Operation incOp
                    = ControlGraph::Assign({-1, iterTag.ctag, Register::Type::Scalar, incExp});
                controlGraph.allocateTag(incOp);
                controlGraph.addEdge({m_loopOp}, {incOp}, ControlGraph::ForLoopIncrement{});
            }

        private:
            TagType m_loopDimTag;

            ExpressionPtr           m_loopSize, m_loopStride;
            ControlGraph::Operation m_loopOp;

            std::map<int, CoordinateTransform::Dimension> m_loopDims;
        };

        KernelGraph
            lowerLinearLoop(KernelGraph k, ExpressionPtr loopSize, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerLinearLoop");
            auto visitor = LoopDistributeVisitor(context, loopSize);
            return rewrite(k, visitor);
        }
    }
}
