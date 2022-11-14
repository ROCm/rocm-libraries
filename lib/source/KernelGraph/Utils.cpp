#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /***********************************
         * Helpers
         */

        /**
         * Create a range-based for loop.
         */
        std::pair<CoordinateTransform::ForLoop, ControlGraph::ForLoopOp>
            rangeFor(CoordinateTransform::HyperGraph& coordGraph,
                     ControlGraph::ControlGraph&      controlGraph,
                     Expression::ExpressionPtr        size)
        {
            auto unit_stride  = Expression::literal(1u);
            auto ctag         = coordGraph.allocateTag();
            auto rangeK       = CoordinateTransform::Linear(ctag, size, unit_stride);
            auto dimK         = CoordinateTransform::ForLoop(ctag);
            auto sizeDataType = Expression::resultVariableType(size);
            auto exprK        = std::make_shared<Expression::Expression>(
                DataFlowTag{ctag, Register::Type::Scalar, sizeDataType});

            ControlGraph::ForLoopOp forK
                = ControlGraph::ForLoopOp{ctag, getTag(rangeK), exprK < size};
            ControlGraph::Operation initK = ControlGraph::Assign{
                -1, ctag, Register::Type::Scalar, Expression::literal(0, sizeDataType)};
            ControlGraph::Operation incrementK
                = ControlGraph::Assign{-1, ctag, Register::Type::Scalar, exprK + unit_stride};

            controlGraph.allocateTag(initK);
            controlGraph.allocateTag(incrementK);

            coordGraph.addEdge({rangeK}, {dimK}, CoordinateTransform::DataFlow());
            controlGraph.addEdge({forK}, {initK}, ControlGraph::Initialize());
            controlGraph.addEdge({forK}, {incrementK}, ControlGraph::ForLoopIncrement());

            return {dimK, forK};
        }

    }
}
