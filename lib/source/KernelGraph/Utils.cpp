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
        std::pair<int, int> rangeFor(KernelHypergraph& graph, Expression::ExpressionPtr size)
        {
            auto unit_stride = Expression::literal(1u);
            auto rangeK      = graph.coordinates.addElement(CoordGraph::Linear(size, unit_stride));
            auto dimK        = graph.coordinates.addElement(CoordGraph::ForLoop(size, unit_stride));
            auto sizeDataType = Expression::resultVariableType(size);
            auto exprK        = std::make_shared<Expression::Expression>(
                DataFlowTag{rangeK, Register::Type::Scalar, sizeDataType});

            auto forK       = graph.control.addElement(ControlHypergraph::ForLoopOp{exprK < size});
            auto initK      = graph.control.addElement(ControlHypergraph::Assign{
                Register::Type::Scalar, Expression::literal(0, sizeDataType)});
            auto incrementK = graph.control.addElement(
                ControlHypergraph::Assign{Register::Type::Scalar, exprK + unit_stride});

            graph.coordinates.addElement(CoordGraph::DataFlow(), {rangeK}, {dimK});
            graph.control.addElement(ControlHypergraph::Initialize(), {forK}, {initK});
            graph.control.addElement(ControlHypergraph::ForLoopIncrement(), {forK}, {incrementK});

            graph.mapper.connect<CoordGraph::Dimension>(forK, rangeK);
            graph.mapper.connect<CoordGraph::Dimension>(initK, rangeK);
            graph.mapper.connect<CoordGraph::ForLoop>(incrementK, rangeK);

            return {dimK, forK};
        }
    }
}
