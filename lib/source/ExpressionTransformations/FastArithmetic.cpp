
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

namespace rocRoller
{
    namespace Expression
    {
        FastArithmetic::FastArithmetic(std::shared_ptr<Context> context)
            : m_context(context)
        {
        }

        ExpressionPtr FastArithmetic::operator()(ExpressionPtr x)
        {
            // TODO: add launchTimeSubExpressions and fix all the tests that are broken by it.
            // x = launchTimeSubExpressions(x, m_context);

            x = fastDivision(x, m_context);

            // TODO: add fastMultiplication when all signed/unsigned arith working
            // x = fastMultiplication(x, m_context);

            return x;

            // // TODO: add fastMultiplication when all signed/unsigned arith working
            // // auto x1 = fastMultiplication(x, m_context);
            // // auto x2 = fastDivision(x1, m_context);
            // // return x2;
            // auto x2 = fastDivision(x, m_context);
            // return x2;
        }
    }
}
