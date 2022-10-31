
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
            if(!x)
            {
                return x;
            }

            x = simplify(x);
            x = fastDivision(x, m_context);
            x = fastMultiplication(x);
            x = fuseAssociative(x);
            x = fuseTernary(x);
            // x = launchTimeSubExpressions(x, m_context); // TODO: Add launchTimeSubExpressions

            return x;
        }
    }
}
