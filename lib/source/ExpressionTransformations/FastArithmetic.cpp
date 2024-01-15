
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

namespace rocRoller
{
    namespace Expression
    {
        FastArithmetic::FastArithmetic(ContextPtr context)
            : m_context(context)
        {
        }

        ExpressionPtr FastArithmetic::operator()(ExpressionPtr x) const
        {
            if(!x)
            {
                return x;
            }
            ExpressionPtr orig = x;

            x = fastDivision(x, m_context);
            x = simplify(x);
            x = fastMultiplication(x);
            x = fuseAssociative(x);
            x = fuseTernary(x);
            x = launchTimeSubExpressions(x, m_context);

            if(!identical(orig, x))
            {
                auto comment = Instruction::Comment(
                    concatenate("FastArithmetic:", ShowValue(orig), ShowValue(x)));
                m_context->schedule(comment);
            }

            return x;
        }
    }
}
