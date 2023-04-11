
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

        ExpressionPtr FastArithmetic::operator()(ExpressionPtr x) const
        {
            if(!x)
            {
                return x;
            }
            ExpressionPtr orig = x;

            x = simplify(x);
            x = fastDivision(x, m_context);
            x = fastMultiplication(x);
            x = fuseAssociative(x);
            x = fuseTernary(x);
            // x = launchTimeSubExpressions(x, m_context); // TODO: Add launchTimeSubExpressions

            if(m_context->kernelOptions().logLevel >= LogLevel::Debug)
            {
                auto comment = Instruction::Comment(
                    concatenate("FastArithmetic:", ShowValue(orig), ShowValue(x)));
                m_context->schedule(comment);
            }

            return x;
        }
    }
}
