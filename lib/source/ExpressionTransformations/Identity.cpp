#include <rocRoller/Expression_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr identity(ExpressionPtr expr)
        {
            return expr;
        }
    }
}
