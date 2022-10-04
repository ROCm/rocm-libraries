

#include <rocRoller/Expression.hpp>

#include <rocRoller/Serialization/Expression.hpp>
#include <rocRoller/Serialization/YAML.hpp>

namespace rocRoller
{
    namespace Expression
    {
        std::string toYAML(ExpressionPtr const& expr)
        {
            return Serialization::toYAML(expr);
        }

        ExpressionPtr fromYAML(std::string const& str)
        {
            return Serialization::fromYAML<ExpressionPtr>(str);
        }
    }
}
