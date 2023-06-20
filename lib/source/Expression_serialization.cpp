

#include <rocRoller/Expression.hpp>

#include <rocRoller/Serialization/Expression.hpp>
#include <rocRoller/Serialization/YAML.hpp>

#ifdef ROCROLLER_USE_LLVM

static_assert(rocRoller::Serialization::
                  CMappedType<rocRoller::Expression::Expression, llvm::yaml::IO, std::string>);
static_assert(
    rocRoller::Serialization::CMappedType<rocRoller::Expression::Expression, llvm::yaml::IO>);
static_assert(rocRoller::Serialization::
                  CMappedType<rocRoller::Expression::ExpressionPtr, llvm::yaml::IO, std::string>);

static_assert(
    rocRoller::Serialization::CMappedType<rocRoller::Expression::ExpressionPtr, llvm::yaml::IO>);
static_assert(
    rocRoller::Serialization::has_SerializationTraits<rocRoller::Expression::ExpressionPtr,
                                                      llvm::yaml::IO>::value);
static_assert(!llvm::yaml::has_FlowTraits<rocRoller::Expression::ExpressionPtr>::value);
#endif

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
