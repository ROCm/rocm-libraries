#pragma once

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>
#include <rocRoller/Serialization/Variant.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO, typename Context>
        struct MappingTraits<Expression::ExpressionPtr, IO, Context>
            : public SharedPointerMappingTraits<Expression::ExpressionPtr, IO, Context, true>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<Expression::Expression, IO, Context>
            : public DefaultVariantMappingTraits<Expression::Expression, IO, Context>
        {
        };

        template <Expression::CBinary TExp, typename IO, typename Context>
        struct MappingTraits<TExp, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TExp& exp, Context& ctx)
            {
                iot::mapRequired(io, "lhs", exp.lhs, ctx);
                iot::mapRequired(io, "rhs", exp.rhs, ctx);
            }

            static void mapping(IO& io, TExp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <Expression::CUnary TExp, typename IO, typename Context>
        struct MappingTraits<TExp, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TExp& exp, Context& ctx)
            {
                iot::mapRequired(io, "arg", exp.arg, ctx);
            }

            static void mapping(IO& io, TExp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Expression::ScaledMatrixMultiply, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Expression::ScaledMatrixMultiply& exp, Context& ctx)
            {
                iot::mapRequired(io, "matA", exp.matA, ctx);
                iot::mapRequired(io, "matB", exp.matB, ctx);
                iot::mapRequired(io, "matC", exp.matC, ctx);
                iot::mapRequired(io, "scaleA", exp.scaleA, ctx);
                iot::mapRequired(io, "scaleB", exp.scaleB, ctx);
            }

            static void mapping(IO& io, Expression::ScaledMatrixMultiply& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <Expression::CTernary TExp, typename IO, typename Context>
        struct MappingTraits<TExp, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TExp& exp, Context& ctx)
            {
                iot::mapRequired(io, "lhs", exp.lhs, ctx);
                iot::mapRequired(io, "r1hs", exp.r1hs, ctx);
                iot::mapRequired(io, "r2hs", exp.r2hs, ctx);
            }

            static void mapping(IO& io, TExp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        static_assert(CNamedVariant<CommandArgumentValue>);
        template <typename IO, typename Context>
        struct MappingTraits<CommandArgumentValue, IO, Context>
            : public DefaultVariantMappingTraits<CommandArgumentValue, IO, Context>
        {
            static const bool flow = true;

            using Base = DefaultVariantMappingTraits<CommandArgumentValue, IO>;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, CommandArgumentValue& val, Context& ctx)
            {
                std::string typeName;

                if(iot::outputting(io))
                {
                    typeName = name(val);
                }

                iot::mapRequired(io, "dataType", typeName);

                if(!iot::outputting(io))
                {
                    val = Base::alternatives.at(typeName)();
                }

                std::visit(
                    [&io, &ctx](auto& theVal) {
                        using U = std::decay_t<decltype(theVal)>;

                        if constexpr(std::is_pointer_v<U>)
                        {
                            Throw<FatalError>("Can't (de)serialize pointer values.");
                        }
                        else
                        {
                            iot::mapRequired(io, "value", theVal);
                        }
                    },
                    val);
            }

            static void mapping(IO& io, CommandArgumentValue& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<CommandArgumentPtr, IO, Context>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, CommandArgumentPtr& val, Context& ctx)
            {
                int           size;
                int           offset;
                std::string   name;
                VariableType  variableType;
                DataDirection direction;

                if(iot::outputting(io))
                {
                    size         = val->size();
                    offset       = val->offset();
                    name         = val->name();
                    variableType = val->variableType();
                    direction    = val->direction();
                }

                iot::mapRequired(io, "size", size);
                iot::mapRequired(io, "offset", offset);
                iot::mapRequired(io, "name", name);
                iot::mapRequired(io, "variableType", variableType);
                iot::mapRequired(io, "direction", direction);

                if(!iot::outputting(io))
                {
                    val = std::make_shared<CommandArgument>(
                        nullptr, variableType, offset, direction, name);
                }
            }

            static void mapping(IO& io, CommandArgumentPtr& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<VariableType, IO, Context>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, VariableType& val, Context& ctx)
            {
                iot::mapRequired(io, "dataType", val.dataType, ctx);
                iot::mapRequired(io, "pointerType", val.pointerType, ctx);
            }

            static void mapping(IO& io, VariableType& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Register::Value, IO, Context>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, Register::Value& val, Context& ctx)
            {
                CommandArgumentValue literalVal;

                if(iot::outputting(io))
                {
                    AssertFatal(val.regType() == Register::Type::Literal);
                    literalVal = val.getLiteralValue();
                }

                iot::mapRequired(io, "literalValue", literalVal, ctx);

                if(!iot::outputting(io))
                {
                    val = *Register::Value::Literal(literalVal);
                }
            }

            static void mapping(IO& io, Register::Value& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Register::ValuePtr, IO, Context>
            : public SharedPointerMappingTraits<Register::ValuePtr, IO, Context, true>
        {
            static const bool flow = true;
        };

        template <typename IO, typename Context>
        struct MappingTraits<AssemblyKernelArgumentPtr, IO, Context>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, AssemblyKernelArgumentPtr& val, Context& ctx)
            {
                if(!iot::outputting(io))
                    val = std::make_shared<AssemblyKernelArgument>();

                iot::mapRequired(io, "name", val->name);
                iot::mapRequired(io, "variableType", val->variableType);
                iot::mapRequired(io, "dataDirection", val->dataDirection);
                iot::mapRequired(io, "expression", val->expression);
                iot::mapRequired(io, "offset", val->offset);
                iot::mapRequired(io, "size", val->size);
            }

            static void mapping(IO& io, AssemblyKernelArgumentPtr& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Expression::DataFlowTag, IO, Context>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, Expression::DataFlowTag& val, Context& ctx)
            {
                iot::mapRequired(io, "tag", val.tag);
                iot::mapRequired(io, "regType", val.regType);
                iot::mapRequired(io, "varType", val.varType);
            }

            static void mapping(IO& io, Expression::DataFlowTag& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Expression::WaveTilePtr, IO, Context>
        {
            static const bool flow = true;
            using iot              = IOTraits<IO>;

            static void mapping(IO& io, Expression::WaveTilePtr& val, Context& ctx)
            {
                AssertFatal(iot::outputting(io));

                iot::mapRequired(io, "size", val->size);
                iot::mapRequired(io, "stride", val->stride);
                iot::mapRequired(io, "rank", val->rank);
                iot::mapRequired(io, "sizes", val->sizes);
                iot::mapRequired(io, "layout", val->layout);
                iot::mapRequired(io, "vgpr", val->vgpr);
            }

            static void mapping(IO& io, Expression::WaveTilePtr& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

    }
}
