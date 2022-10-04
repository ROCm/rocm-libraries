#pragma once

#include "../DataTypes/DataTypes.hpp"

#include "AssemblyKernel.hpp"
#include "Base.hpp"
#include "Enum.hpp"
#include "HasTraits.hpp"
#include "Variant.hpp"

#include "../Expression.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<Expression::ExpressionPtr, IO, EmptyContext>
            : public SharedPointerMappingTraits<Expression::ExpressionPtr, IO, EmptyContext>
        {
        };

        template <typename IO>
        struct MappingTraits<Expression::Expression, IO>
            : public DefaultVariantMappingTraits<Expression::Expression, IO>
        {
        };

        template <CNamedVariant T, typename IO, typename Context>
        const typename DefaultVariantMappingTraits<T, IO, Context>::AlternativeMap
            DefaultVariantMappingTraits<T, IO, Context>::alternatives
            = DefaultVariantMappingTraits<T, IO, Context>::GetAlternatives();

        template <Expression::CBinary TExp, typename IO, typename Context>
        struct MappingTraits<TExp, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TExp& exp, Context& ctx)
            {
                iot::mapRequired(io, "lhs", exp.lhs, ctx);
                iot::mapRequired(io, "rhs", exp.rhs, ctx);
            }

            static std::enable_if_t<std::same_as<EmptyContext, Context>> mapping(IO& io, TExp& val)
            {
                EmptyContext ctx;
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

            static std::enable_if_t<std::same_as<EmptyContext, Context>> mapping(IO& io, TExp& val)
            {
                EmptyContext ctx;
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

            static std::enable_if_t<std::same_as<EmptyContext, Context>> mapping(IO& io, TExp& val)
            {
                EmptyContext ctx;
                mapping(io, val, ctx);
            }
        };

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

                iot::mapRequired(io, "dataType", typeName, ctx);

                if(!iot::outputting(io))
                {
                    val = Base::alternatives.at(typeName)();
                }

                std::visit(
                    [&io, &ctx](auto& theVal) {
                        using U = std::decay_t<decltype(theVal)>;

                        if constexpr(std::is_pointer_v<U>)
                        {
                            Throw<FatalError>("Can't deserialize pointer values.");
                        }
                        else
                        {
                            iot::mapRequired(io, "value", theVal, ctx);
                        }
                    },
                    val);
            }

            static std::enable_if_t<std::same_as<EmptyContext, Context>>
                mapping(IO& io, CommandArgumentValue& val)
            {
                EmptyContext ctx;
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
            }

            static std::enable_if_t<std::same_as<EmptyContext, Context>>
                mapping(IO& io, CommandArgumentPtr& val)
            {
                EmptyContext ctx;
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

            static std::enable_if_t<std::same_as<EmptyContext, Context>> mapping(IO&           io,
                                                                                 VariableType& val)
            {
                EmptyContext ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Register::ValuePtr, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Register::ValuePtr& val, Context& ctx)
            {
                CommandArgumentValue literalVal;

                if(iot::outputting(io))
                {
                    AssertFatal(val->regType() == Register::Type::Literal);
                    literalVal = val->getLiteralValue();
                }

                iot::mapRequired(io, "literalValue", literalVal, ctx);

                if(!iot::outputting(io))
                {
                    val = Register::Value::Literal(literalVal);
                }
            }

            static std::enable_if_t<std::same_as<EmptyContext, Context>>
                mapping(IO& io, Register::ValuePtr& val)
            {
                EmptyContext ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<AssemblyKernelArgumentPtr, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, AssemblyKernelArgumentPtr& val, Context& ctx)
            {
                AssertFatal(iot::outputting(io));

                iot::mapRequired(io, "name", val->name, ctx);
                iot::mapRequired(io, "variableType", val->variableType, ctx);
                iot::mapRequired(io, "dataDirection", val->dataDirection, ctx);
                iot::mapRequired(io, "expression", val->expression, ctx);
                iot::mapRequired(io, "offset", val->offset, ctx);
                iot::mapRequired(io, "size", val->size, ctx);
            }

            static std::enable_if_t<std::same_as<EmptyContext, Context>>
                mapping(IO& io, AssemblyKernelArgumentPtr& val)
            {
                EmptyContext ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Expression::DataFlowTag, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Expression::DataFlowTag& val, Context& ctx)
            {
                iot::mapRequired(io, "tag", val.tag, ctx);
                iot::mapRequired(io, "regType", val.regType, ctx);
                iot::mapRequired(io, "varType", val.varType, ctx);
            }

            static std::enable_if_t<std::same_as<EmptyContext, Context>>
                mapping(IO& io, Expression::DataFlowTag& val)
            {
                EmptyContext ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Expression::WaveTilePtr, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Expression::WaveTilePtr& val, Context& ctx)
            {
                AssertFatal(iot::outputting(io));

                iot::mapRequired(io, "tag", val->tag, ctx);
                iot::mapRequired(io, "output", val->output, ctx);
                iot::mapRequired(io, "size", val->size, ctx);
                iot::mapRequired(io, "stride", val->stride, ctx);
                iot::mapRequired(io, "rank", val->rank, ctx);
                iot::mapRequired(io, "sizes", val->sizes, ctx);
                iot::mapRequired(io, "layout", val->layout, ctx);
                iot::mapRequired(io, "vgpr", val->vgpr, ctx);
            }

            static std::enable_if_t<std::same_as<EmptyContext, Context>>
                mapping(IO& io, Expression::WaveTilePtr& val)
            {
                EmptyContext ctx;
                mapping(io, val, ctx);
            }
        };

    } // namespace Serialization
} // namespace Tensile

#ifdef ROCROLLER_USE_LLVM
static_assert(
    rocRoller::Serialization::CMappedType<rocRoller::Expression::ExpressionPtr, llvm::yaml::IO>);
static_assert(
    rocRoller::Serialization::has_SerializationTraits<rocRoller::Expression::ExpressionPtr,
                                                      llvm::yaml::IO>::value);
static_assert(!llvm::yaml::has_FlowTraits<rocRoller::Expression::ExpressionPtr>::value);
#endif
