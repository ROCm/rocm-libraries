#pragma once

#ifdef ROCROLLER_USE_LLVM
#include <llvm/ObjectYAML/YAML.h>
#endif

#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations_fwd.hpp>

#include "Base.hpp"
#include "Containers.hpp"
#include "Variant.hpp"

namespace rocRoller
{
    namespace Serialization
    {

        template <Operations::CUnaryXOp TOp, typename IO, typename Context>
        struct MappingTraits<TOp, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "dest", op.dest);
                iot::mapRequired(io, "a", op.a);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <Operations::CBinaryXOp TOp, typename IO, typename Context>
        struct MappingTraits<TOp, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "dest", op.dest);
                iot::mapRequired(io, "a", op.a);
                iot::mapRequired(io, "b", op.b);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <Operations::CTernaryXOp TOp, typename IO, typename Context>
        struct MappingTraits<TOp, IO, Context>
        {
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                iot::mapRequired(io, "dest", op.dest);
                iot::mapRequired(io, "a", op.a);
                iot::mapRequired(io, "b", op.b);
                iot::mapRequired(io, "c", op.c);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));
                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Neg>
        {
            static Operations::XOp call()
            {
                return Operations::E_Neg(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Abs>
        {
            static Operations::XOp call()
            {
                return Operations::E_Abs(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Not>
        {
            static Operations::XOp call()
            {
                return Operations::E_Not(Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Cvt>
        {
            static Operations::XOp call()
            {
                return Operations::E_Cvt(Operations::OperationTag(-1), DataType::None);
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_StochasticRoundingCvt>
        {
            static Operations::XOp call()
            {
                return Operations::E_StochasticRoundingCvt(
                    Operations::OperationTag(-1), Operations::OperationTag(-1), DataType::None);
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Add>
        {
            static Operations::XOp call()
            {
                return Operations::E_Add(Operations::OperationTag(-1),
                                         Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Sub>
        {
            static Operations::XOp call()
            {
                return Operations::E_Sub(Operations::OperationTag(-1),
                                         Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Mul>
        {
            static Operations::XOp call()
            {
                return Operations::E_Mul(Operations::OperationTag(-1),
                                         Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Div>
        {
            static Operations::XOp call()
            {
                return Operations::E_Div(Operations::OperationTag(-1),
                                         Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_And>
        {
            static Operations::XOp call()
            {
                return Operations::E_And(Operations::OperationTag(-1),
                                         Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Or>
        {
            static Operations::XOp call()
            {
                return Operations::E_Or(Operations::OperationTag(-1), Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_GreaterThan>
        {
            static Operations::XOp call()
            {
                return Operations::E_GreaterThan(Operations::OperationTag(-1),
                                                 Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_Conditional>
        {
            static Operations::XOp call()
            {
                return Operations::E_Conditional(Operations::OperationTag(-1),
                                                 Operations::OperationTag(-1),
                                                 Operations::OperationTag(-1));
            }
        };

        template <>
        struct DefaultConstruct<Operations::XOp, Operations::E_RandomNumber>
        {
            static Operations::XOp call()
            {
                return Operations::E_RandomNumber(Operations::OperationTag(-1));
            }
        };

        template <typename IO, typename Context>
        struct MappingTraits<Operations::XOp, IO, Context>
            : public DefaultVariantMappingTraits<Operations::XOp, IO, Context>
        {
        };

        template <typename IO, typename Context>
        struct MappingTraits<std::shared_ptr<Operations::XOp>, IO, Context>
        {
            using TOp = std::shared_ptr<Operations::XOp>;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, TOp& op, Context& ctx)
            {
                if(!iot::outputting(io))
                    op = std::make_shared<Operations::XOp>(
                        Operations::E_Neg(Operations::OperationTag{-1}));
                MappingTraits<Operations::XOp, IO, Context>::mapping(io, *op, ctx);
            }

            static void mapping(IO& io, TOp& val)
            {
                AssertFatal((std::same_as<EmptyContext, Context>));

                Context ctx;
                mapping(io, val, ctx);
            }
        };

        template <typename IO>
        struct SequenceTraits<std::vector<std::shared_ptr<Operations::XOp>>, IO>
            : public DefaultSequenceTraits<std::vector<std::shared_ptr<Operations::XOp>>, IO, false>
        {
        };
    }
}
