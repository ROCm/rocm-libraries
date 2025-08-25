/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cmath>

#include <rocRoller/Expression.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>

#include <rocRoller/Utilities/RTTI.hpp>
#include <rocRoller/Utilities/Random.hpp>

#include <libdivide.h>

namespace rocRoller::Expression::EvaluateDetail
{
    template <CUnary UnaryExpr, DataType DESTTYPE = DataType::None>
    struct UnaryEvaluatorVisitor
    {
        UnaryExpr expr;

        using TheEvaluator = OperationEvaluatorVisitor<UnaryExpr, DESTTYPE>;

        template <CCommandArgumentValue ARG>
        requires CCanEvaluateUnary<TheEvaluator, ARG>
            CommandArgumentValue operator()(ARG const& arg) const
        {
            auto evaluator = static_cast<TheEvaluator const*>(this);
            return evaluator->evaluate(arg);
        }

        template <CCommandArgumentValue ARG>
        requires(!CCanEvaluateUnary<TheEvaluator, ARG>) CommandArgumentValue
            operator()(ARG const& arg) const
        {
            if constexpr(CHasTypeInfo<ARG>)
            {
                Throw<FatalError>("Incompatible type ",
                                  TypeInfo<ARG>::Name(),
                                  " for expression ",
                                  typeName<UnaryExpr>());
            }
            else
            {
                Throw<FatalError>("Incompatible type ",
                                  ShowValue(ARG()),
                                  typeName<ARG>(),
                                  " for expression ",
                                  typeName<UnaryExpr>());
            }
        }

        CommandArgumentValue call(CommandArgumentValue const& arg) const
        {
            return std::visit(*this, arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MagicMultiple> : public UnaryEvaluatorVisitor<MagicMultiple>
    {
        uint32_t evaluate(uint32_t const& arg) const
        {
            assertNonNullPointer(arg);
            AssertFatal(arg != 1, "Fast division not supported for denominator == 1");

            if(arg == 0)
                return std::numeric_limits<uint32_t>::max() / 2;

            auto magic = libdivide::libdivide_u32_branchfree_gen(arg);

            return magic.magic;
        }

        int32_t evaluate(int32_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return std::numeric_limits<int32_t>::max() / 2;

            auto magic = libdivide::libdivide_s32_branchfree_gen(arg);

            return magic.magic;
        }

        int64_t evaluate(int64_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return std::numeric_limits<int64_t>::max() / 2;

            auto magic = libdivide::libdivide_s64_branchfree_gen(arg);

            return magic.magic;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Convert>
    {
        Convert exp;

        template <int Idx = 0>
        CommandArgumentValue call(CommandArgumentValue const& arg)
        {
            constexpr auto IdxType = static_cast<DataType>(Idx);

            if constexpr(IdxType == DataType::None)
            {
                Throw<FatalError>("Cannot convert to None type.");
                return 0;
            }
            else if constexpr(IdxType == DataType::Count)
            {
                Throw<FatalError>("Cannot convert to Count type.");
                return 0;
            }
            else
            {
                using ResultType = typename EnumTypeInfo<IdxType>::Type;

                if(exp.destinationType == IdxType)
                {
                    if constexpr(!CCommandArgumentValue<ResultType>)
                    {
                        Throw<FatalError>("Cannot convert to ", friendlyTypeName<ResultType>());
                        return 0;
                    }
                    else
                    {
                        auto theVisitor = [](auto value) -> ResultType {
                            using ExistingType = std::decay_t<decltype(value)>;
                            if constexpr(CCanStaticCastTo<ResultType, ExistingType>)
                            {
                                return static_cast<ResultType>(value);
                            }
                            else
                            {
                                Throw<FatalError>("Cannot cast ",
                                                  friendlyTypeName<ExistingType>(),
                                                  " to ",
                                                  friendlyTypeName<ResultType>());

                                return ResultType{};
                            }
                        };

                        return std::visit(theVisitor, arg);
                    }
                }
                else
                {
                    return call<Idx + 1>(arg);
                }
            }
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MagicShifts> : public UnaryEvaluatorVisitor<MagicShifts>
    {
        int evaluate(uint32_t const& arg) const
        {
            assertNonNullPointer(arg);
            AssertFatal(arg != 1, "Fast division not supported for denominator == 1");

            if(arg == 0)
                return 0;

            auto magic = libdivide::libdivide_u32_branchfree_gen(arg);

            return magic.more & libdivide::LIBDIVIDE_32_SHIFT_MASK;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MagicShiftAndSign>
        : public UnaryEvaluatorVisitor<MagicShiftAndSign>
    {
        static_assert(libdivide::LIBDIVIDE_32_SHIFT_MASK == 31,
                      "magicNumberDivision assumes this is true.");
        static_assert(libdivide::LIBDIVIDE_64_SHIFT_MASK == 63,
                      "magicNumberDivision assumes this is true.");
        static_assert(libdivide::LIBDIVIDE_NEGATIVE_DIVISOR == 1 << 7,
                      "magicNumberDivision assumes this is true.");

        uint32_t evaluate(int32_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return 0;

            auto magic = libdivide::libdivide_s32_branchfree_gen(arg);

            return static_cast<uint32_t>(magic.more);
        }

        uint32_t evaluate(int64_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return 0;

            auto magic = libdivide::libdivide_s64_branchfree_gen(arg);

            return static_cast<uint32_t>(magic.more);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<BitwiseNegate> : public UnaryEvaluatorVisitor<BitwiseNegate>
    {
        template <CIntegral T>
        constexpr T evaluate(T const& arg) const
        {
            return ~arg;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Exponential2> : public UnaryEvaluatorVisitor<Exponential2>
    {
        template <isFP32 T>
        constexpr T evaluate(T const& arg) const
        {
            return std::exp2(arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Exponential> : public UnaryEvaluatorVisitor<Exponential>
    {
        template <isFP32 T>
        constexpr T evaluate(T const& arg) const
        {
            return std::exp(arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Negate> : public UnaryEvaluatorVisitor<Negate>
    {
        template <typename T>
        requires(std::floating_point<T> || std::signed_integral<T>) constexpr T
            evaluate(T const& arg) const
        {
            return -arg;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<LogicalNot> : public UnaryEvaluatorVisitor<LogicalNot>
    {
        template <typename T>
        constexpr bool evaluate(T const& arg) const
        {
            return !arg;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<RandomNumber> : public UnaryEvaluatorVisitor<RandomNumber>
    {
        template <typename T>
        requires(!std::same_as<bool, T> && std::unsigned_integral<T>) constexpr T
            evaluate(T const& arg) const
        {
            return LFSRRandomNumberGenerator(arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<BitFieldExtract>
        : public UnaryEvaluatorVisitor<BitFieldExtract>
    {
        template <typename T>
        constexpr T evaluate(T const& arg) const
        {
            Throw<FatalError>("BitFieldExtract present in runtime expression.");
            return T{};
        }
    };

    template <>
    struct OperationEvaluatorVisitor<ToScalar> : public UnaryEvaluatorVisitor<ToScalar>
    {
        template <typename T>
        constexpr T evaluate(T arg) const
        {
            return arg;
        }
    };

    struct ToBoolVisitor
    {
        template <typename T>
        constexpr bool operator()(T const& val)
        {
            Throw<FatalError>("Invalid bool type: ", typeName<T>());
            return 0;
        }

        template <std::integral T>
        constexpr bool operator()(T const& val)
        {
            return val != 0;
        }

        bool call(CommandArgumentValue const& val)
        {
            return std::visit(*this, val);
        }
    };

    template <CUnary T>
    __attribute__((noinline)) CommandArgumentValue evaluateOp(T const&                    op,
                                                              CommandArgumentValue const& arg)
    {
        OperationEvaluatorVisitor<T> visitor{op};
        return visitor.call(arg);
    }

#define INSTANTIATE_UNARY_OP(Op) \
    template CommandArgumentValue evaluateOp<Op>(Op const& op, CommandArgumentValue const& arg)

    INSTANTIATE_UNARY_OP(MagicMultiple);
    INSTANTIATE_UNARY_OP(MagicShifts);
    INSTANTIATE_UNARY_OP(MagicShiftAndSign);
    INSTANTIATE_UNARY_OP(Negate);
    INSTANTIATE_UNARY_OP(BitwiseNegate);
    INSTANTIATE_UNARY_OP(LogicalNot);
    INSTANTIATE_UNARY_OP(Exponential2);
    INSTANTIATE_UNARY_OP(Exponential);
    INSTANTIATE_UNARY_OP(RandomNumber);
    INSTANTIATE_UNARY_OP(ToScalar);
    INSTANTIATE_UNARY_OP(BitFieldExtract);
    INSTANTIATE_UNARY_OP(Convert);

}
