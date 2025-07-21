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
#include <memory>
#include <stack>
#include <typeinfo>
#include <variant>

#include <rocRoller/Expression.hpp>

#include <rocRoller/AssemblyKernelArgument.hpp>

#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Utilities/RTTI.hpp>
#include <rocRoller/Utilities/Random.hpp>

#include <libdivide.h>

namespace rocRoller
{
    namespace Expression
    {
        template <typename T>
        concept CIntegral = std::integral<T> && !std::same_as<bool, T>;

        template <typename T>
        concept isFP32 = std::same_as<float, T>;

        /**
         * Visitor for a specific Operation expression.  Performs that
         * specific Operation (Add, subtract, etc).  Does not walk the
         * expression tree.
         */
        template <typename T, DataType DESTTYPE = DataType::None>
        struct OperationEvaluatorVisitor
        {
        };

        /**
         * Visitor for an Expression.  Walks the expression tree,
         * calling the OperationEvaluatorVisitor to perform actual
         * operations.
         */
        struct EvaluateVisitor;

        /**
         * Is satisfied if the binary operation associated with TheEvaluator can be
         * applied to LHS and RHS.
         * e.g. TheEvaluator == OperationEvaluatorVisitor<Add>, LHS -> int, RHS -> int
         *
         * Note that this depends on evaluate() not being defined for invalid pairs of
         * types, thus the use of e.g. `CCanAdd` below.
         *
         * @tparam TheEvaluator Specialization of OperationEvaluatorVisitor class for a specific operation
         * @tparam LHS
         * @tparam RHS
         */
        template <typename TheEvaluator, typename LHS, typename RHS>
        concept CCanEvaluateBinary = requires(TheEvaluator ev, LHS lhs, RHS rhs)
        {
            requires CCommandArgumentValue<LHS>;
            requires CCommandArgumentValue<RHS>;

            {
                ev.evaluate(lhs, rhs)
                } -> CCommandArgumentValue;
        };

        template <CBinary BinaryExpr>
        struct BinaryEvaluatorVisitor
        {
            using TheEvaluator = OperationEvaluatorVisitor<BinaryExpr>;

            template <CCommandArgumentValue T>
            void assertNonNullPointer(T const& val) const
            {
                if constexpr(std::is_pointer<T>::value)
                {
                    AssertFatal(val, "Can't offset from nullptr!");
                }
            }

            template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
            requires CCanEvaluateBinary<TheEvaluator, LHS, RHS>
                CommandArgumentValue operator()(LHS const& lhs, RHS const& rhs) const
            {
                auto evaluator = static_cast<TheEvaluator const*>(this);
                return evaluator->evaluate(lhs, rhs);
            }

            template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
            requires(!CCanEvaluateBinary<TheEvaluator, LHS, RHS>) CommandArgumentValue
                operator()(LHS const& lhs, RHS const& rhs) const
            {
                if constexpr(CHasTypeInfo<LHS> && CHasTypeInfo<RHS>)
                {
                    Throw<FatalError>("Type mismatch for expression: ",
                                      typeName<BinaryExpr>(),
                                      ". Argument ",
                                      ShowValue(TypeInfo<LHS>::Var),
                                      " incompatible with ",
                                      ShowValue(TypeInfo<RHS>::Var),
                                      ").");
                }
                else
                {
                    Throw<FatalError>("Type mismatch for expression: ",
                                      typeName<BinaryExpr>(),
                                      ". Argument ",
                                      ShowValue(typeName<LHS>()),
                                      " incompatible with ",
                                      ShowValue(typeName<RHS>()),
                                      ").");
                }
            }

            CommandArgumentValue call(CommandArgumentValue const& lhs,
                                      CommandArgumentValue const& rhs) const
            {
                return std::visit(*this, lhs, rhs);
            }
        };

        template <typename TheEvaluator, typename LHS, typename R1HS, typename R2HS>
        concept CCanEvaluateTernary = requires(TheEvaluator ev, LHS lhs, R1HS r1hs, R2HS r2hs)
        {
            requires CCommandArgumentValue<LHS>;
            requires CCommandArgumentValue<R1HS>;
            requires CCommandArgumentValue<R2HS>;

            {
                ev.evaluate(lhs, r1hs, r2hs)
                } -> CCommandArgumentValue;
        };

        template <CTernary TernaryExpr>
        struct TernaryEvaluatorVisitor
        {
            using TheEvaluator = OperationEvaluatorVisitor<TernaryExpr>;

            template <CCommandArgumentValue T>
            void assertNonNullPointer(T const& val) const
            {
                if constexpr(std::is_pointer<T>::value)
                {
                    AssertFatal(val, "Can't offset from nullptr!");
                }
            }

            template <CCommandArgumentValue LHS,
                      CCommandArgumentValue R1HS,
                      CCommandArgumentValue R2HS>
            requires CCanEvaluateTernary<TheEvaluator, LHS, R1HS, R2HS> CommandArgumentValue
                operator()(LHS const& lhs, R1HS const& r1hs, R2HS const& r2hs) const
            {
                auto evaluator = static_cast<TheEvaluator const*>(this);
                return evaluator->evaluate(lhs, r1hs, r2hs);
            }

            template <CCommandArgumentValue LHS,
                      CCommandArgumentValue R1HS,
                      CCommandArgumentValue R2HS>
            requires(!CCanEvaluateTernary<TheEvaluator, LHS, R1HS, R2HS>) CommandArgumentValue
                operator()(LHS const& lhs, R1HS const& r1hs, R2HS const& r2hs) const
            {
                Throw<FatalError>("Type mismatch for expression: ",
                                  typeName<TernaryExpr>(),
                                  ". Incompatible arguments ",
                                  ShowValue(typeName<LHS>()),
                                  " ",
                                  ShowValue(typeName<R1HS>()),
                                  " ",
                                  ShowValue(typeName<R2HS>()),
                                  ").");
            }

            CommandArgumentValue call(CommandArgumentValue const& lhs,
                                      CommandArgumentValue const& r1hs,
                                      CommandArgumentValue const& r2hs) const
            {
                return std::visit(*this, lhs, r1hs, r2hs);
            }
        };

        /**
         * For example, CCanAdd which satisifes pairs of types that can be added.
         */
#define CAN_OPERATE_CONCEPT(name, op)               \
    template <typename LHS, typename RHS>           \
    concept CCan##name = requires(LHS lhs, RHS rhs) \
    {                                               \
        lhs op rhs;                                 \
    }

        /**
         * Declares a BinaryEvaluatorVisitor that can be defined solely by a single binary expression.
         */
#define BINARY_EVALUATOR_VISITOR(name, op)                                          \
    template <>                                                                     \
    struct OperationEvaluatorVisitor<name> : public BinaryEvaluatorVisitor<name>    \
    {                                                                               \
        template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>             \
        requires CCan##name<LHS, RHS> constexpr auto evaluate(LHS const& lhs,       \
                                                              RHS const& rhs) const \
        {                                                                           \
            return lhs op rhs;                                                      \
        }                                                                           \
    }

#define SIMPLE_BINARY_OP(name, op) \
    CAN_OPERATE_CONCEPT(name, op); \
    BINARY_EVALUATOR_VISITOR(name, op)

        /**
         * Is satisfied if the unary operation associated with TheEvaluator can be
         * applied to ARG.
         * e.g. TheEvaluator == OperationEvaluatorVisitor<Not>, Arg -> bool
         *
         *
         * @tparam TheEvaluator Specialization of OperationEvaluatorVisitor class for a specific operation
         * @tparam ARG
         */
        template <typename TheEvaluator, typename ARG>
        concept CCanEvaluateUnary = requires(TheEvaluator ev, ARG arg)
        {
            requires CCommandArgumentValue<ARG>;

            {
                ev.evaluate(arg)
                } -> CCommandArgumentValue;
        };

        template <CUnary UnaryExpr, DataType DESTTYPE = DataType::None>
        struct UnaryEvaluatorVisitor
        {
            using TheEvaluator = OperationEvaluatorVisitor<UnaryExpr, DESTTYPE>;

            template <CCommandArgumentValue T>
            void assertNonNullPointer(T const& val) const
            {
                if constexpr(std::is_pointer<T>::value)
                {
                    AssertFatal(val, "Can't offset from nullptr!");
                }
            }

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

        CAN_OPERATE_CONCEPT(Add, +);
        static_assert(CCanAdd<int, int>);
        static_assert(CCanAdd<float*, int>);
        static_assert(!CCanAdd<float, int*>);

        /**
         * Custom declared so that we can check for null pointers where appropriate.
         */
        template <>
        struct OperationEvaluatorVisitor<Add> : public BinaryEvaluatorVisitor<Add>
        {
            template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
            requires CCanAdd<LHS, RHS>
            auto evaluate(LHS const& lhs, RHS const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return lhs + rhs;
            }
        };

        CAN_OPERATE_CONCEPT(Subtract, -);

        /**
         * Custom declared so that we can check for null pointers where appropriate.
         */
        template <>
        struct OperationEvaluatorVisitor<Subtract> : public BinaryEvaluatorVisitor<Subtract>
        {
            template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
            requires CCanSubtract<LHS, RHS>
            auto evaluate(LHS const& lhs, RHS const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return lhs - rhs;
            }
        };

        SIMPLE_BINARY_OP(Multiply, *);
        SIMPLE_BINARY_OP(Divide, /);
        SIMPLE_BINARY_OP(Modulo, %);

        SIMPLE_BINARY_OP(ShiftL, <<);
        SIMPLE_BINARY_OP(ArithmeticShiftR, >>);
        SIMPLE_BINARY_OP(BitwiseAnd, &);
        SIMPLE_BINARY_OP(BitwiseOr, |);
        SIMPLE_BINARY_OP(BitwiseXor, ^);

        SIMPLE_BINARY_OP(GreaterThan, >);
        SIMPLE_BINARY_OP(GreaterThanEqual, >=);
        SIMPLE_BINARY_OP(LessThan, <);
        SIMPLE_BINARY_OP(LessThanEqual, <=);
        SIMPLE_BINARY_OP(Equal, ==);
        SIMPLE_BINARY_OP(NotEqual, !=);
        ;
        SIMPLE_BINARY_OP(LogicalAnd, &&);
        SIMPLE_BINARY_OP(LogicalOr, ||);

        template <>
        struct OperationEvaluatorVisitor<MultiplyHigh> : public BinaryEvaluatorVisitor<MultiplyHigh>
        {
            int evaluate(int const& lhs, int const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return (lhs * (int64_t)rhs) >> 32;
            }

            int evaluate(int const& lhs, unsigned int const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return (lhs * (int64_t)rhs) >> 32;
            }

            int evaluate(unsigned int const& lhs, int const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return (lhs * (int64_t)rhs) >> 32;
            }

            unsigned int evaluate(unsigned int const& lhs, unsigned int const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return (lhs * (uint64_t)rhs) >> 32;
            }

            int64_t evaluate(int const& lhs, int64_t const& rhs) const
            {
                return evaluate((int64_t)lhs, rhs);
            }

            int64_t evaluate(int64_t const& lhs, int64_t const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return ((__int128_t)lhs * (__int128_t)rhs) >> 64;
            }

            uint64_t evaluate(uint64_t const& lhs, uint64_t const& rhs) const
            {
                assertNonNullPointer(lhs);
                assertNonNullPointer(rhs);

                return ((__uint128_t)lhs * (__uint128_t)rhs) >> 64;
            }
        };

        template <>
        struct OperationEvaluatorVisitor<LogicalShiftR>
            : public BinaryEvaluatorVisitor<LogicalShiftR>
        {
            template <CIntegral T, CIntegral U>
            constexpr T evaluate(T const& lhs, U const& rhs) const
            {
                return static_cast<typename std::make_unsigned<T>::type>(lhs) >> rhs;
            }
        };

        template <DataType T_DataType>
        struct OperationEvaluatorVisitor<SRConvert<T_DataType>>
            : public BinaryEvaluatorVisitor<SRConvert<T_DataType>>
        {
            using Base       = BinaryEvaluatorVisitor<SRConvert<T_DataType>>;
            using ResultType = typename EnumTypeInfo<T_DataType>::Type;

            template <CArithmeticType T>
            requires CCanStaticCastTo<ResultType, T> ResultType evaluate(T const& arg)
            const
            {
                Base::assertNonNullPointer(arg);
                return static_cast<ResultType>(arg);
            }
        };

        template <>
        struct OperationEvaluatorVisitor<MagicMultiple>
            : public UnaryEvaluatorVisitor<MagicMultiple>
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

        template <DataType DESTTYPE>
        struct OperationEvaluatorVisitor<Convert, DESTTYPE>
            : public UnaryEvaluatorVisitor<Convert, DESTTYPE>
        {
            using Base    = UnaryEvaluatorVisitor<Convert, DESTTYPE>;
            using ResType = typename EnumTypeInfo<DESTTYPE>::Type;

            template <CArithmeticType T>
            requires CCanStaticCastTo<ResType, T> ResType evaluate(T const& arg)
            const
            {
                Base::assertNonNullPointer(arg);
                return static_cast<ResType>(arg);
            }
        };

        static_assert(
            std::same_as<OperationEvaluatorVisitor<Convert, DataType::Double>::ResType, double>);
        static_assert(
            CCanEvaluateUnary<OperationEvaluatorVisitor<Convert, DataType::Double>, float>);
        static_assert(
            CCanEvaluateUnary<OperationEvaluatorVisitor<Convert, DataType::Double>, double>);

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
        struct OperationEvaluatorVisitor<BitwiseNegate>
            : public UnaryEvaluatorVisitor<BitwiseNegate>
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
        struct OperationEvaluatorVisitor<AddShiftL>
        {
            CommandArgumentValue call(CommandArgumentValue const& lhs,
                                      CommandArgumentValue const& r1hs,
                                      CommandArgumentValue const& r2hs) const
            {
                auto adder   = OperationEvaluatorVisitor<Add>();
                auto sum     = adder.call(lhs, r1hs);
                auto shifter = OperationEvaluatorVisitor<ShiftL>();
                return shifter.call(sum, r2hs);
            }
        };

        template <>
        struct OperationEvaluatorVisitor<ShiftLAdd>
        {
            CommandArgumentValue call(CommandArgumentValue const& lhs,
                                      CommandArgumentValue const& r1hs,
                                      CommandArgumentValue const& r2hs) const
            {
                auto adder   = OperationEvaluatorVisitor<Add>();
                auto shifter = OperationEvaluatorVisitor<ShiftL>();

                auto temp = shifter.call(lhs, r1hs);
                return adder.call(temp, r2hs);
            }
        };

        template <>
        struct OperationEvaluatorVisitor<MultiplyAdd>
        {
            CommandArgumentValue call(CommandArgumentValue const& lhs,
                                      CommandArgumentValue const& r1hs,
                                      CommandArgumentValue const& r2hs) const
            {
                auto adder      = OperationEvaluatorVisitor<Add>();
                auto multiplier = OperationEvaluatorVisitor<Multiply>();

                auto product = multiplier.call(lhs, r1hs);
                return adder.call(product, r2hs);
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

        template <>
        struct OperationEvaluatorVisitor<Conditional>
        {

            CommandArgumentValue call(CommandArgumentValue const& lhs,
                                      CommandArgumentValue const& r1hs,
                                      CommandArgumentValue const& r2hs) const
            {
                AssertFatal(r1hs.index() == r2hs.index(),
                            "Types of each conditional option must match!",
                            ShowValue(r1hs),
                            ShowValue(r2hs));

                if(ToBoolVisitor().call(lhs))
                {
                    return r1hs;
                }
                else
                {
                    return r2hs;
                }
            }
        };

        struct EvaluateVisitor
        {
            RuntimeArguments args;

            template <CTernary TernaryExp>
            CommandArgumentValue operator()(TernaryExp const& expr)
            {
                // TODO: Short-circuit logic
                auto arg1      = call(expr.lhs);
                auto arg2      = call(expr.r1hs);
                auto arg3      = call(expr.r2hs);
                auto evaluator = OperationEvaluatorVisitor<TernaryExp>();
                return evaluator.call(arg1, arg2, arg3);
            }

            template <CBinary BinaryExp>
            CommandArgumentValue operator()(BinaryExp const& expr)
            {
                // TODO: Short-circuit logic
                auto lhs       = call(expr.lhs);
                auto rhs       = call(expr.rhs);
                auto evaluator = OperationEvaluatorVisitor<BinaryExp>();
                return evaluator.call(lhs, rhs);
            }

            template <CUnary UnaryExp>
            CommandArgumentValue operator()(UnaryExp const& expr)
            {
                auto arg       = call(expr.arg);
                auto evaluator = OperationEvaluatorVisitor<UnaryExp>();
                return evaluator.call(arg);
            }

            CommandArgumentValue operator()(Convert const& expr)
            {
#define ConvertCase(dtype)                                                      \
    case DataType::dtype:                                                       \
    {                                                                           \
        auto evaluator = OperationEvaluatorVisitor<Convert, DataType::dtype>(); \
        return evaluator.call(arg);                                             \
    }

                auto     arg      = call(expr.arg);
                DataType destType = expr.destinationType;
                switch(expr.destinationType)
                {
                    // clang-format off
                    ConvertCase(Half)
                    ConvertCase(Halfx2)
                    ConvertCase(FP8)
                    ConvertCase(BF8)
                    ConvertCase(FP8x4)
                    ConvertCase(BF8x4)
                    ConvertCase(BFloat16)
                    ConvertCase(BFloat16x2)
                    ConvertCase(Float)
                    ConvertCase(Double)
                    ConvertCase(Int32)
                    ConvertCase(Int64)
                    ConvertCase(UInt32)
                    ConvertCase(UInt64)
                    ConvertCase(Bool)
                    ConvertCase(Bool32)
                    ConvertCase(Bool64)
                default: // clang-format on
                          Throw<FatalError>(
                              "At EvaluateVisitor: No convert operation is supported.",
                              ShowValue(expr));
                }
#undef ConvertCase
            }

            CommandArgumentValue operator()(BitFieldExtract const& expr)
            {
                throw std::runtime_error("BitFieldExtract present in runtime expression.");
            }

            CommandArgumentValue operator()(MatrixMultiply const& expr)
            {
                throw std::runtime_error("Matrix multiply present in runtime expression.");
            }

            CommandArgumentValue operator()(ScaledMatrixMultiply const& expr)
            {
                throw std::runtime_error("Scaled Matrix multiply present in runtime expression.");
            }

            CommandArgumentValue operator()(Register::ValuePtr const& expr)
            {
                if(expr->regType() == Register::Type::Literal)
                    return expr->getLiteralValue();

                Throw<FatalError>("Register present in runtime expression", ShowValue(expr));
            }

            CommandArgumentValue operator()(ToScalar const& expr)
            {
                return call(expr.arg);
            }

            CommandArgumentValue operator()(CommandArgumentPtr const& expr)
            {
                return expr->getValue(args);
            }

            CommandArgumentValue operator()(CommandArgumentValue const& expr)
            {
                return expr;
            }

            CommandArgumentValue operator()(AssemblyKernelArgumentPtr const& expr)
            {
                return call(expr->expression);
            }

            CommandArgumentValue operator()(DataFlowTag const& expr)
            {
                Throw<FatalError>("Data flow tag present in runtime expression", ShowValue(expr));
            }

            CommandArgumentValue operator()(PositionalArgument const& expr)
            {
                Throw<FatalError>("Positional argument present in runtime expression",
                                  ShowValue(expr));
            }

            CommandArgumentValue operator()(WaveTilePtr const& expr)
            {
                Throw<FatalError>("Wave tile present in runtime expression", ShowValue(expr));
            }

            CommandArgumentValue call(Expression const& expr)
            {
                auto rv = std::visit(*this, expr);

                auto exprType = resultVariableType(expr);
                auto result   = resultType(rv).varType;
                AssertFatal(
                    exprType == result, ShowValue(expr), ShowValue(exprType), ShowValue(result));

                return rv;
            }

            CommandArgumentValue call(ExpressionPtr const& expr)
            {
                AssertFatal(expr != nullptr, "Found nullptr in expression");
                return call(*expr);
            }

            CommandArgumentValue call(ExpressionPtr const&    expr,
                                      RuntimeArguments const& runtimeArgs)
            {
                args = runtimeArgs;
                return call(expr);
            }

            CommandArgumentValue call(Expression const& expr, RuntimeArguments const& runtimeArgs)
            {
                args = runtimeArgs;
                return call(expr);
            }
        };

        CommandArgumentValue evaluate(ExpressionPtr const& expr, RuntimeArguments const& args)
        {
            return EvaluateVisitor().call(expr, args);
        }

        CommandArgumentValue evaluate(Expression const& expr, RuntimeArguments const& args)
        {
            return EvaluateVisitor().call(expr, args);
        }

        CommandArgumentValue evaluate(ExpressionPtr const& expr)
        {
            return EvaluateVisitor().call(expr);
        }

        CommandArgumentValue evaluate(Expression const& expr)
        {
            return EvaluateVisitor().call(expr);
        }

        bool canEvaluateTo(CommandArgumentValue val, ExpressionPtr const& expr)
        {
            if(evaluationTimes(expr)[EvaluationTime::Translate])
            {
                return evaluate(expr) == val;
            }
            return false;
        }

        static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Add>, double, double>);
        static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Add>, double, int*>);
        static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Add>, float*, int>);

        static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Subtract>, double, double>);
        static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Subtract>, double, int*>);
        static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Subtract>, float*, int>);

        static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Multiply>, double, double>);
        static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Multiply>, double, int*>);
        static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Multiply>, float*, int>);
    }
}
