

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct CodeGeneratorVisitor
        {
            ContextPtr m_context;
            using RegisterValue = std::variant<Register::ValuePtr>;

            template <typename Operation>
            Generator<Instruction> callArithmeticUnary(ArithmeticPtr const&      arith,
                                                       Register::ValuePtr const& dest,
                                                       Register::ValuePtr const& arg);

            template <typename Operation>
            Generator<Instruction> callArithmeticBinary(ArithmeticPtr const&      arith,
                                                        Register::ValuePtr const& dest,
                                                        Register::ValuePtr const& lhs,
                                                        Register::ValuePtr const& rhs)
            {
                Throw<FatalError>("Unsupported callArithmeticBinaryOperation");
            }

            template <typename Operation>
            Generator<Instruction> callArithmeticTernary(ArithmeticPtr const&      arith,
                                                         Register::ValuePtr const& dest,
                                                         Register::ValuePtr const& lhs,
                                                         Register::ValuePtr const& r1hs,
                                                         Register::ValuePtr const& r2hs);

            Register::ValuePtr resultPlaceholder(ResultType const& resType)
            {
                if(resType.first == Register::Type::Special && resType.second == DataType::Bool)
                {
                    return m_context->getSCC();
                }
                else if(resType.first == Register::Type::Scalar
                        && resType.second == DataType::Bool32)
                {
                    return Register::Value::WavefrontPlaceholder(m_context);
                }
                else
                {
                    return Register::Value::Placeholder(
                        m_context, resType.first, resType.second, 1);
                }
            }

            template <CBinary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                Register::ValuePtr lhsResult, rhsResult;

                // TODO: Combine these using a scheduler.
                std::vector<Generator<Instruction>> subExprs;
                subExprs.push_back(call(lhsResult, expr.lhs));
                subExprs.push_back(call(rhsResult, expr.rhs));

                auto scheduler = Component::Get<Scheduling::Scheduler>(
                    Scheduling::SchedulerProcedure::Sequential, m_context);
                co_yield (*scheduler)(subExprs);

                if(dest == nullptr)
                {
                    dest = resultPlaceholder(resultType(expr));

                    // TODO: Delete once FastDivision uses only libdivide.
                    if constexpr(std::same_as<MultiplyHigh, Operation>)
                        dest = Register::Value::Placeholder(
                            m_context, dest->regType(), DataType::Int32, dest->valueCount());
                }

                auto generator = GetGenerator<Operation>(dest, lhsResult, rhsResult);
                if(generator)
                {
                    co_yield generator->generate(dest, lhsResult, rhsResult);
                    co_return;
                }

                ArithmeticPtr arith;

                // TODO: Delete once FastDivision uses only libdivide.
                if constexpr(std::same_as<MultiplyHigh, Operation>)
                {
                    arith = Component::Get<Arithmetic>(m_context, dest->regType(), DataType::Int32);
                }
                else
                {
                    // This portion should remain after FastDivision fix.
                    if constexpr(CArithmetic<Operation>)
                        arith = Arithmetic::Get(dest, lhsResult, rhsResult);
                    else
                        arith = Arithmetic::GetComparison(dest, lhsResult, rhsResult);
                }

                co_yield callArithmeticBinary<Operation>(arith, dest, lhsResult, rhsResult);
            }

            template <CTernary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                Register::ValuePtr lhsResult, r1hsResult, r2hsResult;

                // TODO: Combine these using a scheduler.
                std::vector<Generator<Instruction>> subExprs;
                subExprs.push_back(call(lhsResult, expr.lhs));
                subExprs.push_back(call(r1hsResult, expr.r1hs));
                subExprs.push_back(call(r2hsResult, expr.r2hs));

                auto scheduler = Component::Get<Scheduling::Scheduler>(
                    Scheduling::SchedulerProcedure::Sequential, m_context);
                co_yield (*scheduler)(subExprs);

                ArithmeticPtr arith;
                arith = Arithmetic::Get(dest, lhsResult, r1hsResult, r2hsResult);
                co_yield callArithmeticTernary<Operation>(
                    arith, dest, lhsResult, r1hsResult, r2hsResult);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, MatrixMultiply expr)
            {
                Register::ValuePtr lhs, rhs;
                int                M, N, K, B;

                if(std::holds_alternative<WaveTilePtr>(*expr.lhs)
                   && std::holds_alternative<WaveTilePtr>(*expr.rhs))
                {
                    auto const atile = *std::get<WaveTilePtr>(*expr.lhs);
                    auto const btile = *std::get<WaveTilePtr>(*expr.rhs);
                    AssertFatal(!atile.sizes.empty(), "WaveTile in invalid state.");
                    AssertFatal(!btile.sizes.empty(), "WaveTile in invalid state.");
                    AssertRecoverable(atile.sizes[1] == btile.sizes[0],
                                      "MatrixMultiply WaveTile size mismatch.",
                                      ShowValue(atile.sizes[1]),
                                      ShowValue(btile.sizes[0]));

                    M   = atile.sizes[0];
                    N   = btile.sizes[1];
                    K   = atile.sizes[1];
                    B   = 1;
                    lhs = atile.vgpr;
                    rhs = btile.vgpr;
                }
                else
                {
                    std::vector<Generator<Instruction>> subExprs;
                    subExprs.push_back(call(lhs, expr.lhs));
                    subExprs.push_back(call(rhs, expr.rhs));

                    auto scheduler = Component::Get<Scheduling::Scheduler>(
                        Scheduling::SchedulerProcedure::Sequential, m_context);
                    co_yield (*scheduler)(subExprs);

                    M = expr.M;
                    N = expr.N;
                    K = expr.K;
                    B = expr.B;
                }

                AssertFatal(lhs->variableType() == rhs->variableType(),
                            "Input types must match ",
                            ShowValue(lhs->variableType()),
                            ShowValue(rhs->variableType()));

                AssertFatal(!lhs->variableType().isPointer(),
                            "Input must not be a pointer. ",
                            ShowValue(lhs->variableType()));

                // accumulator is either f32, f64, or i32
                DataType accType = expr.accumulationPrecision;

                if(dest == nullptr)
                {
                    auto const accRegCount = M * N * B / m_context->kernel()->wavefront_size();

                    dest = Register::Value::Placeholder(
                        m_context, Register::Type::Accumulator, accType, accRegCount);
                }

                co_yield Register::AllocateIfNeeded(dest);

                auto mm = Component::Get<rocRoller::InstructionGenerators::MatrixMultiply>(
                    m_context, accType, lhs->variableType().dataType);

                // TODO: Only zero if we are allocating a new register?
                co_yield mm->zero(dest);
                co_yield mm->mul(dest, lhs, rhs, M, N, K, B);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, WaveTilePtr const& expr)
            {
                Throw<FatalError>("WaveTile can only appear as an argument to MatrixMultiply.");
            }

            template <CUnary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                Register::ValuePtr argResult;

                co_yield call(argResult, expr.arg);

                if(dest == nullptr)
                {
                    dest = resultPlaceholder(resultType(expr));
                }

                auto arith = Arithmetic::Get(dest, argResult);
                co_yield callArithmeticUnary<Operation>(arith, dest, argResult);
            }

            template <CExpression Operation>
            requires(!CKernelExecuteTime<Operation>) Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                Throw<FatalError>("Operation ",
                                  ShowValue(expr),
                                  " not supported at kernel execute time: ",
                                  typeid(Operation).name());
            }

            Generator<Instruction> operator()(Register::ValuePtr&       dest,
                                              Register::ValuePtr const& expr)
            {
                if(dest == nullptr)
                {
                    dest = expr;
                }
                else
                {
                    co_yield m_context->copier()->copy(dest, expr);
                }
            }

            Generator<Instruction> operator()(Register::ValuePtr&              dest,
                                              AssemblyKernelArgumentPtr const& expr)
            {
                co_yield m_context->argLoader()->getValue(expr->name, dest);
            }

            Generator<Instruction> operator()(std::shared_ptr<CommandArgument> const& expr)
            {
                Throw<FatalError>("CommandArgument present in expression.", ShowValue(expr));
            }

            Generator<Instruction> operator()(Register::ValuePtr&         dest,
                                              CommandArgumentValue const& expr)
            {
                auto regLiteral = Register::Value::Literal(expr);
                co_yield (*this)(dest, regLiteral);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, DataFlowTag const& expr)
            {
                auto tagReg = m_context->registerTagManager()->getRegister(
                    expr.tag, expr.regType, expr.varType, 1);

                if(dest == nullptr)
                    dest = tagReg;
                else
                    co_yield m_context->copier()->copy(dest, tagReg);
            }

            Generator<Instruction> call(Register::ValuePtr& dest, Expression const& expr)
            {
                auto evalTimes = evaluationTimes(expr);
                if(evalTimes[EvaluationTime::Translate])
                {
                    auto result = Register::Value::Literal(evaluate(expr));

                    if(dest == nullptr)
                    {
                        dest = result;
                    }
                    else
                    {
                        co_yield m_context->copier()->copy(dest, result);
                    }
                }
                else
                {
                    RegisterValue theDest = dest;
                    co_yield std::visit(*this, theDest, expr);
                    dest = std::get<Register::ValuePtr>(theDest);
                }
            }

            Generator<Instruction> call(Register::ValuePtr& dest, ExpressionPtr const& expr)
            {
                std::string comment = getComment(expr);
                if(comment.length() > 0)
                {
                    co_yield Instruction::Comment(concatenate("BEGIN: ", comment));
                }

                co_yield call(dest, *expr);

                if(comment.length() > 0)
                {
                    co_yield Instruction::Comment(concatenate("END: ", comment));
                }
            }
        };

        Generator<Instruction>
            generate(Register::ValuePtr& dest, ExpressionPtr expr, ContextPtr context)
        {
            co_yield Instruction::Comment(toString(expr));
            CodeGeneratorVisitor v{context};
            co_yield v.call(dest, expr);
        }

#define DEFINE_UNARY_CALL(Op, call)                                                                \
    template <>                                                                                    \
    Generator<Instruction> CodeGeneratorVisitor::callArithmeticUnary<Op>(                          \
        ArithmeticPtr const& arith, Register::ValuePtr const& dest, Register::ValuePtr const& arg) \
    {                                                                                              \
        co_yield arith->call(dest, arg);                                                           \
    }
        DEFINE_UNARY_CALL(Negate, negate);

#undef DEFINE_UNARY_CALL

#define DEFINE_BINARY_CALL(Op, call)                                       \
    template <>                                                            \
    Generator<Instruction> CodeGeneratorVisitor::callArithmeticBinary<Op>( \
        ArithmeticPtr const&      arith,                                   \
        Register::ValuePtr const& dest,                                    \
        Register::ValuePtr const& lhs,                                     \
        Register::ValuePtr const& rhs)                                     \
    {                                                                      \
        co_yield arith->call(dest, lhs, rhs);                              \
    }
        DEFINE_BINARY_CALL(Subtract, sub);
        DEFINE_BINARY_CALL(Multiply, mul);
        DEFINE_BINARY_CALL(MultiplyHigh, mulHi);
        DEFINE_BINARY_CALL(Divide, div);
        DEFINE_BINARY_CALL(Modulo, mod);
        DEFINE_BINARY_CALL(ShiftL, shiftL);
        DEFINE_BINARY_CALL(ShiftR, shiftR);
        DEFINE_BINARY_CALL(BitwiseAnd, bitwiseAnd);
        DEFINE_BINARY_CALL(BitwiseXor, bitwiseXor);
        DEFINE_BINARY_CALL(GreaterThan, gt);
        DEFINE_BINARY_CALL(GreaterThanEqual, ge);
        DEFINE_BINARY_CALL(LessThan, lt);
        DEFINE_BINARY_CALL(LessThanEqual, le);
        DEFINE_BINARY_CALL(Equal, eq);
#undef DEFINE_BINARY_CALL

#define DEFINE_TERNARY_CALL(Op, call)                                       \
    template <>                                                             \
    Generator<Instruction> CodeGeneratorVisitor::callArithmeticTernary<Op>( \
        ArithmeticPtr const&      arith,                                    \
        Register::ValuePtr const& dest,                                     \
        Register::ValuePtr const& lhs,                                      \
        Register::ValuePtr const& r1hs,                                     \
        Register::ValuePtr const& r2hs)                                     \
    {                                                                       \
        co_yield arith->call(dest, lhs, r1hs, r2hs);                        \
    }
        DEFINE_TERNARY_CALL(FusedAddShift, addShiftL);
        DEFINE_TERNARY_CALL(FusedShiftAdd, shiftLAdd);
#undef DEFINE_TERNARY_CALL

    }
}
