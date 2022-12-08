

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

            Register::ValuePtr resultPlaceholder(ResultType const& resType)
            {
                if(resType.first == Register::Type::Special && resType.second == DataType::Bool)
                {
                    return Register::Value::Placeholder(
                        m_context, Register::Type::Scalar, resType.second, 1);
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

                std::vector<Generator<Instruction>> subExprs;
                subExprs.push_back(call(lhsResult, expr.lhs));
                subExprs.push_back(call(rhsResult, expr.rhs));

                auto proc      = Settings::getInstance()->get(Settings::Scheduler);
                auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                    proc, Scheduling::CostProcedure::MinNops, m_context);
                co_yield (*scheduler)(subExprs);

                if(dest == nullptr)
                {
                    dest = resultPlaceholder(resultType(expr));
                }

                co_yield generateOp<Operation>(dest, lhsResult, rhsResult);
            }

            template <CTernary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                Register::ValuePtr lhsResult, r1hsResult, r2hsResult;

                std::vector<Generator<Instruction>> subExprs;
                subExprs.push_back(call(lhsResult, expr.lhs));
                subExprs.push_back(call(r1hsResult, expr.r1hs));
                subExprs.push_back(call(r2hsResult, expr.r2hs));

                auto proc      = Settings::getInstance()->get(Settings::Scheduler);
                auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                    proc, Scheduling::CostProcedure::MinNops, m_context);
                co_yield (*scheduler)(subExprs);

                co_yield generateOp<Operation>(dest, lhsResult, r1hsResult, r2hsResult);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, MatrixMultiply expr)
            {
                Register::ValuePtr lhs, r1hs, r2hs;
                int                M, N, K, B;

                AssertFatal(std::holds_alternative<WaveTilePtr>(*expr.lhs)
                                && std::holds_alternative<WaveTilePtr>(*expr.r1hs),
                            "Expression MatrixMultiply requires WaveTiles");

                auto const atile = *std::get<WaveTilePtr>(*expr.lhs);
                auto const btile = *std::get<WaveTilePtr>(*expr.r1hs);
                AssertFatal(!atile.sizes.empty(), "WaveTile in invalid state.");
                AssertFatal(!btile.sizes.empty(), "WaveTile in invalid state.");
                AssertRecoverable(atile.sizes[1] == btile.sizes[0],
                                  "MatrixMultiply WaveTile size mismatch.",
                                  ShowValue(atile.sizes[1]),
                                  ShowValue(btile.sizes[0]));

                M    = atile.sizes[0];
                N    = btile.sizes[1];
                K    = atile.sizes[1];
                B    = 1;
                lhs  = atile.vgpr;
                r1hs = btile.vgpr;

                AssertFatal(lhs->variableType() == r1hs->variableType(),
                            "Input types must match ",
                            ShowValue(lhs->variableType()),
                            ShowValue(r1hs->variableType()));

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

                r2hs = std::get<Register::ValuePtr>(*expr.r2hs);
                co_yield Register::AllocateIfNeeded(r2hs);
                co_yield mm->mul(dest, lhs, r1hs, r2hs, M, N, K, B);
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

                co_yield generateOp<Operation>(dest, argResult);
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
    }
}
