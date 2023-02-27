

#include <algorithm>

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

            Register::ValuePtr resultPlaceholder(ResultType const& resType,
                                                 bool              allowSpecial = true,
                                                 int               valueCount   = 1)
            {
                if(resType.regType == Register::Type::Special && resType.varType == DataType::Bool)
                {
                    if(allowSpecial)
                        return m_context->getSCC();
                    else
                        return Register::Value::Placeholder(
                            m_context, Register::Type::Scalar, resType.varType, valueCount);
                }
                else if(resType.regType == Register::Type::Scalar
                        && resType.varType == DataType::Bool32)
                {
                    return Register::Value::WavefrontPlaceholder(m_context);
                }

                return Register::Value::Placeholder(
                    m_context, resType.regType, resType.varType, valueCount);
            }

            int resultValueCount(Register::ValuePtr const&              dest,
                                 std::vector<Register::ValuePtr> const& operands)

            {
                if(dest)
                {
                    return dest->valueCount();
                }

                std::vector<int> count;
                std::transform(
                    operands.cbegin(), operands.cend(), std::back_inserter(count), [](auto x) {
                        return x->valueCount() * DataTypeInfo::Get(x->variableType()).packing;
                    });
                return *std::max_element(count.cbegin(), count.cend());
            }

            Register::Type promoteRegisterTypes(std::vector<Register::ValuePtr> const& regs)
            {
                AssertFatal(!regs.empty());
                auto rtype = regs[0]->regType();
                for(int i = 0; i < regs.size(); ++i)
                {
                    rtype = Register::PromoteType(rtype, regs[i]->regType());
                }
                return rtype;
            }

            VariableType promoteVarialbeTypes(std::vector<Register::ValuePtr> const& regs)
            {
                AssertFatal(!regs.empty());
                auto vtype = regs[0]->variableType();
                for(int i = 0; i < regs.size(); ++i)
                {
                    vtype = VariableType::Promote(vtype, regs[i]->variableType());
                }
                return vtype;
            }

            /**
             * Evaluates each expression in `exprs`, storing the results in respective indices of
             * `results`.
             *
             * Up to one result may be stored in `scc`. If this is the case, the scheduler will be locked,
             * and `schedulerLocked` will be set to `true`.  It's the caller's responsibility to unlock
             * the scheduler in this case, once the value has been consumed.
             */
            Generator<Instruction> prepareSourceOperands(std::vector<Register::ValuePtr>& results,
                                                         bool&                      schedulerLocked,
                                                         std::vector<ExpressionPtr> exprs)
            {
                std::vector<char>       done(exprs.size(), false);
                std::vector<ResultType> resultTypes(exprs.size());

                schedulerLocked = false;

                results = std::vector<Register::ValuePtr>(exprs.size(), nullptr);

                int specials = 0;
                for(int i = 0; i < exprs.size(); i++)
                {
                    resultTypes[i] = resultType(exprs[i]);
                    if(resultTypes[i].regType == Register::Type::Special)
                        specials++;
                }

                // Can't use SCC for two temporary values at once.
                if(specials > 1)
                {
                    for(int i = 0; i < exprs.size() && specials > 1; i++)
                    {
                        if(resultTypes[i].regType == Register::Type::Special)
                        {
                            results[i] = resultPlaceholder(resultTypes[i], false);
                            specials--;
                        }
                    }
                }

                // First, schedule any sub-expressions that will go into general-purpose registers.
                {
                    std::vector<Generator<Instruction>> schedulable;
                    for(int i = 0; i < exprs.size(); i++)
                    {
                        if(resultTypes[i].regType != Register::Type::Special
                           || results[i] != nullptr)
                        {
                            schedulable.push_back(call(results[i], exprs[i]));
                            done[i] = true;
                        }
                    }

                    if(!schedulable.empty())
                    {
                        auto proc      = Settings::getInstance()->get(Settings::Scheduler);
                        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                            proc, Scheduling::CostProcedure::MinNops, m_context);

                        co_yield (*scheduler)(schedulable);
                    }
                }

                // Then there might be 1 remaining expression that will go into SCC.
                if(specials > 0)
                {
                    int unscheduled = 0;
                    for(int i = 0; i < exprs.size(); i++)
                    {
                        if(!done[i])
                        {
                            unscheduled++;
                            schedulerLocked = true;
                            co_yield Instruction::Lock(Scheduling::Dependency::SCC,
                                                       "Expression temporary in special register");
                            co_yield call(results[i], exprs[i]);
                        }
                    }

                    AssertFatal(unscheduled == specials && specials <= 1,
                                "Only one special purpose register should have remained.",
                                ShowValue(unscheduled),
                                ShowValue(specials));
                }
            }

            /*
             * Generate code for a "deferred" binary operation.
             *
             * These are typically from element operations in Assign nodes where the result variable
             * type is `DataType::None`.  That is, we have deferred determining the output type (and
             * size) to code-gen time.
             *
             * We need to support, for example,
             * 1. scalar * scalar
             * 2. scalar * vector
             * 3. vector * vector (element-wise product)
             */
            template <CBinary Operation>
            Generator<Instruction> generateDeferredBinary(Register::ValuePtr& dest,
                                                          Operation const&    expr,
                                                          Register::ValuePtr  lhs,
                                                          Register::ValuePtr  rhs)
            {
                auto const lhsInfo = DataTypeInfo::Get(lhs->variableType());
                auto const rhsInfo = DataTypeInfo::Get(rhs->variableType());

                auto regType = promoteRegisterTypes({lhs, rhs});
                auto varType = promoteVarialbeTypes({lhs, rhs});

                // TODO: Delete once FastDivision uses only libdivide.
                if constexpr(std::same_as<MultiplyHigh, Operation>)
                    varType = DataType::Int32;

                int valueCount = resultValueCount(dest, {lhs, rhs});

                // TODO: Should this be pushed to arithmetic generators?
                // If any sources were AGPRs, copy to VGPRs first.
                if(valueCount > 1 && regType == Register::Type::Accumulator)
                {
                    regType = Register::Type::Vector;
                    co_yield m_context->copier()->ensureType(lhs, lhs, regType);
                    co_yield m_context->copier()->ensureType(rhs, rhs, regType);
                }

                if(!dest)
                {
                    dest = resultPlaceholder({regType, varType}, true, valueCount);
                    co_yield dest->allocate();
                }

                if(lhsInfo.packing != rhsInfo.packing)
                {
                    // If the packing values of the datatypes are different, we need to
                    // convert the more packed value into the less packed value type.
                    // We can then perform the operation.
                    int packingRatio = std::max(lhsInfo.packing, rhsInfo.packing)
                                       / std::min(lhsInfo.packing, rhsInfo.packing);

                    for(size_t i = 0; i < valueCount; i += packingRatio)
                    {
                        auto result = dest->element({i, i + packingRatio - 1});

                        Register::ValuePtr lhsVal, rhsVal;
                        if(lhsInfo.packing < rhsInfo.packing)
                        {
                            co_yield generateConvertOp(
                                varType.dataType, result, rhs->element({i / packingRatio}));
                        }
                        else
                        {
                            co_yield generateConvertOp(
                                varType.dataType, result, lhs->element({i / packingRatio}));
                        }

                        for(size_t j = 0; j < packingRatio; j++)
                        {
                            if(lhsInfo.packing < rhsInfo.packing)
                            {
                                lhsVal = lhs->valueCount() == 1 ? lhs : lhs->element({i + j});
                                rhsVal = result->element({j});
                            }
                            else
                            {
                                lhsVal = result->element({j});
                                rhsVal = rhs->valueCount() == 1 ? rhs : rhs->element({i + j});
                            }

                            co_yield generateOp<Operation>(result->element({j}), lhsVal, rhsVal);
                        }
                    }
                }
                else
                {
                    for(size_t k = 0; k < valueCount / lhsInfo.packing; ++k)
                    {
                        auto lhsVal = lhs->valueCount() == 1 ? lhs : lhs->element({k});
                        auto rhsVal = rhs->valueCount() == 1 ? rhs : rhs->element({k});
                        co_yield generateOp<Operation>(dest->element({k}), lhsVal, rhsVal);
                    }
                }
            }

            template <CBinary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.lhs, expr.rhs};

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);

                auto resType  = resultType(expr);
                auto deferred = resType.varType == DataType::None;

                if(deferred)
                {
                    co_yield generateDeferredBinary(dest, expr, results[0], results[1]);
                }
                else
                {
                    if(dest == nullptr)
                        dest = resultPlaceholder(resType);

                    co_yield generateOp<Operation>(dest, results[0], results[1]);
                }

                if(schedulerLocked)
                    co_yield Instruction::Unlock("Expression temporary in special register");
            }

            template <CTernary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)

            {
                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.lhs, expr.r1hs, expr.r2hs};

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);
                auto regType    = promoteRegisterTypes(results);
                auto valueCount = resultValueCount(dest, results);

                if(valueCount > 1 && regType == Register::Type::Accumulator)
                {
                    regType = Register::Type::Vector;
                    for(int i = 0; i < results.size(); ++i)
                    {
                        co_yield m_context->copier()->ensureType(results[i], results[i], regType);
                    }
                }

                auto varType = promoteVarialbeTypes(results);

                if(!dest)
                {
                    dest = resultPlaceholder({regType, varType}, true, valueCount);
                    co_yield dest->allocate();
                }

                for(size_t k = 0; k < valueCount; ++k)
                {
                    auto lhsVal
                        = results[0]->valueCount() == 1 ? results[0] : results[0]->element({k});
                    auto r1hsVal
                        = results[1]->valueCount() == 1 ? results[1] : results[1]->element({k});
                    auto r2hsVal
                        = results[2]->valueCount() == 1 ? results[2] : results[2]->element({k});
                    co_yield generateOp<Operation>(dest->element({k}), lhsVal, r1hsVal, r2hsVal);
                }

                if(schedulerLocked)
                    co_yield Instruction::Unlock("Expression temporary in special register");
            }

            template <CUnary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.arg};

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);

                if(dest == nullptr)
                    dest = resultPlaceholder(resultType(expr));

                co_yield generateOp<Operation>(dest, results[0]);

                if(schedulerLocked)
                    co_yield Instruction::Unlock("Expression temporary in special register");
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
                AssertFatal(atile.sizes[1] == btile.sizes[0],
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

                auto mm = Component::Get<rocRoller::InstructionGenerators::MatrixMultiply>(
                    m_context, accType, lhs->variableType().dataType);

                r2hs = std::get<Register::ValuePtr>(*expr.r2hs);
                co_yield mm->mul(dest, lhs, r1hs, r2hs, M, N, K, B);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, WaveTilePtr const& expr)
            {
                Throw<FatalError>("WaveTile can only appear as an argument to MatrixMultiply.");
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
                co_yield call(dest, regLiteral);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, DataFlowTag const& expr)
            {
                if(m_context->registerTagManager()->hasExpression(expr.tag))
                {
                    auto [tagExpr, tagDT]
                        = m_context->registerTagManager()->getExpression(expr.tag);
                    co_yield call(dest, tagExpr);
                }
                else
                {
                    auto tagReg = m_context->registerTagManager()->getRegister(
                        expr.tag, expr.regType, expr.varType, 1);

                    if(dest == nullptr)
                        dest = tagReg;
                    else
                        co_yield m_context->copier()->copy(dest, tagReg);
                }
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

            // Top-level evaluations can't go into special-purpose registers
            // unless explicitly asked for.
            if(dest == nullptr)
            {
                auto resType = resultType(expr);
                if(resType.regType == Register::Type::Special)
                    dest = v.resultPlaceholder(resType, false);
            }

            co_yield v.call(dest, expr);
        }
    }
}
