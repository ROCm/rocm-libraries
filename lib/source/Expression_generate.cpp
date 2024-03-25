

#include <algorithm>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/CommonSubexpressionElim.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyAdd.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct ExpressionHasDFTagVisitor
        {
            template <CTernary Expr>
            bool operator()(Expr const& expr) const
            {
                return call(expr.lhs) || call(expr.r1hs) || call(expr.r2hs);
            }

            template <CBinary Expr>
            bool operator()(Expr const& expr) const
            {
                return call(expr.lhs) || call(expr.rhs);
            }

            template <CUnary Expr>
            bool operator()(Expr const& expr) const
            {
                return call(expr.arg);
            }

            template <typename T>
            bool operator()(T const& expr) const
            {
                return false;
            }

            bool operator()(DataFlowTag const& expr) const
            {
                return true;
            }

            bool call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            bool call(ExpressionPtr expr) const
            {
                if(!expr)
                    return false;

                return call(*expr);
            }
        };

        /**
         * @brief Returns true if an expression contains a DataFlowTag.
         *
         * @param expr
         * @return true
         * @return false
         */
        bool expressionHasDFTag(ExpressionPtr const& expr)
        {
            auto visitor = ExpressionHasDFTagVisitor();
            return visitor.call(expr);
        }

        bool expressionHasDFTag(Expression const& expr)
        {
            auto visitor = ExpressionHasDFTagVisitor();
            return visitor.call(expr);
        }

        struct CodeGeneratorVisitor
        {
            CodeGeneratorVisitor(ContextPtr& context)
                : m_context(context)
            {
            }

            using RegisterValue = std::variant<Register::ValuePtr>;

            Register::ValuePtr resultPlaceholder(ResultType const& resType,
                                                 bool              allowSpecial = true,
                                                 int               valueCount   = 1)
            {
                if(IsSpecial(resType.regType) && resType.varType == DataType::Bool)
                {
                    if(allowSpecial)
                        return m_context->getSCC();
                    else
                        return Register::Value::Placeholder(
                            m_context,
                            Register::Type::Scalar,
                            resType.varType,
                            valueCount,
                            Register::AllocationOptions::FullyContiguous());
                }
                return Register::Value::Placeholder(m_context,
                                                    resType.regType,
                                                    resType.varType,
                                                    valueCount,
                                                    Register::AllocationOptions::FullyContiguous());
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
                for(int i = 1; i < regs.size(); ++i)
                {
                    rtype = Register::PromoteType(rtype, regs[i]->regType());
                }
                return rtype;
            }

            VariableType promoteVariableTypes(std::vector<Register::ValuePtr> const& regs)
            {
                AssertFatal(!regs.empty());
                auto vtype = regs[0]->variableType();
                for(int i = 1; i < regs.size(); ++i)
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
                    if(IsSpecial(resultTypes[i].regType))
                        specials++;
                }

                // Can't use SCC for two temporary values at once.
                if(specials > 1)
                {
                    for(int i = 0; i < exprs.size() && specials > 1; i++)
                    {
                        if(IsSpecial(resultTypes[i].regType))
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
                        if(!IsSpecial(resultTypes[i].regType) || results[i] != nullptr)
                        {
                            schedulable.push_back(call(results[i], exprs[i]));
                            done[i] = true;
                        }
                    }

                    if(!schedulable.empty())
                    {
                        auto proc = Settings::getInstance()->get(Settings::Scheduler);
                        auto cost = Settings::getInstance()->get(Settings::SchedulerCost);
                        auto scheduler
                            = Component::GetNew<Scheduling::Scheduler>(proc, cost, m_context);

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
             * Generate code for arithemtic binary operation.
             *
             * We need to support, for example,
             * 1. scalar * scalar
             * 2. scalar * vector
             * 3. vector * vector (element-wise product)
             */
            template <typename T>
            requires(CBinary<T>&& CArithmetic<T>) Generator<Instruction> generateArithmeticBinary(
                Register::ValuePtr& dest,
                T const&            expr,
                Register::ValuePtr& lhs,
                Register::ValuePtr& rhs,
                ResultType&         resType)
            {

                auto const lhsInfo  = DataTypeInfo::Get(lhs->variableType());
                auto const rhsInfo  = DataTypeInfo::Get(rhs->variableType());
                auto const destInfo = DataTypeInfo::Get(resType.varType);

                int valueCount = resultValueCount(dest, {lhs, rhs});

                // TODO: Should this be pushed to arithmetic generators?
                // If any sources were AGPRs, copy to VGPRs first.
                if(valueCount > 1 && resType.regType == Register::Type::Accumulator)
                {
                    resType.regType = Register::Type::Vector;
                    co_yield m_context->copier()->ensureType(lhs, lhs, resType.regType);
                    co_yield m_context->copier()->ensureType(rhs, rhs, resType.regType);
                }

                if(dest == nullptr)
                {
                    dest = resultPlaceholder(resType, true, valueCount / destInfo.packing);
                }
                else
                {
                    // TODO Destination/result packing mismatch
                    //
                    // This was added to catch the case where:
                    // - a destination register was given
                    // - the packing of "lhs OP rhs" would be
                    //   different then the packing of the destination
                    //   register
                    //
                    // This should be possible.  See the
                    //
                    //   ReuseInputVGPRsAsOutputVGPRsInArithmeticF16SmallerPacking
                    //
                    // test in ExpressionTest.cpp.

                    auto resPack = DataTypeInfo::Get(resType.varType).packing;
                    auto dstPack = DataTypeInfo::Get(dest->variableType()).packing;
                    AssertFatal(dstPack <= resPack, "Destination/result packing mismatch.");
                }

                if(lhsInfo.packing != rhsInfo.packing)
                {
                    // If the packing values of the datatypes are different, we need to
                    // convert the more packed value into the less packed value type.
                    // We can then perform the operation.
                    int packingRatio = std::max(lhsInfo.packing, rhsInfo.packing)
                                       / std::min(lhsInfo.packing, rhsInfo.packing);

                    auto conversion = resultPlaceholder(resType, true, packingRatio);

                    for(size_t i = 0; i < valueCount; i += packingRatio)
                    {
                        Register::ValuePtr lhsVal, rhsVal;
                        if(lhsInfo.packing < rhsInfo.packing)
                        {
                            co_yield generateConvertOp(resType.varType.dataType,
                                                       conversion,
                                                       rhs->element({i / packingRatio}));
                        }
                        else
                        {
                            co_yield generateConvertOp(resType.varType.dataType,
                                                       conversion,
                                                       lhs->element({i / packingRatio}));
                        }

                        auto result = dest->element({i, i + packingRatio - 1});

                        for(size_t j = 0; j < packingRatio; j++)
                        {
                            if(lhsInfo.packing < rhsInfo.packing)
                            {
                                lhsVal = lhs->valueCount() == 1 ? lhs : lhs->element({i + j});
                                rhsVal = conversion->element({j});
                            }
                            else
                            {
                                lhsVal = conversion->element({j});
                                rhsVal = rhs->valueCount() == 1 ? rhs : rhs->element({i + j});
                            }

                            co_yield generateOp<T>(result->element({j}), lhsVal, rhsVal);
                        }
                    }
                }
                else
                {
                    AssertFatal(destInfo.isIntegral || lhs->variableType() == resType.varType
                                    || rhs->variableType() == resType.varType,
                                "Only one floating point argument can be converted");

                    auto conversion = resultPlaceholder(resType, true, 1);

                    for(size_t k = 0; k < dest->valueCount(); ++k)
                    {
                        auto lhsVal = lhs->regType() == Register::Type::Literal
                                              || IsSpecial(lhs->regType()) || lhs->valueCount() == 1
                                          ? lhs
                                          : lhs->element({k});
                        if(!destInfo.isIntegral && lhs->variableType() != resType.varType)
                        {
                            co_yield generateConvertOp(
                                resType.varType.dataType, conversion, lhsVal);
                            lhsVal = conversion;
                        }

                        auto rhsVal = rhs->regType() == Register::Type::Literal
                                              || IsSpecial(rhs->regType()) || rhs->valueCount() == 1
                                          ? rhs
                                          : rhs->element({k});
                        if(!destInfo.isIntegral && rhs->variableType() != resType.varType)
                        {
                            co_yield generateConvertOp(
                                resType.varType.dataType, conversion, rhsVal);
                            rhsVal = conversion;
                        }

                        co_yield generateOp<T>(dest->element({k}), lhsVal, rhsVal);
                    }
                }
            }

            template <typename T>
            requires(CKernelExecuteTime<T>&& CBinary<T>&& CArithmetic<T>) Generator<Instruction>
            operator()(Register::ValuePtr& dest, T const& expr)
            {
                auto strValue = toString(expr);
                co_yield Instruction::Comment(strValue);
                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.lhs, expr.rhs};

                AssertFatal(
                    !expressionHasDFTag(expr),
                    "expr is not expected to have a DataFlowTag : check DataFlowTagPropagation");

                auto resType = resultType(expr);
                AssertFatal(resType.varType != DataType::None,
                            "expr w/o DataFlowTag(s) doesn't have deferred datatype");

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);

                co_yield generateArithmeticBinary(dest, expr, results[0], results[1], resType);

                if(schedulerLocked)
                    co_yield Instruction::Unlock("Expression temporary in special register");
            }

            template <typename T>
            requires(CKernelExecuteTime<T>&& CBinary<T> && (CLogical<T> || CComparison<T>))
                Generator<Instruction>
            operator()(Register::ValuePtr& dest, T const& expr)
            {
                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.lhs, expr.rhs};

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);

                auto resType = resultType(expr);

                if(dest == nullptr)
                    dest = resultPlaceholder(resType);

                co_yield generateOp<T>(dest, results[0], results[1]);

                if(schedulerLocked)
                    co_yield Instruction::Unlock("Expression temporary in special register");
            }

            template <CTernary Operation>
            requires(
                !CTernaryMixed<Operation> && CKernelExecuteTime<Operation>) Generator<Instruction>
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

                auto varType = promoteVariableTypes(results);

                if(!dest)
                {
                    dest = resultPlaceholder({regType, varType}, true, valueCount);
                }

                for(size_t k = 0; k < valueCount; ++k)
                {
                    auto lhsVal  = results[0]->regType() == Register::Type::Literal
                                          || results[0]->valueCount() == 1
                                       ? results[0]
                                       : results[0]->element({k});
                    auto r1hsVal = results[1]->regType() == Register::Type::Literal
                                           || results[1]->valueCount() == 1
                                       ? results[1]
                                       : results[1]->element({k});
                    auto r2hsVal = results[2]->regType() == Register::Type::Literal
                                           || results[2]->valueCount() == 1
                                       ? results[2]
                                       : results[2]->element({k});
                    co_yield generateOp<Operation>(dest->element({k}), lhsVal, r1hsVal, r2hsVal);
                }

                if(schedulerLocked)
                    co_yield Instruction::Unlock("Expression temporary in special register");
            }

            template <CTernaryMixed Operation>
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

                if(!dest)
                {
                    auto varType = promoteVariableTypes(results);
                    dest         = resultPlaceholder({regType, varType}, true, valueCount);
                }

                //If dest, results have multiple elements, handled inside generateOp
                co_yield generateOp<Operation>(dest, results[0], results[1], results[2]);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, Conditional const& expr)
            {
                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.lhs, expr.r1hs, expr.r2hs};

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);
                auto cond = results[0];
                results.erase(results.begin());
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

                if(!dest)
                {
                    auto varType = promoteVariableTypes(results);
                    dest         = resultPlaceholder({regType, varType}, true, valueCount);
                }

                //If dest, results have multiple elements, handled inside generateOp
                co_yield generateOp<Conditional>(dest, cond, results[0], results[1]);
            }

            template <CUnary Operation>
            requires CKernelExecuteTime<Operation> Generator<Instruction>
            operator()(Register::ValuePtr& dest, Operation const& expr)
            {
                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.arg};

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);

                auto       destType = resultType(expr);
                auto const destInfo = DataTypeInfo::Get(destType.varType);
                auto const argInfo  = DataTypeInfo::Get(results[0]->variableType());

                int packingRatio = std::max(destInfo.packing, argInfo.packing)
                                   / std::min(destInfo.packing, argInfo.packing);

                if(dest == nullptr)
                {
                    dest = resultPlaceholder(
                        destType, true, results[0]->valueCount() / packingRatio);
                }

                if(GetGenerator<Operation>(dest, results[0])->isIdentity(results[0]))
                {
                    dest = results[0];
                }
                else if(dest->valueCount() == 1 && results[0]->valueCount() == 1)
                {
                    co_yield generateOp<Operation>(dest, results[0]);
                }
                else
                {
                    for(size_t i = 0; i < dest->valueCount(); i++)
                    {
                        Register::ValuePtr arg;
                        if(argInfo.packing < destInfo.packing)
                        {
                            arg = results[0]->element(
                                {i * packingRatio, i * packingRatio + packingRatio - 1});
                        }
                        else if(argInfo.packing > destInfo.packing)
                        {
                            Throw<FatalError>("Argument to unary expression cannot be a datatype "
                                              "with packing greater than the destination");
                        }
                        else
                        {
                            arg = results[0]->element({i});
                        }

                        co_yield generateOp<Operation>(dest->element({i}), arg);
                    }
                }

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
                        m_context,
                        Register::Type::Accumulator,
                        accType,
                        accRegCount,
                        Register::AllocationOptions::FullyContiguous());
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
                expr->assertCanUseAsOperand();

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

            Generator<Instruction> operator()(Register::ValuePtr&         dest,
                                              CommandArgumentValue const& expr)
            {
                auto regLiteral = Register::Value::Literal(expr);
                co_yield call(dest, regLiteral);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, DataFlowTag const& expr)
            {
                Throw<FatalError>("DataFlowTag present in the expression.", ShowValue(expr));
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

            Generator<Instruction> call(Register::ValuePtr& dest, ExpressionPtr expr)
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

        private:
            ContextPtr m_context;
        };

        Generator<Instruction> generateFromTree(Register::ValuePtr&   dest,
                                                ExpressionTree&       tree,
                                                CodeGeneratorVisitor& visitor,
                                                ContextPtr            context)
        {
            tree.back().reg = dest;

            std::set<int> metDeps;
            while(metDeps.size() < tree.size())
            {
                std::set<int> tmpMetDeps;

                auto proc      = Settings::getInstance()->get(Settings::Scheduler);
                auto cost      = Settings::getInstance()->get(Settings::SchedulerCost);
                auto scheduler = Component::GetNew<Scheduling::Scheduler>(proc, cost, context);
                std::vector<Generator<Instruction>> schedulable;
                for(int i = 0; i < tree.size(); i++)
                {
                    if(!metDeps.contains(i)
                       && std::includes(metDeps.begin(),
                                        metDeps.end(),
                                        tree.at(i).deps.begin(),
                                        tree.at(i).deps.end()))
                    {
                        if(!(tree.at(i).reg
                             && tree.at(i).reg->regType() == Register::Type::Literal))
                        {
                            schedulable.push_back(visitor.call(tree.at(i).reg, tree.at(i).expr));
                        }
                        tmpMetDeps.insert(i);
                    }
                }

                co_yield (*scheduler)(schedulable);

                for(int i = 0; i < tree.size() - 1; i++)
                {
                    if(tmpMetDeps.contains(i))
                    {
                        tree.at(i).expr = nullptr;
                        tree.at(i).reg  = nullptr;
                    }
                }

                AssertFatal(!tmpMetDeps.empty(), ShowValue(tree.size()), ShowValue(metDeps.size()));
                metDeps.insert(tmpMetDeps.begin(), tmpMetDeps.end());
            }
            dest = tree.back().reg;
        }

        Generator<Instruction>
            generate(Register::ValuePtr& dest, ExpressionPtr expr, ContextPtr context)
        {
            std::string destStr = "nullptr";
            if(dest)
                destStr = dest->toString();
            co_yield Instruction::Comment("Generate " + toString(expr) + " into " + destStr);

            {
                auto fast = FastArithmetic(context);

                // There may be pre-calculated values based on other kernel arguments.
                expr = fast(expr);
                // Resolve DataFlowTags and evaluate exprs with translate time source operands.
                expr = dataFlowTagPropagation(expr, context);
                // There may be additional optimizations after resolving DataFlowTags and kernel arguments.
                expr = fast(expr);
            }

            CodeGeneratorVisitor v{context};

            // Top-level evaluations can't go into special-purpose registers
            // unless explicitly asked for.
            if(dest == nullptr)
            {
                auto resType = resultType(expr);
                if(IsSpecial(resType.regType))
                    dest = v.resultPlaceholder(resType, false);
            }

            ExpressionTree tree;
            tree = consolidateSubExpressions(expr, context);

            if(tree.size() < 2
               || (dest && dest->variableType() == DataType::Bool
                   && getConsolidationCount(tree) == 0))
            {
                // Don't use CSE in this case

                tree.resize(0);
                co_yield v.call(dest, expr);
            }
            else
            {
                co_yield generateFromTree(dest, tree, v, context);
            }
        }
    }
}
