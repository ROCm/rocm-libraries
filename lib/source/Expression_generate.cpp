/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <algorithm>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/CommonSubexpressionElim.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyAdd.hpp>
#include <rocRoller/CodeGen/Arithmetic/ScaledMatrixMultiply.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/RTTI.hpp>
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
             * Generate code for comparison binary operation.
             *
             * We need to support, for example,
             * 1. scalar <=> scalar
             * 2. vector <=> vector
             */
            template <typename T>
            requires(CKernelExecuteTime<T>&& CBinary<T> && (CLogical<T> || CComparison<T>))
                Generator<Instruction> generateComparisonOrLogicalBinary(Register::ValuePtr& dest,
                                                                         T const&            expr,
                                                                         Register::ValuePtr& lhs,
                                                                         Register::ValuePtr& rhs,
                                                                         ResultType& resType)
            {
                auto const lhsInfo = DataTypeInfo::Get(lhs->variableType());
                auto const rhsInfo = DataTypeInfo::Get(rhs->variableType());

                int valueCount = resultValueCount(dest, {lhs, rhs});

                if(!dest)
                {
                    dest = resultPlaceholder(resType, true, valueCount);
                }

                for(size_t k = 0; k < dest->valueCount(); ++k)
                {
                    // TODD: Consolidate with other similar code
                    // that only calls `->element` if conditions are met
                    auto lhsVal = lhs->regType() == Register::Type::Literal
                                          || IsSpecial(lhs->regType()) || lhs->valueCount() == 1
                                      ? lhs
                                      : lhs->element({k});

                    auto rhsVal = rhs->regType() == Register::Type::Literal
                                          || IsSpecial(rhs->regType()) || rhs->valueCount() == 1
                                      ? rhs
                                      : rhs->element({k});

                    co_yield generateOp<T>(dest->element({k}), lhsVal, rhsVal);
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
                co_yield Instruction::Comment(toString(expr));
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

            /*
             * Generate code for Stochastic Rounding Conversion.
             *
             * SR conversion is a binaray expression as it has a value (lhs) to be converted
             * and a seed (rhs) for stochastic rounding.
             *
             */
            template <typename T>
            requires(CKernelExecuteTime<T>&& CBinary<T>&& CConversion<T>) Generator<Instruction>
            operator()(Register::ValuePtr& dest, T const& expr)
            {
                // Currently GPU only supports SR conversion of F32 to FP8/BF8
                static_assert(
                    std::is_same_v<
                        T,
                        SRConvert<DataType::FP8>> || std::is_same_v<T, SRConvert<DataType::BF8>>);

                bool                            schedulerLocked = false;
                std::vector<Register::ValuePtr> results;
                std::vector<ExpressionPtr>      subExprs{expr.lhs, expr.rhs};

                co_yield prepareSourceOperands(results, schedulerLocked, subExprs);

                // Convert one value at a time
                dest = resultPlaceholder(resultType(expr), true, results[0]->valueCount());

                // Assume the seed is a single scalar value. Consider allowing a
                // vector of seeds?
                AssertFatal(results[1]->valueCount() == 1, "Seed should be a scalar value");

                for(size_t i = 0; i < dest->valueCount(); i++)
                {
                    // TODO: the seed should incoporate workitem ID
                    co_yield generateOp<T>(
                        dest->element({i}), results[0]->element({i}), results[1]->element({0}));
                }
            }

            template <typename T>
            requires(CKernelExecuteTime<T>&& CBinary<T> && (CLogical<T> || CComparison<T>))
                Generator<Instruction>
            operator()(Register::ValuePtr& dest, T const& expr)
            {
                co_yield Instruction::Comment(toString(expr));
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

                co_yield generateComparisonOrLogicalBinary(
                    dest, expr, results[0], results[1], resType);

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

                if(dest == nullptr)
                {
                    auto varType = promoteVariableTypes(results);
                    dest         = resultPlaceholder({regType, varType}, true, valueCount);
                }

                for(size_t k = 0; k < valueCount; ++k)
                {
                    auto lhs    = results[0];
                    auto rhs    = results[1];
                    auto lhsVal = lhs->regType() == Register::Type::Literal
                                          || IsSpecial(lhs->regType()) || lhs->valueCount() == 1
                                      ? lhs
                                      : lhs->element({k});

                    auto rhsVal = rhs->regType() == Register::Type::Literal
                                          || IsSpecial(rhs->regType()) || rhs->valueCount() == 1
                                      ? rhs
                                      : rhs->element({k});
                    co_yield generateOp<Conditional>(
                        dest->element({k}), cond->element({k}), lhsVal, rhsVal, expr);
                }
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

                // arg's packing might be larger than dest's packing.
                // For example, this could be a conversion op that
                // converts Halfx2 (packing=2) into Float (packing=1).
                // If this occurs, that means we should `unpack` the
                // arg into dest
                bool const isUnpacking = argInfo.packing > destInfo.packing;

                if(dest == nullptr)
                {
                    if(destType.regType == rocRoller::Register::Type::Accumulator)
                    {
                        // If the expr is a matrix multiply (mfma), the register type might
                        // be ACCVGPR. But Unary operation cannot work on ACCVGPR,
                        // and we have to allocate Vector instead.
                        destType.regType = rocRoller::Register::Type::Vector;
                    }

                    if(isUnpacking)
                    {
                        // unpacking args into (multiple registers) dest
                        dest = resultPlaceholder(
                            destType, true, results[0]->valueCount() * packingRatio);
                    }
                    else
                    {
                        dest = resultPlaceholder(
                            destType, true, results[0]->valueCount() / packingRatio);
                    }
                }

                dest->setName(dest->name() + results[0]->name());

                if(dest->valueCount() == 1 && results[0]->valueCount() == 1)
                {
                    co_yield generateOp<Operation>(dest, results[0], expr);
                }
                else
                {
                    if(isUnpacking)
                    {
                        for(size_t i = 0; i < results[0]->valueCount(); i++)
                        {
                            Register::ValuePtr destRegs;
                            const size_t       index = i * packingRatio;
                            if(packingRatio == 2) // e.g., unpack a Halfx2 to two Float
                                destRegs = dest->element({index, index + 1});
                            else if(packingRatio == 4) // e.g., unpack a FP8x4 to four Float
                                destRegs = dest->element({index, index + 1, index + 2, index + 3});
                            else
                                Throw<FatalError>("Packing ratio not supported yet.");

                            Register::ValuePtr arg = results[0]->element({i});
                            co_yield generateOp<Operation>(destRegs, arg, expr);
                        }
                    }
                    else
                    {
                        for(size_t i = 0; i < dest->valueCount(); i++)
                        {
                            Register::ValuePtr arg;
                            if(argInfo.packing < destInfo.packing)
                            {
                                if(packingRatio == 2)
                                    arg = results[0]->element(
                                        {i * packingRatio, i * packingRatio + 1});
                                else if(packingRatio == 4)
                                    arg = results[0]->element({i * packingRatio,
                                                               i * packingRatio + 1,
                                                               i * packingRatio + 2,
                                                               i * packingRatio + 3});
                                else
                                    Throw<FatalError>("Packing ratio not supported yet.");
                            }
                            else
                            {
                                arg = results[0]->element({i});
                            }

                            co_yield generateOp<Operation>(dest->element({i}), arg, expr);
                        }
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

                auto mm
                    = Component::Get<rocRoller::InstructionGenerators::MatrixMultiply>(m_context);

                r2hs = std::get<Register::ValuePtr>(*expr.r2hs);
                co_yield mm->mul(dest, lhs, r1hs, r2hs, M, N, K, B);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, BitFieldExtract const& expr)
            {
                AssertFatal(std::holds_alternative<Register::ValuePtr>(*expr.arg));
                auto arg = std::get<Register::ValuePtr>(*expr.arg);
                co_yield generateOp<BitFieldExtract>(dest, arg, expr);
            }

            Generator<Instruction> operator()(Register::ValuePtr& dest, ScaledMatrixMultiply expr)
            {

                AssertFatal(std::holds_alternative<WaveTilePtr>(*expr.matA)
                                && std::holds_alternative<WaveTilePtr>(*expr.matB),
                            "Expression MatrixMultiply requires WaveTiles");

                auto const atile = *std::get<WaveTilePtr>(*expr.matA);
                auto const btile = *std::get<WaveTilePtr>(*expr.matB);
                AssertFatal(!atile.sizes.empty(), "WaveTile in invalid state.");
                AssertFatal(!btile.sizes.empty(), "WaveTile in invalid state.");
                AssertFatal(atile.sizes[1] == btile.sizes[0],
                            "MatrixMultiply WaveTile size mismatch.",
                            ShowValue(atile.sizes[1]),
                            ShowValue(btile.sizes[0]));

                auto M  = atile.sizes[0];
                auto N  = btile.sizes[1];
                auto K  = atile.sizes[1];
                auto rA = atile.vgpr;
                auto rB = btile.vgpr;

                auto getRegister = rocRoller::overloaded{
                    [&](WaveTilePtr const& tile) -> Register::ValuePtr { return tile->vgpr; },
                    [&](Register::ValuePtr const& reg) -> Register::ValuePtr { return reg; },
                    [&](auto const& other) -> Register::ValuePtr {
                        Throw<FatalError>("Invalid scale expression type: ",
                                          typeName<decltype(other)>());
                        return nullptr;
                    }};

                auto rScaleA = std::visit(getRegister, *expr.scaleA);
                auto rScaleB = std::visit(getRegister, *expr.scaleB);

                AssertFatal(!rA->variableType().isPointer(),
                            "Input must not be a pointer. ",
                            ShowValue(rA->variableType()));

                // accumulator is either f32, f64, or i32
                DataType accType = expr.accumulationPrecision;

                if(dest == nullptr)
                {
                    auto const accRegCount = M * N / m_context->kernel()->wavefront_size();

                    dest = Register::Value::Placeholder(
                        m_context,
                        Register::Type::Accumulator,
                        accType,
                        accRegCount,
                        Register::AllocationOptions::FullyContiguous());
                }

                auto smm = Component::Get<rocRoller::InstructionGenerators::ScaledMatrixMultiply>(
                    m_context, accType, rA->variableType().dataType);

                auto rC = std::get<Register::ValuePtr>(*expr.matC);

                co_yield smm->mul(dest, rA, rB, rC, rScaleA, rScaleB, M, N, K);
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
                                  typeName<Operation>());
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
                    co_yield m_context->copier()->copy(dest, expr, "reg expression");
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
                        co_yield m_context->copier()->copy(dest, result, "call()");
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
                std::string comment = getComment(expr, false);
                if(comment.length() > 0)
                {
                    co_yield Instruction::Comment(concatenate("BEGIN: ", comment));
                }

                co_yield call(dest, *expr);

                if(comment.length() > 0)
                {
                    co_yield Instruction::Comment(concatenate("END: ", comment));

                    if(dest->name().empty())
                        dest->setName(comment);
                }
                else if(dest->name().empty())
                {
                    comment = getComment(expr, true);
                    if(!comment.empty())
                        dest->setName(comment);
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

            // Replace RandomNumber expression with expressions that implement the PRNG algorithm
            // if PRNG instruction is unavailable
            expr = lowerPRNG(expr, context);

            {
                auto fast = FastArithmetic(context);

                // There may be pre-calculated values based on other kernel arguments.
                expr = fast(expr);
                // Resolve DataFlowTags and evaluate exprs with translate time source operands.
                expr = dataFlowTagPropagation(expr, context);
                // There may be additional optimizations after resolving DataFlowTags and kernel arguments.
                expr = fast(expr);
            }

            expr = lowerBitfieldValues(expr);

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
