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

#pragma once

#include <map>
#include <memory>

#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    inline RegisterTagManager::RegisterTagManager(ContextPtr context)
        : m_context(context)
    {
    }

    inline RegisterTagManager::~RegisterTagManager() = default;

    inline std::pair<Expression::ExpressionPtr, RegisterExpressionAttributes>
        RegisterTagManager::getExpression(int tag) const
    {
        AssertFatal(hasExpression(tag), ShowValue(tag));
        return m_expressions.at(tag);
    }

    inline Register::ValuePtr RegisterTagManager::getRegister(int tag)
    {
        AssertFatal(hasRegister(tag), ShowValue(tag));
        return m_registers.at(tag);
    }

    inline Register::ValuePtr
        RegisterTagManager::getRegister(int                         tag,
                                        Register::Type              regType,
                                        VariableType                varType,
                                        size_t                      valueCount,
                                        Register::AllocationOptions allocOptions)
    {
        if(hasRegister(tag))
        {
            auto reg = m_registers.at(tag);
            if(varType != DataType::None)
            {
                AssertFatal(reg->variableType() == varType,
                            ShowValue(varType),
                            ShowValue(reg->variableType()),
                            ShowValue(tag));
                AssertFatal(reg->valueCount() == valueCount,
                            ShowValue(valueCount),
                            ShowValue(reg->valueCount()));
                AssertFatal(
                    reg->regType() == regType, ShowValue(regType), ShowValue(reg->regType()));
            }
            return reg;
        }
        auto tmpl = Register::Value::Placeholder(
            m_context.lock(), regType, varType, valueCount, allocOptions);
        return getRegister(tag, tmpl);
    }

    inline Register::ValuePtr RegisterTagManager::getRegister(int tag, Register::ValuePtr tmpl)
    {
        AssertFatal(!hasExpression(tag), "Tag already associated with an expression");
        if(hasRegister(tag))
        {
            auto reg = m_registers.at(tag);
            if(tmpl->variableType() != DataType::None)
            {
                AssertFatal(reg->variableType() == tmpl->variableType(),
                            ShowValue(tmpl->variableType()),
                            ShowValue(reg->variableType()));
                AssertFatal(reg->valueCount() == tmpl->valueCount(),
                            ShowValue(tmpl->valueCount()),
                            ShowValue(reg->valueCount()));
                AssertFatal(reg->regType() == tmpl->regType(),
                            ShowValue(tmpl->regType()),
                            ShowValue(reg->regType()));
            }
            return reg;
        }
        auto r = tmpl->placeholder();
        m_registers.emplace(tag, r);
        return m_registers.at(tag);
    }

    inline void RegisterTagManager::addRegister(int tag, Register::ValuePtr value)
    {
        AssertFatal(!hasExpression(tag), "Tag already associated with an expression");
        m_registers.insert(std::pair<int, Register::ValuePtr>(tag, value));
    }

    inline void RegisterTagManager::addExpression(int                          tag,
                                                  Expression::ExpressionPtr    value,
                                                  RegisterExpressionAttributes attrs)
    {
        AssertFatal(!hasRegister(tag), "Tag ", tag, " already associated with a register");
        m_expressions.insert(
            std::pair<int, std::pair<Expression::ExpressionPtr, RegisterExpressionAttributes>>(
                tag, {value, attrs}));
    }

    inline void RegisterTagManager::deleteTag(int tag)
    {
        auto inst = Instruction::Comment(concatenate("Deleting tag ", tag));
        m_context.lock()->schedule(inst);
        m_registers.erase(tag);
        m_expressions.erase(tag);
    }

    inline bool RegisterTagManager::hasRegister(int tag) const
    {
        return m_registers.count(tag) > 0;
    }

    inline bool RegisterTagManager::hasExpression(int tag) const
    {
        return m_expressions.count(tag) > 0;
    }
}
