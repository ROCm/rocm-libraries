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

#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

#include <rocRoller/InstructionValues/Register.hpp>
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
        auto merge = getIndex(tag);
        if(merge)
        {
            auto [dst, index] = *merge;
            AssertFatal(hasRegister(dst), ShowValue(dst));

            auto target = m_registers.at(dst);
            return target->subset({index});
        }

        AssertFatal(hasRegister(tag), ShowValue(tag));
        AssertFatal(!isBorrowed(tag), ShowValue(tag), " has been borrowed.");

        return m_registers.at(tag);
    }

    inline Register::ValuePtr
        RegisterTagManager::getRegister(int                         tag,
                                        Register::Type              regType,
                                        VariableType                varType,
                                        size_t                      valueCount,
                                        Register::AllocationOptions allocOptions)
    {
        auto tmpl = Register::Value::Placeholder(
            m_context.lock(), regType, varType, valueCount, allocOptions);
        AssertFatal(
            tmpl->valueCount() == valueCount, ShowValue(tmpl->toString()), ShowValue(valueCount));
        return getRegister(tag, tmpl);
    }

    inline Register::ValuePtr RegisterTagManager::getRegister(int tag, Register::ValuePtr tmpl)
    {
        AssertFatal(!hasExpression(tag), "Tag already associated with an expression");
        AssertFatal(!isBorrowed(tag), ShowValue(tag), " has already been borrowed.");

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

        std::string aliasMsg;

        auto alias = getAlias(tag);
        if(alias)
        {
            auto dst = *alias;
            if(isBorrowed(dst))
            {
                AssertFatal(!isBorrowed(dst),
                            ShowValue(dst),
                            ShowValue(tag),
                            ShowValue(m_borrowedTags.at(dst)));
            }
            AssertFatal(hasRegister(dst), ShowValue(dst), ShowValue(tag));
            auto target = getRegister(dst);

            // Variable type is not checked as we can alias as long as the other
            // requirements are met.
            AssertFatal(target->registerCount() == tmpl->registerCount(),
                        ShowValue(target->registerCount()),
                        ShowValue(tmpl->registerCount()),
                        ShowValue(tag),
                        ShowValue(dst));
            AssertFatal(target->regType() == tmpl->regType(),
                        ShowValue(target->regType()),
                        ShowValue(tmpl->regType()),
                        ShowValue(tag),
                        ShowValue(dst));

            // Create a new Register::Value that points to the same allocation as target
            r = target->subset(iota(0ul, target->registerCount()).to<std::vector>());

            // Set the data type.
            r->setVariableType(tmpl->variableType());

            m_borrowedTags[dst] = tag;

            aliasMsg = fmt::format(" alias {} -> {}", tag, dst);
        }

        {
            auto inst
                = Instruction::Comment(fmt::format("tag {}: {}{}", tag, r->toString(), aliasMsg));
            m_context.lock()->schedule(inst);
        }

        m_registers.emplace(tag, r);
        return m_registers.at(tag);
    }

    inline void RegisterTagManager::addRegister(int tag, Register::ValuePtr value)
    {
        AssertFatal(!hasExpression(tag), "Tag already associated with an expression");
        AssertFatal(!hasAlias(tag),
                    "Aliasing tags must not be deferred.",
                    ShowValue(tag),
                    ShowValue(getAlias(tag).value_or(-1)));
        AssertFatal(!hasRegister(tag), "Tag ", tag, " already in RegisterTagManager.");

        if(auto existingTag = findRegister(value))
        {
            AssertFatal(value->readOnly(),
                        "Read/write Tag ",
                        tag,
                        ": ",
                        value->toString(),
                        " intersects with existing tag ",
                        *existingTag,
                        ": ",
                        getRegister(*existingTag)->toString());
        }

        m_registers.emplace(tag, value);
    }

    inline void RegisterTagManager::addExpression(int                          tag,
                                                  Expression::ExpressionPtr    value,
                                                  RegisterExpressionAttributes attrs)
    {
        AssertFatal(!isBorrowed(tag), "Tag ", tag, " has been borrowed.");
        AssertFatal(!hasRegister(tag), "Tag ", tag, " already associated with a register");

        AssertFatal(!m_aliases.contains(tag), "Cannot alias an expression tag.");

        m_expressions.emplace(tag, std::make_pair(value, attrs));
    }

    inline bool RegisterTagManager::isBorrowed(int tag) const
    {
        return m_borrowedTags.contains(tag);
    }

    inline bool RegisterTagManager::isAliased(int tag) const
    {
        auto iter = m_aliases.find(tag);
        return iter != m_aliases.end() && iter->second == ALIAS_DEST;
    }

    inline bool RegisterTagManager::hasAlias(int tag) const
    {
        auto iter = m_aliases.find(tag);
        return iter != m_aliases.end() && iter->second != ALIAS_DEST;
    }

    inline std::optional<int> RegisterTagManager::getAlias(int tag) const
    {
        auto iter = m_aliases.find(tag);
        if(iter != m_aliases.end() && iter->second != ALIAS_DEST)
        {
            return iter->second;
        }

        return std::nullopt;
    }

    inline std::optional<std::pair<int, int>> RegisterTagManager::getIndex(int tag) const
    {
        auto iter = m_indexes.find(tag);
        if(iter != m_indexes.end())
        {
            return iter->second;
        }

        return std::nullopt;
    }

    inline void RegisterTagManager::deleteTag(int tag)
    {
        auto comment = fmt::format("Deleting tag {}", tag);
        AssertFatal(!isBorrowed(tag), "Tag ", tag, " has been borrowed.");

        m_registers.erase(tag);
        m_expressions.erase(tag);

        auto alias = getAlias(tag);
        if(alias)
        {
            comment += fmt::format("alias (-> {})", *alias);
            m_borrowedTags.erase(*alias);
        }

        {
            auto inst = Instruction::Comment(comment);
            m_context.lock()->schedule(inst);
        }
    }

    inline bool RegisterTagManager::hasRegister(int tag) const
    {
        return m_registers.contains(tag) && !isBorrowed(tag);
    }

    inline std::optional<int> RegisterTagManager::findRegister(Register::ValuePtr reg) const
    {
        for(auto const& [tag, existingReg] : m_registers)
        {
            if(existingReg->intersects(reg))
                return tag;
        }

        return std::nullopt;
    }

    inline bool RegisterTagManager::hasExpression(int tag) const
    {
        return m_expressions.contains(tag) && !isBorrowed(tag);
    }
}
