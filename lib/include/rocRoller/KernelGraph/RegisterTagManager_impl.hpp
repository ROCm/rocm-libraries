#pragma once

#include <map>
#include <memory>

#include "Context_fwd.hpp"

#include "InstructionValues/Register.hpp"

namespace rocRoller
{
    inline RegisterTagManager::RegisterTagManager(ContextPtr context)
        : m_context(context)
    {
    }

    inline RegisterTagManager::~RegisterTagManager() = default;

    inline std::pair<Expression::ExpressionPtr, DataType>
        RegisterTagManager::getExpression(int tag) const
    {
        AssertFatal(hasExpression(tag), ShowValue(tag));
        return m_expressions.at(tag);
    }

    inline std::shared_ptr<Register::Value> RegisterTagManager::getRegister(int tag)
    {
        AssertFatal(hasRegister(tag), ShowValue(tag));
        return m_registers.at(tag);
    }

    inline std::shared_ptr<Register::Value> RegisterTagManager::getRegister(int            tag,
                                                                            Register::Type regType,
                                                                            VariableType   varType,
                                                                            size_t valueCount)
    {
        AssertFatal(!hasExpression(tag), "Tag already associated with an expression");
        if(hasRegister(tag))
        {
            auto reg = m_registers.at(tag);
            if(varType != DataType::None)
            {
                AssertFatal(reg->variableType() == varType,
                            ShowValue(varType),
                            ShowValue(reg->variableType()));
                AssertFatal(reg->valueCount() == valueCount,
                            ShowValue(valueCount),
                            ShowValue(reg->valueCount()));
                AssertFatal(
                    reg->regType() == regType, ShowValue(regType), ShowValue(reg->regType()));
            }
            return reg;
        }
        auto r = Register::Value::Placeholder(m_context.lock(), regType, varType, valueCount);
        m_registers.emplace(tag, r);
        return m_registers.at(tag);
    }

    inline std::shared_ptr<Register::Value> RegisterTagManager::getRegister(int                tag,
                                                                            Register::ValuePtr tmpl)
    {
        return getRegister(tag, tmpl->regType(), tmpl->variableType(), tmpl->valueCount());
    }

    inline void RegisterTagManager::addRegister(int tag, Register::ValuePtr value)
    {
        AssertFatal(!hasExpression(tag), "Tag already associated with an expression");
        m_registers.insert(std::pair<int, Register::ValuePtr>(tag, value));
    }

    inline void
        RegisterTagManager::addExpression(int tag, Expression::ExpressionPtr value, DataType dt)
    {
        AssertFatal(!hasRegister(tag), "Tag ", tag, " already associated with a register");
        m_expressions.insert(
            std::pair<int, std::pair<Expression::ExpressionPtr, DataType>>(tag, {value, dt}));
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
