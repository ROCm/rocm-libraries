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

    inline std::shared_ptr<Register::Value> RegisterTagManager::getRegister(int tag)
    {
        AssertFatal(m_registers.count(tag) > 0, ShowValue(tag));
        return m_registers.at(tag);
    }

    inline std::shared_ptr<Register::Value> RegisterTagManager::getRegister(int            tag,
                                                                            Register::Type regType,
                                                                            VariableType   varType,
                                                                            size_t valueCount)
    {
        if(m_registers.count(tag) > 0)
        {
            auto reg = m_registers.at(tag);
            AssertFatal(reg->variableType() == varType);
            AssertFatal(reg->registerCount() == valueCount);
            AssertFatal(reg->regType() == regType);
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
        m_registers.insert(std::pair<int, Register::ValuePtr>(tag, value));
    }

    inline void RegisterTagManager::deleteRegister(int tag)
    {
        m_registers.erase(tag);
    }
}
