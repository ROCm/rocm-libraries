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
        AssertFatal(hasRegister(tag), ShowValue(tag));
        return m_registers.at(tag);
    }

    inline std::shared_ptr<Register::Value> RegisterTagManager::getRegister(int            tag,
                                                                            Register::Type regType,
                                                                            VariableType   varType,
                                                                            size_t valueCount)
    {
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
        m_registers.insert(std::pair<int, Register::ValuePtr>(tag, value));
    }

    inline void RegisterTagManager::deleteRegister(int tag)
    {
        m_registers.erase(tag);
    }

    inline bool RegisterTagManager::hasRegister(int tag) const
    {
        return m_registers.count(tag) > 0;
    }
}
