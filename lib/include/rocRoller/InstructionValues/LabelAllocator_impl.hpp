#pragma once

#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    inline LabelAllocator::LabelAllocator(std::string const& prefix)
        : m_prefix(prefix)
    {
    }

    inline Register::ValuePtr LabelAllocator::label(std::string const& name)
    {
        m_count++;
        return Register::Value::Label(concatenate(m_prefix, "_", m_count - 1, "_", name));
    }
}
