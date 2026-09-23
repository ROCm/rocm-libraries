// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "LabelAllocator.hpp"

#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    inline LabelAllocator::LabelAllocator(std::string prefix)
        : m_prefix(std::move(prefix))
    {
    }

    inline Register::ValuePtr LabelAllocator::label(std::string name)
    {
        name    = escapeSymbolName(std::move(name));
        auto rv = fmt::format("{}_{}", m_prefix, name);

        while(m_generatedLabels.contains(rv))
        {
            rv = fmt::format("{}_{}_{}", m_prefix, m_count, name);
            m_count++;
        }
        m_generatedLabels.insert(rv);
        return Register::Value::Label(rv);
    }
}
