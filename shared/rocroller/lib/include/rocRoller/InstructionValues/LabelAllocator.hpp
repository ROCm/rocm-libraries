// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <set>

#include <rocRoller/InstructionValues/Register_fwd.hpp>

namespace rocRoller
{
    class LabelAllocator
    {
    public:
        LabelAllocator(std::string prefix);

        Register::ValuePtr label(std::string name);

    private:
        std::string  m_prefix;
        unsigned int m_count = 0;

        std::set<std::string> m_generatedLabels;
    };
}

#include <rocRoller/InstructionValues/LabelAllocator_impl.hpp>
