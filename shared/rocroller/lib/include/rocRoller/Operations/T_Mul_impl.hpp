// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Operations/T_Mul.hpp>

#include <fmt/core.h>

namespace rocRoller
{
    namespace Operations
    {
        inline T_Mul::T_Mul(OperationTag a, OperationTag b, VariableType accType)
            : BaseOperation()
            , a(a)
            , b(b)
            , accType(accType)
        {
        }

        inline std::unordered_set<OperationTag> T_Mul::getInputs() const
        {
            return {a, b};
        }

        inline std::string T_Mul::toString() const
        {
            return fmt::format("T_Mul {} {} {}", a.value, b.value, rocRoller::toString(accType));
        }

        inline bool T_Mul::operator==(T_Mul const& rhs) const
        {
            return m_tag == rhs.m_tag && a == rhs.a && b == rhs.b && accType == rhs.accType;
        }
    }
}
