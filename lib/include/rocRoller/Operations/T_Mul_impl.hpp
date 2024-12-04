#pragma once

#include "T_Mul.hpp"

namespace rocRoller
{
    namespace Operations
    {
        inline T_Mul::T_Mul(OperationTag a, OperationTag b)
            : BaseOperation()
            , a(a)
            , b(b)
        {
        }

        inline std::unordered_set<OperationTag> T_Mul::getInputs() const
        {
            return {a, b};
        }

        inline std::string T_Mul::toString() const
        {
            return "T_Mul";
        }

        inline bool T_Mul::operator==(T_Mul const& rhs) const
        {
            return m_tag == rhs.m_tag && a == rhs.a && b == rhs.b;
        }
    }
}
