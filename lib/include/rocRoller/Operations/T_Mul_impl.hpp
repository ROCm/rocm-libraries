#pragma once

#include "T_Mul.hpp"

namespace rocRoller
{
    namespace Operations
    {
        inline T_Mul::T_Mul(int a, int b)
            : BaseOperation()
            , a(a)
            , b(b)
        {
        }

        inline std::unordered_set<int> T_Mul::getInputs() const
        {
            return {a, b};
        }

        inline std::string T_Mul::toString() const
        {
            return "T_Mul";
        }
    }
}
