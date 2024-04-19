/**
 * T_Mul (tensor/matrix multiply) command.
 */

#pragma once

#include "Operation.hpp"

#include <memory>
#include <unordered_set>

namespace rocRoller
{
    namespace Operations
    {
        class T_Mul : public BaseOperation
        {
        public:
            T_Mul() = delete;
            T_Mul(int a, int b);

            std::unordered_set<int> getInputs() const;
            std::string             toString() const;

            int a, b;
        };

    }
}

#include "T_Mul_impl.hpp"
