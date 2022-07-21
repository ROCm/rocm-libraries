/**
 * T_Mul (tensor/matrix multiply) command.
 */

#pragma once

#include <memory>
#include <unordered_set>

#include <rocRoller/Operations/Command_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {
        class T_Mul
        {
        public:
            T_Mul() = delete;
            T_Mul(int dest, int a, int b);

            void                    setCommand(std::shared_ptr<Command>);
            int                     getTag() const;
            void                    setTag(int tag);
            std::unordered_set<int> getInputs() const;
            std::string             toString() const;

            int dest, a, b;

        protected:
            std::weak_ptr<Command> m_command;
        };

    }
}

#include "T_Mul_impl.hpp"
