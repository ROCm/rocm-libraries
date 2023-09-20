#pragma once

#include "../../InstructionValues/Register.hpp"

namespace rocRoller
{
    namespace Arithmetic
    {
        /**
         * @brief Represent a single Register::Value as two Register::Values each the size of a single DWord
        */
        void get2LiteralDwords(Register::ValuePtr& lsd,
                               Register::ValuePtr& msd,
                               Register::ValuePtr  input);
    }
}
