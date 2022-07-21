#pragma once

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRollerTest
{
    rocRoller::Generator<rocRoller::Instruction>
        SGEMM_Minimal_Program(rocRoller::ContextPtr context);
    rocRoller::Generator<rocRoller::Instruction>
        SGEMM_Optimized_Program(rocRoller::ContextPtr context);
}
