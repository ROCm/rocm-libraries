#include <rocRoller/CodeGen/InstructionRef.hpp>

#include <rocRoller/CodeGen/Instruction.hpp>

namespace rocRoller
{
    InstructionRef::InstructionRef(Instruction const& inst)
        : m_opCode(inst.getOpCode())
    {
    }
}
