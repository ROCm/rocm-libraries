#pragma once

#include <memory>
#include <string>

namespace rocRoller
{
    class Assembler;
    using AssemblerPtr = std::shared_ptr<Assembler>;

    enum class AssemblerType : int
    {
        InProcess = 0,
        Subprocess,
        Count
    };

    std::string toString(AssemblerType t);
}
