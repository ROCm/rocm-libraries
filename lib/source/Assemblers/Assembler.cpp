#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    RegisterComponentBase(Assembler);

    std::string toString(AssemblerType t)
    {
        switch(t)
        {
        case AssemblerType::InProcess:
            return "InProcess";
        case AssemblerType::Subprocess:
            return "Subprocess";
        case AssemblerType::Count:
            return "Count";
        }

        Throw<FatalError>("Invalid AssemblerType: ", ShowValue(static_cast<int>(t)));
    }

    std::ostream& operator<<(std::ostream& stream, AssemblerType t)
    {
        return stream << toString(t);
    }

    AssemblerPtr Assembler::Get()
    {
        auto setting = Settings::Get(Settings::KernelAssembler);

        return Component::Get<Assembler>(setting);
    }

}
