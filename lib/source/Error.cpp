
#include "Utilities/Error.hpp"

namespace rocRoller
{
    [[noreturn]] void Crash()
    {
        auto will_be_null = GetNullPointer();
        if((*will_be_null = 0))
            throw std::runtime_error("Impossible 1");

        throw std::runtime_error("Impossible 2");
    }

    int* GetNullPointer()
    {
        return nullptr;
    }

}
