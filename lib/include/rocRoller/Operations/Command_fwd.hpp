/**
 */

#pragma once

#include <iostream>

namespace rocRoller
{
    class Command;

    std::ostream& operator<<(std::ostream& stream, Command const& command);

}
